#pragma once

#include "memory_read_write.hpp"
#include "poseidon2.hpp"
#include "../../poseidon2/round_constants.cuh"
#include "util/exception.cuh"
#include "util/rusterror.h"
#include <chrono>
#include <iostream>
#include <ostream>

using namespace ::pico_gpu::poseidon2;

namespace pico_gpu::precompile_poseidon2 {
  using namespace memory_read_write;
  using Poseidon2ColsT = Poseidon2Cols<field_t, half_external_rounds, internal_rounds>;

  /// Convert a Poseidon2 permute event to a trace row.
  __device__ inline void event_to_row(
    const Poseidon2PermuteEvent& event,
    Poseidon2ColsT& cols,
    const field_t* __restrict__ round_constants,
    field_t* byte_trace)
  {
    // Populate basic event metadata.
    cols.chunk = field_t::from_canonical_u32(event.chunk);
    cols.clk = field_t::from_canonical_u32(event.clk);
    cols.input_memory_ptr = field_t::from_canonical_u32(event.input_memory_ptr);
    cols.output_memory_ptr = field_t::from_canonical_u32(event.output_memory_ptr);

    // Populate memory read/write records.
    for (size_t i = 0; i < WIDTH; i++) {
      populate(cols.input_memory[i], event.state_read_records[i], byte_trace);
      populate(cols.output_memory[i], event.state_write_records[i], byte_trace);
    }

    // Execute Poseidon2 permutation.
    Poseidon2ValueColsT& poseidon2_cols = cols.value_cols;
    poseidon2_cols.is_real = field_t::from_bool(true);
    for (int i = 0; i < WIDTH; i++) {
      poseidon2_cols.input[i] = field_t::from_canonical_u32(event.state_values[i]);
    }
    permute_and_populate(poseidon2_cols, round_constants);
    for (int i = 0; i < WIDTH; i++) {
      poseidon2_cols.input[i] = field_t::from_canonical_u32(event.state_values[i]);
    }
  }

  /// Kernel to convert events to trace rows.
  __global__ void events_to_rows_kernel(
    const Poseidon2PermuteEvent* __restrict__ events,
    field_t* __restrict__ trace_matrix,
    const size_t events_count,
    const size_t num_cols,
    const field_t* __restrict__ round_constants,
    field_t* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    Poseidon2ColsT* cols = reinterpret_cast<Poseidon2ColsT*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], *cols, round_constants, byte_trace);
  }

  /// Kernel to pad trace with dummy rows.
  __global__ void pad_dummy_rows_kernel(
    field_t* __restrict__ trace_matrix,
    const size_t pad_count,
    const size_t num_cols,
    const field_t* __restrict__ dummy_row)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    field_t* row = &trace_matrix[idx * num_cols];
    for (size_t i = 0; i < num_cols; i++) {
      row[i] = dummy_row[i];
    }
  }

  /// Generate trace from Poseidon2 permute events.
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const Poseidon2PermuteEvent* events,
    field_t* trace,
    const field_t* dummy_row,
    cudaStream_t stream,
    field_t* byte_trace)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(field_t), stream));

      // Pad dummy rows if needed.
      const size_t num_total_rows = trace_size / num_cols;
      if (num_total_rows > event_size) {
        const int block_size = 256;
        const size_t pad_count = num_total_rows - event_size;
        const int grid_size = int((pad_count + block_size - 1) / block_size);
        const size_t idx_dummy_start = event_size * num_cols;

        pad_dummy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          &trace[idx_dummy_start], pad_count, num_cols, dummy_row);
        CUDA_OK(cudaGetLastError());
      }

      // Initialize round constants and launch main kernel.
      RoundConstants<field_t> rc;
      init_round_constants(rc, stream);

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        events, trace, event_size, num_cols, rc.d_round_constants, byte_trace);
      CUDA_OK(cudaGetLastError());

      free_round_constants(rc, stream);
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::precompile_poseidon2
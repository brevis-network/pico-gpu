#pragma once

#include "ff/ff_config.hpp"
#include "poseidon2.hpp"

using namespace ::pico_gpu::poseidon2;

namespace pico_gpu::riscv_poseidon2 {
  __host__ __device__ inline void
  event_to_row(const Poseidon2Event& event, Poseidon2ValueColsT& cols, const field_t* __restrict__ round_constants)
  {
    cols.is_real = field_t::from_bool(true);
    for (int i = 0; i < WIDTH; i++) {
      cols.input[i] = field_t::from_canonical_u32(event.input[i]);
    }
    permute_and_populate(cols, round_constants);
    // Since permute_and_populate changes the state, we need to copy it back to
    // cols.input
    for (int i = 0; i < WIDTH; i++) {
      cols.input[i] = field_t::from_canonical_u32(event.input[i]);
    }
  }

  __global__ void events_to_rows_kernel(
    const Poseidon2Event* __restrict__ events,
    field_t* __restrict__ trace_matrix,
    size_t events_count,
    size_t num_cols,
    const field_t* __restrict__ round_constants)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    Poseidon2ValueColsT* cols = reinterpret_cast<Poseidon2ValueColsT*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], *cols, round_constants);
  }

  __global__ void pad_dummy_rows_kernel(
    field_t* __restrict__ trace_matrix, size_t pad_count, size_t num_cols, const field_t* __restrict__ dummy_row)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pad_count) return;

    field_t* row = &trace_matrix[idx * num_cols];
    for (size_t i = 0; i < num_cols; i++) {
      row[i] = dummy_row[i];
    }
  }

  inline RustError generate_main(
    size_t num_cols,
    size_t event_size,
    size_t trace_size,
    const Poseidon2Event* events,
    field_t* trace,
    field_t* dummy_row,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) { return RustError{cudaSuccess}; }

      // initialize trace matrix with zeros
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(field_t), stream));

      // launch kernel
      // generate real rows by events
      {
        RoundConstants<field_t> rc;
        init_round_constants(rc, stream);

        const int block_size = 256;
        const int grid_size = (event_size + block_size - 1) / block_size;

        events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          events, trace, event_size, num_cols, rc.d_round_constants);
        CUDA_OK(cudaGetLastError());

        free_round_constants(rc, stream);
      }

      // generate padding rows
      size_t num_total_rows = trace_size / num_cols;
      if (num_total_rows > event_size) {
        const int block_size = 256;
        size_t pad_size = num_total_rows - event_size;
        int grid_size = int((pad_size + block_size - 1) / block_size);

        size_t idx_dummy_start = event_size * num_cols;

        pad_dummy_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          &trace[idx_dummy_start], pad_size, num_cols, &dummy_row[0]);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::riscv_poseidon2

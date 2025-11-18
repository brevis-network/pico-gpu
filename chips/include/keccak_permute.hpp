#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "keccak_consts.hpp"
#include "keccak_populate.hpp"

namespace pico_gpu::keccak {

  /// Kernel to convert Keccak permutation events to trace.
  /// Each event generates NUM_ROUNDS rows in the trace.
  template <typename F>
  __global__ void keccak_events_to_trace_kernel(
    const KeccakPermuteEvent* events,
    const size_t events_count,
    const size_t events_total,
    const size_t rows_total,
    F* trace_matrix,
    const size_t num_cols,
    KeccakCols<F>** d_keccak_ptrs,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_total) return;

    constexpr int ROWS_PER_EVT = NUM_ROUNDS;
    const size_t row0 = idx * ROWS_PER_EVT;
    const size_t row0_offset = row0 * num_cols;

    if (row0 >= rows_total) return;

    F* row0_scalar = trace_matrix + row0_offset;
    KeccakMemCols<F>* rows = reinterpret_cast<KeccakMemCols<F>*>(row0_scalar);

    if (idx < events_count) {
      populate_chunk_gpu<F>(events[idx], rows, d_keccak_ptrs, row0, byte_trace);
    } else {
      populate_dummy_chunk_gpu<F>(rows, d_keccak_ptrs, row0, rows_total);
    }
  }

  /// Generate trace from Keccak permutation events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const KeccakPermuteEvent* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Calculate total events including dummy padding.
      const size_t rows_total = trace_size / num_cols;
      const size_t events_total = (rows_total + NUM_ROUNDS - 1) / NUM_ROUNDS;

      // Allocate device memory for Keccak column pointers.
      KeccakCols<F>** d_keccak_ptrs;
      CUDA_OK(cudaMalloc(&d_keccak_ptrs, rows_total * sizeof(KeccakCols<F>*)));

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const int block_size = 256;
      const int grid_size = (events_total + block_size - 1) / block_size;

      keccak_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, events_total, rows_total, trace, num_cols, d_keccak_ptrs, byte_trace);
      CUDA_OK(cudaGetLastError());

      CUDA_OK(cudaFree(d_keccak_ptrs));
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::keccak
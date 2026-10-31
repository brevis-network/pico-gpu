// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// MleOpenChip GPU trace generation.
// Asserts that a specific (air_id, col_id) MLE column opening equals the claimed value.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::mle_open {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`), alongside every other
  // chip's layout definitions; resolved here via enclosing-namespace lookup.

  // ── kernels ────────────────────────────────────────────────────────────────

  template <class F>
  __global__ void
  mle_open_instrs_to_trace_kernel(const MleOpenInstr<F>* instrs,
                                   size_t count,
                                   F* trace_matrix,
                                   size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MleOpenPreprocessedCols<F>* row =
        reinterpret_cast<MleOpenPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real            = F::one();
    row->col_ref            = F::from_canonical_u32(instrs[idx].col_ref);
    row->opening_value_addr = instrs[idx].opening_value_addr;
  }

  template <class F>
  __global__ void
  mle_open_events_to_trace_kernel(const MleOpenEvent<F>* events,
                                   size_t count,
                                   F* trace_matrix,
                                   size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MleOpenCols<F>* row =
        reinterpret_cast<MleOpenCols<F>*>(&trace_matrix[idx * num_cols]);
    row->opening_value = events[idx].opening_value;
  }

  // ── entry points ───────────────────────────────────────────────────────────

  template <class F>
  inline RustError
  generate_preprocessed(const MleOpenInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      mle_open_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const MleOpenEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      mle_open_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::mle_open

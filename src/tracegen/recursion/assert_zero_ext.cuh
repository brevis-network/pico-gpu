// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// AssertZeroExtChip GPU trace generation.
// The chip asserts that a given extension-field value equals zero.
// Trace is trivial: one Block per event.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::assert_zero_ext {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`), alongside every other
  // chip's layout definitions; resolved here via enclosing-namespace lookup.

  // ── kernels ────────────────────────────────────────────────────────────────

  template <class F>
  __global__ void
  assert_zero_ext_instrs_to_trace_kernel(const AssertZeroExtInstr<F>* instrs,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    AssertZeroExtPreprocessedCols<F>* row =
        reinterpret_cast<AssertZeroExtPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real = F::one();
    row->addr    = instrs[idx].addr;
  }

  template <class F>
  __global__ void
  assert_zero_ext_events_to_trace_kernel(const AssertZeroExtEvent<F>* events,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    AssertZeroExtCols<F>* row =
        reinterpret_cast<AssertZeroExtCols<F>*>(&trace_matrix[idx * num_cols]);
    row->val = events[idx].val;
  }

  // ── entry points ───────────────────────────────────────────────────────────

  template <class F>
  inline RustError
  generate_preprocessed(const AssertZeroExtInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      assert_zero_ext_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const AssertZeroExtEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      assert_zero_ext_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::assert_zero_ext

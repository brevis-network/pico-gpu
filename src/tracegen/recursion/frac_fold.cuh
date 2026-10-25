// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// FracFoldChip GPU trace generation.
// GKR fractional fold: (p1,q1) + (p2,q2) = (p1*q2 + p2*q1, q1*q2).
// The trace stores the six EF values; the arithmetic is verified via AIR constraints.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::frac_fold {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`), alongside every other
  // chip's layout definitions; resolved here via enclosing-namespace lookup.

  // ── kernels ────────────────────────────────────────────────────────────────

  template <class F>
  __global__ void
  frac_fold_instrs_to_trace_kernel(const FracFoldInstr<F>* instrs,
                                    size_t count,
                                    F* trace_matrix,
                                    size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    FracFoldPreprocessedCols<F>* row =
        reinterpret_cast<FracFoldPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real    = F::one();
    row->in_p1_addr = instrs[idx].in_p1_addr;
    row->in_q1_addr = instrs[idx].in_q1_addr;
    row->in_p2_addr = instrs[idx].in_p2_addr;
    row->in_q2_addr = instrs[idx].in_q2_addr;
    row->out_p_addr = instrs[idx].out_p_addr;
    row->out_q_addr = instrs[idx].out_q_addr;
    row->mult       = instrs[idx].mult;
  }

  template <class F>
  __global__ void
  frac_fold_events_to_trace_kernel(const FracFoldEvent<F>* events,
                                    size_t count,
                                    F* trace_matrix,
                                    size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    FracFoldCols<F>* row =
        reinterpret_cast<FracFoldCols<F>*>(&trace_matrix[idx * num_cols]);
    row->p1    = events[idx].p1;
    row->q1    = events[idx].q1;
    row->p2    = events[idx].p2;
    row->q2    = events[idx].q2;
    row->out_p = events[idx].out_p;
    row->out_q = events[idx].out_q;
  }

  // ── entry points ───────────────────────────────────────────────────────────

  template <class F>
  inline RustError
  generate_preprocessed(const FracFoldInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      frac_fold_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const FracFoldEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      frac_fold_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::frac_fold

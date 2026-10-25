// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// GkrSumcheckRoundChip GPU trace generation.
// One GKR sumcheck round per row: copies the EF values and computes the r^2 helper
// (r_sq = r * r) that keeps the cubic-interpolation constraint at degree 4.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/recursion/recursion_ef.cuh"

namespace pico_gpu::gkr_sumcheck_round {

  // Event / Instr / Cols structs live in types.cuh; ef4_mul lives in recursion_ef.cuh
  // (namespace `pico_gpu`, resolved here via enclosing-namespace lookup).

  template <class F>
  __global__ void
  gkr_sumcheck_round_instrs_to_trace_kernel(const GkrSumcheckRoundInstr<F>* instrs,
                                            size_t count,
                                            F* trace_matrix,
                                            size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    GkrSumcheckRoundPreprocessedCols<F>* cols =
        reinterpret_cast<GkrSumcheckRoundPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    const GkrSumcheckRoundInstr<F>& in = instrs[idx];
    cols->is_real = F::one();
    cols->claim_addr = in.claim_addr;
    cols->s1_addr = in.s1_addr;
    cols->s2_addr = in.s2_addr;
    cols->s3_addr = in.s3_addr;
    cols->r_addr = in.r_addr;
    cols->xi_addr = in.xi_addr;
    cols->eq_in_addr = in.eq_in_addr;
    cols->next_claim_addr = in.next_claim_addr;
    cols->eq_out_addr = in.eq_out_addr;
    cols->mult_next = in.mult_next;
    cols->mult_eq = in.mult_eq;
  }

  /// `w` is the binomial extension constant of the field.
  template <class F>
  __global__ void
  gkr_sumcheck_round_events_to_trace_kernel(const GkrSumcheckRoundEvent<F>* events,
                                            size_t count,
                                            F* trace_matrix,
                                            size_t num_cols,
                                            F w)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const GkrSumcheckRoundEvent<F>& ev = events[idx];
    GkrSumcheckRoundCols<F>* cols =
        reinterpret_cast<GkrSumcheckRoundCols<F>*>(&trace_matrix[idx * num_cols]);
    cols->claim = ev.claim;
    cols->s1 = ev.s1;
    cols->s2 = ev.s2;
    cols->s3 = ev.s3;
    cols->r = ev.r;
    cols->xi = ev.xi;
    cols->eq_in = ev.eq_in;
    ef4_mul(ev.r._0, ev.r._0, cols->r_sq._0, w);
    cols->next_claim = ev.next_claim;
    cols->eq_out = ev.eq_out;
  }

  template <class F>
  inline RustError
  generate_preprocessed(const GkrSumcheckRoundInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      gkr_sumcheck_round_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  /// `w` is the binomial extension constant for the field.
  template <class F>
  inline RustError
  generate_main(const GkrSumcheckRoundEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                F w,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      gkr_sumcheck_round_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols, w);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::gkr_sumcheck_round

// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// UnivariateFoldChip GPU trace generation.
// Fills the coefficient slots and the Horner accumulator constraining next_sum = p(r).
// The Rust side flattens Vec<Block<F>> into the fixed-size FlatEvent before passing to CUDA.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::univariate_fold {

  // Event / Instr / Cols structs and UNIVARIATE_FOLD_MAX_COEFFS live in types.cuh
  // (namespace `pico_gpu`); resolved here via enclosing-namespace lookup.
  static constexpr size_t UNIVARIATE_FOLD_OMEGA_ORDER = 16; // (MAX_COEFFS + 1) / 2

  // ── extension-field helpers (degree-4 binomial extension, X^4 - W) ──────────

  template <class F>
  __device__ __forceinline__ void
  ef4_mul(const F* a, const F* b, F* out, const F& w)
  {
    F a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    F b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = a0 * b0 + w * (a1 * b3 + a2 * b2 + a3 * b1);
    out[1] = a0 * b1 + a1 * b0 + w * (a2 * b3 + a3 * b2);
    out[2] = a0 * b2 + a1 * b1 + a2 * b0 + w * (a3 * b3);
    out[3] = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
  }

  template <class F>
  __device__ __forceinline__ void
  ef4_add(const F* a, const F* b, F* out)
  {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
  }

  // ── kernels ────────────────────────────────────────────────────────────────

  template <class F>
  __global__ void
  univariate_fold_instrs_to_trace_kernel(const UnivariateFoldInstr<F>* instrs,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    UnivariateFoldPreprocessedCols<F>* row =
        reinterpret_cast<UnivariateFoldPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real          = F::one();
    row->num_coeffs       = instrs[idx].num_coeffs;
    row->claimed_sum_addr = instrs[idx].claimed_sum_addr;
    row->r_addr           = instrs[idx].r_addr;
    row->next_sum_addr    = instrs[idx].next_sum_addr;
    row->mult             = instrs[idx].mult;
    for (size_t k = 0; k < UNIVARIATE_FOLD_MAX_COEFFS; ++k)
      row->coeffs_addrs[k] = instrs[idx].coeffs_addrs[k];
  }

  /// One thread per event — the Horner recurrence is sequential within each row.
  /// `w` is the binomial extension constant of the field.
  template <class F>
  __global__ void
  univariate_fold_events_to_trace_kernel(const UnivariateFoldFlatEvent<F>* events,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols,
                                          F w)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const UnivariateFoldFlatEvent<F>& ev = events[idx];
    UnivariateFoldCols<F>* row =
        reinterpret_cast<UnivariateFoldCols<F>*>(&trace_matrix[idx * num_cols]);

    for (size_t k = 0; k < UNIVARIATE_FOLD_MAX_COEFFS; ++k)
      row->coeffs[k] = ev.coeffs[k];

    row->claimed_sum = ev.claimed_sum;
    row->r           = ev.r;
    row->next_sum    = ev.next_sum;

    // eval_at_0 = coeffs[0]; eval_at_1 = coeffs[OMEGA_ORDER].
    row->eval_at_0 = ev.coeffs[0];
    row->eval_at_1 = ev.coeffs[UNIVARIATE_FOLD_OMEGA_ORDER];

    // Horner accumulator: acc[0] = coeffs[MAX-1]; acc[i] = acc[i-1] * r + coeffs[MAX-1-i].
    // The last stored slot (acc[MAX-2]) is the partial through coeffs[1]; the final step
    // acc[MAX-2] * r + coeffs[0] = p(r) = next_sum is enforced by the AIR, not stored here.
    row->acc[0] = ev.coeffs[UNIVARIATE_FOLD_MAX_COEFFS - 1];
    for (size_t i = 1; i < UNIVARIATE_FOLD_MAX_COEFFS - 1; ++i) {
      F prod[4], sum[4];
      ef4_mul(row->acc[i - 1]._0, ev.r._0, prod, w);
      ef4_add(prod, ev.coeffs[UNIVARIATE_FOLD_MAX_COEFFS - 1 - i]._0, sum);
      row->acc[i]._0[0] = sum[0];
      row->acc[i]._0[1] = sum[1];
      row->acc[i]._0[2] = sum[2];
      row->acc[i]._0[3] = sum[3];
    }
  }

  // ── entry points ───────────────────────────────────────────────────────────

  template <class F>
  inline RustError
  generate_preprocessed(const UnivariateFoldInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      univariate_fold_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  /// `w` is the binomial extension constant for the field.
  template <class F>
  inline RustError
  generate_main(const UnivariateFoldFlatEvent<F>* events,
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
      univariate_fold_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols, w);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::univariate_fold

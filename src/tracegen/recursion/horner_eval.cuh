// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// HornerEvalChip GPU trace generation.
// main: DATAPAR coefficients + the x^2/x^3/x^4 degree-splitting helpers (computed on the device).
// preproc: each instruction spans len/DATAPAR rows; expanded fully on the device from a two-buffer
// layout (per-instruction metadata + a flat coefficient-address buffer), no host-side row building.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/recursion/recursion_ef.cuh"

namespace pico_gpu::horner_eval {

  // Event / Cols / Meta structs and HORNER_EVAL_DATAPAR live in types.cuh; ef4_mul lives in
  // recursion_ef.cuh (namespace `pico_gpu`, resolved here via enclosing-namespace lookup).

  /// One thread per instruction: writes its len/DATAPAR contiguous rows into the device trace,
  /// reading the coefficient addresses from the flat side buffer.
  template <class F>
  __global__ void
  horner_eval_instrs_to_trace_kernel(const HornerEvalPreprocMeta<F>* metas,
                                     size_t count,
                                     const F* coeff_addrs_flat,
                                     F* trace_matrix,
                                     size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const HornerEvalPreprocMeta<F>& meta = metas[idx];
    uint32_t n_rows = meta.num_coeffs / HORNER_EVAL_DATAPAR;
    for (uint32_t i = 0; i < n_rows; ++i) {
      HornerEvalPreprocessedCols<F>* cols = reinterpret_cast<HornerEvalPreprocessedCols<F>*>(
          &trace_matrix[(meta.row_offset + i) * num_cols]);
      cols->is_real = F::one();
      cols->is_start = (i == 0) ? F::one() : F::zero();
      cols->is_end = (i == n_rows - 1) ? F::one() : F::zero();
      for (uint32_t j = 0; j < HORNER_EVAL_DATAPAR; ++j)
        cols->coeff_addrs[j] = coeff_addrs_flat[meta.coeff_offset + HORNER_EVAL_DATAPAR * i + j];
      cols->x_addr = meta.x_addr;
      cols->out_addr = meta.out_addr;
      cols->mult = (i == n_rows - 1) ? meta.mult : F::zero();
    }
  }

  /// `w` is the binomial extension constant of the field.
  template <class F>
  __global__ void
  horner_eval_events_to_trace_kernel(const HornerEvalEvent<F>* events,
                                     size_t count,
                                     F* trace_matrix,
                                     size_t num_cols,
                                     F w)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const HornerEvalEvent<F>& ev = events[idx];
    HornerEvalCols<F>* cols =
        reinterpret_cast<HornerEvalCols<F>*>(&trace_matrix[idx * num_cols]);
    for (uint32_t j = 0; j < HORNER_EVAL_DATAPAR; ++j) cols->coeffs[j] = ev.coeffs[j];
    cols->x = ev.x;
    // x2 = x*x, x3 = x2*x, x4 = x2*x2.
    ef4_mul(ev.x._0, ev.x._0, cols->x2._0, w);
    ef4_mul(cols->x2._0, ev.x._0, cols->x3._0, w);
    ef4_mul(cols->x2._0, cols->x2._0, cols->x4._0, w);
    cols->acc = ev.acc;
  }

  /// `coeff_addrs_flat` holds all instructions' coefficient addresses concatenated; each meta
  /// carries its offset/count and the device-computed `row_offset`.
  template <class F>
  inline RustError
  generate_preprocessed(const HornerEvalPreprocMeta<F>* metas,
                        size_t num_instrs,
                        const F* coeff_addrs_flat,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      horner_eval_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          metas, num_instrs, coeff_addrs_flat, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  /// `w` is the binomial extension constant for the field.
  template <class F>
  inline RustError
  generate_main(const HornerEvalEvent<F>* events,
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
      horner_eval_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols, w);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::horner_eval

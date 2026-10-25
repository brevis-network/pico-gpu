// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// BinaryFoldStepChip GPU trace generation.
// One WHIR binary k-fold butterfly step per row: copies the inputs and computes the two helpers
// amt = alpha - tw*xi (base lane only) and tih = inv_tw * xi_inv * (1/2). No extension multiply is
// stored (the CPU ef_mul there is only a debug sanity check; `out` is copied verbatim).

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::binary_fold_step {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  binary_fold_step_instrs_to_trace_kernel(const BinaryFoldStepInstr<F>* instrs,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    BinaryFoldStepPreprocessedCols<F>* cols =
        reinterpret_cast<BinaryFoldStepPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    const BinaryFoldStepInstr<F>& in = instrs[idx];
    cols->is_real = F::one();
    cols->tw_addr = in.tw_addr;
    cols->xi_addr = in.xi_addr;
    cols->inv_tw_addr = in.inv_tw_addr;
    cols->xi_inv_addr = in.xi_inv_addr;
    cols->alpha_addr = in.alpha_addr;
    cols->v_lo_addr = in.v_lo_addr;
    cols->v_hi_addr = in.v_hi_addr;
    cols->out_addr = in.out_addr;
    cols->mult = in.mult;
  }

  /// `inv2` is the field element 1/2.
  template <class F>
  __global__ void
  binary_fold_step_events_to_trace_kernel(const BinaryFoldStepEvent<F>* events,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols,
                                          F inv2)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const BinaryFoldStepEvent<F>& ev = events[idx];
    BinaryFoldStepCols<F>* cols =
        reinterpret_cast<BinaryFoldStepCols<F>*>(&trace_matrix[idx * num_cols]);
    cols->tw = ev.tw;
    cols->xi = ev.xi;
    cols->inv_tw = ev.inv_tw;
    cols->xi_inv = ev.xi_inv;
    cols->alpha = ev.alpha;
    cols->v_lo = ev.v_lo;
    cols->v_hi = ev.v_hi;
    // amt = alpha - tw*xi (only the base lane of tw*xi is nonzero).
    for (int k = 0; k < EXTENSION_DEGREE; ++k) cols->amt._0[k] = ev.alpha._0[k];
    cols->amt._0[0] = cols->amt._0[0] - ev.tw * ev.xi;
    cols->tih = ev.inv_tw * ev.xi_inv * inv2;
    cols->out = ev.out;
  }

  template <class F>
  inline RustError
  generate_preprocessed(const BinaryFoldStepInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      binary_fold_step_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  /// `inv2` is the field element 1/2.
  template <class F>
  inline RustError
  generate_main(const BinaryFoldStepEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                F inv2,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      binary_fold_step_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols, inv2);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::binary_fold_step

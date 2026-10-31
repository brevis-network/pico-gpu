// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// LagrangeEvalD2Chip GPU trace generation.
// Stores the extension-field values for a degree-2 Lagrange evaluation round.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::lagrange_eval_d2 {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  lagrange_eval_d2_instrs_to_trace_kernel(const LagrangeEvalD2Instr<F>* instrs,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    LagrangeEvalD2PreprocessedCols<F>* row =
        reinterpret_cast<LagrangeEvalD2PreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real         = F::one();
    row->claim_addr      = instrs[idx].claim_addr;
    row->s1_addr         = instrs[idx].s1_addr;
    row->s2_addr         = instrs[idx].s2_addr;
    row->r_addr          = instrs[idx].r_addr;
    row->next_claim_addr = instrs[idx].next_claim_addr;
    row->mult            = instrs[idx].mult;
  }

  template <class F>
  __global__ void
  lagrange_eval_d2_events_to_trace_kernel(const LagrangeEvalD2Event<F>* events,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    LagrangeEvalD2Cols<F>* row =
        reinterpret_cast<LagrangeEvalD2Cols<F>*>(&trace_matrix[idx * num_cols]);
    row->claim      = events[idx].claim;
    row->s1         = events[idx].s1;
    row->s2         = events[idx].s2;
    row->r          = events[idx].r;
    row->next_claim = events[idx].next_claim;
  }

  template <class F>
  inline RustError
  generate_preprocessed(const LagrangeEvalD2Instr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      lagrange_eval_d2_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const LagrangeEvalD2Event<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      lagrange_eval_d2_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::lagrange_eval_d2

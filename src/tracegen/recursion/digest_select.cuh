// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// DigestSelectChip GPU trace generation.
// Per-lane select of two 8-lane digests. DIGEST_SELECT_DATAPAR independent ops per row
// (one thread per op, mapped to its row + datapar slot).

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::digest_select {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  digest_select_instrs_to_trace_kernel(const DigestSelectInstr<F>* instrs,
                                        size_t count,
                                        F* trace_matrix,
                                        size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t row = idx / DIGEST_SELECT_DATAPAR;
    uint32_t slot = idx % DIGEST_SELECT_DATAPAR;

    DigestSelectPreprocessedCols<F>* cols =
        reinterpret_cast<DigestSelectPreprocessedCols<F>*>(&trace_matrix[row * num_cols]);
    DigestSelectPreprocessedValueCols<F>& vc = cols->values[slot];
    vc.is_real = F::one();
    vc.bit_addr = instrs[idx].bit_addr;
    for (int i = 0; i < 2; ++i) {
      vc.in1_addrs[i] = instrs[idx].in1_addrs[i];
      vc.in2_addrs[i] = instrs[idx].in2_addrs[i];
      vc.out_addrs[i] = instrs[idx].out_addrs[i];
      vc.mults[i] = instrs[idx].mults[i];
    }
  }

  template <class F>
  __global__ void
  digest_select_events_to_trace_kernel(const DigestSelectEvent<F>* events,
                                       size_t count,
                                       F* trace_matrix,
                                       size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t row = idx / DIGEST_SELECT_DATAPAR;
    uint32_t slot = idx % DIGEST_SELECT_DATAPAR;

    DigestSelectCols<F>* cols =
        reinterpret_cast<DigestSelectCols<F>*>(&trace_matrix[row * num_cols]);
    DigestSelectValueCols<F>& vc = cols->values[slot];
    vc.bit = events[idx].bit;
    for (int i = 0; i < 8; ++i) {
      vc.in1[i] = events[idx].in1[i];
      vc.in2[i] = events[idx].in2[i];
      vc.out[i] = events[idx].out[i];
    }
  }

  template <class F>
  inline RustError
  generate_preprocessed(const DigestSelectInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      digest_select_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const DigestSelectEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      digest_select_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::digest_select

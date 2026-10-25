// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// UnpackBlockChip GPU trace generation.
// Unpacks an extension-field Block into its four base-field lanes.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::unpack_block {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  unpack_block_instrs_to_trace_kernel(const UnpackBlockInstr<F>* instrs,
                                       size_t count,
                                       F* trace_matrix,
                                       size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    UnpackBlockPreprocessedCols<F>* row =
        reinterpret_cast<UnpackBlockPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    row->is_real = F::one();
    row->in_addr = instrs[idx].in_addr;
    for (int i = 0; i < 4; ++i) {
      row->out[i].addr = instrs[idx].out_addrs[i];
      row->out[i].mult = instrs[idx].mults[i];
    }
  }

  template <class F>
  __global__ void
  unpack_block_events_to_trace_kernel(const UnpackBlockEvent<F>* events,
                                      size_t count,
                                      F* trace_matrix,
                                      size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    UnpackBlockCols<F>* row =
        reinterpret_cast<UnpackBlockCols<F>*>(&trace_matrix[idx * num_cols]);
    row->val = events[idx].val;
  }

  template <class F>
  inline RustError
  generate_preprocessed(const UnpackBlockInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      unpack_block_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const UnpackBlockEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      unpack_block_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::unpack_block

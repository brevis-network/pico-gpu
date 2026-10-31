// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// PackBlockChip GPU trace generation.
// Packs four base-field lanes into a Block. PACK_BLOCK_DATAPAR independent ops per row
// (one thread per op, mapped to its row + datapar slot, mirroring alu_ext.cuh).

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::pack_block {

  // Event / Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  pack_block_instrs_to_trace_kernel(const PackBlockInstr<F>* instrs,
                                     size_t count,
                                     F* trace_matrix,
                                     size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t row = idx / PACK_BLOCK_DATAPAR;
    uint32_t slot = idx % PACK_BLOCK_DATAPAR;

    PackBlockPreprocessedCols<F>* cols =
        reinterpret_cast<PackBlockPreprocessedCols<F>*>(&trace_matrix[row * num_cols]);
    PackBlockPreprocessedValueCols<F>& vc = cols->values[slot];
    vc.is_real = F::one();
    for (int i = 0; i < 4; ++i) vc.in_addrs[i] = instrs[idx].in_addrs[i];
    vc.out_addr = instrs[idx].out_addr;
    vc.mult = instrs[idx].mult;
  }

  template <class F>
  __global__ void
  pack_block_events_to_trace_kernel(const PackBlockEvent<F>* events,
                                    size_t count,
                                    F* trace_matrix,
                                    size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t row = idx / PACK_BLOCK_DATAPAR;
    uint32_t slot = idx % PACK_BLOCK_DATAPAR;

    PackBlockCols<F>* cols =
        reinterpret_cast<PackBlockCols<F>*>(&trace_matrix[row * num_cols]);
    for (int j = 0; j < 4; ++j) cols->values[slot][j] = events[idx].lanes[j];
  }

  template <class F>
  inline RustError
  generate_preprocessed(const PackBlockInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      pack_block_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

  template <class F>
  inline RustError
  generate_main(const PackBlockEvent<F>* events,
                size_t num_events,
                F* trace,
                size_t num_cols,
                cudaStream_t stream)
  {
    try {
      if (num_events == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      pack_block_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::pack_block

// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Poseidon2PackedChip GPU trace generation (preprocessed only).
// The main columns are the shared Poseidon2ValueCols (one permutation per row), identical to the
// recursion poseidon2 chip, so the main trace reuses that chip's kernel. This file only provides
// the packed preprocessed layout: 4 packed input Block addresses + 4 packed output accesses.

#pragma once

#include "common/rusterror.h"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::poseidon2_packed {

  // Instr / Cols structs live in types.cuh (namespace `pico_gpu`).

  template <class F>
  __global__ void
  poseidon2_packed_instrs_to_trace_kernel(const Poseidon2PackedInstr<F>* instrs,
                                          size_t count,
                                          F* trace_matrix,
                                          size_t num_cols)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    Poseidon2PackedPreprocessedCols<F>* cols =
        reinterpret_cast<Poseidon2PackedPreprocessedCols<F>*>(&trace_matrix[idx * num_cols]);
    const Poseidon2PackedInstr<F>& in = instrs[idx];
    cols->is_real = F::one();
    for (int j = 0; j < 4; ++j) {
      cols->input[j] = in.input_addrs[j];
      cols->output[j].addr = in.output_addrs[j];
      cols->output[j].mult = in.mults[j];
    }
  }

  template <class F>
  inline RustError
  generate_preprocessed(const Poseidon2PackedInstr<F>* instrs,
                        size_t num_instrs,
                        F* trace,
                        size_t num_cols,
                        cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) return RustError{cudaSuccess};
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      poseidon2_packed_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
          instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) { return RustError{e.code()}; }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::poseidon2_packed

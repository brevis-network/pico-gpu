#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::recursion_memory_const {
  // Kernel implementation
  template <class F>
  __global__ void
  mem_instrs_to_trace_kernel(const MemInstrFfi<F>* instrs, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const MemInstrFfi<F>& inst = instrs[idx];
    uint32_t row_index = idx / CONST_MEM_DATAPAR;
    uint32_t datapar_index = idx % CONST_MEM_DATAPAR;
    RecursionMemoryConstPreprocessedCols<F>* cols =
      reinterpret_cast<RecursionMemoryConstPreprocessedCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->values_and_accesses[datapar_index].value = inst.vals.inner;
    cols->values_and_accesses[datapar_index].access.addr = inst.addrs.inner;
    cols->values_and_accesses[datapar_index].access.mult = inst.mult;
  }

  template <class F>
  inline RustError
  generate_preprocessed(const MemInstrFfi<F>* instrs, size_t num_instrs, F* trace, size_t num_cols, cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      mem_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed
} // namespace pico_gpu::recursion_memory_const

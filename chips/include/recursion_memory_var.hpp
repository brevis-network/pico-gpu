#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::recursion_memory_var {
  // Kernel implementation
  template <class F>
  __global__ void mem_accesses_to_trace_kernel(
    const RecursionMemoryAccessCols<F>* instrs, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const RecursionMemoryAccessCols<F>& inst = instrs[idx];
    uint32_t row_index = idx / VAR_MEM_DATAPAR;
    uint32_t datapar_index = idx % VAR_MEM_DATAPAR;
    RecursionMemoryVarPreprocessedCols<F>* cols =
      reinterpret_cast<RecursionMemoryVarPreprocessedCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->accesses[datapar_index] = inst;
  }

  template <class F>
  __global__ void mem_events_to_trace_kernel(const MemEvent<F>* events, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const MemEvent<F>& e = events[idx];
    uint32_t row_index = idx / VAR_MEM_DATAPAR;
    uint32_t datapar_index = idx % VAR_MEM_DATAPAR;
    RecursionVariableMemoryCols<F>* cols =
      reinterpret_cast<RecursionVariableMemoryCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->values[datapar_index] = e.inner;
  }

  template <class F>
  inline RustError generate_preprocessed(
    const RecursionMemoryAccessCols<F>* instrs, size_t num_instrs, F* trace, size_t num_cols, cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      mem_accesses_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(instrs, num_instrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  template <class F>
  inline RustError
  generate_main(const MemEvent<F>* events, size_t num_events, F* trace, size_t num_cols, cudaStream_t stream)
  {
    try {
      if (num_events == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      mem_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::recursion_memory_var

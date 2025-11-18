#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::public_values {
  // Kernel implementation
  template <class F>
  __global__ void public_values_addrs_to_trace_kernel(const F* addrs, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= DIGEST_SIZE) return;

    const F& addr = addrs[idx];
    uint32_t row_index = idx;
    PublicValuesPreprocessedCols<F>* cols =
      reinterpret_cast<PublicValuesPreprocessedCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->pv_idx[idx] = F::one();
    cols->pv_mem.addr = addr;
    cols->pv_mem.mult = -F::one();
  }

  template <class F>
  __global__ void public_values_to_trace_kernel(const F* digest, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= DIGEST_SIZE) return;

    const F& digest_elem = digest[idx];
    uint32_t row_index = idx;
    PublicValuesCols<F>* cols = reinterpret_cast<PublicValuesCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->pv_element = digest_elem;
  }

  template <class F>
  inline RustError generate_preprocessed(const F* addrs, F* trace, size_t num_cols, cudaStream_t stream)
  {
    assert(addrs != NULL);

    try {
      // launch kernel
      const int block_size = DIGEST_SIZE;
      int grid_size = 1;
      public_values_addrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(addrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main

  template <class F>
  inline RustError generate_main(const F* digest, F* trace, size_t num_cols, cudaStream_t stream)
  {
    assert(digest != NULL);

    try {
      // launch kernel
      const int block_size = DIGEST_SIZE;
      int grid_size = 1;
      public_values_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(digest, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::public_values

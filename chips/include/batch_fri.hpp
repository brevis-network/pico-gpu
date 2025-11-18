#pragma once

#include <iostream>
#include <util/rusterror.h>
#include <util/exception.cuh>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::batch_fri {
  // Kernel implementation
  template <class F>
  __global__ void batch_fri_insts_to_trace_kernel(
    const BatchFRIInstrFfi<F>* instrs,
    size_t count,
    const Address<F>* p_at_x,
    const Address<F>* p_at_z,
    const Address<F>* alpha_pow,
    F* trace_matrix,
    size_t num_cols,
    F t,
    F f)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BatchFRIInstrFfi<F>& inst = instrs[idx];

    for (uintptr_t i = 0; i < inst.len; i++) {
      BatchFRIPreprocessedCols<F>* cols =
        reinterpret_cast<BatchFRIPreprocessedCols<F>*>(&trace_matrix[(inst.offset + i) * num_cols]);
      cols->is_real = F::one();
      cols->is_end = i == inst.len - 1 ? t : f;
      cols->acc_addr = inst.ext_single_addrs.acc;
      cols->alpha_pow_addr = alpha_pow[inst.offset + i];
      cols->p_at_z_addr = p_at_z[inst.offset + i];
      cols->p_at_x_addr = p_at_x[inst.offset + i];
    }
  }

  template <class F>
  __global__ void
  batch_fri_to_trace_kernel(const BatchFRIEvent<F>* events, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BatchFRIEvent<F>& e = events[idx];
    BatchFRICols<F>* cols = reinterpret_cast<BatchFRICols<F>*>(&trace_matrix[idx * num_cols]);

    cols->acc = e.ext_single.acc;
    cols->alpha_pow = e.ext_vec.alpha_pow;
    cols->p_at_z = e.ext_vec.p_at_z;
    cols->p_at_x = e.base_vec.p_at_x;
  }

  template <class F>
  inline RustError generate_preprocessed(
    const BatchFRIInstrFfi<F>* instrs,
    size_t num_instrs,
    const Address<F>* p_at_x,
    const Address<F>* p_at_z,
    const Address<F>* alpha_pow,
    F* trace,
    size_t num_cols,
    cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }
      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      F t = F::from_bool(true);
      F f = F::from_bool(false);
      batch_fri_insts_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        instrs, num_instrs, p_at_x, p_at_z, alpha_pow, trace, num_cols, t, f);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  template <class F>
  inline RustError
  generate_main(const BatchFRIEvent<F>* events, size_t num_events, F* trace, size_t num_cols, cudaStream_t stream)
  {
    try {
      if (num_events == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      batch_fri_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, num_events, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::batch_fri

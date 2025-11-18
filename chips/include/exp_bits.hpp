#pragma once

#include <iostream>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::exp_bits {
  // Kernel implementation
  template <class F>
  __global__ void exp_bits_instrs_to_trace_kernel(
    const ExpReverseBitsInstrFfi<F>* instrs, size_t count, const Address<F>* addrs, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const ExpReverseBitsInstrFfi<F>& inst = instrs[idx];
    uint32_t initial_row_index = inst.addrs.initial_row;

    F accum = F::one();

    // write out each row
    for (uintptr_t i = 0; i < inst.addrs.len; i++) {
      ExpReverseBitsLenPreprocessedCols<F>* cols =
        reinterpret_cast<ExpReverseBitsLenPreprocessedCols<F>*>(&trace_matrix[(initial_row_index + i) * num_cols]);
      F addr = addrs[initial_row_index + i];

      F prev_accum = accum;
      F prev_accum_squared = prev_accum * prev_accum;
      F multiplier;
      accum = prev_accum_squared;

      cols->iteration_num = F::from_canonical_u32(i);
      cols->is_real = F::one();
      cols->is_first = i == 0 ? F::one() : F::zero();
      cols->is_last = i == inst.addrs.len - 1 ? F::one() : F::zero();
      cols->x_mem.addr = inst.addrs.base;
      cols->x_mem.mult = i == 0 ? -F::one() : F::zero();
      cols->exponent_mem.addr = addr;
      cols->exponent_mem.mult = -F::one();
      cols->result_mem.addr = inst.addrs.result;
      cols->result_mem.mult = i == inst.addrs.len - 1 ? inst.mult : F::zero();
    }
  }

  template <class F>
  __global__ void exp_bits_events_to_trace_kernel(
    const ExpReverseBitsFfiEvent<F>* events, size_t count, F* bits, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const ExpReverseBitsFfiEvent<F>& e = events[idx];
    uint32_t initial_row_index = e.initial_row;

    F accum = F::one();

    // write out each row
    for (uintptr_t i = 0; i < e.len; i++) {
      ExpReverseBitsLenCols<F>* cols =
        reinterpret_cast<ExpReverseBitsLenCols<F>*>(&trace_matrix[(initial_row_index + i) * num_cols]);
      F bit = bits[initial_row_index + i];

      F prev_accum = accum;
      F prev_accum_squared = prev_accum * prev_accum;
      F multiplier;
      accum = prev_accum_squared;

      if (bit == F::one()) {
        multiplier = e.base;
        accum = accum * e.base;
      } else {
        multiplier = F::one();
      }

      cols->x = e.base;
      cols->current_bit = bit;
      cols->accum = accum;
      cols->accum_squared = accum * accum;
      cols->prev_accum_squared = prev_accum_squared;
      cols->multiplier = multiplier;
      cols->prev_accum_squared_times_multiplier = accum;
    }

    assert(e.result == accum);
  }

  template <class F>
  inline RustError generate_preprocessed(
    const ExpReverseBitsInstrFfi<F>* instrs,
    size_t num_instrs,
    const Address<F>* addrs,
    F* trace,
    size_t num_cols,
    cudaStream_t stream)
  {
    try {
      if (num_instrs == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      exp_bits_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(instrs, num_instrs, addrs, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main

  template <class F>
  inline RustError generate_main(
    const ExpReverseBitsFfiEvent<F>* events, size_t num_events, F* bits, F* trace, size_t num_cols, cudaStream_t stream)
  {
    try {
      if (num_events == 0) { return RustError{cudaSuccess}; }

      // launch kernel
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      exp_bits_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, num_events, bits, trace, num_cols);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::exp_bits

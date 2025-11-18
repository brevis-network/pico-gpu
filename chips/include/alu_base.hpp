#pragma once

#include <cassert>
#include <iostream>
#include <util/exception.cuh>
#include <util/rusterror.h>
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::alu_base {
  // Kernel implementation
  template <class F>
  __global__ void alu_base_instrs_to_trace_kernel(
    const BaseAluInstr<F>* instrs, size_t count, F* trace_matrix, size_t num_cols, F t, F f)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BaseAluInstr<F>& inst = instrs[idx];
    uint32_t row_index = idx / BASE_ALU_DATAPAR;
    uint32_t datapar_index = idx % BASE_ALU_DATAPAR;
    BaseAluPreprocessedCols<F>* cols =
      reinterpret_cast<BaseAluPreprocessedCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->accesses[datapar_index].addrs = inst.addrs;
    cols->accesses[datapar_index].is_add = f;
    cols->accesses[datapar_index].is_sub = f;
    cols->accesses[datapar_index].is_mul = f;
    cols->accesses[datapar_index].is_div = f;
    cols->accesses[datapar_index].mult = inst.mult;

    switch (inst.opcode) {
    case BaseAluOpcode::AddF:
      cols->accesses[datapar_index].is_add = t;
      break;
    case BaseAluOpcode::SubF:
      cols->accesses[datapar_index].is_sub = t;
      break;
    case BaseAluOpcode::MulF:
      cols->accesses[datapar_index].is_mul = t;
      break;
    case BaseAluOpcode::DivF:
      cols->accesses[datapar_index].is_div = t;
      break;
    default:
      assert(false);
      break;
    }
  }

  template <class F>
  __global__ void
  alu_base_events_to_trace_kernel(const BaseAluEvent<F>* events, size_t count, F* trace_matrix, size_t num_cols)
  {
    // Calculate thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const BaseAluEvent<F>& e = events[idx];
    uint32_t row_index = idx / BASE_ALU_DATAPAR;
    uint32_t datapar_index = idx % BASE_ALU_DATAPAR;
    BaseAluCols<F>* cols = reinterpret_cast<BaseAluCols<F>*>(&trace_matrix[row_index * num_cols]);

    cols->values[datapar_index].vals = e;
  }

  template <class F>
  inline RustError generate_preprocessed(
    const BaseAluInstr<F>* instrs, size_t num_instrs, F* trace, size_t num_cols, cudaStream_t stream)
  {
    assert(num_instrs > 0);

    try {
      // launch kernel
      const int block_size = 256;
      int grid_size = (num_instrs + block_size - 1) / block_size;
      F t = F::from_bool(true);
      F f = F::from_bool(false);
      alu_base_instrs_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(instrs, num_instrs, trace, num_cols, t, f);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_preprocessed

  template <class F>
  inline RustError
  generate_main(const BaseAluEvent<F>* events, size_t num_events, F* trace, size_t num_cols, cudaStream_t stream)
  {
    assert(num_events > 0);

    try {
      // launch kernel
      const int block_size = 256;
      int grid_size = (num_events + block_size - 1) / block_size;
      alu_base_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, num_events, trace, num_cols);

      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  } // generate_main
} // namespace pico_gpu::alu_base

// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/divrem.cuh"


namespace pico_gpu::mul_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  template <typename T>
  struct MulValueColsU64 {
    Word<T> a;
    Word<T> b;
    Word<T> c;
    MulGadgetU64<T> mul_gadget;
    T is_mul;
    T is_mulh;
    T is_mulhu;
    T is_mulhsu;
    T is_mulw;
  };

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, MulValueColsU64<F>& cols, F* byte_trace)
  {
    write_word_from_u64(cols.a, event.a);
    write_word_from_u64(cols.b, event.b);
    write_word_from_u64(cols.c, event.c);

    bool is_mulh = event.opcode == Opcode::MULH;
    bool is_mulhsu = event.opcode == Opcode::MULHSU;
    bool is_mulw = event.opcode == Opcode::MULW;

    populate_mul_gadget(cols.mul_gadget, byte_trace,
                        event.b, event.c, is_mulh, is_mulhsu, is_mulw);

    cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
    cols.is_mulh = F::from_bool(is_mulh);
    cols.is_mulhu = F::from_bool(event.opcode == Opcode::MULHU);
    cols.is_mulhsu = F::from_bool(is_mulhsu);
    cols.is_mulw = F::from_bool(is_mulw);
  }

  template <class F>
  __global__ void mul_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MulValueColsU64<F>* cols =
      reinterpret_cast<MulValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events[idx], *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      mul_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::mul_u64
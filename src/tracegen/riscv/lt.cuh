// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/divrem.cuh"


namespace pico_gpu::lt_u64 {
  using namespace byte;
  using divrem::write_word_from_u64;
  using divrem::LtWordU16GadgetU64;

  template <typename T>
  struct LtSignedGadgetU64 {
    LtWordU16GadgetU64<T> result;
    divrem::U16MSBGadgetU64<T> b_msb;
    divrem::U16MSBGadgetU64<T> c_msb;
  };

  template <typename T>
  struct LtValueColsU64 {
    T is_slt;
    T is_slt_u;
    Word<T> a;
    Word<T> b;
    Word<T> c;
    LtSignedGadgetU64<T> lt_signed;
    /// Merged branch comparison row (Rust `LtValueCols::is_cmp`); must stay last, like Rust.
    T is_cmp;
  };

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, LtValueColsU64<F>& cols, F* byte_trace)
  {
    uint64_t a = event.a;
    uint64_t b = event.b;
    uint64_t c = event.c;

    write_word_from_u64<F>(cols.a, a);
    write_word_from_u64<F>(cols.b, b);
    write_word_from_u64<F>(cols.c, c);

    bool is_signed = (event.opcode == Opcode::SLT);
    cols.is_slt = F::from_bool(is_signed);
    cols.is_slt_u = F::from_bool(event.opcode == Opcode::SLTU);
    // Merged branch comparison: `a = lt | gt << 16`, so `cols.a = [lt, gt, 0, 0]` and the gadget
    // below still takes its `bit` from limb 0 (same as the Rust event_to_row).
    cols.is_cmp = F::from_bool(event.is_cmp);

    uint16_t b_limbs[4] = {
      (uint16_t)(b), (uint16_t)(b >> 16),
      (uint16_t)(b >> 32), (uint16_t)(b >> 48)};
    uint16_t c_limbs[4] = {
      (uint16_t)(c), (uint16_t)(c >> 16),
      (uint16_t)(c >> 32), (uint16_t)(c >> 48)};

    if (is_signed) {
      divrem::populate_u16_msb(cols.lt_signed.b_msb, byte_trace, b_limbs[3]);
      divrem::populate_u16_msb(cols.lt_signed.c_msb, byte_trace, c_limbs[3]);

      uint64_t b_converted = b ^ (1ULL << 63);
      uint64_t c_converted = c ^ (1ULL << 63);
      divrem::populate_lt_word_u16(cols.lt_signed.result, byte_trace, a, b_converted, c_converted);
    } else {
      cols.lt_signed.b_msb.msb = F::zero();
      cols.lt_signed.c_msb.msb = F::zero();
      divrem::populate_lt_word_u16(cols.lt_signed.result, byte_trace, a, b, c);
    }
  }

  template <class F>
  __global__ void lt_events_to_trace_kernel(
    const AluEventU64* events, const size_t count,
    F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    LtValueColsU64<F>* cols =
      reinterpret_cast<LtValueColsU64<F>*>(&trace_matrix[idx * num_cols]);
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

      lt_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::lt_u64
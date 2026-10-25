// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cstdlib>
#include "field/ff_config.hpp"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ─── u64 shift left (u16-limb layout) ─────────────────────────────────────

#include "tracegen/riscv/divrem.cuh"

namespace pico_gpu::sll_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  static constexpr size_t SLL_WORD_SIZE = 4;

  /// u64 ShiftLeftValueCols matching Rust layout (#[repr(C)]).
  template <typename T>
  struct ShiftLeftValueColsU64 {
    Word<T> a;
    Word<T> b;
    Word<T> c;
    T c_bits[6];
    T v_01;
    T v_012;
    T v_0123;
    T shift_u16[4];
    Word<T> lower_limb;
    Word<T> higher_limb;
    Word<T> limb_result;
    U16MSBGadgetU64<T> sllw_msb;
    T is_sll;
    T is_sllw;
  };

  template <class F>
  __device__ inline void add_bit_range_check(F* byte_trace, uint16_t a, uint8_t num_bits) {
    uint8_t b = (uint8_t)((a >> 8) & 0xFF);
    uint8_t c = (uint8_t)(a & 0xFF);
    handle_bit_range_check_event<F>(byte_trace, b, c, num_bits);
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, ShiftLeftValueColsU64<F>& cols, F* byte_trace)
  {
    uint64_t a = event.a;
    uint64_t b = event.b;
    uint64_t c = event.c;

    uint16_t c_lo = (uint16_t)(c & 0xFFFF);

    if (event.opcode == Opcode::SLLW) {
      uint32_t sllw_val = (uint32_t)(((int64_t)(int32_t)b) << (c_lo & 0x1F));
      uint16_t sllw_limb1 = (uint16_t)((sllw_val >> 16) & 0xFFFF);
      populate_u16_msb(cols.sllw_msb, byte_trace, sllw_limb1);
    } else {
      cols.sllw_msb.msb = F::zero();
    }

    write_word_from_u64(cols.a, a);
    write_word_from_u64(cols.b, b);
    write_word_from_u64(cols.c, c);

    bool is_sll = (event.opcode == Opcode::SLL);
    cols.is_sll  = F::from_bool(is_sll);
    cols.is_sllw = F::from_bool(event.opcode == Opcode::SLLW);

    for (int i = 0; i < 6; ++i)
      cols.c_bits[i] = F::from_canonical_u16((c_lo >> i) & 1);

    add_bit_range_check<F>(byte_trace, (uint16_t)(c_lo >> 6), 10);

    cols.v_01   = F::from_canonical_u16(1u << (c_lo & 3));
    cols.v_012  = F::from_canonical_u16(1u << (c_lo & 7));
    cols.v_0123 = F::from_canonical_u16(1u << (c_lo & 15));

    uint32_t shift_amount = ((c_lo >> 4) & 1) + 2u * ((c_lo >> 5) & 1) * (is_sll ? 1u : 0u);
    for (int i = 0; i < 4; ++i)
      cols.shift_u16[i] = (i == (int)shift_amount) ? F::one() : F::zero();

    uint16_t b_limbs[4] = {
      (uint16_t)(b),         (uint16_t)(b >> 16),
      (uint16_t)(b >> 32),   (uint16_t)(b >> 48),
    };

    uint8_t bit_shift = (uint8_t)(c_lo & 0xF);

    for (int i = 0; i < 4; ++i) {
      uint32_t limb = (uint32_t)b_limbs[i];
      uint16_t lower_val = (uint16_t)(limb & ((1u << (16 - bit_shift)) - 1));
      uint16_t higher_val = (uint16_t)(limb >> (16 - bit_shift));
      cols.lower_limb._0[i]  = F::from_canonical_u16(lower_val);
      cols.higher_limb._0[i] = F::from_canonical_u16(higher_val);
      add_bit_range_check<F>(byte_trace, lower_val, (uint8_t)(16 - bit_shift));
      add_bit_range_check<F>(byte_trace, higher_val, bit_shift);
    }

    for (int i = 0; i < 4; ++i) {
      cols.limb_result._0[i] = cols.lower_limb._0[i] * F::from_canonical_u32(1u << bit_shift);
      if (i != 0)
        cols.limb_result._0[i] = cols.limb_result._0[i] + cols.higher_limb._0[i - 1];
    }
  }

  template <class F>
  __device__ inline void assign_padding_row(ShiftLeftValueColsU64<F>& cols)
  {
    cols.v_01   = F::one();
    cols.v_012  = F::one();
    cols.v_0123 = F::one();
  }

  template <class F>
  __global__ void sll_events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      ShiftLeftValueColsU64<F>* cols =
        reinterpret_cast<ShiftLeftValueColsU64<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size)
        event_to_row<F>(events[idx], *cols, byte_trace);
      else
        assign_padding_row<F>(*cols);
    }
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
      // No early return on zero events: a whitelist shape can list this chip with no events,
      // and its padding row is not all-zero, so the kernel runs over every row and writes it.

      const int block_size = 256;
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      sll_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sll_u64
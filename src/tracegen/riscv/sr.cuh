// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <tuple>
#include "field/ff_config.hpp"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ─── u64 shift right (u16-limb layout) ──────────────────────────────────

#include "tracegen/riscv/divrem.cuh"

namespace pico_gpu::sr_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  static constexpr size_t SR_WORD_SIZE = 4;

  template <typename T>
  struct ShiftRightValueColsU64 {
    Word<T> a;
    Word<T> b;
    Word<T> c_for_lookup;
    U16MSBGadgetU64<T> b_msb;
    U16MSBGadgetU64<T> srw_msb;
    T c_bits[6];
    T c;
    T imm_c;
    T sra_msb_v0123;
    T v_0123;
    T v_012;
    T v_01;
    Word<T> lower_limb;
    Word<T> higher_limb;
    T limb_result[SR_WORD_SIZE];
    T shift_u16[4];
    T is_srl;
    T is_sra;
    T is_srlw;
    T is_sraw;
    T is_w_imm;
  };

  template <class F>
  __device__ inline void add_bit_range_check(F* byte_trace, uint16_t a, uint8_t num_bits) {
    uint8_t b = (uint8_t)((a >> 8) & 0xFF);
    uint8_t c = (uint8_t)(a & 0xFF);
    handle_bit_range_check_event<F>(byte_trace, b, c, num_bits);
  }

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, ShiftRightValueColsU64<F>& cols, F* byte_trace)
  {
    uint64_t a = event.a;
    uint64_t b = event.b;
    uint64_t c = event.c;

    write_word_from_u64(cols.a, a);
    write_word_from_u64(cols.b, b);
    write_word_from_u64(cols.c_for_lookup, c);

    cols.is_srl  = F::from_bool(event.opcode == Opcode::SRL);
    cols.is_sra  = F::from_bool(event.opcode == Opcode::SRA);
    cols.is_srlw = F::from_bool(event.opcode == Opcode::SRLW);
    cols.is_sraw = F::from_bool(event.opcode == Opcode::SRAW);

    uint16_t c_val = (uint16_t)c;
    for (int i = 0; i < 6; ++i)
      cols.c_bits[i] = F::from_canonical_u32((c_val >> i) & 1);
    cols.c = F::from_canonical_u32(c_val);
    add_bit_range_check<F>(byte_trace, (uint16_t)(c_val >> 6), 10);

    cols.v_01   = F::from_canonical_u32(1u << (4 - (c_val & 3)));
    cols.v_012  = F::from_canonical_u32(1u << (8 - (c_val & 7)));
    cols.v_0123 = F::from_canonical_u32(1u << (16 - (c_val & 15)));

    uint16_t b_limbs[4] = {
      (uint16_t)(b),         (uint16_t)(b >> 16),
      (uint16_t)(b >> 32),   (uint16_t)(b >> 48),
    };

    if (event.opcode == Opcode::SRA) {
      populate_u16_msb(cols.b_msb, byte_trace, b_limbs[3]);
    } else if (event.opcode == Opcode::SRAW) {
      populate_u16_msb(cols.b_msb, byte_trace, b_limbs[1]);
    } else {
      cols.b_msb.msb = F::zero();
    }
    cols.sra_msb_v0123 = cols.b_msb.msb * cols.v_0123;

    bool is_word = event.opcode == Opcode::SRLW || event.opcode == Opcode::SRAW;
    uint32_t not_word = is_word ? 0u : 1u;

    if (is_word) {
      uint16_t a_limb1 = (uint16_t)((a >> 16) & 0xFFFF);
      populate_u16_msb(cols.srw_msb, byte_trace, a_limb1);
    } else {
      cols.srw_msb.msb = F::zero();
    }

    uint8_t bit_shift = (uint8_t)(c_val & 0xF);

    for (int i = 0; i < 4; ++i) {
      uint32_t limb = (uint32_t)b_limbs[i];
      uint16_t lower_val = (uint16_t)(limb & ((1u << bit_shift) - 1));
      uint16_t higher_val = (uint16_t)(limb >> bit_shift);
      cols.lower_limb._0[i]  = F::from_canonical_u16(lower_val);
      cols.higher_limb._0[i] = F::from_canonical_u16(higher_val);
      add_bit_range_check<F>(byte_trace, lower_val, bit_shift);
      add_bit_range_check<F>(byte_trace, higher_val, (uint8_t)(16 - bit_shift));
    }

    for (int i = 0; i < 4; ++i) {
      cols.limb_result[i] = cols.higher_limb._0[i];
      if (i != 3)
        cols.limb_result[i] = cols.limb_result[i]
          + cols.lower_limb._0[i + 1] * F::from_canonical_u32(1u << (16 - bit_shift));
    }

    uint32_t shift_amount = ((c_val >> 4) & 1) + 2 * ((c_val >> 5) & 1) * not_word;
    for (int i = 0; i < 4; ++i)
      cols.shift_u16[i] = (i == (int)shift_amount) ? F::one() : F::zero();

    bool is_w_imm = is_word && event.is_imm;
    cols.is_w_imm = F::from_bool(is_w_imm);
    cols.imm_c    = F::from_bool(event.is_imm);
  }

  template <class F>
  __device__ inline void assign_padding_row(ShiftRightValueColsU64<F>& cols)
  {
    cols.v_01   = F::from_canonical_u32(16);
    cols.v_012  = F::from_canonical_u32(256);
    cols.v_0123 = F::from_canonical_u32(65536);
  }

  template <class F>
  __global__ void sr_events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      ShiftRightValueColsU64<F>* cols =
        reinterpret_cast<ShiftRightValueColsU64<F>*>(&trace_matrix[idx * num_cols]);

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

      sr_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::sr_u64
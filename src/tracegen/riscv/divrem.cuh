// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include "common/rusterror.h"
#include "tracegen/common/is_zero.cuh"
#include "tracegen/common/is_zero_word.cuh"
#include "tracegen/common/is_equal_word.cuh"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"

namespace pico_gpu::divrem {
  using namespace pico_gpu::byte;
  using namespace pico_gpu;

  static constexpr size_t U64_WORD_SIZE = 4;
  static constexpr size_t U64_LONG_WORD_SIZE = 8;
  static constexpr size_t U64_WORD_BYTE_SIZE = 8;
  static constexpr size_t U64_LONG_WORD_BYTE_SIZE = 16;

  // ─── u64-specific gadget structs matching Rust layout ──────────────────

  template <typename T>
  struct U16MSBGadgetU64 {
    T msb;
  };

  template <typename T>
  struct U16ToU8GadgetU64 {
    T low_bytes[U64_WORD_SIZE];
  };

  template <typename T>
  struct MulGadgetU64 {
    T carry[U64_LONG_WORD_BYTE_SIZE];
    T product[U64_LONG_WORD_BYTE_SIZE];
    U16ToU8GadgetU64<T> b_lower_byte;
    U16ToU8GadgetU64<T> c_lower_byte;
    T b_msb;
    T c_msb;
    U16MSBGadgetU64<T> product_msb;
    T b_sign_extend;
    T c_sign_extend;
  };

  template <typename T>
  struct LtWordU16GadgetU64 {
    T u16_flags[U64_WORD_SIZE];
    T not_eq_inv;
    T comparison_limbs[2];
    T bit;
  };

  template <typename T>
  struct AddGadgetU64 {
    Word<T> value;
  };

  template <typename T>
  struct DivRemValueColsU64 {
    Word<T> a;
    Word<T> b;
    Word<T> c;
    // Raw register values, as the CPU put them on the ALU bus. For word
    // operations RV64 leaves the upper 32 bits of a register undefined, so
    // these differ from the computational `b`/`c` above and the AIR ties the
    // two together. Must stay in this position: the layout mirrors the Rust
    // `DivRemValueCols` field for field.
    Word<T> b_raw;
    Word<T> c_raw;
    Word<T> quotient;
    Word<T> quotient_comp;
    Word<T> remainder_comp;
    Word<T> remainder;
    Word<T> abs_remainder;
    Word<T> abs_c;
    Word<T> max_abs_c_or_1;
    T c_times_quotient[U64_LONG_WORD_SIZE];
    MulGadgetU64<T> c_times_quotient_lower;
    MulGadgetU64<T> c_times_quotient_upper;
    LtWordU16GadgetU64<T> remainder_lt_gadget;
    AddGadgetU64<T> c_neg_gadget;
    AddGadgetU64<T> rem_neg_gadget;
    T carry[U64_LONG_WORD_SIZE];
    IsZeroWordOperation<T> is_c_0;
    T is_div;
    T is_divu;
    T is_rem;
    T is_remu;
    T is_divw;
    T is_remw;
    T is_divuw;
    T is_remuw;
    T is_overflow;
    IsEqualWordOperation<T> is_overflow_b;
    IsEqualWordOperation<T> is_overflow_c;
    U16MSBGadgetU64<T> b_msb;
    U16MSBGadgetU64<T> rem_msb;
    U16MSBGadgetU64<T> c_msb;
    U16MSBGadgetU64<T> quot_msb;
    T b_neg;
    T b_neg_not_overflow;
    T b_not_neg_not_overflow;
    T is_real_not_word;
    T rem_neg;
    T c_neg;
    T abs_c_alu_event;
    T abs_rem_alu_event;
    T is_real;
    T remainder_check_multiplicity;
  };

  // ─── Word helpers for u64 (u16-limb encoding) ─────────────────────────

  template <class F>
  __device__ inline void write_word_from_u64(Word<F>& word, uint64_t v)
  {
    word._0[0] = F::from_canonical_u16((uint16_t)(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16((uint16_t)((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16((uint16_t)((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16((uint16_t)((v >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline uint64_t word_to_u64_val(const Word<F>& word)
  {
    return (uint64_t)word._0[0].as_canonical_u32()
         | ((uint64_t)word._0[1].as_canonical_u32() << 16)
         | ((uint64_t)word._0[2].as_canonical_u32() << 32)
         | ((uint64_t)word._0[3].as_canonical_u32() << 48);
  }

  // ─── Opcode classification ────────────────────────────────────────────

  __device__ inline bool is_signed_64bit_op(Opcode op) { return op == Opcode::DIV || op == Opcode::REM; }
  __device__ inline bool is_unsigned_64bit_op(Opcode op) { return op == Opcode::DIVU || op == Opcode::REMU; }
  __device__ inline bool is_64bit_op(Opcode op) { return is_signed_64bit_op(op) || is_unsigned_64bit_op(op); }
  __device__ inline bool is_signed_word_op(Opcode op) { return op == Opcode::DIVW || op == Opcode::REMW; }
  __device__ inline bool is_unsigned_word_op(Opcode op) { return op == Opcode::DIVUW || op == Opcode::REMUW; }
  __device__ inline bool is_word_op(Opcode op) { return is_signed_word_op(op) || is_unsigned_word_op(op); }
  __device__ inline bool is_signed_op(Opcode op) { return is_signed_64bit_op(op) || is_signed_word_op(op); }

  __device__ inline uint8_t get_msb_u64(uint64_t a) { return (uint8_t)((a >> 63) & 1); }

  // ─── Quotient / remainder (matching Rust get_quotient_and_remainder) ──

  __device__ inline void get_qr_u64(
    uint64_t b, uint64_t c, Opcode op, uint64_t& q, uint64_t& r)
  {
    if (c == 0 && is_64bit_op(op)) {
      q = UINT64_MAX; r = b; return;
    }
    if ((int32_t)c == 0 && is_word_op(op)) {
      q = UINT64_MAX; r = (uint64_t)(int64_t)(int32_t)b; return;
    }
    if (is_signed_64bit_op(op)) {
      int64_t sb = (int64_t)b, sc = (int64_t)c;
      if (sb == INT64_MIN && sc == -1) { q = (uint64_t)INT64_MIN; r = 0; }
      else { q = (uint64_t)(sb / sc); r = (uint64_t)(sb % sc); }
      return;
    }
    if (is_signed_word_op(op)) {
      int32_t sb = (int32_t)b, sc = (int32_t)c;
      if (sb == INT32_MIN && sc == -1) { q = (uint64_t)(int64_t)INT32_MIN; r = 0; }
      else { q = (uint64_t)(int64_t)(sb / sc); r = (uint64_t)(int64_t)(sb % sc); }
      return;
    }
    if (is_unsigned_word_op(op)) {
      uint32_t ub = (uint32_t)b, uc = (uint32_t)c;
      q = (uint64_t)(int64_t)(int32_t)(ub / uc);
      r = (uint64_t)(int64_t)(int32_t)(ub % uc);
      return;
    }
    q = b / c; r = b % c;
  }

  // ─── Gadget populate functions ────────────────────────────────────────

  template <class F>
  __device__ inline bool populate_is_zero_word_u64(IsZeroWordOperation<F>& g, uint64_t v)
  {
    Word<F> w; write_word_from_u64(w, v);
    bool z = true;
    for (int i = 0; i < 4; ++i)
      z &= is_zero::populate_from_field_element(g.is_zero_byte[i], w._0[i]) == 1;
    g.is_lower_half_zero = g.is_zero_byte[0].result * g.is_zero_byte[1].result;
    g.is_upper_half_zero = g.is_zero_byte[2].result * g.is_zero_byte[3].result;
    g.result = F::from_bool(z);
    return z;
  }

  template <class F>
  __device__ inline bool populate_is_zero_word_field(IsZeroWordOperation<F>& g, Word<F>& w)
  {
    bool z = true;
    for (int i = 0; i < 4; ++i)
      z &= is_zero::populate_from_field_element(g.is_zero_byte[i], w._0[i]) == 1;
    g.is_lower_half_zero = g.is_zero_byte[0].result * g.is_zero_byte[1].result;
    g.is_upper_half_zero = g.is_zero_byte[2].result * g.is_zero_byte[3].result;
    g.result = F::from_bool(z);
    return z;
  }

  template <class F>
  __device__ inline bool populate_is_equal_word_u64(
    IsEqualWordOperation<F>& g, uint64_t a, uint64_t b)
  {
    uint16_t al[4] = {
      (uint16_t)(a), (uint16_t)(a >> 16), (uint16_t)(a >> 32), (uint16_t)(a >> 48)};
    uint16_t bl[4] = {
      (uint16_t)(b), (uint16_t)(b >> 16), (uint16_t)(b >> 32), (uint16_t)(b >> 48)};
    Word<F> diff;
    for (int i = 0; i < 4; ++i)
      diff._0[i] = F::from_canonical_u16(al[i]) - F::from_canonical_u16(bl[i]);
    populate_is_zero_word_field(g.is_diff_zero, diff);
    return a == b;
  }

  template <class F>
  __device__ inline void populate_u16_msb(
    U16MSBGadgetU64<F>& g, F* byte_trace, uint16_t a_u16)
  {
    uint16_t msb = (a_u16 >> 15) & 1;
    g.msb = F::from_canonical_u16(msb);
    add_u16_range_check<F>(byte_trace, (uint16_t)(a_u16 * 2));
  }

  template <class F>
  __device__ inline void populate_u16_to_u8(U16ToU8GadgetU64<F>& g, F* byte_trace, uint64_t v)
  {
    uint16_t limbs[4] = {
      (uint16_t)(v & 0xFFFF), (uint16_t)((v >> 16) & 0xFFFF),
      (uint16_t)((v >> 32) & 0xFFFF), (uint16_t)((v >> 48) & 0xFFFF)};
    for (int i = 0; i < 4; ++i) {
      uint8_t low_byte  = (uint8_t)(limbs[i] & 0xFF);
      uint8_t high_byte = (uint8_t)((limbs[i] >> 8) & 0xFF);
      g.low_bytes[i] = F::from_canonical_u8(low_byte);
      add_u8_range_check<F>(byte_trace, low_byte, high_byte);
    }
  }

  template <class F>
  __device__ inline void populate_add_gadget(
    AddGadgetU64<F>& g, F* byte_trace, uint64_t a, uint64_t b)
  {
    uint64_t result = a + b;
    write_word_from_u64(g.value, result);
    uint16_t limbs[4] = {
      (uint16_t)(result), (uint16_t)(result >> 16),
      (uint16_t)(result >> 32), (uint16_t)(result >> 48)};
    for (int i = 0; i < 4; ++i)
      add_u16_range_check<F>(byte_trace, limbs[i]);
  }

  template <class F>
  __device__ inline void populate_lt_word_u16(
    LtWordU16GadgetU64<F>& g, F* byte_trace, uint64_t a_u64, uint64_t b_u64, uint64_t c_u64)
  {
    uint16_t al[4] = {
      (uint16_t)(a_u64), (uint16_t)(a_u64 >> 16),
      (uint16_t)(a_u64 >> 32), (uint16_t)(a_u64 >> 48)};
    uint16_t bl[4] = {
      (uint16_t)(b_u64), (uint16_t)(b_u64 >> 16),
      (uint16_t)(b_u64 >> 32), (uint16_t)(b_u64 >> 48)};
    uint16_t cl[4] = {
      (uint16_t)(c_u64), (uint16_t)(c_u64 >> 16),
      (uint16_t)(c_u64 >> 32), (uint16_t)(c_u64 >> 48)};

    uint16_t a_u16 = al[0];

    for (int i = 0; i < 4; ++i) g.u16_flags[i] = F::zero();
    g.not_eq_inv = F::zero();
    g.comparison_limbs[0] = F::zero();
    g.comparison_limbs[1] = F::zero();

    uint16_t comparison_limbs_raw[2] = {0, 0};
    for (int i = 3; i >= 0; --i) {
      if (bl[i] != cl[i]) {
        g.u16_flags[i] = F::one();
        comparison_limbs_raw[0] = bl[i];
        comparison_limbs_raw[1] = cl[i];
        F bf = F::from_canonical_u16(bl[i]);
        F cf = F::from_canonical_u16(cl[i]);
        g.not_eq_inv = (bf - cf).reciprocal();
        g.comparison_limbs[0] = bf;
        g.comparison_limbs[1] = cf;
        break;
      }
    }

    g.bit = F::from_canonical_u16(a_u16);
    uint16_t diff = comparison_limbs_raw[0] - comparison_limbs_raw[1];
    add_u16_range_check<F>(byte_trace, diff);
  }

  template <class F>
  __device__ inline void populate_mul_gadget(
    MulGadgetU64<F>& mg, F* byte_trace,
    uint64_t b_u64, uint64_t c_u64,
    bool is_mulh, bool is_mulhsu, bool is_mulw)
  {
    uint8_t b_bytes[8], c_bytes[8];
    for (int i = 0; i < 8; ++i) { b_bytes[i] = (uint8_t)(b_u64 >> (8 * i)); }
    for (int i = 0; i < 8; ++i) { c_bytes[i] = (uint8_t)(c_u64 >> (8 * i)); }

    if (is_mulw) {
      int32_t mulw_val = (int32_t)b_u64 * (int32_t)c_u64;
      uint64_t mulw_u64 = (uint64_t)(int64_t)mulw_val;
      uint16_t limb1 = (uint16_t)((mulw_u64 >> 16) & 0xFFFF);
      populate_u16_msb(mg.product_msb, byte_trace, limb1);
    } else {
      mg.product_msb.msb = F::zero();
    }

    populate_u16_to_u8(mg.b_lower_byte, byte_trace, b_u64);
    populate_u16_to_u8(mg.c_lower_byte, byte_trace, c_u64);

    uint8_t bm = (b_bytes[7] >> 7) & 1;
    uint8_t cm = (c_bytes[7] >> 7) & 1;
    mg.b_msb = F::from_canonical_u8(bm);
    mg.c_msb = F::from_canonical_u8(cm);

    int b_len = 8, c_len = 8;
    uint8_t b_ext[16] = {0}, c_ext[16] = {0};
    for (int i = 0; i < 8; ++i) { b_ext[i] = b_bytes[i]; c_ext[i] = c_bytes[i]; }

    if ((is_mulh || is_mulhsu) && bm == 1) {
      mg.b_sign_extend = F::one();
      for (int i = 8; i < 16; ++i) b_ext[i] = 0xFF;
      b_len = 16;
    } else {
      mg.b_sign_extend = F::zero();
    }
    if (is_mulh && cm == 1) {
      mg.c_sign_extend = F::one();
      for (int i = 8; i < 16; ++i) c_ext[i] = 0xFF;
      c_len = 16;
    } else {
      mg.c_sign_extend = F::zero();
    }

    handle_byte_lookup_event<F>(byte_trace, ByteOpcode::MSB, b_bytes[7], 0);
    handle_byte_lookup_event<F>(byte_trace, ByteOpcode::MSB, c_bytes[7], 0);

    uint32_t product[16] = {0};
    for (int i = 0; i < b_len; ++i)
      for (int j = 0; j < c_len; ++j)
        if (i + j < 16)
          product[i + j] += (uint32_t)b_ext[i] * (uint32_t)c_ext[j];

    const uint32_t base = 256;
    uint32_t carry_arr[16] = {0};
    for (int i = 0; i < 16; ++i) {
      carry_arr[i] = product[i] / base;
      product[i] %= base;
      if (i + 1 < 16) product[i + 1] += carry_arr[i];
      mg.carry[i] = F::from_canonical_u32(carry_arr[i]);
    }
    for (int i = 0; i < 16; ++i)
      mg.product[i] = F::from_canonical_u32(product[i]);

    uint16_t carry_u16[16];
    for (int i = 0; i < 16; ++i) carry_u16[i] = (uint16_t)(carry_arr[i] & 0xFFFF);
    add_u16_range_checks<F>(byte_trace, carry_u16, 16);

    for (int i = 0; i < 16; i += 2)
      add_u8_range_check<F>(byte_trace, (uint8_t)product[i], (uint8_t)product[i + 1]);
  }

  // ─── Main event_to_row ────────────────────────────────────────────────

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event,
    DivRemValueColsU64<F>& cols,
    F* byte_trace,
    AluEventU64* alu_add_sub_events)
  {
    const Opcode op = event.opcode;

    // Compute the corrected b and c for the operation
    uint64_t b_comp, c_comp;
    if (is_signed_word_op(op)) {
      b_comp = (uint64_t)(int64_t)(int32_t)event.b;
      c_comp = (uint64_t)(int64_t)(int32_t)event.c;
    } else if (is_unsigned_word_op(op)) {
      b_comp = (uint64_t)(uint32_t)event.b;
      c_comp = (uint64_t)(uint32_t)event.c;
    } else {
      b_comp = event.b;
      c_comp = event.c;
    }

    // Basic operands and opcode flags
    write_word_from_u64(cols.a, event.a);
    write_word_from_u64(cols.b, b_comp);
    write_word_from_u64(cols.c, c_comp);
    // The ALU bus carries the raw register values, not the truncated /
    // sign-extended ones the arithmetic uses.
    write_word_from_u64(cols.b_raw, event.b);
    write_word_from_u64(cols.c_raw, event.c);
    cols.is_real = F::one();
    cols.is_div   = F::from_bool(op == Opcode::DIV);
    cols.is_divu  = F::from_bool(op == Opcode::DIVU);
    cols.is_rem   = F::from_bool(op == Opcode::REM);
    cols.is_remu  = F::from_bool(op == Opcode::REMU);
    cols.is_divw  = F::from_bool(op == Opcode::DIVW);
    cols.is_remw  = F::from_bool(op == Opcode::REMW);
    cols.is_divuw = F::from_bool(op == Opcode::DIVUW);
    cols.is_remuw = F::from_bool(op == Opcode::REMUW);

    F not_word = F::one() - cols.is_divw - cols.is_remw - cols.is_divuw - cols.is_remuw;
    cols.is_real_not_word = cols.is_real * not_word;

    populate_is_zero_word_u64(cols.is_c_0, c_comp);

    // Quotient and remainder
    uint64_t quotient, remainder;
    get_qr_u64(event.b, event.c, op, quotient, remainder);
    write_word_from_u64(cols.quotient, quotient);
    write_word_from_u64(cols.remainder, remainder);

    uint64_t quotient_comp = is_unsigned_word_op(op)
      ? (uint64_t)(uint32_t)quotient : quotient;
    uint64_t remainder_comp = is_unsigned_word_op(op)
      ? (uint64_t)(uint32_t)remainder : remainder;
    write_word_from_u64(cols.quotient_comp, quotient_comp);
    write_word_from_u64(cols.remainder_comp, remainder_comp);

    // Sign detection and absolute values
    if (is_signed_64bit_op(op)) {
      cols.rem_neg = F::from_canonical_u8(get_msb_u64(remainder));
      cols.b_neg   = F::from_canonical_u8(get_msb_u64(event.b));
      cols.c_neg   = F::from_canonical_u8(get_msb_u64(event.c));
      cols.is_overflow = F::from_bool((int64_t)event.b == INT64_MIN && (int64_t)event.c == -1);

      uint64_t abs_rem = (uint64_t)( ((int64_t)remainder < 0) ? -(int64_t)remainder : (int64_t)remainder );
      uint64_t abs_c_val = (uint64_t)( ((int64_t)event.c < 0) ? -(int64_t)event.c : (int64_t)event.c );
      write_word_from_u64(cols.abs_remainder, abs_rem);
      write_word_from_u64(cols.abs_c, abs_c_val);
      uint64_t max_c = abs_c_val > 1 ? abs_c_val : 1;
      write_word_from_u64(cols.max_abs_c_or_1, max_c);
    } else if (is_signed_word_op(op)) {
      cols.rem_neg = F::from_canonical_u8(get_msb_u64((uint64_t)(int64_t)(int32_t)remainder));
      cols.b_neg   = F::from_canonical_u8(get_msb_u64((uint64_t)(int64_t)(int32_t)event.b));
      cols.c_neg   = F::from_canonical_u8(get_msb_u64((uint64_t)(int64_t)(int32_t)event.c));
      cols.is_overflow = F::from_bool((int32_t)event.b == INT32_MIN && (int32_t)event.c == -1);

      uint64_t abs_rem = (uint64_t)( ((int64_t)remainder < 0) ? -(int64_t)remainder : (int64_t)remainder );
      uint64_t abs_c_val = (uint64_t)( ((int64_t)c_comp < 0) ? -(int64_t)c_comp : (int64_t)c_comp );
      write_word_from_u64(cols.abs_remainder, abs_rem);
      write_word_from_u64(cols.abs_c, abs_c_val);
      uint64_t max_c = abs_c_val > 1 ? abs_c_val : 1;
      write_word_from_u64(cols.max_abs_c_or_1, max_c);
    } else if (is_unsigned_word_op(op)) {
      cols.abs_remainder = cols.remainder_comp;
      write_word_from_u64(cols.abs_c, (uint64_t)(uint32_t)event.c);
      uint32_t mc = (uint32_t)event.c > 1 ? (uint32_t)event.c : 1;
      write_word_from_u64(cols.max_abs_c_or_1, (uint64_t)mc);
    } else {
      cols.abs_remainder = cols.remainder_comp;
      write_word_from_u64(cols.abs_c, event.c);
      uint64_t mc = event.c > 1 ? event.c : 1;
      write_word_from_u64(cols.max_abs_c_or_1, mc);
    }

    // Overflow checks
    if (is_word_op(op)) {
      populate_is_equal_word_u64(cols.is_overflow_b, (uint64_t)(uint32_t)event.b,
                                 (uint64_t)(uint32_t)INT32_MIN);
      populate_is_equal_word_u64(cols.is_overflow_c, (uint64_t)(uint32_t)event.c,
                                 (uint64_t)(uint32_t)(int32_t)-1);
    } else {
      populate_is_equal_word_u64(cols.is_overflow_b, event.b, (uint64_t)INT64_MIN);
      populate_is_equal_word_u64(cols.is_overflow_c, event.c, (uint64_t)(int64_t)-1);
    }

    cols.b_neg_not_overflow = cols.b_neg * (F::one() - cols.is_overflow);
    cols.b_not_neg_not_overflow = (F::one() - cols.b_neg) * (F::one() - cols.is_overflow);

    cols.abs_c_alu_event   = cols.c_neg * cols.is_real;
    cols.abs_rem_alu_event = cols.rem_neg * cols.is_real;

    // Range-check abs_c and abs_remainder limbs
    for (int i = 0; i < 4; ++i) {
      add_u16_range_check<F>(byte_trace, (uint16_t)cols.abs_c._0[i].as_canonical_u32());
      add_u16_range_check<F>(byte_trace, (uint16_t)cols.abs_remainder._0[i].as_canonical_u32());
    }

    // Populate c_neg / rem_neg gadgets
    if (cols.abs_c_alu_event == F::one())
      populate_add_gadget(cols.c_neg_gadget, byte_trace,
                          word_to_u64_val(cols.c), word_to_u64_val(cols.abs_c));
    if (cols.abs_rem_alu_event == F::one())
      populate_add_gadget(cols.rem_neg_gadget, byte_trace,
                          word_to_u64_val(cols.remainder), word_to_u64_val(cols.abs_remainder));

    // MSB lookups
    if (is_word_op(op)) {
      populate_u16_msb(cols.b_msb, byte_trace, (uint16_t)(event.b >> 16));
      populate_u16_msb(cols.c_msb, byte_trace, (uint16_t)(event.c >> 16));
      populate_u16_msb(cols.rem_msb, byte_trace, (uint16_t)(remainder >> 16));
      populate_u16_msb(cols.quot_msb, byte_trace, (uint16_t)(quotient >> 16));
    } else {
      populate_u16_msb(cols.b_msb, byte_trace, (uint16_t)(b_comp >> 48));
      populate_u16_msb(cols.c_msb, byte_trace, (uint16_t)(c_comp >> 48));
      populate_u16_msb(cols.rem_msb, byte_trace, (uint16_t)(remainder >> 48));
    }

    // Remainder < abs(c) check
    cols.remainder_check_multiplicity = cols.is_real * (F::one() - cols.is_c_0.result);
    if (cols.remainder_check_multiplicity == F::one())
      populate_lt_word_u16(cols.remainder_lt_gadget, byte_trace,
                           1,
                           word_to_u64_val(cols.abs_remainder),
                           word_to_u64_val(cols.max_abs_c_or_1));

    // ── c * quotient + remainder = b verification ──

    // Compute 128-bit product as bytes
    uint64_t prod_lo = quotient_comp * c_comp;
    uint64_t prod_hi;
    if (is_signed_64bit_op(op) || is_signed_word_op(op))
      prod_hi = (uint64_t)__mul64hi((long long)quotient_comp, (long long)c_comp);
    else
      prod_hi = __umul64hi(quotient_comp, c_comp);

    uint8_t prod_bytes[16];
    for (int i = 0; i < 8; ++i) prod_bytes[i]     = (uint8_t)(prod_lo >> (8 * i));
    for (int i = 0; i < 8; ++i) prod_bytes[8 + i]  = (uint8_t)(prod_hi >> (8 * i));

    uint16_t c_times_q_u16[8];
    for (int i = 0; i < 8; ++i)
      c_times_q_u16[i] = (uint16_t)prod_bytes[2 * i]
                        | ((uint16_t)prod_bytes[2 * i + 1] << 8);

    for (int i = 0; i < 8; ++i)
      cols.c_times_quotient[i] = F::from_canonical_u16(c_times_q_u16[i]);

    // Populate MulGadgets
    populate_mul_gadget(cols.c_times_quotient_lower, byte_trace,
                        quotient_comp, c_comp, false, false, false);

    if (is_signed_64bit_op(op))
      populate_mul_gadget(cols.c_times_quotient_upper, byte_trace,
                          quotient_comp, c_comp, true, false, false);
    else if (is_unsigned_64bit_op(op))
      populate_mul_gadget(cols.c_times_quotient_upper, byte_trace,
                          quotient_comp, c_comp, false, false, false);

    // Carry propagation: c*quotient + remainder (sign-extended)
    uint32_t rem_u16[8];
    for (int i = 0; i < 4; ++i) {
      rem_u16[i]     = cols.remainder_comp._0[i].as_canonical_u32();
      rem_u16[i + 4] = cols.rem_neg.as_canonical_u32() * ((1u << 16) - 1);
    }

    uint32_t carry_acc[8] = {0};
    const uint32_t carry_base = 1u << 16;
    for (int i = 0; i < 8; ++i) {
      uint32_t x = (uint32_t)c_times_q_u16[i] + rem_u16[i];
      if (i > 0) x += carry_acc[i - 1];
      carry_acc[i] = x / carry_base;
      cols.carry[i] = F::from_canonical_u32(carry_acc[i]);
      add_u16_range_check<F>(byte_trace, (uint16_t)(x & 0xFFFF));
    }

    // ── Generate dependent ALU events ───────────────────────────────

    // Add events for absolute value computation. At most two, and the packed wrapper in
    // packed_tracegen.cuh sizes a stack array on that — a third emitter here overruns it.
    // Build each event zero-initialized (as cpu.cuh does) and store it whole: the slots come
    // from a reused device pool, so writing only clk/opcode/a/b/c left `is_imm` / `is_cmp` and
    // the padding holding whatever the previous tenant wrote.
    {
      size_t add_idx = 0;
      if (cols.abs_c_alu_event == F::one()) {
        AluEventU64 add_event = {};
        add_event.clk    = event.clk;
        add_event.opcode = Opcode::ADD;
        add_event.a      = 0;
        add_event.b      = c_comp;
        add_event.c      = word_to_u64_val(cols.abs_c);
        alu_add_sub_events[add_idx++] = add_event;
      }
      if (cols.abs_rem_alu_event == F::one()) {
        AluEventU64 add_event = {};
        add_event.clk    = event.clk;
        add_event.opcode = Opcode::ADD;
        add_event.a      = 0;
        add_event.b      = remainder;
        add_event.c      = word_to_u64_val(cols.abs_remainder);
        alu_add_sub_events[add_idx++] = add_event;
      }
    }

    // Range checks for quotient, remainder, c_times_quotient
    {
      uint16_t q_limbs[4] = {
        (uint16_t)(quotient), (uint16_t)(quotient >> 16),
        (uint16_t)(quotient >> 32), (uint16_t)(quotient >> 48)};
      add_u16_range_checks<F>(byte_trace, q_limbs, 4);

      uint16_t r_limbs[4] = {
        (uint16_t)(remainder), (uint16_t)(remainder >> 16),
        (uint16_t)(remainder >> 32), (uint16_t)(remainder >> 48)};
      add_u16_range_checks<F>(byte_trace, r_limbs, 4);

      add_u16_range_checks<F>(byte_trace, c_times_q_u16, 8);
    }
  }

  // ─── Padding row ─────────────────────────────────────────────────────

  template <class F>
  __device__ inline void assign_padding_row(DivRemValueColsU64<F>& cols)
  {
    cols.is_divu = F::one();
    write_word_from_u64(cols.c, 1);
    // Padding is a non-word op, so the b/c alignment constraints require
    // `*_raw == *` on every limb. `b_raw` stays zero with `b`.
    write_word_from_u64(cols.c_raw, 1);
    write_word_from_u64(cols.abs_c, 1);
    write_word_from_u64(cols.max_abs_c_or_1, 1);
    cols.b_not_neg_not_overflow = F::one();
    populate_is_zero_word_u64(cols.is_c_0, 1);
  }

  // ─── Kernel ──────────────────────────────────────────────────────────

  template <class F>
  __global__ void events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace_matrix,
    const size_t num_cols,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEventU64* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      DivRemValueColsU64<F>* cols =
        reinterpret_cast<DivRemValueColsU64<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size) {
        event_to_row<F>(
          events[idx], *cols, byte_trace,
          &alu_add_sub_events[alu_add_sub_events_start_index + extra_event_indices[idx].add]);
      } else {
        assign_padding_row<F>(*cols);
      }
    }
  }

  // ─── Entry point ─────────────────────────────────────────────────────

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEventU64* events,
    F* trace,
    cudaStream_t stream,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEventU64* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      // No early return on zero events: a whitelist shape can list this chip with no events,
      // and its padding row is not all-zero, so the kernel runs over every row and writes it.

      const int block_size = 256;
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, extra_event_indices,
        byte_trace,
        alu_add_sub_events, alu_add_sub_events_start_index);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::divrem

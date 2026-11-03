// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ─── u64 bitwise (BitwiseValueCols matching Rust layout) ────────────────────

#include "tracegen/riscv/divrem.cuh"

namespace pico_gpu::bitwise_u64 {
  using namespace pico_gpu::byte;
  using namespace pico_gpu::divrem;

  static constexpr size_t WORD_BYTE_SIZE = 8;

  /// BitwiseValueColsU64 matching Rust BitwiseValueCols layout (#[repr(C)]).
  /// Layout: b, c, BitWiseU16Gadget { U16ToU8Gadget(b), U16ToU8Gadget(c), BitWiseGadget(result) }, is_xor, is_or, is_and
  template <typename T>
  struct BitwiseValueColsU64 {
    Word<T> b;
    Word<T> c;
    // U16ToU8Gadget for b: low_bytes[4]
    T b_low_bytes[4];
    // U16ToU8Gadget for c: low_bytes[4]
    T c_low_bytes[4];
    // BitWiseGadget: result[8]
    T result[WORD_BYTE_SIZE];
    T is_xor;
    T is_or;
    T is_and;
  };

  template <class F>
  __device__ inline void event_to_row(
    const AluEventU64& event, BitwiseValueColsU64<F>& cols, F* byte_trace)
  {
    uint64_t b = event.b;
    uint64_t c = event.c;
    uint64_t a = event.a;

    write_word_from_u64(cols.b, b);
    write_word_from_u64(cols.c, c);

    cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
    cols.is_or = F::from_bool(event.opcode == Opcode::OR);
    cols.is_and = F::from_bool(event.opcode == Opcode::AND);

    // Populate U16ToU8Gadget: extract low byte from each u16 limb
    // b as 4 u16 limbs
    uint16_t b_limbs[4] = {
      (uint16_t)(b & 0xFFFF), (uint16_t)((b >> 16) & 0xFFFF),
      (uint16_t)((b >> 32) & 0xFFFF), (uint16_t)((b >> 48) & 0xFFFF)
    };
    uint16_t c_limbs[4] = {
      (uint16_t)(c & 0xFFFF), (uint16_t)((c >> 16) & 0xFFFF),
      (uint16_t)((c >> 32) & 0xFFFF), (uint16_t)((c >> 48) & 0xFFFF)
    };
    for (int i = 0; i < 4; ++i) {
      cols.b_low_bytes[i] = F::from_canonical_u8((uint8_t)(b_limbs[i] & 0xFF));
      cols.c_low_bytes[i] = F::from_canonical_u8((uint8_t)(c_limbs[i] & 0xFF));
    }

    const array_t<uint8_t, WORD_BYTE_SIZE> a_bytes = u64_to_le_bytes(a);
    const array_t<uint8_t, WORD_BYTE_SIZE> b_bytes = u64_to_le_bytes(b);
    const array_t<uint8_t, WORD_BYTE_SIZE> c_bytes = u64_to_le_bytes(c);

    ByteOpcode byte_opcode = ByteOpcode::AND;
    if (event.opcode == Opcode::XOR) byte_opcode = ByteOpcode::XOR;
    else if (event.opcode == Opcode::OR) byte_opcode = ByteOpcode::OR;
    else if (event.opcode == Opcode::AND) byte_opcode = ByteOpcode::AND;

    for (uintptr_t i = 0; i < WORD_BYTE_SIZE; ++i) {
      cols.result[i] = F::from_canonical_u8(a_bytes[i]);
      handle_byte_lookup_event(byte_trace, byte_opcode, b_bytes[i], c_bytes[i]);
    }
  }
} // namespace pico_gpu::bitwise_u64
#pragma once

#include <cmath>
#include <cstddef>
#include <tuple>
#include <util/rusterror.h>
#include "prelude.hpp"

namespace pico_gpu {
  // Compiles to a no-op with -O3 and the like.
  __PICO_HOSTDEV__ inline array_t<uint8_t, 4> u32_to_le_bytes(uint32_t n)
  {
    return {
      (uint8_t)(n >> 8 * 0),
      (uint8_t)(n >> 8 * 1),
      (uint8_t)(n >> 8 * 2),
      (uint8_t)(n >> 8 * 3),
    };
  }

  __PICO_HOSTDEV__ inline array_t<uint8_t, 8> u64_to_le_bytes(uint64_t n)
  {
    return {
      (uint8_t)(n >> 8 * 0), (uint8_t)(n >> 8 * 1), (uint8_t)(n >> 8 * 2), (uint8_t)(n >> 8 * 3),
      (uint8_t)(n >> 8 * 4), (uint8_t)(n >> 8 * 5), (uint8_t)(n >> 8 * 6), (uint8_t)(n >> 8 * 7),
    };
  }

  template <class F>
  __PICO_HOSTDEV__ inline void write_word_from_u32_v2(Word<F>& word, const uint32_t value)
  {
    word._0[0] = F::from_canonical_u8(value);
    word._0[1] = F::from_canonical_u8(value >> 8);
    word._0[2] = F::from_canonical_u8(value >> 16);
    word._0[3] = F::from_canonical_u8(value >> 24);
  }

  template <class F>
  __PICO_HOSTDEV__ inline void word_from_le_bytes(Word<F>& word, const array_t<uint8_t, 4> bytes)
  {
    // Coercion to `uint8_t` truncates the number.
    word._0[0] = F::from_canonical_u8(bytes[0]);
    word._0[1] = F::from_canonical_u8(bytes[1]);
    word._0[2] = F::from_canonical_u8(bytes[2]);
    word._0[3] = F::from_canonical_u8(bytes[3]);
  }

  /// Shifts a byte to the right and returns both the shifted byte and the bits that carried.
  __PICO_HOSTDEV__ inline tuple_t<uint8_t, uint8_t> shr_carry(uint8_t input, uint8_t rotation)
  {
    uint8_t c_mod = rotation & 0x7;
    if (c_mod != 0) {
      uint8_t res = input >> c_mod;
      uint8_t c_mod_comp = 8 - c_mod;
      uint8_t carry = (uint8_t)(input << c_mod_comp) >> c_mod_comp;
      return {res, carry};
    } else {
      return {input, 0};
    }
  }

  template <class F>
  __PICO_HOSTDEV__ inline uint32_t word_to_u32(const Word<F>& word)
  {
    return (word._0[0].as_canonical_u32()) + (word._0[1].as_canonical_u32() << 8) +
           (word._0[2].as_canonical_u32() << 16) + (word._0[3].as_canonical_u32() << 24);
  }

  template <class F>
  __PICO_HOSTDEV__ inline void word_from_le_bytes(Word<decltype(F::val)>& word, const array_t<uint8_t, 4> bytes)
  {
    // Coercion to `uint8_t` truncates the number.
    word._0[0] = F::from_canonical_u8(bytes[0]).val;
    word._0[1] = F::from_canonical_u8(bytes[1]).val;
    word._0[2] = F::from_canonical_u8(bytes[2]).val;
    word._0[3] = F::from_canonical_u8(bytes[3]).val;
  }

  __PICO_HOSTDEV__ inline uint8_t get_msb(const array_t<uint8_t, WORD_SIZE> a)
  {
    return (a[WORD_SIZE - 1] >> (BYTE_SIZE - 1)) & 1;
  }

  // return `true` if the given `opcode` is a signed operation
  __PICO_HOSTDEV__ inline bool is_signed_operation(Opcode opcode)
  {
    return opcode == Opcode::DIV || opcode == Opcode::REM;
  }

  // calculate the `quotient` for the given `b` and `c` per RISC-V spec
  __PICO_HOSTDEV__ inline uint32_t get_quotient(uint32_t b, uint32_t c, Opcode opcode)
  {
    if (c == 0) {
      // when c is 0, the quotient is 2^32 - 1
      return UINT32_MAX;
    } else if (is_signed_operation(opcode)) {
      return (uint32_t)((int32_t)b / (int32_t)c);
    } else {
      return b / c;
    }
  }

  // calculate the `remainder` for the given `b` and `c` per RISC-V spec
  __PICO_HOSTDEV__ inline uint32_t get_remainder(uint32_t b, uint32_t c, Opcode opcode)
  {
    if (c == 0) {
      // when c is 0, the remainder is b
      return b;
    } else if (is_signed_operation(opcode)) {
      return (uint32_t)((int32_t)b % (int32_t)c);
    } else {
      return b % c;
    }
  }

  __PICO_HOSTDEV__ inline int next_power_of_two(int x)
  {
    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
  }

  /* --- bit helpers identical to Rust versions ---------------------------- */
  template <class F>
  __PICO_HOSTDEV__ inline array_t<F, 64> u64_to_bits_le(uint64_t v)
  {
    array_t<F, 64> out;
    for (int i = 0; i < 64; ++i)
      out[i] = F::from_bool((v >> i) & 1u);
    return out;
  }

  template <class F>
  __PICO_HOSTDEV__ inline array_t<F, 4> u64_to_16_bit_limbs(uint64_t v)
  {
    array_t<F, 4> out;
    for (int limb = 0; limb < 4; ++limb)
      out[limb] = F::from_canonical_u16((v >> (16 * limb)) & 0xFFFFu);
    return out;
  }

  __PICO_HOSTDEV__ inline constexpr uint64_t rotl64(uint64_t v, unsigned r)
  {
    // return std::rotl(v, r);
    r &= 63u;
    return (v << r) | (v >> ((64u - r) & 63u));
  }

  __PICO_HOSTDEV__ inline constexpr uint64_t rotr64(uint64_t v, unsigned r)
  {
    // return std::rotr(v, r);
    r &= 63u;
    return (v >> r) | (v << ((64u - r) & 63u));
  }
} // namespace pico_gpu

#pragma once

#include "types.hpp"
#include "byte.hpp"

namespace pico_gpu::sha256_operations {
  using namespace byte;

  /// Fixed right rotation operation for SHA256.
  template <typename T>
  struct FixedRotateRightOperation {
    Word<T> value;
    Word<T> shift;
    Word<T> carry;

    __device__ FixedRotateRightOperation() {}

    __device__ static constexpr size_t nb_bytes_to_shift(size_t rotation) { return rotation / 8; }

    __device__ static constexpr size_t nb_bits_to_shift(size_t rotation) { return rotation % 8; }

    __device__ static constexpr uint32_t carry_multiplier(size_t rotation)
    {
      const size_t nb_bits_to_shift = FixedRotateRightOperation::nb_bits_to_shift(rotation);
      return 1 << (8 - nb_bits_to_shift);
    }

    __device__ static void shr_carry(uint8_t input, uint8_t shift_amount, uint8_t* shift_out, uint8_t* carry_out)
    {
      *shift_out = input >> shift_amount;
      *carry_out = input & ((1 << shift_amount) - 1);
    }

    __device__ uint32_t populate(uint32_t input, const size_t rotation, T* byte_trace)
    {
      const uint32_t expected = (input >> rotation) | (input << (32 - rotation));

      // Extract input bytes and convert to field elements.
      uint8_t input_bytes_raw[4] = {
        (uint8_t)(input & 0xFF), (uint8_t)((input >> 8) & 0xFF), (uint8_t)((input >> 16) & 0xFF),
        (uint8_t)((input >> 24) & 0xFF)};

      T input_bytes[4] = {
        T::from_canonical_u8(input_bytes_raw[0]), T::from_canonical_u8(input_bytes_raw[1]),
        T::from_canonical_u8(input_bytes_raw[2]), T::from_canonical_u8(input_bytes_raw[3])};

      // Compute rotation constants.
      const size_t nb_bytes_to_shift = FixedRotateRightOperation::nb_bytes_to_shift(rotation);
      const size_t nb_bits_to_shift = FixedRotateRightOperation::nb_bits_to_shift(rotation);
      T carry_mult = T::from_canonical_u32(FixedRotateRightOperation::carry_multiplier(rotation));

      // Perform byte rotation.
      T input_bytes_rotated[4] = {
        input_bytes[nb_bytes_to_shift % WORD_SIZE], input_bytes[(1 + nb_bytes_to_shift) % WORD_SIZE],
        input_bytes[(2 + nb_bytes_to_shift) % WORD_SIZE], input_bytes[(3 + nb_bytes_to_shift) % WORD_SIZE]};

      T first_shift = T::zero();
      T last_carry = T::zero();

      // Process in reverse order.
      for (int i = WORD_SIZE - 1; i >= 0; i--) {
        const uint8_t b = (uint8_t)(input_bytes_rotated[i].as_canonical_u32() & 0xFF);
        const uint8_t c = (uint8_t)nb_bits_to_shift;

        uint8_t shift_val, carry_val;
        shr_carry(b, c, &shift_val, &carry_val);

        handle_byte_lookup_event(byte_trace, ByteOpcode::ShrCarry, b, c);

        shift._0[i] = T::from_canonical_u8(shift_val);
        carry._0[i] = T::from_canonical_u8(carry_val);

        if (i == WORD_SIZE - 1) {
          first_shift = shift._0[i];
        } else {
          value._0[i] = shift._0[i] + last_carry * carry_mult;
        }

        last_carry = carry._0[i];
      }

      value._0[WORD_SIZE - 1] = first_shift + last_carry * carry_mult;

      return expected;
    }
  };

  /// XOR operation for SHA256.
  template <typename T>
  struct XorOperation {
    Word<T> value;

    __device__ XorOperation() {}

    __device__ uint32_t populate(uint32_t x, uint32_t y, T* byte_trace)
    {
      const uint32_t expected = x ^ y;

      // Extract bytes in little endian order.
      uint8_t x_bytes[4] = {
        (uint8_t)(x & 0xFF), (uint8_t)((x >> 8) & 0xFF), (uint8_t)((x >> 16) & 0xFF), (uint8_t)((x >> 24) & 0xFF)};
      uint8_t y_bytes[4] = {
        (uint8_t)(y & 0xFF), (uint8_t)((y >> 8) & 0xFF), (uint8_t)((y >> 16) & 0xFF), (uint8_t)((y >> 24) & 0xFF)};

      // Populate value field byte by byte.
      for (int i = 0; i < WORD_SIZE; i++) {
        const uint8_t xor_result = x_bytes[i] ^ y_bytes[i];
        value._0[i] = T::from_canonical_u8(xor_result);

        handle_byte_lookup_event(byte_trace, ByteOpcode::XOR, x_bytes[i], y_bytes[i]);
      }

      return expected;
    }
  };
} // namespace pico_gpu::sha256_operations
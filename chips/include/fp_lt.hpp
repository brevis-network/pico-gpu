#pragma once

#include "fp_op.hpp"
#include "byte.hpp"
#include "types.hpp"

namespace pico_gpu::fp_lt {

  /// Populate field element less-than comparison columns.
  /// Compares two field elements byte-by-byte from most significant to least.
  template <class F, const int NumLimbs, const int NumWitnesses>
  __PICO_HOSTDEV__ inline void populate(
    FieldLtCols<F, NumLimbs, NumWitnesses>& cols,
    const uint32_t lhs[NumLimbs / 4],
    const uint32_t rhs[NumLimbs / 4],
    F* byte_trace)
  {
    // Convert words to byte arrays.
    uint8_t value_limbs[NumLimbs];
    fp_op::words_to_bytes_le<NumLimbs / 4, NumLimbs>(lhs, value_limbs);

    uint8_t modulus[NumLimbs];
    fp_op::words_to_bytes_le<NumLimbs / 4, NumLimbs>(rhs, modulus);

    // Initialize byte flags for comparison tracking.
    uint8_t byte_flags[NumLimbs];
    for (int i = 0; i < NumLimbs; ++i)
      byte_flags[i] = 0;

    // Compare bytes from most significant to least significant.
    // Stop at first byte where lhs < rhs.
    for (int i = NumLimbs - 1; i >= 0; --i) {
      const uint8_t byte = value_limbs[i];
      const uint8_t modulus_byte = modulus[i];

      assert(byte <= modulus_byte);

      if (byte < modulus_byte) {
        byte_flags[i] = 1;
        cols.lhs_comparison_byte = F::from_canonical_u8(byte);
        cols.rhs_comparison_byte = F::from_canonical_u8(modulus_byte);

        if (byte_trace) { byte::handle_byte_lookup_event(byte_trace, ByteOpcode::LTU, byte, modulus_byte); }
        break;
      }
    }

    // Store byte flags indicating comparison position.
    for (int i = 0; i < NumLimbs; ++i) {
      cols.byte_flags[i] = F::from_canonical_u8(byte_flags[i]);
    }
  }

} // namespace pico_gpu::fp_lt
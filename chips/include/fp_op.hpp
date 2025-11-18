#pragma once

#include "types.hpp"
#include "memory_read_write.hpp"
#include "byte.hpp"
#include <sys/types.h>

namespace pico_gpu::fp_op {
  using namespace ::pico_gpu::memory_read_write;

  static constexpr int BitsPerLimb = 8;

  // Macro to split a 64-bit literal into two 32-bit words for array initialization (little-endian).
  // Derived from ff/alt_bn128.hpp
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

  // Convert little-endian 32-bit words to a byte array.
  template <const int NumWords, const int NumLimbs>
  __PICO_HOSTDEV__ inline void words_to_bytes_le(const uint32_t words[NumWords], uint8_t bytes[NumLimbs])
  {
    // NumLimbs must be NumWords * 4
    for (int i = 0; i < NumWords; ++i) {
      uint32_t current_word = words[i];
      bytes[4 * i + 0] = static_cast<uint8_t>(current_word);
      bytes[4 * i + 1] = static_cast<uint8_t>(current_word >> 8);
      bytes[4 * i + 2] = static_cast<uint8_t>(current_word >> 16);
      bytes[4 * i + 3] = static_cast<uint8_t>(current_word >> 24);
    }
  }

  // Convert little-endian 32-bit words to an array of T-typed bytes.
  template <class F, const int NumWords, const int NumLimbs>
  __PICO_HOSTDEV__ inline void
  words_to_t_bytes_le(const uint32_t words[NumWords], F t_bytes_out[NumLimbs], F* byte_trace)
  {
    // NumLimbs must be NumWords * 4
    for (int i = 0; i < NumWords; ++i) {
      uint32_t w = words[i];
      t_bytes_out[4 * i + 0] = F(static_cast<int>(w & 0xFF));
      t_bytes_out[4 * i + 1] = F(static_cast<int>((w >> 8) & 0xFF));
      t_bytes_out[4 * i + 2] = F(static_cast<int>((w >> 16) & 0xFF));
      t_bytes_out[4 * i + 3] = F(static_cast<int>((w >> 24) & 0xFF));
    }

    if (byte_trace) {
      for (int i = 0; i < NumWords; i++) {
        uint32_t w = words[i];
        byte::add_u8_range_check(byte_trace, w & 0xff, (w >> 8) & 0xff);
        byte::add_u8_range_check(byte_trace, (w >> 16) & 0xff, (w >> 24) & 0xff);
      }
    }
  }

  // Copy NumWords elements from src to dest.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void copy_words(uint32_t* __restrict__ dest, const uint32_t* __restrict__ src)
  {
    for (int i = 0; i < NumWords; ++i) {
      dest[i] = src[i];
    }
  }

  // Set NumWords elements of an array to zero.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void set_words_to_zero(uint32_t* arr)
  {
    for (int i = 0; i < NumWords; ++i) {
      arr[i] = 0U;
    }
  }

  // Check if all NumWords elements of an array are zero.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline bool are_words_zero(const uint32_t* arr)
  {
    for (int i = 0; i < NumWords; ++i) {
      if (arr[i] != 0U) return false;
    }
    return true;
  }

  // Check if multi-precision integer 'a' is greater than or equal to 'b'.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline bool are_words_greater_equal(const uint32_t* a, const uint32_t* b)
  {
    for (int i = NumWords - 1; i >= 0; --i) { // Start from MSB
      if (a[i] > b[i]) return true;
      if (a[i] < b[i]) return false;
    }
    return true; // Equal
  }

  // Add a 32-bit scalar to a multi-precision integer 'arr' in place.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void add_scalar_to_words(uint32_t* arr, uint32_t val, uint32_t* out)
  {
    uint64_t carry = val;
    for (int i = 0; i < NumWords; ++i) {
      uint64_t sum = static_cast<uint64_t>(arr[i]) + carry;
      out[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }

  // Add a 32-bit scalar to a multi-precision integer 'arr' in place.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void add_scalar_to_words_inplace(uint32_t* arr, uint32_t val)
  {
    uint64_t carry = val;
    for (int i = 0; i < NumWords; ++i) {
      if (carry == 0) break;
      uint64_t sum = static_cast<uint64_t>(arr[i]) + carry;
      arr[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
  }

  // Compute (x + y) mod modulus_words. result_out = x + y - k*modulus.
  // carry_out[0] is 1 if x+y >= modulus_words, 0 otherwise. Other carry_out elements are 0.
  template <const int NumWords, const int NumLimbs>
  __PICO_HOSTDEV__ inline void add_words_mod_p(
    const uint32_t x_words[NumWords],
    const uint32_t y_words[NumWords],
    uint32_t result_out[NumWords],
    uint8_t carry_out[NumLimbs],            // carry_out[0] stores the effective carry for the modular addition.
    const uint32_t modulus_words[NumWords]) // we assume if modulus is zero, modulus = 1 << NumWords * 32)
  {
    // Step 1: Full addition x + y
    uint64_t acc = 0;
    for (int i = 0; i < NumWords; ++i) {
      acc = static_cast<uint64_t>(x_words[i]) + y_words[i] + (acc >> 32);
      result_out[i] = static_cast<uint32_t>(acc);
    }
    uint32_t addition_carry_bit = static_cast<uint32_t>(acc >> 32); // Carry from the most significant word addition

    // Step 2: Check if (x + y) >= modulus_words
    // This happens if addition_carry_bit is 1, or if addition_carry_bit is 0 AND sum >= modulus_words.
    bool overflow = addition_carry_bit || are_words_greater_equal<NumWords>(result_out, modulus_words);

    // Step 3: Conditional subtraction of modulus: result_out = result_out - (overflow ? modulus_words : 0)
    int64_t borrow = 0;
    for (int i = 0; i < NumWords; ++i) {
      int64_t diff_val =
        static_cast<int64_t>(result_out[i]) - (static_cast<int64_t>(modulus_words[i]) * overflow) - borrow;
      result_out[i] = static_cast<uint32_t>(diff_val);
      borrow = (diff_val < 0) ? 1 : 0; // Simpler borrow extraction for subtraction logic
    }

    // Write carry flag (overflow means a reduction by modulus occurred)
    carry_out[0] = static_cast<uint8_t>(overflow);
    for (int i = 1; i < NumLimbs; ++i) { // Zero-fill remaining carry bytes
      carry_out[i] = 0;
    }
  }

  // Adds a and b, stores in res, returns the carry-out. res = a + b.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline uint32_t
  add_words_with_carry(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ res)
  {
    uint64_t carry = 0ULL;
    for (int i = 0; i < NumWords; ++i) {
      uint64_t sum = static_cast<uint64_t>(a[i]) + b[i] + carry;
      res[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    return static_cast<uint32_t>(carry);
  }

  // Adds a and b, stores in res, returns the carry-out. res = a + b.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline uint32_t add_words_with_carry_norestrict(const uint32_t* a, const uint32_t* b, uint32_t* res)
  {
    uint64_t carry = 0ULL;
    for (int i = 0; i < NumWords; ++i) {
      uint64_t sum = static_cast<uint64_t>(a[i]) + b[i] + carry;
      res[i] = static_cast<uint32_t>(sum);
      carry = sum >> 32;
    }
    return static_cast<uint32_t>(carry);
  }

  // Subtracts b from a, stores in res, returns the borrow. res = a - b.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline uint32_t
  sub_words_with_borrow(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, uint32_t* __restrict__ res)
  {
    int64_t borrow = 0LL; // Use signed type for borrow logic
    for (int i = 0; i < NumWords; ++i) {
      int64_t diff = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]) - borrow;
      if (diff < 0) {
        res[i] = static_cast<uint32_t>(diff + (1LL << 32)); // Add 2^32 to make it positive before casting
        borrow = 1;
      } else {
        res[i] = static_cast<uint32_t>(diff);
        borrow = 0;
      }
    }
    return static_cast<uint32_t>(borrow);
  }

  // Subtracts b from a, stores in res, returns the borrow. res = a - b.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline uint32_t sub_words_with_borrow_norestrict(const uint32_t* a, const uint32_t* b, uint32_t* res)
  {
    int64_t borrow = 0LL; // Use signed type for borrow logic
    for (int i = 0; i < NumWords; ++i) {
      int64_t diff = static_cast<int64_t>(a[i]) - static_cast<int64_t>(b[i]) - borrow;
      if (diff < 0) {
        res[i] = static_cast<uint32_t>(diff + (1LL << 32)); // Add 2^32 to make it positive before casting
        borrow = 1;
      } else {
        res[i] = static_cast<uint32_t>(diff);
        borrow = 0;
      }
    }
    return static_cast<uint32_t>(borrow);
  }

  // Calculates mid = floor((low + high) / 2) for multi-precision integers.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void calculate_floored_average_words(
    uint32_t* __restrict__ mid, const uint32_t* __restrict__ low, const uint32_t* __restrict__ high)
  {
    uint32_t temp_sum[NumWords];
    uint32_t carry_from_add = add_words_with_carry<NumWords>(low, high, temp_sum); // temp_sum = low + high

    // Perform right shift by 1 on (carry_from_add << (NumWords*32) | temp_sum)
    uint32_t prev_word_msb_as_carry = carry_from_add & 1U; // MSB of sum is LSB of carry_from_add
    for (int i = NumWords - 1; i >= 0; --i) {              // Iterate from MSW to LSW
      uint32_t current_word = temp_sum[i];
      mid[i] = (current_word >> 1) | (prev_word_msb_as_carry << 31);
      prev_word_msb_as_carry = current_word & 1U; // LSB of current word becomes carry for next (lower) word
    }
  }

  // Compute vanishing polynomial coefficients for an addition:
  // c_i = P(a_limbs_i) + P(b_limbs_i) - P(r_limbs_i) - carry_var * P(m_limbs_i)
  // Inputs are byte limbs.
  template <const int NumLimbs>
  __PICO_HOSTDEV__ inline void compute_vanishing_coeffs_add(
    const uint8_t a_limbs[NumLimbs],
    const uint8_t b_limbs[NumLimbs],
    const uint8_t r_limbs[NumLimbs], // result of (a + b) mod m
    const uint8_t m_limbs[NumLimbs], // modulus
    uint8_t carry_var,               // Expected to be 0 or 1, representing the carry of (a+b) over m
    int64_t vanishing_coeffs_out[NumLimbs + 1],
    const bool modulus_is_zero)
  {
    for (int i = 0; i < NumLimbs; ++i) {
      vanishing_coeffs_out[i] = static_cast<int64_t>(a_limbs[i]) + static_cast<int64_t>(b_limbs[i]) -
                                static_cast<int64_t>(r_limbs[i]) -
                                static_cast<int64_t>(carry_var) * static_cast<int64_t>(m_limbs[i]);
    }
    if (modulus_is_zero)
      vanishing_coeffs_out[NumLimbs] = -static_cast<int64_t>(carry_var);
    else
      vanishing_coeffs_out[NumLimbs] = 0;
  }

  // Multiply two polynomials p1(x) and p2(x) with NumLimbs coefficients each.
  // prod_coeffs is the resulting polynomial of degree 2*NumLimbs - 2.
  template <const int NumLimbs>
  __PICO_HOSTDEV__ inline void
  poly_mul_coeffs(const uint8_t p1[NumLimbs], const uint8_t p2[NumLimbs], int64_t prod_coeffs[2 * NumLimbs - 1])
  {
    for (int k = 0; k < 2 * NumLimbs - 1; ++k) {
      prod_coeffs[k] = 0;
    }
    for (int i = 0; i < NumLimbs; ++i) {
      if (p1[i] == 0) continue; // Optimization for sparse coefficient
      for (int j = 0; j < NumLimbs; ++j) {
        // if (p2[j] == 0) continue; // This inner check might be less beneficial due to unrolling strategy
        prod_coeffs[i + j] += static_cast<int64_t>(p1[i]) * static_cast<int64_t>(p2[j]);
      }
    }
  }

  // Compute vanishing polynomial coefficients for a multiplication:
  // c_k = (P(a) * P(b))_k - P(r)_k - (P(qc) * P(m))_k
  template <const int NumLimbs>
  __PICO_HOSTDEV__ inline void compute_vanishing_coeffs_mul(
    const uint8_t a_limbs[NumLimbs],
    const uint8_t b_limbs[NumLimbs],
    const uint8_t r_limbs[NumLimbs],  // result (remainder) of (a * b) mod m
    const uint8_t qc_limbs[NumLimbs], // quotient of (a * b) / m
    const uint8_t m_limbs[NumLimbs],  // modulus
    int64_t vanishing_coeffs_out[2 * NumLimbs],
    const bool modulus_overflow)
  { // Output size is 2*NumLimbs-1 for poly product

    int64_t p_ab_coeffs[2 * NumLimbs - 1]; // P(a) * P(b)
    poly_mul_coeffs<NumLimbs>(a_limbs, b_limbs, p_ab_coeffs);

    int64_t p_qc_m_coeffs[2 * NumLimbs]; // P(qc) * P(m)
    // modulus = 1 << 8 * NumLimbs
    if (modulus_overflow) {
      for (int i = 0; i < NumLimbs; ++i)
        p_qc_m_coeffs[i] = 0;
      for (int i = NumLimbs; i < 2 * NumLimbs; ++i)
        p_qc_m_coeffs[i] = qc_limbs[i - NumLimbs];
    } else {
      p_qc_m_coeffs[2 * NumLimbs - 1] = 0;
      poly_mul_coeffs<NumLimbs>(qc_limbs, m_limbs, p_qc_m_coeffs);
    }

    vanishing_coeffs_out[2 * NumLimbs - 1] = -p_qc_m_coeffs[2 * NumLimbs - 1];
    for (int k = 0; k < 2 * NumLimbs - 1; ++k) {
      // r_limbs has NumLimbs elements. If k >= NumLimbs, its contribution is 0.
      int64_t r_k_val = (k < NumLimbs) ? static_cast<int64_t>(r_limbs[k]) : 0;
      vanishing_coeffs_out[k] = p_ab_coeffs[k] - r_k_val - p_qc_m_coeffs[k];
    }
  }

  // Synthetic division by (x - base) for addition's vanishing polynomial (NumLimbs coefficients).
  // coeffs_in: p(x) = c_{N-1}x^{N-1} + ... + c_0
  // quotient_coeffs_out: q(x) where p(x) = q(x)(x-base) + p(base)
  // The remainder p(base) should be 0.
  template <const int NumLimbs, const int BitsPerLimb>
  __PICO_HOSTDEV__ inline void
  synthetic_division_add(const int64_t coeffs_in[NumLimbs + 1], int64_t quotient_coeffs_out[NumLimbs])
  {
    constexpr uint64_t horner_base = 1ULL << BitsPerLimb; // Base of the polynomial representation (e.g., 2^8)

    // Horner's method for division:
    // q_{N-1} = 0 (for this specific formulation where quotient has one less degree or padded)
    // q_{i-1} = c_i + base * q_i
    // Here, quotient_coeffs_out[NumLimbs-1] is highest coeff of quotient, which is 0.
    // And coeffs_in[NumLimbs-1] is highest coeff of dividend.
    // Division: q_k = c_{k+1} + base * q_{k+1} (working downwards)
    // q_{N-1} = c_N (if c_N existed, but P is degree N-1)
    // Let q(x) = q_{N-1}x^{N-1} + ... + q_0 (output quotient)
    // Let p(x) = c_{N-1}x^{N-1} + ... + c_0 (input coefficients)
    // q_{N-1} = 0 (coefficient of x^{N-1} in quotient is 0, as q has degree N-2 if p(base)=0)
    // This seems to compute quotient for P(X)/(X-B) where result has same "NumLimbs" size array.
    // The highest coefficient of the quotient q_{N-1} is c_{N-1} if degree of P is N-1 and Q is N-2.
    // This implementation has q_{N-1} = 0, q_{N-2} = c_{N-1}, q_{N-3} = c_{N-2} + base * q_{N-2}, ...

    quotient_coeffs_out[NumLimbs - 1] = coeffs_in[NumLimbs]; // Highest coefficient of q(x) (q_{N-1})
    for (int i = NumLimbs - 2; i >= 0; --i) {                // Compute q_{N-2} down to q_0
      // quotient_coeffs_out[i] is q_i
      // coeffs_in[i+1] is c_{i+1}
      // quotient_coeffs_out[i+1] is q_{i+1}
      quotient_coeffs_out[i] = coeffs_in[i + 1] + static_cast<int64_t>(horner_base) * quotient_coeffs_out[i + 1];
    }
    // Remainder is coeffs_in[0] + horner_base * quotient_coeffs_out[0], should be 0.
  }

  // Synthetic division by (x - base) for multiplication's vanishing polynomial.
  // VanishingCoeffSize = 2*NumLimbs. QuotientCoeffSize = NumWitnesses.
  template <
    const int NumLimbs, // Used for context, not directly in loop bounds if VanishingCoeffSize is primary
    const int BitsPerLimb,
    const int QuotientCoeffSize,
    const int VanishingCoeffSize = 2 * NumLimbs>
  __PICO_HOSTDEV__ inline void
  synthetic_division_mul(const int64_t coeffs_in[VanishingCoeffSize], int64_t quotient_coeffs_out[QuotientCoeffSize])
  {
    constexpr uint64_t horner_base = 1ULL << BitsPerLimb;

    // q_{M-1} = c_M (where M = VanishingCoeffSize-1 = QuotientCoeffSize)
    // q_j = c_{j+1} + base * q_{j+1} for j from M-1 down to 0.
    // quotient_coeffs_out indices are [0, ..., QuotientCoeffSize-1]
    // coeffs_in indices are [0, ..., VanishingCoeffSize-1]
    // Highest coeff of quotient q_{QCS-1} (degree QCS-1) is c_{VCS-1} (coeff of x^{VCS-1} in input)
    quotient_coeffs_out[QuotientCoeffSize - 1] = coeffs_in[QuotientCoeffSize];
    for (int j = QuotientCoeffSize - 2; j >= 0; --j) {
      quotient_coeffs_out[j] = coeffs_in[j + 1] + static_cast<int64_t>(horner_base) * quotient_coeffs_out[j + 1];
    }
    // Remainder is coeffs_in[0] + horner_base * quotient_coeffs_out[0], should be 0.
  }

  // Fill witness limbs from quotient coefficients (typically for ADD/SUB).
  // quotient_coeffs has NumLimbs elements. witness arrays have NumWitnesses elements.
  // If NumWitnesses > NumLimbs, remaining witness elements are derived from a zero quotient coefficient.
  template <class F, const int NumLimbs, const int NumWitnesses, const int WitnessOffset>
  __PICO_HOSTDEV__ inline void fill_witness_limbs(
    const int64_t quotient_coeffs[NumLimbs], F witness_low[NumWitnesses], F witness_high[NumWitnesses], F* byte_trace)
  {
    for (int i = 0; i < NumWitnesses; ++i) {
      // If index 'i' is beyond the actual quotient coefficients, assume coefficient is 0.
      int64_t q_coeff_i = (i < NumLimbs) ? quotient_coeffs[i] : 0;
      uint64_t witness_val = static_cast<uint64_t>(q_coeff_i + WitnessOffset); // Apply offset
      // Split 16-bit (or more accurately, the lower part of witness_val) into two 8-bit limbs
      witness_low[i] = F(static_cast<int>(witness_val & 0xFF));         // Lower 8 bits
      witness_high[i] = F(static_cast<int>((witness_val >> 8) & 0xFF)); // Next 8 bits
    }

    if (byte_trace) {
      for (int i = 0; i < NumWitnesses; i += 2) {
        // If index 'i' is beyond the actual quotient coefficients, assume coefficient is 0.
        int64_t q_coeff_i = (i < NumLimbs) ? quotient_coeffs[i] : 0;
        int64_t q_coeff_ii = (i + 1 < NumLimbs && i + 1 < NumWitnesses) ? quotient_coeffs[i + 1]
                             : i + 1 < NumWitnesses                     ? 0
                                                                        : -WitnessOffset;
        uint64_t witness_val = static_cast<uint64_t>(q_coeff_i + WitnessOffset);   // Apply offset
        uint64_t witness_next = static_cast<uint64_t>(q_coeff_ii + WitnessOffset); // Apply offset
        uint8_t low_val = witness_val & 0xff;
        uint8_t low_next = witness_next & 0xff;
        uint8_t high_val = (witness_val >> 8) & 0xff;
        uint8_t high_next = (witness_next >> 8) & 0xff;
        byte::add_u8_range_check(byte_trace, low_val, low_next);
        byte::add_u8_range_check(byte_trace, high_val, high_next);
      }
    }
  }

  // Fill witness limbs from quotient coefficients (typically for MUL/DIV).
  // quotient_coeffs has NumWitnesses elements.
  template <class F, const int NumWitnesses, const int WitnessOffset>
  __PICO_HOSTDEV__ inline void fill_witness_limbs_mul(
    const int64_t quotient_coeffs[NumWitnesses],
    F witness_low[NumWitnesses],
    F witness_high[NumWitnesses],
    F* byte_trace)
  {
    for (int i = 0; i < NumWitnesses; ++i) {
      int64_t q_coeff_i = quotient_coeffs[i]; // Direct use, assumes quotient_coeffs has NumWitnesses elements
      uint64_t witness_val = static_cast<uint64_t>(q_coeff_i + WitnessOffset);
      witness_low[i] = F(static_cast<int>(witness_val & 0xFF));
      witness_high[i] = F(static_cast<int>((witness_val >> 8) & 0xFF));
    }

    if (byte_trace) {
      for (int i = 0; i < NumWitnesses; i += 2) {
        // If index 'i' is beyond the actual quotient coefficients, assume coefficient is 0.
        int64_t q_coeff_i = quotient_coeffs[i];
        int64_t q_coeff_ii = (i + 1 < NumWitnesses) ? quotient_coeffs[i + 1] : -WitnessOffset;
        uint64_t witness_val = static_cast<uint64_t>(q_coeff_i + WitnessOffset);   // Apply offset
        uint64_t witness_next = static_cast<uint64_t>(q_coeff_ii + WitnessOffset); // Apply offset
        uint8_t low_val = witness_val & 0xff;
        uint8_t low_next = witness_next & 0xff;
        uint8_t high_val = (witness_val >> 8) & 0xff;
        uint8_t high_next = (witness_next >> 8) & 0xff;
        byte::add_u8_range_check(byte_trace, low_val, low_next);
        byte::add_u8_range_check(byte_trace, high_val, high_next);
      }
    }
  }

  // Populate columns for an ADD operation: (x + y) mod modulus
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_add(
    FieldOpCols<F, NumLimbs, NumWitnesses>& cols, // Output columns
    const uint32_t x_words[NumWords],             // Input X
    const uint32_t y_words[NumWords],             // Input Y
    uint32_t result_words[NumWords],              // Output result
    const uint32_t modulus[NumWords],             // Modulus p
    const bool modulus_is_zero,
    F* byte_trace)
  {
    // 1. Compute R = (X+Y) mod p, and C_eff = (X+Y >= p ? 1:0)
    uint8_t c_eff_bytes[NumLimbs]; // C_eff (only c_eff_bytes[0] is used)
    add_words_mod_p<NumWords, NumLimbs>(x_words, y_words, result_words, c_eff_bytes, modulus);

    // 2. Convert R to T-limbs and store in cols.result
    words_to_t_bytes_le<F, NumWords, NumLimbs>(result_words, cols.result, byte_trace);

    if (byte_trace) { byte::add_u8_range_checks(byte_trace, c_eff_bytes, NumLimbs); }

    // Store effective carry C_eff in T-limbs into cols.carry
    for (int i = 0; i < NumLimbs; ++i) {
      cols.carry[i] = F(static_cast<int>(c_eff_bytes[i]));
    }

    // 3. Convert inputs (X, Y), result (R), and modulus (p) to byte arrays for polynomial construction
    uint8_t x_bytes[NumLimbs], y_bytes[NumLimbs], r_bytes[NumLimbs], m_bytes[NumLimbs];
    words_to_bytes_le<NumWords, NumLimbs>(x_words, x_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(y_words, y_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(result_words, r_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(modulus, m_bytes);

    // 4. Compute vanishing polynomial coefficients: v_i = x_i + y_i - r_i - C_eff * m_i
    int64_t v_coeffs[NumLimbs + 1];
    compute_vanishing_coeffs_add<NumLimbs>(
      x_bytes, y_bytes, r_bytes, m_bytes, c_eff_bytes[0], v_coeffs, modulus_is_zero);

    // 5. Compute quotient polynomial: Q(X) = V(X) / (X - base) via synthetic division
    int64_t q_coeffs[NumLimbs]; // Quotient polynomial coefficients
    synthetic_division_add<NumLimbs, BitsPerLimb>(v_coeffs, q_coeffs);

    // 6. Generate witness values from Q(X)
    fill_witness_limbs<F, NumLimbs, NumWitnesses, WitnessOffset>(
      q_coeffs, cols.witness_low, cols.witness_high, byte_trace);
  }

  // Full NumWords-word by NumWords-word multiplication, resulting in a 2*NumWords-word product.
  // a, b: NumWords elements each. prod: 2*NumWords elements.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void full_multiply_words_to_double_words(
    const uint32_t a[NumWords], const uint32_t b[NumWords], uint32_t prod[2 * NumWords])
  {
    for (int k = 0; k < 2 * NumWords; ++k) {
      prod[k] = 0;
    }

    for (int i = 0; i < NumWords; ++i) { // Iterate through words of 'b'
      if (b[i] == 0) continue;           // Optimization: if word of b is 0, it adds nothing to product
      uint64_t carry_inner = 0;
      for (int j = 0; j < NumWords; ++j) { // Iterate through words of 'a'
        uint64_t val = static_cast<uint64_t>(a[j]) * b[i] + prod[i + j] + carry_inner;
        prod[i + j] = static_cast<uint32_t>(val);
        carry_inner = val >> 32;
      }
      int k = i + NumWords;
      while (carry_inner != 0 && k < 2 * NumWords) {
        uint64_t sum_with_carry = static_cast<uint64_t>(prod[k]) + carry_inner;
        prod[k] = static_cast<uint32_t>(sum_with_carry);
        carry_inner = sum_with_carry >> 32;
        k++;
      }
    }
  }

  // Division of a 2*NumWords-word number (dividend) by a NumWords-word number (divisor) using binary search.
  // Produces a NumWords-word quotient and a NumWords-word remainder.
  // This is computationally intensive.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void divide_double_words_by_words_binary_search(
    const uint32_t dividend_double_words[2 * NumWords], // 2*NumWords words dividend
    const uint32_t divisor_words[NumWords],             // NumWords words divisor
    uint32_t quotient_words_out[NumWords],              // NumWords words quotient
    uint32_t remainder_words_out[NumWords],
    const bool divisor_words_overflow) // NumWords words remainder
  {
    uint32_t divisor_padded_double_words[2 * NumWords];
    set_words_to_zero<2 * NumWords>(divisor_padded_double_words);
    copy_words<NumWords>(divisor_padded_double_words, divisor_words);
    if (divisor_words_overflow) { // when modulus = 1 << 8 * NumLimbs
      divisor_padded_double_words[NumWords] = 1;
      copy_words<NumWords>(quotient_words_out, dividend_double_words + NumWords);
      copy_words<NumWords>(remainder_words_out, dividend_double_words); // Remainder is low NumWords of dividend
      return;
    }

    if (!are_words_greater_equal<2 * NumWords>(dividend_double_words, divisor_padded_double_words)) {
      set_words_to_zero<NumWords>(quotient_words_out);
      copy_words<NumWords>(remainder_words_out, dividend_double_words); // Remainder is low NumWords of dividend
      return;
    }

    if (are_words_zero<NumWords>(divisor_words)) {
      for (int i = 0; i < NumWords; ++i)
        quotient_words_out[i] = 0xFFFFFFFFU; // Max quotient
      // Remainder is the lower NumWords part of the dividend when divisor is zero.
      // If dividend fits in NumWords, this is fine. If dividend > 2^(NumWords*32),
      // then original dividend_double_words[0...NumWords-1] is copied.
      copy_words<NumWords>(remainder_words_out, dividend_double_words);
      return;
    }

    uint32_t low_q[NumWords], high_q[NumWords], mid_q[NumWords];
    uint32_t temp_prod_q_times_divisor_double_words[2 * NumWords];
    uint32_t one_words[NumWords] = {0};
    if (NumWords > 0) one_words[0] = 1U;

    set_words_to_zero<NumWords>(low_q);

    // Set high_q to the maximum possible quotient value (all bits set to 1).
    // The quotient is known to fit within NumWords.
    for (int i = 0; i < NumWords; ++i)
      high_q[i] = 0xFFFFFFFFU;

    set_words_to_zero<NumWords>(quotient_words_out);

    uint32_t temp_diff_high_low[NumWords];
    // Loop while low_q <= high_q
    while (sub_words_with_borrow<NumWords>(high_q, low_q, temp_diff_high_low) == 0) {
      calculate_floored_average_words<NumWords>(mid_q, low_q, high_q);

      full_multiply_words_to_double_words<NumWords>(mid_q, divisor_words, temp_prod_q_times_divisor_double_words);

      if (are_words_greater_equal<2 * NumWords>(dividend_double_words, temp_prod_q_times_divisor_double_words)) {
        copy_words<NumWords>(quotient_words_out, mid_q);
        copy_words<NumWords>(low_q, mid_q);
        add_scalar_to_words_inplace<NumWords>(low_q, 1);
      } else {
        // mid_q is too high. high_q = mid_q - 1.
        // Need to handle mid_q = 0 carefully to prevent underflow if one_words is subtracted.
        if (are_words_zero<NumWords>(mid_q)) {
          // If mid_q is already 0 and it's too high, it means dividend < 0, which is not
          // possible for unsigned, or dividend is 0 and divisor is >0.
          // In this case, the loop should terminate with quotient_words_out being 0.
          // Setting high_q < low_q will terminate the loop. e.g. copy_words<NumWords>(high_q, low_q);
          // sub_scalar_from_words_inplace<NumWords>(high_q,1); A simpler break is fine if quotient is already 0.
          if (are_words_zero<NumWords>(quotient_words_out)) break; // defensive break
        }
        sub_words_with_borrow<NumWords>(mid_q, one_words, high_q);
      }
    }

    // Recalculate remainder: remainder = dividend - quotient * divisor
    full_multiply_words_to_double_words<NumWords>(
      quotient_words_out, divisor_words, temp_prod_q_times_divisor_double_words);
    uint32_t remainder_temp_double_words[2 * NumWords];
    sub_words_with_borrow<2 * NumWords>(
      dividend_double_words, temp_prod_q_times_divisor_double_words, remainder_temp_double_words);
    copy_words<NumWords>(
      remainder_words_out, remainder_temp_double_words); // Remainder must be < divisor, so fits in NumWords
  }

  // Populate columns for a MUL operation: (x * y) mod modulus
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_mul(
    FieldOpCols<F, NumLimbs, NumWitnesses>& cols,
    const uint32_t x_words[NumWords],
    const uint32_t y_words[NumWords],
    uint32_t remainder_words[NumWords], // R_words (result of modular multiplication)
    const uint32_t modulus[NumWords],
    const bool modulus_is_zero,
    F* byte_trace)
  {
    uint32_t product_double_words[2 * NumWords];
    // 1. Full product: P_double_words = X * Y
    full_multiply_words_to_double_words<NumWords>(x_words, y_words, product_double_words);

    // 2. Division: P_double_words = Q_words * modulus + R_words
    uint32_t quotient_words[NumWords]; // Q_words (effective carry for mul operation)
    // This division is a major performance factor.
    divide_double_words_by_words_binary_search<NumWords>(
      product_double_words, modulus, quotient_words, remainder_words, modulus_is_zero);

    // 3. Store R_words (result) and Q_words (carry/quotient) into columns
    words_to_t_bytes_le<F, NumWords, NumLimbs>(remainder_words, cols.result, byte_trace);
    words_to_t_bytes_le<F, NumWords, NumLimbs>(quotient_words, cols.carry, byte_trace);

    // 4. Prepare byte arrays for vanishing polynomial: X, Y, R_words, Q_words, modulus
    uint8_t x_bytes[NumLimbs], y_bytes[NumLimbs], r_bytes[NumLimbs], qc_bytes[NumLimbs], m_bytes[NumLimbs];
    words_to_bytes_le<NumWords, NumLimbs>(x_words, x_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(y_words, y_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(remainder_words, r_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(quotient_words, qc_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(modulus, m_bytes);

    // 5. Compute vanishing polynomial coefficients: V(X) = P(x_bytes)*P(y_bytes) - P(r_bytes) - P(qc_bytes)*P(m_bytes)
    // Degree of V(X) is 2*NumLimbs - 2. So, 2*NumLimbs - 1 coefficients, but when modulus = 1 << 8 * NumLimbs, it's
    // 2*NumLimbs.
    constexpr int VanishingCoeffsSize = 2 * NumLimbs;
    int64_t v_coeffs[VanishingCoeffsSize];
    compute_vanishing_coeffs_mul<NumLimbs>(x_bytes, y_bytes, r_bytes, qc_bytes, m_bytes, v_coeffs, modulus_is_zero);

    // 6. Compute quotient polynomial Q_prime(X) = V(X) / (X - base)
    // Degree of Q_prime(X) is NumWitnesses, 2*NumLimbs - 2 or 2*NumLimbs - 1 when the modulus is 1 << 8 * NumLimbs.
    constexpr int QuotientOutputCoeffsSize = NumWitnesses;
    int64_t q_prime_coeffs[QuotientOutputCoeffsSize];
    synthetic_division_mul<NumLimbs, BitsPerLimb, NumWitnesses>(v_coeffs, q_prime_coeffs);

    // 7. Generate witness values from Q_prime(X)
    fill_witness_limbs_mul<F, NumWitnesses, WitnessOffset>(
      q_prime_coeffs, cols.witness_low, cols.witness_high, byte_trace);
  }

  // Modular subtraction: result = (a - b + modulus) % modulus. Ensures result is in [0, modulus-1].
  // If modulus == 0, we assume modulus denotes 1 << NumWords * 32.
  template <const int NumWords>
  __PICO_HOSTDEV__ inline void mod_sub_words(
    const uint32_t a[NumWords],
    const uint32_t b[NumWords],
    uint32_t result[NumWords],
    const uint32_t modulus[NumWords],
    const bool modulus_is_zero)
  {
    uint32_t temp_diff[NumWords];
    if (modulus_is_zero || are_words_greater_equal<NumWords>(a, b)) {
      // If a >= b, then a - b is non-negative. borrow will be 0.
      sub_words_with_borrow<NumWords>(a, b, result);
    } else {
      // If a < b, then a - b is negative. Compute (a + modulus) - b.
      // (a + modulus) will not overflow 2*NumWord range if a, modulus are NumWord.
      uint32_t a_plus_modulus[NumWords];
      add_words_with_carry<NumWords>(
        a, modulus, a_plus_modulus); // This intermediate sum can exceed NumWords capacity if carry=1
                                     // but for (a+modulus-b), it's fine as final result is < modulus.
      sub_words_with_borrow<NumWords>(a_plus_modulus, b, result);
    }
  }

  // Populate columns for a SUB operation: (x - y) mod modulus
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_sub(
    FieldOpCols<F, NumLimbs, NumWitnesses>& cols,
    const uint32_t x_words[NumWords],    // Input X (minuend)
    const uint32_t y_words[NumWords],    // Input Y (subtrahend)
    uint32_t result_sub_words[NumWords], // Output result
    const uint32_t modulus[NumWords],    // Modulus p
    const bool modulus_is_zero,
    F* byte_trace)
  {
    // 1. Compute R_sub = (X - Y) mod p = (X - Y + p) mod p
    mod_sub_words<NumWords>(x_words, y_words, result_sub_words, modulus, modulus_is_zero);

    // 2. Store R_sub into cols.result
    words_to_t_bytes_le<F, NumWords, NumLimbs>(result_sub_words, cols.result, byte_trace);

    // 3. Determine effective carry (C_eff) for the equivalent addition: R_sub + Y = X (mod p).
    //    C_eff is 1 if (R_sub + Y) >= p, and 0 otherwise.
    //    This is because X - Y = R_sub (mod p)  =>  X = R_sub + Y (mod p)
    //    So we are checking the "carry" of R_sub + Y over p.
    uint8_t c_eff_bytes[NumLimbs]; // c_eff_bytes[0] will store C_eff
    uint32_t sum_for_carry_check[NumWords];

    uint64_t acc = 0;
    for (int i = 0; i < NumWords; ++i) {
      acc = static_cast<uint64_t>(result_sub_words[i]) + y_words[i] + (acc >> 32);
      sum_for_carry_check[i] = static_cast<uint32_t>(acc);
    }
    uint32_t addition_carry_bit = static_cast<uint32_t>(acc >> 32);
    bool effective_add_overflows =
      addition_carry_bit || are_words_greater_equal<NumWords>(sum_for_carry_check, modulus);

    c_eff_bytes[0] = static_cast<uint8_t>(effective_add_overflows);
    for (int i = 1; i < NumLimbs; ++i)
      c_eff_bytes[i] = 0;

    if (byte_trace) byte::add_u8_range_checks(byte_trace, c_eff_bytes, NumLimbs);

    // Store C_eff in T-limbs into cols.carry
    for (int i = 0; i < NumLimbs; ++i) {
      cols.carry[i] = F(static_cast<int>(c_eff_bytes[i]));
    }

    // 4. Prepare byte arrays for vanishing polynomial.
    //    Constraint: P(R_sub) + P(Y) - P(X) - C_eff * P(modulus) = 0
    uint8_t p_result_sub_bytes[NumLimbs], p_y_bytes[NumLimbs], p_x_bytes[NumLimbs], p_modulus_bytes[NumLimbs];
    words_to_bytes_le<NumWords, NumLimbs>(result_sub_words, p_result_sub_bytes); // 'a' for compute_vanishing_coeffs_add
    words_to_bytes_le<NumWords, NumLimbs>(y_words, p_y_bytes);                   // 'b'
    words_to_bytes_le<NumWords, NumLimbs>(x_words, p_x_bytes);                   // 'r' (result of effective add)
    words_to_bytes_le<NumWords, NumLimbs>(modulus, p_modulus_bytes);             // 'm'

    // 5. Compute vanishing polynomial coefficients for the effective addition
    int64_t v_coeffs[NumLimbs + 1];
    compute_vanishing_coeffs_add<NumLimbs>(
      p_result_sub_bytes, p_y_bytes, p_x_bytes, p_modulus_bytes, c_eff_bytes[0], v_coeffs, modulus_is_zero);
    // 6. Compute quotient polynomial Q(X) = V(X) / (X - base)
    int64_t q_coeffs[NumLimbs];
    synthetic_division_add<NumLimbs, BitsPerLimb>(v_coeffs, q_coeffs);

    // 7. Generate witness values from Q(X)
    fill_witness_limbs<F, NumLimbs, NumWitnesses, WitnessOffset>(
      q_coeffs, cols.witness_low, cols.witness_high, byte_trace);
  }

  // Modular multiplication: result = (a * b) % mod.
  template <const int NumWords, const int NumLimbs>
  __PICO_HOSTDEV__ inline void mod_mul_words(
    const uint32_t a[NumWords], const uint32_t b[NumWords], const uint32_t mod[NumWords], uint32_t result[NumWords])
  {
    uint32_t product_double_words[2 * NumWords];
    full_multiply_words_to_double_words<NumWords>(a, b, product_double_words);

    uint32_t quotient_dummy[NumWords]; // Quotient part of division is not directly needed here
    // Performance-heavy step:
    divide_double_words_by_words_binary_search<NumWords>(
      product_double_words, mod, quotient_dummy, result /* remainder */, false);
  }

  // Modular exponentiation: result_out = (base_in ^ exp_in) % mod.
  // Uses binary exponentiation (right-to-left). This is very computationally expensive.
  template <const int NumWords, const int NumLimbs>
  __PICO_HOSTDEV__ inline void modpow_words(
    const uint32_t base_in[NumWords],
    const uint32_t exp_in[NumWords], // Exponent
    const uint32_t mod[NumWords],    // Modulus
    uint32_t result_out[NumWords])   // Output
  {
    uint32_t current_power[NumWords]; // Stores base^(2^i)
    copy_words<NumWords>(current_power, base_in);

    uint32_t temp_exp[NumWords]; // Modifiable copy of the exponent
    copy_words<NumWords>(temp_exp, exp_in);

    // Initialize result_out to 1
    set_words_to_zero<NumWords>(result_out);
    if (NumWords > 0) {
      result_out[0] = 1U;
    } else {
      return;
    } // Guard for NumWords = 0

    uint32_t temp_mul_res[NumWords]; // Temporary storage for multiplication results

    while (!are_words_zero<NumWords>(temp_exp)) {
      // If LSB of current exponent is 1, multiply result by current_power
      if ((temp_exp[0] & 1U) == 1U) {
        mod_mul_words<NumWords, NumLimbs>(result_out, current_power, mod, temp_mul_res);
        copy_words<NumWords>(result_out, temp_mul_res);
      }

      // Square the current_power: current_power = (current_power * current_power) % mod
      mod_mul_words<NumWords, NumLimbs>(current_power, current_power, mod, temp_mul_res);
      copy_words<NumWords>(current_power, temp_mul_res);

      // Right shift exponent by 1 bit (temp_exp = temp_exp / 2)
      uint32_t prev_word_lsb_carry = 0;
      for (int i = NumWords - 1; i >= 0; --i) { // MSW to LSW
        uint32_t current_word_val = temp_exp[i];
        temp_exp[i] = (current_word_val >> 1) | (prev_word_lsb_carry << 31);
        prev_word_lsb_carry = current_word_val & 1U;
      }
    }
  }

  // Populate columns for a DIV operation: (x / y) mod modulus = (x * y^-1) mod modulus.
  // Uses Fermat's Little Theorem for inverse: y^-1 = y^(modulus-2) mod modulus.
  // This is extremely computationally expensive due to modpow.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_div(
    FieldOpCols<F, NumLimbs, NumWitnesses>& cols, // Output columns
    const uint32_t x_words[NumWords],             // Dividend 'x'
    const uint32_t y_words[NumWords],             // Divisor 'y'
    uint32_t result_div_words[NumWords],
    const uint32_t modulus[NumWords],       // Modulus 'p'
    const uint32_t mod_minus_two[NumWords], // p-2, for modular inverse
    F* byte_trace)
  {
    // Handle division by zero: if y is 0.
    if (are_words_zero<NumWords>(y_words)) {
      // Division by zero. If x is also 0 (0/0), result is 0 (common in padding).
      // If x is non-zero (x/0), this is an error, but code proceeds, result is 0.
      // Upstream logic should typically prevent non-zero/zero.
      set_words_to_zero<NumWords>(result_div_words);
    } else {
      // Normal case: y_words (divisor) is not zero.
      // 1. Calculate y_inv = y ^ (modulus - 2) % modulus (Fermat's Little Theorem)
      uint32_t y_inv[NumWords];
      // This is the most expensive part of DIV:
      modpow_words<NumWords, NumLimbs>(y_words, mod_minus_two, modulus, y_inv);

      // 2. Calculate result_div_words = (x_words * y_inv) % modulus
      mod_mul_words<NumWords, NumLimbs>(x_words, y_inv, modulus, result_div_words);
    }

    // 3. Store result_div_words into cols.result
    words_to_t_bytes_le<F, NumWords, NumLimbs>(result_div_words, cols.result, byte_trace);

    // For witness generation, we model the equivalent multiplication:
    //   result_div_words * y_words = x_words (mod modulus)
    // So, 'a_eff' = result_div_words, 'b_eff' = y_words, 'r_eff' = x_words.
    // We need to find 'qc_eff' = floor((result_div_words * y_words) / modulus).

    // 4. Calculate product_eff = result_div_words * y_words
    uint32_t prod_eff_double_words[2 * NumWords];
    full_multiply_words_to_double_words<NumWords>(result_div_words, y_words, prod_eff_double_words);

    // 5. Decompose product_eff: product_eff = qc_eff * modulus + r_check_eff
    //    r_check_eff should be equal to x_words if calculations are correct.
    uint32_t qc_eff_div_words[NumWords];  // Effective quotient/carry
    uint32_t r_check_eff_words[NumWords]; // Effective remainder, should match x_words
    divide_double_words_by_words_binary_search<NumWords>(
      prod_eff_double_words, modulus, qc_eff_div_words, r_check_eff_words, false);

    // Store qc_eff_div_words into cols.carry
    words_to_t_bytes_le<F, NumWords, NumLimbs>(qc_eff_div_words, cols.carry, byte_trace);

    // 6. Prepare byte arrays for vanishing polynomial based on the effective multiplication.
    //    VP: P(result_div) * P(y) - P(x) - P(qc_eff) * P(modulus) = 0
    uint8_t p_res_div_bytes[NumLimbs], p_y_bytes[NumLimbs], p_x_bytes[NumLimbs];
    uint8_t p_qc_eff_bytes[NumLimbs], p_modulus_bytes[NumLimbs];

    words_to_bytes_le<NumWords, NumLimbs>(result_div_words, p_res_div_bytes); // 'a' in mul_vp
    words_to_bytes_le<NumWords, NumLimbs>(y_words, p_y_bytes);                // 'b' in mul_vp
    words_to_bytes_le<NumWords, NumLimbs>(x_words, p_x_bytes);                // 'r' in mul_vp (original dividend)
    words_to_bytes_le<NumWords, NumLimbs>(qc_eff_div_words, p_qc_eff_bytes);  // 'qc' in mul_vp
    words_to_bytes_le<NumWords, NumLimbs>(modulus, p_modulus_bytes);          // 'm' in mul_vp

    // 7. Compute vanishing polynomial coefficients
    constexpr int VanishingCoeffsSize = 2 * NumLimbs;
    int64_t v_coeffs[VanishingCoeffsSize];
    compute_vanishing_coeffs_mul<NumLimbs>(
      p_res_div_bytes, p_y_bytes, p_x_bytes, p_qc_eff_bytes, p_modulus_bytes, v_coeffs, false);

    // 8. Synthetic division for quotient polynomial
    constexpr int QuotientOutputCoeffsSize = NumWitnesses;
    int64_t q_coeffs[QuotientOutputCoeffsSize];
    synthetic_division_mul<NumLimbs, BitsPerLimb, NumWitnesses>(v_coeffs, q_coeffs);

    // 9. Fill witness limbs
    fill_witness_limbs_mul<F, NumWitnesses, WitnessOffset>(q_coeffs, cols.witness_low, cols.witness_high, byte_trace);
  }

  // Dispatcher function to populate columns based on the field operation.
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_with_modulus(
    FieldOpCols<F, NumLimbs, NumWitnesses>& cols, // Operation-specific output columns
    const uint32_t x_words[NumWords],
    const uint32_t y_words[NumWords],
    uint32_t result_words[NumWords],
    const uint32_t modulus[NumWords],
    const uint32_t mod_mins_two[NumWords], // p-2, only used for DIV
    FieldOperation op,                     // Operation type
    const bool modulus_is_zero,
    F* byte_trace)
  {
    switch (op) {
    case FieldOperation::Add:
      populate_add<F, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
        cols, x_words, y_words, result_words, modulus, modulus_is_zero, byte_trace);
      break;
    case FieldOperation::Mul:
      populate_mul<F, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
        cols, x_words, y_words, result_words, modulus, modulus_is_zero, byte_trace);
      break;
    case FieldOperation::Sub:
      populate_sub<F, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
        cols, x_words, y_words, result_words, modulus, modulus_is_zero, byte_trace);
      break;
    case FieldOperation::Div:
      assert(!modulus_is_zero);
      populate_div<F, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
        cols, x_words, y_words, result_words, modulus, mod_mins_two, byte_trace);
      break;
    }
  }

  // populate for FieldInnerProductCols
  // calculates a[0] * b[0] + a[1] * b[1] + ... a[n] * b[n] % MODULUS
  template <
    class F,
    const int NumElems,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_inner_prod(
    FieldInnerProductCols<F, NumLimbs, NumWitnesses>& cols, // Output columns
    const uint32_t* a[NumElems],                            // input a[NumWords]
    const uint32_t* b[NumElems],                            // input b[NumWords]
    uint32_t result[NumWords],                              // output result
    const uint32_t modulus[NumWords],                       // modulus
    F* byte_trace)
  {
    uint32_t modmul_result[NumElems][NumWords];
    uint32_t carry[NumElems][NumWords];
    uint32_t final_result[NumWords] = {0};
    uint32_t final_carry[NumWords] = {0};

    // compute carry[i] * MODULUS + modmul_result[i] = a[i] * b[i]
    for (int i = 0; i < NumElems; i++) {
      uint32_t tmp[2 * NumWords];
      full_multiply_words_to_double_words<NumWords>(a[i], b[i], tmp);

      // Performance-heavy step:
      // TODO: maybe optimize this so that product does not have to be twice the size of modulus
      // this way we can add a[i] * b[i] directly together
      divide_double_words_by_words_binary_search<NumWords>(
        tmp, modulus, carry[i], modmul_result[i] /* remainder */, false);
    }

    // start accumulating the dot product addition into final_result and final_carry
    for (int i = 0; i < NumElems; i++) {
      uint8_t tmp_carry[1];

      add_words_mod_p<NumWords, 1>(final_result, modmul_result[i], final_result, tmp_carry, modulus);

      // full addition for final_carry
      // set the initial carry flag by left shifting 32 bits
      uint64_t acc = (uint64_t)tmp_carry[0] << 32;
      for (int j = 0; j < NumWords; j++) {
        acc = static_cast<uint64_t>(final_carry[j]) + carry[i][j] + (acc >> 32);
        final_carry[j] = static_cast<uint32_t>(acc);
      }
    }

    // now that final_result is correct, copy it to result
    for (int i = 0; i < NumWords; i++) {
      result[i] = final_result[i];
    }

    // p_vanishing is a polynomial intended to be evaluated at base 256 i.e.
    // a_0 + a_1 * 2^8 + a_2 * 2^16 + ... + a_n * 2^(8n)
    int64_t p_vanishing[2 * NumLimbs - 1] = {0};

    // p_vanishing = p_a[0] * p_b[0] + ... + p_a[n] * p_b[n]
    for (int i = 0; i < NumElems; i++) {
      uint8_t a_bytes[NumLimbs];
      uint8_t b_bytes[NumLimbs];
      int64_t inner_prod_poly[2 * NumLimbs - 1] = {0};
      words_to_bytes_le<NumWords, NumLimbs>(a[i], a_bytes);
      words_to_bytes_le<NumWords, NumLimbs>(b[i], b_bytes);
      poly_mul_coeffs<NumLimbs>(a_bytes, b_bytes, inner_prod_poly);

      // add to inner_prod poly
      for (int j = 0; j < 2 * NumLimbs - 1; j++) {
        p_vanishing[j] += inner_prod_poly[j];
      }
    }

    uint8_t result_bytes[NumLimbs];
    uint8_t carry_bytes[NumLimbs];
    uint8_t modulus_bytes[NumLimbs];
    int64_t p_carrymod[2 * NumLimbs - 1];
    words_to_bytes_le<NumWords, NumLimbs>(final_result, result_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(final_carry, carry_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(modulus, modulus_bytes);
    poly_mul_coeffs<NumLimbs>(carry_bytes, modulus_bytes, p_carrymod);

    // p_vanishing = p_inner_prod - p_result
    for (int i = 0; i < NumLimbs; i++) {
      p_vanishing[i] -= result_bytes[i];
    }

    // p_vanishing = p_inner_prod - p_result - p_carry * p_modulus
    for (int i = 0; i < 2 * NumLimbs - 1; i++) {
      p_vanishing[i] -= p_carrymod[i];
    }

    // 6. Compute quotient polynomial Q_prime(X) = V(X) / (X - base)
    // Degree of Q_prime(X) is NumWitnesses, 2*NumLimbs - 2 or 2*NumLimbs - 1 when the modulus is 1 << 8 * NumLimbs.
    constexpr int QuotientOutputCoeffsSize = NumWitnesses;
    int64_t q_prime_coeffs[QuotientOutputCoeffsSize];
    synthetic_division_mul<NumLimbs, BitsPerLimb, NumWitnesses>(p_vanishing, q_prime_coeffs);

    // 7. Generate witness values from Q_prime(X)
    fill_witness_limbs_mul<F, NumWitnesses, WitnessOffset>(
      q_prime_coeffs, cols.witness_low._0, cols.witness_high._0, byte_trace);

    // also store result/carry into columns
    words_to_t_bytes_le<F, NumWords, NumLimbs>(final_result, cols.result._0, byte_trace);
    words_to_t_bytes_le<F, NumWords, NumLimbs>(final_carry, cols.carry._0, byte_trace);
  }

  // populate for FieldDenCols
  // calculates a / (1 + b) if sign
  // calculates a / (1 - b) if !sign
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void populate_den(
    FieldDenCols<F, NumLimbs, NumWitnesses>& cols, // output columns
    const uint32_t a[NumWords],                    // input a
    const uint32_t b[NumWords],                    // input b
    uint32_t result[NumWords],                     // output result
    const uint32_t modulus[NumWords],              // modulus
    const uint32_t mod_minus_two[NumWords],        // modulus - 2 for inv
    bool sign,                                     // sign
    F* byte_trace)
  {
    uint32_t minus_b_int[NumWords];
    mod_sub_words<NumWords>(modulus, b, minus_b_int, modulus, false);
    uint32_t b_signed[NumWords];
    if (sign) {
      for (int i = 0; i < NumWords; i++) {
        b_signed[i] = b[i];
      }
    } else {
      for (int i = 0; i < NumWords; i++) {
        b_signed[i] = minus_b_int[i];
      }
    }
    uint32_t denominator[NumWords];
    add_scalar_to_words<NumWords>(b_signed, 1, denominator);
    uint32_t den_inv[NumWords];
    modpow_words<NumWords, NumLimbs>(denominator, mod_minus_two, modulus, den_inv);
    // uint32_t result[NumWords]; // already declared in arg
    mod_mul_words<NumWords, NumLimbs>(a, den_inv, modulus, result);
    uint32_t equation_lhs[2 * NumWords];
    full_multiply_words_to_double_words<NumWords>(b, result, equation_lhs);
    uint32_t doublewide_result_a[2 * NumWords] = {0};
    if (sign) {
      for (int i = 0; i < NumWords; i++) {
        doublewide_result_a[i] = result[i];
      }
    } else {
      for (int i = 0; i < NumWords; i++) {
        doublewide_result_a[i] = a[i];
      }
    }
    add_words_with_carry_norestrict<2 * NumWords>(equation_lhs, doublewide_result_a, equation_lhs);
    uint32_t equation_rhs[2 * NumWords] = {0};
    if (sign) {
      for (int i = 0; i < NumWords; i++) {
        equation_rhs[i] = a[i];
      }
    } else {
      for (int i = 0; i < NumWords; i++) {
        equation_rhs[i] = result[i];
      }
    }
    uint32_t difference[2 * NumWords] = {0};
    sub_words_with_borrow<2 * NumWords>(equation_lhs, equation_rhs, difference);
    uint32_t carry[NumWords];
    uint32_t remainder_should_be_zero[NumWords];
    divide_double_words_by_words_binary_search<NumWords>(difference, modulus, carry, remainder_should_be_zero, false);

    uint8_t a_bytes[NumLimbs];
    uint8_t b_bytes[NumLimbs];
    uint8_t p_bytes[NumLimbs];
    uint8_t result_bytes[NumLimbs];
    uint8_t carry_bytes[NumLimbs];
    int64_t p_b_result[2 * NumLimbs - 1];
    int64_t p_carry_p[2 * NumLimbs - 1];
    words_to_bytes_le<NumWords, NumLimbs>(a, a_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(b, b_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(modulus, p_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(result, result_bytes);
    words_to_bytes_le<NumWords, NumLimbs>(carry, carry_bytes);
    poly_mul_coeffs<NumLimbs>(b_bytes, result_bytes, p_b_result);
    poly_mul_coeffs<NumLimbs>(carry_bytes, p_bytes, p_carry_p);

    int64_t p_vanishing[2 * NumLimbs - 1];
    // p_vanishing = p_b * p_result - p_carry * p_p
    for (int i = 0; i < 2 * NumLimbs - 1; i++) {
      p_vanishing[i] = p_b_result[i] - p_carry_p[i];
    }

    // p_vanishing = p_b * p_result + f(sign) * (p_result - p_a) - p_carry * p_p
    for (int i = 0; i < NumLimbs; i++) {
      int64_t diff = sign ? result_bytes[i] - a_bytes[i] : a_bytes[i] - result_bytes[i];
      p_vanishing[i] += diff;
    }

    // 6. Compute quotient polynomial Q_prime(X) = V(X) / (X - base)
    // Degree of Q_prime(X) is NumWitnesses, 2*NumLimbs - 2 or 2*NumLimbs - 1 when the modulus is 1 << 8 * NumLimbs.
    constexpr int QuotientOutputCoeffsSize = NumWitnesses;
    int64_t q_prime_coeffs[QuotientOutputCoeffsSize];
    synthetic_division_mul<NumLimbs, BitsPerLimb, NumWitnesses>(p_vanishing, q_prime_coeffs);

    // 7. Generate witness values from Q_prime(X)
    fill_witness_limbs_mul<F, NumWitnesses, WitnessOffset>(
      q_prime_coeffs, cols.witness_low._0, cols.witness_high._0, byte_trace);

    // also store result/carry into columns
    words_to_t_bytes_le<F, NumWords, NumLimbs>(result, cols.result._0, byte_trace);
    words_to_t_bytes_le<F, NumWords, NumLimbs>(carry, cols.carry._0, byte_trace);
  }

  // Main device function to convert an FpEvent into a row of the trace matrix (FpOpCols).
  template <
    class F,
    const int NumWords,
    const int NumLimbs,
    const int NumWitnesses,
    const int BitsPerLimb,
    const int WitnessOffset>
  __PICO_HOSTDEV__ inline void event_to_row(
    const FpEvent<NumWords>& event,                      // Input event data
    FpOpCols<F, NumWords, NumLimbs, NumWitnesses>& cols, // Output row columns
    const uint32_t modulus[NumWords],                    // Field modulus p
    const uint32_t mod_mins_two[NumWords],               // p-2, for division
    F* byte_trace)
  {
    // Set boolean flags indicating the operation type
    cols.is_add = F::from_bool(event.op == FieldOperation::Add);
    cols.is_sub = F::from_bool(event.op == FieldOperation::Sub);
    cols.is_mul = F::from_bool(event.op == FieldOperation::Mul);
    cols.is_real = F::from_bool(true);

    cols.chunk = F::from_canonical_u32(event.chunk);
    cols.clk = F::from_canonical_u32(event.clk);
    cols.x_ptr = F::from_canonical_u32(event.x_ptr);
    cols.y_ptr = F::from_canonical_u32(event.y_ptr);

    uint32_t result_words[NumWords];
    populate_with_modulus<F, NumWords, NumLimbs, NumWitnesses, BitsPerLimb, WitnessOffset>(
      cols.output, event.x, event.y, result_words, modulus, mod_mins_two, event.op, false, byte_trace);

    for (int i = 0; i < NumWords; i++) {
      populate(cols.y_access[i], event.y_memory_records[i], byte_trace);
    }
    for (int i = 0; i < NumWords; i++) {
      populate(cols.x_access[i], event.x_memory_records[i], byte_trace);
    }

    // new_byte_lookup_events
    //   .iter()
    //   .for_each(|x| output.add_byte_lookup_event(*x));
  }
} // namespace pico_gpu::fp_op

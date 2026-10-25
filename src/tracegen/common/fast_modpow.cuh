// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cstdint>

// Self-contained Montgomery multiplication for GPU acceleration of modular exponentiation.
// Works on any CUDA compute capability.

namespace fast_mont {

// Montgomery multiplication: result = (a * b * R^(-1)) mod mod
// All inputs and outputs are in Montgomery form (or used to convert).
// Uses the CIOS (Coarsely Integrated Operand Scanning) method.
template <int N>
__device__ __forceinline__
void mont_mul(const uint32_t a[N], const uint32_t b[N],
              const uint32_t mod[N], uint32_t m0,
              uint32_t result[N]) {
  // t has N+2 limbs to hold intermediate results without overflow.
  uint32_t t[N + 2] = {0};

  for (int i = 0; i < N; i++) {
    // Step 1: t = t + a[i] * b
    uint64_t carry = 0;
    for (int j = 0; j < N; j++) {
      uint64_t prod = (uint64_t)a[i] * b[j] + t[j] + carry;
      t[j] = (uint32_t)prod;
      carry = prod >> 32;
    }
    uint64_t sum = (uint64_t)t[N] + carry;
    t[N] = (uint32_t)sum;
    t[N + 1] = (uint32_t)(sum >> 32);

    // Step 2: Montgomery reduction
    // m = t[0] * m0 mod 2^32
    uint32_t m = t[0] * m0;

    // t = (t + m * mod) / 2^32  (i.e., shift right by one limb)
    carry = 0;
    uint64_t prod0 = (uint64_t)m * mod[0] + t[0];
    carry = prod0 >> 32;  // t[0] is discarded (divided out)

    for (int j = 1; j < N; j++) {
      uint64_t prod = (uint64_t)m * mod[j] + t[j] + carry;
      t[j - 1] = (uint32_t)prod;
      carry = prod >> 32;
    }
    sum = (uint64_t)t[N] + carry;
    t[N - 1] = (uint32_t)sum;
    t[N] = t[N + 1] + (uint32_t)(sum >> 32);
    t[N + 1] = 0;
  }

  // Final reduction: if t >= mod, subtract mod
  bool ge = false;
  if (t[N] != 0) {
    ge = true;
  } else {
    for (int i = N - 1; i >= 0; i--) {
      if (t[i] > mod[i]) { ge = true; break; }
      if (t[i] < mod[i]) { ge = false; break; }
    }
    // If all equal, t == mod means we should subtract
    if (!ge) {
      bool all_eq = true;
      for (int i = 0; i < N; i++) {
        if (t[i] != mod[i]) { all_eq = false; break; }
      }
      if (all_eq) ge = true;
    }
  }

  if (ge) {
    uint64_t borrow = 0;
    for (int i = 0; i < N; i++) {
      uint64_t diff = (uint64_t)t[i] - mod[i] - borrow;
      result[i] = (uint32_t)diff;
      borrow = (diff >> 63) & 1;
    }
  } else {
    for (int i = 0; i < N; i++) result[i] = t[i];
  }
}

// Convert to Montgomery form: aR = MonPro(a, R^2 mod p)
template <int N>
__device__ __forceinline__
void to_mont(const uint32_t a[N], const uint32_t rr[N],
             const uint32_t mod[N], uint32_t m0,
             uint32_t result[N]) {
  mont_mul<N>(a, rr, mod, m0, result);
}

// Convert from Montgomery form: a = MonPro(aR, 1)
template <int N>
__device__ __forceinline__
void from_mont(const uint32_t a_mont[N],
               const uint32_t mod[N], uint32_t m0,
               uint32_t result[N]) {
  uint32_t one[N] = {0};
  one[0] = 1;
  mont_mul<N>(a_mont, one, mod, m0, result);
}

// Montgomery modular exponentiation: result = base^exp mod mod
// Uses square-and-multiply with Montgomery multiplication.
template <int N>
__device__ __forceinline__
void mont_modpow(const uint32_t base[N],
                 const uint32_t exp[N],
                 const uint32_t mod[N],
                 const uint32_t rr[N],     // R^2 mod p
                 uint32_t m0,
                 uint32_t result[N]) {
  // Convert base to Montgomery form
  uint32_t base_mont[N];
  to_mont<N>(base, rr, mod, m0, base_mont);

  // result_mont = R mod p = to_mont(1)
  uint32_t result_mont[N];
  uint32_t one_val[N] = {0};
  one_val[0] = 1;
  to_mont<N>(one_val, rr, mod, m0, result_mont);

  uint32_t current_power[N];
  for (int i = 0; i < N; i++) current_power[i] = base_mont[i];

  uint32_t temp_exp[N];
  for (int i = 0; i < N; i++) temp_exp[i] = exp[i];

  uint32_t temp[N];

  // Binary exponentiation (right-to-left)
  while (true) {
    // Check if exponent is zero
    bool all_zero = true;
    for (int i = 0; i < N; i++) {
      if (temp_exp[i] != 0) { all_zero = false; break; }
    }
    if (all_zero) break;

    // If LSB is 1, multiply result by current_power
    if (temp_exp[0] & 1U) {
      mont_mul<N>(result_mont, current_power, mod, m0, temp);
      for (int i = 0; i < N; i++) result_mont[i] = temp[i];
    }

    // Right shift exponent
    uint32_t carry = 0;
    for (int i = N - 1; i >= 0; i--) {
      uint32_t val = temp_exp[i];
      temp_exp[i] = (val >> 1) | (carry << 31);
      carry = val & 1U;
    }

    // Square current_power
    mont_mul<N>(current_power, current_power, mod, m0, temp);
    for (int i = 0; i < N; i++) current_power[i] = temp[i];
  }

  // Convert result from Montgomery form
  from_mont<N>(result_mont, mod, m0, result);
}

} // namespace fast_mont


// =============================================================================
// Precomputed Montgomery parameters for each supported curve.
// R^2 mod p and m0 = -p^(-1) mod 2^32.
// =============================================================================

namespace mont_params {

// secp256k1: p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
__device__ __constant__ const uint32_t SECP256K1_RR[8] = {
  0x000e90a1, 0x000007a2, 0x00000001, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000
};
constexpr uint32_t SECP256K1_M0 = 0xd2253531u;

// secp256r1: p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
__device__ __constant__ const uint32_t SECP256R1_RR[8] = {
  0x00000003, 0xfffffffb, 0xfffffffe, 0x00000004,
  0xfffffffd, 0x00000002, 0x00000001, 0x00000000
};
constexpr uint32_t SECP256R1_M0 = 0x00000001u;

// bn254: p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
__device__ __constant__ const uint32_t BN254_RR[8] = {
  0xf32cfc5b, 0xb786686c, 0x006e22d1, 0x64774b84,
  0x482d9af2, 0xa4a61580, 0x000006c4, 0x0e0a77c1
};
constexpr uint32_t BN254_M0 = 0xe4866389u;

// bls12-381: p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
__device__ __constant__ const uint32_t BLS12381_RR[12] = {
  0x1c341746, 0xf4df1f34, 0x09d104f1, 0x0a76e6a6,
  0x4c95b6d5, 0x8de5476c, 0x939d83c0, 0x67eb88a9,
  0xb519952d, 0x9a793e85, 0x92cae3aa, 0x11988fe5
};
constexpr uint32_t BLS12381_M0 = 0xfffcfffdu;

} // namespace mont_params

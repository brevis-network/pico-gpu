#pragma once

#include "fp_op.hpp"

namespace pico_gpu {
  static constexpr int NumBitsPerLimb = 8;

  namespace params_secp256k1 {
    static constexpr int NumLimbs = 32;
    static constexpr int NumWords =
      NumLimbs / 4; // 32 limbs = 8 u32 words, corresponding to WordsFieldElement in CPU code.
    static constexpr int NumWitnesses = 62;
    static constexpr int WitnessOffset = 1 << 14;

    // Secp256k1 modulus (p) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t MODULUS[NumWords] = {
      TO_CUDA_T(0xfffffffefffffc2f), // LSB
      TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
      TO_CUDA_T(0xffffffffffffffff) // MSB
    };

    // MODULUS - 2 (p-2) in little-endian u32 words.
    // Used for modular inverse via Fermat's Little Theorem: a^(p-2) mod p.
    static __device__ __constant__ __align__(16) const uint32_t MOD_MINUS_TWO[NumWords] = {
      TO_CUDA_T(0xfffffffefffffc2d), // LSB
      TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
      TO_CUDA_T(0xffffffffffffffff) // MSB
    };

    // (MODULUS + 1) / 4 ((p+1)/4) in little-endian u32 words.
    // Used for square root computations
    static __device__ __constant__ __align__(16) const uint32_t SQRT_EXP[NumWords] = {
      TO_CUDA_T(0xffffffffbfffff0c), // LSB
      TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
      TO_CUDA_T(0x3fffffffffffffff) // MSB
    };

    // Secp256k1 coefficient (a) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t WEIERSTRASS_A[NumWords] = {
      TO_CUDA_T(0x0000000000000000ull), // LSB
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull),
      TO_CUDA_T(0x0000000000000000ull) // MSB
    };

    // Secp256k1 coefficient (b) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t WEIERSTRASS_B[NumWords] = {
      TO_CUDA_T(0x0000000000000007ull), // LSB
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull),
      TO_CUDA_T(0x0000000000000000ull) // MSB
    };

    // Secp256k1 generator x in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t GEN_X[NumWords] = {
      TO_CUDA_T(0x59f2815b16f81798), // LSB
      TO_CUDA_T(0x029bfcdb2dce28d9), TO_CUDA_T(0x55a06295ce870b07),
      TO_CUDA_T(0x79be667ef9dcbbac) // MSB
    };

    // Secp256k1 generator y in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t GEN_Y[NumWords] = {
      TO_CUDA_T(0x9c47d08ffb10d4b8), // LSB
      TO_CUDA_T(0xfd17b448a6855419), TO_CUDA_T(0x5da4fbfc0e1108a8),
      TO_CUDA_T(0x483ada7726a3c465) // MSB
    };
  } // namespace params_secp256k1

  namespace params_bn254 {
    static constexpr int NumLimbs = 32;
    static constexpr int NumWords =
      NumLimbs / 4; // 32 limbs = 8 u32 words, corresponding to WordsFieldElement in CPU code.
    static constexpr int NumWitnesses = 62;
    static constexpr int WitnessOffset = 1 << 14;

    // BN254 modulus (p) in little-endian u32 words.
    // Value from ff/alt_bn128.hpp
    static __device__ __constant__ __align__(16) const uint32_t MODULUS[NumWords] = {
      TO_CUDA_T(0x3c208c16d87cfd47), // LSB
      TO_CUDA_T(0x97816a916871ca8d),
      TO_CUDA_T(0xb85045b68181585d),
      TO_CUDA_T(0x30644e72e131a029), // MSB
    };

    // MODULUS - 2 (p-2) in little-endian u32 words.
    // Used for modular inverse via Fermat's Little Theorem: a^(p-2) mod p.
    static __device__ __constant__ __align__(16) const uint32_t MOD_MINUS_TWO[NumWords] = {
      TO_CUDA_T(0x3c208c16d87cfd45), // LSB
      TO_CUDA_T(0x97816a916871ca8d),
      TO_CUDA_T(0xb85045b68181585d),
      TO_CUDA_T(0x30644e72e131a029), // MSB
    };

    // BN254 coefficient (a) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t WEIERSTRASS_A[NumWords] = {
      TO_CUDA_T(0x0000000000000000ull), // LSB
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull),
      TO_CUDA_T(0x0000000000000000ull) // MSB
    };
  } // namespace params_bn254

  namespace params_bls12381 {
    static constexpr int NumLimbs = 48;
    static constexpr int NumWords =
      NumLimbs / 4; // 48 limbs = 12 u32 words, corresponding to WordsFieldElement in CPU code.
    static constexpr int NumWitnesses = 94;
    static constexpr int WitnessOffset = 1 << 15;

    // BLS12-381 modulus (p) in little-endian u32 words.
    // Value from ff/bls12-381.hpp
    static __device__ __constant__ __align__(16) const uint32_t MODULUS[NumWords] = {
      TO_CUDA_T(0xb9feffffffffaaab), // LSB
      TO_CUDA_T(0x1eabfffeb153ffff), TO_CUDA_T(0x6730d2a0f6b0f624), TO_CUDA_T(0x64774b84f38512bf),
      TO_CUDA_T(0x4b1ba7b6434bacd7), TO_CUDA_T(0x1a0111ea397fe69a) // MSB
    };

    // MODULUS - 2 (p-2) in little-endian u32 words.
    // Used for modular inverse via Fermat's Little Theorem: a^(p-2) mod p.
    static __device__ __constant__ __align__(16) const uint32_t MOD_MINUS_TWO[NumWords] = {
      TO_CUDA_T(0xb9feffffffffaaa9), // LSB
      TO_CUDA_T(0x1eabfffeb153ffff), TO_CUDA_T(0x6730d2a0f6b0f624), TO_CUDA_T(0x64774b84f38512bf),
      TO_CUDA_T(0x4b1ba7b6434bacd7), TO_CUDA_T(0x1a0111ea397fe69a) // MSB
    };

    // (MODULUS + 1) / 4 ((p+1)/4) in little-endian u32 words.
    // Used for square root computations
    static __device__ __constant__ __align__(16) const uint32_t SQRT_EXP[NumWords] = {
      TO_CUDA_T(0xee7fbfffffffeaab), // LSB
      TO_CUDA_T(0x07aaffffac54ffff), TO_CUDA_T(0xd9cc34a83dac3d89), TO_CUDA_T(0xd91dd2e13ce144af),
      TO_CUDA_T(0x92c6e9ed90d2eb35), TO_CUDA_T(0x0680447a8e5ff9a6) // MSB
    };

    // BLS12-381 coefficient (a) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t WEIERSTRASS_A[NumWords] = {
      TO_CUDA_T(0x0000000000000000ull), // LSB
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull),
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull) // MSB
    };

    // BLS12-381 coefficient (b) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t WEIERSTRASS_B[NumWords] = {
      TO_CUDA_T(0x0000000000000004ull), // LSB
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull),
      TO_CUDA_T(0x0000000000000000ull), TO_CUDA_T(0x0000000000000000ull) // MSB
    };

    // BLS12-381 generator x in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t GEN_X[NumWords] = {
      TO_CUDA_T(0xFB3AF00ADB22C6BB), // LSB
      TO_CUDA_T(0x6C55E83FF97A1AEF), TO_CUDA_T(0xA14E3A3F171BAC58), TO_CUDA_T(0xC3688C4F9774B905),
      TO_CUDA_T(0x2695638C4FA9AC0F), TO_CUDA_T(0x17F1D3A73197D794) // MSB
    };

    // BLS12-381 generator y in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t GEN_Y[NumWords] = {
      TO_CUDA_T(0x08B3F481E3AAA0F1), // LSB
      TO_CUDA_T(0xA09E30ED741D8AE4), TO_CUDA_T(0xFCF5E095D5D00AF6), TO_CUDA_T(0x00DB18CB2C04B3ED),
      TO_CUDA_T(0xD03CC744A2888AE4), TO_CUDA_T(0x0CAA232946C5E7E1) // MSB
    };
  } // namespace params_bls12381

  namespace params_ed25519 {
    static constexpr int NumLimbs = 32;
    static constexpr int NumWords =
      NumLimbs / 4; // 32 limbs = 8 u32 words, corresponding to WordsFieldElement in CPU code.
    static constexpr int NumWitnesses = 62;
    static constexpr int WitnessOffset = 1 << 14;

    // Ed25519 modulus (p) in little-endian u32 words.
    static __device__ __constant__ __align__(16) const uint32_t MODULUS[NumWords] = {
      TO_CUDA_T(0xffffffffffffffed), // LSB
      TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
      TO_CUDA_T(0x7fffffffffffffff) // MSB
    };

    // MODULUS - 2 (p-2) in little-endian u32 words.
    // Used for modular inverse via Fermat's Little Theorem: a^(p-2) mod p.
    static __device__ __constant__ __align__(16) const uint32_t MOD_MINUS_TWO[NumWords] = {
      TO_CUDA_T(0xffffffffffffffeb), // LSB
      TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff),
      TO_CUDA_T(0x7fffffffffffffff) // MSB
    };

    // Edwards d value (ax^2 + y^2 = 1 + d x^2 + y^2)
    static __device__ __constant__ __align__(16) const uint32_t EDWARDS_D[NumWords] = {
      TO_CUDA_T(0x75eb4dca135978a3), // LSB
      TO_CUDA_T(0x00700a4d4141d8ab),
      TO_CUDA_T(0x8cc740797779e898),
      TO_CUDA_T(0x52036cee2b6ffe73), // MSB
    };
  } // namespace params_ed25519
} // namespace pico_gpu

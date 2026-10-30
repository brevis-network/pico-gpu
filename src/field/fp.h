// Modified by Brevis, from OpenVM stark-backend's version
// Copyright 2024 RISC Zero, Inc.
// SPDX-License-Identifier: Apache-2.0

/*
 * KoalaBear field override for GPU proving kernels.
 *
 * Included as "field/fp.h", so that Fp = kb31_t and FpExt = kb31_4_t for all
 * kernel code. This file directly defines Fp and FpExt wrapping kb31_t/kb31_4_t.
 *
 * KoalaBear prime: p = 2^31 - 2^24 + 1 = 0x7F000001
 * Extension field: KoalaBear[x] / (x^4 - 3)  (W = 3 is a non-residue)
 */

#pragma once

#include <cassert>
#include <cstdint>
#include "ff/koala_bear.hpp"
#include "ff/koala_bear_ext.hpp"

// ---------------------------------------------------------------------------
// Fp = kb31_t wrapper
// ---------------------------------------------------------------------------

class Fp : public kb31_t {
  public:
    static constexpr uint32_t P            = 0x7F000001;   // KoalaBear prime
    static constexpr uint32_t M            = 0x81000001;   // -(p^{-1}) mod 2^32, i.e. M0 negated
    static constexpr uint32_t R2           = 0x17F7EFE4;   // R^2 mod p
    static constexpr uint32_t MONTY_BITS   = 32;
    static constexpr uint32_t MONTY_MASK   = 0xffffffff;
    static constexpr uint32_t HALF_P_PLUS_1 = (P + 1) >> 1;
    static constexpr uint32_t TWO_ADICITY  = 24;           // p-1 = 2^24 * 127
    // INV_2EXP_K = (2^{60})^{-1} mod p   (FIELD_BITS=31, k=2*31-2=60)
    // = pow(pow(2,60,p), p-2, p) = 0x07E81004
    static constexpr uint32_t INV_2EXP_K  = 0x07E81004u;

  private:
    __device__ uint32_t val() const {
        return static_cast<uint32_t>(*this);
    }

  public:
    __device__ constexpr Fp() : kb31_t(0) {}

    __device__ explicit constexpr Fp(const kb31_t& b) : kb31_t(b) {}

    __device__ Fp(const kb31_base& b) : kb31_t(b) {}

    __host__ __device__ constexpr Fp(uint32_t v) : kb31_t(static_cast<int>(v)) {}

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    __host__ __device__ constexpr Fp(I v) : kb31_t(static_cast<int>(v)) {}

    __device__ static constexpr Fp fromRaw(uint32_t val) {
        return Fp(kb31_t(val));
    }

    __device__ static constexpr Fp zero()    { return Fp(0); }
    __device__ static constexpr Fp one()     { return Fp(1); }
    __device__ static constexpr Fp neg_one() { return maxVal(); }
    __device__ static constexpr Fp maxVal()  { return P - 1; }
    __device__ static constexpr Fp invalid() { return Fp::fromRaw(0xfffffffful); }

    __device__ uint32_t asUInt32() const { return val(); }
    __device__ uint32_t asRaw()    const { return kb31_t::operator*(); }
    __device__ uint32_t get()      const { return asRaw(); }
    __device__ void     set(uint32_t input) { kb31_t::operator=(input); }

    // Equality on the stored (Montgomery) words; an int operand is converted to Fp first.
    friend __device__ inline bool operator==(const Fp& lhs, const Fp& rhs) {
        return lhs.asRaw() == rhs.asRaw();
    }
    friend __device__ inline bool operator!=(const Fp& lhs, const Fp& rhs) {
        return lhs.asRaw() != rhs.asRaw();
    }
    friend __device__ inline bool operator==(const Fp& lhs, int rhs) {
        return lhs.asRaw() == Fp(rhs).asRaw();
    }
    friend __device__ inline bool operator==(int lhs, const Fp& rhs) {
        return Fp(lhs).asRaw() == rhs.asRaw();
    }

    __device__ bool operator<(Fp rhs)  const { return val() < rhs.val(); }
    __device__ bool operator<=(Fp rhs) const { return val() <= rhs.val(); }
    __device__ bool operator>(Fp rhs)  const { return val() > rhs.val(); }
    __device__ bool operator>=(Fp rhs) const { return val() >= rhs.val(); }

    __device__ Fp operator++(int) { Fp r = *this; *this += Fp::one(); return r; }
    __device__ Fp operator--(int) { Fp r = *this; *this -= Fp::one(); return r; }
    __device__ Fp operator++()    { *this += Fp::one(); return *this; }
    __device__ Fp operator--()    { *this -= Fp::one(); return *this; }

    static __device__ inline uint32_t halve_u32(uint32_t input) {
        uint32_t shr    = input >> 1;
        uint32_t lo_bit = input & 1;
        return shr + (lo_bit * HALF_P_PLUS_1);
    }

    static __device__ uint32_t monty_reduce(uint64_t x) {
        // M here is the "neg_inv" value: (2^32 - M0) where M0 = 0x7EFFFFFF
        // The standard monty reduce: t = (x * (-1/p mod 2^32)) & mask
        // Standard Montgomery reduction with KoalaBear's P and M.
        uint64_t t        = (x * uint64_t(Fp::M)) & uint64_t(Fp::MONTY_MASK);
        uint64_t u        = t * uint64_t(Fp::P);
        uint64_t x_sub_u  = x - u;
        bool     overflow = x < u;
        uint32_t x_sub_u_hi = uint32_t(x_sub_u >> Fp::MONTY_BITS);
        uint32_t corr     = overflow ? (Fp::P) : 0;
        return x_sub_u_hi + corr;
    }

    __device__ Fp doubled() const { return *this + *this; }
    __device__ Fp halve()   const { return Fp::fromRaw(halve_u32(asRaw())); }

    __device__ Fp mul_2exp_neg_n(uint32_t n) const {
        assert(n < 33 && "n must be less than 33");
        uint64_t value_mul_2exp_neg_n = static_cast<uint64_t>(asRaw()) << (32 - n);
        return Fp::fromRaw(monty_reduce(value_mul_2exp_neg_n));
    }
};

__device__ inline Fp pow(Fp x, uint32_t n) {
    return Fp(static_cast<kb31_t>(x) ^ n);
}

template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
__device__ inline Fp pow(Fp x, I n) {
    return pow(x, static_cast<uint32_t>(n));
}

__device__ inline Fp inv_fermat(Fp x) {
    if (x.asRaw() == 0u) return Fp::zero();
    return Fp(static_cast<kb31_t>(x).reciprocal());
}

__device__ inline Fp inv(Fp x) {
    return inv_fermat(x);
}

// KoalaBear two-adic generators: gen[i] = primitive 2^i-th root of unity in GF(p).
// p-1 = 2^24 * 127.  g = 3 (a primitive root of GF(p)*).
// gen[i] = 3^(127 * 2^(24-i)) mod p.
// Values are in standard (non-Montgomery) form; kb31_t(int) converts to Montgomery.
constexpr __device__ Fp TWO_ADIC_GENERATORS[Fp::TWO_ADICITY + 1] = {
    Fp(0x00000001),  // 2^0  root: 1
    Fp(0x7f000000),  // 2^1  root: p-1 = -1
    Fp(0x7e010002),  // 2^2  root
    Fp(0x6832fe4a),  // 2^3  root
    Fp(0x08dbd69c),  // 2^4  root
    Fp(0x0a28f031),  // 2^5  root
    Fp(0x5c4a5b99),  // 2^6  root
    Fp(0x29b75a80),  // 2^7  root
    Fp(0x17668b8a),  // 2^8  root
    Fp(0x27ad539b),  // 2^9  root
    Fp(0x334d48c7),  // 2^10 root
    Fp(0x7744959c),  // 2^11 root
    Fp(0x768fc6fa),  // 2^12 root
    Fp(0x303964b2),  // 2^13 root
    Fp(0x3e687d4d),  // 2^14 root
    Fp(0x45a60e61),  // 2^15 root
    Fp(0x6e2f4d7a),  // 2^16 root
    Fp(0x163bd499),  // 2^17 root
    Fp(0x6c4a8a45),  // 2^18 root
    Fp(0x143ef899),  // 2^19 root
    Fp(0x514ddcad),  // 2^20 root
    Fp(0x484ef19b),  // 2^21 root
    Fp(0x205d63c3),  // 2^22 root
    Fp(0x68e7dd49),  // 2^23 root
    Fp(0x6ac49f88),  // 2^24 root (primitive root of order 2^24)
};

static_assert(std::is_trivially_copyable<Fp>::value, "Fp must be POD-ish");
static_assert(sizeof(Fp) == 4,    "Fp must be 4 bytes");
static_assert(alignof(Fp) == 4,   "Fp must be 4-byte aligned");
static_assert(sizeof(Fp) == sizeof(kb31_t),    "Fp and kb31_t sizes must match");
static_assert(alignof(Fp) == alignof(kb31_t),  "Fp and kb31_t align must match");

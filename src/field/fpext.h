// Modified by Brevis, from OpenVM stark-backend's version
// Copyright 2024 RISC Zero, Inc.
// SPDX-License-Identifier: Apache-2.0

/*
 * KoalaBear extension field override for GPU proving kernels.
 *
 * Wraps kb31_5_t. Included as "field/fpext.h", so that FpExt = kb31_5_t for all
 * KB kernel code.
 *
 * KoalaBear^5 = KoalaBear[x] / (x^5 + x^2 - 1)
 */

#pragma once

#include <type_traits>

#include "field/fp.h"   // KB override of fp.h => Fp = kb31_t
#include "field/koala_bear_quintic.hpp"

/// Element of GF(p^5) = GF(p)[X] / (X^5 + X^2 - 1), stored as the coefficients
/// elems[0] + elems[1]*X + ... + elems[4]*X^4 with each elems[k] an Fp (kb31_t).
struct FpExt {
    static constexpr uint32_t D = 5;

    union {
        Fp       elems[D];
        kb31_5_t rep;
    };

    /// The zero element.
    __device__ FpExt() : rep(kb31_t{0u}) {}

    /// Embeds a uint32_t value.
    __device__ explicit FpExt(uint32_t v) : rep(kb31_t{v}) {}

    /// Embeds a base-field element.
    __device__ explicit FpExt(Fp base) : rep(static_cast<kb31_t>(base)) {}

    /// Builds an element from its five coefficients, lowest degree first.
    __device__ FpExt(Fp c0, Fp c1, Fp c2, Fp c3, Fp c4) {
        elems[0] = c0;
        elems[1] = c1;
        elems[2] = c2;
        elems[3] = c3;
        elems[4] = c4;
    }

    __device__ FpExt operator+=(FpExt addend) {
        rep += addend.rep;
        return *this;
    }
    __device__ FpExt operator-=(FpExt subtrahend) {
        rep -= subtrahend.rep;
        return *this;
    }
    __device__ FpExt operator+(FpExt addend) const {
        FpExt sum = *this;
        sum += addend;
        return sum;
    }
    __device__ FpExt operator-(FpExt subtrahend) const {
        FpExt diff = *this;
        diff -= subtrahend;
        return diff;
    }
    __device__ FpExt operator-(Fp base_sub) const {
        FpExt diff = *this;
        diff.elems[0] -= base_sub;
        return diff;
    }
    __device__ FpExt operator-() const { return FpExt() - *this; }

    // Scaling by a base-field element (the Fp * FpExt order is a free function below).
    __device__ FpExt operator*=(Fp scalar) {
        rep *= static_cast<kb31_t>(scalar);
        return *this;
    }
    __device__ FpExt operator*(Fp scalar) const {
        FpExt prod = *this;
        prod *= scalar;
        return prod;
    }
    __device__ FpExt operator*=(FpExt factor) {
        rep *= factor.rep;
        return *this;
    }
    __device__ FpExt operator*(FpExt factor) const {
        FpExt prod = *this;
        prod *= factor;
        return prod;
    }

    __device__ bool operator==(FpExt other) const { return rep == other.rep; }
    __device__ bool operator!=(FpExt other) const { return rep != other.rep; }

    __device__ Fp constPart() const { return elems[0]; }
};

/// Fp on the left: multiplication commutes, so defer to the member operator.
__device__ inline FpExt operator*(Fp scalar, FpExt ext) { return ext * scalar; }

/// base^e.
__device__ inline FpExt pow(FpExt base, uint32_t e) {
    FpExt out; out.rep = base.rep ^ e;
    return out;
}

template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
__device__ inline FpExt pow(FpExt base, I e) {
    return pow(base, static_cast<uint32_t>(e));
}

/// Multiplicative inverse, delegated to kb31_5_t::reciprocal().
__device__ inline FpExt inv(FpExt elem) {
    FpExt recip;
    recip.rep = elem.rep.reciprocal();
    return recip;
}

// The Rust side's extension element is QuinticTrinomialExtensionField<KoalaBear>,
// repr(transparent) over [KoalaBear; 5]: 20 packed bytes, 4-byte aligned, and FpExt crosses the
// FFI by value. A non-trivial copy would make C++ pass it by hidden reference while Rust passes
// it by value, so trivial copyability is part of the ABI.
static_assert(sizeof(FpExt) == 20, "FpExt must be 20 packed bytes");
static_assert(alignof(FpExt) == 4, "FpExt must be 4-byte aligned, like [KoalaBear; 5]");
static_assert(sizeof(FpExt) == sizeof(kb31_5_t),   "FpExt and kb31_5_t sizes must match");
static_assert(alignof(FpExt) == alignof(kb31_5_t), "FpExt and kb31_5_t align must match");
static_assert(std::is_trivially_copyable_v<FpExt>, "FpExt crosses the FFI by value");

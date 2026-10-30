// Modified by Brevis, from OpenVM stark-backend's version
// Copyright 2024 RISC Zero, Inc.
// SPDX-License-Identifier: Apache-2.0

/*
 * KoalaBear extension field override for GPU proving kernels.
 *
 * Wraps kb31_4_t. Included as "field/fpext.h", so that FpExt = kb31_4_t for all
 * KB kernel code.
 *
 * KoalaBear^4 = KoalaBear[x] / (x^4 - 3)
 */

#pragma once

#include "field/fp.h"   // KB override of fp.h => Fp = kb31_t

/// Element of GF(p^4) = GF(p)[X] / (X^4 - 3), stored as the coefficients
/// elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*X^3 with each elems[k] an Fp (kb31_t).
struct FpExt {
    union {
        Fp       elems[4];
        kb31_4_t rep;
    };

    /// The zero element.
    __device__ FpExt() : rep(kb31_t{0u}) {}

    /// Embeds a uint32_t value.
    __device__ explicit FpExt(uint32_t v) : rep(kb31_t{v}) {}

    /// Embeds a base-field element.
    __device__ explicit FpExt(Fp base) : rep(static_cast<kb31_t>(base)) {}

    /// Builds an element from its four coefficients, lowest degree first.
    __device__ FpExt(Fp c0, Fp c1, Fp c2, Fp c3) {
        elems[0] = c0;
        elems[1] = c1;
        elems[2] = c2;
        elems[3] = c3;
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

/// Multiplicative inverse, delegated to kb31_4_t::reciprocal().
__device__ inline FpExt inv(FpExt elem) {
    FpExt recip;
    recip.rep = elem.rep.reciprocal();
    return recip;
}

static_assert(sizeof(FpExt) == 16, "FpExt must be 16 bytes");
static_assert(sizeof(FpExt) == sizeof(kb31_4_t),   "FpExt and kb31_4_t sizes must match");
static_assert(alignof(FpExt) == alignof(kb31_4_t), "FpExt and kb31_4_t align must match");

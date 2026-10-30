// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "field/fp.h"
#include "field/fpext.h"

// A fraction p / q over the extension field, kept unreduced (numerator and denominator are never
// normalised against each other).
struct FracExt {
    FpExt p;
    FpExt q;
};

// acc <- acc + term, using a/b + c/d = (a*d + b*c) / (b*d).
__device__ __forceinline__ void frac_add_inplace(FracExt& acc, FracExt const& term) {
    acc.p = acc.p * term.q + acc.q * term.p;
    acc.q = acc.q * term.q;
}

__device__ __forceinline__ FracExt frac_add(FracExt sum_in, const FracExt &addend) {
    frac_add_inplace(sum_in, addend);
    return sum_in;
}

// Inverse of frac_add_inplace: recovers the exact (numerator, denominator) pair that `acc` held
// before `term` was added.
__device__ __forceinline__ void frac_unadd_inplace(FracExt& acc, FracExt const& term) {
    FpExt q_inv = inv(term.q);
    acc.q = acc.q * q_inv;
    acc.p = (acc.p - acc.q * term.p) * q_inv;
}

/// Returns the `c` with frac_add(c, part) == total, matching numerator and denominator exactly.
/// This is not plain subtraction of fractions: subtracting would give an equal value, but with a
/// numerator and denominator scaled by some common factor.
__device__ __forceinline__ FracExt frac_unadd(FracExt total, const FracExt &part) {
    frac_unadd_inplace(total, part);
    return total;
}

__device__ __forceinline__ FpExt make_bool_ext(bool bit) { return FpExt(Fp(bit ? 1u : 0u)); }

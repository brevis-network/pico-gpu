// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/fp.h"
#include <cstdint>

constexpr uint32_t LOG_WARP_SIZE = 5;

/// base^(2^log_exp), via log_exp squarings.
__device__ __forceinline__ Fp exp_power_of_2(Fp base, uint32_t log_exp) {
    Fp acc = base;
    while (log_exp--) {
        acc.sqr();
    }
    return acc;
}

// Mean of the geometric series 1, r, ..., r^(count - 1) for a power-of-two `count`, i.e.
// (1 + r + ... + r^(count - 1)) / count.
__device__ __forceinline__ Fp avg_gp(Fp ratio, uint32_t count) {
#ifdef CUDA_DEBUG
    assert(count && !(count & (count - 1)));
#endif
    Fp mean = Fp::one();
    for (uint32_t step = 1; step < count; step <<= 1) {
        mean *= Fp::one() + ratio;
        mean = mean.halve();
        ratio *= ratio;
    }
    return mean;
}

/// Bit-reverses the low `width` bits of `bits` (the value is assumed to fit in `width` bits).
__device__ __forceinline__ uint32_t rev_len(uint32_t bits, uint32_t width) {
    return width ? (__brev(bits) >> (32 - width)) : 0;
}

__device__ __forceinline__ uint32_t with_rev_bits(uint32_t idx, uint32_t span) { return idx; }

/// Sets bits of `idx` from the top of a power-of-two `span` downward, one flag per bit: the first
/// flag lands on span/2, the next on span/4, and so on. For instance
/// with_rev_bits(0b110, 32, 1, 0) yields 0b10110.
template <typename... Flags>
__device__ __forceinline__ uint32_t
with_rev_bits(uint32_t idx, uint32_t span, bool lead, Flags &&...rest_flags) {
    span >>= 1;
    return with_rev_bits(idx | (lead * span), span, rest_flags...);
}

#define DISPATCH_BOOL(fn, sel, ...) ((sel) ? fn<true>(__VA_ARGS__) : fn<false>(__VA_ARGS__))

// Picks one of the four <bool, bool> instantiations of `fn` from two run-time flags.
#define DISPATCH_BOOL_PAIR(fn, sel1, sel2, ...)                                                    \
    ((sel1) ? ((sel2) ? fn<true, true>(__VA_ARGS__) : fn<true, false>(__VA_ARGS__))                \
            : ((sel2) ? fn<false, true>(__VA_ARGS__) : fn<false, false>(__VA_ARGS__)))

// Declares `dispatcher(n, b1, b2, args...)`, which forwards `args` to
// `target<N, B1, B2>(args...)` for the compile-time N equal to the run-time n, trying
// N = upper, upper - 1, ..., 1 in turn (cudaErrorInvalidValue when none matches). Meant for
// templates of the form <uint32_t N, bool B1, bool B2>.
#define DEFINE_DISPATCH_N_B1_B2(dispatcher, target, upper)                                         \
    template <uint32_t TRY_N, typename... Rest>                                                    \
    inline int dispatcher##_impl(uint32_t want_n, bool flag_a, bool flag_b, Rest &&...rest) {      \
        if (want_n == TRY_N) {                                                                     \
            if (flag_a) {                                                                          \
                return flag_b ? target<TRY_N, true, true>(std::forward<Rest>(rest)...)             \
                              : target<TRY_N, true, false>(std::forward<Rest>(rest)...);           \
            } else {                                                                               \
                return flag_b ? target<TRY_N, false, true>(std::forward<Rest>(rest)...)            \
                              : target<TRY_N, false, false>(std::forward<Rest>(rest)...);          \
            }                                                                                      \
        } else if constexpr (TRY_N == 1) {                                                         \
            return cudaErrorInvalidValue;                                                          \
        } else {                                                                                   \
            return dispatcher##_impl<TRY_N - 1>(want_n, flag_a, flag_b, std::forward<Rest>(rest)...); \
        }                                                                                          \
    }                                                                                              \
    template <typename... Rest>                                                                    \
    inline int dispatcher(uint32_t want_n, bool flag_a, bool flag_b, Rest &&...rest) {             \
        return dispatcher##_impl<upper>(want_n, flag_a, flag_b, std::forward<Rest>(rest)...);      \
    }

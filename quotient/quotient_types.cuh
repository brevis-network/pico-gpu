#pragma once

#include <cstdint>
#include <ff/kb31_extension_t.hpp>

struct EvalOp {
    unsigned char op;
    unsigned char src1_kind;
    unsigned char src2_kind;
    unsigned short dst;
    unsigned short src1;
    unsigned short src2;
};

template <typename Val>
struct LagrangeSelectorsAtPoint {
    Val first_row;
    Val last_row;
    Val transition;
    Val vanishing_inv;
};

template <typename Val>
struct LagrangeSelectors {
    Val *first_row;
    Val *last_row;
    Val *transition;
    Val *vanishing_inv;
};

template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    __host__ __device__ __forceinline__ size_t size() const { return 1 << log_n; }

    __host__ __device__ __forceinline__ LagrangeSelectorsAtPoint<Val> compute_selectors(Val generator, Val point) const {
        Val normalized_pt = point / shift;
        Val vanishing_poly = normalized_pt.exp_power_of_two(log_n) - Val::one();

        Val gen_inv = generator.reciprocal();

        Val first_row = vanishing_poly / (normalized_pt - Val::one());
        Val last_row = vanishing_poly / (normalized_pt - gen_inv);
        Val transition = normalized_pt - gen_inv;
        Val vanishing_inv = vanishing_poly.reciprocal();

        LagrangeSelectorsAtPoint<Val> result;
        result.first_row = first_row;
        result.last_row = last_row;
        result.transition = transition;
        result.vanishing_inv = vanishing_inv;

        return result;
    }
};

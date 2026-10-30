// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "field/fp.h"

// One monomial variable packed into a 32-bit word:
//   [3:0]   entry_type
//   [11:4]  part_index
//   [15:12] offset (row rotation)
//   [31:16] column_index
//
// entry_type codes (FROZEN ABI — must match cuda-mle/src/monomial.rs `mod entry_type`):
//   0   Preprocessed   preprocessed (fixed) trace column
//   1   Main           main (witness) trace column, possibly partitioned
//   2   (unused)       intentionally reserved hole — a retired `Entry` variant
//   3   Public         public value
//   8   IsFirstRow     boundary selector
//   9   IsLastRow      boundary selector
//   10  IsTransition   transition selector
struct PackedVar {
    uint32_t data;
    __device__ __forceinline__ uint8_t entry_type() const { return data & 0xF; }
    __device__ __forceinline__ uint8_t part_index() const { return (data >> 4) & 0xFF; }
    __device__ __forceinline__ uint8_t offset() const { return (data >> 12) & 0xF; }
    __device__ __forceinline__ uint16_t col_index() const { return data >> 16; }
};

// Per-monomial descriptor, 12 bytes; field order is chosen so there is no padding.
struct MonomialHeader {
    uint32_t var_offset;
    uint32_t term_offset;
    uint16_t num_vars;
    uint16_t num_terms;
};

// One (constraint_idx, coefficient) contribution of a monomial, 8 bytes. The device folds a
// monomial's terms into sum_i coefficient_i * lambda_pows[constraint_idx_i].
struct LambdaTerm {
    uint32_t constraint_idx;
    Fp coefficient;
};

// Ties a monomial to the interaction(s) it feeds. The device folds the terms as
//   numerator:   sum_i coefficient_i * eq_3bs[interaction_idx_i]
//   denominator: sum_i coefficient_i * beta_pows[field_idx_i] * eq_3bs[interaction_idx_i]
struct InteractionMonomialTerm {
    Fp coefficient;
    uint16_t interaction_idx;
    uint16_t field_idx; // denominator only: message position selecting the beta power; ignored for numerators
};

// ABI size contracts with cuda-mle/src/monomial.rs (checked on the Rust side by
// `packed_types_abi_contract`). Header-only static checks — zero codegen. Fp is 4 bytes.
static_assert(sizeof(PackedVar) == 4, "ABI contract with cuda-mle/src/monomial.rs");
static_assert(sizeof(MonomialHeader) == 12, "ABI contract with cuda-mle/src/monomial.rs");
static_assert(sizeof(LambdaTerm) == 8, "ABI contract with cuda-mle/src/monomial.rs");
static_assert(sizeof(InteractionMonomialTerm) == 8, "ABI contract with cuda-mle/src/monomial.rs");

// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "prover/dag/codec.cuh"
#include "prover/ntt/device_ntt.cuh"
#include "field/fp.h"
#include <cassert>
#include <cstdint>
#include <vector_types.h>

// Leaf lookup for the symbolic constraint DAG: turns a decoded source operand into its values on
// all evaluation cosets of the round-0 skip domain.
namespace symbolic_dag {

// Per-thread state for resolving DAG leaves on every coset at once. One inverse NTT turns the
// thread's trace sample into a coefficient kept in a register; each coset then re-shifts that
// coefficient and runs its own forward NTT.
template <uint32_t NUM_COSETS> struct NttEvalContext {
    const Fp *__restrict__ preprocessed;
    const Fp *const *__restrict__ main_parts;
    const Fp *__restrict__ public_values;
    Fp *__restrict__ inter_buffer; // per thread: buffer_size rows of NUM_COSETS values
    Fp *ntt_buffer;   // shared-memory NTT workspace, used only by the NEEDS_SHMEM path
    Fp omega_shifts[NUM_COSETS]; // per-coset multiplier g^((c+1)*ntt_idx_rev), filled in ahead of time
    uint32_t skip_domain;        // size of the skip domain, 2^l_skip
    uint32_t height;             // row count of the trace; smaller than num_x * skip_domain when lifted
    uint32_t buffer_stride;
    uint32_t buffer_size;
    uint32_t ntt_idx; // lane position inside the skip domain, in [0, skip_domain)
};

// Produces f(g^c * omega_skip^ntt_idx, x) for each coset c < NUM_COSETS, all from one thread.
//
// Steps: read the trace sample once; an identity coset (index 0 when FIRST_COSET_IS_IDENTITY, or
// the whole call when `identity_only` is set) takes the sample as-is; otherwise one inverse NTT
// yields a coefficient that every remaining coset reuses: scale it by that coset's shift, run a
// forward NTT and write the coset's slot.
//
// Without NEEDS_SHMEM (l_skip <= 5, i.e. the domain fits in a warp) the transforms are pure warp
// shuffles and the coefficient never leaves a register; this is the latency-critical case.
//
// FIRST_COSET_IS_IDENTITY is the compile-time identity flag of the lockstep kernel, while
// `identity_only` is the run-time one used by the coset-parallel kernel.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool FIRST_COSET_IS_IDENTITY = false>
__device__ __forceinline__ void ntt_coset_interpolate(
    Fp *__restrict__ out,          // NUM_COSETS values written here
    const Fp *__restrict__ column, // trace column of num_x * domain_size rows
    const Fp *coset_shift,         // NUM_COSETS multipliers prepared by the caller
    Fp *scratch,  // shared-memory workspace; ignored without NEEDS_SHMEM
    uint32_t lane_pos,
    uint32_t x_pos,
    uint32_t domain_size,
    uint32_t trace_height,
    uint8_t rot,
    bool identity_only = false // coset-parallel kernel working on the identity coset
) {
    uint32_t const log_domain = __ffs(domain_size) - 1;
    uint32_t const first_row = x_pos * domain_size;

#ifdef CUDA_DEBUG
    assert(lane_pos < domain_size);
#endif

    // Each thread fetches its own sample; rotation wraps around the trace. The trace height is a
    // power of two everywhere on this path (other code already takes `__ffs(height) - 1`), which
    // lets a mask stand in for the much slower `%`.
#ifdef CUDA_DEBUG
    assert(trace_height && !(trace_height & (trace_height - 1)));
#endif
    uint32_t const row = (first_row + lane_pos + rot) & (trace_height - 1);
    Fp sample = column[row];

    if (log_domain == 0) {
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            out[k] = sample;
        }
        return;
    }

    // Coset-parallel kernel, identity coset: no transform at all.
    if (identity_only) {
        out[0] = sample;
        return;
    }

    if constexpr (FIRST_COSET_IS_IDENTITY) {
        // Shift 1 means the evaluation is the sample itself.
        out[0] = sample;
    }

    // Every coset from here on needs the coefficient form.
    constexpr uint32_t first_shifted = FIRST_COSET_IS_IDENTITY ? 1 : 0;
    if constexpr (first_shifted < NUM_COSETS) {
        // A single inverse transform serves all remaining cosets.
        if constexpr (NEEDS_SHMEM) {
            scratch[lane_pos] = sample;
            __syncthreads();
        }
        device_ntt::ntt_natural_to_bitrev<true, NEEDS_SHMEM>(sample, scratch, lane_pos, log_domain);

        // Pin the coefficient to its own register; this matters most for the shuffle-only path.
        Fp const coef = sample;

        // One forward transform per coset, all reusing `scratch` in turn.
#pragma unroll
        for (uint32_t k = first_shifted; k < NUM_COSETS; k++) {
            Fp moved = coef * coset_shift[k];
            // No barrier is needed between the inverse transform and these forward ones: whichever
            // NEEDS_SHMEM path runs, the forward transform starts from the register `moved` and
            // rewrites every `scratch` slot before its first __syncthreads.
            device_ntt::ntt_bitrev_to_natural<false, NEEDS_SHMEM>(moved, scratch, lane_pos, log_domain);
            out[k] = moved;
        }
    }
}

// Resolves one DAG leaf for every coset and writes the NUM_COSETS values into `dst`. The whole
// block must enter together, because the NEEDS_SHMEM transforms contain __syncthreads.
//
// The flags mean the same as for ntt_coset_interpolate (`identity_coset` is the run-time one).
// Values that change from one loop iteration to the next (the selectors and x position) arrive as
// arguments instead of living in NttEvalContext: mutating fields of a struct that is also held by
// const reference led to miscompiled code on Blackwell.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool FIRST_COSET_IS_IDENTITY = false>
__device__ __forceinline__ void ntt_eval_dag_entry(
    Fp *__restrict__ dst, // NUM_COSETS values written here
    const SourceInfo &leaf,
    const NttEvalContext<NUM_COSETS> &ec,
    const Fp *first_row_sel, // one first-row selector per coset
    const Fp *last_row_sel,  // one last-row selector per coset
    uint32_t x_pos,
    bool identity_coset = false // coset-parallel kernel working on the identity coset
) {
    switch (leaf.type) {
    case ENTRY_PREPROCESSED: {
        const Fp *column_ptr = ec.preprocessed + ec.height * leaf.index;
        ntt_coset_interpolate<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
            dst,
            column_ptr,
            ec.omega_shifts,
            ec.ntt_buffer,
            ec.ntt_idx,
            x_pos,
            ec.skip_domain,
            ec.height,
            leaf.offset,
            identity_coset
        );
        return;
    }
    case ENTRY_MAIN: {
        auto part_base = ec.main_parts[leaf.part];
        const Fp *column_ptr = part_base + ec.height * leaf.index;
        ntt_coset_interpolate<NUM_COSETS, NEEDS_SHMEM, FIRST_COSET_IS_IDENTITY>(
            dst,
            column_ptr,
            ec.omega_shifts,
            ec.ntt_buffer,
            ec.ntt_idx,
            x_pos,
            ec.skip_domain,
            ec.height,
            leaf.offset,
            identity_coset
        );
        return;
    }
    case ENTRY_PUBLIC: {
        Fp uniform = ec.public_values[leaf.index];
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = uniform;
        }
        return;
    }
    case SRC_CONSTANT: {
        Fp uniform = Fp(leaf.index);
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = uniform;
        }
        return;
    }
    case SRC_INTERMEDIATE:
#ifdef CUDA_DEBUG
        assert(ec.buffer_size > 0);
        assert(leaf.index < ec.buffer_size);
#endif
        // Spilled intermediates sit row-major: one row of NUM_COSETS values per slot.
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = ec.inter_buffer[leaf.index * ec.buffer_stride + k];
        }
        return;
    case SRC_IS_FIRST:
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = first_row_sel[k];
        }
        return;
    case SRC_IS_LAST:
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = last_row_sel[k];
        }
        return;
    case SRC_IS_TRANSITION:
        // The transition selector is the complement of the last-row one.
#pragma unroll
        for (uint32_t k = 0; k < NUM_COSETS; k++) {
            dst[k] = Fp::one() - last_row_sel[k];
        }
        return;
    default:
        assert(false);
    }
#pragma unroll
    for (uint32_t k = 0; k < NUM_COSETS; k++) {
        dst[k] = Fp::zero();
    }
}

} // namespace symbolic_dag

// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "prover/dag/codec.cuh"
#include "field/fp.h"
#include "field/fpext.h"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"

// C++ standard library
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace logup_gkr_input_evaluation {

// ---------------------------------------------------------------------------- device code

// Resolves one DAG operand (trace cell, public value, challenge, buffered intermediate, constant or
// row selector) for row `r` of a trace with n_rows rows; trace reads wrap cyclically.
__device__ __forceinline__ FpExt load_gkr_operand(
    const SourceInfo &entry,
    uint32_t r,
    const Fp *prep,
    const uint64_t *main_parts,
    const Fp *pubs,
    const FpExt *chals,
    const FpExt *scratch,
    const uint32_t scratch_stride,
    const uint32_t n_rows
) {
    FpExt v(0);
    switch (entry.type) {
    case ENTRY_PREPROCESSED:
        v = FpExt(prep[n_rows * entry.index + ((r + entry.offset) % n_rows)]);
        break;
    case ENTRY_MAIN: {
        Fp *part_cols = (Fp *)main_parts[entry.part];
        v = FpExt(part_cols[n_rows * entry.index + ((r + entry.offset) % n_rows)]);
        break;
    }
    case ENTRY_PUBLIC:
        v = FpExt(pubs[entry.index]);
        break;
    case ENTRY_CHALLENGE:
        v = chals[entry.index];
        break;
    case SRC_INTERMEDIATE:
        v = scratch[scratch_stride * entry.index];
        break;
    case SRC_CONSTANT:
        v = FpExt(Fp(entry.index));
        break;
    case SRC_IS_FIRST:
        v = make_bool_ext(r == 0);
        break;
    case SRC_IS_LAST:
        v = make_bool_ext(r == n_rows - 1);
        break;
    case SRC_IS_TRANSITION:
        v = make_bool_ext(r != n_rows - 1);
        break;
    default:
        assert(0);
    }
    return v;
}

// Evaluates the interaction numerator/denominator DAG row by row and writes the requested nodes as
// GKR input fractions. Intermediates live in a small per-thread array, or with SPILL_TO_GLOBAL in a
// global buffer interleaved across threads. Each thread handles rows_per_thread rows strided by the
// total thread count.
template <bool SPILL_TO_GLOBAL>
__global__ void evaluate_interactions_gkr_kernel(
    FracExt *__restrict__ fracs_out,
    const Fp *__restrict__ prep,
    const uint64_t *__restrict__ main_parts,
    const Fp *__restrict__ pubs,
    const FpExt *__restrict__ chals,
    const FpExt *__restrict__ scratch_global,
    const Rule *__restrict__ rules,
    const size_t *__restrict__ used_nodes,
    const uint32_t *__restrict__ pair_slots,
    const size_t n_used,
    const uint32_t n_rows,
    const uint32_t rows_per_thread
) {
    uint32_t tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_threads = gridDim.x * blockDim.x;

    FpExt *scratch;
    uint32_t scratch_stride;
    if constexpr (SPILL_TO_GLOBAL) {
        scratch = (FpExt *)scratch_global + tid0;
        scratch_stride = n_threads;
    } else {
        FpExt local_scratch[10];
        scratch = local_scratch;
        scratch_stride = 1;
    }

    for (uint32_t t = 0; t < rows_per_thread; t++) {
        uint32_t r = tid0 + t * n_threads;

        if (r < n_rows) {
            uint32_t n_done = 0;
            // used_nodes lists the output nodes in numerator / denominator order; rules are
            // evaluated lazily up to the highest node requested so far.
            for (uint32_t u = 0; u < n_used; u++) {
                uint32_t node = used_nodes[u];
                FpExt v(0);
                if (node < n_done) {
                    Rule rw = rules[node];
                    RuleHeader hdr = decode_rule_header(rw);
                    if (hdr.op == OP_VAR) {
                        v = load_gkr_operand(
                            hdr.x,
                            r,
                            prep,
                            main_parts,
                            pubs,
                            chals,
                            scratch,
                            scratch_stride,
                            n_rows
                        );
                    } else {
                        uint32_t slot = decode_z_index(rw);
                        v = scratch[slot * scratch_stride];
                    }
                } else {
                    for (; n_done <= node; n_done++) {
                        Rule rw = rules[n_done];
                        RuleHeader hdr = decode_rule_header(rw);

                        FpExt lhs = load_gkr_operand(
                            hdr.x,
                            r,
                            prep,
                            main_parts,
                            pubs,
                            chals,
                            scratch,
                            scratch_stride,
                            n_rows
                        );
                        FpExt rhs;

                        switch (hdr.op) {
                        case OP_ADD:
                            rhs = load_gkr_operand(
                                decode_y(rw),
                                r,
                                prep,
                                main_parts,
                                pubs,
                                chals,
                                scratch,
                                scratch_stride,
                                n_rows
                            );
                            v = lhs + rhs;
                            break;
                        case OP_SUB:
                            rhs = load_gkr_operand(
                                decode_y(rw),
                                r,
                                prep,
                                main_parts,
                                pubs,
                                chals,
                                scratch,
                                scratch_stride,
                                n_rows
                            );
                            v = lhs - rhs;
                            break;
                        case OP_MUL:
                            rhs = load_gkr_operand(
                                decode_y(rw),
                                r,
                                prep,
                                main_parts,
                                pubs,
                                chals,
                                scratch,
                                scratch_stride,
                                n_rows
                            );
                            lhs *= rhs;
                            v = lhs;
                            break;
                        case OP_NEG:
                            v = -lhs;
                            break;
                        case OP_VAR:
                            v = lhs;
                            break;
                        default:
                            assert(0);
                        }

                        if (hdr.buffer_result) {
                            uint32_t slot = decode_z_index(rw);
                            scratch[slot * scratch_stride] = v;
                        }
                    }
                }

                uint32_t slot_code = pair_slots[u];
                size_t interaction = slot_code >> 1;
                size_t base = (interaction * n_rows + r) * 2;
                reinterpret_cast<FpExt *>(fracs_out)[base + (slot_code & 1)] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------- host entry point

constexpr uint32_t GLOBAL_TASK_THREADS = 1 << 16;

extern "C" int _kb_logup_gkr_input_eval(
    bool spill,
    FracExt *fracs_out,
    const Fp *prep,
    const uint64_t *main_parts,
    const Fp *pubs,
    const FpExt *chals,
    const FpExt *scratch_global,
    const Rule *rules,
    const size_t *used_nodes,
    const uint32_t *pair_slots,
    size_t n_used,
    uint32_t n_rows,
    uint32_t rows_per_thread
) {
    auto n_launch = spill ? GLOBAL_TASK_THREADS : n_rows;
    auto [grid, block] = kernel_launch_params(n_launch, 256);
    if (spill) {
        evaluate_interactions_gkr_kernel<true><<<grid, block>>>(
            fracs_out,
            prep,
            main_parts,
            pubs,
            chals,
            scratch_global,
            rules,
            used_nodes,
            pair_slots,
            n_used,
            n_rows,
            rows_per_thread
        );
    } else {
        evaluate_interactions_gkr_kernel<false><<<grid, block>>>(
            fracs_out,
            prep,
            main_parts,
            pubs,
            chals,
            scratch_global,
            rules,
            used_nodes,
            pair_slots,
            n_used,
            n_rows,
            rows_per_thread
        );
    }
    return CHECK_KERNEL();
}

} // namespace logup_gkr_input_evaluation

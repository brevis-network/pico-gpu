// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "prover/dag/codec.cuh"
#include "prover/logup_zerocheck/eval_config.cuh"
#include "prover/dag/eval_ctx.cuh"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "common/oom_heal.cuh"
#include "prover/reduce.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <vector_types.h>

namespace logup_zerocheck_mle {

struct ZerocheckCtx {
    EvalCoreCtx eval_ctx;
    FpExt *__restrict__ d_intermediates;
    uint32_t num_y;                    // rows per x slice (size of the y cube)
    const FpExt *__restrict__ d_eq_xi; // eq weights indexed by y
    const Rule *__restrict__ d_rules;
    size_t rules_len;
    const size_t *__restrict__ d_used_nodes;
    size_t used_nodes_len;
    uint32_t buffer_size; // intermediate slots per trace row
};

struct LogupCtx {
    EvalCoreCtx eval_ctx;
    FpExt *__restrict__ d_intermediates;
    uint32_t num_y;                         // rows per x slice (size of the y cube)
    const FpExt *__restrict__ d_eq_xi;      // eq weights indexed by y
    const FpExt *__restrict__ d_challenges; // values behind ENTRY_CHALLENGE operands
    const FpExt *__restrict__ d_eq_3bs;
    const Rule *__restrict__ d_rules;
    size_t rules_len;
    const size_t *__restrict__ d_used_nodes;
    const uint32_t *__restrict__ d_pair_idxs; // per used node: 2 * interaction + is_denom
    size_t used_nodes_len;
    uint32_t buffer_size; // intermediate slots per trace row
};

// Per-row operand view handed to evaluate_mle_entry (device side only).
struct EvalCtx {
    const FpExt *__restrict__ d_selectors;
    const MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *__restrict__ d_main;
    const Fp *__restrict__ d_public;
    FpExt *__restrict__ d_intermediates;
    uint32_t height; // total rows, num_x * num_y
};

__device__ __forceinline__ FpExt evaluate_mle_entry(
    const SourceInfo &opnd,
    uint32_t flat_row,
    const EvalCtx &view,
    uint32_t slot_stride,
    const FpExt *__restrict__ chal_vals = nullptr
) {
    switch (opnd.type) {
    case ENTRY_PREPROCESSED: {
#ifdef CUDA_DEBUG
        assert(view.d_preprocessed.data);
#endif
        const auto mat_stride = view.height * view.d_preprocessed.air_width;
        const FpExt *__restrict__ mat_base = view.d_preprocessed.data + mat_stride * opnd.offset;
        const FpExt *__restrict__ col_ptr = mat_base + view.height * opnd.index;
        return col_ptr[flat_row];
    }
    case ENTRY_MAIN: {
        auto main_mat = view.d_main[opnd.part];
        const auto mat_stride = view.height * main_mat.air_width;
        const FpExt *__restrict__ mat_base = main_mat.data + mat_stride * opnd.offset;
        const FpExt *__restrict__ col_ptr = mat_base + view.height * opnd.index;
        return col_ptr[flat_row];
    }
    case SRC_INTERMEDIATE:
        return view.d_intermediates[opnd.index * slot_stride];
    case SRC_IS_FIRST: {
        const FpExt *__restrict__ col_ptr = view.d_selectors;
        return FpExt(col_ptr[flat_row]);
    }
    case SRC_IS_LAST: {
        const FpExt *__restrict__ col_ptr = view.d_selectors + 2 * view.height;
        return FpExt(col_ptr[flat_row]);
    }
    case SRC_IS_TRANSITION: {
        const FpExt *__restrict__ col_ptr = view.d_selectors + view.height;
        return FpExt(col_ptr[flat_row]);
    }
    case ENTRY_CHALLENGE:
#ifdef CUDA_DEBUG
        assert(chal_vals);
#endif
        return chal_vals[opnd.index];
    case ENTRY_PUBLIC:
        return FpExt(view.d_public[opnd.index]);
    case SRC_CONSTANT:
        return FpExt(Fp(opnd.index));
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// ============================================================================
// KERNELS
// ============================================================================

__global__ void zerocheck_batch_mle_kernel(
    FpExt *__restrict__ block_partials,
    const BlockCtx *__restrict__ block_ctx_tab,
    const ZerocheckCtx *__restrict__ zc_ctx_tab,
    const FpExt *__restrict__ lam_pows,
    size_t n_lambdas
) {
    extern __shared__ char smem[];
    FpExt *red_scratch = (FpExt *)smem;

    BlockCtx bctx = block_ctx_tab[blockIdx.x];
    ZerocheckCtx actx = zc_ctx_tab[bctx.air_idx];

    uint32_t x_count = gridDim.y;
    uint32_t xi = blockIdx.y;
    uint32_t yi = threadIdx.x + bctx.local_block_idx_x * blockDim.x;
    bool const in_range = (yi < actx.num_y);

    FpExt weighted_sum(Fp::zero());

    if (in_range) {
        uint32_t rows_total = x_count * actx.num_y;
        uint32_t flat_row = xi * actx.num_y + yi;
        // one row per thread, so the slot stride is the full row count
        uint32_t slot_stride = 0;
        FpExt *__restrict__ slot_base = nullptr;
        if (actx.buffer_size > 0) {
#ifdef CUDA_DEBUG
            assert(actx.d_intermediates != nullptr);
#endif
            slot_base = actx.d_intermediates + flat_row;
            slot_stride = rows_total;
        }
        EvalCtx view{
            actx.eval_ctx.d_selectors,
            actx.eval_ctx.d_preprocessed,
            actx.eval_ctx.d_main,
            actx.eval_ctx.d_public,
            slot_base,
            rows_total
        };
        uint32_t lam_pos = 0;

        for (size_t rule_idx = 0; rule_idx < actx.rules_len; ++rule_idx) {
            Rule raw_rule = actx.d_rules[rule_idx];
            DecodedRule dec = decode_rule(raw_rule);

            FpExt lhs_val = evaluate_mle_entry(dec.x, flat_row, view, slot_stride);
            FpExt contrib;
            switch (dec.op) {
            case OP_ADD: {
                FpExt rhs_val = evaluate_mle_entry(dec.y, flat_row, view, slot_stride);
                contrib = lhs_val + rhs_val;
                break;
            }
            case OP_SUB: {
                FpExt rhs_val = evaluate_mle_entry(dec.y, flat_row, view, slot_stride);
                contrib = lhs_val - rhs_val;
                break;
            }
            case OP_MUL: {
                FpExt rhs_val = evaluate_mle_entry(dec.y, flat_row, view, slot_stride);
                contrib = lhs_val * rhs_val;
                break;
            }
            case OP_NEG:
                contrib = -lhs_val;
                break;
            case OP_VAR:
                contrib = lhs_val;
                break;
            default:
                assert(false);
            }

            if (dec.buffer_result && actx.buffer_size > 0) {
                view.d_intermediates[dec.z_index * slot_stride] = contrib;
            }

            if (dec.is_constraint) {
                while (lam_pos < n_lambdas && lam_pos < actx.used_nodes_len &&
                       actx.d_used_nodes[lam_pos] == rule_idx) {
                    FpExt lam = lam_pows[lam_pos];
                    lam_pos++;
                    weighted_sum += lam * contrib;
                }
            }
        }
        weighted_sum *= actx.d_eq_xi[yi];
    }

    FpExt block_total = reduce::block_reduce_sum(weighted_sum, red_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + xi] = block_total;
    }
}

// ============================================================================
// ZEROCHECK LOCAL+LB: host-predecoded rules + per-thread local intermediates
// ============================================================================
//
// Byte-equal, measured 1.357x standalone on the reth-171 shape. Mechanism: the dominant AIR
// (93% of work) forces 14064 INTERMEDIATE reads + 8560 writes per thread through GLOBAL
// d_intermediates[z * height] (stride 65536) ~ 24 GB/launch. A per-thread
// FpExt inter_local[32] (L1-resident) kills that traffic, flipping the kernel
// latency-bound, after which __launch_bounds__(128, 10) adds +11pt of occupancy
// (LB >= 11 spills the local array; smem-staging the rule stream is a measured
// REGRESSION here — the dominant AIR's 150 KB stream exceeds the cap and the
// reservation just slashes its occupancy — so no staging, unlike the logup PSI below).
//
// PRODUCTION INTEGRATION (mirrors the logup_pred landing in this file):
//   * PackedRule per AIR is built ON FIRST USE and cached KEYED ON THE KEYGEN-STABLE
//     d_rules POINTER (the FUSED_PREIDX rule: cache structural per-AIR DAG data by
//     pointer; never bake a per-chunk-sampled value). Lambda coefficients stay LIVE —
//     d_lambda_pows is read from the kernel arg every launch.
//   * Host gate, per AIR (any failure -> the WHOLE launch falls back to the untouched
//     reference kernel, byte-identical): every operand type in {MAIN(part==0, off<16),
//     INTERMEDIATE(idx < buffer_size <= 32), CONSTANT}; op in {ADD,SUB,MUL,NEG,VAR};
//     0 < buffer_size <= 32; z_index < buffer_size; write-before-read on intermediates;
//     used_nodes == the sorted is_constraint node indices 1:1 (so the reference's lazy
//     while-loop collapses to a running lambda counter); used_nodes_len <= lambda_len
//     (checked per launch). MAIN folding: the whole matrix-stride math becomes one
//     column word = air_width * offset + index baked on the host; constants are baked
//     to Montgomery form (kills the runtime ((u64)v << 32) % P in Fp(int)).
//   * New kernel + trailing d_packed_ptrs arg only — ZerocheckCtx layout untouched
//     (FFI-safe). Fast path additionally requires threads_per_block == 128 to match
//     __launch_bounds__(128, 10).

namespace zc_pred {

// 32 FpExt = 512 B local stack per thread (0 spill at LB(128,10) on sm_120; the reth-171
// dominant AIR has buffer_size=30).
constexpr uint32_t MAX_LOCAL_BUFFER = 32;

// 16 B pre-decoded rule (same footprint as Rule => one aligned uint4 broadcast load per
// node), zero device-side bit-field reassembly:
//   meta: [2:0] op | [3] buffer_result | [4] is_constraint | [11:8] x_type | [15:12] y_type
//   xw:   x operand word — INTERMEDIATE: slot index; MAIN: column word =
//         air_width * offset + index (matrix stride math folded on host); CONSTANT: the
//         constant already in Montgomery form.
//   yw:   y operand word (same convention; valid for binary ops ADD/SUB/MUL).
//   zw:   z_index (buffer slot for buffer_result nodes).
// alignas(16), not merely 16 bytes wide: the kernel reads this array through
// reinterpret_cast<const uint4 *>, and a uint4 load requires 16-byte alignment.
struct alignas(16) PackedRule {
    uint32_t meta;
    uint32_t xw;
    uint32_t yw;
    uint32_t zw;
};
static_assert(sizeof(PackedRule) == 16, "PackedRule must be 16 bytes");
static_assert(alignof(PackedRule) == 16, "PackedRule must be uint4-aligned");

// PK_* bit layout is PRIVATE TO THIS NAMESPACE — packed and unpacked only by this file's own
// kernels. Three namespaces declare these same names, and bit 3 does NOT mean the same thing:
//   logup_pred (batch_mle.cu)        PK_BUF @3
//   zc_pred    (batch_mle.cu)        PK_BUF @3, PK_ISC @4
//   arith_pred (zerocheck_round0.cu) PK_CON @3, PK_BUF @4   <- bit 3 is PK_CON here, not PK_BUF
// Hoisting any set into a shared header without renumbering would silently put every site on one
// layout. Consult this table before touching a bit.
constexpr uint32_t PK_OP_MASK = 7u;
constexpr uint32_t PK_BUF = 1u << 3;
constexpr uint32_t PK_ISC = 1u << 4;

// Pre-decoded operand evaluator. Dispatch ordered by the measured operand histogram of the
// dominant AIR (INTERMEDIATE 14064 >> MAIN 2080 > CONSTANT 1793). The host gate guarantees
// type is one of exactly these three.
__device__ __forceinline__ FpExt eval_packed(
    uint32_t type,
    uint32_t word,
    uint32_t row,
    uint32_t height,
    const FpExt *__restrict__ main0,
    const FpExt *inter_local
) {
    if (__builtin_expect(type == (uint32_t)SRC_INTERMEDIATE, 1)) {
        return inter_local[word];
    }
    if (type == (uint32_t)ENTRY_MAIN) {
        return main0[(size_t)height * word + row];
    }
    // SRC_CONSTANT (host-gated). word is already Montgomery-form: no runtime conversion.
    return FpExt(Fp::fromRaw(word));
}

} // namespace zc_pred

// Lean fast kernel: launched only when EVERY AIR in the batch packed successfully, so it
// carries no fallback interpreter. Byte-identical to zerocheck_batch_mle_kernel by
// construction (proven byte-exact re-encoding; the launcher routes any un-packable batch
// to the reference kernel instead).
__global__ void __launch_bounds__(128, 10) zerocheck_batch_mle_kernel_pred(
    FpExt *__restrict__ block_partials,
    const BlockCtx *__restrict__ block_ctx_tab,
    const ZerocheckCtx *__restrict__ zc_ctx_tab,
    const FpExt *__restrict__ lam_pows,
    const uint4 *const *__restrict__ d_packed_ptrs
) {
    extern __shared__ char smem[];
    FpExt *red_scratch = (FpExt *)smem;

    BlockCtx bctx = block_ctx_tab[blockIdx.x];
    ZerocheckCtx actx = zc_ctx_tab[bctx.air_idx];
    const uint4 *__restrict__ packed = d_packed_ptrs[bctx.air_idx];
    uint32_t rules_len = (uint32_t)actx.rules_len;

    uint32_t x_count = gridDim.y;
    uint32_t xi = blockIdx.y;
    uint32_t yi = threadIdx.x + bctx.local_block_idx_x * blockDim.x;
    bool const in_range = (yi < actx.num_y);

    FpExt weighted_sum(Fp::zero());

    if (in_range) {
        uint32_t rows_total = x_count * actx.num_y;
        uint32_t flat_row = xi * actx.num_y + yi;
        // Single-main-part hoist (host gate: every MAIN operand has part == 0).
        const FpExt *__restrict__ main0 = actx.eval_ctx.d_main[0].data;

        FpExt inter_local[zc_pred::MAX_LOCAL_BUFFER];
        uint32_t lam_pos = 0;

        for (uint32_t rule_idx = 0; rule_idx < rules_len; ++rule_idx) {
            uint4 pr = packed[rule_idx];
            uint32_t const meta = pr.x;

            FpExt lhs_val = zc_pred::eval_packed(
                (meta >> 8) & 0xFu, pr.y, flat_row, rows_total, main0, inter_local);
            uint32_t const op = meta & zc_pred::PK_OP_MASK;
            FpExt contrib = lhs_val; // OP_VAR pass-through default
            if (__builtin_expect(op <= (uint32_t)OP_MUL, 1)) {
                FpExt rhs_val = zc_pred::eval_packed(
                    (meta >> 12) & 0xFu, pr.z, flat_row, rows_total, main0, inter_local);
                if (op == (uint32_t)OP_ADD) {
                    contrib = lhs_val + rhs_val;
                } else if (op == (uint32_t)OP_SUB) {
                    contrib = lhs_val - rhs_val;
                } else { // OP_MUL
                    contrib = lhs_val * rhs_val;
                }
            } else if (op == (uint32_t)OP_NEG) {
                contrib = -lhs_val;
            }

            if (meta & zc_pred::PK_BUF) {
                inter_local[pr.w] = contrib;
            }
            if (meta & zc_pred::PK_ISC) {
                // Host gate verified used_nodes == sorted is_constraint nodes 1:1 and
                // used_nodes_len <= n_lambdas, so the reference's lazy while-loop is
                // exactly one lambda per is_constraint node, in order.
                FpExt lam = lam_pows[lam_pos];
                lam_pos++;
                weighted_sum += lam * contrib;
            }
        }
        weighted_sum *= actx.d_eq_xi[yi];
    }

    FpExt block_total = reduce::block_reduce_sum(weighted_sum, red_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + xi] = block_total;
    }
}

// ============================================================================
// PSI: PRED_FLAT + smem-staged packed stream + per-thread local intermediates
// ============================================================================
//
// Measured 1.344x on the reth-171 GPU-dominant AIR, byte-equal to the reference kernel. The kernel
// is latency- and occupancy-bound, so three stacked levers each cut a different memory stream:
//   * PRED_FLAT  : host one-time re-encode of the 16B `Rule` DAG into a 16B `PackedRule`
//                  (one uint4 broadcast load/node, zero decode_rule bit-twiddling, constants
//                  pre-baked to Montgomery form, frequency-ordered operand dispatch) + a
//                  monotone-used_nodes FLAT walk (collapses the lazy read-back walk into one
//                  monotone pass) when the host verifies used_nodes is monotone.
//   * SMEM_STAGE : stage the (packed) rule stream + used_nodes into dynamic shared memory once
//                  per block (block-uniform broadcast reuse across all threads).
//   * INTER_LOCAL: per-thread FpExt inter_local[MAX_LOCAL_BUFFER] for the DAG intermediates
//                  (L1-resident) instead of the strided global d_intermediates round-trip.
//
// PRODUCTION INTEGRATION (vs a single-AIR kernel): `_kb_logup_batch_eval_mle` batches MANY AIRs
// per launch (a block selects its AIR via block_ctx.air_idx), each with a DIFFERENT rules_len /
// used_nodes_len / buffer_size. So every lever is PER-AIR gated at runtime:
//   1. PRED cache is per-AIR (keyed on that AIR's d_rules pointer, guarded by rules_len, like
//      arith_pred::prep / mono_fused::prep) and a small per-launch (air_idx -> {packed, mode})
//      table `LogupPredRef*` is passed as a NEW trailing kernel arg (LogupCtx layout untouched,
//      FFI-safe). If ANY AIR in the batch fails a packing precondition, the WHOLE launch falls
//      back to the untouched reference kernel (logup_batch_mle_kernel), byte-identical — this
//      keeps the fast kernel lean (no dead mode-0 interpreter inline). In practice every reth
//      AIR packs, so the reference path is essentially never taken.
//   2. INTER_LOCAL is gated on `buffer_size <= MAX_LOCAL_BUFFER`; larger AIRs use the original
//      global d_intermediates path (compile-time array can't be sized per-AIR).
//   3. SMEM_STAGE is gated per-block on the AIR's stream fitting the per-launch shmem budget
//      (`need <= stage_region_bytes`); AIRs whose stream is too large read the packed stream
//      from global for their blocks. The launch allocates max-over-fitting-AIRs, capped at
//      STAGE_CAP_BYTES so it stays under the 48 KB default dynamic-smem limit (no opt-in) and
//      keeps enough occupancy.
// __launch_bounds__ is intentionally NOT used: measured together with the levers above, it overlaps
// them and is neutral-to-harmful once PSI is applied.

namespace logup_pred {

// 32 FpExt = 512 B local stack per thread (0 spill on sm_120 at 30 slots / 480 B, measured). The
// reth-171 dominant AIR has buffer_size=30, so it fits the local fast path.
constexpr uint32_t MAX_LOCAL_BUFFER = 32;
// Per-launch shared-memory staging budget (bytes, excluding the reduction scratch). 24 KB holds
// rules_len up to ~1500; kept well under the 48 KB default dynamic-smem cap so no
// cudaFuncSetAttribute opt-in is needed and >=2 blocks/SM stay resident. AIRs whose packed
// stream exceeds this read from global (still get PRED + INTER_LOCAL, just not staging).
constexpr size_t STAGE_CAP_BYTES = 24u * 1024u;

// 16 B pre-decoded rule (same footprint as Rule => one aligned uint4 broadcast load per node):
//   meta: [2:0] op | [3] buffer_result |
//         [11:8] x_type | [15:12] y_type | [19:16] x_offset | [23:20] y_offset |
//         [27:24] x_part | [31:28] y_part
//   xw:   x operand word (index; for SRC_CONSTANT the constant already in Montgomery form)
//   yw:   y operand word (same convention; valid for binary ops)
//   zw:   z_index (buffer store slot)
// alignas(16), not merely 16 bytes wide: the kernel reads this array through
// reinterpret_cast<const uint4 *>, and a uint4 load requires 16-byte alignment.
struct alignas(16) PackedRule {
    uint32_t meta;
    uint32_t xw;
    uint32_t yw;
    uint32_t zw;
};
static_assert(sizeof(PackedRule) == 16, "PackedRule must be 16 bytes");
static_assert(alignof(PackedRule) == 16, "PackedRule must be uint4-aligned");

// PK_* bit layout is PRIVATE TO THIS NAMESPACE — packed and unpacked only by this file's own
// kernels. Three namespaces declare these same names, and bit 3 does NOT mean the same thing:
//   logup_pred (batch_mle.cu)        PK_BUF @3
//   zc_pred    (batch_mle.cu)        PK_BUF @3, PK_ISC @4
//   arith_pred (zerocheck_round0.cu) PK_CON @3, PK_BUF @4   <- bit 3 is PK_CON here, not PK_BUF
// Hoisting any set into a shared header without renumbering would silently put every site on one
// layout. Consult this table before touching a bit.
constexpr uint32_t PK_OP_MASK = 7u;
constexpr uint32_t PK_BUF = 1u << 3;

// Per-launch, per-AIR fast-path reference (a NEW trailing kernel arg; does NOT touch LogupCtx).
// d_packed == nullptr OR mode == 0 => that AIR falls back to the reference interpreter.
struct LogupPredRef {
    const PackedRule *__restrict__ d_packed;
    uint32_t mode; // 0 = fallback, 1 = packed + FLAT walk, 2 = packed + reference walk
};

// Pre-decoded operand evaluator. Dispatch ordered by measured frequency
// (INTERMEDIATE > MAIN > CHALLENGE > CONSTANT). `inter_ptr != nullptr` selects the per-thread
// local intermediate array; nullptr keeps the strided global round-trip. The cold tail rebuilds
// a SourceInfo and delegates to the reference evaluator (identical semantics for
// PREPROCESSED / PUBLIC / IS_FIRST / IS_LAST / IS_TRANSITION; never sees SRC_INTERMEDIATE).
__device__ __forceinline__ FpExt eval_packed_mle(
    uint32_t type,
    uint32_t part,
    uint32_t offset,
    uint32_t word,
    uint32_t row,
    const EvalCtx &ctx,
    uint32_t buffer_stride,
    const FpExt *__restrict__ d_challenges,
    const FpExt *inter_ptr
) {
    if (__builtin_expect(type == (uint32_t)SRC_INTERMEDIATE, 1)) {
        return inter_ptr ? inter_ptr[word] : ctx.d_intermediates[word * buffer_stride];
    }
    if (type == (uint32_t)ENTRY_MAIN) {
        auto main_ptr = ctx.d_main[part];
        const auto stride = ctx.height * main_ptr.air_width;
        const FpExt *__restrict__ matrix = main_ptr.data + stride * offset;
        const FpExt *__restrict__ column = matrix + ctx.height * word;
        return column[row];
    }
    if (type == (uint32_t)ENTRY_CHALLENGE) {
        return d_challenges[word];
    }
    if (type == (uint32_t)SRC_CONSTANT) {
        return FpExt(Fp::fromRaw(word)); // host-baked Montgomery form: no runtime conversion
    }
    SourceInfo src;
    src.type = (EntryType)type;
    src.part = (uint8_t)part;
    src.offset = (uint8_t)offset;
    src.index = word;
    return evaluate_mle_entry(src, row, ctx, buffer_stride, d_challenges);
}

// Evaluate ONE packed rule -> node_result and perform the buffer store (identical to the
// reference per-node compute). Hot ops ADD/SUB/MUL/VAR fast, NEG cold. OP_INV excluded by the
// host precondition (would trap in the reference default: assert too).
__device__ __forceinline__ FpExt eval_rule_packed(
    uint4 pr,
    uint32_t row,
    const EvalCtx &eval_ctx,
    uint32_t buffer_stride,
    const FpExt *__restrict__ d_challenges,
    uint32_t buffer_size,
    FpExt *inter_ptr
) {
    uint32_t const meta = pr.x;
    uint32_t const op = meta & PK_OP_MASK;

    FpExt x_val = eval_packed_mle(
        (meta >> 8) & 0xFu, (meta >> 24) & 0xFu, (meta >> 16) & 0xFu, pr.y,
        row, eval_ctx, buffer_stride, d_challenges, inter_ptr);

    FpExt node_result = x_val; // OP_VAR pass-through default
    if (__builtin_expect(op <= (uint32_t)OP_MUL, 1)) {
        FpExt y_val = eval_packed_mle(
            (meta >> 12) & 0xFu, (meta >> 28) & 0xFu, (meta >> 20) & 0xFu, pr.z,
            row, eval_ctx, buffer_stride, d_challenges, inter_ptr);
        if (op == (uint32_t)OP_ADD) {
            node_result = x_val + y_val;
        } else if (op == (uint32_t)OP_SUB) {
            node_result = x_val - y_val;
        } else { // OP_MUL
            node_result = x_val * y_val;
        }
    } else if (op == (uint32_t)OP_NEG) {
        node_result = -x_val;
    }

    if ((meta & PK_BUF) && buffer_size > 0) {
        if (inter_ptr) {
            inter_ptr[pr.w] = node_result;
        } else {
            eval_ctx.d_intermediates[pr.w * buffer_stride] = node_result;
        }
    }
    return node_result;
}

} // namespace logup_pred

// PSI (lean fast kernel): launched only when EVERY AIR in the batch packs (mode 1/2), so it
// carries no mode-0 interpreter — just the packed FLAT / packed-walk paths with per-block
// smem-stage and per-thread local-intermediate gates. Byte-identical to logup_batch_mle_kernel
// by construction (proven byte-exact re-encoding; the launcher routes any un-packable batch to
// the reference kernel instead).
__global__ void logup_batch_mle_kernel_pred(
    FracExt *__restrict__ block_partials,
    const BlockCtx *__restrict__ block_ctx_tab,
    const LogupCtx *__restrict__ logup_ctx_tab,
    const logup_pred::LogupPredRef *__restrict__ d_air_refs,
    uint32_t stage_region_bytes
) {
    using logup_pred::PackedRule;

    extern __shared__ char smem[];
    FpExt *red_scratch = reinterpret_cast<FpExt *>(smem);

    BlockCtx bctx = block_ctx_tab[blockIdx.x];
    LogupCtx actx = logup_ctx_tab[bctx.air_idx];

    // Per-AIR fast-path selection (block-uniform).
    const PackedRule *packed_src =
        (d_air_refs != nullptr) ? d_air_refs[bctx.air_idx].d_packed : nullptr;
    uint32_t mode = (packed_src != nullptr) ? d_air_refs[bctx.air_idx].mode : 0u;

    // ---- SMEM_STAGE: cooperatively stage packed stream + used_nodes (block-uniform). MUST run
    // for ALL threads (active AND inactive) so every thread reaches the __syncthreads() before
    // the in_range interp guard. Gated per-block on the AIR's stream fitting the launch's
    // staging budget; too-large AIRs keep reading the packed stream from global. ----
    const size_t *used_src = actx.d_used_nodes;
    if (mode != 0) {
        const uint32_t red_fpext = (blockDim.x + 31u) / 32u;
        size_t need = actx.rules_len * sizeof(PackedRule)
                    + actx.used_nodes_len * sizeof(size_t);
        if (need <= (size_t)stage_region_bytes) {
            char *stage_base = smem + (size_t)red_fpext * sizeof(FpExt);
            uint4 *s_packed4 = reinterpret_cast<uint4 *>(stage_base);
            const uint4 *g_packed4 = reinterpret_cast<const uint4 *>(packed_src);
            for (uint32_t i = threadIdx.x; i < actx.rules_len; i += blockDim.x) {
                s_packed4[i] = g_packed4[i];
            }
            size_t used_off = actx.rules_len * sizeof(PackedRule);
            size_t *s_used = reinterpret_cast<size_t *>(stage_base + used_off);
            for (uint32_t i = threadIdx.x; i < actx.used_nodes_len; i += blockDim.x) {
                s_used[i] = actx.d_used_nodes[i];
            }
            __syncthreads();
            packed_src = reinterpret_cast<const PackedRule *>(stage_base);
            used_src = s_used;
        }
    }

    uint32_t x_count = gridDim.y;
    uint32_t xi = blockIdx.y;
    uint32_t yi = threadIdx.x + bctx.local_block_idx_x * blockDim.x;
    bool const in_range = (yi < actx.num_y);

    FpExt numer_acc = FpExt(Fp::zero());
    FpExt denom_acc = FpExt(Fp::zero());

    if (in_range) {
        uint32_t rows_total = x_count * actx.num_y;
        uint32_t flat_row = xi * actx.num_y + yi;
        uint32_t buffer_size = actx.buffer_size;

        // INTER_LOCAL gate: use the per-thread local array only when the AIR's buffer is small;
        // larger buffers keep the strided global d_intermediates round-trip.
        bool const use_local =
            (mode != 0) && (buffer_size > 0) && (buffer_size <= logup_pred::MAX_LOCAL_BUFFER);
        uint32_t slot_stride = 0;
        FpExt *slot_base = nullptr;
        if (buffer_size > 0 && !use_local) {
            assert(actx.d_intermediates != nullptr);
            slot_base = actx.d_intermediates + flat_row;
            slot_stride = rows_total;
        }

        EvalCtx view{
            actx.eval_ctx.d_selectors,
            actx.eval_ctx.d_preprocessed,
            actx.eval_ctx.d_main,
            actx.eval_ctx.d_public,
            slot_base,
            rows_total
        };

        FpExt inter_local[logup_pred::MAX_LOCAL_BUFFER];
        FpExt *inter_ptr = use_local ? inter_local : nullptr;
        const FpExt *__restrict__ d_challenges = actx.d_challenges;

        if (mode == 1) {
            // PRED_FLAT: monotone single pass — evaluate each rule once, route every used-node
            // entry landing on it (in used_nodes order). Bit-identical to the reference lazy walk
            // under the host-verified monotone precondition.
            const uint4 *__restrict__ packed = reinterpret_cast<const uint4 *>(packed_src);
            size_t use_pos = 0;
            size_t const used_len = actx.used_nodes_len;
            size_t const rules_len = actx.rules_len;
            for (size_t rule_idx = 0; rule_idx < rules_len && use_pos < used_len; ++rule_idx) {
                uint4 pr = packed[rule_idx];
                FpExt node_val = logup_pred::eval_rule_packed(
                    pr, flat_row, view, slot_stride, d_challenges, buffer_size, inter_ptr);
                // A node appearing MORE than once in used_nodes: the reference evaluates it on
                // the FIRST occurrence (contrib = node_val) and READS IT BACK on later ones —
                // and the reference read-back returns node_val only for OP_VAR (re-eval x,
                // pure) or a buffered node (reads the store), otherwise FpExt::zero(). Mirror
                // that quirk so repeated non-buffered non-VAR used-nodes stay byte-identical.
                uint32_t op = pr.x & logup_pred::PK_OP_MASK;
                bool readback_is_result =
                    (op == (uint32_t)OP_VAR) || (buffer_size > 0 && (pr.x & logup_pred::PK_BUF));
                FpExt readback = readback_is_result ? node_val : FpExt(Fp::zero());
                bool first = true;
                while (use_pos < used_len && used_src[use_pos] == rule_idx) {
                    FpExt contrib = first ? node_val : readback;
                    first = false;
                    uint32_t pair_code = actx.d_pair_idxs[use_pos];
                    contrib *= actx.d_eq_3bs[pair_code >> 1];
                    if (pair_code & 1) {
                        denom_acc += contrib;
                    } else {
                        numer_acc += contrib;
                    }
                    ++use_pos;
                }
            }
        } else {
            // mode 2: packed reads, exact reference used_nodes lazy walk (non-monotone AIRs).
            const uint4 *__restrict__ packed = reinterpret_cast<const uint4 *>(packed_src);
            size_t evaluated_upto = 0;
            for (size_t use_pos = 0; use_pos < actx.used_nodes_len; ++use_pos) {
                size_t target_node = used_src[use_pos];
                FpExt contrib(Fp::zero());

                if (target_node < evaluated_upto) {
                    uint4 pr = packed[target_node];
                    uint32_t meta = pr.x;
                    uint32_t op = meta & logup_pred::PK_OP_MASK;
                    if (op == (uint32_t)OP_VAR) {
                        contrib = logup_pred::eval_packed_mle(
                            (meta >> 8) & 0xFu, (meta >> 24) & 0xFu, (meta >> 16) & 0xFu, pr.y,
                            flat_row, view, slot_stride, d_challenges, inter_ptr);
                    } else if (buffer_size > 0 && (meta & logup_pred::PK_BUF)) {
                        contrib = inter_ptr ? inter_ptr[pr.w]
                                           : view.d_intermediates[pr.w * slot_stride];
                    }
                } else {
                    for (; evaluated_upto <= target_node; ++evaluated_upto) {
                        FpExt node_val = logup_pred::eval_rule_packed(
                            packed[evaluated_upto], flat_row, view, slot_stride,
                            d_challenges, buffer_size, inter_ptr);
                        if (evaluated_upto == target_node) {
                            contrib = node_val;
                        }
                    }
                }

                uint32_t pair_code = actx.d_pair_idxs[use_pos];
                contrib *= actx.d_eq_3bs[pair_code >> 1];
                if (pair_code & 1) {
                    denom_acc += contrib;
                } else {
                    numer_acc += contrib;
                }
            }
        }

        FpExt eq_w = actx.d_eq_xi[yi];
        numer_acc *= eq_w;
        denom_acc *= eq_w;
    }

    FpExt numer_total = reduce::block_reduce_sum(numer_acc, red_scratch);
    __syncthreads();
    FpExt denom_total = reduce::block_reduce_sum(denom_acc, red_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + xi] = {numer_total, denom_total};
    }
}

__global__ void logup_batch_mle_kernel(
    FracExt *__restrict__ block_partials,
    const BlockCtx *__restrict__ block_ctx_tab,
    const LogupCtx *__restrict__ logup_ctx_tab
) {
    extern __shared__ char smem[];
    FpExt *red_scratch = reinterpret_cast<FpExt *>(smem);

    BlockCtx bctx = block_ctx_tab[blockIdx.x];
    LogupCtx actx = logup_ctx_tab[bctx.air_idx];

    uint32_t x_count = gridDim.y;
    uint32_t xi = blockIdx.y;
    uint32_t yi = threadIdx.x + bctx.local_block_idx_x * blockDim.x;
    bool const in_range = (yi < actx.num_y);

    FpExt numer_acc = FpExt(Fp::zero());
    FpExt denom_acc = FpExt(Fp::zero());

    if (in_range) {
        uint32_t rows_total = x_count * actx.num_y;
        uint32_t flat_row = xi * actx.num_y + yi;
        // one row per thread, so the slot stride is the full row count
        uint32_t slot_stride = 0;
        FpExt *slot_base = nullptr;
        if (actx.buffer_size > 0) {
            assert(actx.d_intermediates != nullptr);
            slot_base = actx.d_intermediates + flat_row;
            slot_stride = rows_total;
        }

        // The view's d_intermediates already points at this row's first slot, so slot k lives at
        // k * slot_stride.
        EvalCtx view{
            actx.eval_ctx.d_selectors,
            actx.eval_ctx.d_preprocessed,
            actx.eval_ctx.d_main,
            actx.eval_ctx.d_public,
            slot_base,
            rows_total
        };
        size_t evaluated_upto = 0;

        // Walk used_nodes in order; entries alternate between numerator and denominator terms.
        for (size_t use_pos = 0; use_pos < actx.used_nodes_len; ++use_pos) {
            size_t target_node = actx.d_used_nodes[use_pos];
            FpExt contrib(Fp::zero());

            if (target_node < evaluated_upto) {
                // already computed: re-read it (VAR re-evaluates, buffered nodes load their slot)
                Rule raw_rule = actx.d_rules[target_node];
                DecodedRule dec = decode_rule(raw_rule);
                if (dec.op == OP_VAR) {
                    contrib = evaluate_mle_entry(
                        dec.x, flat_row, view, slot_stride, actx.d_challenges
                    );
                } else if (actx.buffer_size > 0 && dec.buffer_result) {
                    contrib = view.d_intermediates[dec.z_index * slot_stride];
                }
            } else {
                // not reached yet: evaluate forward up to and including this node
                for (; evaluated_upto <= target_node; ++evaluated_upto) {
                    Rule raw_rule = actx.d_rules[evaluated_upto];
                    DecodedRule dec = decode_rule(raw_rule);

                    FpExt lhs_val = evaluate_mle_entry(
                        dec.x, flat_row, view, slot_stride, actx.d_challenges
                    );
                    FpExt node_val;
                    switch (dec.op) {
                    case OP_ADD: {
                        FpExt rhs_val = evaluate_mle_entry(
                            dec.y, flat_row, view, slot_stride, actx.d_challenges
                        );
                        node_val = lhs_val + rhs_val;
                        break;
                    }
                    case OP_SUB: {
                        FpExt rhs_val = evaluate_mle_entry(
                            dec.y, flat_row, view, slot_stride, actx.d_challenges
                        );
                        node_val = lhs_val - rhs_val;
                        break;
                    }
                    case OP_MUL: {
                        FpExt rhs_val = evaluate_mle_entry(
                            dec.y, flat_row, view, slot_stride, actx.d_challenges
                        );
                        node_val = lhs_val * rhs_val;
                        break;
                    }
                    case OP_NEG:
                        node_val = -lhs_val;
                        break;
                    case OP_VAR:
                        node_val = lhs_val;
                        break;
                    default:
                        assert(false);
                    }

                    if (dec.buffer_result && actx.buffer_size > 0) {
                        view.d_intermediates[dec.z_index * slot_stride] = node_val;
                    }

                    // capture the value of the node this entry asked for
                    if (evaluated_upto == target_node) {
                        contrib = node_val;
                    }
                }
            }

            // d_pair_idxs maps the entry to its interaction and side:
            // pair_code = 2 * interaction + is_denom
            uint32_t pair_code = actx.d_pair_idxs[use_pos];
            contrib *= actx.d_eq_3bs[pair_code >> 1];

            if (pair_code & 1) {
                denom_acc += contrib;
            } else {
                numer_acc += contrib;
            }
        }

        FpExt eq_w = actx.d_eq_xi[yi];
        numer_acc *= eq_w;
        denom_acc *= eq_w;
    }

    FpExt numer_total = reduce::block_reduce_sum(numer_acc, red_scratch);
    __syncthreads(); // red_scratch is reused by the second reduction
    FpExt denom_total = reduce::block_reduce_sum(denom_acc, red_scratch);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x * x_count + xi] = {numer_total, denom_total};
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

// ============================================================================
// SIZE HELPERS (batch kernels)
// ============================================================================
//
// The batch reference kernels keep intermediates in global memory whenever buffer_size > 0; they
// have no register-resident small-buffer mode. The single-AIR helpers in `eval_mle.cu`
// (`_kb_zerocheck_mle_intermediates_buffer_size`, `_kb_logup_mle_intermediates_buffer_size`) report 0
// for small buffers, so the batch path needs its own sizing.

// Result is a count of FpExt elements.
extern "C" size_t _kb_zerocheck_batch_mle_intermediates_buffer_size(
    uint32_t n_slots,
    uint32_t x_count,
    uint32_t y_count
) {
    size_t rows_total = static_cast<size_t>(x_count) * y_count;
    return rows_total * n_slots;
}

// Result is a count of FpExt elements.
extern "C" size_t _kb_logup_batch_mle_intermediates_buffer_size(
    uint32_t n_slots,
    uint32_t x_count,
    uint32_t y_count
) {
    size_t rows_total = static_cast<size_t>(x_count) * y_count;
    return rows_total * n_slots;
}

namespace zc_pred {

// Host-readable POD mirror of ZerocheckCtx (same trick as logup_pred::LogupCtxPOD below:
// EvalCoreCtx's const value member deletes the default ctor, so a layout-identical POD is
// populated by cudaMemcpy of the real device bytes, never through assignment).
struct EvalCtxPOD {
    const FpExt *d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *d_main;
    const Fp *d_public;
};
struct ZerocheckCtxPOD {
    EvalCtxPOD eval_ctx;
    FpExt *d_intermediates;
    uint32_t num_y; // rows per x slice (size of the y cube)
    const FpExt *d_eq_xi;
    const Rule *d_rules;
    size_t rules_len;
    const size_t *d_used_nodes;
    size_t used_nodes_len;
    uint32_t buffer_size; // intermediate slots per trace row
};
static_assert(sizeof(EvalCtxPOD) == sizeof(EvalCoreCtx), "EvalCtxPOD layout mismatch");
static_assert(sizeof(ZerocheckCtxPOD) == sizeof(ZerocheckCtx),
              "ZerocheckCtxPOD layout mismatch");
static_assert(offsetof(ZerocheckCtxPOD, d_rules) == offsetof(ZerocheckCtx, d_rules),
              "ZerocheckCtxPOD d_rules offset mismatch");
static_assert(offsetof(ZerocheckCtxPOD, used_nodes_len) == offsetof(ZerocheckCtx, used_nodes_len),
              "ZerocheckCtxPOD used_nodes_len offset mismatch");
static_assert(offsetof(ZerocheckCtxPOD, buffer_size) == offsetof(ZerocheckCtx, buffer_size),
              "ZerocheckCtxPOD buffer_size offset mismatch");

struct PackedEntry {
    PackedRule *d_packed;
    size_t rules_len;
    uint32_t buffer_size; // intermediate slots per trace row
    bool ok;
};

// Exact host mirror of mont32_t(int): val = ((uint64_t)a << 32) % MOD.
//
// The modulus is the KoalaBear literal: this file is compiled only into `cuda_kb_all`. The two
// sibling copies below and in zerocheck_round0.cu carry the same literal.
static inline uint32_t host_to_monty(uint32_t v) {
    return (uint32_t)(((uint64_t)v << 32) % 2130706433ull);
}

// Pack one operand. Allowed: MAIN(part==0, off<16), INTERMEDIATE(idx<buffer_size), CONSTANT.
static bool pack_source(const SourceInfo &s, bool used, uint32_t buffer_size,
                        uint32_t air_width, uint32_t &type4, uint32_t &word) {
    if (!used) {
        type4 = 0;
        word = 0;
        return true;
    }
    switch (s.type) {
    case ENTRY_MAIN:
        if (s.part != 0 || (uint32_t)s.offset > 15u) return false;
        type4 = (uint32_t)ENTRY_MAIN;
        word = (uint32_t)air_width * (uint32_t)s.offset + s.index; // fold matrix stride math
        return true;
    case SRC_INTERMEDIATE:
        if (s.index >= buffer_size) return false;
        type4 = (uint32_t)SRC_INTERMEDIATE;
        word = s.index;
        return true;
    case SRC_CONSTANT:
        type4 = (uint32_t)SRC_CONSTANT;
        word = host_to_monty(s.index);
        return true;
    default:
        return false;
    }
}

// Decode the rule stream on the host (same __host__ __device__ codec), build the packed
// stream, verify every packing precondition (see the landing comment above the kernel).
// Any failure => ok = false (cached, so the gate runs once per AIR).
static PackedEntry build_packed(const ZerocheckCtxPOD &c) {
    PackedEntry e{nullptr, c.rules_len, c.buffer_size, false};
    uint32_t const bufsz = c.buffer_size;
    if (c.rules_len == 0 || bufsz == 0 || bufsz > MAX_LOCAL_BUFFER) return e;

    // MAIN air_width for the single hoisted part (structural per-AIR keygen data).
    MainMatrixPtrs<FpExt> main0{nullptr, 0};
    if (cudaMemcpy(&main0, c.eval_ctx.d_main, sizeof(main0),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;
    uint32_t const air_width = main0.air_width;

    std::vector<Rule> rules(c.rules_len);
    if (cudaMemcpy(rules.data(), c.d_rules, c.rules_len * sizeof(Rule),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;
    size_t const used_len = c.used_nodes_len;
    std::vector<size_t> used(used_len ? used_len : 1);
    if (used_len &&
        cudaMemcpy(used.data(), c.d_used_nodes, used_len * sizeof(size_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;

    std::vector<PackedRule> packed(c.rules_len);
    std::vector<bool> written(bufsz, false);
    size_t constraint_seq = 0; // running count of is_constraint nodes
    for (size_t i = 0; i < c.rules_len; i++) {
        DecodedRule r = decode_rule(rules[i]);
        // The reference switch handles ADD/SUB/MUL/NEG/VAR only (default asserts).
        if (!(r.op == OP_ADD || r.op == OP_SUB || r.op == OP_MUL ||
              r.op == OP_NEG || r.op == OP_VAR)) return e;
        bool const binary = (r.op == OP_ADD) || (r.op == OP_SUB) || (r.op == OP_MUL);
        uint32_t xt, xw, yt, yw;
        if (!pack_source(r.x, true, bufsz, air_width, xt, xw)) return e;
        if (!pack_source(r.y, binary, bufsz, air_width, yt, yw)) return e;
        // write-before-read on intermediates (an uninit read would return garbage that
        // differs between the global and local backing).
        if (r.x.type == SRC_INTERMEDIATE && !written[r.x.index]) return e;
        if (binary && r.y.type == SRC_INTERMEDIATE && !written[r.y.index]) return e;
        if (r.buffer_result) {
            if (r.z_index >= bufsz) return e;
            written[r.z_index] = true;
        }
        // used_nodes must be exactly the sorted is_constraint nodes, 1:1, so the lazy
        // while-loop is a running counter (duplicates / non-constraint entries reject).
        if (r.is_constraint) {
            if (constraint_seq >= used_len || used[constraint_seq] != i) return e;
            constraint_seq++;
        }
        packed[i].meta = ((uint32_t)r.op & PK_OP_MASK)
                       | (r.buffer_result ? PK_BUF : 0u)
                       | (r.is_constraint ? PK_ISC : 0u)
                       | (xt << 8) | (yt << 12);
        packed[i].xw = xw;
        packed[i].yw = yw;
        packed[i].zw = r.buffer_result ? r.z_index : 0u;
    }
    if (constraint_seq != used_len) return e; // every lambda must be consumed

    PackedRule *d = nullptr;
    if (mle_malloc_or_fallback(&d, c.rules_len * sizeof(PackedRule)) != cudaSuccess) return e;
    if (cudaMemcpy(d, packed.data(), c.rules_len * sizeof(PackedRule),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    e.d_packed = d;
    e.ok = true;
    return e;
}

// Cache keyed on the device rule pointer (keygen-stable, unique per AIR for a prove's
// lifetime; a changed rules_len/buffer_size forces a rebuild). Mutex-guarded for
// multi-prover-thread use. Mirrors logup_pred::prep below.
// Namespace-scope so `invalidate` (called when the owning pk frees `d_rules`) reaches them: the
// pointer key is only stable while the buffer lives — after a free the pool can hand the same
// address to a DIFFERENT air's rules, and a stale hit would evaluate the wrong constraint program.
static std::mutex &cache_mtx() {
    static std::mutex mtx;
    return mtx;
}
static std::unordered_map<const Rule *, PackedEntry> &cache_map() {
    static std::unordered_map<const Rule *, PackedEntry> cache;
    return cache;
}
static void invalidate(const void *d_rules) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(reinterpret_cast<const Rule *>(d_rules));
    if (it != cache.end()) {
        if (it->second.d_packed) cudaFree(it->second.d_packed);
        cache.erase(it);
    }
}
static PackedEntry prep(const ZerocheckCtxPOD &c) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(c.d_rules);
    if (it != cache.end() && it->second.rules_len == c.rules_len &&
        it->second.buffer_size == c.buffer_size)
        return it->second;
    if (it != cache.end() && it->second.d_packed) cudaFree(it->second.d_packed);
    PackedEntry e = build_packed(c);
    cache[c.d_rules] = e;
    return e;
}

// Runtime escape hatch, and the reason it is permanent rather than measurement scaffolding: the same
// PRED mechanism applied to the logup round-0 kernel produces a wrong proof on large workloads while
// passing every small one, and no test in this crate exercises this fast path — its admission gate
// needs `threads_per_block == 128`, which the unit fixtures never reach. So the path a future shape
// change is most likely to break silently is also the one with the least coverage; MLE_ZC_PRED=0
// reverts to the reference kernel without a rebuild. Read uncached on purpose: this is a dispatch
// site, not a kernel, so a getenv here is microseconds against a launch — and caching it would make
// the two paths unreachable from one process, which is exactly what the equivalence test needs to do.
static bool zc_pred_enabled()
{
    const char *e = getenv("MLE_ZC_PRED");
    return (e == nullptr) || (atoi(e) != 0);
}

// Fast-path launch counter, so a test can tell "both paths agree" from "the fast path never ran".
static unsigned long long g_launches = 0;

} // namespace zc_pred

// Times the zerocheck batch-MLE predecoded fast path has launched in this process. Exposed for tests:
// an equivalence check between fast and reference paths is vacuous if the fast path never fired, and
// its admission gate turns on a launch geometry the caller only controls indirectly.
extern "C" unsigned long long _kb_zc_pred_launch_count() { return zc_pred::g_launches; }

extern "C" int _kb_zerocheck_batch_eval_mle(
    FpExt *block_partials,
    FpExt *final_out,
    const BlockCtx *block_ctx_tab,
    const ZerocheckCtx *zc_ctx_tab,
    const uint32_t *air_block_starts, // size = n_airs + 1, grouped by air_idx
    const FpExt *lam_pows,
    size_t n_lambdas,
    uint32_t n_blocks,
    uint32_t x_count,
    uint32_t n_airs,
    uint32_t block_threads
) {
    if (!valid_grid_y_dim(x_count)) {
        return cudaErrorInvalidValue;
    }
    dim3 launch_grid(n_blocks, x_count);
    dim3 launch_block(block_threads);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // Fast path: build/reuse the per-AIR PackedRule caches (keyed on the keygen-stable
    // d_rules pointer). Launched only when EVERY AIR in the batch packs AND its
    // used_nodes_len fits n_lambdas this launch AND the block shape matches the
    // __launch_bounds__(128, 10); otherwise the whole launch takes the untouched
    // reference kernel (byte-identical). Lambda pows are read LIVE by both kernels.
    const uint4 **d_packed_ptrs = nullptr;
    bool all_fast = false;
    // The hatch is consulted here, before any packing work: gating only the launch would still pay
    // the per-AIR predecode and the device allocation, and would leak the uploaded pointer table on
    // the disabled path since only the fast branch frees it.
    if (n_airs > 0 && block_threads == 128 && zc_pred::zc_pred_enabled()) {
        std::vector<zc_pred::ZerocheckCtxPOD> h_ctxs(n_airs);
        if (cudaMemcpy(h_ctxs.data(), zc_ctx_tab, n_airs * sizeof(zc_pred::ZerocheckCtxPOD),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::vector<const uint4 *> h_ptrs(n_airs);
            bool any_fallback = false;
            for (uint32_t a = 0; a < n_airs; a++) {
                zc_pred::PackedEntry e = zc_pred::prep(h_ctxs[a]);
                if (e.ok && e.d_packed != nullptr &&
                    h_ctxs[a].used_nodes_len <= n_lambdas) {
                    h_ptrs[a] = reinterpret_cast<const uint4 *>(e.d_packed);
                } else {
                    any_fallback = true;
                    break;
                }
            }
            if (!any_fallback) {
                // Stream-ordered upload of the small (air_idx -> packed) table so the
                // paired cudaFreeAsync after the launch is ordered behind the kernel
                // (same pattern as logup_pred::upload_air_refs).
                const uint4 **d = nullptr;
                if (mle_malloc_async_or_fallback(&d, n_airs * sizeof(const uint4 *),
                                                 /*stream=*/0) == cudaSuccess) {
                    if (cudaMemcpyAsync(d, h_ptrs.data(), n_airs * sizeof(const uint4 *),
                                        cudaMemcpyHostToDevice, /*stream=*/0) == cudaSuccess) {
                        d_packed_ptrs = d;
                        all_fast = true;
                    } else {
                        cudaFreeAsync(d, /*stream=*/0);
                    }
                }
            }
        }
    }

    int launch_status;
    if (all_fast) {
        zc_pred::g_launches++;
        zerocheck_batch_mle_kernel_pred<<<launch_grid, launch_block, smem_bytes>>>(
            block_partials, block_ctx_tab, zc_ctx_tab, lam_pows, d_packed_ptrs
        );
        cudaFreeAsync(d_packed_ptrs, /*stream=*/0);
        launch_status = CHECK_KERNEL();
    } else {
        zerocheck_batch_mle_kernel<<<launch_grid, launch_block, smem_bytes>>>(
            block_partials, block_ctx_tab, zc_ctx_tab, lam_pows, n_lambdas
        );
        launch_status = CHECK_KERNEL();
    }
    if (launch_status != 0)
        return launch_status;

    // Second pass: one launch reduces the per-block partials of every AIR
    auto [_, fin_block] = kernel_launch_params(n_blocks / n_airs + 1);
    unsigned int fin_warps = (fin_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);

    dim3 fin_grid(n_airs, x_count);
    reduce::batched_final_reduce_block_sums<<<fin_grid, fin_block, fin_smem>>>(
        block_partials, final_out, air_block_starts, x_count
    );

    return CHECK_KERNEL();
}

namespace logup_pred {

// One-time host prep: re-encode an AIR's Rule stream into the PackedRule stream, cached per
// device rule-buffer pointer (built once at keygen, lives the whole prove). Mirrors
// arith_pred::prep / mono_fused::prep. mode: 0 = fallback, 1 = packed + FLAT, 2 = packed + walk.
struct PackedEntry {
    PackedRule *d_packed;
    size_t rules_len;
    int mode;
};

// Exact host mirror of mont32_t(int): val = ((uint64_t)a << 32) % MOD (KoalaBear). The << 32
// depends only on the low 32 bits, so this matches the device Fp(int) for all v.
static inline uint32_t host_to_monty(uint32_t v) {
    return (uint32_t)(((uint64_t)v << 32) % 2130706433ull);
}

static bool pack_source(const SourceInfo &s, bool used,
                        uint32_t &type4, uint32_t &part4, uint32_t &off4, uint32_t &word) {
    if (!used) {
        type4 = part4 = off4 = 0;
        word = 0;
        return true;
    }
    if ((uint32_t)s.type > 15u) return false;
    if ((uint32_t)s.offset > 15u) return false; // 4-bit field, defensive
    uint32_t part = s.part;
    if (s.type == ENTRY_MAIN || s.type == ENTRY_PREPROCESSED) {
        if (part > 15u) return false;
    } else {
        part = 0; // unused by the evaluator for other types
    }
    type4 = (uint32_t)s.type;
    part4 = part;
    off4 = (uint32_t)s.offset;
    word = (s.type == SRC_CONSTANT) ? host_to_monty(s.index) : s.index;
    return true;
}

// Decode the rule stream on the host (same __host__ __device__ codec), build the packed stream,
// verify all packing preconditions + the FLAT monotone precondition. Any failure => mode 0.
static PackedEntry build_packed(const Rule *d_rules, size_t rules_len,
                                const size_t *d_used, size_t used_nodes_len) {
    PackedEntry e{nullptr, rules_len, 0};
    if (rules_len == 0) return e;

    std::vector<Rule> rules(rules_len);
    if (cudaMemcpy(rules.data(), d_rules, rules_len * sizeof(Rule),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;
    std::vector<size_t> used(used_nodes_len ? used_nodes_len : 1);
    if (used_nodes_len &&
        cudaMemcpy(used.data(), d_used, used_nodes_len * sizeof(size_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;

    // FLAT precondition: used_nodes monotone non-decreasing AND every used node <= rules_len-1.
    bool monotone = true;
    for (size_t i = 0; i + 1 < used_nodes_len; i++) {
        if (used[i] > used[i + 1]) { monotone = false; break; }
    }
    if (used_nodes_len && used[used_nodes_len - 1] >= rules_len) monotone = false;

    std::vector<PackedRule> packed(rules_len);
    for (size_t i = 0; i < rules_len; i++) {
        DecodedRule r = decode_rule(rules[i]);
        // The reference switch handles ADD/SUB/MUL/NEG/VAR only (default asserts). OP_INV would
        // trap in the reference too -> reject to stay a faithful fallback.
        if (!(r.op == OP_ADD || r.op == OP_SUB || r.op == OP_MUL ||
              r.op == OP_NEG || r.op == OP_VAR)) return e;
        bool const binary = (r.op == OP_ADD) || (r.op == OP_SUB) || (r.op == OP_MUL);
        uint32_t xt, xp, xo, xw, yt, yp, yo, yw;
        if (!pack_source(r.x, true, xt, xp, xo, xw)) return e;
        if (!pack_source(r.y, binary, yt, yp, yo, yw)) return e;
        packed[i].meta = (uint32_t)r.op
                       | (r.buffer_result ? PK_BUF : 0u)
                       | (xt << 8) | (yt << 12)
                       | (xo << 16) | (yo << 20)
                       | (xp << 24) | (yp << 28);
        packed[i].xw = xw;
        packed[i].yw = yw;
        packed[i].zw = r.z_index;
    }

    PackedRule *d = nullptr;
    if (mle_malloc_or_fallback(&d, rules_len * sizeof(PackedRule)) != cudaSuccess) return e;
    if (cudaMemcpy(d, packed.data(), rules_len * sizeof(PackedRule),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    e.d_packed = d;
    e.mode = monotone ? 1 : 2;
    return e;
}

// Cache keyed on the device rule pointer (unique per AIR for a prove's lifetime; a changed
// rules_len forces a rebuild). Mutex-guarded for multi-prover-thread use.
// See zc_pred: the pointer key is only stable while the pk's `d_rules` buffer lives, so a free
// must purge the entry before the pool can recycle the address for a different air.
static std::mutex &cache_mtx() {
    static std::mutex mtx;
    return mtx;
}
static std::unordered_map<const Rule *, PackedEntry> &cache_map() {
    static std::unordered_map<const Rule *, PackedEntry> cache;
    return cache;
}
static void invalidate(const void *d_rules) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(reinterpret_cast<const Rule *>(d_rules));
    if (it != cache.end()) {
        if (it->second.d_packed) cudaFree(it->second.d_packed);
        cache.erase(it);
    }
}
static PackedEntry prep(const Rule *d_rules, size_t rules_len,
                        const size_t *d_used, size_t used_nodes_len) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(d_rules);
    if (it != cache.end() && it->second.rules_len == rules_len) return it->second;
    if (it != cache.end() && it->second.d_packed) cudaFree(it->second.d_packed);
    PackedEntry e = build_packed(d_rules, rules_len, d_used, used_nodes_len);
    cache[d_rules] = e;
    return e;
}

// Host-readable POD mirror of LogupCtx. EvalCoreCtx's `d_preprocessed` is a const value member,
// which deletes LogupCtx's copy-assignment/default-ctor -> it can't be a std::vector element or
// cudaMemcpy destination directly. This mirror is layout-identical (asserted below) and is only
// ever populated by cudaMemcpy of the real device bytes, never through assignment.
struct EvalCtxPOD {
    const FpExt *d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *d_main;
    const Fp *d_public;
};
struct LogupCtxPOD {
    EvalCtxPOD eval_ctx;
    FpExt *d_intermediates;
    uint32_t num_y; // rows per x slice (size of the y cube)
    const FpExt *d_eq_xi;
    const FpExt *d_challenges;
    const FpExt *d_eq_3bs;
    const Rule *d_rules;
    size_t rules_len;
    const size_t *d_used_nodes;
    const uint32_t *d_pair_idxs;
    size_t used_nodes_len;
    uint32_t buffer_size; // intermediate slots per trace row
};
static_assert(sizeof(EvalCtxPOD) == sizeof(EvalCoreCtx), "EvalCtxPOD layout mismatch");
static_assert(sizeof(LogupCtxPOD) == sizeof(LogupCtx), "LogupCtxPOD layout mismatch");
static_assert(offsetof(LogupCtxPOD, d_rules) == offsetof(LogupCtx, d_rules),
              "LogupCtxPOD d_rules offset mismatch");
static_assert(offsetof(LogupCtxPOD, d_used_nodes) == offsetof(LogupCtx, d_used_nodes),
              "LogupCtxPOD d_used_nodes offset mismatch");
static_assert(offsetof(LogupCtxPOD, used_nodes_len) == offsetof(LogupCtx, used_nodes_len),
              "LogupCtxPOD used_nodes_len offset mismatch");
static_assert(offsetof(LogupCtxPOD, buffer_size) == offsetof(LogupCtx, buffer_size),
              "LogupCtxPOD buffer_size offset mismatch");

// Upload the small per-launch (air_idx -> ref) table using the stream-ordered allocator so the
// paired cudaFreeAsync (issued right after the launch) is ordered after the kernel reading it,
// even under concurrent prover threads. nullptr on any failure -> every AIR falls back this launch.
static LogupPredRef *upload_air_refs(const std::vector<LogupPredRef> &h_refs, cudaStream_t stream) {
    if (h_refs.empty()) return nullptr;
    LogupPredRef *d_refs = nullptr;
    if (mle_malloc_async_or_fallback(&d_refs, h_refs.size() * sizeof(LogupPredRef), stream) !=
        cudaSuccess)
        return nullptr;
    if (cudaMemcpyAsync(d_refs, h_refs.data(), h_refs.size() * sizeof(LogupPredRef),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFreeAsync(d_refs, stream);
        return nullptr;
    }
    return d_refs;
}

} // namespace logup_pred

extern "C" int _kb_logup_batch_eval_mle(
    FracExt *block_partials,
    FracExt *final_out,
    const BlockCtx *block_ctx_tab,
    const LogupCtx *logup_ctx_tab,
    const uint32_t *air_block_starts, // size = n_airs + 1, grouped by air_idx
    uint32_t n_blocks,
    uint32_t x_count,
    uint32_t n_airs,
    uint32_t block_threads
) {
    if (!valid_frac_grid_y_dim(x_count)) {
        return cudaErrorInvalidValue;
    }
    dim3 launch_grid(n_blocks, x_count);
    dim3 launch_block(block_threads);
    size_t scratch_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // PSI: build/reuse the per-AIR PackedRule caches + a small per-launch (air_idx -> ref) table.
    // logup_ctx_tab lives on the device (uploaded fresh by Rust each call), so D2H a POD mirror to
    // read each AIR's d_rules/rules_len/d_used_nodes/used_nodes_len (the cache key + build input).
    // Size the shared-memory staging region to the largest packed stream that fits STAGE_CAP_BYTES.
    logup_pred::LogupPredRef *d_air_refs = nullptr;
    uint32_t stage_region_bytes = 0;
    // The fast (pred) kernel is launched only when EVERY AIR in the batch packs successfully
    // (mode != 0); a single fallback AIR sends the whole launch to the untouched reference kernel
    // (byte-identical, safe). This keeps the hot kernel lean — no dead mode-0 interpreter inline —
    // and in practice every reth AIR packs, so the reference path is essentially never taken.
    bool all_fast = false;
    if (n_airs > 0) {
        std::vector<logup_pred::LogupCtxPOD> h_ctxs(n_airs);
        if (cudaMemcpy(h_ctxs.data(), logup_ctx_tab, n_airs * sizeof(logup_pred::LogupCtxPOD),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::vector<logup_pred::LogupPredRef> h_refs(n_airs);
            size_t max_need = 0;
            bool any_fallback = false;
            for (uint32_t a = 0; a < n_airs; a++) {
                const logup_pred::LogupCtxPOD &c = h_ctxs[a];
                logup_pred::PackedEntry e =
                    logup_pred::prep(c.d_rules, c.rules_len, c.d_used_nodes, c.used_nodes_len);
                if (e.mode != 0 && e.d_packed != nullptr) {
                    h_refs[a].d_packed = e.d_packed;
                    h_refs[a].mode = (uint32_t)e.mode;
                    size_t need = c.rules_len * sizeof(logup_pred::PackedRule)
                                + c.used_nodes_len * sizeof(size_t);
                    if (need <= logup_pred::STAGE_CAP_BYTES)
                        max_need = std::max(max_need, need);
                } else {
                    h_refs[a].d_packed = nullptr;
                    h_refs[a].mode = 0;
                    any_fallback = true;
                }
            }
            if (!any_fallback) {
                d_air_refs = logup_pred::upload_air_refs(h_refs, /*stream=*/0);
                if (d_air_refs != nullptr) {
                    all_fast = true;
                    stage_region_bytes = (uint32_t)max_need;
                }
            }
        }
    }

    int launch_status;
    if (all_fast) {
        size_t smem_bytes = scratch_bytes + stage_region_bytes;
        logup_batch_mle_kernel_pred<<<launch_grid, launch_block, smem_bytes>>>(
            block_partials, block_ctx_tab, logup_ctx_tab, d_air_refs, stage_region_bytes);
        cudaFreeAsync(d_air_refs, /*stream=*/0);
        launch_status = CHECK_KERNEL();
    } else {
        // Fallback: original reference kernel for the whole batch (any AIR failed to pack, or the
        // ref-table build failed). Byte-identical to the pre-PSI production path.
        if (d_air_refs != nullptr)
            cudaFreeAsync(d_air_refs, /*stream=*/0);
        logup_batch_mle_kernel<<<launch_grid, launch_block, scratch_bytes>>>(
            block_partials, block_ctx_tab, logup_ctx_tab);
        launch_status = CHECK_KERNEL();
    }
    if (launch_status != 0)
        return launch_status;

    // Second pass: one launch reduces the per-block partials of every AIR
    // each FracExt is two FpExt, hence 2 * x_count columns
    auto [_, fin_block] = kernel_launch_params(n_blocks / n_airs + 1);
    unsigned int fin_warps = (fin_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);

    dim3 fin_grid(n_airs, 2 * x_count);
    reduce::batched_final_reduce_block_sums<<<fin_grid, fin_block, fin_smem>>>(
        reinterpret_cast<FpExt *>(block_partials),
        reinterpret_cast<FpExt *>(final_out),
        air_block_starts,
        2 * x_count
    );

    return CHECK_KERNEL();
}

// Purge a freed `d_rules` pointer from both PSI pred caches. Called by the Rust side when the
// owning pk drops the buffer — the pool can then recycle the address for a different air without
// the caches serving that air the previous program.
extern "C" void _kb_batch_mle_pred_invalidate(const void *d_rules) {
    zc_pred::invalidate(d_rules);
    logup_pred::invalidate(d_rules);
}

} // namespace logup_zerocheck_mle

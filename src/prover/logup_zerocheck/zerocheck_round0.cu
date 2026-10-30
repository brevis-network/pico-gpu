// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vector_types.h>

#include "prover/dag/codec.cuh"
#include "prover/dag/dag_entry.cuh"
#include "prover/logup_zerocheck/eval_config.cuh"
#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include "common/oom_heal.cuh"
#include "prover/reduce.cuh"
#include "common/utils.cuh"

using namespace symbolic_dag;

namespace zerocheck_round0 {

constexpr uint32_t BUFFER_THRESHOLD = 16;
// Cutoff on x_count * dom_size that picks between the two kernel shapes.
// Below it, each block owns a single coset (grid.y = n_cosets) so small inputs still fill the GPU.
// At or above it, one thread covers every coset so each inverse NTT is shared across cosets.
constexpr uint32_t COSET_PARALLEL_THRESHOLD = 32768; // 1 << 15

// Single pass over the constraint DAG that evaluates every node on all NUM_COSETS cosets at once
// and returns, per coset, the lambda-weighted sum of the constraint nodes.
// Plain __inline__ rather than forced, so nvcc can weigh inlining against register pressure.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM>
__device__ __inline__ void acc_constraints(
    FpExt *__restrict__ cons_sums, // out: one sum per coset
    const NttEvalContext<NUM_COSETS> &ectx,
    const Fp *is_first, // [NUM_COSETS] first-row selector at this xi
    const Fp *is_last,  // [NUM_COSETS] last-row selector at this xi
    uint32_t xi,
    const FpExt *__restrict__ lam_pows,
    const Rule *__restrict__ rule_buf,
    size_t n_rules,
    const size_t *__restrict__ used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas
) {
    size_t lam_pos = 0;

    // every coset starts from zero
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        cons_sums[c] = FpExt(Fp::zero());
    }

    for (size_t rule_idx = 0; rule_idx < n_rules; ++rule_idx) {
        Rule raw_rule = rule_buf[rule_idx];
        // only the header (op, flags, x operand) is decoded eagerly
        RuleHeader hdr = decode_rule_header(raw_rule);

        // x operand on each coset
        Fp lhs_vals[NUM_COSETS];
        ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(lhs_vals, hdr.x, ectx, is_first, is_last, xi);

        switch (hdr.op) {
        case OP_ADD: {
            // y exists only for the two-operand ops
            SourceInfo rhs_src = decode_y(raw_rule);
            Fp rhs_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(rhs_vals, rhs_src, ectx, is_first, is_last, xi);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] += rhs_vals[c];
            }
            break;
        }
        case OP_SUB: {
            SourceInfo rhs_src = decode_y(raw_rule);
            Fp rhs_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(rhs_vals, rhs_src, ectx, is_first, is_last, xi);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] -= rhs_vals[c];
            }
            break;
        }
        case OP_MUL: {
            SourceInfo rhs_src = decode_y(raw_rule);
            Fp rhs_vals[NUM_COSETS];
            ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(rhs_vals, rhs_src, ectx, is_first, is_last, xi);
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] *= rhs_vals[c];
            }
            break;
        }
        case OP_NEG:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] = -lhs_vals[c];
            }
            break;
        case OP_VAR:
            break;
        case OP_INV:
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] = inv(lhs_vals[c]);
            }
            break;
        }

        // Keep the node value for later reads (one entry per coset);
        // the slot index is decoded on this path only
        if (hdr.buffer_result && ectx.buffer_size > 0) {
            uint32_t slot_idx = decode_z_index(raw_rule);
#ifdef CUDA_DEBUG
            assert(slot_idx < ectx.buffer_size);
#endif
            // per-thread slot layout is [n_slots][NUM_COSETS]
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                ectx.inter_buffer[slot_idx * ectx.buffer_stride + c] = lhs_vals[c];
            }
        }

        // constraint nodes pull their lambda(s) and add into the per-coset sums
        if (hdr.is_constraint) {
            while (lam_pos < n_lambdas && lam_pos < n_used_nodes &&
                   used_node_ids[lam_pos] == rule_idx) {
                FpExt lam = lam_pows[lam_pos];
                lam_pos++;
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    cons_sums[c] += lam * lhs_vals[c];
                }
            }
        }
    }
}

// ============================================================================
// PRED — host-predecoded rule stream (shape-independent, ~-10% on the largest
// size-gated interpreter AIRs, e.g. reth <2,1,0> n_rules=38132).
//
// One-time host re-encode of the 16B Rule stream into a 16B PackedRule: op/flags/
// types/parts/offsets pre-unpacked into `meta`, operand indices in xw/yw, constants
// pre-baked to Montgomery form (kills the runtime ((u64)v<<32)%P per constant eval),
// slot_idx in zw. Device interpreter does ONE uint4 broadcast load per rule, zero
// bit-field reassembly, frequency-ordered operand dispatch (INTERMEDIATE first), and
// — when the lambda-flattening precondition holds — drops the used_node_ids walk.
//
// Byte-identical to acc_constraints (verified). The host prep
// verifies every packing precondition and falls back to the reference kernel (mode 0)
// otherwise. Cached per device rule-buffer pointer; built lazily on first launch (the
// device pkey's rule buffers are allocated once at keygen and live the whole prove).
// ============================================================================
namespace arith_pred {

struct PackedRule {
    uint32_t meta;
    uint32_t xw;
    uint32_t yw;
    uint32_t zw;
};
static_assert(sizeof(PackedRule) == 16, "PackedRule must be 16 bytes");

// PK_* bit layout is PRIVATE TO THIS NAMESPACE — packed and unpacked only by this file's own
// kernels. Three namespaces declare these same names, and bit 3 does NOT mean the same thing:
//   logup_pred (batch_mle.cu)        PK_BUF @3
//   zc_pred    (batch_mle.cu)        PK_BUF @3, PK_ISC @4
//   arith_pred (zerocheck_round0.cu) PK_CON @3, PK_BUF @4   <- bit 3 is PK_CON here, not PK_BUF
// Hoisting any set into a shared header without renumbering would silently put every site on one
// layout. Consult this table before touching a bit.
constexpr uint32_t PK_OP_MASK = 7u;
constexpr uint32_t PK_CON = 1u << 3;
constexpr uint32_t PK_BUF = 1u << 4;

// Device pre-decoded operand evaluator. INTERMEDIATE (dominant) and CONSTANT are
// fast-pathed; every other entry type rebuilds a SourceInfo and delegates to the
// reference ntt_eval_dag_entry (identical semantics).
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM>
__device__ __forceinline__ void eval_packed(
    Fp *__restrict__ results,
    uint32_t type,
    uint32_t part,
    uint32_t offset,
    uint32_t word,
    const NttEvalContext<NUM_COSETS> &ctx,
    const Fp *is_first,
    const Fp *is_last,
    uint32_t xi
) {
    if (__builtin_expect(type == (uint32_t)SRC_INTERMEDIATE, 1)) {
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = ctx.inter_buffer[word * ctx.buffer_stride + c];
        }
        return;
    }
    if (type == (uint32_t)SRC_CONSTANT) {
        Fp val = Fp::fromRaw(word); // host-baked Montgomery form
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            results[c] = val;
        }
        return;
    }
    // Cold tail (MAIN / PREPROCESSED / PUBLIC / IS_FIRST / IS_LAST / IS_TRANSITION):
    // rebuild a SourceInfo and delegate to the reference evaluator.
    SourceInfo src;
    src.type = (EntryType)type;
    src.part = (uint8_t)part;
    src.offset = (uint8_t)offset;
    src.index = word;
    ntt_eval_dag_entry<NUM_COSETS, NEEDS_SHMEM>(results, src, ctx, is_first, is_last, xi);
}

// Interpreter over the pre-decoded stream. LAMBDA_FLAT=true is used only after the host
// prep verified (by simulating the reference walk) that every constraint node consumes
// EXACTLY one lambda in rule order and n_lambdas == n_used_nodes == #constraint nodes;
// under that precondition `if (is_constraint) consume-next` is bit-identical to the
// reference while-walk and the used_node_ids loads + 64-bit compares disappear.
template <uint32_t NUM_COSETS, bool NEEDS_SHMEM, bool LAMBDA_FLAT>
__device__ __inline__ void acc_constraints_pred(
    FpExt *__restrict__ cons_sums,
    const NttEvalContext<NUM_COSETS> &ectx,
    const Fp *is_first,
    const Fp *is_last,
    uint32_t xi,
    const FpExt *__restrict__ lam_pows,
    const PackedRule *__restrict__ d_packed,
    size_t n_rules,
    const size_t *__restrict__ used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas
) {
    size_t lam_pos = 0;
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        cons_sums[c] = FpExt(Fp::zero());
    }

#pragma unroll 2
    for (size_t rule_idx = 0; rule_idx < n_rules; ++rule_idx) {
        // One aligned 16B load, uniform across the warp (broadcast).
        uint4 const pr = *(reinterpret_cast<const uint4 *>(d_packed) + rule_idx);
        uint32_t const meta = pr.x;
        uint32_t const op = meta & PK_OP_MASK;

        Fp lhs_vals[NUM_COSETS];
        eval_packed<NUM_COSETS, NEEDS_SHMEM>(
            lhs_vals, (meta >> 8) & 0xFu, (meta >> 24) & 0xFu, (meta >> 16) & 0xFu, pr.y,
            ectx, is_first, is_last, xi);

        if (__builtin_expect(op <= (uint32_t)OP_MUL, 1)) {
            Fp rhs_vals[NUM_COSETS];
            eval_packed<NUM_COSETS, NEEDS_SHMEM>(
                rhs_vals, (meta >> 12) & 0xFu, (meta >> 28) & 0xFu, (meta >> 20) & 0xFu, pr.z,
                ectx, is_first, is_last, xi);
            if (op == (uint32_t)OP_ADD) {
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    lhs_vals[c] += rhs_vals[c];
                }
            } else if (op == (uint32_t)OP_SUB) {
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    lhs_vals[c] -= rhs_vals[c];
                }
            } else { // OP_MUL
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    lhs_vals[c] *= rhs_vals[c];
                }
            }
        } else if (op == (uint32_t)OP_NEG) {
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] = -lhs_vals[c];
            }
        } else if (__builtin_expect(op == (uint32_t)OP_INV, 0)) {
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                lhs_vals[c] = inv(lhs_vals[c]);
            }
        }
        // OP_VAR: pass through.

        if ((meta & PK_BUF) && ectx.buffer_size > 0) {
            uint32_t const slot_idx = pr.w;
#pragma unroll
            for (uint32_t c = 0; c < NUM_COSETS; c++) {
                ectx.inter_buffer[slot_idx * ectx.buffer_stride + c] = lhs_vals[c];
            }
        }

        if (meta & PK_CON) {
            if constexpr (LAMBDA_FLAT) {
                FpExt lam = lam_pows[lam_pos];
                lam_pos++;
#pragma unroll
                for (uint32_t c = 0; c < NUM_COSETS; c++) {
                    cons_sums[c] += lam * lhs_vals[c];
                }
            } else {
                while (lam_pos < n_lambdas && lam_pos < n_used_nodes &&
                       used_node_ids[lam_pos] == rule_idx) {
                    FpExt lam = lam_pows[lam_pos];
                    lam_pos++;
#pragma unroll
                    for (uint32_t c = 0; c < NUM_COSETS; c++) {
                        cons_sums[c] += lam * lhs_vals[c];
                    }
                }
            }
        }
    }
}

// ---- Host: one-time re-encode + per-rule-buffer cache -----------------------
// mode: 0 = fallback (reference kernel), 1 = packed + flat lambda, 2 = packed + reference walk
struct PackedEntry {
    PackedRule *d_packed;
    size_t rules_len;
    int mode;
};

// Exact host-side mirror of mont32_t(int): val = ((uint64_t)a << 32) % MOD (KoalaBear).
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
    if ((uint32_t)s.offset > 15u) return false; // 4-bit by codec, defensive
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

// Decode the rule stream on the host (same __host__ __device__ codec), build the packed
// stream, and verify the lambda-flattening precondition by simulating the reference walk.
static PackedEntry build_packed(const Rule *rule_buf, size_t n_rules,
                                const size_t *d_used, size_t n_used_nodes, size_t n_lambdas) {
    PackedEntry e{nullptr, n_rules, 0};
    if (n_rules == 0) return e;
    std::vector<Rule> rules(n_rules);
    if (cudaMemcpy(rules.data(), rule_buf, n_rules * sizeof(Rule),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;
    std::vector<size_t> used(n_used_nodes ? n_used_nodes : 1);
    if (n_used_nodes &&
        cudaMemcpy(used.data(), d_used, n_used_nodes * sizeof(size_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;

    std::vector<PackedRule> packed(n_rules);
    size_t li = 0;
    bool flat_ok = (n_used_nodes == n_lambdas);
    for (size_t i = 0; i < n_rules; i++) {
        DecodedRule r = decode_rule(rules[i]);
        if ((uint32_t)r.op > (uint32_t)OP_INV) return e;
        bool const binary = (r.op == OP_ADD) || (r.op == OP_SUB) || (r.op == OP_MUL);
        uint32_t xt, xp, xo, xw, yt, yp, yo, yw;
        if (!pack_source(r.x, true, xt, xp, xo, xw)) return e;
        if (!pack_source(r.y, binary, yt, yp, yo, yw)) return e;
        packed[i].meta = (uint32_t)r.op
                       | (r.is_constraint ? PK_CON : 0u)
                       | (r.buffer_result ? PK_BUF : 0u)
                       | (xt << 8) | (yt << 12)
                       | (xo << 16) | (yo << 20)
                       | (xp << 24) | (yp << 28);
        packed[i].xw = xw;
        packed[i].yw = yw;
        packed[i].zw = r.z_index;

        // Simulate the reference lambda walk for this node.
        unsigned consumed = 0;
        if (r.is_constraint) {
            while (li < n_lambdas && li < n_used_nodes && used[li] == i) {
                li++;
                consumed++;
            }
        }
        if (consumed != (r.is_constraint ? 1u : 0u)) flat_ok = false;
    }
    if (li != n_lambdas || li != n_used_nodes) flat_ok = false;

    PackedRule *d = nullptr;
    if (mle_malloc_or_fallback(&d, n_rules * sizeof(PackedRule)) != cudaSuccess) return e;
    if (cudaMemcpy(d, packed.data(), n_rules * sizeof(PackedRule),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return e;
    }
    e.d_packed = d;
    e.mode = flat_ok ? 1 : 2;
    return e;
}

// Cache keyed on the device rule pointer. The key is only stable while the pk's buffer lives:
// once freed, the pool can hand the address to a DIFFERENT air's rules, so the owning pk's drop
// purges the entry via `invalidate` below. Mutex-guarded for multi-prover-thread use.
static std::mutex &cache_mtx() {
    static std::mutex mtx;
    return mtx;
}
static std::unordered_map<const Rule *, PackedEntry> &cache_map() {
    static std::unordered_map<const Rule *, PackedEntry> cache;
    return cache;
}
static void invalidate(const void *rule_buf) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(reinterpret_cast<const Rule *>(rule_buf));
    if (it != cache.end()) {
        if (it->second.d_packed) cudaFree(it->second.d_packed);
        cache.erase(it);
    }
}
static PackedEntry prep(const Rule *rule_buf, size_t n_rules,
                        const size_t *d_used, size_t n_used_nodes, size_t n_lambdas) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(rule_buf);
    if (it != cache.end() && it->second.rules_len == n_rules) return it->second;
    if (it != cache.end() && it->second.d_packed) cudaFree(it->second.d_packed);
    PackedEntry e = build_packed(rule_buf, n_rules, d_used, n_used_nodes, n_lambdas);
    cache[rule_buf] = e;
    return e;
}

} // namespace arith_pred

// Purge a freed `rule_buf` pointer from the round-0 pred cache (see `arith_pred::invalidate`).
extern "C" void _kb_round0_pred_invalidate(const void *rule_buf) { arith_pred::invalidate(rule_buf); }

// Lockstep kernel: each thread evaluates all NUM_COSETS cosets together.
// Grid is (num_x_blocks, 1); cosets are not a grid dimension here.
// Block is dom_size * x_per_block threads, with
// dom_pos = threadIdx.x % dom_size as the fast index.
//
// Why lockstep: a trace read costs one inverse NTT, and the coefficient it yields is
// reused for every coset's shift and forward NTT.
template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM, bool USE_PACKED = false,
          bool LAMBDA_FLAT = false>
__global__ void zerocheck_ntt_evaluate_constraints_kernel(
    FpExt *__restrict__ block_partials, // [n_partials][NUM_COSETS][dom_size]
    const Fp *__restrict__ sel_cube,    // [3][x_count]
    const Fp *__restrict__ prep_trace,
    const Fp *const *__restrict__ main_traces,
    const FpExt *__restrict__ eq_x, // [x_count]
    const FpExt *__restrict__ lam_pows,
    const Fp *__restrict__ pub_vals,
    const Rule *__restrict__ rule_buf,
    size_t n_rules,
    const size_t *__restrict__ used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas,
    uint32_t n_slots,
    Fp *__restrict__ inter_global,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t trace_h,
    Fp coset_gen,
    // PRED: pre-decoded rule stream, non-null only when USE_PACKED (else reference walk).
    const arith_pred::PackedRule *__restrict__ d_packed = nullptr
) {
    extern __shared__ char smem[];
    // Dynamic smem: blockDim.x FpExt of reduction scratch (reused for each coset), followed,
    // when NEEDS_SHMEM, by blockDim.x Fp of NTT scratch cut into one dom_size slice per xi
    // group.
    FpExt *red_buf = reinterpret_cast<FpExt *>(smem);
    Fp *ntt_scratch_all =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FpExt)) : nullptr;

    uint32_t const log_dom = __ffs(dom_size) - 1;
    bool const trivial_dom = (log_dom == 0);

    // one dom_size-wide NTT scratch slice per xi group of the block
    uint32_t const xi_local = threadIdx.x >> log_dom;
    Fp *ntt_scratch = NEEDS_SHMEM ? (ntt_scratch_all + xi_local * dom_size) : nullptr;

    // dom_pos is the fast-moving index, xi the slow one
    uint32_t const gtid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const dom_pos = gtid & (dom_size - 1);
    uint32_t const xi_start = gtid >> log_dom;

    // per-thread constants shared by every coset
    uint32_t const dom_pos_rev = trivial_dom ? 0 : rev_len(dom_pos, log_dom);

    // scale factors for the first/last-row selectors, one per coset
    uint32_t const log_h = __ffs(trace_h) - 1;
    uint32_t const log_seg = min(log_dom, log_h);
    uint32_t const seg_len = 1u << log_seg;
    uint32_t const log_step = log_dom - log_seg;

    Fp first_scale[NUM_COSETS];
    Fp last_scale[NUM_COSETS];
    Fp const eta_step = trivial_dom ? Fp::one() : TWO_ADIC_GENERATORS[log_dom - log_step];
    Fp const dom_root =
        trivial_dom ? Fp::one() : device_ntt::get_twiddle(log_dom, dom_pos);

    Fp coset_pow = coset_gen; // running power g, g^2, g^3, ...
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        Fp eval_pt = coset_pow * dom_root;
        Fp seg_root = exp_power_of_2(eval_pt, log_step);
        first_scale[c] = avg_gp(seg_root, seg_len);
        last_scale[c] = avg_gp(seg_root * eta_step, seg_len);
        coset_pow *= coset_gen; // power for the next coset
    }

    // Node-value slots: registers when they fit, otherwise the global spill buffer
    Fp reg_slots[GLOBAL ? 1 : BUFFER_THRESHOLD * NUM_COSETS];
    Fp *slot_base;
    uint32_t slot_stride;
    if constexpr (GLOBAL) {
        // spill layout [n_slots][num_threads][NUM_COSETS]
        slot_base = inter_global + gtid * NUM_COSETS;
        slot_stride = gridDim.x * blockDim.x * NUM_COSETS;
    } else {
        slot_base = reg_slots;
        slot_stride = NUM_COSETS;
    }

    // running per-coset totals
    FpExt coset_acc[NUM_COSETS] = {};

    // the launch config keeps blockDim.x a multiple of dom_size
    uint32_t const xi_step = (gridDim.x * blockDim.x) >> log_dom;

    // fields that stay fixed for the whole xi loop
    NttEvalContext<NUM_COSETS> ectx{
        prep_trace,
        main_traces,
        pub_vals,
        slot_base,
        ntt_scratch,
        {}, // omega_shifts, filled just below
        dom_size,
        trace_h,
        slot_stride,
        n_slots,
        dom_pos,
    };
    // omega_shifts[c] = shift_base^(c + 1), by repeated multiplication
    Fp const shift_base = trivial_dom ? Fp::one() : pow(coset_gen, dom_pos_rev);
    Fp shift_pow = shift_base;
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        ectx.omega_shifts[c] = shift_pow;
        shift_pow *= shift_base;
    }

    // grid-stride over xi, so the slot buffer stays per-thread instead of per-row
    for (uint32_t xi = xi_start; xi < x_count; xi += xi_step) {
        // row selectors at this xi, scaled for each coset
        Fp is_first[NUM_COSETS];
        Fp is_last[NUM_COSETS];
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            is_first[c] = first_scale[c] * sel_cube[xi];
            is_last[c] = last_scale[c] * sel_cube[2 * x_count + xi];
        }

        FpExt cons_sums[NUM_COSETS];
        if constexpr (USE_PACKED) {
            arith_pred::acc_constraints_pred<NUM_COSETS, NEEDS_SHMEM, LAMBDA_FLAT>(
                cons_sums,
                ectx,
                is_first,
                is_last,
                xi,
                lam_pows,
                d_packed,
                n_rules,
                used_node_ids,
                n_used_nodes,
                n_lambdas
            );
        } else {
            acc_constraints<NUM_COSETS, NEEDS_SHMEM>(
                cons_sums,
                ectx,
                is_first,
                is_last,
                xi,
                lam_pows,
                rule_buf,
                n_rules,
                used_node_ids,
                n_used_nodes,
                n_lambdas
            );
        }

        FpExt eq_w = eq_x[xi];
#pragma unroll
        for (uint32_t c = 0; c < NUM_COSETS; c++) {
            coset_acc[c] += cons_sums[c] * eq_w;
        }
    }

    // Reduce one coset at a time so smem needs only one FpExt per thread.
    // eval_pt and the vanishing value are rebuilt here rather than held live across the loop.
    coset_pow = coset_gen; // restart from g
#pragma unroll
    for (uint32_t c = 0; c < NUM_COSETS; c++) {
        Fp eval_pt = coset_pow * dom_root;
        Fp vanishing = exp_power_of_2(eval_pt, log_dom) - Fp::one();

        // Assumes no coset power coset_gen^(c+1) lies in the skip domain, so eval_pt is outside it
        // as well and the vanishing value is invertible.
        red_buf[threadIdx.x] = coset_acc[c] * inv(vanishing);
        __syncthreads();

        if (threadIdx.x < dom_size) {
            FpExt col_sum = red_buf[threadIdx.x];
            for (uint32_t grp = 1; grp < (blockDim.x >> log_dom); ++grp) {
                col_sum += red_buf[(grp << log_dom) + threadIdx.x];
            }
            // one row per block, NUM_COSETS * dom_size wide: the [n_partials][d] shape that
            // final_reduce_block_sums consumes
            block_partials[blockIdx.x * NUM_COSETS * dom_size + c * dom_size + dom_pos] =
                col_sum;
        }
        __syncthreads();
        coset_pow *= coset_gen;
    }
}

__global__ void fold_selectors_round0_kernel(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    uint32_t x_count
) {
    size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= x_count)
        return;

    out[gtid] = is_first * in[gtid];                              // is_first
    out[2 * x_count + gtid] = is_last * in[2 * x_count + gtid];       // is_last
    out[x_count + gtid] = FpExt(Fp::one()) - out[2 * x_count + gtid]; // is_transition
}

// Per-coset kernel: blockIdx.y selects the coset and a block works on that coset only.
// It runs acc_constraints with NUM_COSETS = 1, passing the selectors by pointer.
// Picked for small x_count * dom_size, where it puts more blocks on the GPU.
template <bool GLOBAL, bool NEEDS_SHMEM>
__global__ void zerocheck_ntt_evaluate_constraints_coset_parallel_kernel(
    FpExt *__restrict__ block_partials, // [n_partials][n_cosets * dom_size]
    const Fp *__restrict__ sel_cube,    // [3][x_count]
    const Fp *__restrict__ prep_trace,
    const Fp *const *__restrict__ main_traces,
    const FpExt *__restrict__ eq_x, // [x_count]
    const FpExt *__restrict__ lam_pows,
    const Fp *__restrict__ pub_vals,
    const Rule *__restrict__ rule_buf,
    size_t n_rules,
    const size_t *__restrict__ used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas,
    uint32_t n_slots,
    Fp *__restrict__ inter_global,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t trace_h,
    Fp coset_gen
) {
    extern __shared__ char smem[];
    FpExt *red_buf = reinterpret_cast<FpExt *>(smem);
    Fp *ntt_scratch_all =
        NEEDS_SHMEM ? reinterpret_cast<Fp *>(smem + blockDim.x * sizeof(FpExt)) : nullptr;

    uint32_t const log_dom = __ffs(dom_size) - 1;
    bool const trivial_dom = (log_dom == 0);
    uint32_t const n_cosets = gridDim.y;

    // one dom_size-wide NTT scratch slice per xi group of the block
    uint32_t const xi_local = threadIdx.x >> log_dom;
    Fp *ntt_scratch = NEEDS_SHMEM ? (ntt_scratch_all + xi_local * dom_size) : nullptr;

    // dom_pos is the fast-moving index, xi the slow one
    uint32_t const gtid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t const dom_pos = gtid & (dom_size - 1);
    uint32_t const xi_start = gtid >> log_dom;

    // unlike lockstep, the coset comes from the grid's y dimension
    uint32_t const coset_id = blockIdx.y;

    // per-thread constants for this coset
    uint32_t const dom_pos_rev = trivial_dom ? 0 : rev_len(dom_pos, log_dom);

    uint32_t const log_h = __ffs(trace_h) - 1;
    uint32_t const log_seg = min(log_dom, log_h);
    uint32_t const seg_len = 1u << log_seg;
    uint32_t const log_step = log_dom - log_seg;

    Fp const eta_step = trivial_dom ? Fp::one() : TWO_ADIC_GENERATORS[log_dom - log_step];
    Fp const dom_root =
        trivial_dom ? Fp::one() : device_ntt::get_twiddle(log_dom, dom_pos);

    // this coset's power g^(coset_id + 1)
    Fp const coset_pow = pow(coset_gen, coset_id + 1);
    Fp const eval_pt = coset_pow * dom_root;
    Fp const seg_root = exp_power_of_2(eval_pt, log_step);
    Fp const first_scale = avg_gp(seg_root, seg_len);
    Fp const last_scale = avg_gp(seg_root * eta_step, seg_len);
    Fp const shift_pow = trivial_dom ? Fp::one() : pow(coset_pow, dom_pos_rev);

    // node-value slots for a single coset
    Fp reg_slots[GLOBAL ? 1 : BUFFER_THRESHOLD];
    Fp *slot_base;
    uint32_t slot_stride;
    if constexpr (GLOBAL) {
        // spill layout [n_slots][n_cosets * threads_per_coset]
        uint32_t flat_tid = coset_id * gridDim.x * blockDim.x + gtid;
        slot_base = inter_global + flat_tid;
        slot_stride = gridDim.x * gridDim.y * blockDim.x;
    } else {
        slot_base = reg_slots;
        slot_stride = 1;
    }

    FpExt acc_total = FpExt(Fp::zero());

    uint32_t const xi_step = (gridDim.x * blockDim.x) >> log_dom;

    // one-coset evaluation context; nothing in it changes across the xi loop
    NttEvalContext<1> ectx{
        prep_trace,
        main_traces,
        pub_vals,
        slot_base,
        ntt_scratch,
        {shift_pow}, // the single omega shift
        dom_size,
        trace_h,
        slot_stride,
        n_slots,
        dom_pos,
    };

    for (uint32_t xi = xi_start; xi < x_count; xi += xi_step) {
        Fp is_first = first_scale * sel_cube[xi];
        Fp is_last = last_scale * sel_cube[2 * x_count + xi];

        FpExt cons_sums[1];
        acc_constraints<1, NEEDS_SHMEM>(
            cons_sums,
            ectx,
            &is_first,
            &is_last,
            xi,
            lam_pows,
            rule_buf,
            n_rules,
            used_node_ids,
            n_used_nodes,
            n_lambdas
        );

        acc_total += cons_sums[0] * eq_x[xi];
    }

    // reduce this block's one coset
    Fp vanishing = exp_power_of_2(eval_pt, log_dom) - Fp::one();
    // Assumes no coset power coset_gen^(c+1) lies in the skip domain, so eval_pt is outside it
    // as well and the vanishing value is invertible.
    red_buf[threadIdx.x] = acc_total * inv(vanishing);
    __syncthreads();

    if (threadIdx.x < dom_size) {
        FpExt col_sum = red_buf[threadIdx.x];
        for (uint32_t grp = 1; grp < (blockDim.x >> log_dom); ++grp) {
            col_sum += red_buf[(grp << log_dom) + threadIdx.x];
        }
        // same [num_x_blocks][n_cosets * dom_size] layout as the lockstep kernel, so
        // both modes share the final reduction
        block_partials[blockIdx.x * n_cosets * dom_size + coset_id * dom_size + dom_pos] =
            col_sum;
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================
constexpr uint32_t MAX_THREADS = 128;

// True when the workload is small enough for the per-coset grid
inline bool use_coset_parallel_mode(uint32_t x_count, uint32_t dom_size) {
    return (x_count * dom_size) < COSET_PARALLEL_THRESHOLD;
}

// Sizing helper (launches nothing): number of FpExt the caller must allocate for the
// per-block partial-sum buffer.
extern "C" size_t _kb_zerocheck_r0_temp_sums_buffer_size(
    uint32_t n_slots,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t n_cosets,
    size_t temp_budget
) {
    // the mode flag only selects the template instantiation
    return DISPATCH_BOOL(
        round0_config_impl::temp_sums_buffer_size,
        use_coset_parallel_mode(x_count, dom_size),
        n_slots,
        dom_size,
        x_count,
        n_cosets,
        temp_budget,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

extern "C" size_t _kb_zerocheck_r0_intermediates_buffer_size(
    uint32_t n_slots,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t n_cosets,
    size_t temp_budget
) {
    // the mode flag only selects the template instantiation
    return DISPATCH_BOOL(
        round0_config_impl::intermediates_buffer_size,
        use_coset_parallel_mode(x_count, dom_size),
        n_slots,
        dom_size,
        x_count,
        n_cosets,
        temp_budget,
        BUFFER_THRESHOLD,
        MAX_THREADS
    );
}

template <uint32_t NUM_COSETS, bool GLOBAL, bool NEEDS_SHMEM>
int launch_zerocheck_ntt_evaluate_constraints(
    FpExt *block_partials,
    FpExt *final_out,
    const Fp *sel_cube, // [3][x_count]
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_x, // [x_count]
    const FpExt *lam_pows,
    const Fp *pub_vals,
    const Rule *rule_buf,
    size_t n_rules,
    const size_t *used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas,
    uint32_t n_slots,
    Fp *inter_global,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t trace_h,
    Fp coset_gen,
    size_t temp_budget
) {
    auto [launch_grid, launch_block] = coset_round0_config::eval_constraints_launch_params(
        n_slots, dom_size, x_count, NUM_COSETS, temp_budget, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t red_bytes = sizeof(FpExt) * launch_block.x;
    // plus one Fp of NTT scratch per thread when NEEDS_SHMEM
    size_t ntt_scratch_bytes = NEEDS_SHMEM ? sizeof(Fp) * launch_block.x : 0;
    size_t dyn_smem_bytes = red_bytes + ntt_scratch_bytes;

    // PRED: one-time host re-encode of the rule DAG into a pre-decoded stream (cached per
    // rule buffer). Shape-independent; byte-identical to the reference walk. mode 0 => the
    // packing precondition failed and we launch the reference kernel unchanged.
    arith_pred::PackedEntry packed_info =
        arith_pred::prep(rule_buf, n_rules, used_node_ids, n_used_nodes, n_lambdas);
#define LAUNCH_ZC(UP, LF, PKPTR)                                                                   \
    zerocheck_ntt_evaluate_constraints_kernel<NUM_COSETS, GLOBAL, NEEDS_SHMEM, UP, LF>             \
        <<<launch_grid, launch_block, dyn_smem_bytes>>>(                                           \
            block_partials, sel_cube, prep_trace, main_traces, eq_x, lam_pows,                     \
            pub_vals, rule_buf, n_rules, used_node_ids, n_used_nodes, n_lambdas,                   \
            n_slots, inter_global, dom_size, x_count, trace_h, coset_gen, PKPTR)
    if (packed_info.mode == 1) {
        LAUNCH_ZC(true, true, packed_info.d_packed);
    } else if (packed_info.mode == 2) {
        LAUNCH_ZC(true, false, packed_info.d_packed);
    } else {
        LAUNCH_ZC(false, false, nullptr);
    }
#undef LAUNCH_ZC
    int launch_status = CHECK_KERNEL();
    if (launch_status != 0) {
        return launch_status;
    }

    // second pass: add the per-block partials of every (coset, dom_pos) column
    auto n_partials = launch_grid.x;
    auto [fin_grid, fin_block] = kernel_launch_params(n_partials);
    unsigned int fin_warps = div_ceil(fin_block.x, WARP_SIZE);
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);
    reduce::final_reduce_block_sums<<<NUM_COSETS * dom_size, fin_block, fin_smem>>>(
        block_partials, final_out, n_partials
    );

    return CHECK_KERNEL();
}

// runtime-to-template dispatch over n_cosets in 1..4, spill_global and wide_dom
DEFINE_DISPATCH_N_B1_B2(dispatch_zerocheck, launch_zerocheck_ntt_evaluate_constraints, 4)

// Per-coset launcher: blockIdx.y picks the coset
template <bool GLOBAL, bool NEEDS_SHMEM>
int launch_zerocheck_coset_parallel(
    FpExt *block_partials,
    FpExt *final_out,
    const Fp *sel_cube,
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_x,
    const FpExt *lam_pows,
    const Fp *pub_vals,
    const Rule *rule_buf,
    size_t n_rules,
    const size_t *used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas,
    uint32_t n_slots,
    Fp *inter_global,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t trace_h,
    uint32_t n_cosets,
    Fp coset_gen,
    size_t temp_budget
) {
    auto [launch_grid, launch_block] = coset_parallel_round0_config::eval_constraints_launch_params(
        n_slots, dom_size, x_count, n_cosets, temp_budget, BUFFER_THRESHOLD, MAX_THREADS
    );

    size_t red_bytes = sizeof(FpExt) * launch_block.x;
    size_t ntt_scratch_bytes = NEEDS_SHMEM ? sizeof(Fp) * launch_block.x : 0;
    size_t dyn_smem_bytes = red_bytes + ntt_scratch_bytes;

    zerocheck_ntt_evaluate_constraints_coset_parallel_kernel<GLOBAL, NEEDS_SHMEM>
        <<<launch_grid, launch_block, dyn_smem_bytes>>>(
            block_partials,
            sel_cube,
            prep_trace,
            main_traces,
            eq_x,
            lam_pows,
            pub_vals,
            rule_buf,
            n_rules,
            used_node_ids,
            n_used_nodes,
            n_lambdas,
            n_slots,
            inter_global,
            dom_size,
            x_count,
            trace_h,
            coset_gen
        );
    int launch_status = CHECK_KERNEL();
    if (launch_status != 0) {
        return launch_status;
    }

    // second pass, shared with the lockstep path: add the per-block partials per column
    auto n_partials = launch_grid.x;
    auto [fin_grid, fin_block] = kernel_launch_params(n_partials);
    unsigned int fin_warps = div_ceil(fin_block.x, WARP_SIZE);
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);
    reduce::final_reduce_block_sums<<<n_cosets * dom_size, fin_block, fin_smem>>>(
        block_partials, final_out, n_partials
    );

    return CHECK_KERNEL();
}

extern "C" int _kb_zerocheck_ntt_eval_constraints(
    FpExt *block_partials, // [n_partials][n_cosets * dom_size]
    FpExt *final_out,      // [n_cosets * dom_size]
    const Fp *sel_cube,    // [3][x_count]
    const Fp *prep_trace,
    const Fp *const *main_traces,
    const FpExt *eq_x, // [x_count]
    const FpExt *lam_pows,
    const Fp *pub_vals,
    const Rule *rule_buf,
    size_t n_rules,
    const size_t *used_node_ids,
    size_t n_used_nodes,
    size_t n_lambdas,
    uint32_t n_slots,
    Fp *inter_global,
    uint32_t dom_size,
    uint32_t x_count,
    uint32_t trace_h,
    uint32_t n_cosets,
    Fp coset_gen,
    size_t temp_budget
) {
    bool spill_global = n_slots > BUFFER_THRESHOLD;
    bool per_coset = use_coset_parallel_mode(x_count, dom_size);
    if (!per_coset && n_cosets > 4) {
        return cudaErrorInvalidValue;
    }

#define KERNEL_ARGS                                                                                \
    block_partials, final_out, sel_cube, prep_trace, main_traces, eq_x, lam_pows,                  \
        pub_vals, rule_buf, n_rules, used_node_ids, n_used_nodes, n_lambdas, n_slots,              \
        inter_global, dom_size, x_count, trace_h

    if (dom_size == 1) {
        if (per_coset) {
            return spill_global ? launch_zerocheck_coset_parallel<true, false>(
                                   KERNEL_ARGS, n_cosets, coset_gen, temp_budget
                               )
                             : launch_zerocheck_coset_parallel<false, false>(
                                   KERNEL_ARGS, n_cosets, coset_gen, temp_budget
                               );
        }
        return dispatch_zerocheck(
            n_cosets, spill_global, false, KERNEL_ARGS, coset_gen, temp_budget
        );
    }

    bool wide_dom = dom_size > WARP_SIZE;

    // small workloads take the per-coset grid
    if (per_coset) {
        // grid.y = n_cosets, one coset per block
        return DISPATCH_BOOL_PAIR(
            launch_zerocheck_coset_parallel,
            spill_global,
            wide_dom,
            KERNEL_ARGS,
            n_cosets,
            coset_gen,
            temp_budget
        );
    } else {
        // larger workloads: lockstep, one thread covers every coset
        return dispatch_zerocheck(
            n_cosets, spill_global, wide_dom, KERNEL_ARGS, coset_gen, temp_budget
        );
    }
#undef KERNEL_ARGS
}

extern "C" int _kb_fold_selectors_round0(
    FpExt *out,
    const Fp *in,
    FpExt is_first,
    FpExt is_last,
    uint32_t x_count
) {
    auto [launch_grid, launch_block] = kernel_launch_params(x_count);
    fold_selectors_round0_kernel<<<launch_grid, launch_block>>>(out, in, is_first, is_last, x_count);
    return CHECK_KERNEL();
}

} // namespace zerocheck_round0

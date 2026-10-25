// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "prover/logup_zerocheck/eval_config.cuh"
#include "prover/dag/eval_ctx.cuh"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "prover/dag/monomial.cuh"
#include "common/oom_heal.cuh"
#include "prover/reduce.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <vector_types.h>

namespace logup_zerocheck_mle {

// Load an FpExt as one 16B vector op (uint4 __ldg) instead of letting the compiler split
// a plain struct copy into 2x LDG.64. FpExt (kb31_4_t) is only 4-byte aligned per its type
// layout, so the compiler can't assume 128-bit width on its own. Every FpExt read below
// indexes a contiguous FpExt[] array at `base + k * sizeof(FpExt)` for an integer k, and the
// backing allocation is a cudaMalloc'd device buffer (>=16B aligned in practice for any
// realistically-sized allocation) — so the address is always 16B-aligned regardless of
// alignof(FpExt). Pure register-reinterpretation after -O3, so this is byte-exact by
// construction (double-checked by the MLE_VERIFY=1 correctness gate).
__device__ __forceinline__ FpExt ldg_fpext(const FpExt *p) {
    uint4 r = __ldg(reinterpret_cast<const uint4 *>(p));
    FpExt out;
    __builtin_memcpy(&out, &r, sizeof(FpExt));
    return out;
}

__device__ __forceinline__ FpExt
eval_variable(PackedVar packed, uint32_t row_idx, const EvalCoreCtx &core, uint32_t col_len) {
    uint8_t kind = packed.entry_type();
    uint8_t rot_off = packed.offset();

    switch (kind) {
    case 1: { // MAIN
        auto part = core.d_main[packed.part_index()];
        const auto mat_stride = col_len * part.air_width;
        const FpExt *__restrict__ mat = part.data + mat_stride * rot_off;
        const FpExt *__restrict__ col_base = mat + col_len * packed.col_index();
        return ldg_fpext(col_base + row_idx);
    }
    case 0: { // PREPROCESSED
        const auto mat_stride = col_len * core.d_preprocessed.air_width;
        const FpExt *__restrict__ mat = core.d_preprocessed.data + mat_stride * rot_off;
        const FpExt *__restrict__ col_base = mat + col_len * packed.col_index();
        return ldg_fpext(col_base + row_idx);
    }
    case 3: // PUBLIC
        return FpExt(core.d_public[packed.col_index()]);
    case 8: // IS_FIRST
        return ldg_fpext(core.d_selectors + row_idx);
    case 9: // IS_LAST
        return ldg_fpext(core.d_selectors + 2 * col_len + row_idx);
    case 10: // IS_TRANSITION
        return ldg_fpext(core.d_selectors + col_len + row_idx);
    default:
        assert(false);
    }
    return FpExt(Fp::zero());
}

// Per-AIR descriptor for the batched zerocheck monomial kernels (mirrors the Rust struct).
struct MonomialAirCtx {
    const MonomialHeader *__restrict__ d_headers;
    const PackedVar *__restrict__ d_variables;
    const FpExt *__restrict__ d_lambda_combinations; // one lambda combination per monomial
    uint32_t num_monomials;
    EvalCoreCtx eval_ctx;
    const FpExt *__restrict__ d_eq_xi;
    uint32_t num_y;
};

// Per-launch (local air_idx -> structural cache) reference, passed as a NEW trailing
// kernel argument (MonomialAirCtx's layout is left untouched). A null `d_fused` means this
// AIR falls back to the original d_headers/d_variables loop for this launch.
struct MonoFusedRef {
    const uint4 *__restrict__ d_fused;
};

// Host-readable mirror of MonomialAirCtx: EvalCoreCtx's `d_preprocessed` field is
// const-qualified (a value member, not just a pointer-to-const), which deletes
// MonomialAirCtx's default constructor and copy-assignment -- so it cannot be used as a
// std::vector element or cudaMemcpy destination directly. This mirror is layout-identical
// (verified by the static_asserts below) and is only ever populated via cudaMemcpy of the
// real device bytes, never through assignment.
struct EvalCtxPOD {
    const FpExt *d_selectors;
    MainMatrixPtrs<FpExt> d_preprocessed;
    const MainMatrixPtrs<FpExt> *d_main;
    const Fp *d_public;
};
struct AirCtxPOD {
    const MonomialHeader *d_headers;
    const PackedVar *d_variables;
    const FpExt *d_lambda_combinations;
    uint32_t num_monomials;
    EvalCtxPOD eval_ctx;
    const FpExt *d_eq_xi;
    uint32_t num_y;
};
static_assert(sizeof(EvalCtxPOD) == sizeof(EvalCoreCtx), "EvalCtxPOD layout mismatch");
static_assert(sizeof(AirCtxPOD) == sizeof(MonomialAirCtx), "AirCtxPOD layout mismatch");
static_assert(offsetof(AirCtxPOD, eval_ctx) == offsetof(MonomialAirCtx, eval_ctx),
             "AirCtxPOD eval_ctx offset mismatch");
static_assert(offsetof(AirCtxPOD, d_eq_xi) == offsetof(MonomialAirCtx, d_eq_xi),
             "AirCtxPOD d_eq_xi offset mismatch");
static_assert(offsetof(AirCtxPOD, num_y) == offsetof(MonomialAirCtx, num_y),
             "AirCtxPOD num_y offset mismatch");

// ============================================================================
// FUSED_PREIDX (structural variant): per-AIR packed-monomial cache
// ============================================================================
//
// A structural variant of PREIDX, for correctness:
//
// PREIDX proper bakes an ABSOLUTE element index for MAIN-part-0 / selector variables:
// idx = height*main0_width*offset + height*col. That index is only valid for the exact `height`
// (= num_x * num_y) and the exact main/selector buffer ADDRESSES present when the cache was built.
//
// In production, `_kb_zerocheck_monomial_par_y_batched` is called once per
// sumcheck round (mod.rs: `for round in 1..=n_max` -> sumcheck_polys_batch_eval
// -> sumcheck_polys_batch_eval_gpu). Per round, a trace's num_y is
// `1 << (n_lift - round)` — it HALVES every round — and the backing main/preprocessed/selector
// buffers are freshly folded/allocated each round (interpolate_columns_gpu into a new
// DeviceMatrix). So for the SAME AIR (same d_headers pointer, which is stable for the whole prove),
// `height` and the main/selector buffer addresses are DIFFERENT on essentially every call. Baking
// either into a cache keyed on d_headers would force invalidation on every single call (zero reuse
// -- the entire point of caching lost), or worse, silently read the wrong row if an invalidation
// guard were imperfect.
//
// `d_headers`/`d_variables` THEMSELVES are safe to cache (built once in AirDataGpu::new, never
// rebuilt). What we cache here is therefore the STRUCTURAL flattening only: one 16B record per
// monomial holding `num_vars` plus the raw 32-bit PackedVar word for each of up to 3 variables:
// fused[m] = uint4{num_vars, var0.data, var1.data, var2.data}. This removes the two-level pointer
// chase (d_headers[m], then a data-dependent d_variables[hdr.var_offset+v] per variable) down to
// ONE aligned 128-bit load per monomial. The per-variable decode (switch on entry_type,
// height-scaled stride, live main/selector pointer) still runs exactly as eval_variable() does
// today, fed by the CURRENT actx.eval_ctx / height at every launch -- nothing height- or
// address-dependent is ever cached, so the record is valid for the AIR's entire prove lifetime,
// independent of round-to-round churn. `d_lambda_combinations` is never touched by this cache
// either (per-chunk pool-allocated, unsafe to bake) -- it stays a live `ldg_fpext` read every
// launch.
namespace mono_fused {

struct PackedEntry {
    uint4 *d_fused;
    uint32_t num_monomials;
    bool usable;
};

// Decode d_headers/d_variables on the host and build the fused record stream. Any monomial
// with num_vars > 3 (the fixed record width) fails the WHOLE AIR back to the reference loop.
static PackedEntry build_packed(const MonomialHeader *d_headers, const PackedVar *d_variables,
                                uint32_t num_monomials) {
    PackedEntry e{nullptr, num_monomials, false};
    if (num_monomials == 0 || d_headers == nullptr)
        return e;

    std::vector<MonomialHeader> hdrs(num_monomials);
    if (cudaMemcpy(hdrs.data(), d_headers, num_monomials * sizeof(MonomialHeader),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return e;

    uint32_t total_vars = 0;
    for (const MonomialHeader &h : hdrs) {
        if (h.num_vars > 3)
            return e; // whole-AIR fallback: fixed 3-slot record can't hold this monomial
        total_vars = std::max(total_vars, h.var_offset + (uint32_t)h.num_vars);
    }

    std::vector<PackedVar> vars(total_vars ? total_vars : 1);
    if (total_vars) {
        if (d_variables == nullptr)
            return e;
        if (cudaMemcpy(vars.data(), d_variables, total_vars * sizeof(PackedVar),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return e;
    }

    std::vector<uint4> fused(num_monomials);
    for (uint32_t m = 0; m < num_monomials; m++) {
        const MonomialHeader &h = hdrs[m];
        uint32_t w[3] = {0, 0, 0};
        for (uint32_t v = 0; v < h.num_vars; v++)
            w[v] = vars[h.var_offset + v].data;
        fused[m] = uint4{(uint32_t)h.num_vars, w[0], w[1], w[2]};
    }

    uint4 *d = nullptr;
    if (mle_malloc_or_fallback(&d, num_monomials * sizeof(uint4)) != cudaSuccess)
        return e;
    if (cudaMemcpy(d, fused.data(), num_monomials * sizeof(uint4), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(d);
        return e;
    }
    e.d_fused = d;
    e.usable = true;
    return e;
}

// Cache keyed on the AIR's device headers pointer. The key is only stable while the pk's
// buffer lives: after a free the pool can hand the same address to a DIFFERENT air's monomials,
// so the owning pk's drop purges the entry via `invalidate` below. Guarded by num_monomials as a
// defensive trigger, mirroring arith_pred::prep's rules_len guard. Mutex-guarded for
// multi-prover-thread use (same reason as arith_pred::prep).
static std::mutex &cache_mtx() {
    static std::mutex mtx;
    return mtx;
}
static std::unordered_map<const MonomialHeader *, PackedEntry> &cache_map() {
    static std::unordered_map<const MonomialHeader *, PackedEntry> cache;
    return cache;
}
static void invalidate(const void *d_headers) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(reinterpret_cast<const MonomialHeader *>(d_headers));
    if (it != cache.end()) {
        if (it->second.d_fused) cudaFree(it->second.d_fused);
        cache.erase(it);
    }
}
static PackedEntry prep(const MonomialHeader *d_headers, const PackedVar *d_variables,
                        uint32_t num_monomials) {
    std::lock_guard<std::mutex> lock(cache_mtx());
    auto &cache = cache_map();
    auto it = cache.find(d_headers);
    if (it != cache.end() && it->second.num_monomials == num_monomials)
        return it->second;
    if (it != cache.end() && it->second.d_fused)
        cudaFree(it->second.d_fused);
    PackedEntry e = build_packed(d_headers, d_variables, num_monomials);
    cache[d_headers] = e;
    return e;
}

// Uploads the small per-launch (local air_idx -> ref) table using the stream-ordered
// allocator, so the paired cudaFreeAsync (issued by the caller right after the kernel launch)
// is safely ordered after the kernel that reads it, even under concurrent prover threads --
// matching the existing cudaMallocFromPoolAsync usage for d_lambda_combinations elsewhere in
// this codebase. Returns nullptr on any allocation failure (caller then launches with a null
// table, i.e. every AIR falls back for this one launch -- safe, just misses the fast path).
static MonoFusedRef *upload_air_refs(const std::vector<MonoFusedRef> &h_refs,
                                     cudaStream_t stream) {
    if (h_refs.empty())
        return nullptr;
    MonoFusedRef *d_refs = nullptr;
    if (mle_malloc_async_or_fallback(&d_refs, h_refs.size() * sizeof(MonoFusedRef), stream) !=
        cudaSuccess)
        return nullptr;
    if (cudaMemcpyAsync(d_refs, h_refs.data(), h_refs.size() * sizeof(MonoFusedRef),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFreeAsync(d_refs, stream);
        return nullptr;
    }
    return d_refs;
}

} // namespace mono_fused

// Purge a freed monomial-headers pointer from the fused-monomial cache (see
// `mono_fused::invalidate`).
extern "C" void _kb_mono_fused_invalidate(const void *d_headers) {
    mono_fused::invalidate(d_headers);
}

// ============================================================================
// Lambda folding: per monomial, sum of lambda^c * coeff over its constraint terms
// ============================================================================

__global__ void fold_lambda_per_monomial_kernel(
    FpExt *__restrict__ combos,
    const MonomialHeader *__restrict__ hdrs,
    const LambdaTerm *__restrict__ terms_in,
    const FpExt *__restrict__ pows,
    uint32_t n_mono
) {
    uint32_t mono = blockIdx.x * blockDim.x + threadIdx.x;
    if (mono >= n_mono)
        return;

    MonomialHeader h = hdrs[mono];
    FpExt acc(Fp::zero());
    for (uint16_t k = 0; k < h.num_terms; ++k) {
        LambdaTerm lt = terms_in[h.term_offset + k];
        acc += pows[lt.constraint_idx] * lt.coefficient;
    }
    combos[mono] = acc;
}

extern "C" int _kb_precompute_lambda_combinations(
    FpExt *combos,
    const MonomialHeader *hdrs,
    const LambdaTerm *terms_in,
    const FpExt *pows,
    uint32_t n_mono
) {
    if (n_mono == 0)
        return 0;

    constexpr uint32_t block_threads = 256;
    uint32_t grid_blocks = div_ceil(n_mono, block_threads);
    fold_lambda_per_monomial_kernel<<<grid_blocks, block_threads>>>(
        combos, hdrs, terms_in, pows, n_mono
    );
    return CHECK_KERNEL();
}

// ============================================================================
// One-thread-per-monomial zerocheck kernel; every AIR of the batch in one launch
// ============================================================================

__global__ void zc_mono_per_thread_kernel(
    FpExt *__restrict__ partials,
    const BlockCtx *__restrict__ blk_ctxs,
    const MonomialAirCtx *__restrict__ airs,
    uint32_t tpb
) {
    extern __shared__ char smem[];
    FpExt *warp_buf = (FpExt *)smem;

    BlockCtx blk = blk_ctxs[blockIdx.x];
    MonomialAirCtx air = airs[blk.air_idx];

    uint32_t n_x = gridDim.y;
    uint32_t x_pos = blockIdx.y;

    // local_block_idx_x = y * tiles_per_y + tile
    uint32_t tiles_per_y = (air.num_monomials + tpb - 1) / tpb;
    uint32_t y = blk.local_block_idx_x / tiles_per_y;
    uint32_t tile = blk.local_block_idx_x % tiles_per_y;

    uint32_t col_len = n_x * air.num_y;
    uint32_t row_idx = x_pos * air.num_y + y;

    FpExt acc(Fp::zero());

    // one monomial per thread
    uint32_t mono = tile * tpb + threadIdx.x;
    if (mono < air.num_monomials) {
        MonomialHeader h = air.d_headers[mono];

        // multiply the monomial's variables together
        FpExt prod(Fp::one());
        for (uint16_t vi = 0; vi < h.num_vars; ++vi) {
            PackedVar pv = air.d_variables[h.var_offset + vi];
            prod *= eval_variable(pv, row_idx, air.eval_ctx, col_len);
        }

        acc = prod * air.d_lambda_combinations[mono];
    }

    // sum over the block's monomials
    FpExt blk_sum = reduce::block_reduce_sum(acc, warp_buf);

    if (threadIdx.x == 0) {
        // eq_xi(y) is shared by every x and every monomial of this block, so
        // multiply it in once, after the reduction
        blk_sum *= air.d_eq_xi[y];
        partials[blockIdx.x * n_x + x_pos] = blk_sum;
    }
}

extern "C" int _kb_zerocheck_monomial_batched(
    FpExt *partials,
    FpExt *out,
    const BlockCtx *blk_ctxs,
    const MonomialAirCtx *airs,
    const uint32_t *air_offsets,
    uint32_t n_blocks,
    uint32_t n_x,
    uint32_t n_airs,
    uint32_t tpb
) {
    if (n_blocks == 0) {
        return 0;
    }
    if (!valid_grid_y_dim(n_x)) {
        return cudaErrorInvalidValue;
    }
    // Phase 2 divides by n_airs and launches a grid of that width; the Rust side never
    // builds an empty batch, but the invariant lives there, not here.
    if (n_airs == 0) {
        return cudaErrorInvalidValue;
    }

    dim3 launch_grid(n_blocks, n_x);
    dim3 launch_block(tpb);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // pass 1: per-block partial sums
    zc_mono_per_thread_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        partials, blk_ctxs, airs, tpb
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // pass 2: one block per (air, x) folds that AIR's partials
    // (grid = n_airs x n_x)
    auto [ignored_grid, fin_block] = kernel_launch_params(n_blocks / n_airs + 1);
    unsigned int fin_warps = (fin_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);

    dim3 fin_grid(n_airs, n_x);
    reduce::batched_final_reduce_block_sums<<<fin_grid, fin_block, fin_smem>>>(
        partials, out, air_offsets, n_x
    );

    return CHECK_KERNEL();
}

// ============================================================================
// PAR-Y KERNEL: threads split the y_int range, monomials are walked in tiles
// ============================================================================

// Launch width for the par-y kernel. The `__launch_bounds__` and the launcher must both use it:
// Rust owns the launcher's width (THREADS_PER_BLOCK_PAR_Y in
// logup_zerocheck/batch_mle_monomial.rs) and passes it in, so the entry point below rejects any
// other value rather than letting the two drift.
#define PAR_Y_BLOCK 128

// OCCUPANCY: unbounded the kernel uses 82 registers/thread, which binds it to 5 resident blocks
// per SM against a warp limit of 12 -- registers, not warps, are what limits it. Asking for 8
// blocks budgets 64 instead: occupancy 41.1 -> 65.4 %, duration -10.6 % over 60 launches paired
// by grid size, 21 B of spill. The min-blocks form is measured, per merkle_tree.cu's rule that
// an unmeasured min-blocks target risks forcing spills to reach a bound it cannot meet.
__global__ void __launch_bounds__(PAR_Y_BLOCK, 8) zc_mono_par_y_kernel(
    FpExt *__restrict__ partials,
    const BlockCtx *__restrict__ blk_ctxs,
    const MonomialAirCtx *__restrict__ airs,
    uint32_t tpb,
    uint32_t monos_per_chunk, // monomials per chunk
    // FUSED_PREIDX: per-(local air_idx) structural cache ref, or nullptr if none is available
    // this launch (e.g. alloc failure) -- every AIR then falls back to the original loop.
    const MonoFusedRef *__restrict__ fused_refs
) {
    extern __shared__ char smem[];
    FpExt *warp_buf = (FpExt *)smem;

    BlockCtx blk = blk_ctxs[blockIdx.x];
    MonomialAirCtx air = airs[blk.air_idx];

    uint32_t n_x = gridDim.y;
    uint32_t x_pos = blockIdx.y;

    // Decode y_tile and chunk from local_block_idx_x
    // Encoding: local_block_idx_x = y_tile * chunks_per_y + chunk
    uint32_t chunks_per_y = (air.num_monomials + monos_per_chunk - 1) / monos_per_chunk;
    uint32_t y_tile = blk.local_block_idx_x / chunks_per_y;
    uint32_t chunk = blk.local_block_idx_x % chunks_per_y;

    // Decode y from thread position within y_tile
    uint32_t y = threadIdx.x + y_tile * tpb;
    bool y_valid = (y < air.num_y);

    uint32_t col_len = n_x * air.num_y;
    uint32_t row_idx = x_pos * air.num_y + y;

    // this block's contiguous range of monomials
    uint32_t first_mono = chunk * monos_per_chunk;
    uint32_t last_mono = min(first_mono + monos_per_chunk, air.num_monomials);

    // FUSED_PREIDX fast path for this block's AIR, if the host-side cache produced one.
    const uint4 *__restrict__ recs =
        (fused_refs != nullptr) ? fused_refs[blk.air_idx].d_fused : nullptr;

    FpExt acc(Fp::zero());
    if (y_valid) {
        // Each thread loops over its chunk of monomials.
        // Seed the product with the per-monomial lambda combination instead of
        // FpExt::one(): field multiplication is commutative/associative and
        // exact, so folding lambda in as the initial factor is bit-identical to
        // (v0*v1*...)*lambda, while eliminating one full quartic ext*ext mul per
        // monomial (the wasted one*v0 identity multiply, plus the separate
        // product*lambda mul). For num_vars==0 the product is exactly lambda
        // (the constant monomial term).
        if (recs != nullptr) {
            // FUSED_PREIDX fast path: one 16B record per monomial (num_vars + up to 3 raw
            // PackedVar words), replacing the d_headers[mono] + d_variables[var_offset+vi] chase.
            // Same per-variable decode (eval_variable), same accumulation order => byte-exact.
            for (uint32_t mono = first_mono; mono < last_mono; ++mono) {
                uint4 r = recs[mono];
                FpExt prod = ldg_fpext(&air.d_lambda_combinations[mono]);
                if (r.x >= 1)
                    prod *= eval_variable(PackedVar{r.y}, row_idx, air.eval_ctx, col_len);
                if (r.x >= 2)
                    prod *= eval_variable(PackedVar{r.z}, row_idx, air.eval_ctx, col_len);
                if (r.x >= 3)
                    prod *= eval_variable(PackedVar{r.w}, row_idx, air.eval_ctx, col_len);

                acc += prod;
            }
        } else {
            for (uint32_t mono = first_mono; mono < last_mono; ++mono) {
                MonomialHeader h = air.d_headers[mono];

                FpExt prod = ldg_fpext(&air.d_lambda_combinations[mono]);
                for (uint16_t vi = 0; vi < h.num_vars; ++vi) {
                    PackedVar pv = air.d_variables[h.var_offset + vi];
                    prod *= eval_variable(pv, row_idx, air.eval_ctx, col_len);
                }

                acc += prod;
            }
        }

        // Apply eq_xi
        acc *= ldg_fpext(&air.d_eq_xi[y]);
    }

    // Block reduction sums across y values in this block
    FpExt blk_sum = reduce::block_reduce_sum(acc, warp_buf);

    if (threadIdx.x == 0) {
        partials[blockIdx.x * n_x + x_pos] = blk_sum;
    }
}

extern "C" int _kb_zerocheck_monomial_par_y_batched(
    FpExt *partials,
    FpExt *out,
    const BlockCtx *blk_ctxs,
    const MonomialAirCtx *airs,
    const uint32_t *air_offsets,
    uint32_t n_blocks,
    uint32_t n_x,
    uint32_t n_airs,
    uint32_t monos_per_chunk,
    uint32_t tpb
) {
    if (n_blocks == 0) {
        return 0;
    }
    if (!valid_grid_y_dim(n_x)) {
        return cudaErrorInvalidValue;
    }
    // Phase 2 divides by n_airs and launches a grid of that width; the Rust side never
    // builds an empty batch, but the invariant lives there, not here.
    if (n_airs == 0) {
        return cudaErrorInvalidValue;
    }
    // PAR_Y_BLOCK is the kernel's `__launch_bounds__` width, so a wider block fails the launch
    // outright and a narrower one budgets registers for a shape the kernel is never launched at.
    // Both are compile-time drift between this file and the Rust constant, never runtime input,
    // so reject either here instead of letting one fail late and the other pass silently.
    if (tpb != PAR_Y_BLOCK) {
        return cudaErrorInvalidValue;
    }

    dim3 launch_grid(n_blocks, n_x);
    dim3 launch_block(tpb);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // FUSED_PREIDX: build/reuse a per-AIR structural cache and a small per-launch
    // (local air_idx -> ref) table. `airs` lives on the device (uploaded fresh by Rust
    // every call), so we D2H a small mirror to read each AIR's d_headers/d_variables/
    // num_monomials -- that's the cache key and rebuild input; see the mono_fused namespace
    // above for why this does NOT bake height or any device buffer address.
    MonoFusedRef *dev_refs = nullptr;
    if (n_airs > 0) {
        std::vector<AirCtxPOD> host_airs(n_airs);
        if (cudaMemcpy(host_airs.data(), airs, n_airs * sizeof(AirCtxPOD),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::vector<MonoFusedRef> host_refs(n_airs);
            for (uint32_t ai = 0; ai < n_airs; ai++) {
                const AirCtxPOD &air = host_airs[ai];
                mono_fused::PackedEntry entry =
                    mono_fused::prep(air.d_headers, air.d_variables, air.num_monomials);
                host_refs[ai].d_fused = entry.usable ? entry.d_fused : nullptr;
            }
            dev_refs = mono_fused::upload_air_refs(host_refs, /*stream=*/0);
        }
    }

    // pass 1: par-y partial sums
    zc_mono_par_y_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        partials, blk_ctxs, airs, tpb, monos_per_chunk, dev_refs
    );
    if (dev_refs != nullptr) {
        cudaFreeAsync(dev_refs, /*stream=*/0);
    }
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // pass 2: fold each AIR's partials, one block per (air, x)
    auto [ignored_grid, fin_block] = kernel_launch_params(n_blocks / n_airs + 1);
    unsigned int fin_warps = (fin_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);

    dim3 fin_grid(n_airs, n_x);
    reduce::batched_final_reduce_block_sums<<<fin_grid, fin_block, fin_smem>>>(
        partials, out, air_offsets, n_x
    );

    return CHECK_KERNEL();
}

// ============================================================================
// LogUp numerator / denominator monomials
// ============================================================================

template <bool IsDenom>
__global__ void fold_interaction_per_monomial_kernel(
    FpExt *__restrict__ combos,
    const MonomialHeader *__restrict__ hdrs,
    const InteractionMonomialTerm *__restrict__ iterms,
    const FpExt *__restrict__ betas,
    const FpExt *__restrict__ eq_bus,
    uint32_t n_mono
) {
    uint32_t mono = blockIdx.x * blockDim.x + threadIdx.x;
    if (mono >= n_mono)
        return;

    MonomialHeader h = hdrs[mono];
    FpExt acc(Fp::zero());
    for (uint16_t k = 0; k < h.num_terms; ++k) {
        InteractionMonomialTerm it = iterms[h.term_offset + k];
        FpExt contrib = eq_bus[it.interaction_idx] * it.coefficient;
        if constexpr (IsDenom) {
            contrib *= betas[it.field_idx];
        }
        acc += contrib;
    }
    combos[mono] = acc;
}

extern "C" int _kb_precompute_logup_numer_combinations(
    FpExt *combos,
    const MonomialHeader *hdrs,
    const InteractionMonomialTerm *iterms,
    const FpExt *eq_bus,
    uint32_t n_mono
) {
    if (n_mono == 0)
        return 0;

    constexpr uint32_t block_threads = 256;
    uint32_t grid_blocks = div_ceil(n_mono, block_threads);
    fold_interaction_per_monomial_kernel<false><<<grid_blocks, block_threads>>>(
        combos, hdrs, iterms, nullptr, eq_bus, n_mono
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_precompute_logup_denom_combinations(
    FpExt *combos,
    const MonomialHeader *hdrs,
    const InteractionMonomialTerm *iterms,
    const FpExt *betas,
    const FpExt *eq_bus,
    uint32_t n_mono
) {
    if (n_mono == 0)
        return 0;

    constexpr uint32_t block_threads = 256;
    uint32_t grid_blocks = div_ceil(n_mono, block_threads);
    fold_interaction_per_monomial_kernel<true><<<grid_blocks, block_threads>>>(
        combos, hdrs, iterms, betas, eq_bus, n_mono
    );
    return CHECK_KERNEL();
}

// Per-AIR inputs shared by the numerator and denominator passes of the LogUp monomial kernel;
// layout mirrors the Rust LogupMonomial*Ctx structs.
struct LogupMonomialCommonCtx {
    EvalCoreCtx eval_ctx;
    const FpExt *__restrict__ d_eq_xi;
    FpExt bus_term_sum; // precomputed sum_i beta[message_len_i] * (bus_idx[i] + 1) * eq_3bs[i]
    uint32_t num_y;
    uint32_t mono_blocks;
};

struct LogupMonomialCtx {
    const MonomialHeader *__restrict__ d_headers;
    const PackedVar *__restrict__ d_variables;
    const FpExt *__restrict__ d_combinations;
    uint32_t num_monomials;
};

// LogUp monomial pass. One block per (y, tile of monomials), one thread per monomial; the
// block sum lands in one half of a FracExt partial so the usual reduction can finish it.
//
// IsDenom = false: writes .p (numerator).
// IsDenom = true:  writes .q (denominator); the tile-0 block also adds bus_term_sum * eq_xi(y).
//
// The two passes are separate launches over the same blk_ctxs / air_commons; they differ
// only in the per-AIR monomial set (`side_ctxs`) and in which half of the FracExt they
// write.
template <bool IsDenom>
__global__ void logup_mono_tile_kernel(
    FracExt *__restrict__ partials,
    const BlockCtx *__restrict__ blk_ctxs,
    const LogupMonomialCommonCtx *__restrict__ air_commons,
    const LogupMonomialCtx *__restrict__ side_ctxs
) {
    extern __shared__ char smem[];
    FpExt *warp_buf = (FpExt *)smem;

    BlockCtx blk = blk_ctxs[blockIdx.x];
    LogupMonomialCommonCtx common = air_commons[blk.air_idx];
    LogupMonomialCtx side = side_ctxs[blk.air_idx];

    uint32_t n_x = gridDim.y;
    uint32_t x_pos = blockIdx.y;
    uint32_t col_len = n_x * common.num_y;

    uint32_t mono_blocks = common.mono_blocks;
#ifdef CUDA_DEBUG
    assert(mono_blocks > 0);
#endif
    uint32_t y = blk.local_block_idx_x / mono_blocks;
    uint32_t tile = blk.local_block_idx_x % mono_blocks;
    uint32_t row_idx = x_pos * common.num_y + y;
    uint32_t mono = tile * blockDim.x + threadIdx.x;

    FpExt acc(Fp::zero());
    if (y < common.num_y && mono < side.num_monomials) {
        MonomialHeader h = side.d_headers[mono];
        FpExt term_val = side.d_combinations[mono];
        for (uint16_t vi = 0; vi < h.num_vars; ++vi) {
            PackedVar pv = side.d_variables[h.var_offset + vi];
            term_val *= eval_variable(pv, row_idx, common.eval_ctx, col_len);
        }
        acc = term_val * common.d_eq_xi[y];
    }

    FpExt blk_sum = reduce::block_reduce_sum(acc, warp_buf);

    if (threadIdx.x == 0) {
        if constexpr (IsDenom) {
            if (tile == 0 && y < common.num_y) {
                blk_sum += common.bus_term_sum * common.d_eq_xi[y];
            }
            partials[blockIdx.x * n_x + x_pos].q = blk_sum;
        } else {
            partials[blockIdx.x * n_x + x_pos].p = blk_sum;
        }
    }
}

extern "C" int _kb_logup_monomial_batched(
    FracExt *partials,
    FracExt *out,
    const BlockCtx *blk_ctxs,
    const LogupMonomialCommonCtx *air_commons,
    const LogupMonomialCtx *numer_sides,
    const LogupMonomialCtx *denom_sides,
    const uint32_t *air_offsets,
    uint32_t n_blocks,
    uint32_t n_x,
    uint32_t n_airs,
    uint32_t tpb
) {
    if (n_blocks == 0)
        return 0;
    if (!valid_frac_grid_y_dim(n_x)) {
        return cudaErrorInvalidValue;
    }
    // Phase 2 divides by n_airs and launches a grid of that width; the Rust side never
    // builds an empty batch, but the invariant lives there, not here.
    if (n_airs == 0) {
        return cudaErrorInvalidValue;
    }

    dim3 launch_grid(n_blocks, n_x);
    dim3 launch_block(tpb);
    size_t smem_bytes = div_ceil(tpb, WARP_SIZE) * sizeof(FpExt);

    // pass 1a: numerator halves
    logup_mono_tile_kernel<false><<<launch_grid, launch_block, smem_bytes>>>(
        partials, blk_ctxs, air_commons, numer_sides
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // pass 1b: denominator halves
    logup_mono_tile_kernel<true><<<launch_grid, launch_block, smem_bytes>>>(
        partials, blk_ctxs, air_commons, denom_sides
    );
    rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // pass 2: fold the partials; a FracExt is two FpExt, so each AIR
    // contributes 2 * n_x columns
    auto [ignored_grid, fin_block] = kernel_launch_params(n_blocks / n_airs + 1);
    unsigned int fin_warps = (fin_block.x + WARP_SIZE - 1) / WARP_SIZE;
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);

    dim3 fin_grid(n_airs, 2 * n_x);
    reduce::batched_final_reduce_block_sums<<<fin_grid, fin_block, fin_smem>>>(
        reinterpret_cast<FpExt *>(partials),
        reinterpret_cast<FpExt *>(out),
        air_offsets,
        2 * n_x
    );

    return CHECK_KERNEL();
}

} // namespace logup_zerocheck_mle

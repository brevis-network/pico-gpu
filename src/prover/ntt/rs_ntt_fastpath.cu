// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

/*
 * RS-encode NTT fast path — KoalaBear build unit only (cuda_kb_all).
 *
 * rs_encode_block_in_place (stacked_pcs.rs) runs, per column-block:
 *
 *     batch_intt_zeta_fused  ->  bit_rev(2^24)  ->  3x kb_ct_mixed_radix_narrow
 *
 * which is DRAM-bandwidth bound and moves the (half-zero) codeword buffer
 * through memory five times. This file implements the fast path that cuts the
 * traffic roughly in half (14.9ms -> 7.8ms for the standalone encode on a 5090,
 * bit-exact output):
 *
 *  1. The fused iNTT+zeta launcher (batch_ntt_small.cu, CUDA_KB_RS_FASTPATH
 *     block) writes its result directly to a *dense bit-reversed* scratch
 *     instead of the buffer: since rows [2^23, 2^24) are the RS blowup zeros,
 *     bit_rev(2^24) would put all real data at even rows — out[2q] =
 *     in[rev23(q)], out[2q+1] = 0 — so only the dense even half D is stored.
 *     It sets `kb_rs_dense_prepared`.
 *  2. _kb_bit_rev (ntt_bitrev_kb.cu) sees `prepared`, flips it to `pending`
 *     and returns without touching memory.
 *  3. The kb_ct_mixed_radix_narrow launcher (cuda/supra/ntt.cu, hook gated on
 *     CUDA_KB_RS_FASTPATH) forwards to _kb_rs_ntt_fastpath_dispatch below:
 *     the three (stage=0/8/16, radix-8) launches become TWO custom passes of
 *     a textbook DIT NTT (bit-reversed input, natural output, level-l twiddle
 *     w_{2^{l+1}}^{i mod 2^l}, identical values to the sppark chain):
 *       pass A: levels 0..13 — contiguous 2^14 segments, reads the dense
 *               scratch (implicit zero partners; level 0 duplicates);
 *       pass B: levels 14..23 — 2^10-point NTTs over elements strided 2^14,
 *               the group index g contributing a per-level correction factor
 *               w_{2^{l+1}}^g derived by repeated squaring of w_{2^24}^g;
 *     and the stage=16 call is a no-op.
 *  4. Streamed-open gather (WHIR round 0): `rs_code_streamed_open` used
 *     to re-run this whole pipeline per column-block only to read back ~3k
 *     queried rows (0.02% of the codeword). `_kb_rs_open_pass_a` consumes the
 *     same dense handshake but stops after pass A; `_kb_rs_open_gather_rows`
 *     then evaluates exactly the queried rows as 1024-term dot products over
 *     their pass-A fibers (see `_rs2_open_dot` for the twiddle derivation),
 *     skipping pass B entirely. Bit-exact: pass B never mixes different
 *     low-14-bit offsets, so the dot product reproduces its output values.
 *     D-full additionally prunes pass A's own output: only the <= n_queries
 *     distinct residues `q mod 2^14` are ever read, so the pruned pass A
 *     (`_rs2_ntt_pass_a<true>`) writes just those fibers to a compact buffer
 *     (~1.2% of the codeword) instead of the whole block.
 *
 * Thread safety: prover worker threads run this pipeline concurrently (one
 * stream each via --default-stream=per-thread), so all handshake state is
 * thread_local and the scratch is a per-pipeline stream-ordered allocation
 * (cudaMallocAsync), freed right after pass A consumes it. Twiddle tables are
 * per-device, built once under a mutex.
 *
 * Opt-out: RS_DENSE=0 disables the whole fast path (fully legacy pipeline).
 * C1_VERIFY also disables it, because that flag compares the fused kernel's
 * buffer output against the staged pipeline, and the fast path no longer
 * writes the buffer at that point.
 */

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "common/launcher.cuh"
#include "ntt/ntt.cuh"
#include "common/oom_heal.cuh"

// ── pipeline handshake state (shared with batch_ntt_small.cu / ntt_bitrev_kb.cu)
thread_local fr_t* kb_rs_dense_scratch = nullptr;
thread_local bool kb_rs_dense_prepared = false;
thread_local bool kb_rs_dense_pending = false;
// 0 idle, 1 after pass A, 2 after pass B, 3 tiled pipeline done (stage 8/16
// pending no-ops), 4 tiled stage-8 seen.
static thread_local int rs2_state = 0;

// ── deferred-K1 handshake (L2-tiled scheduling layer) ────────────────────────
// batch_ntt_small.cu records the fused-iNTT (K1) launch here instead of running
// it eagerly; the stage-0 dispatch below replays [K1 ; pass A ; pass B] per
// column group on a software pipeline so each column's intermediates stay
// L2-resident. thread_local: prover workers run this concurrently under
// --default-stream=per-thread, so each thread owns its own handshake.
thread_local void* kb_rs_defer_buffer = nullptr;
thread_local uint32_t kb_rs_defer_chunks = 0;
thread_local uint32_t kb_rs_defer_real_chunks = 0;
thread_local uint32_t kb_rs_defer_lg_real = 0;
thread_local uint32_t kb_rs_defer_width = 0;
thread_local bool kb_rs_defer_pending = false;

// Deferred-K1 launchers (batch_ntt_small.cu): the full group and a block-range
// sub-launch used to split K1 across the previous group's B1/B2 windows.
extern "C" int kb_rs_launch_dense_group(
    const void* buf, void* dense, uint32_t chunks_per_col,
    uint32_t real_chunks_per_col, uint32_t lg_real, uint32_t ncols,
    cudaStream_t stream);
extern "C" int kb_rs_launch_dense_group_range(
    const void* buf, void* dense, uint32_t chunks_per_col,
    uint32_t real_chunks_per_col, uint32_t lg_real, uint32_t ncols,
    uint32_t blk_off, uint32_t nblk, cudaStream_t stream);

bool kb_rs_fastpath_enabled()
{
    static const bool enabled = [] {
        const char* e = getenv("RS_DENSE");
        if (e != nullptr && e[0] == '0')
            return false;
        if (getenv("C1_VERIFY") != nullptr)
            return false;
        // The per-pipeline scratch below allocates from the device default pool,
        // which must run with a raised release threshold or every alloc/free pair
        // round-trips physical memory. That is set per device by the Rust side's
        // ensure_default_pool_retains (memory_manager.rs) before the first
        // allocation lands on a card — NOT here: this is a function-local static,
        // so it would run once per process against whichever card was current and
        // leave the other N-1 unretained.
        return true;
    }();
    return enabled;
}

// ── kernels ──────────────────────────────────────────────────────────────────

__global__ void _rs2_powers(fr_t* out, fr_t base, uint32_t n)
{
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n)
        out[tid] = base^tid;
}

// Shoup double-width twiddle table for pass A. For each w14 entry store the
// PLAIN-domain twiddle w (= Montgomery form x R^{-1}, via operator uint32_t)
// next to its Shoup quotient w' = floor(w * 2^32 / p). Multiplying a
// Montgomery-form value y*R by the plain constant w via Shoup yields (w*y)*R
// mod p — the Montgomery form of the product, bit-identical to the Montgomery
// product of the Montgomery-form twiddle with y. Every pass-A twiddle index
// used by levels 1..5 is a multiple of 32 (index = j << (13-l), l <= 5), so
// out[k] mirrors w14[k << 5]; 256 entries = 2 KiB.
__global__ void _rs2_shoup_fill(uint2* out, const fr_t* w14, uint32_t n)
{
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n)
        return;
    uint32_t w = (uint32_t)w14[tid << 5]; // operator uint32_t == plain domain
    uint2 e;
    e.x = w;
    e.y = (uint32_t)(((uint64_t)w << 32) / fr_t::MOD); // exact integer floor
    out[tid] = e;
}

// Constant-memory mirror of the pass-A Shoup table. Every access is
// warp-uniform with a compile-time index (the ideal constant-cache broadcast
// pattern), keeping these loads off the L1/LSU path that the per-lane w14
// twiddle loads (levels 6..13) contend on.
__constant__ uint2 rs2_w14sh_c[1u << 8];

// Shoup modular mul by a precomputed constant twiddle (tw.x = plain w,
// tw.y = Shoup quotient w'). q = hi32(x*w') underestimates floor(x*w/p) by at
// most 1, so z = x*w - q*p lands in [0, 2p); one conditional subtract returns
// the canonical residue. 3 mul-class instructions / 3-deep dependency chain vs
// Montgomery's 5 instructions / 4-deep chain.
__device__ __forceinline__ fr_t rs2_shoup_mul(uint2 tw, fr_t y)
{
    const uint32_t x = reinterpret_cast<const uint32_t&>(y);
    uint32_t q = __umulhi(x, tw.y);
    uint32_t z = x * tw.x - q * fr_t::MOD;
    if (z >= fr_t::MOD)
        z -= fr_t::MOD;
    return fr_t{z};
}

// F table for pass B split: ftab[b][g] = w_{2^{15+b}}^g = w24g[g]^(2^(9-b)),
// g < 2^14, b < 10. Replaces the per-thread repeated-squaring chain with one
// warp-uniform cached load per level.
__global__ void _rs2_ftab_fill(fr_t* ftab, const fr_t* w24g, uint32_t n)
{
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n)
        return;
    fr_t f = w24g[tid];
    ftab[9 * n + tid] = f;
    for (int b = 9; b > 0; b--) {
        f.sqr();
        ftab[(b - 1) * n + tid] = f;
    }
}

// Pass A: one 2^14-element segment per block, 512 threads x 32 elements.
// Phase 1 owns 32 contiguous elements (levels 0..4 in registers, 5..8 via
// warp shuffles over 1024-element warp spans); a swizzled shared-memory
// transpose then hands each thread 32 elements strided 512 (levels 9..13).
//
// PRUNED (D-full, streamed open only): identical arithmetic, but instead of
// writing the whole 2^14-element segment to d_inout, emit only the n_res
// low-14-bit residues listed in d_res (the distinct `q mod 2^14` of the round-0
// queries, <= num_queries ~ 1.2% of the segment) into the compact buffer
// d_compact, laid out [col][residue][segment] so each gather-dot block later
// reads one contiguous 1024-element fiber. This prunes pass A's full-codeword
// DRAM write, the last remaining output-sized pass of the open pipeline.
template<bool PRUNED>
__launch_bounds__(512) __global__
void _rs2_ntt_pass_a(fr_t* __restrict__ d_inout, const fr_t* __restrict__ d_dense,
                     const fr_t* __restrict__ w14,
                     fr_t* __restrict__ d_compact, const uint32_t* __restrict__ d_res,
                     const uint32_t n_res,
                     const uint32_t padded_poly_size, const uint32_t poly_count)
{
    // Reverse segment order for LRU-friendly dense reads (the freshest, still
    // L2-resident dense words sit at the high-seg end). The output is written
    // by absolute segment index, so this block reordering is bit-exact for both
    // the full and the pruned (compact, seg-indexed) output paths.
    const uint32_t seg = gridDim.x - 1 - blockIdx.x;
    const uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= poly_count)
        return;
    const uint32_t t = threadIdx.x;
    // Dynamic shared layout: 16384 + 512 transpose/swizzle words, then a
    // separate 8192 + 256 twiddle-cache region for levels 9..13 (see below).
    extern __shared__ fr_t rs2_sm[];
    fr_t* tw_sm = rs2_sm + 16896;

    const fr_t* D = d_dense + (size_t)col * (padded_poly_size >> 1) + ((size_t)seg << 13);
    fr_t y[32];
    {
        const uint4* s4 = reinterpret_cast<const uint4*>(D + ((size_t)t << 4));
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint4 v = s4[k];
            y[8*k+0] = y[8*k+1] = fr_t{v.x};
            y[8*k+2] = y[8*k+3] = fr_t{v.y};
            y[8*k+4] = y[8*k+5] = fr_t{v.z};
            y[8*k+6] = y[8*k+7] = fr_t{v.w};
        }
        // The dense slice is dead after this one read but its lines are dirty
        // in L2 (K1 wrote them) and would write back to DRAM on eviction.
        // Discard them: each thread read its exclusive 64B; lanes t and t^1
        // share one 128B line, so after a warp sync the even lane invalidates
        // the line without writeback.
        __syncwarp();
        if ((t & 1) == 0)
            asm volatile("discard.global.L2 [%0], 128;" ::
                         "l"(D + ((size_t)t << 4)) : "memory");
    }

    // Early twiddle staging (consumed by levels 9..13 after the transpose; the
    // transpose's own __syncthreads publishes these stores). Staging at kernel
    // START overlaps the w14 L2 round trip with the dense load and levels 1..8
    // compute. Swizzled placement k + (k>>5) keeps the stride-2^k reads
    // bank-conflict free.
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t i2 = t + (k << 9);
        tw_sm[i2 + (i2 >> 5)] = w14[i2];
    }

    // levels 1..4: pairs inside this thread's 32 contiguous elements; the
    // twiddle only depends on ii mod 2^l, so group pairs by twiddle. The
    // twiddle index j << (13-l) is a multiple of 32, so serve it from the
    // constant-memory Shoup table (warp-uniform broadcast).
    #pragma unroll
    for (uint32_t l = 1; l <= 4; l++) {
        const uint32_t half = 1u << l;
        #pragma unroll
        for (uint32_t j = 0; j < half; j++) {
            const uint2 tw = rs2_w14sh_c[j << (8 - l)];
            #pragma unroll
            for (uint32_t h = 0; h < (32u >> (l + 1)); h++) {
                const uint32_t ii = (h << (l + 1)) + j;
                fr_t b = rs2_shoup_mul(tw, y[ii + half]);
                y[ii + half] = y[ii] - b;
                y[ii] = y[ii] + b;
            }
        }
    }

    // level 5: partner thread t ^ 1 holds the same slot ii; the twiddle index
    // ii << 8 is warp-uniform and a multiple of 32, so serve it from the
    // constant-memory Shoup table. Levels 6..8 have lane-divergent indices —
    // constant-cache replays lose there — so they stay on w14/L1 (Montgomery).
    {
        const bool up = (t & 1) != 0;
        #pragma unroll
        for (uint32_t ii = 0; ii < 32; ii++) {
            const uint2 tw = rs2_w14sh_c[ii << 3];
            fr_t o = y[ii];
            o.shfl_bfly(1);
            fr_t tb = rs2_shoup_mul(tw, fr_t::csel(y[ii], o, up));
            fr_t a = fr_t::csel(o, y[ii], up);
            y[ii] = fr_t::csel(a - tb, a + tb, up);
        }
    }

    // levels 6..8: partner thread t ^ 2^(l-5) holds the same slot ii
    #pragma unroll
    for (uint32_t l = 6; l <= 8; l++) {
        const uint32_t toff = 1u << (l - 5);
        const bool up = (t & toff) != 0;
        #pragma unroll
        for (uint32_t ii = 0; ii < 32; ii++) {
            fr_t tw = w14[(((t & (toff - 1)) << 5) + ii) << (13 - l)];
            fr_t o = y[ii];
            o.shfl_bfly(toff);
            // both sides compute tw*b with b = the upper element
            fr_t tb = tw * fr_t::csel(y[ii], o, up);
            fr_t a = fr_t::csel(o, y[ii], up);
            y[ii] = fr_t::csel(a - tb, a + tb, up);
        }
    }

    // swizzled transpose: element i lives at sm[i + (i >> 5)]
    #pragma unroll
    for (uint32_t ii = 0; ii < 32; ii++)
        rs2_sm[33 * t + ii] = y[ii];
    __syncthreads();
    #pragma unroll
    for (uint32_t m = 0; m < 32; m++) {
        uint32_t i2 = t + (m << 9);
        y[m] = rs2_sm[i2 + (i2 >> 5)];
    }

    // levels 9..13: pairs on the m (stride-512) axis, grouped by twiddle;
    // twiddles were staged into tw_sm at kernel start.
    #pragma unroll
    for (uint32_t l = 9; l <= 13; l++) {
        const uint32_t hm = 1u << (l - 9);
        #pragma unroll
        for (uint32_t j = 0; j < hm; j++) {
            const uint32_t twi = (t + (j << 9)) << (13 - l);
            const fr_t tw = tw_sm[twi + (twi >> 5)];
            #pragma unroll
            for (uint32_t h = 0; h < (32u >> (l - 8)); h++) {
                const uint32_t m = (h << (l - 8)) + j;
                fr_t b = tw * y[m + hm];
                y[m + hm] = y[m] - b;
                y[m] = y[m] + b;
            }
        }
    }

    if (!PRUNED) {
        fr_t* O = d_inout + (size_t)col * padded_poly_size + ((size_t)seg << 14);
        // evict_last hint: this output is re-read by pass B shortly; bias L2 to
        // keep it so the intermediate is not flushed to DRAM prematurely.
        uint64_t pol;
        asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
        #pragma unroll
        for (uint32_t m = 0; m < 32; m++) {
            uint32_t raw = reinterpret_cast<const uint32_t&>(y[m]);
            asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;" ::
                         "l"(O + t + (m << 9)), "r"(raw), "l"(pol) : "memory");
        }
    } else {
        // Stage the finished segment in shared (reusing rs2_sm with the same
        // j + (j>>5) swizzle as the transpose above — bank-conflict-free for both
        // the strided stores and the residue reads), then emit only the requested
        // residues. y[] cannot be indexed by the runtime residue directly without
        // spilling the register file to local memory.
        __syncthreads(); // transpose reads of rs2_sm are complete on all threads
        #pragma unroll
        for (uint32_t m = 0; m < 32; m++) {
            const uint32_t j = t + (m << 9);
            rs2_sm[j + (j >> 5)] = y[m];
        }
        __syncthreads();
        for (uint32_t gi = t; gi < n_res; gi += 512) {
            const uint32_t j = d_res[gi];
            d_compact[(((size_t)col * n_res + gi) << 10) | seg] = rs2_sm[j + (j >> 5)];
        }
    }
}

// Pass B: levels 14..23. Block = 8 consecutive groups g (two warps each) x
// 1024 strided elements r at column offset g + r*2^14. Level 14+b twiddle is
// F_{14+b}(g) * w_{2^{b+1}}^{r mod 2^b}, with F_23(g) = w_{2^24}^g and
// F_l = F_{l+1}^2.
__launch_bounds__(512) __global__
void _rs2_ntt_pass_b(fr_t* __restrict__ d_inout, const fr_t* __restrict__ w14,
                     const fr_t* __restrict__ w24g,
                     const uint32_t padded_poly_size, const uint32_t poly_count)
{
    const uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= poly_count)
        return;
    const uint32_t gbase = blockIdx.x << 3;
    const uint32_t t = threadIdx.x; // 512 threads: 8 g-slices x 2 warps
    __shared__ fr_t tile[1024 * 9]; // [r][g], row stride 9 keeps reads conflict-free

    fr_t* base = d_inout + (size_t)col * padded_poly_size + gbase;
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        uint32_t e = t + (k << 9);
        uint32_t g = e & 7, r = e >> 3;
        tile[r * 9 + g] = base[(size_t)g + ((size_t)r << 14)];
    }
    __syncthreads();

    const uint32_t wp = t >> 6, q = t & 63; // slice, r-low bits (2 warps/slice)
    fr_t y[16];                             // element m holds r = q + 64*m
    #pragma unroll
    for (uint32_t m = 0; m < 16; m++)
        y[m] = tile[(q + (m << 6)) * 9 + wp];

    // F[b] = w_{2^(15+b)}^g via repeated squaring down from F[9] = w_{2^24}^g
    fr_t F[10];
    F[9] = w24g[gbase + wp];
    #pragma unroll
    for (int k = 8; k >= 0; k--) {
        F[k] = F[k + 1];
        F[k].sqr();
    }

    // levels 14..18 (b = 0..4): cross-lane on the q axis; twiddle is
    // m-invariant, so compute it once per level.
    #pragma unroll
    for (uint32_t b = 0; b <= 4; b++) {
        const uint32_t loff = 1u << b;
        const bool up = (q & loff) != 0;
        const fr_t tw = F[b] * w14[((q & (loff - 1)) << (9 - b)) << 4];
        #pragma unroll
        for (uint32_t m = 0; m < 16; m++) {
            fr_t o = y[m];
            o.shfl_bfly(loff);
            fr_t tb = tw * fr_t::csel(y[m], o, up);
            fr_t a = fr_t::csel(o, y[m], up);
            y[m] = fr_t::csel(a - tb, a + tb, up);
        }
    }

    // level 19 (b = 5): partner is q ^ 32 = the other warp of this slice;
    // exchange the pre-butterfly values through shared memory (reusing tile).
    {
        const bool up = (q & 32) != 0;
        const fr_t tw = F[5] * w14[(q & 31) << 8];
        __syncthreads(); // everyone is done reading tile's load-phase contents
        #pragma unroll
        for (uint32_t m = 0; m < 16; m++)
            tile[(m << 9) + t] = y[m];
        __syncthreads();
        #pragma unroll
        for (uint32_t m = 0; m < 16; m++) {
            fr_t o = tile[(m << 9) + (t ^ 32)];
            fr_t tb = tw * fr_t::csel(y[m], o, up);
            fr_t a = fr_t::csel(o, y[m], up);
            y[m] = fr_t::csel(a - tb, a + tb, up);
        }
    }

    // levels 20..23 (b = 6..9): pairs on the m axis, grouped by twiddle
    #pragma unroll
    for (uint32_t b = 6; b <= 9; b++) {
        const uint32_t hm = 1u << (b - 6);
        #pragma unroll
        for (uint32_t j = 0; j < hm; j++) {
            const fr_t tw = F[b] * w14[((q + (j << 6)) << (9 - b)) << 4];
            #pragma unroll
            for (uint32_t h = 0; h < (8u >> (b - 6)); h++) {
                const uint32_t m = (h << (b - 5)) + j;
                fr_t bb = tw * y[m + hm];
                y[m + hm] = y[m] - bb;
                y[m] = y[m] + bb;
            }
        }
    }

    __syncthreads(); // level-19 exchange reads are done; reuse tile for output
    #pragma unroll
    for (uint32_t m = 0; m < 16; m++)
        tile[(q + (m << 6)) * 9 + wp] = y[m];
    __syncthreads();
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        uint32_t e = t + (k << 9);
        uint32_t g = e & 7, r = e >> 3;
        base[(size_t)g + ((size_t)r << 14)] = tile[r * 9 + g];
    }
}

// ============================================================================
// Split pass B: levels 14..23 as two register-resident kernels (B1: 14..18,
// B2: 19..23). Redesigned thread mapping: a warp's 32 lanes run along g (the
// contiguous axis), so every global access is one full 128-byte transaction,
// and each thread owns 32 r-elements IN REGISTERS (r bits 0..4 for B1 at
// stride 2^14, r bits 5..9 for B2 at stride 2^19) so all butterflies are
// in-lane — no shared memory, no shuffles, and every w14 twiddle load is
// warp-uniform. Both kernels are in-place, so the extra pass costs only
// L2<->SM bandwidth, not DRAM traffic or L2 footprint. RS_BSPLIT=0 falls back
// to the classic single-kernel _rs2_ntt_pass_b above.
// ============================================================================

// B1: levels 14..18 (b = 0..4). Block = 8 warps x 32 g = 256 g, one 32-r tile.
__launch_bounds__(256) __global__
void _rs2_ntt_pass_b1(fr_t* __restrict__ d_inout, const fr_t* __restrict__ w14,
                      const fr_t* __restrict__ ftab,
                      const uint32_t padded_poly_size, const uint32_t poly_count)
{
    const uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= poly_count)
        return;
    const uint32_t gtile = blockIdx.x & 63;  // 64 tiles of 256 g
    // Pass A runs its segments in descending order, so its freshest (still
    // L2-resident) output is at LOW addresses — read those first.
    const uint32_t rtile = blockIdx.x >> 6;  // 32 tiles of 32 r
    const uint32_t g = (gtile << 8) + threadIdx.x;
    const uint32_t r0 = rtile << 5;

    fr_t* base = d_inout + (size_t)col * padded_poly_size + g + ((size_t)r0 << 14);
    fr_t y[32]; // element i holds r = r0 + i
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++)
        y[i] = base[(size_t)i << 14];

    // F(b) for b = 0..4 from ONE ftab load: _rs2_ftab_fill wrote
    // ftab[b-1] = sqr(ftab[b]), so squaring F(4) down reproduces F(3)..F(0)
    // bit-exactly and removes 4 of the 5 per-thread ftab loads.
    fr_t F[5];
    F[4] = ftab[((size_t)4 << 14) + g]; // lane-coalesced 128B
    #pragma unroll
    for (int fb = 4; fb > 0; fb--) {
        F[fb - 1] = F[fb];
        F[fb - 1].sqr();
    }

    #pragma unroll
    for (uint32_t b = 0; b <= 4; b++) {
        const uint32_t half = 1u << b;
        const fr_t Fb = F[b];
        #pragma unroll
        for (uint32_t j = 0; j < half; j++) {
            // r mod 2^b = j: warp-uniform w14 broadcast
            const fr_t tw = Fb * w14[(j << (9 - b)) << 4];
            #pragma unroll
            for (uint32_t h = 0; h < (32u >> (b + 1)); h++) {
                const uint32_t i = (h << (b + 1)) + j;
                fr_t bb = tw * y[i + half];
                y[i + half] = y[i] - bb;
                y[i] = y[i] + bb;
            }
        }
    }

    // evict_last: B2 re-reads this output shortly; keep it in L2.
    uint64_t pol;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    uint32_t* gdst = reinterpret_cast<uint32_t*>(base);
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        uint32_t raw = reinterpret_cast<const uint32_t&>(y[i]);
        asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;" ::
                     "l"(gdst + ((size_t)i << 14)), "r"(raw), "l"(pol) : "memory");
    }
}

// B2: levels 19..23 (b = 5..9). Same shape; the in-lane axis is r bits 5..9
// (stride 2^19 elements), q0 = the tile's fixed r bits 0..4.
__launch_bounds__(256) __global__
void _rs2_ntt_pass_b2(fr_t* __restrict__ d_inout, const fr_t* __restrict__ w14,
                      const fr_t* __restrict__ ftab,
                      const uint32_t padded_poly_size, const uint32_t poly_count)
{
    const uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= poly_count)
        return;
    const uint32_t gtile = blockIdx.x & 63;
    const uint32_t q0 = blockIdx.x >> 6; // r bits 0..4
    const uint32_t g = (gtile << 8) + threadIdx.x;

    fr_t* base = d_inout + (size_t)col * padded_poly_size + g + ((size_t)q0 << 14);
    fr_t y[32]; // element j holds r = q0 + 32*j
    #pragma unroll
    for (uint32_t j = 0; j < 32; j++)
        y[j] = base[(size_t)j << 19];

    // B1's F-squaring trick loses on B2 (its 4-deep dependent squaring chain
    // costs more than the loads it would replace), so load F(b) per level.
    #pragma unroll
    for (uint32_t b = 5; b <= 9; b++) {
        const uint32_t hm = 1u << (b - 5);
        const fr_t Fb = ftab[((size_t)b << 14) + g];
        #pragma unroll
        for (uint32_t jm = 0; jm < hm; jm++) {
            // r mod 2^b = q0 + 32*jm: warp-uniform w14 broadcast
            const fr_t tw = Fb * w14[((q0 + (jm << 5)) << (9 - b)) << 4];
            #pragma unroll
            for (uint32_t h = 0; h < (32u >> (b - 4)); h++) {
                const uint32_t j = (h << (b - 4)) + jm;
                fr_t bb = tw * y[j + hm];
                y[j + hm] = y[j] - bb;
                y[j] = y[j] + bb;
            }
        }
    }

    // Final stores: evict-first, the output is not re-read this run.
    uint32_t* gdst = reinterpret_cast<uint32_t*>(base);
    #pragma unroll
    for (uint32_t j = 0; j < 32; j++) {
        fr_t o = y[j];
        __stcs(gdst + ((size_t)j << 19), reinterpret_cast<const uint32_t&>(o));
    }
}

// Streamed-open gather-dot (D-lite): computes the WHIR round-0 opened codeword
// rows directly from the pass-A intermediate, replacing pass B (which would finish
// all 2^24 rows only for ~3k of them to be read back).
//
// Math — why one 1024-term dot product per row is exact: pass B is levels 14..23 of
// the global DIT NTT (bit-reversed input, natural-order output). Its butterflies at
// level 14+b pair memory indices i and i + 2^{14+b}, which share g = i mod 2^14, so
// pass B never mixes different low-14-bit offsets: output row n = g + s*2^14
// (s < 2^10) depends only on the 1024-element fiber A_g[r] = passA[g + r*2^14].
// Unrolling the butterfly network along the path from input position r to output
// position s: at level 14+b the twiddle at fiber index i = g + r*2^14 is
//   w_{2^{15+b}}^{i mod 2^{14+b}} = w_{2^{15+b}}^g * w_{2^{b+1}}^{r mod 2^b}
// (exactly pass B's F[b] * w14[...] schedule); it multiplies the input whose r has
// bit b set, and the butterfly output with s-bit-b = 1 takes the minus sign
// (-1 = w_{2^{b+1}}^{2^b}). Collecting over the set bits b of r:
//   prod_b [ w_{2^{15+b}}^g * w_{2^{b+1}}^{s mod 2^{b+1}} ]
//     = prod_b w_{2^24}^{(g + s*2^14) * 2^{9-b}}  =  w_{2^24}^{n * rev10(r)} .
// Substituting rho = rev10(r) (an involution) gives the closed form
//
//   X[n] = sum_{rho=0}^{1023} passA[g + rev10(rho)*2^14] * w_{2^24}^{n * rho} .
//
// Field ops are exact (fully reduced Montgomery arithmetic), so any evaluation
// order yields the same canonical 32-bit word — bit-exact vs the pass-B output.
//
// Block (qi, col) handles the rows_per_query sibling rows of query qi. Siblings
// differ by multiples of query_stride, which the Rust caller gates to be a multiple
// of 2^14, so they share g and hence one fiber: it is staged in shared memory once
// (in rho-order, i.e. bit-reversed fiber order, swizzled i + (i>>5) so both the
// bit-reversed stores and the sequential reads are bank-conflict free), then each
// sibling row is a dot with the geometric sequence of base = w_{2^24}^n.
//
// PRUNED (D-full): d_src is the compact [col][residue][segment] buffer written by
// the pruned pass A; the fiber of query qi is the contiguous 1024-element row at
// residue slot d_qslot[qi] (coalesced load) instead of the stride-2^14 walk.
template<bool PRUNED>
__launch_bounds__(256) __global__
void _rs2_open_dot(fr_t* __restrict__ d_out, const fr_t* __restrict__ d_src,
                   const uint32_t* __restrict__ d_rows,
                   const uint32_t* __restrict__ d_qslot,
                   const fr_t* __restrict__ w24g, const fr_t* __restrict__ w10,
                   const uint32_t rows_per_query, const uint32_t n_res,
                   const uint32_t padded_poly_size, const uint32_t poly_count)
{
    const uint32_t qi = blockIdx.x;
    const uint32_t col = blockIdx.y + blockIdx.z * gridDim.y;
    if (col >= poly_count)
        return;
    const uint32_t t = threadIdx.x;
    __shared__ fr_t sh[1024 + 32]; // fiber in rho-order, swizzled i + (i >> 5)
    __shared__ fr_t red[256];

    if (PRUNED) {
        const fr_t* S = d_src + (((size_t)col * n_res + d_qslot[qi]) << 10);
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t r = t + (k << 8);      // fiber index (pass-A segment)
            const uint32_t rho = __brev(r) >> 22; // rev10(r)
            sh[rho + (rho >> 5)] = S[r];
        }
    } else {
        const uint32_t g = d_rows[(size_t)qi * rows_per_query] & ((1u << 14) - 1);
        const fr_t* S = d_src + (size_t)col * padded_poly_size + g;
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t r = t + (k << 8);      // fiber index (pass-A segment)
            const uint32_t rho = __brev(r) >> 22; // rev10(r)
            sh[rho + (rho >> 5)] = S[(size_t)r << 14];
        }
    }
    __syncthreads();

    for (uint32_t ro = 0; ro < rows_per_query; ro++) {
        const uint32_t n = d_rows[(size_t)qi * rows_per_query + ro];
        // base = w_{2^24}^n = w_{2^24}^{n mod 2^14} * (w_{2^24}^{2^14})^{n >> 14}
        const fr_t base = w24g[n & ((1u << 14) - 1)] * w10[n >> 14];
        const fr_t step = base ^ 256u;
        fr_t tw = base ^ t; // w_{2^24}^{n * rho} for rho = t + k*256, stepped below
        fr_t acc;
        acc.zero();
        #pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
            const uint32_t rho = t + (k << 8);
            acc += sh[rho + (rho >> 5)] * tw;
            tw = tw * step;
        }
        red[t] = acc;
        __syncthreads();
        for (uint32_t s = 128; s > 0; s >>= 1) {
            if (t < s)
                red[t] = red[t] + red[t + s];
            __syncthreads();
        }
        if (t == 0)
            d_out[((size_t)qi * rows_per_query + ro) * poly_count + col] = red[0];
        // No barrier needed before the next ro: red[0] is only rewritten by t == 0
        // itself, and all other red writes are fenced by the __syncthreads above.
    }
}

// 8-column-coarsened variant of _rs2_open_dot for the PRUNED (D-full) path — the
// production baseline launches one block per (query, col), so every query's
// poly_count blocks each re-gather the same per-column src segment AND re-walk the
// d_rows/bit-rev index math per column, with a serial twiddle chain per (row, col).
// This kernel covers 8 columns per block (grid.y = ceil(poly_count/8), block 512):
// the 8 columns' fibers are staged ONCE into transposed smem — layout [si][c] in two
// uint4 planes so the inner accumulate does 2x LDS.128 conflict-free instead of
// 8x LDS.32 — and one d_rows read + one base = w24g*w10 twiddle per row is reused
// across all 8 column accumulators (tw = base^lane, step = base^32), amortizing the
// pow chain 8x. Same total src bytes, 8x less index/twiddle redundancy, wider ILP.
// Exact field ops (fully reduced Montgomery), so the warp-butterfly re-association
// is byte-equal to the tree reduce. 6.89x standalone (36.5 vs 251 us @ the reth-171
// shape). The non-pruned path keeps the original template above (COLST was only
// validated at production shapes for PRUNED=1).
//
// PRECONDITION, enforced by the caller not by this kernel: the fiber geometry below is hardcoded
// for a 2^24 codeword — 1024 segments, `__brev(rr) >> 22` for the 10-bit reversal, and the
// `shT0[1024 + 32]` tile. `_kb_rs_open_gather_rows` rejects `padded_poly_size != (1u << 24)` before
// reaching here, which is the only reason dropping the `padded_poly_size` parameter is sound. A
// future shape change must revisit all three constants together.
__launch_bounds__(512) __global__
void _rs2_open_dot_pruned_colst(fr_t* __restrict__ d_out, const fr_t* __restrict__ d_src,
                                const uint32_t* __restrict__ d_rows,
                                const uint32_t* __restrict__ d_qslot,
                                const fr_t* __restrict__ w24g, const fr_t* __restrict__ w10,
                                const uint32_t rows_per_query, const uint32_t n_res,
                                const uint32_t poly_count)
{
    constexpr uint32_t C = 8;
    constexpr uint32_t BLK = 512;
    const uint32_t qi = blockIdx.x;
    const uint32_t col0 = (blockIdx.y + blockIdx.z * gridDim.y) * C;
    if (col0 >= poly_count)
        return;
    const uint32_t t = threadIdx.x;
    __shared__ __align__(16) uint32_t shT0[1024 + 32][4]; // cols 0..3 of each si
    __shared__ __align__(16) uint32_t shT1[1024 + 32][4]; // cols 4..7 of each si

    const uint32_t qslot = d_qslot[qi];
    const uint32_t* srcw = reinterpret_cast<const uint32_t*>(d_src);
    #pragma unroll
    for (uint32_t r = 0; r < 1024; r += BLK) {
        const uint32_t rr = r + t;
        const uint32_t rho = __brev(rr) >> 22; // rev10(rr)
        const uint32_t si = rho + (rho >> 5);
        uint32_t w[C];
        #pragma unroll
        for (uint32_t c = 0; c < C; c++) {
            const uint32_t col = col0 + c;
            w[c] = (col < poly_count) ? srcw[(((size_t)col * n_res + qslot) << 10) + rr] : 0u;
        }
        *reinterpret_cast<uint4*>(shT0[si]) = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4*>(shT1[si]) = make_uint4(w[4], w[5], w[6], w[7]);
    }
    __syncthreads();

    const uint32_t warp = t >> 5;
    const uint32_t lane = t & 31;
    const uint32_t nwarp = BLK >> 5;
    for (uint32_t ro = warp; ro < rows_per_query; ro += nwarp) {
        const uint32_t n = d_rows[(size_t)qi * rows_per_query + ro];
        // base = w_{2^24}^n = w_{2^24}^{n mod 2^14} * (w_{2^24}^{2^14})^{n >> 14}
        const fr_t base = w24g[n & ((1u << 14) - 1)] * w10[n >> 14];
        const fr_t step = base ^ 32u;
        fr_t tw = base ^ lane; // w_{2^24}^{n * rho} for rho = 32j + lane, stepped below
        fr_t acc[C];
        #pragma unroll
        for (uint32_t c = 0; c < C; c++)
            acc[c].zero();
        #pragma unroll 8
        for (uint32_t j = 0; j < 32; j++) {
            const uint32_t rho = (j << 5) + lane;
            const uint32_t si = rho + (rho >> 5);
            const uint4 q0 = *reinterpret_cast<const uint4*>(shT0[si]);
            const uint4 q1 = *reinterpret_cast<const uint4*>(shT1[si]);
            fr_t v0, v1, v2, v3, v4, v5, v6, v7;
            *reinterpret_cast<uint32_t*>(&v0) = q0.x;
            *reinterpret_cast<uint32_t*>(&v1) = q0.y;
            *reinterpret_cast<uint32_t*>(&v2) = q0.z;
            *reinterpret_cast<uint32_t*>(&v3) = q0.w;
            *reinterpret_cast<uint32_t*>(&v4) = q1.x;
            *reinterpret_cast<uint32_t*>(&v5) = q1.y;
            *reinterpret_cast<uint32_t*>(&v6) = q1.z;
            *reinterpret_cast<uint32_t*>(&v7) = q1.w;
            acc[0] += v0 * tw;
            acc[1] += v1 * tw;
            acc[2] += v2 * tw;
            acc[3] += v3 * tw;
            acc[4] += v4 * tw;
            acc[5] += v5 * tw;
            acc[6] += v6 * tw;
            acc[7] += v7 * tw;
            tw = tw * step;
        }
        #pragma unroll
        for (uint32_t c = 0; c < C; c++) {
            // intra-warp butterfly (exact field add: re-association is byte-equal)
            #pragma unroll
            for (uint32_t m = 16; m; m >>= 1) {
                fr_t o = acc[c];
                o.shfl_bfly(m);
                acc[c] += o;
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < C; c++) {
                if (col0 + c < poly_count)
                    d_out[((size_t)qi * rows_per_query + ro) * poly_count + col0 + c] = acc[c];
            }
        }
    }
}

// ── per-device twiddle tables ────────────────────────────────────────────────

namespace {
constexpr uint32_t RS2_W14_LEN = 1u << 13;   // w_{2^14}^k, k < 2^13
constexpr uint32_t RS2_W24G_LEN = 1u << 14;  // w_{2^24}^g, g < 2^14
constexpr uint32_t RS2_W10_LEN = 1u << 10;   // w_{2^10}^h, h < 2^10 (streamed-open gather)
constexpr uint32_t RS2_FTAB_LEN = 10u << 14; // ftab[b][g] = w_{2^{15+b}}^g, b < 10 (pass-B split)
constexpr uint32_t RS2_W14SH_N = 1u << 8;    // 256 pass-A Shoup pairs (uint2)
constexpr int RS2_MAX_DEVICES = 64;

// Pass-A dynamic shared: 16384 + 512 transpose/swizzle words + 8192 + 256
// twiddle-cache words (levels 9..13 staged at kernel entry). = 101376 B, the
// sm_120 opt-in max; the same 100KB carveout bucket the 67.6KB version used.
constexpr size_t RS2_PASS_A_SMEM = (size_t)(16384 + 512 + 8192 + 256) * sizeof(fr_t);

struct Rs2DevTables {
    fr_t* w14 = nullptr;
    fr_t* w24g = nullptr;
    fr_t* w10 = nullptr;
    fr_t* ftab = nullptr;
};
std::mutex rs2_mu;
Rs2DevTables rs2_dev[RS2_MAX_DEVICES];
} // namespace

static int rs2_get_tables(fr_t*& w14, fr_t*& w24g, fr_t*& w10, fr_t*& ftab)
{
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= RS2_MAX_DEVICES)
        return cudaErrorInvalidDevice;

    std::lock_guard<std::mutex> lk(rs2_mu);
    if (rs2_dev[dev].w14 != nullptr) {
        w14 = rs2_dev[dev].w14;
        w24g = rs2_dev[dev].w24g;
        w10 = rs2_dev[dev].w10;
        ftab = rs2_dev[dev].ftab;
        return cudaSuccess;
    }
    // Per DEVICE, not per process: cudaFuncSetAttribute applies to the current device's
    // instance of the kernel, so it must run on every card that will launch it. The per-device
    // table miss below happens exactly once per card (mutex-held), which makes this the right
    // spot — a process-wide "already set" flag here once let card 1 launch with the default
    // 48KB dynamic-shmem cap and fail.
    {
        cudaError_t aerr = cudaFuncSetAttribute(_rs2_ntt_pass_a<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, RS2_PASS_A_SMEM);
        if (aerr == cudaSuccess)
            aerr = cudaFuncSetAttribute(_rs2_ntt_pass_a<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, RS2_PASS_A_SMEM);
        if (aerr != cudaSuccess)
            return aerr;
    }
    fr_t* tbl = nullptr;
    if (cudaMalloc(&tbl, (RS2_W14_LEN + RS2_W24G_LEN + RS2_W10_LEN + RS2_FTAB_LEN) *
                             sizeof(fr_t)) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    fr_t* p_w14 = tbl;
    fr_t* p_w24g = tbl + RS2_W14_LEN;
    fr_t* p_w10 = p_w24g + RS2_W24G_LEN;
    fr_t* p_ftab = p_w10 + RS2_W10_LEN;
    _rs2_powers<<<RS2_W14_LEN / 256, 256>>>(p_w14, forward_roots_of_unity[14], RS2_W14_LEN);
    _rs2_powers<<<RS2_W24G_LEN / 256, 256>>>(p_w24g, forward_roots_of_unity[24], RS2_W24G_LEN);
    _rs2_powers<<<RS2_W10_LEN / 256, 256>>>(p_w10, forward_roots_of_unity[10], RS2_W10_LEN);
    // ftab depends on w24g being filled first (same stream ordering).
    _rs2_ftab_fill<<<RS2_W24G_LEN / 256, 256>>>(p_ftab, p_w24g, RS2_W24G_LEN);
    // Pass-A Shoup constant table (per-device __constant__ mirror of w14[k<<5]).
    uint2* d_sh = nullptr;
    if (cudaMalloc(&d_sh, RS2_W14SH_N * sizeof(uint2)) != cudaSuccess) {
        cudaFree(tbl);
        return cudaErrorMemoryAllocation;
    }
    _rs2_shoup_fill<<<1, RS2_W14SH_N>>>(d_sh, p_w14, RS2_W14SH_N);
    cudaError_t cerr = cudaMemcpyToSymbolAsync(rs2_w14sh_c, d_sh,
        RS2_W14SH_N * sizeof(uint2), 0, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    cudaStreamSynchronize(cudaStreamPerThread);
    cudaFree(d_sh);
    int err = CHECK_KERNEL();
    if (err == cudaSuccess && cerr != cudaSuccess)
        err = cerr;
    if (err != cudaSuccess) {
        cudaFree(tbl);
        return err;
    }
    rs2_dev[dev].w14 = p_w14;
    rs2_dev[dev].w24g = p_w24g;
    rs2_dev[dev].w10 = p_w10;
    rs2_dev[dev].ftab = p_ftab;
    w14 = p_w14;
    w24g = p_w24g;
    w10 = p_w10;
    ftab = p_ftab;
    return cudaSuccess;
}

// RS_BSPLIT=0 falls back to the classic single-kernel pass B.
static bool rs_bsplit_enabled()
{
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("RS_BSPLIT");
        v = (e != nullptr && e[0] == '0') ? 0 : 1;
    }
    return v != 0;
}

// RS_TILE = columns per group for the L2-tiled deferred pipeline (default 1 =
// per-column; 0 disables the whole scheduling layer and falls back to the
// eager-K1 non-tiled fast path). One 2^24 column's intermediates (32 MiB dense
// + 64 MiB pass-A output) fit the 5090's 96 MiB L2, so per-column replay keeps
// them cache-resident. A naive SINGLE-stream tiling regresses (per-group launch
// overhead swamps the L2 win); the win requires the full scheduling layer
// below: deferred-K1 + double-stream (K1(grp+1) overlaps pass B(grp)) + CUDA
// graph (strips launch overhead) + K1-split. RS_TILE is read here AND by
// rs_defer_enabled in batch_ntt_small.cu, with the same default (1) but different
// meanings: there it collapses to a boolean gate (`tile > 0`), here the value is
// the column-group width. The two must stay consistent about being on or off, so
// changing the default means changing it in both files.
static uint32_t rs_tile_cols()
{
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("RS_TILE");
        v = (e != nullptr) ? atoi(e) : 1;
        if (v < 0) v = 0;
    }
    return (uint32_t)v;
}

// Concurrent column pipelines (RS_STREAMS, default 2). Per-column grids fill the
// GPU for only ~6 waves, so each kernel's partial last wave wastes bandwidth;
// pipelining K1 of the next group with pass B of the current one (the only
// cross-column overlap that does not grow the L2 working set — K1 needs the
// group scratch, free once A(grp-1) read it, and B never touches it) recovers it.
static uint32_t rs_num_streams()
{
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("RS_STREAMS");
        v = (e != nullptr) ? atoi(e) : 2;
        if (v < 1) v = 1;
        if (v > 2) v = 2;
    }
    return (uint32_t)v;
}

// RS_GRAPH=0 disables the CUDA-graph capture/replay of the tiled pipeline (kept
// as an A/B knob; the graph strips ~100 per-group launch/gap overheads per run
// but re-captures whenever the target buffer/shape changes).
static bool rs_graph_enabled()
{
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("RS_GRAPH");
        v = (e != nullptr && e[0] == '0') ? 0 : 1;
    }
    return v != 0;
}

// RS_K1SPLIT: blocks of K1(grp) to overlap with B1(grp-1) (the rest overlap
// B2(grp-1)). -1 (default) = auto (~11/32 of the grid, the standalone optimum);
// 0 = no split (K1 runs after B1 done, overlapping only B2); >0 = forced count.
static int rs_k1split_blocks()
{
    static int v = -2;
    if (v == -2) {
        const char* e = getenv("RS_K1SPLIT");
        v = (e != nullptr) ? atoi(e) : -1;
        if (v < -1) v = -1;
    }
    return v;
}

// Launch pass B (levels 14..23) for `poly_count` columns on `strm`, either as
// the split B1+B2 register-resident kernels (default) or the classic single
// kernel. Both are in-place and bit-exact; the split is faster (128B stores,
// no shared/shuffle) but needs ftab.
static void rs2_launch_pass_b(fr_t* d_inout, uint32_t grid_y, uint32_t grid_z,
                              fr_t* w14, fr_t* w24g, fr_t* ftab,
                              uint32_t padded_poly_size, uint32_t poly_count,
                              cudaStream_t strm)
{
    if (rs_bsplit_enabled()) {
        _rs2_ntt_pass_b1<<<dim3(1u << 11, grid_y, grid_z), 256, 0, strm>>>(
            d_inout, w14, ftab, padded_poly_size, poly_count);
        _rs2_ntt_pass_b2<<<dim3(1u << 11, grid_y, grid_z), 256, 0, strm>>>(
            d_inout, w14, ftab, padded_poly_size, poly_count);
    } else {
        _rs2_ntt_pass_b<<<dim3(1u << 11, grid_y, grid_z), 512, 0, strm>>>(
            d_inout, w14, w24g, padded_poly_size, poly_count);
    }
}

// ── L2-tiled deferred scheduling layer (thread_local / per-thread) ───────────
// Every mutable scheduling resource — the two side streams, the pipeline events, the persistent
// group-sized dense scratch, and the captured graph — is thread_local, so each prover worker thread
// runs a fully independent pipeline with no cross-thread shared mutable state. The only shared
// state is the per-device twiddle tables (mutex-guarded above).

static thread_local cudaStream_t rs2_streams[2] = {nullptr, nullptr};
static thread_local cudaEvent_t rs2_stream_done[2] = {nullptr, nullptr};
static thread_local cudaEvent_t rs2_k1ev[2] = {nullptr, nullptr};   // K1(grp) done
static thread_local cudaEvent_t rs2_aev[2] = {nullptr, nullptr};    // B1(grp) done
static thread_local cudaEvent_t rs2_postA_ev[2] = {nullptr, nullptr}; // A(grp) done
static thread_local cudaEvent_t rs2_fork_ev = nullptr;
static thread_local uint32_t rs2_streams_ready = 0;

// Persistent per-thread group-sized dense scratch (K1 writes it, pass A reads
// it). A plain cudaMalloc (not cudaMallocAsync) so it survives graph replays and
// is never allocated/freed inside a capture; reused across groups so it stays
// L2-resident. Distinct from kb_rs_dense_scratch (the eager/open path's
// stream-ordered allocation).
static thread_local fr_t* rs2_tile_scratch = nullptr;
static thread_local size_t rs2_tile_scratch_cap = 0;

// Per-thread captured-graph cache, keyed on target buffer + shape.
static thread_local cudaGraphExec_t rs2_graph_exec = nullptr;
static thread_local void* rs2_graph_buf = nullptr;
static thread_local uint32_t rs2_graph_w = 0;
static thread_local uint32_t rs2_graph_pad = 0;

// ── per-thread state, recycled across threads ────────────────────────────────
// Everything above is per THREAD, and this process starts a fresh set of prover threads for every
// block it proves. A raw `thread_local` has no destructor, so each block's threads left their
// streams, events, captured graph and 32 MiB of scratch behind — memory outside every memory pool,
// which no `cudaMemPoolTrimTo` can reach, growing until the card refused the next allocation.
//
// The state is handed to the next thread rather than destroyed. Destroying would work, but
// `cudaFree` synchronises the whole device and runs here while other lanes are still proving,
// and re-allocating 32 MiB on the next block costs more than passing it along. What the process
// holds is then bounded by how many threads are alive at once, not by how many it has spawned.

struct Rs2ThreadState {
    cudaStream_t streams[2];
    cudaEvent_t stream_done[2];
    cudaEvent_t k1ev[2];
    cudaEvent_t aev[2];
    cudaEvent_t postA_ev[2];
    cudaEvent_t fork_ev;
    uint32_t streams_ready;
    fr_t* tile_scratch;
    size_t tile_scratch_cap;
};

// Never destroyed, deliberately: a thread can end after the static-destruction phase has begun,
// and the hand-back below must not reach into an object that is already gone. The `atexit` flag
// stops the CUDA calls; this stops the container from being the hazard.
static std::mutex& rs2_idle_mtx() {
    static std::mutex* m = new std::mutex();
    return *m;
}

// Per device: the scratch belongs to the card that allocated it, and a stream to the card's
// context. Sized like the other per-device tables in this build unit.
static std::vector<Rs2ThreadState>* rs2_idle_for(int dev) {
    static std::vector<Rs2ThreadState>* idle = new std::vector<Rs2ThreadState>[RS2_MAX_DEVICES];
    if (dev < 0 || dev >= RS2_MAX_DEVICES)
        return nullptr;
    return &idle[dev];
}

// Set from an `atexit` handler, so a thread ending late in teardown skips the CUDA calls. It is
// only half the guard: glibc runs `__call_tls_dtors()` at the top of `exit()`, BEFORE atexit
// handlers, so the main thread's hand-back still runs with the flag clear. What makes that safe is
// that the free list and its mutex above are never destroyed.
static std::atomic<bool> rs2_shutting_down{false};

// The card this thread's state belongs to, recorded when it is taken or first built. Read at
// hand-back instead of asking the driver again, so a thread that rebound in between cannot file
// its scratch under a card whose context cannot use it.
static thread_local int rs2_state_device = -1;

static void rs2_release_thread_state();

// Registers the hand-back for this thread. A `thread_local` with a destructor only runs it if the
// object is actually used, which is what `arm()` is for.
struct Rs2ThreadGuard {
    bool armed = false;
    void arm() { armed = true; }
    ~Rs2ThreadGuard() {
        if (armed)
            rs2_release_thread_state();
    }
};
static thread_local Rs2ThreadGuard rs2_guard;

// Take an idle thread's state if this card has one. Called before the lazy `rs_ensure_*` paths, so
// a recycled buffer satisfies them and they allocate nothing.
static void rs2_acquire_thread_state() {
    if (rs2_streams_ready != 0 || rs2_tile_scratch != nullptr)
        return; // this thread is already equipped
    rs2_guard.arm();
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit([] { rs2_shutting_down.store(true, std::memory_order_relaxed); });
    });
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return;
    rs2_state_device = dev;
    std::lock_guard<std::mutex> lock(rs2_idle_mtx());
    std::vector<Rs2ThreadState>* idle = rs2_idle_for(dev);
    if (idle == nullptr || idle->empty())
        return;
    const Rs2ThreadState st = idle->back();
    idle->pop_back();
    for (int i = 0; i < 2; i++) {
        rs2_streams[i] = st.streams[i];
        rs2_stream_done[i] = st.stream_done[i];
        rs2_k1ev[i] = st.k1ev[i];
        rs2_aev[i] = st.aev[i];
        rs2_postA_ev[i] = st.postA_ev[i];
    }
    rs2_fork_ev = st.fork_ev;
    rs2_streams_ready = st.streams_ready;
    rs2_tile_scratch = st.tile_scratch;
    rs2_tile_scratch_cap = st.tile_scratch_cap;
}

static void rs2_release_thread_state() {
    if (rs2_shutting_down.load(std::memory_order_relaxed))
        return;
    if (rs2_streams_ready == 0 && rs2_tile_scratch == nullptr)
        return;
    // Drained by this thread before anything moves. Every side stream is waited on by name, NOT
    // just the per-thread stream they join back into: that join is the last thing `rs2_tiled_body`
    // does, so an error return from the middle of the pipeline skips it and leaves K1/passA/passB
    // still running against the scratch this function is about to hand to another thread.
    // Normally all of these return at once.
    cudaStreamSynchronize(cudaStreamPerThread);
    for (uint32_t s = 0; s < rs2_streams_ready; s++) {
        if (rs2_streams[s] != nullptr)
            cudaStreamSynchronize(rs2_streams[s]);
    }
    // The exec bakes in this thread's scratch pointer, and the next block re-captures anyway
    // because the data buffer moves (see the staleness check in `rs2_run_tiled`), so destroying it
    // here costs nothing and keeps a captured graph on the thread that captured it.
    if (rs2_graph_exec != nullptr) {
        cudaGraphExecDestroy(rs2_graph_exec);
        rs2_graph_exec = nullptr;
    }
    rs2_graph_buf = nullptr;
    rs2_graph_w = 0;
    rs2_graph_pad = 0;
    Rs2ThreadState st;
    for (int i = 0; i < 2; i++) {
        st.streams[i] = rs2_streams[i];
        st.stream_done[i] = rs2_stream_done[i];
        st.k1ev[i] = rs2_k1ev[i];
        st.aev[i] = rs2_aev[i];
        st.postA_ev[i] = rs2_postA_ev[i];
        rs2_streams[i] = nullptr;
        rs2_stream_done[i] = nullptr;
        rs2_k1ev[i] = nullptr;
        rs2_aev[i] = nullptr;
        rs2_postA_ev[i] = nullptr;
    }
    st.fork_ev = rs2_fork_ev;
    st.streams_ready = rs2_streams_ready;
    st.tile_scratch = rs2_tile_scratch;
    st.tile_scratch_cap = rs2_tile_scratch_cap;
    rs2_fork_ev = nullptr;
    rs2_streams_ready = 0;
    rs2_tile_scratch = nullptr;
    rs2_tile_scratch_cap = 0;
    // Normally recorded when the state was taken or first built; asking the driver is the fallback
    // for a thread whose `cudaGetDevice` failed back then, so a 32 MiB scratch is not dropped on
    // the floor just because one query did not answer.
    int dev = rs2_state_device;
    if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess)
        return;
    rs2_state_device = -1;
    std::lock_guard<std::mutex> lock(rs2_idle_mtx());
    std::vector<Rs2ThreadState>* idle = rs2_idle_for(dev);
    if (idle != nullptr)
        idle->push_back(st);
}

// Return the real CUDA code rather than flattening every failure to cudaErrorUnknown: on the
// graph path the code is the diagnosis, and StreamCaptureUnsupported / Invalidated / Implicit
// each name a different collision with the other prover lane.
#define RS2_TRY(expr)                                    \
    do {                                                 \
        cudaError_t _rs2_e = (expr);                     \
        if (_rs2_e != cudaSuccess) return _rs2_e;        \
    } while (0)

static int rs_ensure_streams(uint32_t n)
{
    if (rs2_fork_ev == nullptr)
        RS2_TRY(cudaEventCreateWithFlags(&rs2_fork_ev, cudaEventDisableTiming));
    for (uint32_t s = rs2_streams_ready; s < n; s++) {
        RS2_TRY(cudaStreamCreateWithFlags(&rs2_streams[s], cudaStreamNonBlocking));
        RS2_TRY(cudaEventCreateWithFlags(&rs2_stream_done[s], cudaEventDisableTiming));
        RS2_TRY(cudaEventCreateWithFlags(&rs2_k1ev[s], cudaEventDisableTiming));
        RS2_TRY(cudaEventCreateWithFlags(&rs2_aev[s], cudaEventDisableTiming));
        RS2_TRY(cudaEventCreateWithFlags(&rs2_postA_ev[s], cudaEventDisableTiming));
        rs2_streams_ready = s + 1;
    }
    return cudaSuccess;
}

static int rs_ensure_tile_scratch(size_t need)
{
    if (rs2_tile_scratch_cap >= need)
        return cudaSuccess;
    if (rs2_tile_scratch != nullptr)
        cudaFree(rs2_tile_scratch);
    rs2_tile_scratch = nullptr;
    rs2_tile_scratch_cap = 0;
    // A realloc moves the pointer captured graphs baked in; invalidate them.
    if (rs2_graph_exec != nullptr) {
        cudaGraphExecDestroy(rs2_graph_exec);
        rs2_graph_exec = nullptr;
    }
    if (cudaMalloc(&rs2_tile_scratch, need * sizeof(fr_t)) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    rs2_tile_scratch_cap = need;
    return cudaSuccess;
}

// The launch body (fork side streams + per-group software pipeline + join back
// into the caller's stream). Factored so it can run directly or under graph
// capture with an identical instruction stream. All streams/events/scratch are
// pre-created by the caller; no host-blocking or allocation happens here, so it
// is capture-safe.
static int rs2_tiled_body(fr_t* d_inout, uint32_t poly_count, uint32_t padded_poly_size,
                          fr_t* w14, fr_t* w24g, fr_t* ftab, uint32_t G, uint32_t nstreams)
{
    const uint32_t ngroups = (poly_count + G - 1) / G;
    const bool pipelined = nstreams > 1;
    cudaStream_t strmAB = rs2_streams[0];
    cudaStream_t strmK = rs2_streams[pipelined ? 1 : 0];

    // Fork: the side streams must not begin before work already queued on the
    // caller's stream (the producer of d_inout) has been reached.
    RS2_TRY(cudaEventRecord(rs2_fork_ev, cudaStreamPerThread));
    for (uint32_t s = 0; s < nstreams; s++)
        RS2_TRY(cudaStreamWaitEvent(rs2_streams[s], rs2_fork_ev, 0));

    const uint32_t grid_x = (kb_rs_defer_real_chunks + 255) / 256;
    const int k1s = rs_k1split_blocks();
    uint32_t h1cfg = (k1s < 0) ? (grid_x * 11u / 32u) : (uint32_t)k1s;

    for (uint32_t grp = 0; grp < ngroups; grp++) {
        const uint32_t c0 = grp * G;
        const uint32_t g = (poly_count - c0 < G) ? (poly_count - c0) : G;
        fr_t* colbase = d_inout + (size_t)c0 * padded_poly_size;

        // ── K1(grp): fill the shared group scratch from the input columns ──
        // The scratch is one group wide and reused, so K1(grp) may only
        // overwrite it once A(grp-1) has finished reading (postA), and its
        // output is consumed by A(grp) (k1ev). K1(grp) therefore overlaps
        // B(grp-1) — the one overlap that does not grow the L2 footprint.
        int e;
        uint32_t h1 = h1cfg;
        if (pipelined && h1 > 0 && grp > 0) {
            if (h1 > grid_x) h1 = grid_x;
            const uint32_t h2 = grid_x - h1;
            // First h1 blocks after A(grp-1): overlaps B1(grp-1) (L2-bound,
            // DRAM idle), soaking up K1's cold input read there.
            RS2_TRY(cudaStreamWaitEvent(strmK, rs2_postA_ev[(grp - 1) & 1], 0));
            e = kb_rs_launch_dense_group_range(colbase, rs2_tile_scratch,
                    kb_rs_defer_chunks, kb_rs_defer_real_chunks, kb_rs_defer_lg_real,
                    g, 0, h1, strmK);
            if (e != cudaSuccess) return e;
            if (h2 > 0) {
                RS2_TRY(cudaStreamWaitEvent(strmK, rs2_aev[(grp - 1) & 1], 0));
                e = kb_rs_launch_dense_group_range(colbase, rs2_tile_scratch,
                        kb_rs_defer_chunks, kb_rs_defer_real_chunks, kb_rs_defer_lg_real,
                        g, h1, h2, strmK);
                if (e != cudaSuccess) return e;
            }
        } else {
            if (pipelined && grp > 0)
                RS2_TRY(cudaStreamWaitEvent(strmK, rs2_aev[(grp - 1) & 1], 0));
            e = kb_rs_launch_dense_group(colbase, rs2_tile_scratch,
                    kb_rs_defer_chunks, kb_rs_defer_real_chunks, kb_rs_defer_lg_real,
                    g, strmK);
            if (e != cudaSuccess) return e;
        }
        if (pipelined) {
            RS2_TRY(cudaEventRecord(rs2_k1ev[grp & 1], strmK));
            RS2_TRY(cudaStreamWaitEvent(strmAB, rs2_k1ev[grp & 1], 0));
        }

        // ── pass A(grp): levels 0..13, reads scratch, writes colbase ──
        _rs2_ntt_pass_a<false><<<dim3(1u << 10, g, 1), 512, RS2_PASS_A_SMEM, strmAB>>>(
            colbase, rs2_tile_scratch, w14, nullptr, nullptr, 0, padded_poly_size, g);
        if (pipelined)
            RS2_TRY(cudaEventRecord(rs2_postA_ev[grp & 1], strmAB));

        // ── pass B(grp): levels 14..23, in place; aev fires after B1 ──
        if (rs_bsplit_enabled()) {
            _rs2_ntt_pass_b1<<<dim3(1u << 11, g, 1), 256, 0, strmAB>>>(
                colbase, w14, ftab, padded_poly_size, g);
            if (pipelined)
                RS2_TRY(cudaEventRecord(rs2_aev[grp & 1], strmAB));
            _rs2_ntt_pass_b2<<<dim3(1u << 11, g, 1), 256, 0, strmAB>>>(
                colbase, w14, ftab, padded_poly_size, g);
        } else {
            _rs2_ntt_pass_b<<<dim3(1u << 11, g, 1), 512, 0, strmAB>>>(
                colbase, w14, w24g, padded_poly_size, g);
            if (pipelined)
                RS2_TRY(cudaEventRecord(rs2_aev[grp & 1], strmAB));
        }
    }

    // Join the side streams back so downstream work (and timing) on the caller's
    // stream waits for the whole pipeline.
    for (uint32_t s = 0; s < nstreams; s++) {
        RS2_TRY(cudaEventRecord(rs2_stream_done[s], rs2_streams[s]));
        RS2_TRY(cudaStreamWaitEvent(cudaStreamPerThread, rs2_stream_done[s], 0));
    }
    return cudaSuccess;
}

// Run the tiled pipeline for one full-encode call (all poly_count columns),
// optionally as a captured+replayed CUDA graph. Streams/events/scratch are set
// up here (outside any capture); the graph is cached per (buffer, shape).
static int rs2_run_tiled(fr_t* d_inout, uint32_t poly_count, uint32_t padded_poly_size,
                         fr_t* w14, fr_t* w24g, fr_t* ftab)
{
    rs2_acquire_thread_state();
    uint32_t G = rs_tile_cols();
    if (G == 0 || G > poly_count)
        G = poly_count;
    const uint32_t ngroups = (poly_count + G - 1) / G;
    uint32_t nstreams = rs_num_streams();
    if (nstreams > ngroups)
        nstreams = ngroups;
    int serr = rs_ensure_streams(nstreams);
    if (serr != cudaSuccess)
        return serr;
    const size_t slice = (size_t)G << (kb_rs_defer_lg_real + 4);
    serr = rs_ensure_tile_scratch(slice);
    if (serr != cudaSuccess)
        return serr;

    const bool use_graph = rs_graph_enabled();
    if (use_graph && rs2_graph_exec != nullptr && rs2_graph_buf == (void*)d_inout &&
        rs2_graph_w == poly_count && rs2_graph_pad == padded_poly_size) {
        RS2_TRY(cudaGraphLaunch(rs2_graph_exec, cudaStreamPerThread));
        return CHECK_KERNEL();
    }
    if (rs2_graph_exec != nullptr) { // stale buffer/shape: re-capture
        cudaGraphExecDestroy(rs2_graph_exec);
        rs2_graph_exec = nullptr;
    }
    if (!use_graph)
        return rs2_tiled_body(d_inout, poly_count, padded_poly_size, w14, w24g, ftab,
                              G, nstreams);

    RS2_TRY(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    int berr = rs2_tiled_body(d_inout, poly_count, padded_poly_size, w14, w24g, ftab,
                              G, nstreams);
    cudaGraph_t graph = nullptr;
    cudaError_t eerr = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (berr != cudaSuccess) {
        if (graph != nullptr)
            cudaGraphDestroy(graph);
        return berr;
    }
    if (eerr != cudaSuccess)
        return eerr;
    if (graph == nullptr)
        return cudaErrorUnknown;
    cudaError_t gerr = cudaGraphInstantiate(&rs2_graph_exec, graph, 0);
    cudaGraphDestroy(graph);
    if (gerr != cudaSuccess) {
        rs2_graph_exec = nullptr;
        return gerr;
    }
    rs2_graph_buf = (void*)d_inout;
    rs2_graph_w = poly_count;
    rs2_graph_pad = padded_poly_size;
    RS2_TRY(cudaGraphLaunch(rs2_graph_exec, cudaStreamPerThread));
    return CHECK_KERNEL();
}

#undef RS2_TRY

// ── dispatch, called from kb_ct_mixed_radix_narrow (cuda/supra/ntt.cu) ───────

extern "C" int _kb_rs_ntt_fastpath_dispatch(
    fr_t* d_inout, uint32_t radix, uint32_t lg_domain_size, uint32_t stage,
    uint32_t iterations, uint32_t padded_poly_size, uint32_t poly_count,
    bool is_intt, int* handled)
{
    *handled = 0;
    (void)radix;
    (void)iterations;
    if (is_intt || lg_domain_size != 24 || padded_poly_size != (1u << 24) ||
        poly_count == 0)
        return cudaSuccess;

    const uint32_t MAX_Y = 65535;
    uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
    uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;

    // Deferred-K1 L2-tiled pipeline (RS_TILE>0): K1 recorded its launch instead
    // of running, so replay [K1 ; pass A ; pass B] per column group here, with
    // K1(grp+1) overlapping pass B(grp) on a second stream. Checked before the
    // eager dense_pending path because in tiled mode BOTH flags are set (K1 set
    // prepared -> bit_rev set pending).
    if (stage == 0 && kb_rs_defer_pending) {
        kb_rs_defer_pending = false;
        *handled = 1; // the buffer holds no usable data — must not fall back
        fr_t* w14 = nullptr;
        fr_t* w24g = nullptr;
        fr_t* w10 = nullptr;
        fr_t* ftab = nullptr;
        int terr = rs2_get_tables(w14, w24g, w10, ftab);
        if (terr != cudaSuccess)
            return terr;
        const bool match = kb_rs_dense_pending &&
            (void*)d_inout == kb_rs_defer_buffer &&
            poly_count == kb_rs_defer_width &&
            padded_poly_size == (kb_rs_defer_chunks << 4);
        if (match) {
            kb_rs_dense_pending = false;
            int rerr = rs2_run_tiled(d_inout, poly_count, padded_poly_size, w14, w24g, ftab);
            if (rerr != cudaSuccess)
                return rerr;
            rs2_state = 3; // tiled: pass B is done; stage 8/16 become no-ops
            return cudaSuccess;
        }
        // Unexpected shape (should not happen for the fixed 2^24 commit): fall
        // back to eager. Materialize the full dense half now with the recorded
        // args into the stream-ordered scratch, then drop through to the
        // classic non-tiled dense_pending path (kb_rs_dense_pending stays true).
        size_t need = (size_t)kb_rs_defer_width << (kb_rs_defer_lg_real + 4);
        if (mle_malloc_async_healing(&kb_rs_dense_scratch, need * sizeof(fr_t),
                                     cudaStreamPerThread) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        int lerr = kb_rs_launch_dense_group(kb_rs_defer_buffer, kb_rs_dense_scratch,
            kb_rs_defer_chunks, kb_rs_defer_real_chunks, kb_rs_defer_lg_real,
            kb_rs_defer_width, cudaStreamPerThread);
        if (lerr != cudaSuccess)
            return lerr;
    }
    if (stage == 0 && kb_rs_dense_pending) {
        kb_rs_dense_pending = false;
        *handled = 1; // the buffer holds no usable data — must not fall back
        fr_t* w14 = nullptr;
        fr_t* w24g = nullptr;
        fr_t* w10 = nullptr;
        fr_t* ftab = nullptr;
        int terr = rs2_get_tables(w14, w24g, w10, ftab);
        if (terr != cudaSuccess)
            return terr;
        _rs2_ntt_pass_a<false><<<dim3(1u << 10, grid_y, grid_z), 512, RS2_PASS_A_SMEM>>>(
            d_inout, kb_rs_dense_scratch, w14, nullptr, nullptr, 0,
            padded_poly_size, poly_count);
        // stream-ordered free: pass A is the last consumer of the scratch
        cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
        kb_rs_dense_scratch = nullptr;
        rs2_state = 1;
        return CHECK_KERNEL();
    }
    if (stage == 8 && rs2_state == 1) {
        rs2_state = 2;
        *handled = 1;
        fr_t* w14 = nullptr;
        fr_t* w24g = nullptr;
        fr_t* w10 = nullptr;
        fr_t* ftab = nullptr;
        int terr = rs2_get_tables(w14, w24g, w10, ftab);
        if (terr != cudaSuccess)
            return terr;
        rs2_launch_pass_b(d_inout, grid_y, grid_z, w14, w24g, ftab,
                          padded_poly_size, poly_count, cudaStreamPerThread);
        return CHECK_KERNEL();
    }
    if (stage == 8 && rs2_state == 3) { // tiled loop already ran pass B
        rs2_state = 4;
        *handled = 1;
        return cudaSuccess;
    }
    if (stage == 16 && (rs2_state == 2 || rs2_state == 4)) {
        rs2_state = 0;
        *handled = 1;
        return cudaSuccess;
    }
    return cudaSuccess;
}

// ── streamed-open entry points, called from rs_code_streamed_open (Rust) ─────

// Pass-A-only continuation of the dense handshake: consumes the `pending` state
// (set by _kb_batch_intt_zeta_fused + _kb_bit_rev) and materializes the pass-A
// intermediate WITHOUT running pass B — into d_inout (full segments, D-lite) or,
// when d_compact/d_res are given, only the n_res requested low-14-bit residues
// into the compact [col][residue][segment] buffer (D-full). Does not touch
// rs2_state, so a subsequent unrelated NTT on this thread starts from a clean
// handshake. `*handled = 0` (and no work) when the dense fast path did not
// engage — the caller must then finish the classic pipeline (full big NTT).
extern "C" int _kb_rs_open_pass_a(fr_t* d_inout, fr_t* d_compact,
                                  const uint32_t* d_res, uint32_t n_res,
                                  uint32_t padded_poly_size, uint32_t poly_count,
                                  int* handled)
{
    *handled = 0;
    // Streamed-open stays EAGER: the deferred/tiled commit pipeline is bypassed
    // here. If K1 deferred (RS_TILE>0), materialize the FULL dense half now —
    // exactly what an eager K1 would have produced — into the stream-ordered
    // scratch, then continue on the normal pass-A-only open path. bit_rev has
    // already set kb_rs_dense_pending, so the handshake below proceeds unchanged.
    if (kb_rs_defer_pending) {
        kb_rs_defer_pending = false;
        if (kb_rs_dense_scratch != nullptr) {
            cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
            kb_rs_dense_scratch = nullptr;
        }
        size_t need = (size_t)kb_rs_defer_width << (kb_rs_defer_lg_real + 4);
        if (mle_malloc_async_healing(&kb_rs_dense_scratch, need * sizeof(fr_t),
                                     cudaStreamPerThread) != cudaSuccess)
            return cudaErrorMemoryAllocation;
        int lerr = kb_rs_launch_dense_group(kb_rs_defer_buffer, kb_rs_dense_scratch,
            kb_rs_defer_chunks, kb_rs_defer_real_chunks, kb_rs_defer_lg_real,
            kb_rs_defer_width, cudaStreamPerThread);
        if (lerr != cudaSuccess)
            return lerr;
    }
    if (!kb_rs_dense_pending)
        return cudaSuccess;
    kb_rs_dense_pending = false;
    if (padded_poly_size != (1u << 24) || poly_count == 0) {
        // `pending` is only ever set for the in-place 2^24 shape (_kb_bit_rev), so
        // this means the pipeline contract was violated. The scratch holds the only
        // copy of the iNTT result — fail loudly rather than silently fall back.
        if (kb_rs_dense_scratch != nullptr) {
            cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
            kb_rs_dense_scratch = nullptr;
        }
        return cudaErrorInvalidValue;
    }
    *handled = 1;
    fr_t* w14 = nullptr;
    fr_t* w24g = nullptr;
    fr_t* w10 = nullptr;
    fr_t* ftab = nullptr;
    int terr = rs2_get_tables(w14, w24g, w10, ftab);
    if (terr != cudaSuccess)
        return terr;
    const uint32_t MAX_Y = 65535;
    uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
    uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;
    if (d_compact != nullptr && d_res != nullptr && n_res > 0) {
        _rs2_ntt_pass_a<true><<<dim3(1u << 10, grid_y, grid_z), 512, RS2_PASS_A_SMEM>>>(
            d_inout, kb_rs_dense_scratch, w14, d_compact, d_res, n_res,
            padded_poly_size, poly_count);
    } else {
        _rs2_ntt_pass_a<false><<<dim3(1u << 10, grid_y, grid_z), 512, RS2_PASS_A_SMEM>>>(
            d_inout, kb_rs_dense_scratch, w14, nullptr, nullptr, 0,
            padded_poly_size, poly_count);
    }
    // stream-ordered free: pass A is the last consumer of the scratch
    cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
    kb_rs_dense_scratch = nullptr;
    return CHECK_KERNEL();
}

// Gather-dot open: d_out[(qi*rows_per_query + ro) * poly_count + col] receives
// codeword row d_rows[qi*rows_per_query + ro] of column col, computed from the
// pass-A intermediate in d_src. With pruned == 0 (D-lite) d_src is the full block
// _kb_rs_open_pass_a wrote (poly_count columns of padded_poly_size); with
// pruned != 0 (D-full) it is the compact buffer and d_qslot[qi] gives each query's
// residue slot. Output layout matches _kb_matrix_get_rows_fp (row-major opened rows),
// so the host-side reassembly in rs_code_streamed_open is unchanged.
extern "C" int _kb_rs_open_gather_rows(fr_t* d_out, const fr_t* d_src,
                                       const uint32_t* d_rows,
                                       const uint32_t* d_qslot, uint32_t n_queries,
                                       uint32_t rows_per_query, uint32_t n_res,
                                       uint32_t padded_poly_size, uint32_t poly_count,
                                       int pruned)
{
    if (n_queries == 0 || poly_count == 0)
        return cudaSuccess;
    if (padded_poly_size != (1u << 24) || rows_per_query == 0)
        return cudaErrorInvalidValue;
    if (pruned && (d_qslot == nullptr || n_res == 0))
        return cudaErrorInvalidValue;
    fr_t* w14 = nullptr;
    fr_t* w24g = nullptr;
    fr_t* w10 = nullptr;
    fr_t* ftab = nullptr;
    int terr = rs2_get_tables(w14, w24g, w10, ftab);
    if (terr != cudaSuccess)
        return terr;
    const uint32_t MAX_Y = 65535;
    if (pruned) {
        // 8-column-coarsened variant (see _rs2_open_dot_pruned_colst): grid.y counts
        // 8-column groups instead of columns.
        uint32_t ycount = (poly_count + 7u) / 8u;
        uint32_t grid_y = ycount < MAX_Y ? ycount : MAX_Y;
        uint32_t grid_z = (ycount + grid_y - 1) / grid_y;
        _rs2_open_dot_pruned_colst<<<dim3(n_queries, grid_y, grid_z), 512>>>(
            d_out, d_src, d_rows, d_qslot, w24g, w10, rows_per_query, n_res,
            poly_count);
    } else {
        uint32_t grid_y = poly_count < MAX_Y ? poly_count : MAX_Y;
        uint32_t grid_z = (poly_count + grid_y - 1) / grid_y;
        _rs2_open_dot<false><<<dim3(n_queries, grid_y, grid_z), 256>>>(
            d_out, d_src, d_rows, nullptr, w24g, w10, rows_per_query, 0,
            padded_poly_size, poly_count);
    }
    return CHECK_KERNEL();
}

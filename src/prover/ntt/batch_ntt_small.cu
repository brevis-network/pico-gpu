// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "prover/ntt/device_ntt.cuh"
#include "field/fp.h"
#include "common/launcher.cuh"
#include "common/oom_heal.cuh"
#include "common/utils.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector_types.h>

using namespace device_ntt;

// ---------------------------------------------------------------------------- twiddle table

namespace device_ntt {

// Storage for the table that device_ntt.cuh declares extern.
__constant__ Fp DEVICE_NTT_TWIDDLES[DEVICE_NTT_TWIDDLES_SIZE];

} // namespace device_ntt

// Fills the packed twiddle table for every level 1..MAX_NTT_LEVEL, one entry per thread: the entry
// at position `power` within level L is g_L^power with g_L the 2^L-th root of unity.
__global__ void generate_device_ntt_twiddles_kernel(Fp *out_twiddles) {
    uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= DEVICE_NTT_TWIDDLES_SIZE)
        return;

    // Levels are stored back to back; level L holds 2^L entries starting at 2^L - 2.
    // Walk the levels until the one containing `slot` is reached.
    uint32_t lvl = 1;
    uint32_t lvl_start = 0;
    while (lvl <= MAX_NTT_LEVEL) {
        uint32_t lvl_len = 1u << lvl;
        if (slot < lvl_start + lvl_len) {
            break;
        }
        lvl_start += lvl_len;
        lvl++;
    }

    uint32_t power = slot - lvl_start;
    out_twiddles[slot] = pow(TWO_ADIC_GENERATORS[lvl], power);
}

// Builds the twiddle table in `scratch` (room for DEVICE_NTT_TWIDDLES_SIZE elements) and then copies
// it into DEVICE_NTT_TWIDDLES. Returns 0 on success, a CUDA error code otherwise.
extern "C" int _kb_generate_device_ntt_twiddles(Fp *scratch) {
    constexpr uint32_t TWIDDLE_THREADS = 256;
    uint32_t grid_blocks = div_ceil(DEVICE_NTT_TWIDDLES_SIZE, TWIDDLE_THREADS);
    cudaStream_t pt_stream = cudaStreamPerThread;
    generate_device_ntt_twiddles_kernel<<<grid_blocks, TWIDDLE_THREADS, 0, pt_stream>>>(scratch);

    // The symbol copy is queued behind the kernel on the same per-thread stream, so it reads the
    // finished table.
    cudaMemcpyToSymbolAsync(
        DEVICE_NTT_TWIDDLES,
        scratch,
        DEVICE_NTT_TWIDDLES_SIZE * sizeof(Fp),
        0,
        cudaMemcpyDeviceToDevice,
        pt_stream
    );
    cudaStreamSynchronize(pt_stream);

    return CHECK_KERNEL();
}

// ---------------------------------------------------------------------------- small batched NTT

// Transforms n_chunks independent chunks of 2^log_n elements in place, leaving each result in
// bit-reversed order. threadIdx.y selects the chunk and threadIdx.x the element; above warp size the
// chunk is staged through shared memory, otherwise the butterflies use warp shuffles.

template <bool INVERSE, bool USE_SMEM>
__global__ void batch_ntt_kernel(
    Fp *__restrict__ data,
    uint32_t const log_n,
    uint32_t const n_chunks
) {
    uint32_t const chunk = blockIdx.x * blockDim.y + threadIdx.y;
    bool const live = (chunk < n_chunks);
    data += chunk << log_n;

#ifdef CUDA_DEBUG
    assert(blockDim.x <= (1 << log_n));
#endif
    uint32_t const lane = threadIdx.x;

    Fp val;
    if constexpr (USE_SMEM) {
        extern __shared__ Fp smem[];
        auto chunk_smem = smem + (threadIdx.y << log_n);
        if (live) {
            chunk_smem[lane] = data[lane];
        }
        __syncthreads();

        ntt_natural_to_bitrev<INVERSE, true>(val, chunk_smem, lane, log_n, live);
    } else if (live) {
        // Small transforms pack several chunks into one warp. The shuffle helper masks by subgroup,
        // so lanes whose chunk is out of range can simply stay out without breaking the shuffles
        // of the chunks that are live.
        val = data[lane];
        ntt_natural_to_bitrev<INVERSE, false>(val, nullptr, lane, log_n, true);
    }

    if (live) {
        auto const dst_idx = rev_len(lane, log_n);
        data[dst_idx] = val;
    }
}

template <bool INVERSE, bool USE_SMEM>
int launch_batch_ntt_small(Fp *data, size_t const log_n, size_t const n_chunks) {
    uint32_t const block_threads = 1024;
    uint32_t const tx = 1 << log_n;
    assert(block_threads >> log_n);
    uint32_t const ty = block_threads / tx;
    size_t const smem_bytes = USE_SMEM ? (sizeof(Fp) * block_threads) : 0;

    batch_ntt_kernel<INVERSE, USE_SMEM>
        <<<div_ceil(n_chunks, ty), dim3{tx, ty, 1}, smem_bytes>>>(
            data, log_n, n_chunks
        );

    return CHECK_KERNEL();
}

extern "C" int _kb_batch_ntt_small(
    Fp *data,
    size_t const log_n,
    size_t const n_chunks,
    bool const inverse
) {
    if (log_n == 0 || n_chunks == 0) {
        return 0;
    }
    if (log_n > MAX_NTT_LEVEL) {
        return cudaErrorInvalidValue;
    }

    bool const use_smem = log_n > LOG_WARP_SIZE;
    assert((1 << log_n) <= (1 << MAX_NTT_LEVEL));
    return DISPATCH_BOOL_PAIR(
        launch_batch_ntt_small, inverse, use_smem, data, log_n, n_chunks
    );
}

// ============================================================================
// Fused per-chunk iNTT + subset-zeta (coeffs→evals)
// ============================================================================
// Replaces the [iNTT (batch_ntt_small) ; zeta (mle_interpolate_stages
// 0..l_skip-1)] pair in rs_code_matrix with a single pass. One thread owns one
// 2^l_skip chunk entirely in registers:
//   1. DIF iNTT, natural → bit-reversed order — a register transcription of
//      ntt_natural_to_bitrev<intt=true> (same halving, same inverse-twiddle
//      schedule from DEVICE_NTT_TWIDDLES, same rev_len output permutation);
//   2. the l_skip coeffs→evals zeta stages, which the staged pipeline applies
//      to the stored (bit-reversed) layout with steps 1,2,..,2^(l_skip-1):
//      y[j] += y[j - step] for every j with the `step` bit set (disjoint pairs
//      per stage, so the sequential form is identical to the parallel kernel).
// Bit-exact with the two-kernel pipeline by construction (C1_VERIFY compares).
// Chunks in the RS blowup region (chunk_in_col >= real_chunks_per_col) hold
// zeros and are skipped: iNTT(0)=0 and zeta preserves 0 (pad-skip).
template <uint32_t LOG_N>
__global__ void batch_intt_zeta_fused_kernel(
    Fp *__restrict__ buffer,
    uint32_t chunks_per_col,
    uint32_t real_chunks_per_col,
    uint32_t total_chunks
) {
    constexpr uint32_t N = 1u << LOG_N;
    uint32_t const c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= total_chunks) return;
    if (c % chunks_per_col >= real_chunks_per_col) return;

    Fp *p = buffer + ((size_t)c << LOG_N);
    Fp x[N];
    if constexpr (N % 4 == 0) {
        // Vectorized 16B loads (chunks are chunk-size aligned); fromRaw is a
        // bit-for-bit passthrough, so values are identical to scalar loads.
        const uint4 *p4 = reinterpret_cast<const uint4 *>(p);
#pragma unroll
        for (uint32_t k = 0; k < N / 4; k++) {
            uint4 v = p4[k];
            x[4 * k + 0] = Fp::fromRaw(v.x);
            x[4 * k + 1] = Fp::fromRaw(v.y);
            x[4 * k + 2] = Fp::fromRaw(v.z);
            x[4 * k + 3] = Fp::fromRaw(v.w);
        }
    } else {
#pragma unroll
        for (uint32_t i = 0; i < N; i++) x[i] = p[i];
    }

    // Per-lane inverse twiddles, squared once per level (transcribes the
    // per-thread `twiddle` of ntt_natural_to_bitrev; every lane squares every
    // level because a lane that is "upper" at one level can be "lower" later).
    Fp tw[N];
#pragma unroll
    for (uint32_t i = 0; i < N; i++) tw[i] = get_twiddle(LOG_N, i ? N - i : 0);

#pragma unroll
    for (uint32_t log_len = LOG_N; log_len-- > 0;) {
        uint32_t const len = 1u << log_len;
#pragma unroll
        for (uint32_t i = 0; i < N; i++) {
            if (!(i & len)) {
                Fp const a = x[i];
                Fp const b = x[i + len];
                x[i] = sum_or_semi_sum<true>(a + b);
                x[i + len] = sum_or_semi_sum<true>(a - b) * tw[i];
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < N; i++) tw[i] = tw[i] * tw[i];
    }

    // natural → bit-reversed output order (batch_ntt_kernel stores buffer[rev_len(i)])
    Fp y[N];
#pragma unroll
    for (uint32_t i = 0; i < N; i++) y[rev_len(i, LOG_N)] = x[i];

    // zeta stages (coeffs→evals) on the stored layout
#pragma unroll
    for (uint32_t step = 1; step < N; step <<= 1) {
#pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            if (j & step) y[j] += y[j - step];
        }
    }

    if constexpr (N % 4 == 0) {
        uint4 *p4 = reinterpret_cast<uint4 *>(p);
#pragma unroll
        for (uint32_t k = 0; k < N / 4; k++) {
            uint4 v;
            v.x = y[4 * k + 0].asRaw();
            v.y = y[4 * k + 1].asRaw();
            v.z = y[4 * k + 2].asRaw();
            v.w = y[4 * k + 3].asRaw();
            p4[k] = v;
        }
    } else {
#pragma unroll
        for (uint32_t i = 0; i < N; i++) p[i] = y[i];
    }
}

#ifdef CUDA_KB_RS_FASTPATH
// ============================================================================
// Fused iNTT + zeta + half-domain bit-reverse into the dense RS scratch
// (KoalaBear build unit only; see rs_ntt_fastpath.cu).
// ============================================================================
// In the rs_encode pipeline this fused-iNTT pass is immediately followed by a
// full-domain bit_rev whose only real content is the bottom (non-blowup) half.
// Writing the iNTT+zeta result straight to its bit-reversed dense position
// saves an entire read+write of the real half.
//
// Thread u of column `col` handles chunk cc = rev(u): output row 16*cc + i of
// the classic pipeline lands at dense position rev(row) = rev4(i)*2^lg | u, so
// writes at fixed i are fully coalesced in u, and reads are 64-byte chunk
// granules (full 32B sectors) fetched cooperatively. Arithmetic is identical
// to the in-place kernel, so the values are bit-exact.

// handshake state + kill switch, owned by rs_ntt_fastpath.cu
extern thread_local kb31_t *kb_rs_dense_scratch;
extern thread_local bool kb_rs_dense_prepared;
bool kb_rs_fastpath_enabled();

// Deferred-K1 handshake (L2-tiled scheduling layer), owned by
// rs_ntt_fastpath.cu. When the tiling is on, _kb_batch_intt_zeta_fused records
// its launch arguments here instead of running the dense kernel eagerly, and
// the stage-0 fast-path dispatch replays [K1 ; pass A ; pass B] per column
// group on a software pipeline so each column's intermediates stay L2-resident.
// All state is thread_local: prover worker threads run this concurrently under
// --default-stream=per-thread, so each thread owns its own handshake.
extern thread_local void *kb_rs_defer_buffer;
extern thread_local uint32_t kb_rs_defer_chunks;
extern thread_local uint32_t kb_rs_defer_real_chunks;
extern thread_local uint32_t kb_rs_defer_lg_real;
extern thread_local uint32_t kb_rs_defer_width;
extern thread_local bool kb_rs_defer_pending;

// Deferred-K1 is on unless RS_TILE=0 turns the L2 tiling off. RS_TILE>0 (the
// default is 1 = per-column) drives both this deferral and the tiled dispatch;
// the two translation units must agree, so both read RS_TILE the same way.
static bool rs_defer_enabled()
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("RS_TILE");
        int tile = (e != nullptr) ? atoi(e) : 1;
        v = (tile > 0) ? 1 : 0;
    }
    return v != 0;
}

template <uint32_t LOG_N>
__global__ void batch_intt_zeta_bitrev_dense_kernel(
    const Fp *__restrict__ buffer,
    Fp *__restrict__ dense,
    uint32_t chunks_per_col,
    uint32_t real_chunks_per_col,
    uint32_t lg_real_chunks,
    uint32_t width,
    uint32_t blk_off
) {
    static_assert(LOG_N == 4, "dense RS path is specialized for l_skip = 4");
    constexpr uint32_t N = 1u << LOG_N;
    constexpr uint32_t CH = 256; // chunks (= threads) per block

    uint32_t const t = threadIdx.x;
    // blk_off shifts the block's chunk window so the grid can be launched over a
    // sub-range of chunk-blocks (K1 split: first h1 blocks overlap B1 of the
    // previous group, the rest overlap B2).
    uint32_t const u0 = (blockIdx.x + blk_off) * CH;
    uint32_t const col = blockIdx.y;
    if (col >= width) return;

    // Cooperative chunk load: threads t..t+3 fetch the four uint4 quarters of
    // one 64-byte chunk (full 32B sectors, no read amplification; per-thread
    // 16B loads of rev-scattered chunks would fetch 2x). tile is [quarter]
    // [chunk]; the +2 row pad makes the row stride 8 words mod 32 so both the
    // strided writes and the row reads are bank-conflict free for 16B accesses.
    __shared__ uint4 tile[4][CH + 2];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
        uint32_t const P = t + CH * k;
        uint32_t const ci = P >> 2;
        uint32_t const pi = P & 3;
        uint32_t const u_load = u0 + ci;
        if (u_load < real_chunks_per_col) {
            uint32_t const cc = rev_len(u_load, lg_real_chunks);
            const uint4 *src = reinterpret_cast<const uint4 *>(
                buffer + (((size_t)col * chunks_per_col + cc) << LOG_N));
            // Streaming load: the input column is read exactly once per pipeline
            // run, so keep it from evicting L2-resident data.
            tile[pi][ci] = __ldcs(src + pi);
        }
    }
    __syncthreads();

    uint32_t const u = u0 + t;
    if (u >= real_chunks_per_col) return;

    Fp x[N];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
        uint4 const v = tile[k][t];
        x[4 * k + 0] = Fp::fromRaw(v.x);
        x[4 * k + 1] = Fp::fromRaw(v.y);
        x[4 * k + 2] = Fp::fromRaw(v.z);
        x[4 * k + 3] = Fp::fromRaw(v.w);
    }

    // Twiddle-by-lookup: the classic per-lane running table tw[i] (seeded with
    // level-LOG_N inverse twiddles and squared once per level) satisfies, at the
    // level with half-size `len`, tw[i] = w_{2len}^{-(i mod 2len)} — exactly the
    // level-(log_len+1) inverse twiddle already stored in constant memory. Look
    // it up directly (indices are compile-time constants after unrolling),
    // replacing N registers + N*LOG_N squarings per thread with a handful of
    // broadcast constant-cache loads. Same values, bit-exact.
#pragma unroll
    for (uint32_t log_len = LOG_N; log_len-- > 0;) {
        uint32_t const len = 1u << log_len;
#pragma unroll
        for (uint32_t i = 0; i < N; i++) {
            if (!(i & len)) {
                uint32_t const idx = i & (2 * len - 1);
                Fp const tw = get_twiddle(log_len + 1, idx ? 2 * len - idx : 0);
                Fp const a = x[i];
                Fp const b = x[i + len];
                x[i] = sum_or_semi_sum<true>(a + b);
                x[i + len] = sum_or_semi_sum<true>(a - b) * tw;
            }
        }
    }

    Fp y[N];
#pragma unroll
    for (uint32_t i = 0; i < N; i++) y[rev_len(i, LOG_N)] = x[i];

#pragma unroll
    for (uint32_t step = 1; step < N; step <<= 1) {
#pragma unroll
        for (uint32_t j = 0; j < N; j++) {
            if (j & step) y[j] += y[j - step];
        }
    }

    Fp *d = dense + ((size_t)col << (lg_real_chunks + LOG_N)) + u;
#pragma unroll
    for (uint32_t i = 0; i < N; i++)
        d[(size_t)rev_len(i, LOG_N) << lg_real_chunks] = y[i];
}

// Launch the dense (K1) kernel for `ncols` columns starting at `buf` (already
// offset to the group's first column) into `dense` (group-relative column 0).
// Used by the L2-tiled deferred pipeline in rs_ntt_fastpath.cu.
extern "C" int kb_rs_launch_dense_group(
    const void *buf, void *dense, uint32_t chunks_per_col,
    uint32_t real_chunks_per_col, uint32_t lg_real, uint32_t ncols,
    cudaStream_t stream)
{
    uint32_t const block = 256;
    uint32_t const grid_x = (real_chunks_per_col + block - 1) / block;
    batch_intt_zeta_bitrev_dense_kernel<4>
        <<<dim3(grid_x, ncols, 1), block, 0, stream>>>(
            reinterpret_cast<const Fp *>(buf), reinterpret_cast<Fp *>(dense),
            chunks_per_col, real_chunks_per_col, lg_real, ncols, 0);
    return CHECK_KERNEL();
}

// Launch only chunk-blocks [blk_off, blk_off+nblk) of the dense group, so K1 can
// be split into two half-grid launches placed on separate pipeline events (the
// first half overlapping B1 of the previous group, the rest overlapping B2).
extern "C" int kb_rs_launch_dense_group_range(
    const void *buf, void *dense, uint32_t chunks_per_col,
    uint32_t real_chunks_per_col, uint32_t lg_real, uint32_t ncols,
    uint32_t blk_off, uint32_t nblk, cudaStream_t stream)
{
    if (nblk == 0) return cudaSuccess;
    batch_intt_zeta_bitrev_dense_kernel<4>
        <<<dim3(nblk, ncols, 1), 256, 0, stream>>>(
            reinterpret_cast<const Fp *>(buf), reinterpret_cast<Fp *>(dense),
            chunks_per_col, real_chunks_per_col, lg_real, ncols, blk_off);
    return CHECK_KERNEL();
}
#endif // CUDA_KB_RS_FASTPATH

extern "C" int _kb_batch_intt_zeta_fused(
    Fp *buffer,
    size_t const l_skip,
    size_t const chunks_per_col,
    size_t const real_chunks_per_col,
    size_t const width
) {
    if (l_skip == 0 || width == 0 || chunks_per_col == 0) {
        return 0;
    }
#ifdef CUDA_KB_RS_FASTPATH
    // RS dense-half fast path: blowup-2 shape whose following bit_rev call
    // (in-place, lg = l_skip + log2(chunks_per_col) = 24) will accept the
    // staged dense scratch. Writes the bit-reversed dense half directly and
    // leaves `buffer` untouched; the in-place kernel below remains the path
    // for every other shape and when RS_DENSE=0 / C1_VERIFY is set.
    if (l_skip == 4 && chunks_per_col == real_chunks_per_col * 2 &&
        (real_chunks_per_col & (real_chunks_per_col - 1)) == 0 &&
        real_chunks_per_col > 1 &&
        (chunks_per_col << l_skip) == ((size_t)1 << 24) &&
        width <= 64 && kb_rs_fastpath_enabled()) {
        uint32_t const lg_real = 31 - __builtin_clz((uint32_t)real_chunks_per_col);
        if (rs_defer_enabled()) {
            // L2-tiled deferred pipeline: record the launch and let the stage-0
            // fast-path dispatch replay [K1 ; pass A ; pass B] per column group.
            // Leaves `buffer` untouched (still holds the iNTT input the deferred
            // K1 will consume) and allocates no scratch here; the tiled dispatch
            // owns a persistent per-thread group-sized scratch. The subsequent
            // bit_rev sees `prepared` and flips it to `pending` as usual.
            kb_rs_defer_buffer = buffer;
            kb_rs_defer_chunks = (uint32_t)chunks_per_col;
            kb_rs_defer_real_chunks = (uint32_t)real_chunks_per_col;
            kb_rs_defer_lg_real = lg_real;
            kb_rs_defer_width = (uint32_t)width;
            kb_rs_defer_pending = true;
            kb_rs_dense_prepared = true;
            return 0;
        }
        size_t const need = (size_t)width << (lg_real + l_skip);
        if (kb_rs_dense_scratch != nullptr) {
            // stale scratch from an aborted pipeline
            cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
            kb_rs_dense_scratch = nullptr;
        }
        if (mle_malloc_async_or_fallback(&kb_rs_dense_scratch, need * sizeof(Fp),
                                         cudaStreamPerThread) == cudaSuccess) {
            uint32_t const block = 256;
            uint32_t const grid_x =
                (uint32_t)((real_chunks_per_col + block - 1) / block);
            batch_intt_zeta_bitrev_dense_kernel<4>
                <<<dim3(grid_x, (uint32_t)width, 1), block>>>(
                    buffer, reinterpret_cast<Fp *>(kb_rs_dense_scratch),
                    (uint32_t)chunks_per_col, (uint32_t)real_chunks_per_col,
                    lg_real, (uint32_t)width, 0);
            int err = CHECK_KERNEL();
            if (err == 0) {
                kb_rs_dense_prepared = true;
            } else {
                cudaFreeAsync(kb_rs_dense_scratch, cudaStreamPerThread);
                kb_rs_dense_scratch = nullptr;
            }
            return err;
        }
        kb_rs_dense_scratch = nullptr;
        // allocation failed: fall through to the classic in-place pipeline
    }
#endif // CUDA_KB_RS_FASTPATH
    size_t const total = width * chunks_per_col;
    uint32_t const block = 256;
    uint32_t const grid = (uint32_t)((total + block - 1) / block);
    switch (l_skip) {
    case 1:
        batch_intt_zeta_fused_kernel<1><<<grid, block>>>(
            buffer, chunks_per_col, real_chunks_per_col, total);
        break;
    case 2:
        batch_intt_zeta_fused_kernel<2><<<grid, block>>>(
            buffer, chunks_per_col, real_chunks_per_col, total);
        break;
    case 3:
        batch_intt_zeta_fused_kernel<3><<<grid, block>>>(
            buffer, chunks_per_col, real_chunks_per_col, total);
        break;
    case 4:
        batch_intt_zeta_fused_kernel<4><<<grid, block>>>(
            buffer, chunks_per_col, real_chunks_per_col, total);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return CHECK_KERNEL();
}

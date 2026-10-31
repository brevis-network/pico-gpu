// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

// ============================================================================
// SHARED FpExt REDUCTION TOOLBOX
// ============================================================================
//
// The crate's shared extension-field reduction primitives — warp, chunk and block sums, u64
// atomic accumulators, and the cross-block final-reduce kernels. Included across `poly`, `whir`,
// `stacked_reduction` and all of `logup_zerocheck`, and compiled for `kb31_t`.
//
// SHARED MEMORY. `block_reduce_sum` and `chunk_reduce_sum` do not allocate: the caller passes an
// `FpExt *` into its launch's dynamic shared memory, at least `(blockDim.x + 31) / 32` slots (one
// per warp). Neither zeroes it nor barriers after its last read, so reducing twice through one
// allocation needs the caller's own `__syncthreads()` between calls.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include <cstdint>
#include <device_atomic_functions.h>

// ----------------------------------------------------------------------------
// Launch-shape probe — CUDA_DEBUG builds only
// ----------------------------------------------------------------------------
//
// Under `-DCUDA_DEBUG` (set by `CUDA_DEBUG=1` in `build.rs`) the reducers take their caller's source
// location as defaulted parameters and, from thread 0, print the block shape: a non-1-D block is a
// bug (lane bookkeeping here is 1-D-only), a ragged block is legal and reported only as a census.
// Compiled out of every normal build, so the production path is unaffected.
#ifdef CUDA_DEBUG
#define REDUCE_CALLER_LOC_PARAMS                                                                 \
    , const char *caller_file = __builtin_FILE(), int caller_line = __builtin_LINE()
#define REDUCE_CALLER_LOC_ARGS , caller_file, caller_line
#define REDUCE_REPORT_LAUNCH_SHAPE()                                                             \
    do {                                                                                           \
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 &&         \
            blockIdx.y == 0 && blockIdx.z == 0) {                                                  \
            /* A defect: the lane bookkeeping here is 1-D-only. */                                 \
            if (blockDim.y != 1u || blockDim.z != 1u) {                                            \
                printf(                                                                            \
                    "[CUDA_DEBUG] BUG: non-1-D block reduced at %s:%d"                             \
                    " - block=(%u,%u,%u) grid=(%u,%u,%u)\n",                                       \
                    caller_file,                                                                   \
                    caller_line,                                                                   \
                    blockDim.x,                                                                    \
                    blockDim.y,                                                                    \
                    blockDim.z,                                                                    \
                    gridDim.x,                                                                     \
                    gridDim.y,                                                                     \
                    gridDim.z                                                                      \
                );                                                                                 \
            }                                                                                      \
            /* Not a defect: a ragged block reduces correctly. Reported to census the shapes. */   \
            if (blockDim.x % WARP_SIZE != 0) {                                                     \
                printf(                                                                            \
                    "[CUDA_DEBUG] ragged block reduced (legal) at %s:%d"                           \
                    " - block=(%u,%u,%u) grid=(%u,%u,%u)\n",                                       \
                    caller_file,                                                                   \
                    caller_line,                                                                   \
                    blockDim.x,                                                                    \
                    blockDim.y,                                                                    \
                    blockDim.z,                                                                    \
                    gridDim.x,                                                                     \
                    gridDim.y,                                                                     \
                    gridDim.z                                                                      \
                );                                                                                 \
            }                                                                                      \
        }                                                                                          \
    } while (0)
#else
#define REDUCE_CALLER_LOC_PARAMS
#define REDUCE_CALLER_LOC_ARGS
#define REDUCE_REPORT_LAUNCH_SHAPE() ((void)0)
#endif

namespace reduce {

// How many lanes of the calling thread's warp the launch created, in [1, 32], from `blockDim.x`.
// 1-D blocks only (the lane id is `threadIdx.x`).
static __device__ __forceinline__ unsigned warp_live_lanes() {
    unsigned warp_base = threadIdx.x & ~31u; // first lane of this warp; WARP_SIZE == 32
    unsigned tail = blockDim.x - warp_base;  // >= 1, since threadIdx.x < blockDim.x
    return tail < 32u ? tail : 32u;
}

// Sums FpExt across the live lanes of one warp (down-sweep of `__shfl_down_sync` at offsets 16..1),
// leaving the complete sum in lane 0 only. `live` comes from `warp_live_lanes()`, so a ragged block
// is correct — but every lane of the warp must be created and converged here: accumulate under
// `if (...)` and fall through, never `return` over a subset. The mask is `__activemask()` as a sync
// argument only; do not use it to derive `live`, and do not replace it with a literal `0xffffffff`.
// Callers needing the sum in every lane want `warp_reduce_sum_n` instead.
static __device__ inline FpExt warp_reduce_sum(FpExt val REDUCE_CALLER_LOC_PARAMS) {
    REDUCE_REPORT_LAUNCH_SHAPE();
    unsigned mask = __activemask(); // sync argument only — see above
    unsigned lane = threadIdx.x & 31u;
    unsigned live = warp_live_lanes();

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        // The shuffles stay outside the guard: every thread of the mask must execute each one.
        FpExt other;
        other.elems[0] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[0].asRaw(), offset));
        other.elems[1] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[1].asRaw(), offset));
        other.elems[2] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[2].asRaw(), offset));
        other.elems[3] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[3].asRaw(), offset));
        if (lane + offset < live) {
            val = val + other;
        }
    }
    return val;
}

// Reduces FpExt across `n` threads (butterfly of `__shfl_xor_sync`), leaving the sum in every
// participating thread. `n` a power of two, 1 <= n <= 32. Every `n`-lane group must be wholly
// created and converged here (the accumulate is unconditional). Mask stays `__activemask()`, never
// a literal `0xffffffff`.
__device__ __forceinline__ FpExt warp_reduce_sum_n(FpExt val, uint32_t n) {
    unsigned mask = __activemask();

    for (uint32_t offset = n >> 1; offset > 0; offset >>= 1) {
        FpExt other;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            other.elems[i] = Fp::fromRaw(__shfl_xor_sync(mask, val.elems[i].asRaw(), offset));
        }
        val = val + other;
    }
    return val;
}

// Reduces FpExt across the `chunk_size` threads of one chunk (the block partitioned into equal
// chunks). `chunk_size` a power of 2; `tid_in_chunk` in [0, chunk_size); `smem` >=
// total_warps_in_block slots. Read the result under `tid_in_chunk == 0` — for `chunk_size > 32`
// only that thread holds the full sum. The `chunk_size > 32` branch reaches a block-wide
// `__syncthreads()`, so `chunk_size` must be uniform and every thread of the block must call, and
// chunks must start at a multiple of 32 (`chunk_size` caps at 1024).
__device__ inline FpExt chunk_reduce_sum(
    FpExt acc,
    FpExt *smem,
    uint32_t tid_in_chunk,
    uint32_t chunk_size,
    uint32_t chunk_in_block REDUCE_CALLER_LOC_PARAMS
) {
    if (chunk_size <= 32) {
        return warp_reduce_sum_n(acc, chunk_size);
    }

    // Chunk spans several warps.
    uint32_t chunk_warps = chunk_size >> 5;
    uint32_t warp_rank = tid_in_chunk >> 5;
    uint32_t lane = tid_in_chunk & 31;

    // (1) every warp folds its own lanes
    acc = warp_reduce_sum(acc REDUCE_CALLER_LOC_ARGS);

    // (2) publish one partial per warp in this chunk's region of `smem`
    FpExt *chunk_partials = smem + chunk_in_block * chunk_warps;
    if (lane == 0) {
        chunk_partials[warp_rank] = acc;
    }
    __syncthreads();

    // (3) the chunk's leading warp folds those partials
    if (warp_rank == 0) {
        FpExt zero_ext = {0, 0, 0, 0};
        FpExt partial = (lane < chunk_warps) ? chunk_partials[lane] : zero_ext;
        acc = warp_reduce_sum(partial REDUCE_CALLER_LOC_ARGS);
    }

    return acc;
}

// ============================================================================
// U64 ATOMIC SUMS (the modular reduction is deferred to the host)
// ============================================================================
//
// `Fp::asRaw()` is a canonical residue in [0, p) and both moduli are below 2^31, so a uint64_t
// accumulator has headroom to ~2^33 summands. The device never reduces modulo p; the final modular
// reduction is done on CPU after kernel completion.

// Adds the canonical residue of `x` to a 64-bit counter without reducing.
__device__ __forceinline__ void atomic_add_fp_to_u64(uint64_t *acc_u64, Fp x) {
    atomicAdd((unsigned long long *)acc_u64, static_cast<uint64_t>(x.asRaw()));
}

// Same for an extension element: limb k of `x` goes into acc_u64[k], k = 0..3.
__device__ __forceinline__ void atomic_add_fpext_to_u64(uint64_t *acc_u64, FpExt x) {
#pragma unroll
    for (int limb = 0; limb < 4; ++limb) {
        atomic_add_fp_to_u64(acc_u64 + limb, x.elems[limb]);
    }
}

// Sums FpExt across all threads of a block: each warp reduces via `warp_reduce_sum`, then the first
// warp reduces the per-warp partials through `shared`. Result in thread 0 only. `shared` sized as
// in the SHARED MEMORY note; block-wide `__syncthreads()`, so every thread must call; <= 32 warps.
static __device__ inline FpExt block_reduce_sum(
    FpExt acc,
    FpExt *shared REDUCE_CALLER_LOC_PARAMS
) {
    int t = threadIdx.x;
    int warp_no = t / 32;
    int lane = t % 32;
    // Rounds up, so a ragged block still counts (and stage 2 still reads) its tail warp's slot.
    int warp_count = (blockDim.x + 31) / 32;

    // Stage 1: intra-warp fold.
    acc = warp_reduce_sum(acc REDUCE_CALLER_LOC_ARGS);

    // One partial per warp goes to `shared`.
    if (lane == 0) {
        shared[warp_no] = acc;
    }
    __syncthreads();

    // Stage 2: warp 0 folds the partials, lane k picking up warp k's slot.
    FpExt zero_ext = {0, 0, 0, 0};
    if (warp_no == 0) {
        FpExt partial = (lane < warp_count) ? shared[lane] : zero_ext;
        acc = warp_reduce_sum(partial REDUCE_CALLER_LOC_ARGS);
    }

    return acc;
}

// Combines the `num_blocks` partial sums a previous launch produced into one result per output.
// Launch with `gridDim.x == D` (block `i` owns output `i`); `D` is a compile-time constant, which
// distinguishes this from `final_reduce_block_sums` below. 1-D block with dynamic shared memory for
// the `block_reduce_sum` it ends in; a ragged block size is legal.
template <int D>
static __global__ void static_final_reduce_block_sums(
    const FpExt *partials, // num_blocks rows of D values
    FpExt *dst,            // D values
    uint32_t n_partials
) {
    extern __shared__ char smem[];
    FpExt *scratch = (FpExt *)smem;

    int t = threadIdx.x;

    // This block's output column.
    int col = blockIdx.x;
    if (col >= D)
        return;

    FpExt total = {0, 0, 0, 0};

    // Block-stride walk over the partial rows.
    for (int row = t; row < n_partials; row += blockDim.x) {
        total += partials[row * D + col];
    }

    total = block_reduce_sum(total, scratch);

    if (t == 0) {
        dst[col] = total;
    }
}

// As `static_final_reduce_block_sums`, but the output count is runtime: `d = gridDim.x`, so
// `gridDim.x` must equal `d`. With `ADD_TO_OUTPUT` the result is added onto `output` (which must
// already hold the addend) instead of overwriting. Launch shape as above.
template <bool ADD_TO_OUTPUT = false>
static __global__ void final_reduce_block_sums(
    const FpExt *partials, // num_blocks rows of d values
    FpExt *dst,            // d values
    uint32_t n_partials
) {
    extern __shared__ char smem[];
    FpExt *scratch = (FpExt *)smem;

    int t = threadIdx.x;

    // One block per output column; the grid width is the column count.
    int col = blockIdx.x;
    int n_cols = gridDim.x;

    FpExt total = {0, 0, 0, 0};

    // Block-stride walk over the partial rows.
    for (int row = t; row < n_partials; row += blockDim.x) {
        total += partials[row * n_cols + col];
    }

    total = block_reduce_sum(total, scratch);

    if (t == 0) {
        if constexpr (ADD_TO_OUTPUT) {
            dst[col] += total;
        } else {
            dst[col] = total;
        }
    }
}

// Several independent reductions (one per segment, e.g. per AIR) in one launch. Block (s, c) of the
// (num_segments, d) grid sums column c over rows segment_offsets[s] .. segment_offsets[s + 1] - 1
// and writes output[s * d + c].
//
// Blocks are 1-D with dynamic shared memory as for `static_final_reduce_block_sums`.
static __global__ void batched_final_reduce_block_sums(
    const FpExt *partials,             // total_blocks rows of d values
    FpExt *dst,                        // num_segments rows of d values
    const uint32_t *segment_offsets,   // num_segments + 1 row boundaries (device memory)
    uint32_t n_cols
) {
    extern __shared__ char smem[];
    FpExt *scratch = (FpExt *)smem;

    uint32_t seg = blockIdx.x;
    uint32_t col = blockIdx.y;
    int t = threadIdx.x;

    uint32_t row_begin = segment_offsets[seg];
    uint32_t row_end = segment_offsets[seg + 1];
    uint32_t seg_rows = row_end - row_begin;

    FpExt total = {0, 0, 0, 0};

    // Block-stride walk over this segment's rows.
    for (uint32_t r = t; r < seg_rows; r += blockDim.x) {
        total += partials[(row_begin + r) * n_cols + col];
    }

    total = block_reduce_sum(total, scratch);

    if (t == 0) {
        dst[seg * n_cols + col] = total;
    }
}

} // namespace reduce

// Scoped to this header — nothing outside it should be growing a caller-location parameter.
#undef REDUCE_CALLER_LOC_PARAMS
#undef REDUCE_CALLER_LOC_ARGS
#undef REDUCE_REPORT_LAUNCH_SHAPE

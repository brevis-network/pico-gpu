// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

// MLE-interpolate butterfly kernels: one stage does, per `2*step`-aligned pair `(u, v)` that are
// `step` apart, `v -= u` (eval->coeff) or `v += u` (coeff->eval).
//
// Four kernels, dispatched from Rust (`src/cuda/mle_interpolate.rs::mle_interpolate_stages`):
//   - butterfly_flat_kernel             1-D over a flat buffer. Only the FpExt instantiation is
//                                       live (WHIR within-chunk zeta); the Fp launcher was removed
//                                       as it had no caller in either link unit.
//   - butterfly_warp_kernel             warp-shuffle, fuses 2..5 stages (log_step < LOG_WARP_SIZE).
//   - butterfly_tile_kernel             shared-memory tile (log_step < MLE_SHARED_TILE_LOG_SIZE).
//   - butterfly_column_kernel           per-stage fallback for log_step >= tile size.
//
// This file compiles into `cuda_kb_all` (`field/fp.h` supplies the field). Its `extern "C"`
// launchers are named `_kb_*`, the names the Rust side binds; a new one follows the same
// convention.
//
// Caller-side preconditions the 2-D kernels rely on (see the Rust wrappers for the debug_asserts):
//   - `width * padded_height < 2^32` — the 2-D kernels index cells with `uint32_t`.
//   - warp shuffle: `meaningful_count >= 2 * max_step`, else the partner lane is inactive.
//
// CPU counterpart: `mle_core::poly::mle::Mle::{evals_to_coeffs,coeffs_to_evals}_inplace` per chunk.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include "common/utils.cuh"
// standard library / CUDA runtime
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <vector_types.h>

// ---------------------------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------------------------

// One butterfly stage over a flat buffer; one thread per (low, high) pair `gap` apart.
template <typename Elem, bool ToCoeffs>
__global__ void butterfly_flat_kernel(Elem *buf, size_t pair_count, uint32_t gap) {
    size_t pair_span = size_t(gap) << 1;
    size_t pair = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (pair >= pair_count) {
        return;
    }

    size_t group = pair / gap;
    uint32_t within = pair % gap;
    size_t low_pos = group * pair_span + within;
    size_t high_pos = low_pos + gap;
    if constexpr (ToCoeffs) {
        buf[high_pos] -= buf[low_pos];
    } else {
        buf[high_pos] += buf[low_pos];
    }
}

// Per-stage fallback for one column (grid.y = width). `pair_count = height >> 1` bounds the
// meaningful threads; the guard makes the kernel safe under any launch geometry (F19), matching the
// 1-D kernel above. `column * col_height + ...` is a `uint32_t`, so the caller must keep
// `width * padded_height < 2^32` (N8).
template <typename Elem, bool ToCoeffs>
__global__ void butterfly_column_kernel(
    Elem *buf,
    uint32_t pair_count,
    uint32_t col_height,
    uint32_t pair_span,
    uint32_t gap
) {
    uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) {
        return;
    }
    uint32_t column = blockIdx.y;

    auto group = pair / gap;
    auto within = pair % gap;
    uint32_t low_pos = column * col_height + group * pair_span + within;
    uint32_t high_pos = low_pos + gap;

    if constexpr (ToCoeffs) {
        buf[high_pos] -= buf[low_pos];
    } else {
        buf[high_pos] += buf[low_pos];
    }
}

// Runs `Passes` consecutive butterfly stages in registers, exchanging partners with warp shuffles.
// Stage gaps go `first_gap, 2*first_gap, ..., first_gap << (Passes - 1)`; one thread owns one
// element. The high element of each pair becomes `high - low` (ToCoeffs) or `high + low`.
//
// Precondition (warp shuffle): the partner lane is `elem ^ gap`, which is in-lane only while
// `live_count >= 2 * max_gap` (max_gap = first_gap << (Passes - 1)). This holds by
// construction because `live_count = col_height >> log_spacing = 2^(l_skip + n_stack)` and
// `max_gap <= 2^l_skip`; the Rust wrapper `debug_assert!`s it. `column * col_height + ...` is a
// `uint32_t`, so the caller must keep `width * padded_height < 2^32` (N8).
//
// `Strided`: element `elem` lives at row `elem << log_spacing` (right-padded layout) instead of
// row `elem`. `Passes` is 1..5 (at most LOG_WARP_SIZE).
template <bool ToCoeffs, int Passes, bool Strided>
__global__ void butterfly_warp_kernel(
    Fp *buf,
    uint32_t col_height,
    uint32_t log_spacing, // live rows per column = col_height >> log_spacing
    uint32_t first_gap    // power-of-two gap of the first fused stage
) {
    uint32_t elem = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t column = blockIdx.y;

    uint32_t live_count = col_height >> log_spacing;
    if (elem >= live_count)
        return;

    uint32_t slot;
    if constexpr (Strided) {
        slot = elem << log_spacing;
    } else {
        slot = elem;
    }

    uint32_t addr = column * col_height + slot;
    Fp mine = buf[addr];

    unsigned active = __activemask();
#pragma unroll
    for (int pass = 0; pass < Passes; pass++) {
        uint32_t gap = first_gap << pass;

        // partner is the lane whose index differs only in the `gap` bit
        Fp theirs = Fp::fromRaw(__shfl_xor_sync(active, mine.asRaw(), gap));

        bool in_low_half = (elem & gap) == 0;

        // only the high element changes; the low one is read by its partner and kept
        if constexpr (ToCoeffs) {
            if (!in_low_half) {
                mine = mine - theirs;
            }
        } else {
            if (!in_low_half) {
                mine = mine + theirs;
            }
        }
    }

    buf[addr] = mine;
}

// Shared-memory index with one pad word after every 32: power-of-two gaps would otherwise map
// every lane of a warp onto the same bank.
__device__ __forceinline__ uint32_t skew_index(uint32_t pos) {
    return pos + (pos >> 5); // pos + pos/32
}

// Runs stages `first_log_gap..=last_log_gap` (all gaps < tile length) on one tile of one column held
// in shared memory. One block per (tile, column).
//
// `LogTile` = log2 of the tile length (sets the dynamic shared-memory size).
// `Strided`: live element `k` sits at row `k << log_spacing`; otherwise at row `k`.
template <bool ToCoeffs, int LogTile, bool Strided>
__global__ void butterfly_tile_kernel(
    Fp *buf,
    uint32_t col_height,
    uint32_t log_spacing,   // live rows per column = col_height >> log_spacing
    uint32_t first_log_gap, // log2 of the smallest gap handled here
    uint32_t last_log_gap   // log2 of the largest gap handled here (inclusive)
) {
    constexpr uint32_t tile_len = 1 << LogTile;
    extern __shared__ Fp smem_mle[];

    uint32_t tile = blockIdx.x;
    uint32_t column = blockIdx.y;
    uint32_t t = threadIdx.x;

    uint32_t live_count = col_height >> log_spacing;
    uint32_t tile_base = tile * tile_len; // first logical element of this tile
    uint32_t column_base = column * col_height;

// stage the tile in shared memory; out-of-range slots are zero-filled
#pragma unroll 4
    for (uint32_t k = t; k < tile_len; k += blockDim.x) {
        uint32_t logical = tile_base + k;
        if (logical < live_count) {
            uint32_t slot;
            if constexpr (Strided) {
                slot = logical << log_spacing;
            } else {
                slot = logical;
            }
            smem_mle[skew_index(k)] = buf[column_base + slot];
        } else {
            smem_mle[skew_index(k)] = Fp(0);
        }
    }
    __syncthreads();

    for (uint32_t lg = first_log_gap; lg <= last_log_gap; lg++) {
        uint32_t gap = 1 << lg;
        uint32_t pair_span = gap << 1;

// threads stride over the tile's pairs
#pragma unroll 4
        for (uint32_t k = t; k < tile_len / 2; k += blockDim.x) {
            uint32_t group = k / gap;
            uint32_t within = k % gap;
            uint32_t low_pos = group * pair_span + within;
            uint32_t high_pos = low_pos + gap;

            // skip pairs whose high element lies past the live rows
            uint32_t hi_logical = tile_base + high_pos;
            if (hi_logical < live_count) {
                if constexpr (ToCoeffs) {
                    smem_mle[skew_index(high_pos)] -= smem_mle[skew_index(low_pos)];
                } else {
                    smem_mle[skew_index(high_pos)] += smem_mle[skew_index(low_pos)];
                }
            }
        }
        __syncthreads();
    }

// copy the live part of the tile back out
#pragma unroll 4
    for (uint32_t k = t; k < tile_len; k += blockDim.x) {
        uint32_t logical = tile_base + k;
        if (logical < live_count) {
            uint32_t slot;
            if constexpr (Strided) {
                slot = logical << log_spacing;
            } else {
                slot = logical;
            }
            buf[column_base + slot] = smem_mle[skew_index(k)];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------------------------

template <typename Elem, bool ToCoeffs>
int run_flat_stage(Elem *buf, size_t len, uint32_t gap) {
    size_t pair_count = len >> 1;
    auto [launch_grid, launch_block] = kernel_launch_params(pair_count);
    butterfly_flat_kernel<Elem, ToCoeffs><<<launch_grid, launch_block>>>(buf, pair_count, gap);
    return CHECK_KERNEL();
}

// Only the extension-field 1-D launcher exists: the base-field one was removed because
// neither link unit called it.
extern "C" int _kb_mle_interpolate_stage_ext(
    FpExt *buf,
    size_t len,
    uint32_t gap,
    bool is_eval_to_coeff
) {
    if (len < 2 || gap == 0) {
        return 0;
    }

    if (is_eval_to_coeff) {
        return run_flat_stage<FpExt, true>(buf, len, gap);
    } else {
        return run_flat_stage<FpExt, false>(buf, len, gap);
    }
}

template <typename Elem, bool ToCoeffs>
int run_column_stage(
    Elem *buf,
    uint16_t width,
    uint32_t height,
    uint32_t col_height,
    uint32_t gap
) {
    auto pair_span = gap * 2;
    uint32_t pair_count = height >> 1;
    auto [launch_grid, launch_block] = kernel_launch_params(pair_count);
    launch_grid.y = width;
    butterfly_column_kernel<Elem, ToCoeffs>
        <<<launch_grid, launch_block>>>(buf, pair_count, col_height, pair_span, gap);
    return CHECK_KERNEL();
}

extern "C" int _kb_mle_interpolate_stage_2d(
    Fp *buf,
    uint16_t width,
    uint32_t height,
    uint32_t col_height,
    uint32_t gap,
    bool is_eval_to_coeff
) {
    if (width == 0) {
        return cudaErrorInvalidValue;
    }
    if (is_eval_to_coeff) {
        return run_column_stage<Fp, true>(
            buf, width, height, col_height, gap
        );
    } else {
        return run_column_stage<Fp, false>(
            buf, width, height, col_height, gap
        );
    }
}

// Launches `butterfly_warp_kernel`: `Passes` stages starting at gap `first_gap` (a power of two
// with `first_gap << (Passes - 1) <= 16`), one thread per live row (`col_height >> log_spacing`).
template <bool ToCoeffs, int Passes, bool Strided>
int run_warp_fused(
    Fp *buf,
    uint16_t width,
    uint32_t col_height,
    uint32_t log_spacing,
    uint32_t first_gap
) {
    uint32_t live_count = col_height >> log_spacing;
    auto [launch_grid, launch_block] = kernel_launch_params(live_count);
    launch_grid.y = width;
    butterfly_warp_kernel<ToCoeffs, Passes, Strided>
        <<<launch_grid, launch_block>>>(buf, col_height, log_spacing, first_gap);
    return CHECK_KERNEL();
}

// Maps the runtime pass count onto the `Passes` template argument, counting down from LOG_WARP_SIZE.
template <bool ToCoeffs, bool Strided, int Stages = LOG_WARP_SIZE>
int pick_warp_fused(
    Fp *buf,
    uint16_t width,
    uint32_t col_height,
    uint32_t log_spacing,
    uint32_t first_gap,
    uint32_t pass_count
) {
    if constexpr (Stages == 0) {
        return cudaErrorInvalidValue;
    } else {
        if (pass_count == Stages) {
            return run_warp_fused<ToCoeffs, Stages, Strided>(
                buf, width, col_height, log_spacing, first_gap
            );
        }
        return pick_warp_fused<ToCoeffs, Strided, Stages - 1>(
            buf, width, col_height, log_spacing, first_gap, pass_count
        );
    }
}

extern "C" int _kb_mle_interpolate_fused_2d(
    Fp *buf,
    uint16_t width,
    uint32_t col_height,
    uint32_t log_spacing,
    uint32_t first_gap,
    uint32_t pass_count,
    bool is_eval_to_coeff,
    bool right_pad
) {
    if (width == 0) {
        return cudaErrorInvalidValue;
    }
    return DISPATCH_BOOL_PAIR(
        pick_warp_fused,
        is_eval_to_coeff,
        right_pad,
        buf,
        width,
        col_height,
        log_spacing,
        first_gap,
        pass_count
    );
}

// log2 of the shared-memory tile length: 4096 Fp = 16 KiB per block, so gaps up to 2^11 fit.
// A bigger tile fuses more stages per launch at the cost of occupancy.
constexpr int MLE_SHARED_TILE_LOG_SIZE = 12;

template <bool ToCoeffs, bool Strided>
int run_tile_fused(
    Fp *buf,
    uint16_t width,
    uint32_t col_height,
    uint32_t log_spacing,
    uint32_t first_log_gap,
    uint32_t last_log_gap
) {
    constexpr uint32_t tile_len = 1 << MLE_SHARED_TILE_LOG_SIZE;
    constexpr uint32_t threads_per_block = 256;
    // one pad word per 32 entries, see skew_index
    size_t smem_bytes = (tile_len + tile_len / 32) * sizeof(Fp);

    uint32_t live_count = col_height >> log_spacing;
    uint32_t tile_count = (live_count + tile_len - 1) / tile_len;
    dim3 launch_grid(tile_count, width);

    butterfly_tile_kernel<ToCoeffs, MLE_SHARED_TILE_LOG_SIZE, Strided>
        <<<launch_grid, threads_per_block, smem_bytes>>>(
            buf, col_height, log_spacing, first_log_gap, last_log_gap
        );
    return CHECK_KERNEL();
}

extern "C" int _kb_mle_interpolate_shared_2d(
    Fp *buf,
    uint16_t width,
    uint32_t col_height,
    uint32_t log_spacing,
    uint32_t first_log_gap,
    uint32_t last_log_gap,
    bool is_eval_to_coeff,
    bool right_pad
) {
    if (width == 0) {
        return cudaErrorInvalidValue;
    }
    // every gap must stay inside one tile
    if (last_log_gap >= MLE_SHARED_TILE_LOG_SIZE) {
        return cudaErrorInvalidValue;
    }
    return DISPATCH_BOOL_PAIR(
        run_tile_fused,
        is_eval_to_coeff,
        right_pad,
        buf,
        width,
        col_height,
        log_spacing,
        first_log_gap,
        last_log_gap
    );
}

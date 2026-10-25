// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/fp.h"
#include "common/utils.cuh"
#include <cstdint>

// Small in-kernel NTTs over the round-0 skip domain: one element per thread, warp shuffles for the
// levels inside a warp and shared memory for the levels above it.
namespace device_ntt {

// ----------------------------------------------------------------------------
// Twiddle table (constant memory)
// ----------------------------------------------------------------------------
//
// The table is the concatenation, for lvl = 1 .. MAX_NTT_LEVEL, of the powers w^0 .. w^(2^lvl - 1)
// of w = TWO_ADIC_GENERATORS[lvl]. Block `lvl` therefore begins at 2^lvl - 2, and the whole table
// holds 2 + 4 + ... + 2^MAX_NTT_LEVEL = 2^(MAX_NTT_LEVEL + 1) - 2 entries, which is 1022 values
// (about 4 KiB) for MAX_NTT_LEVEL = 9.

constexpr uint32_t MAX_NTT_LEVEL = 9;
constexpr uint32_t DEVICE_NTT_TWIDDLES_SIZE = (1u << (MAX_NTT_LEVEL + 1)) - 2;

// Defined in batch_ntt_small.cu, which also uploads its contents.
extern __constant__ Fp DEVICE_NTT_TWIDDLES[DEVICE_NTT_TWIDDLES_SIZE];

// Start of the block for level `lvl` (lvl >= 1).
__device__ __host__ __forceinline__ constexpr uint32_t twiddle_offset(uint32_t lvl) {
    return (1u << lvl) - 2;
}

// w_lvl^power, read from the table; valid for 1 <= lvl <= MAX_NTT_LEVEL and power < 2^lvl.
__device__ __forceinline__ Fp get_twiddle(uint32_t lvl, uint32_t power) {
    return DEVICE_NTT_TWIDDLES[twiddle_offset(lvl) + power];
}

__device__ __forceinline__ uint32_t linear_thread_lane() {
    uint32_t flat_tid =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return flat_tid & ((1u << LOG_WARP_SIZE) - 1u);
}

__device__ __forceinline__ unsigned ntt_subgroup_mask(uint32_t pos, uint32_t log_group) {
    if (log_group >= LOG_WARP_SIZE) {
        return 0xffffffffu;
    }

    uint32_t group_size = 1u << log_group;
    uint32_t lane_no = linear_thread_lane();
    uint32_t group_first_lane = lane_no - (pos & (group_size - 1u));
    return ((1u << group_size) - 1u) << group_first_lane;
}

template <bool inverse> __device__ __forceinline__ Fp sum_or_semi_sum(Fp &&v) {
    if constexpr (inverse) {
        return v.halve();
    } else {
        return v;
    }
}

// Size-2^log_size NTT computed jointly by 2^log_size threads, each owning one element.
// Decimation in frequency: input in natural order, output in bit-reversed order.
//
// Input is the evaluation at omega^pos. With `smem_path` it must already sit in `smem_buf[pos]`;
// without it, in `my_val`. Either way the result ends up in `my_val`. Domains larger than a warp go
// through shared memory, smaller ones use only shuffles.
//
// Threads with `active == false` carry no data but must still reach every barrier.
//
// Requirement: the threads of one transform occupy consecutive lanes. Whole transforms may be left
// out, but every lane covered by a shuffle mask has to execute the shuffle.
template <bool inverse, bool smem_path>
__device__ __forceinline__ void ntt_natural_to_bitrev(
    Fp &my_val,
    Fp *smem_buf, // this transform's shared-memory slots, 1 << log_size of them
    uint32_t const pos,      // this thread's element index, in [0, 1 << log_size)
    uint32_t const log_size, // log2 of the transform length
    bool const active = true
) {
    if (log_size == 0) {
        return;
    }

    uint32_t shuffle_levels;
    if constexpr (smem_path) {
        shuffle_levels = LOG_WARP_SIZE;
    } else {
        shuffle_levels = log_size;
    }
    unsigned lane_mask = ntt_subgroup_mask(pos, shuffle_levels);
    // The inverse transform walks the twiddle table backwards.
    uint32_t const tw_exp = inverse ? (pos ? (1 << log_size) - pos : 0) : pos;
    Fp tw = get_twiddle(log_size, tw_exp);

    // Levels above the warp (only when log_size > LOG_WARP_SIZE).
    if constexpr (smem_path) {
        for (uint32_t lvl = log_size; lvl-- > LOG_WARP_SIZE;) {
            if (active) {
                uint32_t const stride = 1u << lvl;
                if (!(pos & stride)) {
                    Fp const top = smem_buf[pos];
                    Fp const bottom = smem_buf[pos + stride];
                    smem_buf[pos] = sum_or_semi_sum<inverse>(top + bottom);
                    smem_buf[pos + stride] = sum_or_semi_sum<inverse>(top - bottom) * tw;
                }
                tw *= tw;
            }
            __syncthreads();
        }
        my_val = smem_buf[pos];
    }
    // Levels inside the warp, via shuffles.
    for (uint32_t lvl = shuffle_levels; lvl-- > 0;) {
        uint32_t const stride = 1u << lvl;
        Fp const partner =
            Fp::fromRaw(__shfl_xor_sync(lane_mask, my_val.asRaw(), stride));
        if (!(pos & stride)) {
            // Upper half of the butterfly: keep the sum.
            my_val = sum_or_semi_sum<inverse>(my_val + partner);
        } else {
            // Lower half: keep the twiddled difference.
            my_val = sum_or_semi_sum<inverse>(my_val - partner) * tw;
        }
        tw *= tw;
    }
}

// Decimation in time: input in bit-reversed order, output in natural order.
//
// Same thread layout as ntt_natural_to_bitrev. For both values of `smem_path` the input is taken
// from `my_val` (element number rev_len(pos, log_size)) and the result is returned there.
//
// Requirement: the threads of one transform occupy consecutive lanes, and every lane covered by a
// shuffle mask executes this call.
template <bool inverse, bool smem_path>
__device__ __forceinline__ void ntt_bitrev_to_natural(
    Fp &my_val,
    Fp *smem_buf,
    uint32_t const pos,
    uint32_t const log_size
) {
    uint32_t const shuffle_levels = smem_path ? LOG_WARP_SIZE : log_size;
    unsigned lane_mask = ntt_subgroup_mask(pos, shuffle_levels);

    // Warp-local levels, smallest stride first.
    for (uint32_t lvl = 0; lvl < shuffle_levels; lvl++) {
        uint32_t const stride = 1u << lvl;
        // Root of order 2^(lvl+1), raised to the thread's offset inside its butterfly group.
        uint32_t tw_exp = pos & (stride - 1);
        if constexpr (inverse)
            tw_exp = tw_exp ? ((2u << lvl) - tw_exp) : 0;
        Fp tw = get_twiddle(lvl + 1, tw_exp);

        Fp partner = Fp::fromRaw(__shfl_xor_sync(lane_mask, my_val.asRaw(), stride));

        if (pos & stride) {
            // Lower output: upper input minus twiddled lower input.
            Fp scaled = my_val * tw;
            my_val = sum_or_semi_sum<inverse>(partner - scaled);
        } else {
            // Upper output: upper input plus twiddled lower input.
            Fp scaled = partner * tw;
            my_val = sum_or_semi_sum<inverse>(my_val + scaled);
        }
    }

    // Remaining levels, LOG_WARP_SIZE .. log_size - 1, through shared memory.
    if constexpr (smem_path) {
        smem_buf[pos] = my_val;
        __syncthreads();

        for (uint32_t lvl = LOG_WARP_SIZE; lvl < log_size; lvl++) {
            uint32_t const stride = 1u << lvl;
            uint32_t tw_exp = pos & (stride - 1);
            if constexpr (inverse)
                tw_exp = tw_exp ? ((2u << lvl) - tw_exp) : 0;
            Fp tw = get_twiddle(lvl + 1, tw_exp);

            if (!(pos & stride)) {
                Fp top = smem_buf[pos];
                Fp scaled = smem_buf[pos + stride] * tw;
                smem_buf[pos] = sum_or_semi_sum<inverse>(top + scaled);
                smem_buf[pos + stride] = sum_or_semi_sum<inverse>(top - scaled);
            }
            __syncthreads();
        }
        my_val = smem_buf[pos];
    }
}

} // namespace device_ntt

// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

// Device kernels and host launchers for the fractional-sum GKR prover.

#include "field/fp.h"
#include "field/fpext.h"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"
#include "common/utils.cuh"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <utility>
#include <vector_types.h>

// Tree build, per-round sumcheck evaluation and folding for the GKR layers; the LogUp
// leaves fed into this prover are produced in interactions.cu.
namespace fractional_sumcheck_gkr {
// s'(X) is evaluated at X = 1 and X = 2 (the leading eq factor is pulled out).
constexpr int ROUND_EVALS = 2;
// ----------------------------------------------------------------------------
// Device side
// ----------------------------------------------------------------------------
template <bool undo, bool with_alpha = false>
__global__ void frac_build_tree_layer_kernel(
    FracExt *__restrict__ nodes,
    uint32_t h_off,
    FpExt den_shift = FpExt(Fp(0))
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= h_off) {
        return;
    }

    if constexpr (with_alpha) {
        // Shift both denominators by den_shift first, then merge the pair through
        // local copies.
        FracExt left_v = nodes[pos];
        FracExt right_v = nodes[pos | h_off];
        left_v.q = left_v.q + den_shift;
        right_v.q = right_v.q + den_shift;

        if constexpr (undo) {
            frac_unadd_inplace(left_v, right_v);
        } else {
            frac_add_inplace(left_v, right_v);
        }

        nodes[pos] = left_v;
    } else {
        // No shift: merge directly through references into the buffer.
        FracExt &left = nodes[pos];
        FracExt const &right = nodes[pos | h_off];

        if constexpr (undo) {
            frac_unadd_inplace(left, right);
        } else {
            frac_add_inplace(left, right);
        }
    }
}

// FpExt/FracExt have only 4-byte alignment, so plain struct copies compile to 4/8
// scalar accesses per element. All elements in these buffers sit at 16/32-byte-aligned
// offsets, so moving them as uint4 vectors is legal and byte-identical, with 4x fewer
// memory transactions (same mechanism as the vectorized k2 bit-reversal kernel).
union ExtVec16 {
    uint4 v;
    FpExt f;
    __device__ ExtVec16() {}
};
__device__ __forceinline__ FpExt ext_load16(const FpExt *in_ptr) {
    ExtVec16 u;
    u.v = *reinterpret_cast<const uint4 *>(in_ptr);
    return u.f;
}
__device__ __forceinline__ void ext_store16(FpExt *out_ptr, FpExt val) {
    ExtVec16 u;
    u.f = val;
    *reinterpret_cast<uint4 *>(out_ptr) = u.v;
}
union FracVec16 {
    uint4 v[2];
    FracExt f;
    __device__ FracVec16() {}
};
__device__ __forceinline__ FracExt frac_load16(const FracExt *in_ptr) {
    FracVec16 u;
    u.v[0] = reinterpret_cast<const uint4 *>(in_ptr)[0];
    u.v[1] = reinterpret_cast<const uint4 *>(in_ptr)[1];
    return u.f;
}
__device__ __forceinline__ void frac_store16(FracExt *out_ptr, FracExt val) {
    FracVec16 u;
    u.f = val;
    reinterpret_cast<uint4 *>(out_ptr)[0] = u.v[0];
    reinterpret_cast<uint4 *>(out_ptr)[1] = u.v[1];
}

// The eq table is stored as two factors: entry pos equals
// eq_lo[pos & (2^lo_bits - 1)] * eq_hi[pos >> lo_bits].
__device__ __forceinline__ FpExt eq_split_lookup(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    uint32_t const lo_bits,
    uint32_t const pos
) {
    return ext_load16(eq_lo + (pos & ((1u << lo_bits) - 1))) *
           ext_load16(eq_hi + (pos >> lo_bits));
}

// Add this point's contributions to the running sums for s'(1) and s'(2).
__device__ __forceinline__ void accum_round_terms(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    uint32_t pos,
    uint32_t lo_bits,
    FpExt lam,
    FpExt n0_at0, FpExt d0_at0, FpExt n0_at1, FpExt d0_at1,
    FpExt n1_at0, FpExt d1_at0, FpExt n1_at1, FpExt d1_at1,
    FpExt &acc_s1, FpExt &acc_s2
) {
    FpExt eq_w = eq_split_lookup(eq_lo, eq_hi, lo_bits, pos);

    FpExt n0_step = n0_at1 - n0_at0;
    FpExt d0_step = d0_at1 - d0_at0;
    FpExt n1_step = n1_at1 - n1_at0;
    FpExt d1_step = d1_at1 - d1_at0;

    FpExt n0_cur = n0_at0 + lam * d0_at0;
    FpExt d0_cur = d0_at0;
    FpExt n1_cur = n1_at0;
    FpExt d1_cur = d1_at0;

    auto const lam_d0_step = lam * d0_step;

#pragma unroll
    for (int i = 0; i < ROUND_EVALS; ++i) {
        n0_cur += n0_step;
        n0_cur += lam_d0_step;
        d0_cur += d0_step;
        n1_cur += n1_step;
        d1_cur += d1_step;

        FpExt term = eq_w * (n0_cur * d1_cur + n1_cur * d0_cur);
        if (i == 0) acc_s1 += term;
        else acc_s2 += term;
    }
}

// Block-reduce both running sums and have thread 0 write them to blk_sums.
__device__ __forceinline__ void store_pair_block_sums(
    FpExt *red_smem,
    FpExt acc_s1, FpExt acc_s2,
    FpExt *__restrict__ blk_sums
) {
    {
        FpExt blk_total = reduce::block_reduce_sum(acc_s1, red_smem);
        if (threadIdx.x == 0) blk_sums[blockIdx.x * ROUND_EVALS + 0] = blk_total;
    }
    __syncthreads();
    {
        FpExt blk_total = reduce::block_reduce_sum(acc_s2, red_smem);
        if (threadIdx.x == 0) blk_sums[blockIdx.x * ROUND_EVALS + 1] = blk_total;
    }
}

// Dynamic shared memory: one FpExt per warp (at least one).
// Each block writes its partial sums of s'(1) and s'(2) to blk_sums.
__global__ void compute_round_block_sum_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    const FracExt *__restrict__ pq_buf,
    uint32_t x_count,
    uint32_t lo_bits,
    FpExt lam,
    FpExt *__restrict__ blk_sums  // [gridDim.x][2]
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;

    const FpExt zero(Fp::zero());
    FpExt acc_s[ROUND_EVALS] = {zero, zero};
    uint32_t first_pos = threadIdx.x + blockIdx.x * blockDim.x;
    // Grid-stride loop accumulating per-thread partial sums.
    for (uint32_t pos = first_pos; pos < x_count / 2; pos += blockDim.x * gridDim.x) {
        FpExt eq_w = eq_split_lookup(eq_lo, eq_hi, lo_bits, pos);

        // Loads the four leaves (poly 0/1) x (current variable 0/1); the values at the
        // current variable's 0 and 1 points are extrapolated linearly to X = 1, 2.
        FracExt const at00 = frac_load16(pq_buf + pos);
        FracExt const at10 = frac_load16(pq_buf + with_rev_bits(pos, pq_len, 1, 0));
        FracExt const at01 = frac_load16(pq_buf + with_rev_bits(pos, pq_len, 0, 1));
        FracExt const at11 = frac_load16(pq_buf + with_rev_bits(pos, pq_len, 1, 1));
        auto const &[n0_at0, d0_at0] = at00;
        auto const &[n1_at0, d1_at0] = at10;
        auto const &[n0_at1, d0_at1] = at01;
        auto const &[n1_at1, d1_at1] = at11;

        FpExt n0_step = n0_at1 - n0_at0;
        FpExt d0_step = d0_at1 - d0_at0;
        FpExt n1_step = n1_at1 - n1_at0;
        FpExt d1_step = d1_at1 - d1_at0;

        FpExt n0_cur = n0_at0 + lam * d0_at0;
        FpExt d0_cur = d0_at0;
        FpExt n1_cur = n1_at0;
        FpExt d1_cur = d1_at0;

        auto const lam_d0_step = lam * d0_step;

#pragma unroll
        for (int i = 0; i < ROUND_EVALS; ++i) {
            n0_cur += n0_step;
            n0_cur += lam_d0_step;
            d0_cur += d0_step;
            n1_cur += n1_step;
            d1_cur += d1_step;

            acc_s[i] += eq_w * (n0_cur * d1_cur + n1_cur * d0_cur);
        }
    }
// Collapse each evaluation point across the block separately.
#pragma unroll
    for (int i = 0; i < ROUND_EVALS; ++i) {
        FpExt blk_total = reduce::block_reduce_sum(acc_s[i], red_smem);

        if (threadIdx.x == 0) {
            blk_sums[blockIdx.x * ROUND_EVALS + i] = blk_total;
        }
        __syncthreads();
    }
}

// Linear fold over the FpExt view of a FracExt buffer: numerator and denominator
// fold by the same rule, so the buffer can be handled as a flat FpExt array of twice
// the length.
//
// Thread pos combines (pos, pos + q_off) into out_ptr[pos] and
// (pos + h_off, pos + h_off + q_off) into out_ptr[pos + q_off]. Every thread reads its
// own inputs before writing, so in_ptr may equal out_ptr.
__global__ void fold_ef_columns_kernel(const FpExt *in_ptr, FpExt *out_ptr, uint32_t q_off, FpExt chal) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= q_off) {
        return;
    }

    FpExt x0 = ext_load16(in_ptr + pos);
    FpExt x1 = ext_load16(in_ptr + (pos | q_off));
    ext_store16(out_ptr + pos, x0 + chal * (x1 - x0));

    uint32_t h_off = q_off << 1;

    FpExt x2 = ext_load16(in_ptr + (pos | h_off));
    FpExt x3 = ext_load16(in_ptr + (pos | h_off | q_off));
    ext_store16(out_ptr + (pos | q_off), x2 + chal * (x3 - x2));
}

// One inner sumcheck round with the previous round's fold merged in (separate buffers).
// pq_in holds the unfolded layer (2 * pq_len entries); each thread folds its entries by
// fold_r, stores them to pq_out (pq_len entries) and accumulates s'(1), s'(2).
//
// Merging the fold into the next round saves a launch per round and one full pass
// over the buffer.
//
// pq_in and pq_out must be distinct allocations; the in-place variant below handles
// the aliased case.
//
// Each pair is loaded, folded and stored inside its own small scope so that only the
// folded scalars stay live.
//
// Dynamic shared memory: one FpExt per warp (at least one).
__global__ void compute_round_and_fold_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    const FracExt *__restrict__ pq_in,  // unfolded, 2 * pq_len entries
    uint32_t x_count,  // after the fold, pq_len / 2
    uint32_t lo_bits,
    FpExt lam,
    FpExt fold_r,  // challenge of the previous round
    FpExt *__restrict__ blk_sums,  // [gridDim.x * 2]
    FracExt *__restrict__ pq_out  // folded, pq_len entries
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;

    // Offsets of the four leaves; the in-place kernel below derives them.
    const uint32_t h_off = pq_len;  // half of the unfolded length
    const uint32_t q_off = pq_len >> 1;  // pq_len / 2
    const uint32_t e_off = pq_len >> 2;  // pq_len / 4
    const uint32_t e3_off = e_off * 3;  // 3 * pq_len / 4

    // Two scalar sums rather than an array keeps them in registers.
    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;
    uint32_t first_pos = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid-stride loop accumulating per-thread partial sums.
    for (uint32_t pos = first_pos; pos < x_count / 2; pos += blockDim.x * gridDim.x) {
        // Folded leaf values consumed by the round evaluation.
        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;

        // Fold each pair and store it right away.
        // leaf (0,0) -> pos
        {
            FracExt lo_f = frac_load16(pq_in + pos);
            FracExt hi_f = frac_load16(pq_in + pos + q_off);
            n0_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store16(pq_out + pos, {n0_at0, d0_at0});
        }
        // leaf (1,0) -> pos + q_off
        {
            FracExt lo_f = frac_load16(pq_in + pos + h_off);
            FracExt hi_f = frac_load16(pq_in + pos + h_off + q_off);
            n1_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store16(pq_out + pos + q_off, {n1_at0, d1_at0});
        }
        // leaf (0,1) -> pos + e_off
        {
            FracExt lo_f = frac_load16(pq_in + pos + e_off);
            FracExt hi_f = frac_load16(pq_in + pos + e3_off);
            n0_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store16(pq_out + pos + e_off, {n0_at1, d0_at1});
        }
        // leaf (1,1) -> pos + e3_off
        {
            FracExt lo_f = frac_load16(pq_in + pos + h_off + e_off);
            FracExt hi_f = frac_load16(pq_in + pos + h_off + e3_off);
            n1_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store16(pq_out + pos + e3_off, {n1_at1, d1_at1});
        }

        accum_round_terms(
            eq_lo, eq_hi, pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }

    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// Same as compute_round_and_fold_kernel above, but the folded layer overwrites the first pq_len
// entries of pq (which holds 2 * pq_len unfolded entries on entry).
//
// Why in place is safe: a thread only stores to {pos, pos+q_off, pos+e_off, pos+e3_off},
// which are exactly the low-half slots it loaded itself, and distinct threads own distinct
// pos. The high half [pq_len, 2 * pq_len) is read but never stored to.
//
// pq carries no __restrict__ because loads and stores go through the same pointer.
//
// Dynamic shared memory: one FpExt per warp (at least one).
__global__ void compute_round_and_fold_inplace_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    FracExt *pq,  // reads 2 * pq_len entries, rewrites the first pq_len
    uint32_t x_count,  // after the fold, pq_len / 2
    uint32_t lo_bits,
    FpExt lam,
    FpExt fold_r,  // challenge of the previous round
    FpExt *__restrict__ blk_sums  // [gridDim.x * 2]
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;

    // Where the leaves sit: viewed as FpExt, a FracExt slot i occupies elements 2i, 2i+1.
    // The unfolded layer has 2 * pq_len FracExt (4 * pq_len FpExt); the fold
    // maps
    //   out[j] = fold(in[j], in[j + pq_len/2])            for j < pq_len/2
    //   out[j] = fold(in[j + pq_len/2], in[j + pq_len])   for pq_len/2 <= j < pq_len
    //
    // The round then reads folded slots pos, pos + pq_len/2, pos + pq_len/4 and
    // pos + 3*pq_len/4.
    const uint32_t h_off = pq_len;  // half of the unfolded length
    const uint32_t q_off = pq_len >> 1;  // pq_len / 2
    const uint32_t e_off = pq_len >> 2;  // pq_len / 4
    const uint32_t e3_off = e_off * 3;  // 3 * pq_len / 4

    // Two scalar sums rather than an array keeps them in registers.
    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;
    uint32_t first_pos = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid-stride loop accumulating per-thread partial sums.
    for (uint32_t pos = first_pos; pos < x_count / 2; pos += blockDim.x * gridDim.x) {
        // All eight inputs are loaded before any store (in-place), but each pair is
        // folded immediately so only four folded fractions remain live; this cut the
        // register count from roughly 107 to 60-70 per thread.

        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;

        // leaf (0,0): slots pos, pos + q_off
        {
            FracExt lo_f = frac_load16(pq + pos);
            FracExt hi_f = frac_load16(pq + pos + q_off);
            n0_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (1,0): slots pos + h_off, pos + h_off + q_off
        {
            FracExt lo_f = frac_load16(pq + pos + h_off);
            FracExt hi_f = frac_load16(pq + pos + h_off + q_off);
            n1_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (0,1): slots pos + e_off, pos + e3_off
        {
            FracExt lo_f = frac_load16(pq + pos + e_off);
            FracExt hi_f = frac_load16(pq + pos + e3_off);
            n0_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (1,1): slots pos + h_off + e_off, pos + h_off + e3_off
        {
            FracExt lo_f = frac_load16(pq + pos + h_off + e_off);
            FracExt hi_f = frac_load16(pq + pos + h_off + e3_off);
            n1_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // Store the folded leaves; every load of this thread has already happened.
        frac_store16(pq + pos, {n0_at0, d0_at0});
        frac_store16(pq + pos + q_off, {n1_at0, d1_at0});
        frac_store16(pq + pos + e_off, {n0_at1, d0_at1});
        frac_store16(pq + pos + e3_off, {n1_at1, d1_at1});

        accum_round_terms(
            eq_lo, eq_hi, pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }

    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// First inner round of a layer: undoes the tree merge and evaluates the round in one
// pass (frac_build_tree_layer_kernel<undo = true> then compute_round_block_sum_kernel).
//
// The low half of nodes holds merged fractions and the high half the right operands;
// each low slot i is restored as frac_unadd(nodes[i], nodes[i + h_off]), leaving the
// high half as it is.
//
// Slots touched per thread:
//   pos, pos | q_off                  (low half, restored and stored back)
//   pos | h_off, pos | h_off | q_off  (high half, read only)
//
// Since pos ranges over [0, x_count/2), the pairs {pos, pos + q_off} partition [0, h_off),
// so each low slot has exactly one writer.
//
// Dynamic shared memory: one FpExt per warp (at least one).
__global__ void compute_round_and_revert_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    FracExt *__restrict__ nodes,  // low half restored in place
    uint32_t x_count,
    uint32_t lo_bits,
    FpExt lam,
    FpExt *__restrict__ blk_sums  // [gridDim.x * 2]
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;
    const uint32_t h_off = pq_len >> 1;  // pq_len / 2
    const uint32_t q_off = pq_len >> 2;  // pq_len / 4

    // Two scalar sums rather than an array keeps them in registers.
    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;
    uint32_t first_pos = threadIdx.x + blockIdx.x * blockDim.x;

    // Grid-stride loop accumulating per-thread partial sums.
    for (uint32_t pos = first_pos; pos < x_count / 2; pos += blockDim.x * gridDim.x) {
        // Unpack each FracExt into scalars right away rather than keeping four FracExt
        // values around, which lowers register pressure.

        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;

        // leaf (0,0) recovered from its merge with leaf (1,0)
        {
            FracExt left = frac_load16(nodes + pos);
            FracExt right = frac_load16(nodes + pos + h_off);
            FpExt right_q_inv = inv(right.q);
            d0_at0 = left.q * right_q_inv;
            n0_at0 = (left.p - d0_at0 * right.p) * right_q_inv;
            n1_at0 = right.p;
            d1_at0 = right.q;
            frac_store16(nodes + pos, {n0_at0, d0_at0});
        }

        // leaf (0,1) recovered from its merge with leaf (1,1)
        {
            FracExt left = frac_load16(nodes + pos + q_off);
            FracExt right = frac_load16(nodes + pos + h_off + q_off);
            FpExt right_q_inv = inv(right.q);
            d0_at1 = left.q * right_q_inv;
            n0_at1 = (left.p - d0_at1 * right.p) * right_q_inv;
            n1_at1 = right.p;
            d1_at1 = right.q;
            frac_store16(nodes + pos + q_off, {n0_at1, d0_at1});
        }

        accum_round_terms(
            eq_lo, eq_hi, pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }

    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// Tail points staged in shared memory per iteration of precompute_m_build_partial_kernel.
// One stage takes (4 * (2^w + 1) * TAIL_CHUNK + TAIL_CHUNK) FpExt: four padded operand tiles
// plus the eq weights. For w = 3 and a chunk of 16 that is about 9.25 KB. A bigger
// chunk means fewer barriers per point but more shared memory per block, which can
// cost occupancy or make the launch fail.
constexpr uint32_t TAIL_CHUNK = 16;

// Partial M matrices: block x sums over its own range [t_begin, t_end) of tail points
// and writes a complete 2^w x 2^w matrix. Thread (ri, ci) of the 2D block owns entry
// (ri, ci). Shared tiles have row pitch side + 1 so that column reads do not conflict.
template <bool fold_inline, uint32_t LOGW>
__global__ void precompute_m_build_partial_kernel(
    const FracExt *__restrict__ pq,
    uint32_t n_rem,  // variable count after the fold
    FpExt lam,
    FpExt fold_r,  // only read when fold_inline
    const FpExt *__restrict__ tail_eq_lo,
    const FpExt *__restrict__ tail_eq_hi,
    uint32_t tail_lo_bits,
    uint32_t tile_len,
    FpExt *__restrict__ partials
) {
    constexpr uint32_t side = 1u << LOGW;
    uint32_t ri = threadIdx.y;
    uint32_t ci = threadIdx.x;
    if (ri >= side || ci >= side) {
        return;
    }

    uint32_t n_tail = n_rem - LOGW;
    uint32_t tail_len = 1u << n_tail;
    // With fold_inline the buffer still has n_rem + 1 variables and is folded here;
    // otherwise it already has n_rem variables.
    uint32_t fold_off = fold_inline ? (1u << n_rem) : 0;
    uint32_t poly_off = fold_inline ? (1u << (n_rem + 1)) : (1u << n_rem);

    uint32_t t_begin = blockIdx.x * tile_len;
    uint32_t t_end = min(t_begin + tile_len, tail_len);

    // Odd pitch (9 for side = 8) spreads a column across banks.
    constexpr uint32_t pad_stride = side + 1;
    extern __shared__ FpExt smem_pool[];
    FpExt *s_l0 = smem_pool;
    FpExt *s_l1 = s_l0 + pad_stride * TAIL_CHUNK;
    FpExt *s_r0 = s_l1 + pad_stride * TAIL_CHUNK;
    FpExt *s_r1 = s_r0 + pad_stride * TAIL_CHUNK;
    FpExt *s_w = s_r1 + pad_stride * TAIL_CHUNK;

    FpExt zero(Fp::zero());
    constexpr uint32_t ACC_LANES = 4;
    FpExt accs[ACC_LANES];
#pragma unroll
    for (uint32_t ai = 0; ai < ACC_LANES; ++ai) accs[ai] = zero;

    uint32_t flat_tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t n_threads = blockDim.x * blockDim.y;

    for (uint32_t chunk0 = t_begin; chunk0 < t_end; chunk0 += TAIL_CHUNK) {
        uint32_t chunk_len = min(TAIL_CHUNK, t_end - chunk0);
        uint32_t chunk_elems = chunk_len * side;

        // Stage the eq weights of this chunk of tail points.
        for (uint32_t slot = flat_tid; slot < chunk_len; slot += n_threads) {
            uint32_t tail_pt = chunk0 + slot;
            s_w[slot] = eq_split_lookup(
                tail_eq_lo, tail_eq_hi, tail_lo_bits, tail_pt
            );
        }
        __syncthreads();

        // Stage the operands (folded on the fly if needed); the left ones carry the weight.
        for (uint32_t pos = flat_tid; pos < chunk_elems; pos += n_threads) {
            uint32_t hi_bits = pos / chunk_len;
            uint32_t slot = pos - hi_bits * chunk_len;
            uint32_t tail_pt = chunk0 + slot;
            uint32_t flat = hi_bits * tail_len + tail_pt;
            FpExt n0, d0, n1, d1;
            if constexpr (fold_inline) {
                // Fold each component as lo + fold_r * (hi - lo).
                FracExt lo0 = frac_load16(&pq[flat]);
                FracExt hi0 = frac_load16(&pq[flat + fold_off]);
                FracExt lo1 = frac_load16(&pq[poly_off + flat]);
                FracExt hi1 = frac_load16(&pq[poly_off + flat + fold_off]);
                n0 = lo0.p + fold_r * (hi0.p - lo0.p);
                d0 = lo0.q + fold_r * (hi0.q - lo0.q);
                n1 = lo1.p + fold_r * (hi1.p - lo1.p);
                d1 = lo1.q + fold_r * (hi1.q - lo1.q);
            } else {
                // Already folded: plain loads.
                FracExt fr0 = frac_load16(&pq[flat]);
                FracExt fr1 = frac_load16(&pq[poly_off + flat]);
                n0 = fr0.p; d0 = fr0.q;
                n1 = fr1.p; d1 = fr1.q;
            }
            uint32_t cell = slot * pad_stride + hi_bits;
            FpExt w_b = s_w[slot];
            s_l0[cell] = w_b * (n0 + lam * d0);
            s_l1[cell] = w_b * n1;
            s_r0[cell] = d0;
            s_r1[cell] = d1;
        }
        __syncthreads();

        // Spread the products over ACC_LANES independent sums so the long FpExt
        // multiply latency overlaps; field addition is exact, so the result is unchanged.
        uint32_t slot = 0;
        for (; slot + ACC_LANES <= chunk_len; slot += ACC_LANES) {
#pragma unroll
            for (uint32_t lane_j = 0; lane_j < ACC_LANES; ++lane_j) {
                uint32_t base_row = (slot + lane_j) * pad_stride;
                accs[lane_j] += s_l0[base_row + ri] * s_r1[base_row + ci]
                        + s_l1[base_row + ri] * s_r0[base_row + ci];
            }
        }
        for (; slot < chunk_len; ++slot) {
            uint32_t base_row = slot * pad_stride;
            accs[0] += s_l0[base_row + ri] * s_r1[base_row + ci]
                    + s_l1[base_row + ri] * s_r0[base_row + ci];
        }

        __syncthreads();
    }

    FpExt m_entry = accs[0];
#pragma unroll
    for (uint32_t ai = 1; ai < ACC_LANES; ++ai) m_entry += accs[ai];
    partials[blockIdx.x * (side * side) + ri * side + ci] = m_entry;
}

template <bool fold_inline, uint32_t LOGW>
inline void launch_m_window_partial(
    dim3 launch_grid,
    const FracExt *pq,
    uint32_t n_rem,
    FpExt lam,
    FpExt fold_r,
    const FpExt *tail_eq_lo,
    const FpExt *tail_eq_hi,
    uint32_t tail_lo_bits,
    uint32_t tile_len,
    FpExt *partials
) {
    constexpr uint32_t side = 1u << LOGW;
    dim3 launch_block(side, side);
    constexpr uint32_t pad_stride = side + 1;  // same padded pitch as the kernel
    size_t smem_bytes =
        (4 * pad_stride * TAIL_CHUNK + TAIL_CHUNK) * sizeof(FpExt);
    precompute_m_build_partial_kernel<fold_inline, LOGW><<<launch_grid, launch_block, smem_bytes>>>(
        pq,
        n_rem,
        lam,
        fold_r,
        tail_eq_lo,
        tail_eq_hi,
        tail_lo_bits,
        tile_len,
        partials
    );
}

template <bool fold_inline>
inline int dispatch_m_window_partial(
    uint32_t wbits,
    dim3 launch_grid,
    const FracExt *pq,
    uint32_t n_rem,
    FpExt lam,
    FpExt fold_r,
    const FpExt *tail_eq_lo,
    const FpExt *tail_eq_hi,
    uint32_t tail_lo_bits,
    uint32_t tile_len,
    FpExt *partials
) {
    using PartialFn = void (*)(
        dim3, const FracExt *, uint32_t, FpExt, FpExt, const FpExt *, const FpExt *, uint32_t,
        uint32_t, FpExt *
    );
    static constexpr PartialFn table[] = {
        &launch_m_window_partial<fold_inline, 1>,
        &launch_m_window_partial<fold_inline, 2>,
        &launch_m_window_partial<fold_inline, 3>,
        &launch_m_window_partial<fold_inline, 4>,
        &launch_m_window_partial<fold_inline, 5>,
    };

    constexpr uint32_t w_lo = 1;
    constexpr uint32_t w_hi = static_cast<uint32_t>(sizeof(table) / sizeof(table[0]));
    if (wbits < w_lo || wbits > w_hi) {
        // 2^w x 2^w threads per block: w > 5 would exceed the 1024-thread limit.
        return cudaErrorInvalidValue;
    }

    table[wbits - w_lo](
        launch_grid,
        pq,
        n_rem,
        lam,
        fold_r,
        tail_eq_lo,
        tail_eq_hi,
        tail_lo_bits,
        tile_len,
        partials
    );
    return 0;
}

// Sum the per-block partial matrices into m_mat.
__global__ void precompute_m_reduce_partials_kernel(
    const FpExt *__restrict__ part_in,
    uint32_t n_parts,
    uint32_t n_entries,
    FpExt *__restrict__ m_mat
) {
    uint32_t slot_e = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_e >= n_entries) {
        return;
    }

    FpExt zero(Fp::zero());
    FpExt sum_e = zero;
    // Consecutive threads read consecutive entries of each partial matrix, so the
    // loads coalesce.
    for (uint32_t blk_i = 0; blk_i < n_parts; ++blk_i) {
        sum_e += part_in[(size_t)blk_i * n_entries + slot_e];
    }
    m_mat[slot_e] = sum_e;
}

// One round of the M window: combines m_mat with the eq prefix/suffix weights.
// Writes evals_out[0] = s'(1) and evals_out[1] = s'(2).
__global__ void precompute_m_eval_round_kernel(
    const FpExt *__restrict__ m_mat,
    uint32_t wbits,
    uint32_t rnd,
    const FpExt *__restrict__ pre_eq,
    const FpExt *__restrict__ suf_eq,
    FpExt *__restrict__ evals_out
) {
    extern __shared__ FpExt red_smem[];

    uint32_t side = 1u << wbits;
    uint32_t pre_bits = rnd;
    uint32_t suf_bits = wbits - rnd - 1;
    uint32_t pre_len = 1u << pre_bits;
    uint32_t suf_len = 1u << suf_bits;

    uint64_t n_terms = (uint64_t)pre_len * pre_len * suf_len;
    uint32_t split_bit = 1u << suf_bits;
    FpExt zero(Fp::zero());
    FpExt one_e(Fp::one());
    FpExt two_e = one_e + one_e;

    FpExt sum_at1 = zero;
    FpExt sum_at2 = zero;

    for (uint64_t pos = threadIdx.x; pos < n_terms; pos += blockDim.x) {
        uint32_t suf = (uint32_t)(pos % suf_len);
        uint64_t rest = pos / suf_len;
        uint32_t pb = (uint32_t)(rest % pre_len);
        uint32_t pa = (uint32_t)(rest / pre_len);

        FpExt wgt = pre_eq[pa] * pre_eq[pb] * suf_eq[suf];

        uint32_t pre_shift = suf_bits + 1;
        uint32_t ra0 = (pa << pre_shift) | suf;
        uint32_t ra1 = ra0 | split_bit;
        uint32_t cb0 = (pb << pre_shift) | suf;
        uint32_t cb1 = cb0 | split_bit;

        FpExt e00 = m_mat[ra0 * side + cb0];
        FpExt e01 = m_mat[ra0 * side + cb1];
        FpExt e10 = m_mat[ra1 * side + cb0];
        FpExt e11 = m_mat[ra1 * side + cb1];

        sum_at1 += wgt * e11;
        sum_at2 += wgt * (e00 - two_e * (e01 + e10 - e11 - e11));
    }

    {
        FpExt blk_total = reduce::block_reduce_sum(sum_at1, red_smem);
        if (threadIdx.x == 0) evals_out[0] = blk_total;
    }
    __syncthreads();
    {
        FpExt blk_total = reduce::block_reduce_sum(sum_at2, red_smem);
        if (threadIdx.x == 0) evals_out[1] = blk_total;
    }
}

// Folds wbits variables in a single pass using the eq table of the window challenges;
// each thread produces column col of both polynomials.
template <uint32_t LOGW>
__global__ void multifold_kernel(
    const FracExt *in_ptr,
    FracExt *out_ptr,
    uint32_t n_cols,
    const FpExt *__restrict__ win_eq
) {
    constexpr uint32_t n_beta = 1u << LOGW;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }

    FpExt zero(Fp::zero());
    FpExt n0_sum = zero;
    FpExt d0_sum = zero;
    FpExt n1_sum = zero;
    FpExt d1_sum = zero;
    const FracExt *src_hi = in_ptr + ((size_t)n_cols << LOGW);
    for (uint32_t bk = 0; bk < n_beta; ++bk) {
        uint32_t pos = bk * n_cols + col;
        FracExt fv0 = in_ptr[pos];
        FracExt fv1 = src_hi[pos];
        FpExt wgt_r = win_eq[bk];
        n0_sum += wgt_r * fv0.p;
        d0_sum += wgt_r * fv0.q;
        n1_sum += wgt_r * fv1.p;
        d1_sum += wgt_r * fv1.q;
    }
    out_ptr[col] = {n0_sum, d0_sum};
    out_ptr[n_cols + col] = {n1_sum, d1_sum};
}

__global__ void add_alpha_kernel(FracExt *fracs, size_t n_fracs, FpExt den_shift) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < n_fracs) {
        fracs[pos].q = fracs[pos].q + den_shift;
    }
}

template <typename ScalarT, typename ExtT>
__global__ void frac_vector_scalar_multiply_kernel(
    std::pair<ExtT, ExtT> *pair_buf,
    ScalarT factor,
    uint32_t count
) {
    size_t slot_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot_i >= count)
        return;

    pair_buf[slot_i].first *= factor;
}

// Two tree layers merged into one pass. The intermediate right halves of the first
// layer are kept in the buffer because undoing the second layer needs them.
//
// Let h_outer = N >> (i+1) and h_inner = N >> (i+2) for layers i, i+1. Thread t < h_inner
// loads fa = nodes[t], fb = nodes[t + h_inner], fc = nodes[t + h_outer] and
// fd = nodes[t + h_outer + h_inner], then forms
//   left   = fa + fc   (layer i at t, never stored)
//   right  = fb + fd   (layer i at t + h_inner)
//   merged = left + right (layer i+1 at t)
// and stores merged to nodes[t] and right to nodes[t + h_inner]. fc and fd stay where
// they are, which is what undoing layer i requires.
//
// Per group of four this does 4 loads + 2 stores instead of 6 loads + 3 stores for two
// single-layer passes, about a third less traffic.
__global__ void frac_build_tree_two_layers_kernel(
    FracExt *__restrict__ nodes,
    uint32_t h_inner  // N >> (i+2), i = first of the two layers
) {
    uint32_t lane_t = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane_t >= h_inner) return;

    uint32_t h_outer = h_inner << 1;  // N >> (i+1)

    FracExt fa = frac_load16(nodes + lane_t);
    FracExt fb = frac_load16(nodes + lane_t + h_inner);
    FracExt fc = frac_load16(nodes + lane_t + h_outer);
    FracExt fd = frac_load16(nodes + lane_t + h_outer + h_inner);

    // layer i (results stay in registers)
    FracExt left = frac_add(fa, fc);
    FracExt right = frac_add(fb, fd);
    // layer i+1
    FracExt merged = frac_add(left, right);

    frac_store16(nodes + lane_t, merged);
    frac_store16(nodes + lane_t + h_inner, right);  // kept for undoing layer i+1
    // nodes[t + h_outer] and nodes[t + h_outer + h_inner] are left untouched; undoing
    // layer i reads them.
}

// ----------------------------------------------------------------------------
// Host side
// ----------------------------------------------------------------------------
template <bool undo, bool with_alpha>
int launch_frac_build_tree_layer(FracExt *nodes, size_t node_count, FpExt den_shift) {
    auto [launch_grid, launch_block] = kernel_launch_params(node_count);
    frac_build_tree_layer_kernel<undo, with_alpha><<<launch_grid, launch_block>>>(nodes, node_count, den_shift);
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_build_tree_layer(
    FracExt *nodes,
    size_t node_count,
    bool undo,
    FpExt den_shift,
    bool with_alpha
) {
    if (node_count == 0) {
        return 0;
    }
    assert(node_count % 2 == 0);
    node_count /= 2;

    return DISPATCH_BOOL_PAIR(
        launch_frac_build_tree_layer, undo, with_alpha, nodes, node_count, den_shift
    );
}

// Two-layer tree build. h_inner = N >> (i+2) with i the first of the two layers;
// needs h_inner >= 1 and 4 * h_inner <= buffer length.
extern "C" int _kb_frac_build_tree_two_layers(FracExt *nodes, size_t h_inner) {
    if (h_inner == 0) return 0;
    // 256 threads per block leave more registers per thread than 1024 would; this
    // kernel keeps four fractions and three merges in flight, more than the
    // single-layer kernel.
    auto [launch_grid, launch_block] = kernel_launch_params(h_inner, 256);
    frac_build_tree_two_layers_kernel<<<launch_grid, launch_block>>>(nodes, (uint32_t)h_inner);
    return CHECK_KERNEL();
}

inline uint32_t device_block_goal(uint32_t per_sm, uint32_t fallback_n) {
    int dev_id = 0;
    if (cudaGetDevice(&dev_id) == cudaSuccess) {
        int n_sms = 0;
        if (cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev_id) ==
                cudaSuccess &&
            n_sms > 0) {
            return static_cast<uint32_t>(n_sms) * per_sm;
        }
    }
    return fallback_n;
}

inline std::pair<dim3, dim3> round_launch_shape(uint32_t x_count) {
    // Small x_count would give too few blocks at the default size, so the block size
    // is lowered until there are enough blocks to cover the SMs.

    uint32_t work_items = x_count >> 1;

    // Aim for two blocks per SM, which hides memory latency without splitting the
    // work too finely; small inputs get smaller blocks to reach that count.
    constexpr uint32_t PER_SM_GOAL = 2;
    constexpr uint32_t FALLBACK_GOAL = 228;
    uint32_t block_goal =
        device_block_goal(
            PER_SM_GOAL, FALLBACK_GOAL
        );
    constexpr uint32_t TPB_DEFAULT = 256;
    constexpr uint32_t TPB_FLOOR = 64;  // occupancy floor

    uint32_t tpb_sel = TPB_DEFAULT;
    uint32_t n_blocks_sel = (work_items + tpb_sel - 1) / tpb_sel;

    // Too few blocks at the default size: shrink the block.
    if (n_blocks_sel < block_goal && work_items >= TPB_FLOOR) {
        // block size that would just reach the goal,
        tpb_sel = (work_items + block_goal - 1) / block_goal;
        // rounded up to whole warps,
        tpb_sel = std::max(TPB_FLOOR, ((tpb_sel + 31) / 32) * 32);
        // and the matching block count.
        n_blocks_sel = (work_items + tpb_sel - 1) / tpb_sel;
    }

    return {dim3(n_blocks_sel), dim3(tpb_sel)};
}

extern "C" uint32_t _kb_frac_compute_round_temp_buffer_size(uint32_t x_count) {
    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    return launch_grid.x * ROUND_EVALS;
}

// Host helper: second-stage reduction of the per-block sums via
// reduce::static_final_reduce_block_sums.
inline int gkr_final_reduce(FpExt *scratch_sums, FpExt *evals_out, uint32_t n_parts) {
    auto [ignored_grid, fin_block] = kernel_launch_params(n_parts);
    (void)ignored_grid;
    unsigned int fin_warps = div_ceil(fin_block.x, WARP_SIZE);
    size_t fin_smem = std::max(1u, fin_warps) * sizeof(FpExt);
    reduce::static_final_reduce_block_sums<ROUND_EVALS>
        <<<ROUND_EVALS, fin_block, fin_smem>>>(scratch_sums, evals_out, n_parts);
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_compute_round(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    const FracExt *pq_buf,
    size_t x_count,
    size_t eq_lo_len,
    FpExt lam,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(x_count > 1);

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // first stage: per-block partial sums into scratch_sums
    compute_round_block_sum_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        pq_buf,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        scratch_sums
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // second stage: fold scratch_sums into evals_out
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

// Launcher for the first inner round: undo the tree merge and evaluate the round
// in one kernel.
extern "C" int _kb_frac_compute_round_and_revert(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    FracExt *nodes,  // low half restored in place
    size_t x_count,
    size_t eq_lo_len,
    FpExt lam,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(x_count > 1);

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // undo + round evaluation
    compute_round_and_revert_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        nodes,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        scratch_sums
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // second stage reduction
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

// Launcher for a round with the fold merged in (separate buffers). src_len is the
// unfolded FracExt count (2 * pq_len); after folding x_count = src_len / 4.
extern "C" int _kb_frac_compute_round_and_fold(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    const FracExt *pq_in_buf,
    FracExt *pq_out_buf,
    size_t src_len,  // unfolded FracExt count
    size_t eq_lo_len,
    FpExt lam,
    FpExt fold_r,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(src_len > 2);
    // sizes after the fold
    size_t pq_len = src_len >> 1;
    size_t x_count = pq_len >> 1;
    assert(x_count > 0);

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // round + fold; fills scratch_sums and pq_out_buf
    compute_round_and_fold_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        pq_in_buf,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        fold_r,
        scratch_sums,
        pq_out_buf
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // second stage: fold scratch_sums into evals_out
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

// In-place variant of the launcher above: the folded layer overwrites the first half
// of pq_buf. src_len is the unfolded FracExt count; after folding x_count = src_len / 4.
extern "C" int _kb_frac_compute_round_and_fold_inplace(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    FracExt *pq_buf,  // unfolded on entry, folded low half on exit
    size_t src_len,  // unfolded FracExt count
    size_t eq_lo_len,
    FpExt lam,
    FpExt fold_r,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(src_len > 2);
    // sizes after the fold
    size_t pq_len = src_len >> 1;
    size_t x_count = pq_len >> 1;
    assert(x_count > 0);

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);

    // round + in-place fold; fills scratch_sums and the low half of pq_buf
    compute_round_and_fold_inplace_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        pq_buf,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        fold_r,
        scratch_sums
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // second stage: fold scratch_sums into evals_out
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

extern "C" int _kb_frac_precompute_m_build(
    const FracExt *pq,
    size_t n_rem,  // variable count after the fold
    size_t wbits,
    FpExt lam,
    FpExt fold_r,  // only read when fold_inline
    bool fold_inline,  // true: pq still has n_rem + 1 variables
    const FpExt *tail_eq_lo,
    const FpExt *tail_eq_hi,
    size_t tail_eq_lo_len,
    size_t tile_len,
    FpExt *partials,
    size_t partials_len,
    FpExt *m_mat
) {
    assert(n_rem > 0);
    assert(wbits > 0 && wbits <= n_rem);
    assert(tile_len > 0);

    uint32_t w32 = (uint32_t)wbits;
    uint32_t n_rem32 = (uint32_t)n_rem;
    uint32_t side = 1u << w32;
    uint32_t n_tail = (uint32_t)n_rem - (uint32_t)wbits;
    uint32_t tail_len = 1u << n_tail;
    uint32_t tile32 = (uint32_t)tile_len;
    uint32_t n_parts = div_ceil(tail_len, tile32);
    size_t n_entries = (size_t)side * (size_t)side;
    assert(partials_len >= (size_t)n_parts * n_entries);

    dim3 launch_grid(n_parts);
    uint32_t tail_lo_bits = __builtin_ctz((uint32_t)tail_eq_lo_len);
    int dispatch_rc = 0;
    if (fold_inline) {
        dispatch_rc = dispatch_m_window_partial<true>(
            w32,
            launch_grid,
            pq,
            n_rem32,
            lam,
            fold_r,
            tail_eq_lo,
            tail_eq_hi,
            tail_lo_bits,
            tile32,
            partials
        );
    } else {
        dispatch_rc = dispatch_m_window_partial<false>(
            w32,
            launch_grid,
            pq,
            n_rem32,
            lam,
            fold_r,
            tail_eq_lo,
            tail_eq_hi,
            tail_lo_bits,
            tile32,
            partials
        );
    }
    if (dispatch_rc != 0) {
        return dispatch_rc;
    }
    int rc = CHECK_KERNEL();
    if (rc != 0) {
        return rc;
    }

    dim3 fin_block(128);
    dim3 fin_grid(div_ceil((uint32_t)n_entries, fin_block.x));
    precompute_m_reduce_partials_kernel<<<fin_grid, fin_block>>>(
        partials,
        n_parts,
        (uint32_t)n_entries,
        m_mat
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_precompute_m_eval_round(
    const FpExt *m_mat,
    size_t wbits,
    size_t rnd,
    const FpExt *pre_eq,
    const FpExt *suf_eq,
    FpExt *evals_out
) {
    assert(wbits > 0);
    assert(rnd < wbits);

    dim3 launch_block(256);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);
    precompute_m_eval_round_kernel<<<1, launch_block, smem_bytes>>>(
        m_mat,
        (uint32_t)wbits,
        (uint32_t)rnd,
        pre_eq,
        suf_eq,
        evals_out
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_multifold(
    const FracExt *in_ptr,
    FracExt *out_ptr,
    size_t n_rem,
    size_t wbits,
    const FpExt *win_eq
) {
    assert(n_rem > 0);
    assert(wbits > 0 && wbits <= n_rem);

    size_t col_count = 1u << (n_rem - wbits);
    auto [launch_grid, launch_block] = kernel_launch_params(col_count, 256);

#define DISPATCH_MULTIFOLD(LOGW) \
    multifold_kernel<LOGW><<<launch_grid, launch_block>>>(in_ptr, out_ptr, (uint32_t)col_count, win_eq); \
    return CHECK_KERNEL();

    switch (wbits) {
        case 2: { DISPATCH_MULTIFOLD(2) }
        case 3: { DISPATCH_MULTIFOLD(3) }
        case 4: { DISPATCH_MULTIFOLD(4) }
        case 5: { DISPATCH_MULTIFOLD(5) }
        default: assert(false && "unsupported w for multifold"); return -1;
    }
#undef DISPATCH_MULTIFOLD
}

extern "C" int _kb_frac_fold_fpext_columns(const FracExt *in_ptr, FracExt *out_ptr, size_t n_frac, FpExt chal) {
    if (n_frac <= 2) {
        return 0;
    }
    // Kernel works on the FpExt view: each FracExt is two FpExt, so a quarter of the
    // FpExt length is (n_frac / 4) * 2 = n_frac / 2.
    uint32_t q_fpext = n_frac >> 1;
    auto [launch_grid, launch_block] = kernel_launch_params(q_fpext);
    fold_ef_columns_kernel<<<launch_grid, launch_block>>>(
        reinterpret_cast<const FpExt *>(in_ptr), reinterpret_cast<FpExt *>(out_ptr), q_fpext, chal
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_add_alpha(FracExt *fracs, size_t n_fracs, FpExt den_shift) {
    auto [launch_grid, launch_block] = kernel_launch_params(n_fracs);
    add_alpha_kernel<<<launch_grid, launch_block>>>(fracs, n_fracs, den_shift);
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_vector_scalar_multiply_ext_fp(FracExt *pair_buf, Fp factor, uint32_t count) {
    auto [launch_grid, launch_block] = kernel_launch_params(count);
    frac_vector_scalar_multiply_kernel<Fp, FpExt>
        <<<launch_grid, launch_block>>>(reinterpret_cast<std::pair<FpExt, FpExt> *>(pair_buf), factor, count);
    return CHECK_KERNEL();
}


// ----------------------------------------------------------------------------
// Bit-reversal permutation fused with the first GKR tree layers
// ----------------------------------------------------------------------------

// Local copies of index_t and bit_rev(): the NTT headers that define them are not on
// this library's include path.
typedef unsigned int index_t;
template<typename Word>
__device__ __forceinline__
Word bit_rev(Word word, unsigned int width)
{
    return __brev(word) >> (8 * sizeof(unsigned int) - width);
}

// In-place bit-reversal of one FracExt buffer that also shifts every denominator by
// den_shift and applies two tree layers (K = 2) while the data sits in shared memory,
// saving two separate passes over the buffer. The permutation follows the 8-wide
// tile scheme of bit_rev_permutation_z in supra/ntt_bitrev.cu (tile width fixed at 8,
// one polynomial, a single buffer instead of in/out, and __syncwarp() as the barrier
// since a tile never spans more than one warp).
//
// Each tile row written back (slots i*z_stride + rev_base, i = 0..7) contains:
//   [0,1]  merged twice     (second layer output)
//   [2,3]  merged once      (first layer right halves, needed to undo layer 1)
//   [4..7] shifted leaves   (needed to undo layer 0)
//
// The 256-thread bound is a tuning choice, not a requirement: every 8-lane tile lives
// inside one warp and warps use disjoint tiles, so any multiple of 32 works. 256 measured
// about twice as fast as 32 standalone (1525 vs 754 GB/s, lg=24, RTX 5090). The
// resulting 64 KB of dynamic shared memory needs an opt-in attribute, which the
// launcher sets once per device.
__launch_bounds__(256) __global__
void bit_rev_frac_build_k2_kernel(
    FracExt* buf, uint32_t log_n, FpExt den_shift)
{
    constexpr uint32_t TILE_W = 8;
    constexpr uint32_t LOG_TILE_W = 3;

    extern __shared__ unsigned char tile_mem[];
    FracExt (*tiles)[TILE_W][TILE_W] =
        reinterpret_cast<FracExt (*)[TILE_W][TILE_W]>(tile_mem);

    uint32_t tile_id = threadIdx.x / TILE_W;
    uint32_t pos = threadIdx.x % TILE_W;
    uint32_t lane_rev = bit_rev(pos, LOG_TILE_W);

    index_t z_stride = (index_t)1 << (log_n - LOG_TILE_W);
    index_t walk = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    #pragma unroll 1
    do {
        index_t grp = walk >> LOG_TILE_W;
        index_t grp_rev = bit_rev(grp, log_n - 2*LOG_TILE_W);

        if (grp > grp_rev)
            continue;

        index_t fwd_base = grp * TILE_W + pos;
        index_t rev_base = grp_rev * TILE_W + pos;

        FracExt held[TILE_W];

        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++) {
            tiles[tile_id][i][lane_rev] = (held[i] = buf[i * z_stride + fwd_base]);
            if (grp != grp_rev)
                held[i] = buf[i * z_stride + rev_base];
        }

        __syncwarp();

        // shift, then layers 0 and 1, on the transposed tile; result goes to rev_base
        {
            FracExt* row = tiles[tile_id][lane_rev];
            #pragma unroll
            for (uint32_t i = 0; i < TILE_W; i++)
                row[i].q = row[i].q + den_shift;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)  // layer 0: (0,4) (1,5) (2,6) (3,7)
                frac_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)  // layer 1: (0,2) (1,3)
                frac_add_inplace(row[i], row[i + 2]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++)
            buf[i * z_stride + rev_base] = tiles[tile_id][lane_rev][i];

        if (grp == grp_rev)
            continue;

        __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++)
            tiles[tile_id][i][lane_rev] = held[i];

        __syncwarp();

        // same for the mirrored tile; result goes to fwd_base
        {
            FracExt* row = tiles[tile_id][lane_rev];
            #pragma unroll
            for (uint32_t i = 0; i < TILE_W; i++)
                row[i].q = row[i].q + den_shift;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)
                frac_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)
                frac_add_inplace(row[i], row[i + 2]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++)
            buf[i * z_stride + fwd_base] = tiles[tile_id][lane_rev][i];

    } while ((walk += blockDim.x*gridDim.x) < z_stride);
    // The tile width is 8, always within a warp, so the warp-size guard used by the
    // generic permutation is not needed here.
}

// uint4-based variant of bit_rev_frac_build_k2_kernel. A FracExt is 8
// u32 with only 4-byte alignment, so a struct copy turns into 8 scalar 4-byte accesses;
// with the 32-byte stride between threads that kept LSU/L2 at 96% while DRAM sat at
// 26% (ncu, lg=27). Each FracExt starts on a 32-byte boundary, so it can be moved as two
// 16-byte uint4 instead: the same bytes with a quarter of the transactions. The tile
// arithmetic is unchanged from the scalar kernel.
__device__ __forceinline__ void frac_copy16(FracExt *out_ptr, const FracExt *in_ptr) {
    const uint4 *s = reinterpret_cast<const uint4 *>(in_ptr);
    uint4 *d = reinterpret_cast<uint4 *>(out_ptr);
    uint4 a = s[0];
    uint4 b = s[1];
    d[0] = a;
    d[1] = b;
}

__launch_bounds__(256) __global__
void bit_rev_frac_build_k2_vec_kernel(
    FracExt* buf, uint32_t log_n, FpExt den_shift)
{
    constexpr uint32_t TILE_W = 8;
    constexpr uint32_t LOG_TILE_W = 3;
    // Rows padded to 9 FracExt: with a 256-byte row pitch, the transposed read
    // tile[tile_id][lane_rev][i] would put all 8 lanes of a tile on one bank quad (8-way
    // conflict); a 288-byte pitch brings that down to 2-way.
    constexpr uint32_t TILE_PITCH = TILE_W + 1;

    extern __shared__ unsigned char tile_mem[];
    FracExt *tile_flat = reinterpret_cast<FracExt *>(tile_mem);
#define K2_CELL(g, i, r) tile_flat[((g) * TILE_W + (i)) * TILE_PITCH + (r)]

    uint32_t tile_id = threadIdx.x / TILE_W;
    uint32_t pos = threadIdx.x % TILE_W;
    uint32_t lane_rev = bit_rev(pos, LOG_TILE_W);

    index_t z_stride = (index_t)1 << (log_n - LOG_TILE_W);
    index_t walk = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    #pragma unroll 1
    do {
        index_t grp = walk >> LOG_TILE_W;
        index_t grp_rev = bit_rev(grp, log_n - 2*LOG_TILE_W);

        if (grp > grp_rev)
            continue;

        index_t fwd_base = grp * TILE_W + pos;
        index_t rev_base = grp_rev * TILE_W + pos;

        uint4 held_lo[TILE_W], held_hi[TILE_W];

        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++) {
            frac_copy16(&K2_CELL(tile_id, i, lane_rev), buf + i * z_stride + fwd_base);
            if (grp != grp_rev) {
                const uint4 *s = reinterpret_cast<const uint4 *>(buf + i * z_stride + rev_base);
                held_lo[i] = s[0];
                held_hi[i] = s[1];
            }
        }

        __syncwarp();

        // shift, then layers 0..2, on the transposed tile; result goes to rev_base
        {
            FracExt* row = &K2_CELL(tile_id, lane_rev, 0);
            #pragma unroll
            for (uint32_t i = 0; i < TILE_W; i++)
                row[i].q = row[i].q + den_shift;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)  // layer 0: (0,4) (1,5) (2,6) (3,7)
                frac_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)  // layer 1: (0,2) (1,3)
                frac_add_inplace(row[i], row[i + 2]);
            // layer 2 (K=3): the pair (0,1) inside the tile; the buffer ends up exactly
            // as three separate single-layer passes would leave it.
            frac_add_inplace(row[0], row[1]);
        }
        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++)
            frac_copy16(buf + i * z_stride + rev_base, &K2_CELL(tile_id, lane_rev, i));

        if (grp == grp_rev)
            continue;

        __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++) {
            uint4 *d = reinterpret_cast<uint4 *>(&K2_CELL(tile_id, i, lane_rev));
            d[0] = held_lo[i];
            d[1] = held_hi[i];
        }

        __syncwarp();

        // same for the mirrored tile; result goes to fwd_base
        {
            FracExt* row = &K2_CELL(tile_id, lane_rev, 0);
            #pragma unroll
            for (uint32_t i = 0; i < TILE_W; i++)
                row[i].q = row[i].q + den_shift;
            #pragma unroll
            for (uint32_t i = 0; i < 4; i++)
                frac_add_inplace(row[i], row[i + 4]);
            #pragma unroll
            for (uint32_t i = 0; i < 2; i++)
                frac_add_inplace(row[i], row[i + 2]);
            frac_add_inplace(row[0], row[1]);  // layer 2 (K=3)
        }
        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++)
            frac_copy16(buf + i * z_stride + fwd_base, &K2_CELL(tile_id, lane_rev, i));

    } while ((walk += blockDim.x*gridDim.x) < z_stride);
#undef K2_CELL
}

// Bit-reversal plus fused tree layers for one FracExt buffer (needs at least 256
// elements for the tiled kernel). Denominators are shifted by den_shift first.
extern "C" int _kb_gkr_bit_rev_frac_ext_build_k2(
    FracExt* buf, uint32_t log_n, FpExt den_shift)
{
    constexpr uint32_t TILE_W = 8;
    constexpr uint32_t tpb = 256;
    constexpr size_t smem_bytes = (size_t)tpb * TILE_W * sizeof(FracExt);  // 64 KB
    size_t n_elems = (size_t)1 << log_n;
    if (n_elems < (size_t)(tpb * TILE_W))
        return cudaErrorInvalidValue;
    uint32_t n_grid = (uint32_t)(n_elems / TILE_W / tpb);

    // The vectorized kernel is the default; FRAC_K2_VEC=0 selects the scalar
    // one. fractional.rs picks start_layer_i to match, so keep both in sync.
    static const bool vec_path = [] {
        const char *env = getenv("FRAC_K2_VEC");
        return env == nullptr || env[0] != '0';
    }();
    // The shared-memory opt-in is per kernel and per device: a single static flag would
    // only cover the first device and leave the others at the 48 KB default. Keep one
    // flag word per device; setting the attribute twice in a race is harmless.
    static std::atomic<int> smem_cap_set[16] = {};
    int dev_slot = 0;
    cudaGetDevice(&dev_slot);
    if (dev_slot < 0 || dev_slot >= 16) return cudaErrorInvalidDevice;
    if (vec_path) {
        // padded rows (TILE_W + 1 per row) -> 72 KB of dynamic shared memory
        constexpr size_t smem_bytes_vec =
            (size_t)tpb * (TILE_W + 1) * sizeof(FracExt);
        if (!(smem_cap_set[dev_slot].load(std::memory_order_acquire) & 1)) {
            cudaError_t rc = cudaFuncSetAttribute(
                bit_rev_frac_build_k2_vec_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem_bytes_vec);
            if (rc != cudaSuccess) return rc;
            smem_cap_set[dev_slot].fetch_or(1, std::memory_order_release);
        }
        bit_rev_frac_build_k2_vec_kernel<<<dim3(n_grid), tpb, smem_bytes_vec>>>(
            buf, log_n, den_shift);
        return CHECK_KERNEL();
    }

    if (!(smem_cap_set[dev_slot].load(std::memory_order_acquire) & 2)) {
        cudaError_t rc = cudaFuncSetAttribute(
            bit_rev_frac_build_k2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        if (rc != cudaSuccess) return rc;
        smem_cap_set[dev_slot].fetch_or(2, std::memory_order_release);
    }
    bit_rev_frac_build_k2_kernel<<<dim3(n_grid), tpb, smem_bytes>>>(
        buf, log_n, den_shift);
    return CHECK_KERNEL();
}

} // namespace fractional_sumcheck_gkr

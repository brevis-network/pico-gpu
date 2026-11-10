// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

// FpExt is 20 packed bytes (FracExt 40), 4-byte aligned: an element at an odd index is not
// 8- or 16-byte aligned, so it moves as plain 32-bit words. Packed 20-byte loads still run at
// full DRAM bandwidth (a fold measured at 1539 GB/s, the same as aligned 16-byte loads).
__device__ __forceinline__ FpExt ext_load(const FpExt *in_ptr) { return *in_ptr; }
__device__ __forceinline__ void ext_store(FpExt *out_ptr, FpExt val) { *out_ptr = val; }
__device__ __forceinline__ FracExt frac_load(const FracExt *in_ptr) { return *in_ptr; }
__device__ __forceinline__ void frac_store(FracExt *out_ptr, FracExt val) { *out_ptr = val; }

// ----------------------------------------------------------------------------
// Virtual input padding
// ----------------------------------------------------------------------------
// The leaf count is padded to a power of two, logical_len = L, but only the first real_len = R
// leaves are real (L/2 <= R <= L); the rest are (0, den_shift) once the denominators are shifted.
// A virtual buffer holds R entries instead of L. A node of the active layer (`active` entries,
// each covering L / active leaves) whose leaves are all padding is never stored: its value is
// (0, den_shift^(L / active)), recomputed on read. A stored node at dense position pos < R sits in
// its own slot; only leaf-level right siblings can have pos >= R, and they sit in the slot of
// their natural (non-reversed) leaf index, which is a padding-only hole of the layout.
// After openvm-org/stark-backend#331 (MIT/Apache-2.0). Every in-place pass over a virtual buffer
// must only overwrite slots no other thread still reads; each kernel below states why it does.
__device__ __forceinline__ uint32_t log2_pow2(uint32_t x) { return 31u - __clz(x); }

// pad_dens[k] = den_shift^(2^k), k < 32: the denominator of an all-padding subtree of 2^k leaves,
// read from a table the host fills once per proof (squaring den_shift on every padding read put
// up to 30 extension multiplications into bandwidth-bound kernels).
__device__ __forceinline__ FracExt virt_load(
    const FracExt *nodes,
    uint32_t pos,
    uint32_t active,
    uint32_t real_len,
    uint32_t logical_len,
    const FpExt *__restrict__ pad_dens
) {
    uint32_t subtree_len = logical_len / active;
    uint32_t first_leaf = rev_len(pos, log2_pow2(active)) * subtree_len;
    if (first_leaf >= real_len) {
        return {FpExt(Fp::zero()), pad_dens[log2_pow2(subtree_len)]};
    }
    return frac_load(nodes + (pos < real_len ? pos : first_leaf));
}

// Stores a node of the active layer; all-padding nodes are dropped.
__device__ __forceinline__ void virt_store(
    FracExt *nodes,
    uint32_t pos,
    uint32_t active,
    uint32_t real_len,
    uint32_t logical_len,
    FracExt val
) {
    uint32_t subtree_len = logical_len / active;
    uint32_t first_leaf = rev_len(pos, log2_pow2(active)) * subtree_len;
    if (first_leaf >= real_len) {
        return;
    }
    frac_store(nodes + (pos < real_len ? pos : first_leaf), val);
}

// The real/logical lengths of a virtual buffer and its padding table, passed to the kernels by
// value.
struct VirtLens {
    uint32_t real_len;
    uint32_t logical_len;
    const FpExt *pad_dens;
};

// A segmented input (cuda-mle's `seg_virtual`): the leaves are cut into aligned
// power-of-two blocks of at least 2^g leaves, each all real or all one padding leaf, and only the
// real blocks are stored, back to back, each as the dense in-place tree over its own leaves. In
// the position space of a layer or folded vector below level g, block entries are the positions
// whose low (log_len - log) bits equal its id; the high bits are its local position.
struct SegBlk {
    uint32_t base;  // storage index of the block's first leaf (real blocks)
    uint32_t log;  // log2 of its leaf count
    uint32_t id;
    uint32_t konst;  // its padding leaf in the constants table (pad blocks)
};

// The real blocks in storage order, and the real block of each storage granule (2^g entries).
struct SegTab {
    const SegBlk *blk;
    const uint32_t *cgran;
    uint32_t g;
    uint32_t log_len;
};

// The real block holding storage index `at`.
__device__ __forceinline__ SegBlk seg_block_at(SegTab const &tab, size_t at) {
    return tab.blk[tab.cgran[at >> tab.g]];
}

// Loads entry pos of a buffer whose active layer has `active` entries: plain in a dense instance,
// through virt_load in a virtual one.
template <bool VIRT>
__device__ __forceinline__ FracExt node_load(
    const FracExt *nodes,
    uint32_t pos,
    uint32_t active,
    VirtLens const &vl
) {
    if constexpr (VIRT) {
        return virt_load(nodes, pos, active, vl.real_len, vl.logical_len, vl.pad_dens);
    } else {
        return frac_load(nodes + pos);
    }
}

template <bool VIRT>
__device__ __forceinline__ void node_store(
    FracExt *nodes,
    uint32_t pos,
    uint32_t active,
    VirtLens const &vl,
    FracExt val
) {
    if constexpr (VIRT) {
        virt_store(nodes, pos, active, vl.real_len, vl.logical_len, val);
    } else {
        frac_store(nodes + pos, val);
    }
}

template <bool undo, bool with_alpha = false, bool VIRT = false>
__global__ void frac_build_tree_layer_kernel(
    FracExt *__restrict__ nodes,
    uint32_t h_off,
    FpExt den_shift,
    VirtLens vl
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= h_off) {
        return;
    }

    if constexpr (VIRT) {
        // Internal layers only (the host densifies inputs that need the alpha-shifting leaf
        // pass). Merging reads the layer below (2 * h_off entries) and stores into the layer
        // above; undoing reads the parent from the layer above and stores the left child below.
        static_assert(!with_alpha, "virtual leaf-level build is not supported");
        uint32_t below = h_off << 1;
        FracExt left = node_load<true>(nodes, pos, undo ? h_off : below, vl);
        FracExt right = node_load<true>(nodes, pos | h_off, below, vl);
        if constexpr (undo) {
            frac_unadd_inplace(left, right);
        } else {
            frac_add_inplace(left, right);
        }
        node_store<true>(nodes, pos, undo ? below : h_off, vl, left);
    } else if constexpr (with_alpha) {
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

// The eq table is stored as two factors: entry pos equals
// eq_lo[pos & (2^lo_bits - 1)] * eq_hi[pos >> lo_bits].
__device__ __forceinline__ FpExt eq_split_lookup(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    uint32_t const lo_bits,
    uint32_t const pos
) {
    return ext_load(eq_lo + (pos & ((1u << lo_bits) - 1))) *
           ext_load(eq_hi + (pos >> lo_bits));
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
        FracExt const at00 = frac_load(pq_buf + pos);
        FracExt const at10 = frac_load(pq_buf + with_rev_bits(pos, pq_len, 1, 0));
        FracExt const at01 = frac_load(pq_buf + with_rev_bits(pos, pq_len, 0, 1));
        FracExt const at11 = frac_load(pq_buf + with_rev_bits(pos, pq_len, 1, 1));
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

    FpExt x0 = ext_load(in_ptr + pos);
    FpExt x1 = ext_load(in_ptr + (pos | q_off));
    ext_store(out_ptr + pos, x0 + chal * (x1 - x0));

    uint32_t h_off = q_off << 1;

    FpExt x2 = ext_load(in_ptr + (pos | h_off));
    FpExt x3 = ext_load(in_ptr + (pos | h_off | q_off));
    ext_store(out_ptr + (pos | q_off), x2 + chal * (x3 - x2));
}

// fold_ef_columns_kernel over a virtual layer of `size` entries, on FracExt entries: thread pos
// writes out_ptr[pos] = fold(in[pos], in[pos + size/4]) and
// out_ptr[pos + size/4] = fold(in[pos + size/2], in[pos + 3*size/4]). out_ptr is dense and
// never aliases in_ptr.
__global__ void fold_frac_columns_virt_kernel(
    const FracExt *__restrict__ in_ptr,
    FracExt *__restrict__ out_ptr,
    uint32_t size,
    FpExt chal,
    VirtLens vl
) {
    uint32_t q_off = size >> 2;
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= q_off) {
        return;
    }
    uint32_t h_off = q_off << 1;

    FracExt x0 = virt_load(in_ptr, pos, size, vl.real_len, vl.logical_len, vl.pad_dens);
    FracExt x1 = virt_load(in_ptr, pos + q_off, size, vl.real_len, vl.logical_len, vl.pad_dens);
    frac_store(out_ptr + pos, {x0.p + chal * (x1.p - x0.p), x0.q + chal * (x1.q - x0.q)});

    FracExt x2 = virt_load(in_ptr, pos + h_off, size, vl.real_len, vl.logical_len, vl.pad_dens);
    FracExt x3 =
        virt_load(in_ptr, pos + h_off + q_off, size, vl.real_len, vl.logical_len, vl.pad_dens);
    frac_store(out_ptr + pos + q_off, {x2.p + chal * (x3.p - x2.p), x2.q + chal * (x3.q - x2.q)});
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
// Virtual: pq_in is a virtual layer (2 * pq_len entries); pq_out is always dense.
//
// Dynamic shared memory: one FpExt per warp (at least one).
template <bool VIRT>
__global__ void compute_round_and_fold_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    const FracExt *__restrict__ pq_in,  // unfolded, 2 * pq_len entries
    uint32_t x_count,  // after the fold, pq_len / 2
    uint32_t lo_bits,
    FpExt lam,
    FpExt fold_r,  // challenge of the previous round
    FpExt *__restrict__ blk_sums,  // [gridDim.x * 2]
    FracExt *__restrict__ pq_out,  // folded, pq_len entries
    VirtLens vl
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;
    const uint32_t in_len = pq_len << 1;  // active entries of pq_in

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
            FracExt lo_f = node_load<VIRT>(pq_in, pos, in_len, vl);
            FracExt hi_f = node_load<VIRT>(pq_in, pos + q_off, in_len, vl);
            n0_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store(pq_out + pos, {n0_at0, d0_at0});
        }
        // leaf (1,0) -> pos + q_off
        {
            FracExt lo_f = node_load<VIRT>(pq_in, pos + h_off, in_len, vl);
            FracExt hi_f = node_load<VIRT>(pq_in, pos + h_off + q_off, in_len, vl);
            n1_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store(pq_out + pos + q_off, {n1_at0, d1_at0});
        }
        // leaf (0,1) -> pos + e_off
        {
            FracExt lo_f = node_load<VIRT>(pq_in, pos + e_off, in_len, vl);
            FracExt hi_f = node_load<VIRT>(pq_in, pos + e3_off, in_len, vl);
            n0_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store(pq_out + pos + e_off, {n0_at1, d0_at1});
        }
        // leaf (1,1) -> pos + e3_off
        {
            FracExt lo_f = node_load<VIRT>(pq_in, pos + h_off + e_off, in_len, vl);
            FracExt hi_f = node_load<VIRT>(pq_in, pos + h_off + e3_off, in_len, vl);
            n1_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
            frac_store(pq_out + pos + e3_off, {n1_at1, d1_at1});
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
            FracExt lo_f = frac_load(pq + pos);
            FracExt hi_f = frac_load(pq + pos + q_off);
            n0_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (1,0): slots pos + h_off, pos + h_off + q_off
        {
            FracExt lo_f = frac_load(pq + pos + h_off);
            FracExt hi_f = frac_load(pq + pos + h_off + q_off);
            n1_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (0,1): slots pos + e_off, pos + e3_off
        {
            FracExt lo_f = frac_load(pq + pos + e_off);
            FracExt hi_f = frac_load(pq + pos + e3_off);
            n0_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // leaf (1,1): slots pos + h_off + e_off, pos + h_off + e3_off
        {
            FracExt lo_f = frac_load(pq + pos + h_off + e_off);
            FracExt hi_f = frac_load(pq + pos + h_off + e3_off);
            n1_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d1_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }

        // Store the folded leaves; every load of this thread has already happened.
        frac_store(pq + pos, {n0_at0, d0_at0});
        frac_store(pq + pos + q_off, {n1_at0, d1_at0});
        frac_store(pq + pos + e_off, {n0_at1, d0_at1});
        frac_store(pq + pos + e3_off, {n1_at1, d1_at1});

        accum_round_terms(
            eq_lo, eq_hi, pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }

    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// The fused in-place round over a virtual leaf layer, split in two launches.
//
// In place on a virtual layer, thread pos would overwrite slots that other threads still read
// spilled right siblings from (openvm-org/stark-backend#331 does exactly that; it only goes
// unnoticed while every thread sits in one warp). The leaves other threads may read are all in the
// upper half [pq_len, 2 * pq_len), and only the upper folded entries (pos + q_off and
// pos + e3_off) consume them, so:
//   fold_upper_virt_kernel: reads the layer only and writes those two folded entries to
//     upper_out[pos] and upper_out[pos + e_off] (pq_len / 2 entries, dense);
//   compute_round_and_fold_split_kernel: reads the lower half, which every thread owns, plus
//     upper_out, evaluates the round, and stores all four folded entries densely into the low
//     pq_len slots of the layer.
// After the pair the folded layer is dense, so every later round of this layer takes the dense
// kernels. Layout and offsets are those of compute_round_and_fold_inplace_kernel.
__global__ void fold_upper_virt_kernel(
    const FracExt *__restrict__ pq,  // virtual, 2 * pq_len entries
    uint32_t x_count,  // after the fold, pq_len / 2
    FpExt fold_r,
    FracExt *__restrict__ upper_out,  // pq_len / 2 entries
    VirtLens vl
) {
    const uint32_t pq_len = 2 * x_count;
    const uint32_t in_len = pq_len << 1;
    const uint32_t h_off = pq_len;
    const uint32_t q_off = pq_len >> 1;
    const uint32_t e_off = pq_len >> 2;
    const uint32_t e3_off = e_off * 3;

    for (uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < x_count / 2;
         pos += blockDim.x * gridDim.x) {
        {
            FracExt lo_f = node_load<true>(pq, pos + h_off, in_len, vl);
            FracExt hi_f = node_load<true>(pq, pos + h_off + q_off, in_len, vl);
            frac_store(
                upper_out + pos,
                {lo_f.p + fold_r * (hi_f.p - lo_f.p), lo_f.q + fold_r * (hi_f.q - lo_f.q)}
            );
        }
        {
            FracExt lo_f = node_load<true>(pq, pos + h_off + e_off, in_len, vl);
            FracExt hi_f = node_load<true>(pq, pos + h_off + e3_off, in_len, vl);
            frac_store(
                upper_out + pos + e_off,
                {lo_f.p + fold_r * (hi_f.p - lo_f.p), lo_f.q + fold_r * (hi_f.q - lo_f.q)}
            );
        }
    }
}

// Dynamic shared memory: one FpExt per warp (at least one).
__global__ void compute_round_and_fold_split_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    FracExt *pq,  // virtual, 2 * pq_len entries on entry; the low pq_len slots dense on exit
    const FracExt *__restrict__ upper_in,  // from fold_upper_virt_kernel
    uint32_t x_count,  // after the fold, pq_len / 2
    uint32_t lo_bits,
    FpExt lam,
    FpExt fold_r,
    FpExt *__restrict__ blk_sums,  // [gridDim.x * 2]
    VirtLens vl
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t pq_len = 2 * x_count;
    const uint32_t in_len = pq_len << 1;
    const uint32_t q_off = pq_len >> 1;
    const uint32_t e_off = pq_len >> 2;
    const uint32_t e3_off = e_off * 3;

    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;

    for (uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos < x_count / 2;
         pos += blockDim.x * gridDim.x) {
        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;

        // Lower half: the four source entries are this thread's own slots (or padding).
        {
            FracExt lo_f = node_load<true>(pq, pos, in_len, vl);
            FracExt hi_f = node_load<true>(pq, pos + q_off, in_len, vl);
            n0_at0 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at0 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }
        {
            FracExt lo_f = node_load<true>(pq, pos + e_off, in_len, vl);
            FracExt hi_f = node_load<true>(pq, pos + e3_off, in_len, vl);
            n0_at1 = lo_f.p + fold_r * (hi_f.p - lo_f.p);
            d0_at1 = lo_f.q + fold_r * (hi_f.q - lo_f.q);
        }
        // Upper half, already folded.
        {
            FracExt f = frac_load(upper_in + pos);
            n1_at0 = f.p;
            d1_at0 = f.q;
        }
        {
            FracExt f = frac_load(upper_in + pos + e_off);
            n1_at1 = f.p;
            d1_at1 = f.q;
        }

        frac_store(pq + pos, {n0_at0, d0_at0});
        frac_store(pq + pos + q_off, {n1_at0, d1_at0});
        frac_store(pq + pos + e_off, {n0_at1, d0_at1});
        frac_store(pq + pos + e3_off, {n1_at1, d1_at1});

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
// Virtual: parents are entries of the layer above (h_off entries), children of the layer below
// (pq_len entries). A child slot written here is the thread's own parent slot or is dropped as
// padding, and the spilled right siblings other threads read sit in padding holes, so the
// in-place revert stays race-free.
//
// Dynamic shared memory: one FpExt per warp (at least one).
template <bool VIRT>
__global__ void compute_round_and_revert_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    FracExt *__restrict__ nodes,  // low half restored in place
    uint32_t x_count,
    uint32_t lo_bits,
    FpExt lam,
    FpExt *__restrict__ blk_sums,  // [gridDim.x * 2]
    VirtLens vl
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
            FracExt left = node_load<VIRT>(nodes, pos, h_off, vl);
            FracExt right = node_load<VIRT>(nodes, pos + h_off, pq_len, vl);
            FpExt right_q_inv = inv(right.q);
            d0_at0 = left.q * right_q_inv;
            n0_at0 = (left.p - d0_at0 * right.p) * right_q_inv;
            n1_at0 = right.p;
            d1_at0 = right.q;
            node_store<VIRT>(nodes, pos, pq_len, vl, {n0_at0, d0_at0});
        }

        // leaf (0,1) recovered from its merge with leaf (1,1)
        {
            FracExt left = node_load<VIRT>(nodes, pos + q_off, h_off, vl);
            FracExt right = node_load<VIRT>(nodes, pos + h_off + q_off, pq_len, vl);
            FpExt right_q_inv = inv(right.q);
            d0_at1 = left.q * right_q_inv;
            n0_at1 = (left.p - d0_at1 * right.p) * right_q_inv;
            n1_at1 = right.p;
            d1_at1 = right.q;
            node_store<VIRT>(nodes, pos + q_off, pq_len, vl, {n0_at1, d0_at1});
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
// Virtual (fold_inline only): pq is a virtual layer of 2^(n_rem + 2) entries.
template <bool fold_inline, uint32_t LOGW, bool VIRT>
__global__ void precompute_m_build_partial_kernel(
    const FracExt *__restrict__ pq,
    uint32_t n_rem,  // variable count after the fold
    FpExt lam,
    FpExt fold_r,  // only read when fold_inline
    const FpExt *__restrict__ tail_eq_lo,
    const FpExt *__restrict__ tail_eq_hi,
    uint32_t tail_lo_bits,
    uint32_t tile_len,
    FpExt *__restrict__ partials,
    VirtLens vl
) {
    static_assert(fold_inline || !VIRT, "a virtual source is only read with the inline fold");
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
    uint32_t src_len = poly_off << 1;  // active entries of pq

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
                FracExt lo0 = node_load<VIRT>(pq, flat, src_len, vl);
                FracExt hi0 = node_load<VIRT>(pq, flat + fold_off, src_len, vl);
                FracExt lo1 = node_load<VIRT>(pq, poly_off + flat, src_len, vl);
                FracExt hi1 = node_load<VIRT>(pq, poly_off + flat + fold_off, src_len, vl);
                n0 = lo0.p + fold_r * (hi0.p - lo0.p);
                d0 = lo0.q + fold_r * (hi0.q - lo0.q);
                n1 = lo1.p + fold_r * (hi1.p - lo1.p);
                d1 = lo1.q + fold_r * (hi1.q - lo1.q);
            } else {
                // Already folded: plain loads.
                FracExt fr0 = frac_load(&pq[flat]);
                FracExt fr1 = frac_load(&pq[poly_off + flat]);
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

template <bool fold_inline, bool VIRT, uint32_t LOGW>
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
    FpExt *partials,
    VirtLens vl
) {
    constexpr uint32_t side = 1u << LOGW;
    dim3 launch_block(side, side);
    constexpr uint32_t pad_stride = side + 1;  // same padded pitch as the kernel
    size_t smem_bytes =
        (4 * pad_stride * TAIL_CHUNK + TAIL_CHUNK) * sizeof(FpExt);
    precompute_m_build_partial_kernel<fold_inline, LOGW, VIRT><<<launch_grid, launch_block, smem_bytes>>>(
        pq,
        n_rem,
        lam,
        fold_r,
        tail_eq_lo,
        tail_eq_hi,
        tail_lo_bits,
        tile_len,
        partials,
        vl
    );
}

template <bool fold_inline, bool VIRT>
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
    FpExt *partials,
    VirtLens vl
) {
    using PartialFn = void (*)(
        dim3, const FracExt *, uint32_t, FpExt, FpExt, const FpExt *, const FpExt *, uint32_t,
        uint32_t, FpExt *, VirtLens
    );
    static constexpr PartialFn table[] = {
        &launch_m_window_partial<fold_inline, VIRT, 1>,
        &launch_m_window_partial<fold_inline, VIRT, 2>,
        &launch_m_window_partial<fold_inline, VIRT, 3>,
        &launch_m_window_partial<fold_inline, VIRT, 4>,
        &launch_m_window_partial<fold_inline, VIRT, 5>,
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
        partials,
        vl
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
//
// Virtual: in_ptr is a virtual layer of 2^(LOGW + 1) * n_cols entries; out_ptr is dense and does
// not alias it (a virtual source can hand one thread's entry to another's slot).
template <uint32_t LOGW, bool VIRT>
__global__ void multifold_kernel(
    const FracExt *in_ptr,
    FracExt *out_ptr,
    uint32_t n_cols,
    const FpExt *__restrict__ win_eq,
    VirtLens vl
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
    const uint32_t hi_off = n_cols << LOGW;
    const uint32_t src_len = hi_off << 1;  // active entries of in_ptr
    const FracExt *src_hi = in_ptr + ((size_t)n_cols << LOGW);
    for (uint32_t bk = 0; bk < n_beta; ++bk) {
        uint32_t pos = bk * n_cols + col;
        FracExt fv0 = VIRT ? node_load<VIRT>(in_ptr, pos, src_len, vl) : in_ptr[pos];
        FracExt fv1 = VIRT ? node_load<VIRT>(in_ptr, hi_off + pos, src_len, vl) : src_hi[pos];
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
template <bool VIRT>
__global__ void frac_build_tree_two_layers_kernel(
    FracExt *__restrict__ nodes,
    uint32_t h_inner,  // N >> (i+2), i = first of the two layers
    VirtLens vl
) {
    uint32_t lane_t = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane_t >= h_inner) return;

    uint32_t h_outer = h_inner << 1;  // N >> (i+1)
    uint32_t below = h_inner << 2;  // entries of layer i's input

    FracExt fa = node_load<VIRT>(nodes, lane_t, below, vl);
    FracExt fb = node_load<VIRT>(nodes, lane_t + h_inner, below, vl);
    FracExt fc = node_load<VIRT>(nodes, lane_t + h_outer, below, vl);
    FracExt fd = node_load<VIRT>(nodes, lane_t + h_outer + h_inner, below, vl);

    // layer i (results stay in registers)
    FracExt left = frac_add(fa, fc);
    FracExt right = frac_add(fb, fd);
    // layer i+1
    FracExt merged = frac_add(left, right);

    node_store<VIRT>(nodes, lane_t, h_inner, vl, merged);
    node_store<VIRT>(nodes, lane_t + h_inner, h_outer, vl, right);  // kept for undoing layer i+1
    // nodes[t + h_outer] and nodes[t + h_outer + h_inner] are left untouched; undoing
    // layer i reads them.
}

// ----------------------------------------------------------------------------
// Host side
// ----------------------------------------------------------------------------
// Lengths of a buffer for the launchers: real_len < logical_len selects the virtual instance.
inline VirtLens make_virt_lens(size_t real_len, size_t logical_len, const FpExt *pad_dens) {
    return VirtLens{(uint32_t)real_len, (uint32_t)logical_len, pad_dens};
}

template <bool undo, bool with_alpha>
int launch_frac_build_tree_layer(FracExt *nodes, size_t node_count, FpExt den_shift) {
    auto [launch_grid, launch_block] = kernel_launch_params(node_count);
    frac_build_tree_layer_kernel<undo, with_alpha><<<launch_grid, launch_block>>>(
        nodes, node_count, den_shift, VirtLens{0, 0, nullptr}
    );
    return CHECK_KERNEL();
}

template <bool undo>
int launch_frac_build_tree_layer_virt(FracExt *nodes, size_t node_count, VirtLens vl) {
    auto [launch_grid, launch_block] = kernel_launch_params(node_count);
    frac_build_tree_layer_kernel<undo, false, true><<<launch_grid, launch_block>>>(
        nodes, node_count, FpExt(Fp::zero()), vl
    );
    return CHECK_KERNEL();
}

// real_len < logical_len: nodes is virtual (see virt_load); only internal layers, no shift.
extern "C" int _kb_frac_build_tree_layer(
    FracExt *nodes,
    size_t node_count,
    size_t real_len,
    size_t logical_len,
    bool undo,
    FpExt den_shift,
    bool with_alpha,
    const FpExt *pad_dens  // virtual only, see virt_load
) {
    if (node_count == 0) {
        return 0;
    }
    assert(node_count % 2 == 0);
    node_count /= 2;

    if (real_len < logical_len) {
        if (with_alpha) {
            return cudaErrorInvalidValue;
        }
        VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);
        return undo ? launch_frac_build_tree_layer_virt<true>(nodes, node_count, vl)
                    : launch_frac_build_tree_layer_virt<false>(nodes, node_count, vl);
    }
    return DISPATCH_BOOL_PAIR(
        launch_frac_build_tree_layer, undo, with_alpha, nodes, node_count, den_shift
    );
}

// Two-layer tree build. h_inner = N >> (i+2) with i the first of the two layers;
// needs h_inner >= 1 and 4 * h_inner <= buffer length.
extern "C" int _kb_frac_build_tree_two_layers(
    FracExt *nodes,
    size_t h_inner,
    size_t real_len,
    size_t logical_len,
    const FpExt *pad_dens
) {
    if (h_inner == 0) return 0;
    // 256 threads per block leave more registers per thread than 1024 would; this
    // kernel keeps four fractions and three merges in flight, more than the
    // single-layer kernel.
    auto [launch_grid, launch_block] = kernel_launch_params(h_inner, 256);
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);
    if (real_len < logical_len) {
        frac_build_tree_two_layers_kernel<true><<<launch_grid, launch_block>>>(nodes, (uint32_t)h_inner, vl);
    } else {
        frac_build_tree_two_layers_kernel<false><<<launch_grid, launch_block>>>(nodes, (uint32_t)h_inner, vl);
    }
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
    size_t real_len,  // < logical_len: nodes is virtual
    size_t logical_len,
    size_t eq_lo_len,
    FpExt lam,
    const FpExt *pad_dens,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(x_count > 1);

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);

    // undo + round evaluation
    auto *kernel = real_len < logical_len ? compute_round_and_revert_kernel<true>
                                          : compute_round_and_revert_kernel<false>;
    kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        nodes,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        scratch_sums,
        vl
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
    size_t real_len,  // < logical_len: pq_in_buf is virtual (pq_out_buf is dense)
    size_t logical_len,
    size_t eq_lo_len,
    FpExt lam,
    FpExt fold_r,
    const FpExt *pad_dens,
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
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);

    // round + fold; fills scratch_sums and pq_out_buf
    auto *kernel = real_len < logical_len ? compute_round_and_fold_kernel<true>
                                          : compute_round_and_fold_kernel<false>;
    kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        pq_in_buf,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        fold_r,
        scratch_sums,
        pq_out_buf,
        vl
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // second stage: fold scratch_sums into evals_out
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

// The fused in-place round over a virtual layer (see fold_upper_virt_kernel): upper_scratch
// receives src_len / 4 folded entries and must not alias pq_buf; on return the low src_len / 2
// slots of pq_buf hold the folded layer, dense.
extern "C" int _kb_frac_compute_round_and_fold_split(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    FracExt *pq_buf,  // virtual, src_len entries on entry
    FracExt *upper_scratch,  // src_len / 4 entries
    size_t src_len,  // unfolded FracExt count
    size_t real_len,
    size_t logical_len,
    size_t eq_lo_len,
    FpExt lam,
    FpExt fold_r,
    const FpExt *pad_dens,
    FpExt *evals_out,  // s'(1), s'(2)
    FpExt *scratch_sums  // gridDim.x * 2 per-block sums
) {
    assert(src_len > 4);
    assert(real_len < logical_len);
    size_t pq_len = src_len >> 1;
    size_t x_count = pq_len >> 1;

    auto [launch_grid, launch_block] = round_launch_shape(x_count);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);

    fold_upper_virt_kernel<<<launch_grid, launch_block>>>(
        pq_buf, (uint32_t)x_count, fold_r, upper_scratch, vl
    );
    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;
    compute_round_and_fold_split_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo,
        eq_hi,
        pq_buf,
        upper_scratch,
        (uint32_t)x_count,
        __builtin_ctz((uint32_t)eq_lo_len),
        lam,
        fold_r,
        scratch_sums,
        vl
    );
    rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;
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

// real_len < logical_len: pq is virtual, which only the inline-fold (first window) read allows.
extern "C" int _kb_frac_precompute_m_build(
    const FracExt *pq,
    size_t real_len,
    size_t logical_len,
    size_t n_rem,  // variable count after the fold
    size_t wbits,
    FpExt lam,
    FpExt fold_r,  // only read when fold_inline
    const FpExt *pad_dens,
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
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);
    if (real_len < logical_len) {
        if (!fold_inline) {
            return cudaErrorInvalidValue;
        }
        dispatch_rc = dispatch_m_window_partial<true, true>(
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
            partials,
            vl
        );
    } else if (fold_inline) {
        dispatch_rc = dispatch_m_window_partial<true, false>(
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
            partials,
            vl
        );
    } else {
        dispatch_rc = dispatch_m_window_partial<false, false>(
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
            partials,
            vl
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

// real_len < logical_len: in_ptr is virtual; out_ptr must not alias it then.
extern "C" int _kb_frac_multifold(
    const FracExt *in_ptr,
    FracExt *out_ptr,
    size_t real_len,
    size_t logical_len,
    size_t n_rem,
    size_t wbits,
    const FpExt *pad_dens,
    const FpExt *win_eq
) {
    assert(n_rem > 0);
    assert(wbits > 0 && wbits <= n_rem);

    size_t col_count = 1u << (n_rem - wbits);
    auto [launch_grid, launch_block] = kernel_launch_params(col_count, 256);
    VirtLens vl = make_virt_lens(real_len, logical_len, pad_dens);
    bool virt = real_len < logical_len;

#define DISPATCH_MULTIFOLD(LOGW) \
    if (virt) { \
        multifold_kernel<LOGW, true><<<launch_grid, launch_block>>>(in_ptr, out_ptr, (uint32_t)col_count, win_eq, vl); \
    } else { \
        multifold_kernel<LOGW, false><<<launch_grid, launch_block>>>(in_ptr, out_ptr, (uint32_t)col_count, win_eq, vl); \
    } \
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

// real_len < logical_len: in_ptr is virtual; out_ptr is dense and must not alias it then.
extern "C" int _kb_frac_fold_fpext_columns(
    const FracExt *in_ptr,
    FracExt *out_ptr,
    size_t n_frac,
    size_t real_len,
    size_t logical_len,
    FpExt chal,
    const FpExt *pad_dens
) {
    if (n_frac <= 2) {
        return 0;
    }
    if (real_len < logical_len) {
        uint32_t q_frac = n_frac >> 2;
        auto [launch_grid, launch_block] = kernel_launch_params(q_frac);
        fold_frac_columns_virt_kernel<<<launch_grid, launch_block>>>(
            in_ptr, out_ptr, (uint32_t)n_frac, chal, make_virt_lens(real_len, logical_len, pad_dens)
        );
        return CHECK_KERNEL();
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

// Slot access for the bit-reversal kernels. In a virtual buffer (real_len < 2^log_n) an input slot
// at or past real_len is padding, zero before the shift, and an output goes through virt_store at
// the layer its tile row belongs to: row i of a written tile holds a node of the layer with
// 2^log_n >> layer_shift entries. Every slot a tile pair stores to lies in the pair's own slots
// (an output at pos < real_len in its own slot, a leaf past it at its natural index, which the
// bit reversal maps into the mirrored tile), so the in-place pass stays race-free.
template <bool VIRT>
__device__ __forceinline__ FracExt k2_in(const FracExt *buf, index_t pos, uint32_t real_len) {
    if constexpr (VIRT) {
        if (pos >= real_len) {
            return {FpExt(Fp::zero()), FpExt(Fp::zero())};
        }
    }
    return buf[pos];
}

template <bool VIRT>
__device__ __forceinline__ void k2_out(
    FracExt *buf,
    index_t pos,
    uint32_t layer_shift,
    uint32_t real_len,
    uint32_t log_n,
    FracExt const &val
) {
    if constexpr (VIRT) {
        uint32_t logical = 1u << log_n;
        virt_store(buf, pos, logical >> layer_shift, real_len, logical, val);
    } else {
        buf[pos] = val;
    }
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
template <bool VIRT>
__launch_bounds__(256) __global__
void bit_rev_frac_build_k2_kernel(
    FracExt* buf, uint32_t log_n, FpExt den_shift, uint32_t real_len)
{
    // layer of each written tile row: [0,1] two merges up, [2,3] one, [4..7] leaves
    constexpr uint32_t ROW_LAYER[8] = {2, 2, 1, 1, 0, 0, 0, 0};
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
            tiles[tile_id][i][lane_rev] = (held[i] = k2_in<VIRT>(buf, i * z_stride + fwd_base, real_len));
            if (grp != grp_rev)
                held[i] = k2_in<VIRT>(buf, i * z_stride + rev_base, real_len);
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
            k2_out<VIRT>(buf, i * z_stride + rev_base, ROW_LAYER[i], real_len, log_n, tiles[tile_id][lane_rev][i]);

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
            k2_out<VIRT>(buf, i * z_stride + fwd_base, ROW_LAYER[i], real_len, log_n, tiles[tile_id][lane_rev][i]);

    } while ((walk += blockDim.x*gridDim.x) < z_stride);
    // The tile width is 8, always within a warp, so the warp-size guard used by the
    // generic permutation is not needed here.
}

// Variant of bit_rev_frac_build_k2_kernel with padded tile rows and tree layer 2 fused (see
// below). Over KoalaBear^4 it also moved each 32-byte FracExt as two uint4; a KoalaBear^5
// FracExt is 40 bytes, 4-byte aligned, and moves as plain words (see ext_load). The tile
// arithmetic is unchanged from the scalar kernel.
__device__ __forceinline__ void frac_copy(FracExt *out_ptr, const FracExt *in_ptr) { *out_ptr = *in_ptr; }

//
// SEG: buf is a segmented input's storage and every real block is permuted and built on its own;
// the grid has one block per 2^11 storage entries, so each block of 256 threads makes exactly one
// pass (a real block is at least a granule, 2^11 or more entries).
template <bool VIRT, bool SEG = false>
__launch_bounds__(256) __global__
void bit_rev_frac_build_k2_vec_kernel(
    FracExt* buf, uint32_t log_n, FpExt den_shift, uint32_t real_len, SegTab seg_tab)
{
    index_t grid_first = blockIdx.x;
    if constexpr (SEG) {
        SegBlk b = seg_block_at(seg_tab, (size_t)blockIdx.x << 11);
        buf += b.base;
        log_n = b.log;
        grid_first = blockIdx.x - (b.base >> 11);
    }
    // layer of each written tile row: [0] three merges up, [1] two, [2,3] one, [4..7] leaves
    constexpr uint32_t ROW_LAYER[8] = {3, 2, 1, 1, 0, 0, 0, 0};
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
    index_t walk = threadIdx.x + blockDim.x * grid_first;

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
            if constexpr (VIRT)
                K2_CELL(tile_id, i, lane_rev) = k2_in<VIRT>(buf, i * z_stride + fwd_base, real_len);
            else
                frac_copy(&K2_CELL(tile_id, i, lane_rev), buf + i * z_stride + fwd_base);
            if (grp != grp_rev)
                held[i] = k2_in<VIRT>(buf, i * z_stride + rev_base, real_len);
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
        for (uint32_t i = 0; i < TILE_W; i++) {
            if constexpr (VIRT)
                k2_out<VIRT>(buf, i * z_stride + rev_base, ROW_LAYER[i], real_len, log_n, K2_CELL(tile_id, lane_rev, i));
            else
                frac_copy(buf + i * z_stride + rev_base, &K2_CELL(tile_id, lane_rev, i));
        }

        if (grp == grp_rev)
            continue;

        __syncwarp();

        #pragma unroll
        for (uint32_t i = 0; i < TILE_W; i++) {
            K2_CELL(tile_id, i, lane_rev) = held[i];
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
        for (uint32_t i = 0; i < TILE_W; i++) {
            if constexpr (VIRT)
                k2_out<VIRT>(buf, i * z_stride + fwd_base, ROW_LAYER[i], real_len, log_n, K2_CELL(tile_id, lane_rev, i));
            else
                frac_copy(buf + i * z_stride + fwd_base, &K2_CELL(tile_id, lane_rev, i));
        }

    } while ((walk += blockDim.x*gridDim.x) < z_stride);
#undef K2_CELL
}

// Bit-reversal plus fused tree layers for one FracExt buffer (needs at least 256
// elements for the tiled kernel). Denominators are shifted by den_shift first. real_len <
// 2^log_n selects the virtual layout: buf then holds only real_len leaves.
template <bool VIRT>
int launch_bit_rev_frac_build_k2(FracExt* buf, uint32_t log_n, FpExt den_shift, uint32_t real_len)
{
    constexpr uint32_t TILE_W = 8;
    constexpr uint32_t tpb = 256;
    constexpr size_t smem_bytes = (size_t)tpb * TILE_W * sizeof(FracExt);  // 80 KB
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
        // padded rows (TILE_W + 1 per row) -> 90 KB of dynamic shared memory
        constexpr size_t smem_bytes_vec =
            (size_t)tpb * (TILE_W + 1) * sizeof(FracExt);
        if (!(smem_cap_set[dev_slot].load(std::memory_order_acquire) & 1)) {
            cudaError_t rc = cudaFuncSetAttribute(
                bit_rev_frac_build_k2_vec_kernel<VIRT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)smem_bytes_vec);
            if (rc != cudaSuccess) return rc;
            smem_cap_set[dev_slot].fetch_or(1, std::memory_order_release);
        }
        bit_rev_frac_build_k2_vec_kernel<VIRT><<<dim3(n_grid), tpb, smem_bytes_vec>>>(
            buf, log_n, den_shift, real_len, SegTab{});
        return CHECK_KERNEL();
    }

    if (!(smem_cap_set[dev_slot].load(std::memory_order_acquire) & 2)) {
        cudaError_t rc = cudaFuncSetAttribute(
            bit_rev_frac_build_k2_kernel<VIRT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        if (rc != cudaSuccess) return rc;
        smem_cap_set[dev_slot].fetch_or(2, std::memory_order_release);
    }
    bit_rev_frac_build_k2_kernel<VIRT><<<dim3(n_grid), tpb, smem_bytes>>>(
        buf, log_n, den_shift, real_len);
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_bit_rev_frac_ext_build_k2(
    FracExt* buf, size_t real_len, uint32_t log_n, FpExt den_shift)
{
    if (real_len < ((size_t)1 << log_n))
        return launch_bit_rev_frac_build_k2<true>(buf, log_n, den_shift, (uint32_t)real_len);
    return launch_bit_rev_frac_build_k2<false>(buf, log_n, den_shift, (uint32_t)real_len);
}

// ----------------------------------------------------------------------------
// Segmented input (see SegBlk)
// ----------------------------------------------------------------------------
// Every kernel below runs a dense kernel's body on each real block, with threads in storage
// order: work item w covers 2^sh storage entries from w << sh, so its block is the one of storage
// granule (w << sh) >> g (sh <= g), and its local index is w - (base >> sh). A round's eq weight of
// local position p is the dense table's at (p << (log_len - log)) | id. Pad blocks never reach these
// kernels: their contribution to a round, eq(id) times a constant, is gkr_seg_pad_sum's.

// Pad nodes of each level per constant: padtab[k * SEG_PAD_LEVELS + l] is 2^l copies of constant
// k, its denominator shifted, added up.
constexpr uint32_t SEG_PAD_LEVELS = 32;

// frac_build_tree_layer_kernel (merge, no shift) on every real block: level -> level + 1.
__global__ void seg_build_layer_kernel(FracExt *nodes, SegTab tab, uint32_t level, uint32_t n_work) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_work) return;
    uint32_t sh = level + 1;
    SegBlk b = seg_block_at(tab, (size_t)w << sh);
    uint32_t h_off = 1u << (b.log - sh);
    uint32_t pos = w - (b.base >> sh);
    FracExt *loc = nodes + b.base;
    FracExt left = loc[pos];
    frac_add_inplace(left, loc[pos | h_off]);
    loc[pos] = left;
}

// frac_build_tree_two_layers_kernel on every real block: level -> level + 2.
__global__ void seg_build_two_layers_kernel(FracExt *nodes, SegTab tab, uint32_t level, uint32_t n_work) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_work) return;
    uint32_t sh = level + 2;
    SegBlk b = seg_block_at(tab, (size_t)w << sh);
    uint32_t h_inner = 1u << (b.log - sh);
    uint32_t h_outer = h_inner << 1;
    uint32_t t = w - (b.base >> sh);
    FracExt *loc = nodes + b.base;
    FracExt fa = loc[t];
    FracExt fb = loc[t + h_inner];
    FracExt fc = loc[t + h_outer];
    FracExt fd = loc[t + h_outer + h_inner];
    FracExt left = frac_add(fa, fc);
    FracExt right = frac_add(fb, fd);
    loc[t] = frac_add(left, right);
    loc[t + h_inner] = right;
}

// compute_round_and_revert_kernel on every real block of the layer at `level`, in place.
__global__ void seg_round_revert_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    FracExt *__restrict__ nodes,
    SegTab tab,
    uint32_t level,
    uint32_t n_work,
    uint32_t lo_bits,
    FpExt lam,
    FpExt *__restrict__ blk_sums
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t sh = level + 2;
    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;
    for (uint32_t w = threadIdx.x + blockIdx.x * blockDim.x; w < n_work;
         w += blockDim.x * gridDim.x) {
        SegBlk b = seg_block_at(tab, (size_t)w << sh);
        uint32_t pos = w - (b.base >> sh);
        uint32_t pq_len = 1u << (b.log - level);
        uint32_t h_off = pq_len >> 1;
        uint32_t q_off = pq_len >> 2;
        FracExt *loc = nodes + b.base;

        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;
        {
            FracExt left = frac_load(loc + pos);
            FracExt right = frac_load(loc + pos + h_off);
            FpExt right_q_inv = inv(right.q);
            d0_at0 = left.q * right_q_inv;
            n0_at0 = (left.p - d0_at0 * right.p) * right_q_inv;
            n1_at0 = right.p;
            d1_at0 = right.q;
            frac_store(loc + pos, {n0_at0, d0_at0});
        }
        {
            FracExt left = frac_load(loc + pos + q_off);
            FracExt right = frac_load(loc + pos + h_off + q_off);
            FpExt right_q_inv = inv(right.q);
            d0_at1 = left.q * right_q_inv;
            n0_at1 = (left.p - d0_at1 * right.p) * right_q_inv;
            n1_at1 = right.p;
            d1_at1 = right.q;
            frac_store(loc + pos + q_off, {n0_at1, d0_at1});
        }
        uint32_t eq_pos = (pos << (tab.log_len - b.log)) | b.id;
        accum_round_terms(
            eq_lo, eq_hi, eq_pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }
    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// compute_round_and_fold_kernel on every real block of a segmented vector at level `c_in` (block
// entries 2^(log - c_in)), stored from base >> src_sh in src; the folded vector goes to
// base >> dst_sh in dst. src == dst with src_sh == dst_sh folds in place: a thread stores only
// to low-half slots it loaded itself, as in compute_round_and_fold_inplace_kernel.
__global__ void seg_round_fold_kernel(
    const FpExt *__restrict__ eq_lo,
    const FpExt *__restrict__ eq_hi,
    const FracExt *src,
    uint32_t src_sh,
    FracExt *dst,
    uint32_t dst_sh,
    SegTab tab,
    uint32_t c_in,
    uint32_t n_work,
    uint32_t lo_bits,
    FpExt lam,
    FpExt fold_r,
    FpExt *__restrict__ blk_sums
) {
    extern __shared__ FpExt red_smem[];
    const uint32_t sh = c_in + 3;
    const FpExt zero(Fp::zero());
    FpExt acc_s1 = zero, acc_s2 = zero;
    for (uint32_t w = threadIdx.x + blockIdx.x * blockDim.x; w < n_work;
         w += blockDim.x * gridDim.x) {
        SegBlk b = seg_block_at(tab, (size_t)w << sh);
        uint32_t pos = w - (b.base >> sh);
        uint32_t pq_len = 1u << (b.log - c_in - 1);  // after the fold
        const uint32_t h_off = pq_len;
        const uint32_t q_off = pq_len >> 1;
        const uint32_t e_off = pq_len >> 2;
        const uint32_t e3_off = e_off * 3;
        const FracExt *in = src + (b.base >> src_sh);
        FracExt *out = dst + (b.base >> dst_sh);

        FpExt n0_at0, d0_at0, n1_at0, d1_at0;
        FpExt n0_at1, d0_at1, n1_at1, d1_at1;
        // Every load comes before any store: in place, the stores overwrite loaded slots.
        FracExt a0 = frac_load(in + pos);
        FracExt a1 = frac_load(in + pos + q_off);
        FracExt b0 = frac_load(in + pos + h_off);
        FracExt b1 = frac_load(in + pos + h_off + q_off);
        FracExt c0 = frac_load(in + pos + e_off);
        FracExt c1 = frac_load(in + pos + e3_off);
        FracExt e0 = frac_load(in + pos + h_off + e_off);
        FracExt e1 = frac_load(in + pos + h_off + e3_off);
        n0_at0 = a0.p + fold_r * (a1.p - a0.p);
        d0_at0 = a0.q + fold_r * (a1.q - a0.q);
        n1_at0 = b0.p + fold_r * (b1.p - b0.p);
        d1_at0 = b0.q + fold_r * (b1.q - b0.q);
        n0_at1 = c0.p + fold_r * (c1.p - c0.p);
        d0_at1 = c0.q + fold_r * (c1.q - c0.q);
        n1_at1 = e0.p + fold_r * (e1.p - e0.p);
        d1_at1 = e0.q + fold_r * (e1.q - e0.q);
        frac_store(out + pos, {n0_at0, d0_at0});
        frac_store(out + pos + q_off, {n1_at0, d1_at0});
        frac_store(out + pos + e_off, {n0_at1, d0_at1});
        frac_store(out + pos + e3_off, {n1_at1, d1_at1});

        uint32_t eq_pos = (pos << (tab.log_len - b.log)) | b.id;
        accum_round_terms(
            eq_lo, eq_hi, eq_pos, lo_bits, lam,
            n0_at0, d0_at0, n0_at1, d0_at1,
            n1_at0, d1_at0, n1_at1, d1_at1,
            acc_s1, acc_s2
        );
    }
    store_pair_block_sums(red_smem, acc_s1, acc_s2, blk_sums);
}

// The dense vector (2^n_log entries, bit-reversed positions) of a segmented one at level c whose
// real blocks are stored from base >> src_sh in src; pad entries are the level-`pad_level` pad
// node of their block's constant. lgran: per leaf granule, the real block index or
// 0x80000000 | konst.
__global__ void seg_gather_kernel(
    const FracExt *__restrict__ src,
    uint32_t src_sh,
    SegTab tab,
    const uint32_t *__restrict__ lgran,
    const FracExt *__restrict__ padtab,
    uint32_t pad_level,
    uint32_t c,
    FracExt *__restrict__ dst,
    uint32_t n_log
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= (1u << n_log)) return;
    size_t leaf = (size_t)bit_rev(pos, n_log) << c;
    uint32_t e = lgran[leaf >> tab.g];
    FracExt v;
    if (e & 0x80000000u) {
        v = padtab[(size_t)(e & 0x7fffffffu) * SEG_PAD_LEVELS + pad_level];
    } else {
        SegBlk b = tab.blk[e];
        v = src[(b.base >> src_sh) + (pos >> (tab.log_len - b.log))];
    }
    dst[pos] = v;
}

__global__ void seg_pad_table_kernel(
    const FracExt *__restrict__ consts,
    uint32_t n_consts,
    FpExt den_shift,
    FracExt *__restrict__ padtab
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_consts) return;
    FracExt v = consts[k];
    v.q = v.q + den_shift;
    for (uint32_t l = 0; l < SEG_PAD_LEVELS; l++) {
        padtab[(size_t)k * SEG_PAD_LEVELS + l] = v;
        v = frac_add(v, v);
    }
}

// One block: sum over pad blocks of eq(id) * term, where term is a round kernel's summand at four
// equal level-`level` pad nodes (n, d): (n + lam d) d + n d, and eq(id) the eq table summed over the
// block's positions, the product over its low (log_len - log) bits i of x_i or 1 - x_i, x_i =
// xi_rev[i] (the coordinate of position bit i, the last coordinate first).
__global__ void seg_pad_sum_kernel(
    const SegBlk *__restrict__ pads,
    uint32_t n_pads,
    const FracExt *__restrict__ padtab,
    uint32_t level,
    FpExt lam,
    const FpExt *__restrict__ xi_rev,
    uint32_t log_len,
    FpExt *__restrict__ out
) {
    extern __shared__ FpExt red_smem[];
    const FpExt one(Fp::one());
    FpExt acc(Fp::zero());
    for (uint32_t j = threadIdx.x; j < n_pads; j += blockDim.x) {
        SegBlk b = pads[j];
        FpExt e = one;
        uint32_t k = log_len - b.log;
        for (uint32_t i = 0; i < k; i++) {
            FpExt x = xi_rev[i];
            e = e * (((b.id >> i) & 1u) ? x : one - x);
        }
        FracExt v = padtab[(size_t)b.konst * SEG_PAD_LEVELS + level];
        FpExt term = (v.p + lam * v.q) * v.q + v.p * v.q;
        acc += e * term;
    }
    FpExt total = reduce::block_reduce_sum(acc, red_smem);
    if (threadIdx.x == 0) out[0] = total;
}

// precompute_m_build_partial_kernel over a segmented vector at level c_src (block entries
// 2^(log - c_src), stored from base >> src_sh): a tail point is a (block, local tail point) pair,
// enumerated in storage order (sh storage entries each), its eq weight the tail table's at
// (lt << (log_len - log)) | id. Pad blocks add the same constant to every M entry, which the host
// adds to the round sums instead.
template <bool fold_inline, uint32_t LOGW>
__global__ void seg_precompute_m_build_partial_kernel(
    const FracExt *__restrict__ src,
    uint32_t src_sh,
    SegTab tab,
    uint32_t c_src,
    uint32_t n_items,
    FpExt lam,
    FpExt fold_r,
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
    const uint32_t sh = c_src + (fold_inline ? 2 : 1) + LOGW;
    uint32_t t_begin = blockIdx.x * tile_len;
    uint32_t t_end = min(t_begin + tile_len, n_items);

    constexpr uint32_t pad_stride = side + 1;
    extern __shared__ FpExt smem_pool[];
    FpExt *s_l0 = smem_pool;
    FpExt *s_l1 = s_l0 + pad_stride * TAIL_CHUNK;
    FpExt *s_r0 = s_l1 + pad_stride * TAIL_CHUNK;
    FpExt *s_r1 = s_r0 + pad_stride * TAIL_CHUNK;
    FpExt *s_w = s_r1 + pad_stride * TAIL_CHUNK;
    // Per tail point: where its local tail point starts in src, its block's tail length and
    // per-polynomial half.
    uint32_t *s_in = reinterpret_cast<uint32_t *>(s_w + TAIL_CHUNK);
    uint32_t *s_tl = s_in + TAIL_CHUNK;
    uint32_t *s_half = s_tl + TAIL_CHUNK;

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

        for (uint32_t slot = flat_tid; slot < chunk_len; slot += n_threads) {
            uint32_t u = chunk0 + slot;
            SegBlk b = seg_block_at(tab, (size_t)u << sh);
            uint32_t lt = u - (b.base >> sh);
            uint32_t half = 1u << (b.log - c_src - 1);
            s_in[slot] = (b.base >> src_sh) + lt;
            s_tl[slot] = 1u << (b.log - sh);
            s_half[slot] = half;
            uint32_t eq_pt = (lt << (tab.log_len - b.log)) | b.id;
            s_w[slot] = eq_split_lookup(tail_eq_lo, tail_eq_hi, tail_lo_bits, eq_pt);
        }
        __syncthreads();

        for (uint32_t pos = flat_tid; pos < chunk_elems; pos += n_threads) {
            uint32_t hi_bits = pos / chunk_len;
            uint32_t slot = pos - hi_bits * chunk_len;
            const FracExt *in = src + s_in[slot] + hi_bits * s_tl[slot];
            uint32_t half = s_half[slot];
            FpExt n0, d0, n1, d1;
            if constexpr (fold_inline) {
                uint32_t fold_off = half >> 1;
                FracExt lo0 = frac_load(in);
                FracExt hi0 = frac_load(in + fold_off);
                FracExt lo1 = frac_load(in + half);
                FracExt hi1 = frac_load(in + half + fold_off);
                n0 = lo0.p + fold_r * (hi0.p - lo0.p);
                d0 = lo0.q + fold_r * (hi0.q - lo0.q);
                n1 = lo1.p + fold_r * (hi1.p - lo1.p);
                d1 = lo1.q + fold_r * (hi1.q - lo1.q);
            } else {
                FracExt fr0 = frac_load(in);
                FracExt fr1 = frac_load(in + half);
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

// multifold_kernel on every real block of a segmented vector at level c_src, folding LOGW
// variables; the result, at level c_src + LOGW, goes to base >> dst_sh in dst. In place
// (src == dst, src_sh == dst_sh) is safe: column col stores to slots col and n_cols + col, both
// among the slots it loaded itself (bk = 0 and bk = 1 of the first polynomial).
template <uint32_t LOGW>
__global__ void seg_multifold_kernel(
    const FracExt *src,
    uint32_t src_sh,
    FracExt *dst,
    uint32_t dst_sh,
    SegTab tab,
    uint32_t c_src,
    uint32_t n_work,
    const FpExt *__restrict__ win_eq
) {
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_work) return;
    const uint32_t sh = c_src + 1 + LOGW;
    SegBlk b = seg_block_at(tab, (size_t)w << sh);
    uint32_t col = w - (b.base >> sh);
    uint32_t half = 1u << (b.log - c_src - 1);
    uint32_t n_cols = half >> LOGW;
    const FracExt *in = src + (b.base >> src_sh);
    FracExt *out = dst + (b.base >> dst_sh);
    FpExt zero(Fp::zero());
    FpExt n0_sum = zero, d0_sum = zero, n1_sum = zero, d1_sum = zero;
#pragma unroll
    for (uint32_t bk = 0; bk < (1u << LOGW); ++bk) {
        uint32_t pos = bk * n_cols + col;
        FracExt fv0 = in[pos];
        FracExt fv1 = in[half + pos];
        FpExt wgt = win_eq[bk];
        n0_sum += wgt * fv0.p;
        d0_sum += wgt * fv0.q;
        n1_sum += wgt * fv1.p;
        d1_sum += wgt * fv1.q;
    }
    out[col] = {n0_sum, d0_sum};
    out[n_cols + col] = {n1_sum, d1_sum};
}

extern "C" int _kb_gkr_seg_precompute_m_build(
    const FracExt *src,
    uint32_t src_sh,
    size_t storage_len,
    SegTab tab,
    uint32_t c_src,
    uint32_t wbits,
    bool fold_inline,
    FpExt lam,
    FpExt fold_r,
    const FpExt *tail_eq_lo,
    const FpExt *tail_eq_hi,
    size_t tail_eq_lo_len,
    size_t tile_len,
    FpExt *partials,
    size_t partials_len,
    FpExt *m_mat
) {
    uint32_t sh = c_src + (fold_inline ? 2 : 1) + wbits;
    if (sh > tab.g || wbits < 1 || wbits > 5 || tile_len == 0) return cudaErrorInvalidValue;
    uint32_t n_items = (uint32_t)(storage_len >> sh);
    uint32_t tile32 = (uint32_t)tile_len;
    uint32_t n_parts = div_ceil(n_items, tile32);
    uint32_t side = 1u << wbits;
    size_t n_entries = (size_t)side * side;
    if (partials_len < (size_t)n_parts * n_entries) return cudaErrorInvalidValue;
    uint32_t tail_lo_bits = __builtin_ctz((uint32_t)tail_eq_lo_len);
    dim3 grid(n_parts), block(side, side);
    size_t smem = (4 * (side + 1) * TAIL_CHUNK + TAIL_CHUNK) * sizeof(FpExt) + 3 * TAIL_CHUNK * sizeof(uint32_t);
#define SEG_M_LAUNCH(FI, W) \
    seg_precompute_m_build_partial_kernel<FI, W><<<grid, block, smem>>>( \
        src, src_sh, tab, c_src, n_items, lam, fold_r, tail_eq_lo, tail_eq_hi, tail_lo_bits, tile32, partials)
#define SEG_M_W(W) if (fold_inline) { SEG_M_LAUNCH(true, W); } else { SEG_M_LAUNCH(false, W); } break;
    switch (wbits) {
        case 1: SEG_M_W(1)
        case 2: SEG_M_W(2)
        case 3: SEG_M_W(3)
        case 4: SEG_M_W(4)
        case 5: SEG_M_W(5)
    }
#undef SEG_M_W
#undef SEG_M_LAUNCH
    int rc = CHECK_KERNEL();
    if (rc != 0) return rc;
    dim3 fin_block(128);
    dim3 fin_grid(div_ceil((uint32_t)n_entries, fin_block.x));
    precompute_m_reduce_partials_kernel<<<fin_grid, fin_block>>>(partials, n_parts, (uint32_t)n_entries, m_mat);
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_seg_multifold(
    const FracExt *src,
    uint32_t src_sh,
    FracExt *dst,
    uint32_t dst_sh,
    size_t storage_len,
    SegTab tab,
    uint32_t c_src,
    uint32_t wbits,
    const FpExt *win_eq
) {
    uint32_t sh = c_src + 1 + wbits;
    if (sh > tab.g) return cudaErrorInvalidValue;
    uint32_t n_work = (uint32_t)(storage_len >> sh);
    auto [grid, block] = kernel_launch_params(n_work, 256);
#define SEG_MF(W) seg_multifold_kernel<W><<<grid, block>>>(src, src_sh, dst, dst_sh, tab, c_src, n_work, win_eq); break;
    switch (wbits) {
        case 2: SEG_MF(2)
        case 3: SEG_MF(3)
        case 4: SEG_MF(4)
        case 5: SEG_MF(5)
        case 6: SEG_MF(6)
        default: return cudaErrorInvalidValue;
    }
#undef SEG_MF
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_seg_k3(FracExt *storage, size_t storage_len, SegTab tab, FpExt den_shift) {
    constexpr uint32_t TILE_W = 8;
    constexpr uint32_t tpb = 256;
    if (storage_len == 0) return 0;
    if (storage_len % (tpb * TILE_W) != 0 || tab.g < 11) return cudaErrorInvalidValue;
    constexpr size_t smem_bytes_vec = (size_t)tpb * (TILE_W + 1) * sizeof(FracExt);
    static std::atomic<int> smem_set[16] = {};
    int dev_slot = 0;
    cudaGetDevice(&dev_slot);
    if (dev_slot < 0 || dev_slot >= 16) return cudaErrorInvalidDevice;
    if (!smem_set[dev_slot].load(std::memory_order_acquire)) {
        cudaError_t rc = cudaFuncSetAttribute(
            bit_rev_frac_build_k2_vec_kernel<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes_vec);
        if (rc != cudaSuccess) return rc;
        smem_set[dev_slot].store(1, std::memory_order_release);
    }
    uint32_t n_grid = (uint32_t)(storage_len / (tpb * TILE_W));
    bit_rev_frac_build_k2_vec_kernel<false, true><<<dim3(n_grid), tpb, smem_bytes_vec>>>(
        storage, 0, den_shift, 0, tab);
    return CHECK_KERNEL();
}

// Builds every real block's tree from `level` up to `level + 1` (two = false) or `level + 2`.
extern "C" int _kb_gkr_seg_build(FracExt *storage, size_t storage_len, SegTab tab, uint32_t level, bool two) {
    uint32_t sh = level + (two ? 2 : 1);
    if (sh > tab.g) return cudaErrorInvalidValue;
    uint32_t n_work = (uint32_t)(storage_len >> sh);
    if (n_work == 0) return 0;
    auto [grid, block] = kernel_launch_params(n_work, 256);
    if (two) {
        seg_build_two_layers_kernel<<<grid, block>>>(storage, tab, level, n_work);
    } else {
        seg_build_layer_kernel<<<grid, block>>>(storage, tab, level, n_work);
    }
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_seg_round_revert(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    FracExt *storage,
    size_t storage_len,
    SegTab tab,
    uint32_t level,
    size_t eq_lo_len,
    FpExt lam,
    FpExt *evals_out,
    FpExt *scratch_sums
) {
    uint32_t sh = level + 2;
    if (sh > tab.g) return cudaErrorInvalidValue;
    uint32_t n_work = (uint32_t)(storage_len >> sh);
    auto [launch_grid, launch_block] = round_launch_shape(2 * n_work);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);
    seg_round_revert_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo, eq_hi, storage, tab, level, n_work, __builtin_ctz((uint32_t)eq_lo_len), lam, scratch_sums
    );
    int rc = CHECK_KERNEL();
    if (rc != 0) return rc;
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

extern "C" int _kb_gkr_seg_round_fold(
    const FpExt *eq_lo,
    const FpExt *eq_hi,
    const FracExt *src,
    uint32_t src_sh,
    FracExt *dst,
    uint32_t dst_sh,
    size_t storage_len,
    SegTab tab,
    uint32_t c_in,
    size_t eq_lo_len,
    FpExt lam,
    FpExt fold_r,
    FpExt *evals_out,
    FpExt *scratch_sums
) {
    uint32_t sh = c_in + 3;
    if (sh > tab.g) return cudaErrorInvalidValue;
    uint32_t n_work = (uint32_t)(storage_len >> sh);
    auto [launch_grid, launch_block] = round_launch_shape(2 * n_work);
    size_t smem_bytes = div_ceil(launch_block.x, WARP_SIZE) * sizeof(FpExt);
    seg_round_fold_kernel<<<launch_grid, launch_block, smem_bytes>>>(
        eq_lo, eq_hi, src, src_sh, dst, dst_sh, tab, c_in, n_work,
        __builtin_ctz((uint32_t)eq_lo_len), lam, fold_r, scratch_sums
    );
    int rc = CHECK_KERNEL();
    if (rc != 0) return rc;
    return gkr_final_reduce(scratch_sums, evals_out, launch_grid.x);
}

extern "C" int _kb_gkr_seg_gather(
    const FracExt *src,
    uint32_t src_sh,
    SegTab tab,
    const uint32_t *lgran,
    const FracExt *padtab,
    uint32_t pad_level,
    uint32_t c,
    FracExt *dst,
    uint32_t n_log
) {
    if (pad_level >= SEG_PAD_LEVELS || n_log == 0 || n_log > 31) return cudaErrorInvalidValue;
    auto [grid, block] = kernel_launch_params((size_t)1 << n_log, 256);
    seg_gather_kernel<<<grid, block>>>(src, src_sh, tab, lgran, padtab, pad_level, c, dst, n_log);
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_seg_pad_table(const FracExt *consts, uint32_t n_consts, FpExt den_shift, FracExt *padtab) {
    if (n_consts == 0) return 0;
    auto [grid, block] = kernel_launch_params(n_consts, 128);
    seg_pad_table_kernel<<<grid, block>>>(consts, n_consts, den_shift, padtab);
    return CHECK_KERNEL();
}

extern "C" int _kb_gkr_seg_pad_sum(
    const SegBlk *pads,
    uint32_t n_pads,
    const FracExt *padtab,
    uint32_t level,
    FpExt lam,
    const FpExt *xi_rev,
    uint32_t log_len,
    FpExt *out
) {
    if (level >= SEG_PAD_LEVELS) return cudaErrorInvalidValue;
    constexpr uint32_t tpb = 256;
    size_t smem_bytes = (tpb / WARP_SIZE) * sizeof(FpExt);
    seg_pad_sum_kernel<<<1, tpb, smem_bytes>>>(pads, n_pads, padtab, level, lam, xi_rev, log_len, out);
    return CHECK_KERNEL();
}

} // namespace fractional_sumcheck_gkr

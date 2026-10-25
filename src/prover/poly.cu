// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <vector_types.h>

// Evaluates sum_k cs[k] * at^k by Horner's rule, highest coefficient first.
template <typename Elem>
__device__ __forceinline__ Elem horner_eval(const Elem *cs, size_t n, Elem at) {
    Elem val = cs[n - 1];
    for (size_t k = n - 1; k > 0; k--) {
        val = val * at + cs[k - 1];
    }
    return val;
}

// ---------------------------------------------------------------------------- device kernels

// Grid-parallel polynomial evaluation with coalesced loads ("stride-Horner").
// Each FpExt coefficient is split across 4 base-field columns of `len` entries (column-major).
//
// Global thread `t` owns coefficients t, t+T, t+2T, ... where T = gridDim.x * blockDim.x:
//   p(x) = sum_t x^t * q_t(x^T),  q_t(y) = sum_k c_{t+kT} y^k   (Horner in y, per thread).
// At each Horner step the active threads of a warp load consecutive indices, so each base
// column costs one 128-byte transaction per warp -- unlike the previous single-block
// chunk-Horner, whose per-warp load addresses were `chunk_size` apart (one transaction per
// lane) and which occupied a single SM regardless of `len`.
//
// Each block reduces its threads to one partial via reduce::block_reduce_sum (warp shuffle
// intra-warp, shared memory only for inter-warp communication); eval_poly_ext_reduce_kernel
// folds the <= MAX_EVAL_POLY_BLOCKS partials. Field sums are exact, so splitting the sum
// cannot change the result, and the reduction is a fixed tree with no atomics, so it stays
// bitwise deterministic across runs.
static const uint32_t EVAL_POLY_BLOCK = 256;
static const uint32_t EVAL_POLY_MIN_COEFFS_PER_THREAD = 64;
static const uint32_t MAX_EVAL_POLY_BLOCKS = 256;

__global__ void eval_poly_ext_partial_kernel(
    const Fp *__restrict__ coeffs,
    size_t len,
    FpExt x,
    FpExt *__restrict__ block_partials
) {
    // Dynamic shared memory: num_warps elements for the block reduction.
    extern __shared__ FpExt smem[];

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    FpExt acc = FpExt(Fp(0));
    if (t < len) {
        // Highest k with t + k * stride <= len - 1.
        const size_t k_max = (len - 1 - t) / stride;
        const FpExt y = pow(x, stride);
        for (size_t k = k_max + 1; k > 0; k--) {
            const size_t idx = t + (k - 1) * stride;
            FpExt c;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                c.elems[i] = coeffs[i * len + idx];
            }
            acc = acc * y + c;
        }
        acc = acc * pow(x, t);
    }

    FpExt result = reduce::block_reduce_sum(acc, smem);
    if (threadIdx.x == 0) {
        block_partials[blockIdx.x] = result;
    }
}

// Folds the per-block partials into the single output value. One block; `num_partials` is at
// most MAX_EVAL_POLY_BLOCKS, so each thread accumulates at most one partial before the block
// reduction.
__global__ void eval_poly_ext_reduce_kernel(
    const FpExt *__restrict__ partials,
    uint32_t num_partials,
    FpExt *__restrict__ out
) {
    extern __shared__ FpExt smem[];

    FpExt acc = FpExt(Fp(0));
    for (uint32_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        acc = acc + partials[i];
    }

    FpExt result = reduce::block_reduce_sum(acc, smem);
    if (threadIdx.x == 0) {
        *out = result;
    }
}

// One in-place step of building the eq(r, .) table: entries [0, half) are split by r_i into
// entry j (times 1 - r_i) and entry j + half (times r_i), i.e. r_i becomes the top variable.
__global__ void eq_hypercube_stage_ext_kernel(FpExt *__restrict__ eq, FpExt r_i, uint32_t half) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half)
        return;
    FpExt upper = eq[j] * r_i;
    eq[j | half] = upper;
    eq[j] -= upper; // v * (1 - r_i) written as v - v * r_i to reuse the product
}

// Out-of-place variant of the step above: reads `src`, writes both halves to `dst`.
__global__ void eq_hypercube_nonoverlapping_stage_ext_kernel(
    FpExt *__restrict__ dst,
    const FpExt *__restrict__ src,
    FpExt r_i,
    uint32_t half
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half)
        return;
    FpExt cur = src[j];
    FpExt upper = cur * r_i;
    dst[j | half] = upper;
    dst[j] = cur - upper; // reuses the r_i product instead of multiplying by (1 - r_i)
}

// Out-of-place step that makes r_i the lowest variable: the two children of entry j land at
// 2j and 2j + 1.
__global__ void eq_hypercube_interleaved_stage_ext_kernel(
    FpExt *__restrict__ dst,
    const FpExt *__restrict__ src,
    FpExt r_i,
    uint32_t half
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half)
        return;
    FpExt cur = src[j];
    FpExt upper = cur * r_i;
    dst[(j << 1) | 1] = upper;
    dst[j << 1] = cur - upper;
}

// v[pos] *= s for every pos < n.
template <typename Elem>
__global__ void vector_scalar_multiply_kernel(Elem *v, Elem s, uint32_t n) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n)
        return;

    v[pos] *= s;
}

// Gathers four base-field columns of length n (one per extension limb) into n extension elements.
__global__ void transpose_fp_to_fpext_vec_kernel(
    FpExt *__restrict__ dst,
    const Fp *__restrict__ src,
    uint32_t n
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n)
        return;

#pragma unroll
    for (int limb = 0; limb < 4; limb++) {
        dst[j].elems[limb] = src[limb * n + j];
    }
}

// ---------------------------------------------------------------------------- host entry points

extern "C" int _kb_eq_hypercube_stage_ext(FpExt *eq, FpExt r_i, uint32_t half) {
    auto [grid, block] = kernel_launch_params(half);
    eq_hypercube_stage_ext_kernel<<<grid, block>>>(eq, r_i, half);
    return CHECK_KERNEL();
}

extern "C" int _kb_eq_hypercube_nonoverlapping_stage_ext(
    FpExt *dst,
    const FpExt *src,
    FpExt r_i,
    uint32_t half
) {
    auto [grid, block] = kernel_launch_params(half);
    eq_hypercube_nonoverlapping_stage_ext_kernel<<<grid, block>>>(dst, src, r_i, half);
    return CHECK_KERNEL();
}

extern "C" int _kb_eq_hypercube_interleaved_stage_ext(
    FpExt *dst,
    const FpExt *src,
    FpExt r_i,
    uint32_t half
) {
    auto [grid, block] = kernel_launch_params(half);
    eq_hypercube_interleaved_stage_ext_kernel<<<grid, block>>>(dst, src, r_i, half);
    return CHECK_KERNEL();
}

// Grid size for `_kb_eval_poly_ext_at_point`. The Rust wrapper calls this first so it can size the
// `partials` scratch buffer through the shared device-memory pool.
extern "C" uint32_t _kb_eval_poly_ext_num_blocks(size_t len) {
    size_t blocks = div_ceil(len, (size_t)EVAL_POLY_BLOCK * EVAL_POLY_MIN_COEFFS_PER_THREAD);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > (size_t)MAX_EVAL_POLY_BLOCKS) {
        blocks = (size_t)MAX_EVAL_POLY_BLOCKS;
    }
    return (uint32_t)blocks;
}

extern "C" int _kb_eval_poly_ext_at_point(
    const Fp *coeffs,
    size_t len,
    FpExt x,
    FpExt *partials,
    uint32_t num_blocks,
    FpExt *out
) {
    // Shared memory: num_warps elements for the block reduction's inter-warp communication.
    constexpr uint32_t num_warps = (EVAL_POLY_BLOCK + WARP_SIZE - 1) / WARP_SIZE;
    const size_t smem_size = num_warps * sizeof(FpExt);

    eval_poly_ext_partial_kernel<<<num_blocks, EVAL_POLY_BLOCK, smem_size>>>(
        coeffs, len, x, partials
    );
    int err = CHECK_KERNEL();
    if (err != 0) {
        return err;
    }

    eval_poly_ext_reduce_kernel<<<1, EVAL_POLY_BLOCK, smem_size>>>(partials, num_blocks, out);
    return CHECK_KERNEL();
}

extern "C" int _kb_vector_scalar_multiply_ext(FpExt *v, FpExt s, uint32_t n) {
    auto [grid, block] = kernel_launch_params(n);
    vector_scalar_multiply_kernel<FpExt><<<grid, block>>>(v, s, n);
    return CHECK_KERNEL();
}

extern "C" int _kb_transpose_fp_to_fpext_vec(FpExt *dst, const Fp *src, uint32_t n) {
    auto [grid, block] = kernel_launch_params(n);
    transpose_fp_to_fpext_vec_kernel<<<grid, block>>>(dst, src, n);
    return CHECK_KERNEL();
}

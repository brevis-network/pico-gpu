// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

// ---------------------------------------------------------------------------- device kernels

// Collapses the univariate-skip dimension: every group of coset_size consecutive rows of a column is
// read as evaluations on the coset D and replaced by the interpolant's value at r_0 (supplied through
// the precomputed omega powers and inverse Lagrange denominators). With USE_NEXT_ROW the column is
// read one row ahead (cyclically), giving the folded rotated trace.
// One thread owns one output cell and sums its coset_size terms serially; exact field addition makes
// that order-independent, so no cross-thread reduction is needed.
// Launch shape: x covers output rows, y is the column index.
template <bool USE_NEXT_ROW>
__global__ void fold_ple_from_evals_kernel(
    const Fp *__restrict__ evals,      // column-major, n_rows per column
    FpExt *__restrict__ folded,        // column-major, n_out_rows per column
    const Fp *__restrict__ omega_pows, // coset_size entries
    const FpExt *inv_denoms,           // coset_size entries
    uint32_t n_rows,
    uint32_t coset_size, // == 1 << log_coset
    uint32_t log_coset,
    uint32_t n_out_rows // lifted row count divided by coset_size
) {
    uint32_t c = blockIdx.y;
    uint32_t out_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= n_out_rows) {
        return;
    }

    uint32_t shift = USE_NEXT_ROW ? 1 : 0;
    FpExt sum(Fp::zero());
    for (uint32_t k = 0; k < coset_size; k++) {
        uint32_t src_row = ((out_row << log_coset) + k + shift) % n_rows;
        uint32_t src_pos = c * n_rows + src_row;
        Fp v = evals[src_pos];

        // k-th Lagrange basis weight at r_0 times the k-th coset evaluation
        sum = sum + inv_denoms[k] * omega_pows[k] * v;
    }

    folded[c * n_out_rows + out_row] = sum;
}

// For each column and each pair (col[2j], col[2j+1]) writes the line through them evaluated at
// 1, 2, ..., n_points into n_points consecutive output slabs of half_len values.
__global__ void interpolate_columns_kernel(
    FpExt *__restrict__ dst,
    const FpExt *__restrict__ const *__restrict__ col_ptrs,
    uint32_t n_points,
    uint32_t half_len,
    uint32_t n_cols
) {
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t slot = gid % half_len;
    uint32_t c = gid / half_len;
    if (c >= n_cols)
        return;

    const FpExt *__restrict__ col = col_ptrs[c];
    auto at0 = col[slot << 1];
    auto at1 = col[(slot << 1) | 1];
    FpExt *__restrict__ col_out = dst + c * n_points * half_len;

    for (int k = 0; k < n_points; k++) {
        col_out[k * half_len + slot] = at0 + (at1 - at0) * Fp(k + 1u);
    }
}

// Periodically extends each column of (numerator, denominator) pairs from src_rows to dst_rows rows.
__global__ void frac_matrix_vertically_repeat_kernel(
    std::pair<FpExt, FpExt> *__restrict__ dst,
    const std::pair<FpExt, FpExt> *__restrict__ src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y + blockIdx.z * gridDim.y;
    if (r >= dst_rows || c >= n_cols) {
        return;
    }
    dst[c * dst_rows + r].first = src[c * src_rows + (r % src_rows)].first;
    dst[c * dst_rows + r].second = src[c * src_rows + (r % src_rows)].second;
}

// Same periodic extension when numerators and denominators live in two separate buffers, as used
// by the GKR input path.
__global__ void frac_matrix_vertically_repeat_mixed_kernel(
    FpExt *__restrict__ num_dst,
    FpExt *__restrict__ den_dst,
    const FpExt *__restrict__ num_src,
    const FpExt *__restrict__ den_src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y + blockIdx.z * gridDim.y;
    if (r >= dst_rows || c >= n_cols) {
        return;
    }
    uint32_t r_src = r % src_rows;
    size_t from = c * src_rows + r_src;
    size_t to = c * dst_rows + r;
    num_dst[to] = num_src[from];
    den_dst[to] = den_src[from];
}

// ---------------------------------------------------------------------------- host entry points

constexpr uint32_t GRID_YZ_LIMIT = 65535u;

extern "C" int _kb_fold_ple_from_evals(
    const Fp *evals,
    FpExt *folded,
    const Fp *omega_pows,
    const FpExt *inv_denoms,
    uint32_t n_rows,
    uint32_t n_cols,
    uint32_t log_coset,
    uint32_t n_out_rows,
    bool use_next_row
) {
    uint32_t coset_size = 1u << log_coset;

    // One thread per output cell, so no shared memory is requested.
    uint32_t threads = 256u;

    // grid.x tiles the output rows, grid.y walks the columns
    dim3 grid(div_ceil(n_out_rows, threads), n_cols);
    dim3 block(threads);

    if (use_next_row) {
        fold_ple_from_evals_kernel<true><<<grid, block>>>(
            evals,
            folded,
            omega_pows,
            inv_denoms,
            n_rows,
            coset_size,
            log_coset,
            n_out_rows
        );
    } else {
        fold_ple_from_evals_kernel<false><<<grid, block>>>(
            evals,
            folded,
            omega_pows,
            inv_denoms,
            n_rows,
            coset_size,
            log_coset,
            n_out_rows
        );
    }
    return CHECK_KERNEL();
}

extern "C" int _kb_interpolate_columns(
    FpExt *dst,
    const FpExt *const *col_ptrs,
    size_t n_points,
    size_t half_len,
    size_t n_cols
) {
    auto [grid, block] = kernel_launch_params(half_len * n_cols, 512);

    interpolate_columns_kernel<<<grid, block>>>(dst, col_ptrs, n_points, half_len, n_cols);
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_matrix_vertically_repeat(
    std::pair<FpExt, FpExt> *dst,
    const std::pair<FpExt, FpExt> *src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    auto [grid, block] = kernel_launch_params(dst_rows);
    grid.y = std::min(n_cols, GRID_YZ_LIMIT);
    grid.z = (n_cols + grid.y - 1) / grid.y;
    assert(grid.z <= GRID_YZ_LIMIT);
    frac_matrix_vertically_repeat_kernel<<<grid, block>>>(dst, src, n_cols, dst_rows, src_rows);
    return CHECK_KERNEL();
}

extern "C" int _kb_frac_matrix_vertically_repeat_ext(
    FpExt *num_dst,
    FpExt *den_dst,
    const FpExt *num_src,
    const FpExt *den_src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    auto [grid, block] = kernel_launch_params(dst_rows);
    grid.y = std::min(n_cols, GRID_YZ_LIMIT);
    grid.z = (n_cols + grid.y - 1) / grid.y;
    assert(grid.z <= GRID_YZ_LIMIT);
    frac_matrix_vertically_repeat_mixed_kernel<<<grid, block>>>(
        num_dst,
        den_dst,
        num_src,
        den_src,
        n_cols,
        dst_rows,
        src_rows
    );
    return CHECK_KERNEL();
}

} // namespace

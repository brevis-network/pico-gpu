// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

// Edge length of the square transpose tile; also the block size, so the kernel relies on it being 32.
const size_t TRANSPOSE_TILE = 32;

template <typename T>
__global__ void __launch_bounds__(
    TRANSPOSE_TILE
) matrix_transpose_kernel(T *dst, const T *src, size_t ncols, size_t nrows) {

    // Logically `__shared__ T tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1]` (the +1 column avoids bank
    // conflicts). It is declared as raw bytes so the file also parses under clang-tidy.
    __shared__ __align__(
        alignof(T)
    ) unsigned char tile_storage[TRANSPOSE_TILE * (TRANSPOSE_TILE + 1) * sizeof(T)];
    T(*tile)[TRANSPOSE_TILE + 1] = reinterpret_cast<T(*)[TRANSPOSE_TILE + 1]>(tile_storage);

    size_t tiles_across = (ncols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    size_t flat_block = blockIdx.x; // 1-D grid, unflattened into tile coordinates below
    size_t tile_r = flat_block / tiles_across;
    size_t tile_c = flat_block % tiles_across;

    size_t lane = threadIdx.x;
    size_t src_base = tile_r * TRANSPOSE_TILE * ncols + tile_c * TRANSPOSE_TILE + lane;
    size_t dst_base = tile_c * TRANSPOSE_TILE * nrows + tile_r * TRANSPOSE_TILE + lane;

    // Load phase: each lane reads one column of the tile, row by row, zero-filling outside the matrix.
    bool lane_ok = tile_c * TRANSPOSE_TILE + lane < ncols;
    size_t first_row = tile_r * TRANSPOSE_TILE + 0;
    for (auto k = 0; k < TRANSPOSE_TILE; ++k) {
        bool in_bounds = lane_ok && (first_row + k < nrows);
        tile[k][lane] = (in_bounds) ? src[src_base + k * ncols] : T(0);
    }
    __syncthreads();

    // Store phase: read the tile transposed and write only the in-range cells.
    lane_ok = tile_r * TRANSPOSE_TILE + lane < nrows;
    first_row = tile_c * TRANSPOSE_TILE + 0;
    for (auto k = 0; k < TRANSPOSE_TILE; ++k) {
        bool in_bounds = lane_ok && (first_row + k < ncols);
        if (in_bounds)
            dst[dst_base + k * nrows] = tile[lane][k];
    }
}

// Instantiate for the two element types the launchers need.
template __global__ void matrix_transpose_kernel<Fp>(Fp *, const Fp *, size_t, size_t);
template __global__ void matrix_transpose_kernel<FpExt>(FpExt *, const FpExt *, size_t, size_t);

__global__ void matrix_get_rows_fp_kernel(
    Fp *__restrict__ dst,
    const Fp *__restrict__ src,
    uint32_t *__restrict__ picked_rows,
    uint64_t n_cols,
    uint64_t n_rows
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cols) {
        return;
    }
    // blockIdx.y indexes picked_rows, not the matrix, so it may legitimately exceed n_rows.

    uint64_t src_row = picked_rows[blockIdx.y];
    uint64_t dst_row = blockIdx.y;
    uint64_t src_pos = c * n_rows + src_row; // source is column-major
    uint64_t dst_pos = dst_row * n_cols + c; // gathered rows are written row-major
    dst[dst_pos] = src[src_pos];
}

__global__ void split_ext_to_base_col_major_matrix_kernel(
    Fp *__restrict__ base_cols,
    const FpExt *__restrict__ ext_cols,
    uint64_t total_len,
    uint32_t rows
) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }

    uint32_t n_ext_cols = (total_len / rows);
    for (uint32_t c = 0; c < n_ext_cols; c++) {
        FpExt x = ext_cols[c * rows + r];
        base_cols[(c * 4 + 0) * rows + r] = x.elems[0];
        base_cols[(c * 4 + 1) * rows + r] = x.elems[1];
        base_cols[(c * 4 + 2) * rows + r] = x.elems[2];
        base_cols[(c * 4 + 3) * rows + r] = x.elems[3];
    }
}

// Each of the n_cols columns of `src` has seg_len * segments rows; each segment of seg_len rows is
// written to `dst` as padded_len rows. The values are shifted up by one row, with wrap-around
// over the whole column (seg_len * segments rows), while zero padding is applied per segment.
__global__ void batch_rotate_pad_kernel(
    Fp *__restrict__ dst,
    const Fp *__restrict__ src,
    uint32_t n_cols,
    uint32_t segments,
    uint32_t seg_len,
    uint32_t padded_len
) {
    auto pos = threadIdx.x + blockIdx.x * blockDim.x;
    auto poly = blockIdx.y + blockIdx.z * gridDim.y;

    if (poly >= n_cols * segments) {
        return;
    }
    if (pos < seg_len) {
        auto next_pos = pos + 1;
        auto next_poly = poly;
        if (next_pos == seg_len) {
            next_pos = 0;
            next_poly += 1;
            // wrap to the start of this column once its last segment is passed
            if (next_poly % segments == 0) {
                next_poly -= segments;
            }
        }
        dst[padded_len * poly + pos] = src[seg_len * next_poly + next_pos];
    } else if (pos < padded_len) {
        dst[padded_len * poly + pos] = Fp(0);
    }
}

// In-place lift of a column-major buffer whose columns are col_stride apart: rows [0, lifted_rows) of
// each column become the first base_rows rows repeated periodically. Rows past lifted_rows keep their
// current contents.
__global__ void lift_padded_matrix_evals_kernel(
    Fp *__restrict__ buf,
    uint32_t n_cols,
    uint32_t base_rows,
    uint32_t lifted_rows,
    uint32_t col_stride
) {
    auto r = threadIdx.x + blockIdx.x * blockDim.x;
    auto c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r >= lifted_rows || c >= n_cols) {
        return;
    }
    // for r < base_rows this rewrites the value already there
    buf[c * col_stride + r] = buf[c * col_stride + (r % base_rows)];
}

// Inverse of the lift: keeps every step-th row of each column. Caller ensures src_rows == dst_rows * step.
__global__ void collapse_strided_matrix_kernel(
    Fp *__restrict__ dst,
    const Fp *__restrict__ src,
    const uint32_t n_cols,
    const uint32_t src_rows,
    const uint32_t dst_rows,
    const uint32_t step
) {
    auto r = threadIdx.x + blockIdx.x * blockDim.x;
    auto c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r >= dst_rows || c >= n_cols) {
        return;
    }

    dst[c * dst_rows + r] = src[c * src_rows + r * step];
}

// Zero-extends each of num_polys contiguous polynomials from src_len to dst_len values.
// For example with 2 polynomials and dst_len = 2 * src_len:
//   src: | a | b |          dst: | a | 0 | b | 0 |
__global__ void batch_expand_pad_kernel(
    Fp *dst,
    const Fp *src,
    const uint32_t num_polys,
    const uint32_t dst_len,
    const uint32_t src_len
) {
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < dst_len) {
        for (uint32_t p = 0; p < num_polys; p++) {
            Fp v = Fp(0);
            if (pos < src_len) {
                v = src[p * src_len + pos];
            }
            dst[p * dst_len + pos] = v;
        }
    }
}

// Same zero-extension for column-major matrices (src_rows -> dst_rows rows per column), with the
// column index spread over grid y/z. Meant for very many short columns, e.g. ~2^20 columns of
// fewer than 2^10 rows.
__global__ void batch_expand_pad_wide_kernel(
    Fp *__restrict__ dst,
    const Fp *__restrict__ src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y + blockIdx.z * gridDim.y;
    if (c >= n_cols) {
        return;
    }
    if (r < src_rows) {
        dst[c * dst_rows + r] = src[c * src_rows + r];
    } else if (r < dst_rows) {
        dst[c * dst_rows + r] = Fp(0);
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

constexpr uint32_t GRID_YZ_LIMIT = 65535u;

template <typename T>
int launch_transpose(T *dst, const T *src, size_t ncols, size_t nrows) {
    uint32_t tiles_x = (ncols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    uint32_t tiles_y = (nrows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;

    dim3 grid(tiles_x * tiles_y);
    dim3 block(TRANSPOSE_TILE);

    matrix_transpose_kernel<T><<<grid, block>>>(dst, src, ncols, nrows);

    return CHECK_KERNEL();
}

extern "C" int _kb_matrix_transpose_fp(Fp *dst, const Fp *src, size_t ncols, size_t nrows) {
    return launch_transpose(dst, src, ncols, nrows);
}

extern "C" int _kb_matrix_transpose_fpext(
    FpExt *dst,
    const FpExt *src,
    size_t ncols,
    size_t nrows
) {
    return launch_transpose(dst, src, ncols, nrows);
}

extern "C" int _kb_matrix_get_rows_fp(
    Fp *dst,
    const Fp *src,
    uint32_t *picked_rows,
    uint64_t n_cols,
    uint64_t n_rows,
    uint32_t n_picked
) {
    if (n_cols == 0 || n_picked == 0) {
        return cudaSuccess;
    }
    if (n_picked > UINT16_MAX) {
        return cudaErrorInvalidValue;
    }
    auto block = WARP_SIZE;
    dim3 grid = dim3(div_ceil(n_cols, WARP_SIZE), n_picked);
    matrix_get_rows_fp_kernel<<<grid, block>>>(
        dst, src, picked_rows, n_cols, n_rows
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_split_ext_to_base_col_major_matrix(
    Fp *base_cols,
    FpExt *ext_cols,
    uint64_t total_len,
    uint32_t rows
) {
    auto [grid, block] = kernel_launch_params(rows);
    split_ext_to_base_col_major_matrix_kernel<<<grid, block>>>(
        base_cols, ext_cols, total_len, rows
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_batch_rotate_pad(
    Fp *dst,
    const Fp *src,
    uint32_t n_cols,
    uint32_t segments, // segments per column
    uint32_t seg_len,
    uint32_t padded_len
) {
    auto [grid, block] = kernel_launch_params(padded_len);
    auto n_polys = n_cols * segments;
    grid.y = std::min(n_polys, GRID_YZ_LIMIT);
    grid.z = (n_polys + grid.y - 1) / grid.y;
    batch_rotate_pad_kernel<<<grid, block>>>(dst, src, n_cols, segments, seg_len, padded_len);
    return CHECK_KERNEL();
}

extern "C" int _kb_lift_padded_matrix_evals(
    Fp *buf,
    uint32_t n_cols,
    uint32_t base_rows,
    uint32_t lifted_rows,
    uint32_t col_stride
) {
    auto [grid, block] = kernel_launch_2d_params(lifted_rows, n_cols);
    lift_padded_matrix_evals_kernel<<<grid, block>>>(
        buf, n_cols, base_rows, lifted_rows, col_stride
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_collapse_strided_matrix(
    Fp *dst,
    const Fp *src,
    uint32_t n_cols,
    uint32_t dst_rows,
    uint32_t step
) {
    auto src_rows = dst_rows * step;
    auto [grid, block] = kernel_launch_2d_params(dst_rows, n_cols);
    collapse_strided_matrix_kernel<<<grid, block>>>(dst, src, n_cols, src_rows, dst_rows, step);
    return CHECK_KERNEL();
}

extern "C" int _kb_batch_expand_pad(
    Fp *dst,
    const Fp *src,
    const uint32_t num_polys,
    const uint32_t dst_len,
    const uint32_t src_len
) {
    auto [grid, block] = kernel_launch_params(dst_len);
    batch_expand_pad_kernel<<<grid, block>>>(dst, src, num_polys, dst_len, src_len);
    return CHECK_KERNEL();
}

extern "C" int _kb_batch_expand_pad_wide(
    Fp *dst,
    const Fp *src,
    const uint32_t n_cols,
    const uint32_t dst_rows,
    const uint32_t src_rows
) {
    auto [grid, block] = kernel_launch_params(dst_rows);
    grid.y = std::min(n_cols, GRID_YZ_LIMIT);
    grid.z = (n_cols + grid.y - 1) / grid.y;
    assert(grid.z <= GRID_YZ_LIMIT);
    batch_expand_pad_wide_kernel<<<grid, block>>>(dst, src, n_cols, dst_rows, src_rows);
    return CHECK_KERNEL();
}

// ============================================================================
// Stacked-trace gather
// ============================================================================
// One pass building the expanded stacked buffer that `stack_traces_into_expanded`
// assembles with a full-buffer memset + one D2D memcpy / expand-pad launch per
// unstacked column (~633 launches for the reth RISC-V set). Every output cell
// binary-searches the descriptor table (sorted by dst_start, L1-resident) and
// either gathers its source value or writes the zero padding inline.
struct StackGatherDesc {
    uint64_t src;       // device pointer to the column's source data
    uint64_t dst_start; // flat output index where this slice starts
    uint32_t len;       // slice length in the output (s_len = stride * src_len)
    uint32_t stride;    // power of two; 1 for plain copies, else src[i] lands at offset i*stride
};

// Count of descriptors with `dst_start <= idx`; the owner is `result - 1`, and 0 means padding.
__device__ __forceinline__ uint32_t gather_desc_upper_bound(
    const StackGatherDesc *__restrict__ descs,
    uint32_t n_descs,
    size_t idx
) {
    uint32_t lo = 0, hi = n_descs;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (descs[mid].dst_start <= idx) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void stack_gather_kernel(
    Fp *__restrict__ out,
    const StackGatherDesc *__restrict__ descs,
    uint32_t n_descs,
    size_t total_cells
) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;

    uint32_t const lo = gather_desc_upper_bound(descs, n_descs, idx);
    Fp v = Fp(0);
    if (lo > 0) {
        const StackGatherDesc d = descs[lo - 1];
        size_t o = idx - d.dst_start;
        if (o < d.len) {
            const Fp *src = reinterpret_cast<const Fp *>(d.src);
            if (d.stride == 1) {
                v = src[o];
            } else if ((o & (size_t)(d.stride - 1)) == 0) {
                // `stride` is a power of two, so mask+shift, not a 64-bit division.
                v = src[o >> (__ffs(d.stride) - 1)];
            }
        }
    }
    out[idx] = v;
}

// 4-wide twin of `stack_gather_kernel`: one thread emits one 16-byte group, so the descriptor
// search runs once per four cells and the store is a single 128-bit access.
//
// Precondition, checked on the host by `stack_traces_gathered` and never assumed here: every
// `dst_start` and `len`, and `total_cells`, are multiples of 4. Hence a group never straddles a
// descriptor and both the store and the stride-1 load are 16-byte aligned.
//
// `out` is `Fp *`, cast at the store, because these sources compile into two link units that share
// one binary: `Fp` mangles per unit, a `uint4 *` signature would not, and the link would fail with
// `duplicate symbol`.
union GatherVec4 {
    uint4 v;
    Fp e[4];
    __device__ GatherVec4() {}
};

__global__ void stack_gather_vec4_kernel(
    Fp *__restrict__ out,
    const StackGatherDesc *__restrict__ descs,
    uint32_t n_descs,
    size_t total_groups
) {
    size_t const g = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (g >= total_groups) return;
    size_t const idx = g << 2;

    // A block-uniform fast path (thread 0 resolves the block via shared memory) was measured and
    // discarded: four cells per thread already cut the searches 4x, and the rest hides under load.
    uint32_t const lo = gather_desc_upper_bound(descs, n_descs, idx);

    GatherVec4 u;
    // Montgomery zero is bitwise zero (same premise as `stack_gather_any`).
    u.v = make_uint4(0u, 0u, 0u, 0u);
    if (lo > 0) {
        const StackGatherDesc d = descs[lo - 1];
        size_t const o = idx - d.dst_start;
        if (o < d.len) {
            const Fp *src = reinterpret_cast<const Fp *>(d.src);
            if (d.stride == 1) {
                u.v = *reinterpret_cast<const uint4 *>(src + o);
            } else {
                uint32_t const log_stride = __ffs(d.stride) - 1;
                size_t const mask = (size_t)(d.stride - 1);
#pragma unroll
                for (uint32_t k = 0; k < 4; ++k) {
                    size_t const ok = o + k;
                    if ((ok & mask) == 0) u.e[k] = src[ok >> log_stride];
                }
            }
        }
    }
    reinterpret_cast<uint4 *>(out)[g] = u.v;
}

extern "C" int _kb_stack_gather(
    Fp *out,
    const void *descs,
    uint32_t n_descs,
    size_t total_cells,
    bool vectorizable
) {
    if (total_cells == 0) return 0;
    uint32_t const block = 256;
    auto const *d = reinterpret_cast<const StackGatherDesc *>(descs);
    if (vectorizable) {
        size_t const groups = total_cells >> 2;
        size_t const grid = (groups + block - 1) / block;
        stack_gather_vec4_kernel<<<grid, block>>>(out, d, n_descs, groups);
    } else {
        size_t const grid = (total_cells + block - 1) / block;
        stack_gather_kernel<<<grid, block>>>(out, d, n_descs, total_cells);
    }
    return CHECK_KERNEL();
}

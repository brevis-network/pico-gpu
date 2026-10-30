// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "prover/ntt/device_ntt.cuh"
#include "field/fp.h"
#include "field/fpext.h"
#include "prover/frac_ext.cuh"
#include "common/launcher.cuh"
#include "prover/reduce.cuh"
#include "common/utils.cuh"
// standard library / CUDA runtime
#include <algorithm> // std::min / std::max
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>
#include <vector_types.h>

// Device NTT helpers are visible unqualified in this file.
using namespace device_ntt;

namespace {

constexpr uint32_t MAX_GRID_DIM = 65535u;
// Degree of the per-round sumcheck polynomial for rounds after round 0.
constexpr int S_DEG = 2;
// Round 0 emits three partial polynomials per z: G0, G1 and G2.
constexpr int NUM_G = 3;

} // namespace

// Layout must match the Rust `stacked_reduction::UnstackedSlice` (repr(C)).
struct UnstackedSlice {
    uint32_t commit_idx;
    uint32_t log_height;
    uint32_t stacked_row_idx;
    uint32_t stacked_col_idx;
};

struct ColumnFoldArgs {
    const Fp *__restrict__ src;
    FpExt *__restrict__ dst;
    uint32_t height;
    uint32_t width;
};

// Reads entry `pos` of the eq table for a hypercube of `n_cube` points (segments are stored at
// offset `n_cube`).
__device__ __forceinline__ FpExt eq_seg_at(const FpExt *segments, uint32_t n_cube, uint32_t pos) {
    return segments[n_cube + pos];
}

// Cyclic predecessor of `pos` on a cube of `n_cube` points.
__device__ __forceinline__ uint32_t prev_on_cube(uint32_t pos, uint32_t n_cube) {
    return pos == 0 ? n_cube - 1 : pos - 1;
}

// Multilinear eq for one variable: eq(a, b) = a*b + (1-a)*(1-b).
__device__ __forceinline__ Fp eq_bit(Fp a, Fp b) { return Fp::one() - a - b + Fp(2) * a * b; }

__device__ __forceinline__ Fp bary_eval_strided(
    const Fp *vals,
    uint32_t vals_len,
    uint32_t pt,
    uint32_t blowup,               // large domain size / skip domain size
    const Fp *inv_denoms_at_pt,    // Lagrange denominators already specialised to `pt`
    const Fp *omega_pows,
    uint32_t dom
) {
    Fp acc = Fp::zero();
    uint32_t step = vals_len >= dom ? 1 : dom / vals_len;
    if (pt % blowup == 0) {
        auto j = pt / blowup;
        if (j % step == 0)
            acc = vals[j / step];
    } else {
        for (int k = 0; k < (dom / step); k++) {
            acc += vals[k] * omega_pows[k * step] * inv_denoms_at_pt[k * step];
        }
    }
    return acc;
}

// Round 0 (SP_DEG = 1). Produces per-block partial sums of three polynomials, evaluated on the
// identity coset only; the CPU later recombines them with NTT-based multiplication, so no
// (i)NTT runs here.
//
// threadIdx.x enumerates a tile of z values; the block cooperates on the x-sum of one column
// selected by blockIdx.y.
//
//   G0(Z) = sum over (col, x) of  w_eq[col]  * eq(x)                  * q_{col,x}(Z)
//   G1(Z) = sum over (col, x) of  w_rot[col] * eq(x)                  * q_{col,x}(Z)
//   G2(Z) = sum over (col, x) of  w_rot[col] * (eq(x - 1) - eq(x))    * q_{col,x}(Z)
//
// with w_eq[col] = col_coeffs[2*col] and w_rot[col] = col_coeffs[2*col + 1]; `x - 1` is cyclic.
__global__ void r0_partial_sums_kernel(
    const FpExt *__restrict__ eq_segments, // EqEvalSegments base
    const Fp *__restrict__ trace,
    const FpExt *__restrict__ col_coeffs, // lambda powers, already offset to the window start
    FpExt *__restrict__ partials,         // [gridDim.x * gridDim.y][NUM_G * skip_domain]
    uint32_t rows,                        // trace height
    uint32_t cols,                        // trace width
    uint32_t log_skip,
    uint32_t z_mask,   // skip_domain - 1
    uint32_t cube_len, // 2^n_lift
    uint32_t log_z_step
) {
    extern __shared__ char smem[];
    // pitch is blockDim.x + 1 so the two rows do not alias the same banks
    const uint32_t row_pitch = blockDim.x + 1;
    FpExt *sum_tile = reinterpret_cast<FpExt *>(smem); // FpExt[2][row_pitch]

    uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t z = gtid & z_mask;
    uint32_t col = blockIdx.y;

    // The original kernel (81% SM / latency-bound on the per-block reduction tail) did all the
    // ext math per (x, z, col) thread and reduced per 256-thread block (~2M block tails).
    //  1. eq(x) / rot_delta depend only on x: for skip_domain <= 32 one leader lane per
    //     x-group computes them and warp-shuffles them to its 2^l_skip siblings;
    //  2. G0 and G1 share the inner product sum eq*q, so each thread accumulates only
    //     acc_eq = eq*q and acc_rot = rot_delta*q; per-column coefficients are applied once per
    //     (block, z) after the reduction (exact by distributivity);
    //  3. each thread strides over R0_X_PER_THREAD x-values, accumulating in registers — dividing
    //     block count, barrier waits and final-reduce work by the stride.
    uint32_t const z_count = z_mask + 1;
    bool const fits_warp = z_count <= 32;
    uint32_t const lane_id = threadIdx.x & 31u;
    uint32_t const x_first = gtid >> log_skip;
    uint32_t const x_step = (gridDim.x * blockDim.x) >> log_skip;
    uint32_t const n_trips = (cube_len + x_step - 1) / x_step;
    auto const z_step_mask = (1u << log_z_step) - 1;
    bool const z_sampled = (z & z_step_mask) == 0;

    FpExt acc_eq(Fp::zero());
    FpExt acc_rot(Fp::zero());
    for (uint32_t trip = 0; trip < n_trips; trip++) {
        uint32_t const x = x_first + trip * x_step;
        bool const x_valid = x < cube_len;

        FpExt eq_x, rot_delta;
        if (x_valid && (!fits_warp || (lane_id & z_mask) == 0)) {
            eq_x = eq_seg_at(eq_segments, cube_len, x);
            FpExt eq_prev = eq_seg_at(eq_segments, cube_len, prev_on_cube(x, cube_len));
            rot_delta = eq_prev - eq_x;
        }
        if (fits_warp) {
            // whole warp shuffles; values received by groups with no valid x are ignored
            uint32_t const leader = lane_id & ~z_mask;
            uint32_t *eq_words = reinterpret_cast<uint32_t *>(&eq_x);
            uint32_t *rot_words = reinterpret_cast<uint32_t *>(&rot_delta);
#pragma unroll
            for (int w = 0; w < 4; w++) {
                eq_words[w] = __shfl_sync(0xffffffffu, eq_words[w], leader);
                rot_words[w] = __shfl_sync(0xffffffffu, rot_words[w], leader);
            }
        }

        if (x_valid && z_sampled) {
            // identity coset: read the trace value directly
            auto col_rows = trace + col * rows + (x << (log_skip - log_z_step));
            Fp q_val = col_rows[z >> log_z_step];
            acc_eq += eq_x * q_val;
            acc_rot += rot_delta * q_val;
        }
    }

    sum_tile[0 * row_pitch + threadIdx.x] = acc_eq;
    sum_tile[1 * row_pitch + threadIdx.x] = acc_rot;

    __syncthreads();

    // the first x-slice of the block folds the other slices for its z
    if ((threadIdx.x >> log_skip) == 0) {
        FpExt sum_eq = sum_tile[0 * row_pitch + z];
        FpExt sum_rot = sum_tile[1 * row_pitch + z];

        for (int slice = 1; slice < (blockDim.x >> log_skip); ++slice) {
            sum_eq += sum_tile[0 * row_pitch + (slice << log_skip) + z];
            sum_rot += sum_tile[1 * row_pitch + (slice << log_skip) + z];
        }

        FpExt w_eq = col_coeffs[2 * col];      // G0 weight
        FpExt w_rot = col_coeffs[2 * col + 1]; // G1 / G2 weight

        // NUM_G consecutive runs of z_len values per block
        uint32_t z_len = 1u << log_skip;
        FpExt *dst =
            partials + (col * gridDim.x + blockIdx.x) * (NUM_G * z_len);
        dst[0 * z_len + z] = w_eq * sum_eq;
        dst[1 * z_len + z] = w_rot * sum_eq;
        dst[2 * z_len + z] = w_rot * sum_rot;
    }
}

// PLE fold, one group of `dom` threads per output cell; blockIdx.y picks the column.
// No longer the production path — the thread-per-cell kernel below took that over; this one now
// serves only launches too small for it (rows_out < 256), where per-cell threading would leave
// blocks nearly empty.
__global__ void fold_ple_chunked_kernel(
    const Fp *__restrict__ trace,
    FpExt *__restrict__ folded,
    const Fp *__restrict__ omega_pows,
    const FpExt *__restrict__ inv_denoms,
    uint32_t rows_in,
    uint32_t rows_out,
    uint32_t dom
) {
    extern __shared__ char smem_raw[];
    FpExt *scratch = reinterpret_cast<FpExt *>(smem_raw);

    uint32_t col = blockIdx.y;
    uint32_t cells_per_block = blockDim.x / dom;
    uint32_t cell_slot = threadIdx.x / dom;
    uint32_t term = threadIdx.x % dom;
    uint32_t out_row = blockIdx.x * cells_per_block + cell_slot;

    // no early exit: chunk_reduce_sum synchronises the whole block
    bool const cell_live = (out_row < rows_out);

    // n_terms <= dom, so every thread owns at most one barycentric term
    FpExt part(Fp::zero());
    if (cell_live) {
        uint32_t n_terms = std::min(rows_in, dom);
        uint32_t step = dom / n_terms;
        const Fp *cell_in = trace + col * rows_in + out_row * n_terms;

        if (term < n_terms) {
            uint32_t j = term;
            part =
                cell_in[j] * omega_pows[j * step] * inv_denoms[j * step];
        }
    }

    // every thread joins; dead cells add zero
    FpExt total =
        reduce::chunk_reduce_sum(part, scratch, term, dom, cell_slot);

    // each cell owns a distinct output, and nothing runs after the store, so no barrier
    if (cell_live && term == 0) {
        folded[col * rows_out + out_row] = total;
    }
}

// Thread-per-output-cell restructure of the single-trace fold (production-hot path).
// Bit-identical to the chunk_reduce_sum butterfly, on two counts: field addition is exact and
// associative, so serialising the accumulation cannot change the sum; and each term's own
// multiplication chain is reassociated, which is safe only because these multiplies are fully
// reduced Montgomery ops (each ends in an unconditional final_sub, so every intermediate is
// canonical and the product is order-independent). Drops the shuffles and the smem stage per
// output cell — measured 1.82x at the work-dominant shape. The chunk kernel above is kept for
// degenerate tiny launches (rows_out < 256).
__global__ void fold_ple_per_cell_kernel(
    const Fp *__restrict__ trace,
    FpExt *__restrict__ folded,
    const Fp *__restrict__ omega_pows,
    const FpExt *__restrict__ inv_denoms,
    uint32_t rows_in,
    uint32_t rows_out,
    uint32_t dom
) {
    uint32_t col = blockIdx.y;
    uint32_t out_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= rows_out)
        return;

    uint32_t n_terms = std::min(rows_in, dom);
    uint32_t step = dom / n_terms;
    const Fp *cell_in = trace + col * rows_in + out_row * n_terms;

    FpExt sum(Fp::zero());
    for (uint32_t j = 0; j < n_terms; j++) {
        sum += inv_denoms[j * step] * omega_pows[j * step] * cell_in[j];
    }
    folded[col * rows_out + out_row] = sum;
}

// Fills every k_rot segment (levels 0..=max_n, blockIdx.y = level) from the matching eq segment.
__global__ void k_rot_from_eq_kernel(
    const FpExt *eq_segments,
    FpExt *k_rot_segments,
    FpExt uni_eq_coeff,
    FpExt uni_rot_coeff
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level = blockIdx.y;
    uint32_t len = 1 << level;

    if (pos >= len)
        return;

    auto eq_here = eq_seg_at(eq_segments, len, pos);
    auto eq_before = eq_seg_at(eq_segments, len, prev_on_cube(pos, len));

    k_rot_segments[len + pos] = uni_eq_coeff * eq_here + uni_rot_coeff * (eq_before - eq_here);
}

// General round (n = n_lift > 0). Each thread owns one y and walks the window with stride
// gridDim.y. Each of the S_DEG sums is block-reduced through `warp_buf` (one FpExt per warp,
// reused, hence the barrier between degrees) and thread 0 atomically adds it to the u64 output.
__global__ void mle_round_window_kernel(
    const FpExt *__restrict__ const
        *__restrict__ q_mats,                 // one matrix per commit, all of height q_rows
    const FpExt *__restrict__ eq_segments,    // EqEvalSegments base
    const FpExt *__restrict__ k_rot_segments, // k_rot segments base
    const UnstackedSlice *__restrict__ slices, // window of unstacked columns
    const FpExt *__restrict__ coeffs,          // lambda powers for the same window
    uint64_t *__restrict__ acc_out, // S_DEG x 4 u64 limbs, summed atomically, finished on the CPU
    uint32_t q_rows,
    uint32_t n_slices,
    uint32_t y_count
) {
    extern __shared__ char smem[];
    FpExt *warp_buf = (FpExt *)smem;
    FpExt partial[S_DEG];
#pragma unroll
    for (int d = 0; d < S_DEG; d++) {
        partial[d] = FpExt(0);
    }

    uint32_t first_slice = blockIdx.y;
    uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
    bool const y_valid = (y < y_count);

    if (y_valid) {
        uint32_t eval_len = y_count * 2;

        auto eq_lo = eq_seg_at(eq_segments, eval_len, y << 1);
        auto eq_hi = eq_seg_at(eq_segments, eval_len, (y << 1) | 1);
        auto eq_slope = eq_hi - eq_lo;
        auto rot_lo = eq_seg_at(k_rot_segments, eval_len, y << 1);
        auto rot_hi = eq_seg_at(k_rot_segments, eval_len, (y << 1) | 1);
        auto rot_slope = rot_hi - rot_lo;

        // gridDim.y (chosen by the launcher) is the window stride; blockDim.y is 1
        for (uint32_t w = first_slice; w < n_slices;
             w += gridDim.y) {
            UnstackedSlice slice = slices[w];
            const FpExt *__restrict__ mat = q_mats[slice.commit_idx];

            auto lg_h = slice.log_height;
            auto col = slice.stacked_col_idx;
            auto row0 = (slice.stacked_row_idx >> lg_h) * eval_len;
            auto base = col * q_rows + row0;

            auto q_lo = mat[base + (y << 1)];
            auto q_hi = mat[base + (y << 1) + 1];
            auto q_slope = q_hi - q_lo;

#pragma unroll
            for (int xi = 1; xi <= S_DEG; ++xi) {
                Fp xf = Fp(xi);
                auto q_at = q_lo + q_slope * xf;
                auto eq_at_x = eq_lo + eq_slope * xf;
                auto rot_at = rot_lo + rot_slope * xf;

                partial[xi - 1] +=
                    (coeffs[2 * w] * eq_at_x + coeffs[2 * w + 1] * rot_at) *
                    q_at;
            }
        }
    }

#pragma unroll
    for (int d = 0; d < S_DEG; d++) {
        FpExt blk_total = reduce::block_reduce_sum(partial[d], warp_buf);
        if (threadIdx.x == 0) {
            reduce::atomic_add_fpext_to_u64(acc_out + d * 4, blk_total);
        }
        __syncthreads();
    }
}

// Degenerate rounds (no hypercube left): a single block whose threads stride over the window.
// Same reduction scheme as mle_round_window_kernel: per-degree block reduce through `warp_buf`
// (one FpExt per warp, barrier between degrees), then one atomic u64 add by thread 0.
__global__ void mle_round_tail_kernel(
    const FpExt *__restrict__ const
        *__restrict__ q_mats, // one matrix per commit, all of height q_rows
    const FpExt *__restrict__ eq_upper, // per-slice eq over the upper bits, at the window start
    FpExt eq_low,                       // eq factor shared by every slice
    FpExt rot_low,                      // k_rot factor shared by every slice
    const UnstackedSlice *__restrict__ slices, // window of unstacked columns
    const FpExt *__restrict__ coeffs,          // lambda powers for the same window
    uint64_t *__restrict__ acc_out, // S_DEG x 4 u64 limbs, summed atomically, finished on the CPU
    uint32_t q_rows,
    uint32_t n_slices,
    uint32_t shift // l_skip + round index
) {
    extern __shared__ char smem[];
    FpExt *warp_buf = (FpExt *)smem;

    FpExt partial[S_DEG];
#pragma unroll
    for (int d = 0; d < S_DEG; d++) {
        partial[d] = FpExt(0);
    }

    for (uint32_t w = threadIdx.x; w < n_slices; w += blockDim.x) {
        UnstackedSlice slice = slices[w];
        const FpExt *__restrict__ mat = q_mats[slice.commit_idx];

        auto col = slice.stacked_col_idx;
        auto row = slice.stacked_row_idx;
        auto row0 = (row >> shift) << 1;
        auto base = col * q_rows + row0;

        auto q_lo = mat[base];
        auto q_hi = mat[base + 1];
        auto q_slope = q_hi - q_lo;

        uint32_t bit = (row >> (shift - 1)) & 1;
        Fp bit_f = Fp(bit);

        auto eq_up = eq_upper[w];

#pragma unroll
        for (int xi = 1; xi <= S_DEG; ++xi) {
            Fp xf = Fp(xi);
            auto eq_up_x = eq_up * eq_bit(xf, bit_f);
            auto eq_at_x = eq_low * eq_up_x;
            auto rot_at = rot_low * eq_up_x;

            auto q_at = q_lo + q_slope * xf;

            partial[xi - 1] +=
                (coeffs[2 * w] * eq_at_x + coeffs[2 * w + 1] * rot_at) * q_at;
        }
    }

#pragma unroll
    for (int d = 0; d < S_DEG; d++) {
        FpExt blk_total = reduce::block_reduce_sum(partial[d], warp_buf);
        if (threadIdx.x == 0) {
            reduce::atomic_add_fpext_to_u64(acc_out + d * 4, blk_total);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------------------------

// x-values are accumulated per thread before the in-block reduction. ncu showed the
// kernel latency-bound on its per-block reduction tail (0.91 eligible warps/cycle at 2M
// blocks); striding x by XS in registers divides blocks, barriers and final-reduce work by XS.
// Geometry retile: the dominant shapes (lifted_height >= 2^20) get XS=128,
// shrinking grid.x (and the paired block_sums scratch + final_reduce work) by another 16x
// on top of the above; tiny traces keep XS=8 so their block/grid geometry doesn't degrade.
// Field addition is exact/associative, so the reduced output is byte-identical either way.
constexpr uint32_t R0_X_PER_THREAD_SMALL = 8;
constexpr uint32_t R0_X_PER_THREAD_LARGE = 128;

inline std::pair<dim3, dim3> round0_geometry(
    uint32_t rows,
    uint32_t cols,
    uint32_t log_skip
) {
    uint32_t z_len = 1u << log_skip;
    auto lifted = std::max(rows, z_len);
    uint32_t xs = lifted >= (1u << 20) ? R0_X_PER_THREAD_LARGE : R0_X_PER_THREAD_SMALL;
    // a block must hold the whole skip domain
    auto block_cap = std::max(z_len, 256u);
    // one thread per `xs` x-values, but never fewer threads than one skip domain
    auto threads_needed = std::max(lifted / xs, (uint32_t)z_len);
    auto [launch_grid, launch_block] = kernel_launch_params(threads_needed, block_cap);
    // `lifted` = z_len * cube_len with cube_len a power of two, so block.x divides it and every
    // thread of a block is live.
    launch_grid.y = cols;
    return {launch_grid, launch_block};
}

// Not a launcher: number of FpExt entries the round-0 `partials` scratch buffer needs
// (NUM_G values per z per block).
extern "C" uint32_t _kb_stacked_reduction_r0_required_temp_buffer_size(
    uint32_t rows,
    uint32_t cols,
    uint32_t log_skip
) {
    auto [launch_grid, launch_block] = round0_geometry(rows, cols, log_skip);
    return ((launch_grid.x * launch_grid.y) << log_skip) * NUM_G;
}

// Round 0: per-block G0/G1/G2 partials, then a final reduction that ADDS into `out` (so several
// buckets can accumulate into the same buffer). Identity coset only.
extern "C" int _kb_stacked_reduction_sumcheck_round0(
    const FpExt *eq_segments,
    const Fp *trace,
    const FpExt *col_coeffs,
    FpExt *partials,
    FpExt *out, // NUM_G * skip_domain entries; the result is added to what is there
    uint32_t rows,
    uint32_t cols,
    uint32_t log_skip,
    uint32_t cube_len
) {
    uint32_t z_len = 1u << log_skip;
    uint32_t z_step = std::max(z_len / rows, 1u);
    auto [launch_grid, launch_block] = round0_geometry(rows, cols, log_skip);

    // two padded rows (pitch block.x + 1): the kernel keeps only acc_eq / acc_rot in shared memory
    // and forms the three G outputs at the block-reduction tail.
    size_t tile_bytes = sizeof(FpExt) * (launch_block.x + 1) * 2;

    r0_partial_sums_kernel<<<launch_grid, launch_block, tile_bytes>>>(
        eq_segments, trace, col_coeffs, partials,
        rows, cols, log_skip,
        z_len - 1, cube_len, 31 - __builtin_clz(z_step)
    );

    int rc = CHECK_KERNEL();
    if (rc != 0)
        return rc;

    // fold the per-block partials and add them into `out`
    auto n_partials = launch_grid.x * launch_grid.y;
    auto n_outputs = NUM_G * z_len;
    auto [fin_grid, fin_block] = kernel_launch_params(n_partials);
    size_t fin_smem = div_ceil(fin_block.x, WARP_SIZE) * sizeof(FpExt);
    reduce::final_reduce_block_sums<true>
        <<<n_outputs, fin_block, fin_smem>>>(partials, out, n_partials);

    return CHECK_KERNEL();
}

// Folds each column's 2^l_skip-row chunks into one extension-field value by barycentric
// interpolation.
extern "C" int _kb_stacked_reduction_fold_ple(
    const Fp *trace,
    FpExt *folded,
    const Fp *omega_pows,
    const FpExt *inv_denoms,
    uint32_t rows_in,
    uint32_t cols,
    uint32_t log_skip
) {
    uint32_t dom = 1u << log_skip;
    uint32_t rows_out = std::max(rows_in, dom) / dom;

    // at least one skip domain per block, 256 threads when that is smaller
    uint32_t threads = std::max(256u, dom);
    uint32_t cells_per_block = threads / dom;

    // grid.x covers output rows, grid.y columns
    dim3 chunk_grid(div_ceil(rows_out, cells_per_block), cols);
    dim3 chunk_block(threads);

    // one FpExt per warp for the cross-warp step, only needed once a cell spans several warps
    uint32_t warps = threads / WARP_SIZE;
    size_t scratch_bytes = (dom > WARP_SIZE) ? warps * sizeof(FpExt) : 0;

    if (rows_out >= 256) {
        // Thread-per-cell fast path (see fold_ple_per_cell_kernel): bit-identical, ~1.8x.
        dim3 cell_grid(div_ceil(rows_out, 256u), cols);
        fold_ple_per_cell_kernel<<<cell_grid, 256>>>(
            trace, folded, omega_pows, inv_denoms, rows_in, rows_out, dom
        );
        return CHECK_KERNEL();
    }
    fold_ple_chunked_kernel<<<chunk_grid, chunk_block, scratch_bytes>>>(
        trace, folded, omega_pows, inv_denoms, rows_in, rows_out, dom
    );

    return CHECK_KERNEL();
}

extern "C" int _kb_initialize_k_rot_from_eq_segments(
    const FpExt *eq_segments,
    FpExt *k_rot_segments,
    FpExt uni_eq_coeff,
    FpExt uni_rot_coeff,
    uint32_t top_level
) {
    auto [launch_grid, launch_block] = kernel_launch_params(1 << top_level);
    launch_grid.y = top_level + 1;

    k_rot_from_eq_kernel<<<launch_grid, launch_block>>>(
        eq_segments, k_rot_segments, uni_eq_coeff, uni_rot_coeff
    );

    return CHECK_KERNEL();
}

extern "C" int _kb_stacked_reduction_sumcheck_mle_round(
    const FpExt *const *q_mats,
    const FpExt *eq_segments,
    const FpExt *k_rot_segments,
    const UnstackedSlice *slices,
    const FpExt *coeffs,
    uint64_t *acc_out, // S_DEG x D_EF u64 limbs, summed atomically, finished on the CPU
    uint32_t q_rows,
    uint32_t n_slices,
    uint32_t num_y,
    uint32_t sm_count
) {
    // 256 threads: more resident warps to cover memory latency
    auto [grid, block] = kernel_launch_params(num_y, 256);
    assert(sm_count);

    // Choose the window stride (grid.y): large enough to fill the GPU when num_y is small, small
    // enough that each thread still loops over several window entries (fewer atomics).
    constexpr uint32_t BLOCKS_PER_SM = 4;    // occupancy target
    constexpr uint32_t MIN_ITERS = 4;        // aim for at least this many window steps per thread
    constexpr uint32_t MAX_ITERS = 16;       // ... and at most this many

    uint32_t for_occupancy = div_ceil(sm_count * BLOCKS_PER_SM, grid.x);
    uint32_t from_max_iters = div_ceil(n_slices, MAX_ITERS);
    uint32_t from_min_iters = div_ceil(n_slices, MIN_ITERS);

    uint32_t floor_s = std::max(1u, std::max(for_occupancy, from_max_iters));
    uint32_t ceil_s = std::min(std::min(n_slices, MAX_GRID_DIM), from_min_iters);

    uint32_t window_stride = (floor_s <= ceil_s) ? floor_s : std::min(floor_s, std::min(n_slices, MAX_GRID_DIM));
    grid.y = window_stride;

    // the u64 atomic accumulators must not overflow
    assert((size_t)num_y * grid.y < (size_t)1u << 32);
    size_t warp_buf_bytes = div_ceil(block.x, WARP_SIZE) * sizeof(FpExt);

    mle_round_window_kernel<<<grid, block, warp_buf_bytes>>>(
        q_mats, eq_segments, k_rot_segments, slices, coeffs, acc_out, q_rows, n_slices, num_y
    );

    return CHECK_KERNEL();
}

extern "C" int _kb_stacked_reduction_sumcheck_mle_round_degenerate(
    const FpExt *const *q_mats,
    const FpExt *eq_upper,
    FpExt eq_low,
    FpExt rot_low,
    const UnstackedSlice *slices,
    const FpExt *coeffs,
    uint64_t *acc_out, // S_DEG x 4 u64 limbs, summed atomically, finished on the CPU
    uint32_t q_rows,
    uint32_t n_slices,
    uint32_t log_skip,
    uint32_t round_idx
) {
    auto shift = log_skip + round_idx;
    dim3 tail_block(std::min(n_slices, 256u));
    dim3 tail_grid(1);
    // at most 256 terms per atomic, far below the u64 overflow bound
    size_t warp_buf_bytes = div_ceil(tail_block.x, WARP_SIZE) * sizeof(FpExt);

    mle_round_tail_kernel<<<tail_grid, tail_block, warp_buf_bytes>>>(
        q_mats,
        eq_upper,
        eq_low,
        rot_low,
        slices,
        coeffs,
        acc_out,
        q_rows,
        n_slices,
        shift
    );

    return CHECK_KERNEL();
}

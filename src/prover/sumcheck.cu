// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>
#include <vector_types.h>

namespace sumcheck {

// Folds one cell for every fold kernel in this file: thread `cell` of matrix `m` combines rows 2*row
// and 2*row + 1 of column `col` into dst[row] = lo + r * (hi - lo). Matrices are column-major.
__device__ __forceinline__ void fold_mle(
    const FpExt *__restrict__ const *__restrict__ srcs,
    FpExt *__restrict__ const *__restrict__ dsts,
    const uint32_t *__restrict__ mat_widths, // column count per matrix
    const uint32_t n_mats,
    const uint8_t log_half,
    const FpExt &r,
    uint32_t cell,
    uint32_t m
) {
    if (m >= n_mats)
        return;
    uint32_t n_cols = mat_widths[m];
    uint32_t half = 1 << log_half;
    if (cell >= half * n_cols)
        return;
    uint32_t row = cell & (half - 1);
    uint32_t col = cell >> log_half;

    const FpExt *src = srcs[m];
    FpExt *dst = dsts[m];

    auto dst_col = col * half;
    auto src_col = dst_col << 1;

    auto lo_at = src_col + (row << 1);
    auto hi_at = src_col + (row << 1) + 1;
    auto dst_at = dst_col + row;

    FpExt lo = src[lo_at];
    FpExt hi = src[hi_at];

    dst[dst_at] = lo + r * (hi - lo);
}


// Binds the lowest MLE variable to r_val in every column of every matrix, halving the column length
// (2^(n+1) rows in, 2^n rows out; all matrices share log_output_height). grid.x covers the cells of
// one matrix and the y dimension picks the matrix.
__global__ void fold_mle_kernel(
    const FpExt *__restrict__ const *__restrict__ input_matrices, // Array of input matrix pointers
    FpExt *__restrict__ const *__restrict__ output_matrices,      // Array of output matrix pointers
    const uint32_t *__restrict__ widths,                          // Width of each matrix
    const uint32_t num_matrices,
    const uint8_t log_output_height,
    const FpExt r_val
) {
    uint32_t mat_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

    // The y coordinate (mat_idx) picks the matrix; x picks the cell inside it.
    fold_mle(
        input_matrices,
        output_matrices,
        widths,
        num_matrices,
        log_output_height,
        r_val,
        tidx,
        mat_idx
    );
}

// Flat-grid variant of fold_mle_kernel: one block per FOLD_MLE_FLAT_BLOCK output cells
// of the CONCATENATED batch. The legacy (grid.x, num_matrices) grid gives every matrix
// as many threads as the widest one, so width-skewed batches (e.g. widths [197,4,1]
// sharing one height on reth traces) launch mostly dead threads (~34% utilization).
// Each block finds its matrix by binary search over block_prefix (block_prefix[i] =
// first flat block of matrix i), then reuses the same per-cell fold. Scalar-height
// sibling of batch_fold_mle_flat_kernel below.
__global__ void fold_mle_flat_kernel(
    const FpExt *__restrict__ const *__restrict__ input_matrices, // Array of input matrix pointers
    FpExt *__restrict__ const *__restrict__ output_matrices,      // Array of output matrix pointers
    const uint32_t *widths,                                       // Width of each matrix
    const uint32_t num_matrices,
    const uint8_t log_output_height,
    const uint32_t *__restrict__ block_prefix, // [num_matrices + 1], ascending, [0] = 0
    const FpExt r_val
) {
    uint32_t b = blockIdx.x;
    // mat_idx = max i such that block_prefix[i] <= b (tiny L2/L1-resident table)
    uint32_t lo = 0;
    uint32_t hi = num_matrices;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) >> 1;
        if (block_prefix[mid] <= b) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint32_t mat_idx = lo;
    uint32_t tidx = (b - block_prefix[mat_idx]) * blockDim.x + threadIdx.x;

    sumcheck::fold_mle(
        input_matrices,
        output_matrices,
        widths,
        num_matrices,
        log_output_height,
        r_val,
        tidx,
        mat_idx
    );
}

// Same fold as fold_mle_kernel, except that each matrix carries its own output height.
__global__ void batch_fold_mle_kernel(
    const FpExt *__restrict__ const *__restrict__ input_matrices, // Array of input matrix pointers
    FpExt *__restrict__ const *__restrict__ output_matrices,      // Array of output matrix pointers
    const uint32_t *__restrict__ widths,                          // Width of each matrix
    const uint32_t num_matrices,
    const uint8_t *__restrict__ log_output_heights,
    const FpExt r_val
) {
    uint32_t mat_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mat_idx >= num_matrices) {
        return;
    }
    uint8_t log_output_height = log_output_heights[mat_idx];

    fold_mle(
        input_matrices,
        output_matrices,
        widths,
        num_matrices,
        log_output_height,
        r_val,
        tidx,
        mat_idx
    );
}

// Flat-grid variant of batch_fold_mle_kernel: one block per BATCH_FOLD_FLAT_BLOCK output
// cells of the CONCATENATED batch. The legacy grid gives every matrix as many threads as
// the largest one, so height-skewed batches launch mostly dead threads (~9% utilization
// on reth traces). Each block finds its matrix by binary search over block_prefix
// (block_prefix[i] = first flat block of matrix i), then reuses the same per-cell fold.
__global__ void batch_fold_mle_flat_kernel(
    const FpExt *__restrict__ const *__restrict__ input_matrices, // Array of input matrix pointers
    FpExt *__restrict__ const *__restrict__ output_matrices,      // Array of output matrix pointers
    const uint32_t *widths,                                       // Width of each matrix
    const uint32_t num_matrices,
    const uint8_t *log_output_heights,
    const uint32_t *__restrict__ block_prefix, // [num_matrices + 1], ascending, [0] = 0
    const FpExt r_val
) {
    uint32_t b = blockIdx.x;
    // mat_idx = max i such that block_prefix[i] <= b (tiny L2/L1-resident table)
    uint32_t lo = 0;
    uint32_t hi = num_matrices;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) >> 1;
        if (block_prefix[mid] <= b) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint32_t mat_idx = lo;
    uint32_t tidx = (b - block_prefix[mat_idx]) * blockDim.x + threadIdx.x;

    sumcheck::fold_mle(
        input_matrices,
        output_matrices,
        widths,
        num_matrices,
        log_output_heights[mat_idx],
        r_val,
        tidx,
        mat_idx
    );
}

// Folds every layer of the input segment tree into the output segment tree in one flat pass.
// The input is a segment tree of size 2 * 2^{output_max_n + 1}, the output one of size
// 2 * 2^{output_max_n}; input[1] and output[0] are unused. Layer `n`'s entry `x` lives at slot
// `t = 2^n + x`, and its input pair at `2^{n+1} + 2x = 2t`. So across every layer at once the fold
// is simply `output[t] = fold(input[2t], input[2t+1])` for `t in [1, num_slots]` — no per-layer
// grid dimension and no early-exit, unlike the previous launch which sized `grid.x` for the
// largest layer and let smaller layers' surplus threads return.
__global__ void triangular_fold_mle_kernel(
    FpExt *output, const FpExt *input, FpExt r, uint32_t num_slots
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_slots)
        return;
    uint32_t t = i + 1;
    FpExt t0 = input[t << 1];
    FpExt t1 = input[(t << 1) + 1];
    output[t] = t0 + (t1 - t0) * r;
}

// ============================================================================
// LAUNCHERS
// ============================================================================

// Both arguments must be non-zero. This helper returns a `dim3` pair and so has no error channel
// of its own, and a zero block extent reaches a host-side division by zero -- SIGFPE, not an error
// return -- so the launchers below reject both degenerate arguments before calling it.
std::pair<dim3, dim3> fold_mle_launch_params(uint32_t max_output_cells, uint32_t num_matrices) {
    if (max_output_cells <= WARP_SIZE) {
        return kernel_launch_2d_params(max_output_cells, num_matrices);
    } else {
        auto [grid, block] = kernel_launch_params(max_output_cells);
        grid.y = num_matrices;
        return {grid, block};
    }
}

// Flat-path launch constants for _kb_fold_mle. 512 output cells per block beat 256 in both bench
// sweeps on this shape class. MAX_MATRICES bounds the host-side `h_widths`/`h_block_prefix` stack
// arrays and the device scratch, so a batch above it takes the legacy grid instead; 1024 is far
// above production (real batches are <= ~21) and is chosen so the stack arrays stay small.
// `_kb_batch_fold_mle` below carries its own pair of constants with the same MAX_MATRICES for the
// same reason — they are separate only because each entry point sizes its own scratch.
constexpr uint32_t FOLD_MLE_FLAT_BLOCK = 512;
// ── per-thread block-prefix scratch, recycled across threads ─────────────────
// The fold paths below stage a small index table per launch on the per-thread default stream, so
// the buffer is per thread — but must not have a thread's LIFETIME, because the prover starts a
// fresh set of threads per block and a raw `thread_local` has no destructor. Handed on
// rather than freed, for the reason `rs_ntt_fastpath.cu` states at its own idle list.
namespace tl_prefix {

// Never destroyed, for the same reason as the free list below: a thread can end after static
// destruction has started, and the hand-back must not touch a destroyed object.
static std::mutex& mtx() {
    static std::mutex* m = new std::mutex();
    return *m;
}

// Matches `RS2_MAX_DEVICES` in the sibling fast path rather than cuda-mle's Rust-side 16: these
// are C-side tables and the two build units size them the same way.
constexpr int MAX_DEVICES = 64;

// Both call sites cap their matrix count at the same 1024; one buffer size serves both, and the
// `static_assert`s beside those constants keep that true if either ever moves.
constexpr uint32_t ENTRIES = 1024 + 1;

static std::vector<uint32_t*>* idle_for(int dev) {
    static std::vector<uint32_t*>* idle = new std::vector<uint32_t*>[MAX_DEVICES];
    return (dev < 0 || dev >= MAX_DEVICES) ? nullptr : &idle[dev];
}

static std::atomic<bool> shutting_down{false};

// One per call site, named so a bare 0/1 does not have to be decoded at the use.
enum Site { FOLD = 0, BATCH_FOLD = 1, SITES = 2 };

// Each site keeps one; both go back together when the thread ends.
static thread_local uint32_t* slot[SITES] = {nullptr, nullptr};
// Per slot, not per thread: the buffers are device memory, and filing one under the wrong card
// would hand it to a thread whose context cannot use it.
static thread_local int slot_device[SITES] = {-1, -1};

struct Guard {
    bool armed = false;
    ~Guard() {
        if (!armed || shutting_down.load(std::memory_order_relaxed))
            return;
        // This thread's own staging copies are stream-ordered on the per-thread stream; drain it
        // before the buffers can be handed to someone else.
        cudaStreamSynchronize(cudaStreamPerThread);
        std::lock_guard<std::mutex> lock(mtx());
        for (int i = 0; i < SITES; i++) {
            std::vector<uint32_t*>* idle = idle_for(slot_device[i]);
            if (slot[i] != nullptr && idle != nullptr)
                idle->push_back(slot[i]);
            slot[i] = nullptr;
            slot_device[i] = -1;
        }
    }
};
static thread_local Guard guard;

/// This thread's buffer for call site `which`, taken from the card's idle list when one is
/// waiting. Reports the driver's own code in `*err` rather than flattening every failure to one
/// stand-in, for the reason the sibling fast path gives where it refuses to flatten to
/// `cudaErrorUnknown`: the code is the diagnosis.
static uint32_t* get(Site which, cudaError_t* err) {
    *err = cudaSuccess;
    if (slot[which] != nullptr)
        return slot[which];
    guard.armed = true;
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit([] { shutting_down.store(true, std::memory_order_relaxed); });
    });
    int dev = 0;
    *err = cudaGetDevice(&dev);
    if (*err != cudaSuccess)
        return nullptr;
    slot_device[which] = dev;
    {
        std::lock_guard<std::mutex> lock(mtx());
        std::vector<uint32_t*>* idle = idle_for(dev);
        if (idle != nullptr && !idle->empty()) {
            slot[which] = idle->back();
            idle->pop_back();
            return slot[which];
        }
    }
    uint32_t* p = nullptr;
    *err = cudaMalloc(&p, ENTRIES * sizeof(uint32_t));
    if (*err != cudaSuccess)
        return nullptr;
    slot[which] = p;
    return p;
}

} // namespace tl_prefix

constexpr uint32_t FOLD_MLE_FLAT_MAX_MATRICES = 1024;
static_assert(FOLD_MLE_FLAT_MAX_MATRICES + 1 == tl_prefix::ENTRIES,
              "the recycled prefix buffer must hold this path's whole table");

extern "C" int _kb_fold_mle(
    const FpExt *const *input_matrices,
    FpExt *const *output_matrices,
    const uint32_t *widths,
    const uint16_t num_matrices,
    const uint32_t output_height,
    const uint32_t max_output_cells,
    const FpExt r_val
) {
    // Defence in depth, not a bug fix -- no live caller can reach these branches -- but each
    // failure would be silent rather than diagnosable: `__builtin_clz(0)` is undefined, a
    // non-power-of-two `output_height` is truncated by the `clz` to the next lower power of two
    // and folds the wrong rows, and a zero count crashes `fold_mle_launch_params`.
    if (output_height == 0 || (output_height & (output_height - 1)) != 0 || num_matrices == 0 ||
        max_output_cells == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    uint8_t log_output_height = static_cast<uint8_t>(31 - __builtin_clz(output_height));

    // Fast path: flat 1-D grid over the total output cells (see fold_mle_flat_kernel).
    // Same recipe as _kb_batch_fold_mle below: block_prefix is built host-side from a tiny
    // D2H readback of widths — the fold challenge r_val is derived from the previous
    // round's results, so the stream has already been drained by that readback when we
    // get here and this one costs only PCIe latency.
    if (max_output_cells > WARP_SIZE && num_matrices > 1 &&
        num_matrices <= FOLD_MLE_FLAT_MAX_MATRICES) {
        uint32_t h_widths[FOLD_MLE_FLAT_MAX_MATRICES];
        uint32_t h_block_prefix[FOLD_MLE_FLAT_MAX_MATRICES + 1];

        cudaError_t err =
            cudaMemcpy(h_widths, widths, num_matrices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return err;
        }

        uint64_t num_blocks = 0;
        h_block_prefix[0] = 0;
        for (uint32_t i = 0; i < num_matrices; i++) {
            uint64_t cells = static_cast<uint64_t>(h_widths[i]) << log_output_height;
            num_blocks += (cells + FOLD_MLE_FLAT_BLOCK - 1) / FOLD_MLE_FLAT_BLOCK;
            h_block_prefix[i + 1] = static_cast<uint32_t>(num_blocks);
        }

        if (num_blocks > 0 && num_blocks < (uint64_t(1) << 31)) {
            // Per-thread scratch: kernels and copies here run on the per-thread default
            // stream, so a thread-local buffer cannot race across prover threads.
            uint32_t *d_block_prefix = tl_prefix::get(tl_prefix::FOLD, &err);
            if (d_block_prefix == nullptr) {
                return err;
            }
            // Async H2D from pageable memory returns after staging, so the stack array
            // may go out of scope; the copy is stream-ordered before the launch below.
            err = cudaMemcpyAsync(
                d_block_prefix,
                h_block_prefix,
                (num_matrices + 1) * sizeof(uint32_t),
                cudaMemcpyHostToDevice
            );
            if (err != cudaSuccess) {
                return err;
            }
            fold_mle_flat_kernel<<<static_cast<uint32_t>(num_blocks), FOLD_MLE_FLAT_BLOCK>>>(
                input_matrices,
                output_matrices,
                widths,
                num_matrices,
                log_output_height,
                d_block_prefix,
                r_val
            );
            return CHECK_KERNEL();
        }
        // Oversize batch: fall through to the legacy grid.
    }
    auto [grid, block] = fold_mle_launch_params(max_output_cells, num_matrices);
    fold_mle_kernel<<<grid, block>>>(
        input_matrices, output_matrices, widths, num_matrices, log_output_height, r_val
    );
    return CHECK_KERNEL();
}

// Flat-path launch constants for _kb_batch_fold_mle. Same roles as the `_kb_fold_mle` pair above (see
// the note there); the block width differs because this path folds per-matrix heights.
constexpr uint32_t BATCH_FOLD_FLAT_BLOCK = 256;
constexpr uint32_t BATCH_FOLD_FLAT_MAX_MATRICES = 1024;
static_assert(BATCH_FOLD_FLAT_MAX_MATRICES + 1 == tl_prefix::ENTRIES,
              "the recycled prefix buffer must hold this path's whole table");

extern "C" int _kb_batch_fold_mle(
    const FpExt *const *input_matrices,
    FpExt *const *output_matrices,
    const uint32_t *widths,
    const uint16_t num_matrices,
    const uint8_t *log_output_heights,
    const uint32_t max_output_cells, // max(height * width)
    const FpExt r_val
) {
    // As on `_kb_fold_mle`, minus the height checks: this entry point is handed the logs directly
    // and never runs a `clz`.
    if (num_matrices == 0 || max_output_cells == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // Fast path: flat 1-D grid over the total output cells (see
    // batch_fold_mle_flat_kernel). block_prefix is built host-side from a tiny D2H
    // readback of widths/log_output_heights: the fold challenge r_val is derived from
    // the previous round's results, so the stream has already been drained by that
    // readback when we get here and this one costs only PCIe latency.
    if (max_output_cells > WARP_SIZE && num_matrices > 1 &&
        num_matrices <= BATCH_FOLD_FLAT_MAX_MATRICES) {
        uint32_t h_widths[BATCH_FOLD_FLAT_MAX_MATRICES];
        uint8_t h_log_output_heights[BATCH_FOLD_FLAT_MAX_MATRICES];
        uint32_t h_block_prefix[BATCH_FOLD_FLAT_MAX_MATRICES + 1];

        cudaError_t err =
            cudaMemcpy(h_widths, widths, num_matrices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaMemcpy(
            h_log_output_heights,
            log_output_heights,
            num_matrices * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess) {
            return err;
        }

        uint64_t num_blocks = 0;
        h_block_prefix[0] = 0;
        for (uint32_t i = 0; i < num_matrices; i++) {
            uint64_t cells = static_cast<uint64_t>(h_widths[i]) << h_log_output_heights[i];
            num_blocks += (cells + BATCH_FOLD_FLAT_BLOCK - 1) / BATCH_FOLD_FLAT_BLOCK;
            h_block_prefix[i + 1] = static_cast<uint32_t>(num_blocks);
        }

        if (num_blocks > 0 && num_blocks < (uint64_t(1) << 31)) {
            // Per-thread scratch: kernels and copies here run on the per-thread default
            // stream, so a thread-local buffer cannot race across prover threads.
            uint32_t *d_block_prefix = tl_prefix::get(tl_prefix::BATCH_FOLD, &err);
            if (d_block_prefix == nullptr) {
                return err;
            }
            // Async H2D from pageable memory returns after staging, so the stack array
            // may go out of scope; the copy is stream-ordered before the launch below.
            err = cudaMemcpyAsync(
                d_block_prefix,
                h_block_prefix,
                (num_matrices + 1) * sizeof(uint32_t),
                cudaMemcpyHostToDevice
            );
            if (err != cudaSuccess) {
                return err;
            }
            batch_fold_mle_flat_kernel<<<static_cast<uint32_t>(num_blocks), BATCH_FOLD_FLAT_BLOCK>>>(
                input_matrices,
                output_matrices,
                widths,
                num_matrices,
                log_output_heights,
                d_block_prefix,
                r_val
            );
            return CHECK_KERNEL();
        }
        // Oversize batch: fall through to the legacy grid.
    }
    auto [grid, block] = fold_mle_launch_params(max_output_cells, num_matrices);
    batch_fold_mle_kernel<<<grid, block>>>(
        input_matrices, output_matrices, widths, num_matrices, log_output_heights, r_val
    );
    return CHECK_KERNEL();
}

extern "C" int _kb_triangular_fold_mle(
    FpExt *output,
    const FpExt *input,
    FpExt r,
    uint32_t output_max_n
) {
    // As on `_kb_fold_mle`. `output_max_n == 0` is legitimate and reachable (the last round folds a
    // single-cell layer); the bound keeps `2 << output_max_n` from overflowing the u32 slot count.
    if (output_max_n >= 31) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // Slots span every layer's segment: layer n contributes 2^n entries, so the whole tree minus
    // the unused slot 0 is (2^{output_max_n+1} - 1) = (2 << output_max_n) - 1 slots.
    uint32_t num_slots = (2u << output_max_n) - 1;
    auto [grid, block] = kernel_launch_params(num_slots);

    triangular_fold_mle_kernel<<<grid, block>>>(output, input, r, num_slots);

    return CHECK_KERNEL();
}

} // namespace sumcheck

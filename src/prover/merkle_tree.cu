// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * KoalaBear Poseidon2 Merkle tree commitment kernels — the production commit path.
 *
 * Field kb31_t, permutation poseidon2_kb::poseidon2_mix_kb.
 *
 * LINK UNIT: compiled into the `mle-kernels` library, by name, not into `cuda_kb_all` (it
 * declares its own `using Fp = kb31_t;`). Its exports are named `_kb_*` like every other
 * launcher the Rust side binds.
 *
 * LEAF LAYOUT: one leaf digest per `query_stride`, collapsing the `2^log_rows_per_query` rows
 * `{stride_idx + t*query_stride}`. `height == query_stride << log_rows_per_query` and is a power of
 * two — the caller pads. Absorb matches p3's PaddingFreeSponge<_, CELLS, CELLS_RATE, CELLS_OUT>;
 * the pair compress matches TruncatedPermutation<_, 2, CELLS_OUT, CELLS>.
 *
 * OCCUPANCY: launch widths are measured and set statically, never env-selected. An earlier
 * getenv-switched `__launch_bounds__` clone (MERKLE_LB) was deleted as unmeasured
 * and, on the shape a test pins, unreachable.
 */

#include "ff/koala_bear.hpp"
#include "common/launcher.cuh"
#include "hash/poseidon2/poseidon2.cuh"
#include <cstddef>
#include <cstdint>

using Fp = kb31_t;

// CELLS, CELLS_RATE and CELLS_OUT are #defines from hash/poseidon2/poseidon2.cuh, so they need no `using`.

struct kb_digest_t {
    Fp cells[CELLS_OUT];  // [KoalaBear; 8]
};

// Launch width for the thread-per-query serial tree. Same invariant as ABSORB_BLOCK below: the
// `__launch_bounds__` and the launcher both read this, so the two cannot drift. Bounds the block only
// — the absorb kernels' `(ABSORB_BLOCK, 4)` min-blocks form was settled by measurement, and no such
// measurement exists for this kernel; asserting a min-blocks target it cannot meet would force spills.
constexpr uint32_t FINALIZE_BLOCK = 512;

// Launch width for the streaming absorb kernel. The `__launch_bounds__` and the launcher must both
// use it — bounding at a width the kernel is never launched at only costs registers.
#define ABSORB_BLOCK 256

// ── Row hash + merkle combine kernel (base field) ──────────────────────────

__global__ void kb_poseidon2_compressing_row_hashes_kernel(
    kb_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[];
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);

    if (stride_idx < query_stride) {
        // Whole rate blocks unrolled with constant cells[] indices, then the tail peeled. The
        // obvious `cells[used++]` counter loop is simpler but retires ~6% more instructions,
        // because its eight loads issue one per iteration behind a dependent compare.
        int col = 0;
        for (; col + CELLS_RATE <= (int)width; col += CELLS_RATE) {
#pragma unroll
            for (int i = 0; i < CELLS_RATE; i++) cells[i] = matrix[(col + i) * height + row];
            poseidon2_kb::poseidon2_mix_kb(cells);
        }
        int used = 0;
#pragma unroll
        for (int i = 0; i < CELLS_RATE; i++)
            if (col + i < (int)width) { cells[i] = matrix[(col + i) * height + row]; used = i + 1; }
        if (used != 0) poseidon2_kb::poseidon2_mix_kb(cells);
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1u << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                shared[i * shared_stride + shared_offset] = cells[i];
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            poseidon2_kb::poseidon2_mix_kb(cells);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++)
            out[stride_idx].cells[i] = cells[i];
    }
}

// ── Streaming (column-block) leaf hash ─────────────────────────────────────
//
// Two passes so the RS codeword never has to be fully resident: `absorb_columns` folds one column
// block into a per-row sponge state, `finalize_leaf_digests` runs the tree reduction over the
// finished per-row digests.
//
// Bit-identical to the monolithic kernel ONLY IF every block but the last is a multiple of
// CELLS_RATE wide. That is what lands each block boundary right after a permute, so `used` is 0 at
// the start of every launch and never has to be persisted.
//
// state layout: `state[i * height + row]`, i in [0, CELLS) — the 16 sponge lanes column-major, so
// consecutive threads (consecutive `row`) read and write coalesced.

__global__ __launch_bounds__(ABSORB_BLOCK, 4) void kb_absorb_columns_kernel(
    Fp *state,
    const Fp *block,
    size_t width_block,
    size_t height,
    int is_first,
    int is_last
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    Fp cells[CELLS];
    if (is_first) {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);
    } else {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = state[(size_t)i * height + row];
    }

    // Same unrolled rate-block shape as the row-hash kernel above; `used` starts at 0 because
    // every non-last block is rate-aligned.
    size_t col = 0;
    for (; col + CELLS_RATE <= width_block; col += CELLS_RATE) {
#pragma unroll
        for (int i = 0; i < CELLS_RATE; i++) cells[i] = block[(col + (size_t)i) * height + row];
        poseidon2_kb::poseidon2_mix_kb(cells);
    }
    int used = 0;
#pragma unroll
    for (int i = 0; i < CELLS_RATE; i++)
        if (col + (size_t)i < width_block) {
            cells[i] = block[(col + (size_t)i) * height + row];
            used = i + 1;
        }
    // Trailing partial permute only on the final block (mirrors the monolithic `if (used != 0)`).
    if (is_last && used != 0) poseidon2_kb::poseidon2_mix_kb(cells);

#pragma unroll
    for (int i = 0; i < CELLS; i++) state[(size_t)i * height + row] = cells[i];
}

// ── Shape-specialized absorb fast paths ────────────────────────────────────
// Production shape capture (reth-171, 16 commits / 110 absorb calls): ~90% of absorb work
// runs rate-aligned block widths (wb in {16,24,32,40}, wb % CELLS_RATE == 0); the hot shape
// is (wb=24, is_first=0) at ~54% of work. For rate-aligned widths is_last needs NO dispatch
// axis: the trailing partial permute is dead code either way, so specializing on
// (width, is_first) stays bit-exact for every call.
//   * wb in {16, 24}: full-unroll template — the per-column `used` counter/branch is
//     eliminated, is_first dead-coded, and the 2-3 mix calls schedule statically
//     (-4.6% at the hot shape vs the generic kernel; wants 64 regs = 4 blk/SM at (256,4)).
//   * other rate-aligned wb: half-specialized — is_first dead-coded but the group loop
//     stays a RUNTIME loop (one inlined mix): the full unroll's code size grows with wb/8
//     and REGRESSES at wb >= 32 (icache), this flavor keeps -2.7%/-2.8% at 32/40.
//   * non-rate-aligned tails (~10% of work, small widths, is_last=1): generic kernel above.
template <int WB, bool IS_FIRST_T>
__global__ __launch_bounds__(ABSORB_BLOCK, 4) void kb_absorb_columns_kernel_tpl(
    Fp *state,
    const Fp *block,
    size_t height
) {
    static_assert(WB % CELLS_RATE == 0, "full-unroll fast path requires a rate-aligned width");
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    Fp cells[CELLS];
    if (IS_FIRST_T) {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);
    } else {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = state[(size_t)i * height + row];
    }

    constexpr int GROUPS = WB / CELLS_RATE;
#pragma unroll
    for (int g = 0; g < GROUPS; g++) {
#pragma unroll
        for (int j = 0; j < CELLS_RATE; j++)
            cells[j] = block[(size_t)(g * CELLS_RATE + j) * height + row];
        poseidon2_kb::poseidon2_mix_kb(cells);
    }

#pragma unroll
    for (int i = 0; i < CELLS; i++) state[(size_t)i * height + row] = cells[i];
}

template <bool IS_FIRST_T>
__global__ __launch_bounds__(ABSORB_BLOCK, 4) void kb_absorb_columns_kernel_grp(
    Fp *state,
    const Fp *block,
    size_t width_block,   // caller guarantees width_block % CELLS_RATE == 0
    size_t height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    Fp cells[CELLS];
    if (IS_FIRST_T) {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);
    } else {
#pragma unroll
        for (int i = 0; i < CELLS; i++) cells[i] = state[(size_t)i * height + row];
    }

    size_t groups = width_block / CELLS_RATE;
    for (size_t g = 0; g < groups; g++) {
#pragma unroll
        for (int j = 0; j < CELLS_RATE; j++)
            cells[j] = block[(g * CELLS_RATE + (size_t)j) * height + row];
        poseidon2_kb::poseidon2_mix_kb(cells);
    }

#pragma unroll
    for (int i = 0; i < CELLS; i++) state[(size_t)i * height + row] = cells[i];
}

__global__ void kb_finalize_leaf_digests_kernel(
    kb_digest_t *out,
    const Fp *state,
    size_t height,
    size_t query_stride,
    size_t log_rows_per_query
) {
    extern __shared__ char smem[];
    Fp *shared = reinterpret_cast<Fp *>(smem);
    const uint32_t shared_stride = blockDim.x * (blockDim.y >> 1);

    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t leaf_idx = threadIdx.y;
    const uint32_t row = leaf_idx * query_stride + stride_idx;

    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS; i++) cells[i] = Fp(0);

    // Load this row's finished sponge digest (first CELLS_OUT lanes); the rest are overwritten by
    // the tree reduction below, so they need no init beyond the zero above.
    if (stride_idx < query_stride) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) cells[i] = state[(size_t)i * height + row];
    }

    for (int layer = 0; layer < (int)log_rows_per_query; ++layer) {
        uint32_t mask = (1 << (layer + 1)) - 1;
        auto shared_offset = ((leaf_idx >> (layer + 1)) << layer) * blockDim.x + threadIdx.x;
        if ((leaf_idx & mask) == (1u << layer)) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                shared[i * shared_stride + shared_offset] = cells[i];
        }
        __syncthreads();
        if ((leaf_idx & mask) == 0) {
#pragma unroll
            for (int i = 0; i < CELLS_OUT; i++)
                cells[CELLS_OUT + i] = shared[i * shared_stride + shared_offset];
            poseidon2_kb::poseidon2_mix_kb(cells);
        }
        __syncthreads();
    }

    if (leaf_idx == 0 && stride_idx < query_stride) {
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++)
            out[stride_idx].cells[i] = cells[i];
    }
}


// ── Finalize leaf digests: thread-per-query serial recursive tree ──────────
// One thread computes its query column's ENTIRE 2^lrq-leaf merkle tree depth-first
// in registers (template recursion => no runtime-indexed level stack, 0 spills).
// Byte-equal pairing to kb_finalize_leaf_digests_kernel: lower leaf index = left =
// cells[0..7], higher = right = cells[8..15]; node = mix output cells[0..7].
// Removes the warp-per-leaf kernel's idle warps / __syncthreads / smem tree
// (1.348x standalone at lrq=4). FINALIZE_BLOCK is bounded on the kernel; 512 giving 2 blocks/SM
// at 64 regs is a past measurement, not an invariant enforced here.
template <int LEVEL>
__device__ __forceinline__ void kb_finalize_reduce_subtree(
    Fp *out, const Fp *state, size_t height, size_t query_stride,
    uint32_t stride_idx, int leaf_base
) {
    if constexpr (LEVEL == 0) {
        const size_t row = (size_t)leaf_base * query_stride + stride_idx;
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) out[i] = state[(size_t)i * height + row];
    } else {
        Fp left[CELLS_OUT], right[CELLS_OUT];
        kb_finalize_reduce_subtree<LEVEL - 1>(left, state, height, query_stride, stride_idx, leaf_base);
        kb_finalize_reduce_subtree<LEVEL - 1>(right, state, height, query_stride, stride_idx,
                                              leaf_base + (1 << (LEVEL - 1)));
        Fp cells[CELLS];
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) { cells[i] = left[i]; cells[CELLS_OUT + i] = right[i]; }
        poseidon2_kb::poseidon2_mix_kb(cells);
#pragma unroll
        for (int i = 0; i < CELLS_OUT; i++) out[i] = cells[i];
    }
}

template <int LRQ_T>
__global__ __launch_bounds__(FINALIZE_BLOCK) void kb_finalize_leaf_digests_serial_kernel(
    kb_digest_t *out,
    const Fp *state,
    size_t height,
    size_t query_stride
) {
    const uint32_t stride_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (stride_idx >= query_stride) return;
    Fp root[CELLS_OUT];
    kb_finalize_reduce_subtree<LRQ_T>(root, state, height, query_stride, stride_idx, 0);
#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) out[stride_idx].cells[i] = root[i];
}

// ── Adjacent compress layer ────────────────────────────────────────────────

__global__ void kb_poseidon2_strided_compress_layer_kernel(
    kb_digest_t *output,
    const kb_digest_t *prev_layer,
    size_t output_size,
    size_t stride
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    // Non-overlapping pair indexing: pair (2*idx, 2*idx+1) for stride=1
    // Generalized: x = idx / stride, y = idx % stride → (2*x*stride+y, (2*x+1)*stride+y)
    uint32_t x = idx / stride;
    uint32_t y = idx % stride;
    Fp cells[CELLS];
#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) cells[i] = prev_layer[2 * x * stride + y].cells[i];
#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) cells[CELLS_OUT + i] = prev_layer[(2 * x + 1) * stride + y].cells[i];

    poseidon2_kb::poseidon2_mix_kb(cells);

#pragma unroll
    for (int i = 0; i < CELLS_OUT; i++) output[idx].cells[i] = cells[i];
}

/*
Digest gather for batched Merkle openings. The output is laid out as
query[0][0,...layers-1]
query[1][0,...layers-1]
...
query[k][0,...layers-1]
*/
// Does NO field arithmetic — it moves CELLS_OUT 32-bit words per digest — so it is typed on
// uint32_t. Words stay in Montgomery form; the host reinterprets them, it does not re-encode them.
__global__ void kb_query_digest_layers_kernel(
    uint32_t *d_digest_matrix,    // CELLS_OUT=8 words per digest
    const uint64_t *d_layers_ptr, // array of digest layers
    uint64_t *d_indices,          // indices to query, size = num_query * num_layer
    uint64_t num_query,           // e.g. 100
    uint64_t num_layer
) // e.g. 23
{
    const uint32_t ELEM_PER_DIGEST = CELLS_OUT; // 8 words
    uint64_t gidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t layer_idx = gidx / ELEM_PER_DIGEST;
    uint64_t elem_offset = gidx % ELEM_PER_DIGEST; // thread group: [0,..7]
    uint64_t query_idx = blockIdx.y;               // [0, num_query -1]
    if (layer_idx >= num_layer) {                  // [0, layers - 1]
        return;
    }

    const uint32_t *d_layer = (const uint32_t *)d_layers_ptr[layer_idx];
    uint64_t digest_offset = d_indices[query_idx * num_layer + layer_idx];
    uint32_t digest_elem = d_layer[digest_offset * ELEM_PER_DIGEST + elem_offset];
    // Each of the ELEM_PER_DIGEST threads of this (query, layer) pair copies one digest word.

    uint64_t output_query_offset = query_idx * num_layer * ELEM_PER_DIGEST;
    uint64_t output_layer_offset = layer_idx * ELEM_PER_DIGEST + elem_offset;
    d_digest_matrix[output_query_offset + output_layer_offset] = digest_elem;
}

// ── Extern C launchers ─────────────────────────────────────────────────────

static inline size_t div_ceil_kb(size_t a, size_t b) { return (a + b - 1) / b; }

extern "C" int _kb_poseidon2_compressing_row_hashes(
    kb_digest_t *out,
    const Fp *matrix,
    size_t width,
    size_t query_stride,
    size_t log_rows_per_query
) {
    if (log_rows_per_query > 10) return cudaErrorInvalidValue;
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil_kb(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    kb_poseidon2_compressing_row_hashes_kernel<<<grid, block, shmem_bytes>>>(
        out, matrix, width, height, query_stride, log_rows_per_query);
    return CHECK_KERNEL();
}

extern "C" int _kb_poseidon2_adjacent_compress_layer(
    kb_digest_t *output,
    const kb_digest_t *prev_layer,
    size_t output_size
) {
    // block=64: block=1024 caps occupancy at 1024/1536 threads/SM on the big layers and is
    // pure poison on the small-layer ladder (grid < #SM => one serial 32-warp block per SM).
    // 1.33x on the 2^19 layer, ~1.86x over the whole per-tree ladder.
    auto [grid, block] = kernel_launch_params(output_size, 64);
    kb_poseidon2_strided_compress_layer_kernel<<<grid, block>>>(
        output, prev_layer, output_size, 1);
    return CHECK_KERNEL();
}

extern "C" int _kb_query_digest_layers(
    uint32_t *d_digest_matrix,
    const uint64_t *d_layers_ptr,
    uint64_t *d_indices,
    uint64_t num_query,
    uint64_t num_layer
) {
    if (num_query == 0 || num_layer == 0) {
        return cudaSuccess;
    }
    if (num_query > UINT16_MAX) {
        return cudaErrorInvalidValue;
    }
    const size_t QUERY_DIGEST_THREADS = 128;
    const size_t DIGEST_WIDTH = 8;

    auto block = QUERY_DIGEST_THREADS;
    dim3 grid = dim3(div_ceil_kb(num_layer * DIGEST_WIDTH, block), num_query);
    kb_query_digest_layers_kernel<<<grid, block>>>(
        d_digest_matrix, d_layers_ptr, d_indices, num_query, num_layer
    );
    return CHECK_KERNEL();
}

// ── Streaming leaf-hash launchers ──────────────────────────────────────────

extern "C" int _kb_poseidon2_absorb_columns(
    Fp *state,
    const Fp *block,
    size_t width_block,
    size_t height,
    int is_first,
    int is_last
) {
    // Launch geometry from one constant: ABSORB_BLOCK is the measured value
    // (256 with min-blocks 4) and feeding it through kernel_launch_params keeps the launcher and
    // the kernel's __launch_bounds__ tied to one constant.
    auto [grid, blk] = kernel_launch_params(height, ABSORB_BLOCK);
    // Tiered fast-path dispatch (see kb_absorb_columns_kernel_tpl/_grp above): full-unroll
    // template for wb in {16,24}, half-specialized group kernel for other rate-aligned
    // widths, generic kernel for the non-rate-aligned tails.
    if (width_block > 0 && width_block % CELLS_RATE == 0) {
        switch (width_block) {
            case 16:
                if (is_first) kb_absorb_columns_kernel_tpl<16, true><<<grid, blk>>>(state, block, height);
                else          kb_absorb_columns_kernel_tpl<16, false><<<grid, blk>>>(state, block, height);
                break;
            case 24:
                if (is_first) kb_absorb_columns_kernel_tpl<24, true><<<grid, blk>>>(state, block, height);
                else          kb_absorb_columns_kernel_tpl<24, false><<<grid, blk>>>(state, block, height);
                break;
            default:
                if (is_first) kb_absorb_columns_kernel_grp<true><<<grid, blk>>>(state, block, width_block, height);
                else          kb_absorb_columns_kernel_grp<false><<<grid, blk>>>(state, block, width_block, height);
                break;
        }
        return CHECK_KERNEL();
    }
    kb_absorb_columns_kernel<<<grid, blk>>>(state, block, width_block, height, is_first, is_last);
    return CHECK_KERNEL();
}

extern "C" int _kb_poseidon2_finalize_leaf_digests(
    kb_digest_t *out,
    const Fp *state,
    size_t query_stride,
    size_t log_rows_per_query
) {
    if (log_rows_per_query > 10) return cudaErrorInvalidValue;
    size_t block_y = size_t{1} << log_rows_per_query;
    size_t threads_x = std::max<size_t>(1, size_t{512} / block_y);
    auto [grid, block] = kernel_launch_params(query_stride, threads_x);
    block.y = block_y;
    size_t shared_stride = block.x * div_ceil_kb(block.y, 2);
    size_t shmem_bytes = CELLS_OUT * shared_stride * sizeof(Fp);
    auto height = query_stride << log_rows_per_query;

    // Thread-per-query serial tree for the hot lrq 1..4 shapes (1.348x standalone);
    // lrq==0 and lrq>4 keep the warp-per-leaf kernel (deep template recursion would
    // blow up code size, cf. the leaf_hash unroll regression).
    if (log_rows_per_query >= 1 && log_rows_per_query <= 4) {
        size_t sgrid = div_ceil_kb(query_stride, FINALIZE_BLOCK);
        switch ((int)log_rows_per_query) {
            case 1: kb_finalize_leaf_digests_serial_kernel<1><<<(unsigned)sgrid, (unsigned)FINALIZE_BLOCK>>>(out, state, height, query_stride); break;
            case 2: kb_finalize_leaf_digests_serial_kernel<2><<<(unsigned)sgrid, (unsigned)FINALIZE_BLOCK>>>(out, state, height, query_stride); break;
            case 3: kb_finalize_leaf_digests_serial_kernel<3><<<(unsigned)sgrid, (unsigned)FINALIZE_BLOCK>>>(out, state, height, query_stride); break;
            default: kb_finalize_leaf_digests_serial_kernel<4><<<(unsigned)sgrid, (unsigned)FINALIZE_BLOCK>>>(out, state, height, query_stride); break;
        }
        return CHECK_KERNEL();
    }

    kb_finalize_leaf_digests_kernel<<<grid, block, shmem_bytes>>>(
        out, state, height, query_stride, log_rows_per_query);
    return CHECK_KERNEL();
}

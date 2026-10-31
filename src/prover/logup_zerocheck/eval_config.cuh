// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

// Host-side launch geometry and scratch-buffer sizing for the constraint-evaluation kernels.

#include "field/fp.h"
#include "field/fpext.h"
#include "common/launcher.cuh"

// Standard library and CUDA vector types.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector_types.h>

// Hardware limit on gridDim.y.
constexpr uint32_t CUDA_GRID_Y_DIM_MAX = 65535;

inline bool valid_grid_y_dim(uint32_t rows) {
    return rows > 0 && rows <= CUDA_GRID_Y_DIM_MAX;
}

inline bool valid_frac_grid_y_dim(uint32_t x_count) {
    return x_count > 0 && static_cast<uint64_t>(x_count) * 2 <= CUDA_GRID_Y_DIM_MAX;
}

// Common code behind the two coset-based round-0 configurations. COSET_PARALLEL selects the grid
// shape: false (lockstep) keeps grid.y = 1 and lets each thread loop over every coset through the
// NUM_COSETS template argument; true (coset-parallel) sets grid.y = coset_count, one coset per block.
namespace round0_config_impl {

template <bool COSET_PARALLEL>
inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t inter_len,
    uint32_t domain_size,
    uint32_t x_count,
    uint32_t coset_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_thread_cap
) {
    // A whole skip domain has to live inside a single block.
    assert(domain_size <= 1024 && "skip_domain exceeds CUDA max threads per block (1024)");
    auto thread_limit = std::max(domain_size, block_thread_cap);
    auto [grid_dim, block_dim] = kernel_launch_params(domain_size * x_count, thread_limit);

    if constexpr (COSET_PARALLEL) {
        grid_dim.y = coset_count; // second grid dimension enumerates the cosets
    }
    // Lockstep leaves grid.y at its default of 1.

    // Spill size is identical in both shapes: lockstep needs grid.x * block.x * coset_count
    // * inter_len slots, coset-parallel grid.x * grid.y * block.x * inter_len with
    // grid.y == coset_count.
    size_t spill_bytes =
        inter_len <= spill_threshold
            ? 0
            : (size_t)grid_dim.x * block_dim.x * coset_count * inter_len * sizeof(Fp);

    uint32_t x_per_thread = std::max(
        div_ceil(spill_bytes, std::max(temp_budget, (size_t)1)), (size_t)1
    );
    if (x_per_thread <= grid_dim.x) {
        grid_dim.x = grid_dim.x / x_per_thread;
    } else {
        grid_dim.x = 1;
        block_dim.x = std::max(x_count / x_per_thread, 1u) * domain_size;
    }

    return {grid_dim, block_dim};
}

template <bool COSET_PARALLEL>
inline uint32_t temp_sums_buffer_size(
    uint32_t inter_len,
    uint32_t domain_size,
    uint32_t x_count,
    uint32_t coset_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_threads
) {
    auto [grid_dim, _] = eval_constraints_launch_params<COSET_PARALLEL>(
        inter_len,
        domain_size,
        x_count,
        coset_count,
        temp_budget,
        spill_threshold,
        block_threads
    );
    // One row of coset_count * domain_size partial sums per block.
    return grid_dim.x * coset_count * domain_size;
}

template <bool COSET_PARALLEL>
inline uint32_t intermediates_buffer_size(
    uint32_t inter_len,
    uint32_t domain_size,
    uint32_t x_count,
    uint32_t coset_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_threads
) {
    if (inter_len <= spill_threshold) {
        return 0;
    }
    auto [grid_dim, block_dim] = eval_constraints_launch_params<COSET_PARALLEL>(
        inter_len,
        domain_size,
        x_count,
        coset_count,
        temp_budget,
        spill_threshold,
        block_threads
    );
    // Indexed as [inter_len][thread][coset].
    return grid_dim.x * block_dim.x * inter_len * coset_count;
}

} // namespace round0_config_impl

// Lockstep shape: a 1-D grid in which every thread evaluates all cosets (NUM_COSETS template).
namespace coset_round0_config {

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t inter_len,
    uint32_t domain_size,
    uint32_t x_count,
    uint32_t coset_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_thread_cap
) {
    return round0_config_impl::eval_constraints_launch_params<false>(
        inter_len,
        domain_size,
        x_count,
        coset_count,
        temp_budget,
        spill_threshold,
        block_thread_cap
    );
}

} // namespace coset_round0_config

// Coset-parallel shape: grid.y == coset_count, so a block serves the single coset blockIdx.y.
namespace coset_parallel_round0_config {

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t inter_len,
    uint32_t domain_size,
    uint32_t x_count,
    uint32_t coset_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_thread_cap
) {
    return round0_config_impl::eval_constraints_launch_params<true>(
        inter_len,
        domain_size,
        x_count,
        coset_count,
        temp_budget,
        spill_threshold,
        block_thread_cap
    );
}

} // namespace coset_parallel_round0_config

// (x_int, z_int) is flattened onto a 1-D threadIdx.x so that each warp sees a single x_int. A block
// covers up to WARP_SIZE values of z and fills the rest with x; how many x blocks go into grid.x is
// set by temp_budget, which has to hold the temp sums plus, if they spill, the intermediates.
namespace align_x_round0_config {

// Gives {z_blocks, z_per_block}. Divergence is lowest when `z_range` is a multiple of WARP_SIZE (32).
inline std::pair<uint32_t, uint32_t> get_z_dim(uint32_t z_range) {
    // Nothing is shared across z through shared memory, so the only requirement is a uniform
    // `x_int` per warp. Below WARP_SIZE some divergence cannot be avoided; then the aim is simply to
    // pack the block as densely as possible.
    uint32_t z_per_block = std::min(z_range, static_cast<uint32_t>(WARP_SIZE));
    uint32_t z_blocks = div_ceil(z_range, z_per_block);
    return {z_blocks, z_per_block};
}

inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t inter_len,
    uint32_t z_range, // number of z_int values
    uint32_t x_count, // number of x_int values
    size_t temp_budget,
    uint32_t spill_threshold, // above this many intermediates they spill to global memory
    uint32_t block_threads
) {
    auto [z_blocks, z_per_block] = get_z_dim(z_range);
    auto x_per_block =
        std::min(static_cast<size_t>(x_count), div_ceil(block_threads, z_per_block));
    dim3 block_dim(x_per_block * z_per_block);
    // grid.x = x_blocks * z_blocks, and x_blocks is what trades parallelism against global memory.
    // Per x block the temp sums take z_range slots (see temp_sums_buffer_size).
    size_t slots_per_x = z_range;
    if (inter_len > spill_threshold) {
        // Intermediates spill to global memory: another x_per_block * z_range * inter_len slots
        // per x block (see intermediates_buffer_size).
        slots_per_x += static_cast<size_t>(x_per_block) * z_range * inter_len;
    }
    size_t x_blocks = temp_budget / (slots_per_x * sizeof(FpExt));
    // Clamp to [1, blocks needed to cover x_count] so a launch is always possible.
    x_blocks = std::max(x_blocks, static_cast<size_t>(1));
    x_blocks = std::min(x_blocks, div_ceil(x_count, x_per_block));
    dim3 grid_dim(x_blocks * z_blocks);

    return {grid_dim, block_dim};
}

inline size_t temp_sums_buffer_size(
    uint32_t inter_len,
    uint32_t z_range,
    uint32_t x_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_threads
) {
    auto [grid_dim, block_dim] = eval_constraints_launch_params(
        inter_len, z_range, x_count, temp_budget, spill_threshold, block_threads
    );
    (void)block_dim;
    auto [z_blocks, z_per_block] = get_z_dim(z_range);
    (void)z_per_block;
    auto x_blocks = grid_dim.x / z_blocks;
    return static_cast<size_t>(z_range) * x_blocks;
}

inline size_t intermediates_buffer_size(
    uint32_t inter_len,
    uint32_t z_range,
    uint32_t x_count,
    size_t temp_budget,
    uint32_t spill_threshold,
    uint32_t block_threads
) {
    if (inter_len <= spill_threshold) {
        return 0;
    }
    auto [grid_dim, block_dim] = eval_constraints_launch_params(
        inter_len, z_range, x_count, temp_budget, spill_threshold, block_threads
    );
    auto [z_blocks, z_per_block] = get_z_dim(z_range);
    auto x_per_block = block_dim.x / z_per_block;
    auto x_blocks = grid_dim.x / z_blocks;
    size_t thread_slots = static_cast<size_t>(x_blocks) * x_per_block * z_range;
    return thread_slots * inter_len;
}

} // namespace align_x_round0_config

namespace mle_rounds_config {
inline std::pair<dim3, dim3> eval_constraints_launch_params(
    uint32_t x_count,
    uint32_t y_count,
    size_t block_thread_cap
) {
    auto block_threads = std::min(std::max(WARP_SIZE, (size_t)y_count), block_thread_cap);
    auto y_blocks = div_ceil(y_count, block_threads);
    dim3 grid_dim = dim3(y_blocks, x_count);
    dim3 block_dim = dim3(block_threads);
    return {grid_dim, block_dim};
}

inline size_t temp_sums_buffer_size(uint32_t x_count, uint32_t y_count, uint32_t block_threads) {
    auto [grid_dim, block_dim] = eval_constraints_launch_params(x_count, y_count, block_threads);
    (void)block_dim;
    return static_cast<size_t>(grid_dim.x) * grid_dim.y;
}

inline size_t intermediates_buffer_size(
    uint32_t inter_len,
    uint32_t x_count,
    uint32_t y_count,
    uint32_t spill_threshold,
    uint32_t block_threads
) {
    if (inter_len <= spill_threshold) {
        return 0;
    }
    auto [grid_dim, block_dim] = eval_constraints_launch_params(x_count, y_count, block_threads);
    return static_cast<size_t>(block_dim.x) * grid_dim.x * grid_dim.y * inter_len;
}
} // namespace mle_rounds_config

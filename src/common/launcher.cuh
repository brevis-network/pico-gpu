// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#ifdef CUDA_DEBUG
#include <cstdio>
#endif

// Host-side helpers for sizing kernel launches and checking their status.

static const size_t MAX_THREADS = 1024;
static const size_t WARP_SIZE = 32;

inline size_t div_ceil(size_t num, size_t den) { return (num + den - 1) / den; }

// 1-D launch covering `n_items` threads, with at most `block_cap` threads per block.
inline std::pair<dim3, dim3> kernel_launch_params(
    size_t n_items,
    size_t block_cap = MAX_THREADS
) {
    size_t threads = std::min(n_items, block_cap);
    size_t blocks = div_ceil(n_items, threads);
    return std::make_pair(dim3(blocks, 1, 1), dim3(threads, 1, 1));
}

// 2-D launch over a cols x rows domain, using blocks of up to 32 x 32 threads.
inline std::pair<dim3, dim3> kernel_launch_2d_params(size_t cols, size_t rows) {
    dim3 tile = dim3(std::min(cols, WARP_SIZE), std::min(rows, WARP_SIZE));
    dim3 tiles = dim3(div_ceil(cols, tile.x), div_ceil(rows, tile.y));
    return std::make_pair(tiles, tile);
}

// Evaluates a CUDA runtime call and prints an error report if it did not return cudaSuccess.
#define CUDA_OK(expr) do {                                  \
    cudaError_t cuda_ok_status = expr;                      \
    if (cuda_ok_status != cudaSuccess) {                    \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(cuda_ok_status)); \
    }                                                       \
} while(0)

#ifdef CUDA_DEBUG
    // Debug builds wait for the device after each launch so a failing kernel is reported by name.
    inline int cuda_check_kernel(const char* fn_name) {
        cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            fprintf(stderr, "[ERROR] Kernel '%s' failed: %s\n",
                    fn_name, cudaGetErrorString(sync_status));
        }
        return sync_status;
    }
#   define CHECK_KERNEL() cuda_check_kernel(__func__)
#else
#   define CHECK_KERNEL() cudaGetLastError()
#endif

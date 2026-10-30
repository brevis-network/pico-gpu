// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

/*
 * oss_check_layout.cu — Host/device ABI checks for the trace-generation FFI.
 *
 * Called once per device from `init_mle_gpu_pool` (vm/src/mle/machine/riscv.rs)
 * before any chip's trace builder runs, so an ABI mismatch aborts at start-up
 * rather than silently corrupting a trace.
 *
 * What actually crosses this boundary: gpu-api/src/chips.rs declares every
 * `cuda_*_generate_*` entry point with raw pointers, `usize`, `u64` and a
 * stream — no struct is passed by value. So what has to agree is the width of
 * the scalars the two sides index with, and the element type of every trace
 * buffer, which CUDA writes and Rust reads back as KoalaBear.
 *
 * NOT covered here: the event structs behind the opaque `d_events` arguments.
 * Their layouts do have to match, but they are per-chip and this file has never
 * checked them — the trace-compare runs are what catch those.
 *
 * This file used to also assert DeviceMatrixRef, RustVecStruct, InputTuple and
 * Matrix<kb31_t>. Those belonged to the NTT / FRI / coset-LDE kernels, which
 * this crate no longer carries, and nothing that remains declares them.
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "field/kb31_t.hpp"

#define CHECK_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("OSS LAYOUT CHECK FAILED: %s\n  Expected: %s\n", msg, #cond); \
            fflush(stdout); \
            abort(); \
        } \
    } while (0)

extern "C" void oss_check_layout() {
    // ---- Scalar widths: Rust `usize` and raw pointers against size_t / void* ----
    CHECK_ASSERT(sizeof(size_t) == 8,    "size_t must be 8 bytes (64-bit)");
    CHECK_ASSERT(alignof(size_t) == 8,   "size_t must be 8-aligned");
    CHECK_ASSERT(sizeof(void*) == 8,     "pointer must be 8 bytes (64-bit)");
    CHECK_ASSERT(alignof(void*) == 8,    "pointer must be 8-aligned");

    // ---- kb31_t: the element type of every trace buffer ----
    CHECK_ASSERT(sizeof(kb31_t) == 4,    "kb31_t must be 4 bytes");
    CHECK_ASSERT(alignof(kb31_t) == 4,   "kb31_t must be 4-aligned");

    printf("OSS layout check passed: scalar widths and kb31_t verified.\n");
    fflush(stdout);
}

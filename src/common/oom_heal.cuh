// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include <cuda_runtime.h>

// All four implemented on the Rust side (`memory_manager`), because the external pool's handle
// lives there: trim both pools and log the attempt; log a retry that still failed; count and log a
// per-launch fallback taken; log a cache that will now stay unbuilt.
extern "C" void mle_oom_trim_pools(int device, size_t bytes);
extern "C" void mle_oom_retry_failed(int device, size_t bytes);
extern "C" void mle_oom_note_fallback(int device, size_t bytes);
extern "C" void mle_oom_note_cache_fallback(int device, size_t bytes);

// A failed `cudaMallocAsync` leaves `cudaErrorMemoryAllocation` in the thread's last-error slot,
// and in release `CHECK_KERNEL()` is `cudaGetLastError()`: the caller's own fallback then reads
// that stale OOM after its next launch and fails the prove anyway. Every path out of these helpers
// therefore leaves the slot clean. The templates never look at `T`: the two CUDA libraries define
// some of these element types differently, and an instantiation may be linked from either.
inline void mle_oom_clear() { (void)cudaGetLastError(); }

// Plain `cudaMalloc` for a grow-once cache whose caller runs a correct slower path when it fails:
// the clean slot that path needs, and a log line every time -- the failed build is cached like a
// successful one, so the AIR stays on the slower path for as long as its proving key lives. No
// retry, for the reason given below.
template <class T>
inline cudaError_t mle_malloc_or_fallback(T **ptr, size_t bytes) {
    cudaError_t rc = cudaMalloc(reinterpret_cast<void **>(ptr), bytes);
    if (rc != cudaErrorMemoryAllocation)
        return rc;
    mle_oom_clear();
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
        mle_oom_note_cache_fallback(device, bytes);
    mle_oom_clear();
    return rc;
}

// For a site whose caller has a correct (if slower) path when the allocation fails: no retry, just
// the clean slot that fallback needs, and a counted log line so the near-miss is visible. Kept
// deliberately free of any device-wide sync, which would be refused by -- and would kill -- a
// CUDA-graph capture another lane has open on the card, to save a fast path.
template <class T>
inline cudaError_t mle_malloc_async_or_fallback(T **ptr, size_t bytes, cudaStream_t stream) {
    cudaError_t rc = cudaMallocAsync(reinterpret_cast<void **>(ptr), bytes, stream);
    if (rc != cudaErrorMemoryAllocation)
        return rc;
    mle_oom_clear();
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
        mle_oom_note_fallback(device, bytes);
    mle_oom_clear();
    return rc;
}

// For a site whose failure is the prove's failure: the shape of `d_malloc`'s self-heal -- sync,
// trim both pools, retry once -- minus its guarantee, since the retry runs after the manager lock
// is dropped and another lane may take the released backing first. It carries the same risk
// `d_malloc` accepts: the device-wide sync is refused by, and invalidates, a capture another lane
// has open. Scratch from these units lands in the device default pool, where a card whose external
// pool has reserved everything refuses even a small request. The success path is the one call it
// always was.
template <class T>
inline cudaError_t mle_malloc_async_healing(T **ptr, size_t bytes, cudaStream_t stream) {
    void **raw = reinterpret_cast<void **>(ptr);
    cudaError_t rc = cudaMallocAsync(raw, bytes, stream);
    if (rc != cudaErrorMemoryAllocation)
        return rc;
    mle_oom_clear();
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        mle_oom_clear();
        return rc; // no card to trim against; fail as before, on a clean slot
    }
    mle_oom_trim_pools(device, bytes);
    mle_oom_clear(); // the sync or the trim may have recorded an error of its own
    rc = cudaMallocAsync(raw, bytes, stream);
    if (rc != cudaSuccess) {
        mle_oom_retry_failed(device, bytes);
        mle_oom_clear();
    }
    return rc;
}

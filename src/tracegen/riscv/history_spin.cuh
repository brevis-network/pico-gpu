// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// For tests only: a kernel that holds its stream for a while, so that the tests of an ordering edge
// between history readers can delay the edge's producer, and a consumer the edge no longer orders
// runs first. Only the FFI entry the ordering tests call uses it; history_cpu_memory_lt.cuh includes
// it too, for FFI files that predate this header. In the namespace of the CPU, memory and less-than
// entries, beside which that FFI entry is declared.
#pragma once

#include <cstdint>

#include "common/exception.cuh"
#include "common/rusterror.h"

namespace pico_gpu::cpu_memory_lt_history {
  // Holds its stream for `nanoseconds` of the device's global clock, so that whatever is enqueued
  // on the stream next starts that much later.
  static __global__ void spin_kernel(const uint64_t nanoseconds)
  {
    uint64_t start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
      // Sleeping between reads exists from sm_70 only; earlier targets read the clock back to back.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      __nanosleep(1000);
#endif
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
  }

  inline RustError spin(const uint64_t nanoseconds, cudaStream_t stream)
  {
    try {
      spin_kernel<<<1, 1, 0, stream>>>(nanoseconds);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::cpu_memory_lt_history

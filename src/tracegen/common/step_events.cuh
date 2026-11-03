// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pico_gpu {
  /// Timing events a caller hands a launcher to have its steps bracketed on the launch's own
  /// stream; a launcher given null records nothing. Step `i` is bracketed by `events[2 * i]` and
  /// `events[2 * i + 1]`, and bit `i` of `recorded` is set once both are recorded, so a step the
  /// launch skipped, or whose events failed to record, reads as not run. The events are the
  /// caller's, created before the launch and read after the caller's own wait on the stream:
  /// recording them adds no wait and moves no launch. `started` is the launcher's own.
  struct StepEvents {
    const cudaEvent_t* events;
    size_t steps;
    uint32_t started;
    uint32_t recorded;
  };

  /// Records event `which` of `step` on `stream`. A launch error still pending is the launcher's
  /// to report, so nothing is recorded over it; a failed record clears only its own error, which
  /// would otherwise fail the launch at its next error check.
  inline bool record_step_event(const StepEvents* steps, size_t step, size_t which, cudaStream_t stream)
  {
    if (steps == nullptr || step >= steps->steps || step >= 32) return false;
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (cudaEventRecord(steps->events[2 * step + which], stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    return true;
  }

  inline void step_begin(StepEvents* steps, size_t step, cudaStream_t stream)
  {
    if (record_step_event(steps, step, 0, stream)) steps->started |= 1u << step;
  }

  inline void step_end(StepEvents* steps, size_t step, cudaStream_t stream)
  {
    if (record_step_event(steps, step, 1, stream) && (steps->started & (1u << step)) != 0) {
      steps->recorded |= 1u << step;
    }
  }
} // namespace pico_gpu

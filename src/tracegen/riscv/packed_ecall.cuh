// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// Packed-event (MLE_PACKED_EVENTS=1) Ecall trace-gen kernel. Additive: reconstructs the fat
// FfiCpuEventU64 from the packed stream (exact by the packing invariants, see
// packed_events.cuh) and runs the unmodified ecall event_to_row. Include from chips_group2.cu.

#pragma once

#include "common/exception.cuh"
#include "common/rusterror.h"

#include "tracegen/riscv/packed_events.cuh"
#include "tracegen/riscv/ecall.cuh"

namespace pico_gpu::ecall_u64_packed {
  using namespace pico_gpu::ecall_u64;

  template <class F, size_t NumCols>
  __global__ void ecall_events_to_trace_kernel_packed(
    const FfiCpuEventPacked* events,
    const uint32_t chunk,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const size_t stride = (NumCols > 0) ? NumCols : num_cols;
    EcallChipColsU64<F>* cols =
      reinterpret_cast<EcallChipColsU64<F>*>(&trace_matrix[idx * stride]);
    const FfiCpuEventU64 e = unpack_cpu_event(events[idx], chunk);
    event_to_row<F>(e, *cols, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main_packed(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const FfiCpuEventPacked* events,
    const uint32_t chunk,
    F* trace,
    cudaStream_t stream,
    F* byte_trace)
  {
    try {
      // A width that disagrees with the mirrored struct is refused (see the unpacked launcher).
      constexpr int NUM_ECALL_CHIP_COLS = sizeof(EcallChipColsU64<F>) / sizeof(F);
      if (num_cols != NUM_ECALL_CHIP_COLS) return RustError{cudaErrorInvalidValue};

      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;
      ecall_events_to_trace_kernel_packed<F, NUM_ECALL_CHIP_COLS>
        <<<grid_size, block_size, 0, stream>>>(
          events, chunk, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
      deferred_tally<F>(trace, num_cols, event_size, trace_size / num_cols, stream);
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::ecall_u64_packed

// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "field/ff_config.hpp"
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/riscv/byte.cuh"


namespace pico_gpu::syscall_u64 {
  template <class F>
  __device__ void event_to_row(
    const SyscallEventU64& event, SyscallColsU64<F>* cols,
    GlobalInteractionEvent* global_events, const bool is_receive,
    F* byte_trace)
  {
    cols->chunk = F::from_canonical_u32(event.chunk);
    cols->clk = F::from_canonical_u32(static_cast<uint32_t>(event.clk));
    cols->syscall_id = F::from_canonical_u32(event.syscall_id);

    uint32_t arg1_0 = static_cast<uint32_t>(event.arg1 & 0xFFFF);
    uint32_t arg1_1 = static_cast<uint32_t>((event.arg1 >> 16) & 0xFFFF);
    uint32_t arg1_2 = static_cast<uint32_t>((event.arg1 >> 32) & 0xFFFF);
    cols->arg1[0] = F::from_canonical_u32(arg1_0);
    cols->arg1[1] = F::from_canonical_u32(arg1_1);
    cols->arg1[2] = F::from_canonical_u32(arg1_2);

    uint32_t arg2_0 = static_cast<uint32_t>(event.arg2 & 0xFFFF);
    uint32_t arg2_1 = static_cast<uint32_t>((event.arg2 >> 16) & 0xFFFF);
    uint32_t arg2_2 = static_cast<uint32_t>((event.arg2 >> 32) & 0xFFFF);
    cols->arg2[0] = F::from_canonical_u32(arg2_0);
    cols->arg2[1] = F::from_canonical_u32(arg2_1);
    cols->arg2[2] = F::from_canonical_u32(arg2_2);

    cols->is_real = F::one();

    GlobalInteractionEvent global_event = {
      .message = {
        static_cast<uint32_t>(event.clk),
        event.syscall_id + arg1_0 * 256,
        arg1_1,
        arg1_2,
        arg2_0,
        arg2_1,
        arg2_2,
        event.chunk,
      },
      .is_receive = is_receive,
      .kind = static_cast<uint8_t>(LookupType::Syscall),
    };
    *global_events = global_event;

    // Byte lookup events (guarded for null byte_trace in bench tests)
    if (byte_trace != nullptr) {
      // U8Range for syscall_id (pair with 0)
      byte::add_u8_range_check<F>(byte_trace,
        static_cast<uint8_t>(event.syscall_id & 0xFF), 0);

      // U16Range for arg1[0] (u16 value)
      byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(arg1_0));
    }
  }

  template <class F>
  __global__ void syscall_events_to_trace_kernel(
    const SyscallEventU64* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* global_events,
    const bool is_receive,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SyscallColsU64<F>* cols = reinterpret_cast<SyscallColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], cols, &global_events[idx], is_receive, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const SyscallEventU64* events,
    F* trace,
    cudaStream_t stream,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    const SyscallChunkKind chunk_kind,
    F* byte_trace)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const bool is_receive = chunk_kind == SyscallChunkKind::Precompile;

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      syscall_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, &global_events[global_events_start_index], is_receive,
        byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::syscall_u64
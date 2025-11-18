#pragma once

#include <ff/ff_config.hpp>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"

namespace pico_gpu::syscall {
  /// Convert a syscall event to a trace row and generate global interaction.
  /// Syscalls are used to interact with precompiles and other system functions.
  /// Each syscall generates a global interaction event for cross-chip lookups.
  template <class F>
  __device__ void event_to_row(
    const SyscallEvent& event, SyscallCols<F>* cols, GlobalInteractionEvent* global_events, const bool is_receive)
  {
    // Populate basic syscall information.
    cols->chunk = F::from_canonical_u32(event.chunk);
    cols->clk = F::from_canonical_u32(event.clk);
    cols->syscall_id = F::from_canonical_u32(event.syscall_id);
    cols->arg1 = F::from_canonical_u32(event.arg1);
    cols->arg2 = F::from_canonical_u32(event.arg2);
    cols->is_real = F::one();

    // Create global interaction event for syscall lookup.
    // The message contains: [chunk, clk, syscall_id, arg1, arg2, 0, 0]
    // is_receive depends on whether this is a precompile (receive) or
    // CPU-side syscall (send).
    GlobalInteractionEvent global_event = {
      .message = {event.chunk, event.clk, event.syscall_id, event.arg1, event.arg2, 0, 0},
      .is_receive = is_receive,
      .kind = static_cast<uint8_t>(LookupType::Syscall),
    };
    *global_events = global_event;
  }

  /// Kernel to convert syscall events to trace rows.
  template <class F>
  __global__ void syscall_events_to_trace_kernel(
    const SyscallEvent* events,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* global_events,
    const bool is_receive)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SyscallCols<F>* cols = reinterpret_cast<SyscallCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], cols, &global_events[idx], is_receive);
  }

  /// Generate trace from syscall events.
  /// Syscalls can be either precompile calls (receive) or CPU-side calls (send).
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const SyscallEvent* events,
    F* trace,
    cudaStream_t stream,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    const SyscallChunkKind chunk_kind)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Determine direction: precompile syscalls are receives, others are sends.
      const bool is_receive = chunk_kind == SyscallChunkKind::Precompile;

      // Launch kernel.
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      syscall_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, trace, num_cols, &global_events[global_events_start_index], is_receive);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::syscall
#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "utils.hpp"

namespace pico_gpu::riscv_memory_local {

  struct MemoryLocalExtraEventIndices {
    size_t global;
  };

  /// Convert a memory local event to a trace row with extra record.
  template <class F>
  __device__ inline void event_to_row_with_extra_record(
    const MemoryLocalEvent& event,
    SingleMemoryLocal<F>& cols,
    GlobalInteractionEvent* global_events,
    const size_t global_event_idx)
  {
    // Set the main trace columns.
    cols.addr = F::from_canonical_u32(event.addr);
    cols.initial_chunk = F::from_canonical_u32(event.initial_mem_access.chunk);
    cols.initial_clk = F::from_canonical_u32(event.initial_mem_access.timestamp);
    write_word_from_u32_v2<F>(cols.initial_value, event.initial_mem_access.value);
    cols.final_chunk = F::from_canonical_u32(event.final_mem_access.chunk);
    cols.final_clk = F::from_canonical_u32(event.final_mem_access.timestamp);
    write_word_from_u32_v2<F>(cols.final_value, event.final_mem_access.value);
    cols.is_real = F::one();

    // Generate global lookup events for initial and final memory access.
    if (global_events != nullptr) {
      // Generate global lookup events for initial memory access.
      GlobalInteractionEvent& initial_event = global_events[global_event_idx];
      initial_event.message[0] = event.initial_mem_access.chunk;
      initial_event.message[1] = event.initial_mem_access.timestamp;
      initial_event.message[2] = event.addr;
      initial_event.message[3] = event.initial_mem_access.value & 255;
      initial_event.message[4] = (event.initial_mem_access.value >> 8) & 255;
      initial_event.message[5] = (event.initial_mem_access.value >> 16) & 255;
      initial_event.message[6] = (event.initial_mem_access.value >> 24) & 255;
      initial_event.is_receive = true;
      initial_event.kind = static_cast<uint8_t>(LookupType::Memory);

      // Generate global lookup events for final memory access.
      GlobalInteractionEvent& final_event = global_events[global_event_idx + 1];
      final_event.message[0] = event.final_mem_access.chunk;
      final_event.message[1] = event.final_mem_access.timestamp;
      final_event.message[2] = event.addr;
      final_event.message[3] = event.final_mem_access.value & 255;
      final_event.message[4] = (event.final_mem_access.value >> 8) & 255;
      final_event.message[5] = (event.final_mem_access.value >> 16) & 255;
      final_event.message[6] = (event.final_mem_access.value >> 24) & 255;
      final_event.is_receive = false;
      final_event.kind = static_cast<uint8_t>(LookupType::Memory);
    }
  }

  /// Kernel to convert events to trace with extra record.
  template <class F>
  __global__ void events_to_trace_with_extra_record_kernel(
    const size_t event_size,
    const size_t trace_size,
    const MemoryLocalEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_size) return;

    // Direct mapping: each event corresponds to one SingleMemoryLocal position.
    SingleMemoryLocal<F>* cols = reinterpret_cast<SingleMemoryLocal<F>*>(&trace_matrix[idx * num_cols]);

    // Process the event and generate global events.
    const size_t global_event_idx = global_events_start_index + idx * 2;
    event_to_row_with_extra_record<F>(events[idx], *cols, global_events, global_event_idx);
  }

  /// Generate trace from memory local events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const MemoryLocalEvent* events,
    F* trace,
    cudaStream_t stream,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      events_to_trace_with_extra_record_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, global_events, global_events_start_index);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::riscv_memory_local
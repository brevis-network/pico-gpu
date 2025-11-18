#pragma once

#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <util/rusterror.h>
#include "field_bit_decomposition.hpp"
#include "is_zero.hpp"
#include "lt_gadgets.hpp"

namespace pico_gpu::memory_initialize_finalize {

  /// Convert memory initialization/finalization event to trace row.
  /// Handles both initialization (receive) and finalization (send) events.
  template <class F>
  __device__ void event_to_row(
    const MemoryInitializeFinalizeEvent* event,
    MemoryInitializeFinalizeCols<F>* cols,
    const uint32_t* previous_addr_bits,
    const size_t event_count,
    const size_t idx,
    GlobalInteractionEvent* global_events,
    const bool is_receive)
  {
    cols->addr = F::from_canonical_u32(event->addr);
    field_bit_decomposition::populate(cols->addr_bits, event->addr);
    cols->chunk = F::from_canonical_u32(event->chunk);
    cols->timestamp = F::from_canonical_u32(event->timestamp);

    // Decompose value into bits.
    for (int i = 0; i < 32; i++) {
      cols->value[i] = F::from_canonical_u32(((event->value) >> i) & 1);
    }
    cols->is_real = F::from_canonical_u32(event->used);

    // Handle first row comparison with previous address.
    if (idx == 0) {
      uint32_t prev_addr = 0;
      for (int j = 0; j < 32; ++j) {
        prev_addr += previous_addr_bits[j] << j;
      }
      is_zero::populate(cols->is_prev_addr_zero, prev_addr);

      cols->is_first_comp = F::from_bool(prev_addr != 0);
      if (prev_addr != 0) {
        uint32_t addr_bits[32];
        for (int k = 0; k < 32; ++k) {
          addr_bits[k] = (event->addr >> k) & 1;
        }
        lt_gadgets::populate(cols->lt_cols, previous_addr_bits, addr_bits);
      }
    } else {
      // Compare with previous event in sequence.
      const auto prev_event = event - 1;
      const uint32_t prev_is_real = prev_event->used;
      cols->is_next_comp = F::from_canonical_u32(prev_is_real);

      const uint32_t previous_addr = prev_event->addr;

      uint32_t addr_bits[32], prev_addr_bits[32];
      for (size_t k = 0; k < 32; ++k) {
        addr_bits[k] = (event->addr >> k) & 1;
        prev_addr_bits[k] = (previous_addr >> k) & 1;
      }
      lt_gadgets::populate(cols->lt_cols, prev_addr_bits, addr_bits);
    }

    // Mark last address in sequence.
    if (idx == event_count - 1) cols->is_last_addr = F::one();

    // Create global interaction event.
    uint32_t interaction_chunk = 0;
    if (is_receive) interaction_chunk = event->chunk;

    uint32_t interaction_clk = 0;
    if (is_receive) interaction_clk = event->timestamp;

    GlobalInteractionEvent global_event = {
      .message =
        {interaction_chunk, interaction_clk, event->addr, event->value & 0xFF, (event->value >> 8) & 0xFF,
         (event->value >> 16) & 0xFF, (event->value >> 24) & 0xFF},
      .is_receive = is_receive,
      .kind = static_cast<uint8_t>(LookupType::Memory),
    };
    *global_events = global_event;
  }

  /// Kernel to convert memory init/finalize events to trace.
  template <class F>
  __global__ void memory_initialize_finalize_events_to_trace_kernel(
    const MemoryInitializeFinalizeEvent* events,
    const size_t count,
    const uint32_t* previous_addr_bits,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* global_events,
    const bool is_receive)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MemoryInitializeFinalizeCols<F>* cols =
      reinterpret_cast<MemoryInitializeFinalizeCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events + idx, cols, previous_addr_bits, count, idx, &global_events[idx], is_receive);
  }

  /// Generate trace from memory initialization/finalization events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const MemoryInitializeFinalizeEvent* events,
    const uint32_t* previous_addr_bits,
    F* trace,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    const bool is_receive,
    cudaStream_t stream)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      memory_initialize_finalize_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, previous_addr_bits, trace, num_cols, &global_events[global_events_start_index], is_receive);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::memory_initialize_finalize
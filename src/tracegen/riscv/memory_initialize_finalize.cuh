// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "common/rusterror.h"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/common/field_bit_decomposition.cuh"
#include "tracegen/common/is_zero.cuh"
#include "tracegen/common/lt_gadgets.cuh"


// ─── U64 Memory Initialize/Finalize ─────────────────────────────────
namespace pico_gpu::memory_initialize_finalize_u64 {
  using namespace pico_gpu;
  using namespace pico_gpu::byte;
  using namespace pico_gpu::is_zero;

  template <class F>
  __device__ inline void populate_lt_word_u16(
    LtWordU16Gadget<F>& g, F* byte_trace, uint64_t a_u64, uint64_t b_u64, uint64_t c_u64)
  {
    uint16_t al[4] = {
      (uint16_t)(a_u64), (uint16_t)(a_u64 >> 16),
      (uint16_t)(a_u64 >> 32), (uint16_t)(a_u64 >> 48)};
    uint16_t bl[4] = {
      (uint16_t)(b_u64), (uint16_t)(b_u64 >> 16),
      (uint16_t)(b_u64 >> 32), (uint16_t)(b_u64 >> 48)};
    uint16_t cl[4] = {
      (uint16_t)(c_u64), (uint16_t)(c_u64 >> 16),
      (uint16_t)(c_u64 >> 32), (uint16_t)(c_u64 >> 48)};

    uint16_t a_u16 = al[0];

    for (int i = 0; i < 4; ++i) g.u16_flags[i] = F::zero();
    g.not_eq_inv = F::zero();
    g.comparison_limbs[0] = F::zero();
    g.comparison_limbs[1] = F::zero();

    uint16_t comparison_limbs_raw[2] = {0, 0};
    for (int i = 3; i >= 0; --i) {
      if (bl[i] != cl[i]) {
        g.u16_flags[i] = F::one();
        comparison_limbs_raw[0] = bl[i];
        comparison_limbs_raw[1] = cl[i];
        F bf = F::from_canonical_u16(bl[i]);
        F cf = F::from_canonical_u16(cl[i]);
        g.not_eq_inv = (bf - cf).reciprocal();
        g.comparison_limbs[0] = bf;
        g.comparison_limbs[1] = cf;
        break;
      }
    }

    g.bit = F::from_canonical_u16(a_u16);
    uint16_t diff = comparison_limbs_raw[0] - comparison_limbs_raw[1];
    add_u16_range_check<F>(byte_trace, diff);
  }

  __device__ inline void build_global_message_u64(
    uint32_t* message, uint32_t chunk, uint32_t clk, uint64_t addr, uint64_t value)
  {
    const uint32_t a0 = static_cast<uint32_t>(addr & 0xFFFF);
    const uint32_t a1 = static_cast<uint32_t>((addr >> 16) & 0xFFFF);
    const uint32_t a2 = static_cast<uint32_t>((addr >> 32) & 0xFFFF);
    const uint32_t v0 = static_cast<uint32_t>(value & 0xFFFF);
    const uint32_t v1 = static_cast<uint32_t>((value >> 16) & 0xFFFF);
    const uint32_t v2_lower = static_cast<uint32_t>((value >> 32) & 0xFF);
    const uint32_t v2_upper = static_cast<uint32_t>((value >> 40) & 0xFF);
    const uint32_t v3 = static_cast<uint32_t>((value >> 48) & 0xFFFF);

    message[0] = chunk;
    message[1] = clk;
    message[2] = a0;
    message[3] = a1;
    message[4] = a2;
    message[5] = v0 + (v2_lower << 16);
    message[6] = v1 + (v2_upper << 16);
    message[7] = v3;
  }

  template <class F>
  __device__ void event_to_row(
    const MemoryInitializeFinalizeEventU64* __restrict__ event,
    MemoryInitializeFinalizeColsU64<F>* __restrict__ cols,
    const uint32_t* __restrict__ previous_addr_limbs,
    const size_t event_count,
    const size_t idx,
    GlobalInteractionEvent* __restrict__ global_events,
    const bool is_receive,
    F* __restrict__ byte_trace)
  {
    const uint64_t addr = event->addr;
    const uint64_t value = event->value;

    cols->addr[0] = F::from_canonical_u16(static_cast<uint16_t>(addr & 0xFFFF));
    cols->addr[1] = F::from_canonical_u16(static_cast<uint16_t>((addr >> 16) & 0xFFFF));
    cols->addr[2] = F::from_canonical_u16(static_cast<uint16_t>((addr >> 32) & 0xFFFF));

    cols->chunk = F::from_canonical_u32(event->chunk);
    cols->timestamp = F::from_canonical_u32(event->timestamp);

    cols->value._0[0] = F::from_canonical_u16(static_cast<uint16_t>(value & 0xFFFF));
    cols->value._0[1] = F::from_canonical_u16(static_cast<uint16_t>((value >> 16) & 0xFFFF));
    cols->value._0[2] = F::from_canonical_u16(static_cast<uint16_t>((value >> 32) & 0xFFFF));
    cols->value._0[3] = F::from_canonical_u16(static_cast<uint16_t>((value >> 48) & 0xFFFF));

    cols->value_lower = F::from_canonical_u32(static_cast<uint32_t>((value >> 32) & 0xFF));
    cols->value_upper = F::from_canonical_u32(static_cast<uint32_t>((value >> 40) & 0xFF));

    cols->is_real = F::from_canonical_u32(event->used);

    uint64_t prev_addr;
    if (idx == 0) {
      prev_addr = (uint64_t)previous_addr_limbs[0]
                | ((uint64_t)previous_addr_limbs[1] << 16)
                | ((uint64_t)previous_addr_limbs[2] << 32);

      cols->prev_addr[0] = F::from_canonical_u32(previous_addr_limbs[0]);
      cols->prev_addr[1] = F::from_canonical_u32(previous_addr_limbs[1]);
      cols->prev_addr[2] = F::from_canonical_u32(previous_addr_limbs[2]);

      uint32_t prev_addr_limb_sum = previous_addr_limbs[0]
                                  + previous_addr_limbs[1]
                                  + previous_addr_limbs[2];
      populate(cols->is_prev_addr_zero, prev_addr_limb_sum);
      cols->is_comp = (prev_addr != 0) ? F::one() : F::zero();
    } else {
      const auto* prev_event = event - 1;
      prev_addr = prev_event->addr;
      cols->prev_addr[0] = F::from_canonical_u16(static_cast<uint16_t>(prev_addr & 0xFFFF));
      cols->prev_addr[1] = F::from_canonical_u16(static_cast<uint16_t>((prev_addr >> 16) & 0xFFFF));
      cols->prev_addr[2] = F::from_canonical_u16(static_cast<uint16_t>((prev_addr >> 32) & 0xFFFF));
      cols->is_comp = F::from_canonical_u32(event->used);
    }

    if (cols->is_comp == F::one()) {
      populate_lt_word_u16(cols->lt_cols, byte_trace, 1, prev_addr, addr);
    }

    // Byte lookups: value u16 range checks
    for (int i = 0; i < 4; ++i)
      add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(value >> (16 * i)));

    // Byte lookups: addr u16 range checks
    for (int i = 0; i < 3; ++i)
      add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(addr >> (16 * i)));

    // Byte lookups: prev_addr u16 range checks
    for (int i = 0; i < 3; ++i)
      add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(prev_addr >> (16 * i)));

    // Byte lookups: v2 packing u8 range checks
    add_u8_range_check<F>(byte_trace,
      static_cast<uint8_t>((value >> 32) & 0xFF),
      static_cast<uint8_t>((value >> 40) & 0xFF));

    // Global interaction event
    uint32_t interaction_chunk = is_receive ? event->chunk : 0;
    uint32_t interaction_clk = is_receive ? event->timestamp : 0;

    GlobalInteractionEvent global_event;
    build_global_message_u64(global_event.message, interaction_chunk, interaction_clk, addr, value);
    global_event.is_receive = is_receive;
    global_event.kind = static_cast<uint8_t>(LookupType::Memory);
    *global_events = global_event;
  }

  template <class F>
  __global__ void memory_init_finalize_events_to_trace_kernel(
    const MemoryInitializeFinalizeEventU64* __restrict__ events,
    const size_t count,
    const uint32_t* __restrict__ previous_addr_limbs,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* __restrict__ global_events,
    const bool is_receive,
    F* __restrict__ byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MemoryInitializeFinalizeColsU64<F>* cols =
      reinterpret_cast<MemoryInitializeFinalizeColsU64<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(events + idx, cols, previous_addr_limbs, count, idx,
                    &global_events[idx], is_receive, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const MemoryInitializeFinalizeEventU64* events,
    const uint32_t* previous_addr_limbs,
    F* trace,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    const bool is_receive,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      memory_init_finalize_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_size, previous_addr_limbs, trace, num_cols,
        &global_events[global_events_start_index], is_receive, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::memory_initialize_finalize_u64
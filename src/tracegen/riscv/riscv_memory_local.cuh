// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"


// ─── U64 Memory Local ───────────────────────────────────────────────
namespace pico_gpu::riscv_memory_local_u64 {
  using namespace pico_gpu;

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
  __device__ inline void write_word_from_u64(Word<F>& word, uint64_t value)
  {
    word._0[0] = F::from_canonical_u16(static_cast<uint16_t>(value & 0xFFFF));
    word._0[1] = F::from_canonical_u16(static_cast<uint16_t>((value >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16(static_cast<uint16_t>((value >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16(static_cast<uint16_t>((value >> 48) & 0xFFFF));
  }

  template <class F>
  __device__ inline void event_to_row_with_extra_record(
    const MemoryLocalEventU64& event,
    SingleMemoryLocalU64<F>& cols,
    GlobalInteractionEvent* global_events,
    const size_t global_event_idx,
    F* byte_trace)
  {
    const uint64_t addr = event.addr;
    cols.addr[0] = F::from_canonical_u16(static_cast<uint16_t>(addr & 0xFFFF));
    cols.addr[1] = F::from_canonical_u16(static_cast<uint16_t>((addr >> 16) & 0xFFFF));
    cols.addr[2] = F::from_canonical_u16(static_cast<uint16_t>((addr >> 32) & 0xFFFF));

    cols.initial_chunk = F::from_canonical_u32(event.initial_mem_access.chunk);
    cols.initial_clk = F::from_canonical_u32(event.initial_mem_access.timestamp);
    cols.final_chunk = F::from_canonical_u32(event.final_mem_access.chunk);
    cols.final_clk = F::from_canonical_u32(event.final_mem_access.timestamp);

    const uint64_t init_val = event.initial_mem_access.value;
    write_word_from_u64(cols.initial_value, init_val);
    const uint8_t init_v2_lower = static_cast<uint8_t>((init_val >> 32) & 0xFF);
    const uint8_t init_v2_upper = static_cast<uint8_t>((init_val >> 40) & 0xFF);
    cols.initial_value_lower = F::from_canonical_u32(static_cast<uint32_t>(init_v2_lower));
    cols.initial_value_upper = F::from_canonical_u32(static_cast<uint32_t>(init_v2_upper));

    const uint64_t final_val = event.final_mem_access.value;
    write_word_from_u64(cols.final_value, final_val);
    const uint8_t final_v2_lower = static_cast<uint8_t>((final_val >> 32) & 0xFF);
    const uint8_t final_v2_upper = static_cast<uint8_t>((final_val >> 40) & 0xFF);
    cols.final_value_lower = F::from_canonical_u32(static_cast<uint32_t>(final_v2_lower));
    cols.final_value_upper = F::from_canonical_u32(static_cast<uint32_t>(final_v2_upper));

    cols.is_real = F::one();
    // Host-decided (Rust is_fresh_local_mem): a fresh entry declares its zero initial state and
    // receives nothing from the Global bus.
    cols.has_initial = event.has_initial ? F::one() : F::zero();

    // Which shape this entry has, and the normalized clk that follows. Mirrors
    // normalize_initial() on the Rust side.
    const bool in_chunk =
      event.initial_mem_access.chunk == event.final_mem_access.chunk;
    const uint32_t clk_regional = in_chunk ? event.initial_mem_access.timestamp : 0u;
    const uint64_t clk_diff =
      in_chunk ? ((uint64_t)event.final_mem_access.timestamp
                  - (uint64_t)event.initial_mem_access.timestamp - 1)
               : 0u;
    cols.initial_in_chunk = in_chunk ? F::one() : F::zero();
    cols.initial_clk_regional = F::from_canonical_u32(clk_regional);
    cols.clk_diff_16bit_limb =
      F::from_canonical_u16(static_cast<uint16_t>(clk_diff & 0xFFFF));
    cols.clk_diff_hi9 =
      F::from_canonical_u16(static_cast<uint16_t>((clk_diff >> 16) & 0x1FF));

    // --- Byte lookups (matching Rust extra_record) ---
    if (byte_trace != nullptr) {
      // Initial value: u8 range check for v2 packing cols
      byte::add_u8_range_check<F>(byte_trace, init_v2_lower, init_v2_upper);
      // Initial value: u16 range check for limbs 0 and 1 only (the two Global packing slots);
      // the AIR binds the other limbs through the memory bus (Rust local/constraints.rs).
      byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(init_val & 0xFFFF));
      byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>((init_val >> 16) & 0xFFFF));

      // Final value: u8 range check for v2 packing cols
      byte::add_u8_range_check<F>(byte_trace, final_v2_lower, final_v2_upper);
      // No U16 events for the final value or the address: the AIR has no range checks on them.

      // Whichever shape the entry has, it pays exactly one range check: the clk gap inside the
      // chunk, or the chunk gap across chunks.
      if (in_chunk) {
        byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(clk_diff & 0xFFFF));
        byte::add_bit_range_check<F>(
          byte_trace, static_cast<uint16_t>((clk_diff >> 16) & 0x1FF), 9);
      } else {
        const uint32_t chunk_diff =
          event.final_mem_access.chunk - event.initial_mem_access.chunk - 1;
        byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(chunk_diff));
      }
    }

    // Global events, in the order and at the positions the host's extra_record writes them:
    // the initial receive (only when the entry has one), then the final send. `global_event_idx`
    // already includes this entry's `global_slot` (the events the entries before it wrote), so
    // the layout has no holes where a fresh entry writes one event instead of two.
    if (global_events != nullptr) {
      size_t slot = global_event_idx;
      if (event.has_initial) {
        GlobalInteractionEvent& init_ev = global_events[slot];
        build_global_message_u64(init_ev.message,
          event.initial_mem_access.chunk, event.initial_mem_access.timestamp,
          addr, init_val);
        init_ev.is_receive = true;
        init_ev.kind = static_cast<uint8_t>(LookupType::Memory);
        slot += 1;
      }

      GlobalInteractionEvent& final_ev = global_events[slot];
      build_global_message_u64(final_ev.message,
        event.final_mem_access.chunk, event.final_mem_access.timestamp,
        addr, final_val);
      final_ev.is_receive = false;
      final_ev.kind = static_cast<uint8_t>(LookupType::Memory);
    }
  }

  template <class F>
  __global__ void events_to_trace_with_extra_record_kernel(
    const size_t event_size,
    const MemoryLocalEventU64* events,
    F* trace_matrix,
    const size_t num_cols,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_size) return;

    SingleMemoryLocalU64<F>* cols =
      reinterpret_cast<SingleMemoryLocalU64<F>*>(&trace_matrix[idx * num_cols]);

    // Not `idx * 2`: a fresh entry writes one Global event, so the host hands each entry the
    // prefix sum of what the entries before it write (MemoryLocalEventU64::global_slot).
    const size_t global_event_idx = global_events_start_index + events[idx].global_slot;
    event_to_row_with_extra_record<F>(events[idx], *cols, global_events, global_event_idx, byte_trace);
  }

  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const MemoryLocalEventU64* events,
    F* trace,
    cudaStream_t stream,
    GlobalInteractionEvent* global_events,
    const size_t global_events_start_index,
    F* byte_trace)
  {
    // `num_cols` is the per-entry stride the kernel writes SingleMemoryLocalU64 at, and Rust
    // derives it from its own struct -- so this checks that the hand-maintained CUDA mirror still
    // matches. A CUDA struct wider than the stride overlaps the next entry and runs past the pool
    // on the last one; a narrower one silently leaves each entry's tail columns zero. Exact
    // equality is the invariant. The production caller asserts on the returned code; the bench in
    // its test module does not, and would surface a trace mismatch instead.
    if (num_cols != sizeof(SingleMemoryLocalU64<F>) / sizeof(F)) {
      return RustError{cudaErrorInvalidValue};
    }

    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      events_to_trace_with_extra_record_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, events, trace, num_cols, global_events, global_events_start_index, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::riscv_memory_local_u64
// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include "field/ff_config.hpp"
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/riscv/byte.cuh"

// The Global chip: one row per cross-chunk event, in the event buffer's order. A row carries the
// event itself (message, kind, multiplicity: the chunk's event table) and the range-check
// witnesses of the chunk's own proof. Mirrors the host trace generation
// (`vm/src/chips/chips/riscv_global/traces.rs`), byte lookups included.
namespace pico_gpu::riscv_global {
  template <class F>
  __device__ inline void event_to_row(const GlobalInteractionEvent& event, GlobalCols<F>* cols, F* byte_trace)
  {
    for (int i = 0; i < 8; i++) {
      cols->message[i] = F::from_canonical_u32(event.message[i]);
    }
    cols->kind = F::from_canonical_u8(event.kind);
    cols->is_real = F::one();
    if (event.is_receive) {
      cols->is_receive = F::one();
      cols->mu = -F::one();
    } else {
      cols->is_send = F::one();
      cols->mu = F::one();
    }

    const uint32_t m0 = event.message[0];
    const uint32_t m1 = event.message[1];
    // message[0]: 16 + 8 + 1 limbs (the boolean high bit needs no lookup).
    cols->message_0_16bit_limb = F::from_canonical_u32(m0 & 0xFFFF);
    cols->message_0_8bit_limb = F::from_canonical_u32((m0 >> 16) & 0xFF);
    cols->message_0_high_bit = F::from_canonical_u32((m0 >> 24) & 1);
    // message[1]: 16 + 9 limbs.
    cols->message_1_16bit_limb = F::from_canonical_u32(m1 & 0xFFFF);
    cols->message_1_hi9 = F::from_canonical_u32((m1 >> 16) & 0x1FF);

    // The host's byte lookups, one for one. The U8 pair's slot order is the constraint's: b is
    // the 8-bit limb, c is zero.
    byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(m0 & 0xFFFF));
    byte::add_u8_range_check<F>(byte_trace, static_cast<uint8_t>((m0 >> 16) & 0xFF), 0);
    byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(m1 & 0xFFFF));
    byte::add_bit_range_check<F>(byte_trace, static_cast<uint16_t>((m1 >> 16) & 0x1FF), 9);
    byte::add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(event.message[7]));
  }

  template <class F>
  __global__ void events_to_trace_kernel(
    const GlobalInteractionEvent* events, const size_t count, F* trace_matrix, const size_t num_cols, F* byte_trace)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    GlobalCols<F>* cols = reinterpret_cast<GlobalCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row(events[idx], cols, byte_trace);
  }

  /// Writes `event_size` rows from `events` at the top of `trace` and leaves the rest zero.
  /// `trace_size` is the trace's size in field elements, padding rows included.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const GlobalInteractionEvent* events,
    F* trace,
    cudaStream_t stream,
    F* byte_trace)
  {
    try {
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));
      if (event_size == 0) return RustError{cudaSuccess};

      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(events, event_size, trace, num_cols, byte_trace);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::riscv_global

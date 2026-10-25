// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "common/rusterror.h"
#include "common/exception.cuh"
#include "tracegen/precompiles/keccak_consts.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/riscv/syscall_addr.cuh"
#include "tracegen/riscv/byte.cuh"

// KeccakControlChip: one row per KECCAK_PERMUTE syscall -- syscall receive, state-pointer
// gadgets and the 25 u64 state reads (at clk) / writes (at clk + 1). Mirrors the Rust
// `KeccakControlChip::event_to_row` (`keccak256/control.rs`); modelled on the ShaCompress
// control kernels in `precompile_sha256_compress.cuh`.
//
// Byte lookups per event: SyscallAddrGadget (3 U16 + 1 BitRange13), 25 x AddrAddGadget (3 U16),
// 50 x memory access (1 U16 + 1 BitRange9). None for the state limbs: the KeccakPermute state bus
// equates them with p3's bit-decomposed columns.
namespace pico_gpu::keccak {

  template <typename F>
  __device__ inline void
  event_to_row_control(const KeccakPermuteEvent& ev, KeccakControlCols<F>* cols, F* byte_trace)
  {
    using memory_read_write_u64::populate_access_u64;
    using memory_read_write_u64::write_word_from_u64;

    cols->chunk = F::from_canonical_u32(ev.chunk);
    cols->clk = F::from_canonical_u32(ev.clk);
    const uint64_t addr64 = static_cast<uint64_t>(ev.state_addr);
    // 25 words x 8 bytes.
    cols->state_addr.populate(byte_trace, addr64, 200);
    for (int i = 0; i < STATE_NUM_WORDS; ++i) {
      cols->state_addrs[i].populate(byte_trace, addr64, static_cast<uint64_t>(8 * i));

      const auto& r = ev.state_read_records[i];
      populate_access_u64(cols->state_read[i].access,
        r.chunk, r.timestamp, r.value,
        r.prev_chunk, r.prev_timestamp, r.value,
        byte_trace);

      const auto& w = ev.state_write_records[i];
      write_word_from_u64(cols->state_write[i].prev_value, w.prev_value);
      populate_access_u64(cols->state_write[i].access,
        w.chunk, w.timestamp, w.value,
        w.prev_chunk, w.prev_timestamp, w.prev_value,
        byte_trace);
    }
    cols->is_real = F::one();
  }

  template <typename F>
  __global__ void control_events_to_rows_kernel(
    const KeccakPermuteEvent* __restrict__ events,
    F* __restrict__ trace_matrix,
    F* byte_trace,
    const size_t events_count,
    const size_t num_cols)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= events_count) return;

    KeccakControlCols<F>* cols = reinterpret_cast<KeccakControlCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row_control(events[idx], cols, byte_trace);
  }

  /// One row per event, zero padding to `total_rows` (the memset below is the padding).
  template <class F>
  inline RustError generate_control_extra_record_and_main(
    const KeccakPermuteEvent* events,
    const size_t events_count,
    F* trace,
    const size_t total_rows,
    const size_t num_cols,
    F* byte_trace,
    cudaStream_t stream)
  {
    try {
      if (events_count == 0) return RustError{cudaSuccess};

      CUDA_OK(cudaMemsetAsync(trace, 0, (total_rows * num_cols) * sizeof(F), stream));

      // populate_access_u64 writes the byte table unconditionally; never run the row kernel
      // without one (same guard as the ShaCompress control kernel).
      if (byte_trace != nullptr) {
        const int block_size = 256;
        const int grid_size = (events_count + block_size - 1) / block_size;
        control_events_to_rows_kernel<<<grid_size, block_size, 0, stream>>>(
          events, trace, byte_trace, events_count, num_cols);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::keccak

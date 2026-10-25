// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "tracegen/precompiles/keccak_consts.cuh"
#include "tracegen/precompiles/keccak_round.cuh"

namespace pico_gpu::keccak {

  /// Populate the 24 trace rows of one Keccak permutation event: the p3 rows for `pre_state`
  /// (without a preimage copy), tagged with chunk / clk / is_real. `export_in` is set on the first
  /// step and `export_flag` on the final step: those rows carry the permutation's input (`a`) and
  /// output onto the KeccakPermute state bus (the p3 generator leaves both at 0). The memory
  /// accesses and address gadgets are populated by `keccak_control.cuh`; this AIR emits no byte
  /// lookups. Mirrors Rust `populate_chunk`.
  template <class F>
  __device__ inline void populate_chunk_gpu(
    const KeccakPermuteEvent& ev,
    KeccakMemCols<F>* rows,
    KeccakCols<F>** d_keccak_ptrs,
    const size_t row0)
  {
    // Initialize state from event.
    uint64_t state[25];
    for (int i = 0; i < 25; ++i)
      state[i] = ev.pre_state[i];

    // Set up pointers for all rounds.
    for (int i = 0; i < NUM_ROUNDS; ++i) {
      d_keccak_ptrs[row0 + i] = &rows[i].keccak;
    }

    // Generate trace rows for full permutation.
    generate_trace_rows_for_perm(&d_keccak_ptrs[row0], NUM_ROUNDS, state);

    const F f_chunk = F::from_canonical_u32(ev.chunk);
    const F f_clk = F::from_canonical_u32(ev.clk);
    const F f_one = F::one();
    for (int r = 0; r < NUM_ROUNDS; ++r) {
      rows[r].chunk = f_chunk;
      rows[r].clk = f_clk;
      rows[r].is_real = f_one;
    }
    rows[0].export_in = f_one;
    rows[NUM_ROUNDS - 1].keccak.export_flag = f_one;
  }

  /// Populate dummy padding rows with zero state.
  template <class F>
  __device__ inline void populate_dummy_chunk_gpu(
    KeccakMemCols<F>* rows, KeccakCols<F>** d_keccak_ptrs, const size_t row0, const size_t rows_total)
  {
    uint64_t state[25] = {0};

    // Set up pointers only for valid rows.
    int valid_rows = 0;
    for (int i = 0; i < NUM_ROUNDS; ++i) {
      const std::size_t current_row = row0 + i;
      if (current_row < rows_total) {
        d_keccak_ptrs[current_row] = &rows[i].keccak;
        valid_rows++;
      } else {
        break;
      }
    }

    generate_trace_rows_for_perm(&d_keccak_ptrs[row0], valid_rows, state);
  }
} // namespace pico_gpu::keccak
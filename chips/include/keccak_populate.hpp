#pragma once

#include "keccak_consts.hpp"
#include "keccak_round.hpp"
#include "memory_read_write.hpp"
#include "byte.hpp"

namespace pico_gpu::keccak {
  using namespace memory_read_write;

  /// Populate trace rows for a Keccak permutation event.
  /// Generates NUM_ROUNDS (24) rows per event with memory read/write records.
  template <class F>
  __device__ inline void populate_chunk_gpu(
    const KeccakPermuteEvent& ev,
    KeccakMemCols<F>* rows,
    KeccakCols<F>** d_keccak_ptrs,
    const size_t row0,
    F* byte_trace)
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

    // Populate bookkeeping fields for all rows.
    const F f_chunk = F::from_canonical_u32(ev.chunk);
    const F f_clk = F::from_canonical_u32(ev.clk);
    const F f_state_addr = F::from_canonical_u32(ev.state_addr);
    const F f_one = F::one();

    for (int r = 0; r < NUM_ROUNDS; ++r) {
      rows[r].chunk = f_chunk;
      rows[r].clk = f_clk;
      rows[r].state_addr = f_state_addr;
      rows[r].is_real = f_one;

      // First row: populate memory reads.
      if (r == 0) {
        for (int j = 0; j < STATE_NUM_WORDS; ++j) {
          populate_read(rows[r].state_mem[j], ev.state_read_records[j], byte_trace);

          const array_t<uint8_t, WORD_SIZE> read_record_value_bytes = u32_to_le_bytes(ev.state_read_records[j].value);
          add_u8_range_checks(byte_trace, read_record_value_bytes);
        }
        rows[r].do_memory_check = f_one;
        rows[r].receive_ecall = f_one;
      }

      // Last row: populate memory writes.
      if (r == NUM_ROUNDS - 1) {
        for (int j = 0; j < STATE_NUM_WORDS; ++j) {
          populate_write(rows[r].state_mem[j], ev.state_write_records[j], byte_trace);

          const array_t<uint8_t, WORD_SIZE> write_record_value_bytes = u32_to_le_bytes(ev.state_write_records[j].value);
          add_u8_range_checks(byte_trace, write_record_value_bytes);
        }
        rows[r].do_memory_check = f_one;
      }
    }
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
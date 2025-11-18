#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <util/rusterror.h>
#include "is_equal_word.hpp"
#include "is_zero_word.hpp"
#include "prelude.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::divrem {
  using namespace pico_gpu::byte;
  static constexpr size_t NUM_MUL_EVENTS_PER_DIVREM = 2;

  /// Initialize padding row with dummy division (0 / 1 = 0).
  template <class F>
  __device__ inline void assign_padding_row(DivRemCols<F>& cols)
  {
    cols.is_divu = F::one();
    write_word_from_u32_v2<F>(cols.c, 1);
    write_word_from_u32_v2<F>(cols.abs_c, 1);
    write_word_from_u32_v2<F>(cols.max_abs_c_or_1, 1);

    is_zero_word::populate(cols.is_c_0, 1);
  }

  /// Convert division/remainder event to trace row.
  /// Handles DIVU, REMU, DIV, REM operations with proper sign handling.
  template <class F>
  __device__ inline void event_to_row(
    const AluEvent& event,
    DivRemCols<F>& cols,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    AluEvent* alu_mul_events,
    AluEvent* alu_lt_events)
  {
    assert(
      event.opcode == Opcode::DIVU || event.opcode == Opcode::REMU || event.opcode == Opcode::REM ||
      event.opcode == Opcode::DIV);

    // Populate basic operands and operation flags.
    {
      write_word_from_u32_v2<F>(cols.a, event.a);
      write_word_from_u32_v2<F>(cols.b, event.b);
      write_word_from_u32_v2<F>(cols.c, event.c);
      cols.is_real = F::one();
      cols.is_divu = F::from_bool(event.opcode == Opcode::DIVU);
      cols.is_remu = F::from_bool(event.opcode == Opcode::REMU);
      cols.is_div = F::from_bool(event.opcode == Opcode::DIV);
      cols.is_rem = F::from_bool(event.opcode == Opcode::REM);
      is_zero_word::populate(cols.is_c_0, event.c);
    }

    const uint32_t quotient = get_quotient(event.b, event.c, event.opcode);
    const uint32_t remainder = get_remainder(event.b, event.c, event.opcode);
    write_word_from_u32_v2<F>(cols.quotient, quotient);
    write_word_from_u32_v2<F>(cols.remainder, remainder);

    // Calculate sign detection flags and absolute values.
    {
      cols.rem_msb = F::from_canonical_u8(get_msb(u32_to_le_bytes(remainder)));
      cols.b_msb = F::from_canonical_u8(get_msb(u32_to_le_bytes(event.b)));
      cols.c_msb = F::from_canonical_u8(get_msb(u32_to_le_bytes(event.c)));

      is_equal_word::populate(cols.is_overflow_b, event.b, (uint32_t)INT32_MIN);
      is_equal_word::populate(cols.is_overflow_c, event.c, (uint32_t)(int32_t)-1);

      if (is_signed_operation(event.opcode)) {
        cols.rem_neg = cols.rem_msb;
        cols.b_neg = cols.b_msb;
        cols.c_neg = cols.c_msb;
        cols.is_overflow = F::from_bool((int32_t)event.b == INT32_MIN && (int32_t)event.c == -1);
        write_word_from_u32_v2<F>(cols.abs_remainder, abs(int32_t(remainder)));
        const uint32_t abs_c = abs(int32_t(event.c));
        write_word_from_u32_v2<F>(cols.abs_c, abs_c);
        write_word_from_u32_v2<F>(cols.max_abs_c_or_1, abs_c > 1 ? abs_c : 1);
      } else {
        cols.abs_remainder = cols.remainder;
        cols.abs_c = cols.c;
        write_word_from_u32_v2<F>(cols.max_abs_c_or_1, event.c > 1 ? event.c : 1);
      }

      // Set ALU event flags for absolute value computations.
      cols.abs_c_alu_event = cols.c_neg * cols.is_real;
      cols.abs_rem_alu_event = cols.rem_neg * cols.is_real;

      // Insert MSB lookup events for sign detection.
      {
        const uint32_t words[3] = {event.b, event.c, remainder};
        for (const uint32_t word : words) {
          const array_t<uint8_t, WORD_SIZE> word_bytes = u32_to_le_bytes(word);
          const uint8_t most_significant_byte = word_bytes[WORD_SIZE - 1];
          handle_byte_lookup_event<F>(byte_trace, ByteOpcode::MSB, most_significant_byte, 0);
        }
      }
    }

    // Calculate remainder check multiplicity.
    {
      cols.remainder_check_multiplicity = cols.is_real * (F::one() - cols.is_c_0.result);
    }

    // Verify division: c * quotient + remainder = b.
    {
      array_t<uint8_t, LONG_WORD_SIZE> c_times_quotient;
      if (is_signed_operation(event.opcode)) {
        c_times_quotient = u64_to_le_bytes((int64_t)(int32_t)quotient * (int64_t)(int32_t)event.c);
      } else {
        c_times_quotient = u64_to_le_bytes((uint64_t)quotient * (uint64_t)event.c);
      }

      array_t<uint8_t, LONG_WORD_SIZE> remainder_bytes;
      if (is_signed_operation(event.opcode)) {
        remainder_bytes = u64_to_le_bytes((int64_t)(int32_t)remainder);
      } else {
        remainder_bytes = u64_to_le_bytes((uint64_t)remainder);
      }

      // Add remainder to product with carry propagation.
      uint32_t carry[8] = {0};
      const uint32_t base = 1 << BYTE_SIZE;
      for (uintptr_t i = 0; i < LONG_WORD_SIZE; ++i) {
        cols.c_times_quotient[i] = F::from_canonical_u8(c_times_quotient[i]);

        uint32_t x = (uint32_t)c_times_quotient[i] + (uint32_t)remainder_bytes[i];
        if (i > 0) x += carry[i - 1];
        carry[i] = x / base;
        cols.carry[i] = F::from_canonical_u32(carry[i]);
      }

      // Generate dependent ALU events for verification.
      {
        // Insert absolute value computation events.
        {
          size_t add_sub_index = 0;
          if (cols.abs_c_alu_event == F::one()) {
            alu_add_sub_events[add_sub_index] = {
              .pc = event.pc,
              .opcode = Opcode::ADD,
              .a = 0,
              .b = event.c,
              .c = static_cast<uint32_t>(abs(static_cast<int32_t>(event.c))),
            };
            add_sub_index++;
          }
          if (cols.abs_rem_alu_event == F::one()) {
            alu_add_sub_events[add_sub_index] = {
              .pc = event.pc,
              .opcode = Opcode::ADD,
              .a = 0,
              .b = remainder,
              .c = static_cast<uint32_t>(abs(static_cast<int32_t>(remainder))),
            };
            add_sub_index++;
          }
        }

        // Extract lower and upper words from product.
        uint32_t lower_word = 0;
        for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
          lower_word += (c_times_quotient[i] << (i * BYTE_SIZE));
        }

        uint32_t upper_word = 0;
        for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
          upper_word += (c_times_quotient[WORD_SIZE + i] << (i * BYTE_SIZE));
        }

        // Insert multiplication events for verification.
        alu_mul_events[0] = {
          .pc = event.pc,
          .opcode = Opcode::MUL,
          .a = lower_word,
          .b = quotient,
          .c = event.c,
        };

        alu_mul_events[1] = {
          .pc = event.pc,
          .opcode = is_signed_operation(event.opcode) ? Opcode::MULH : Opcode::MULHU,
          .a = upper_word,
          .b = quotient,
          .c = event.c,
        };

        // Insert comparison event: remainder < abs(divisor).
        if (cols.remainder_check_multiplicity == F::one()) {
          uint32_t lt_b, lt_c;
          if (is_signed_operation(event.opcode)) {
            lt_b = abs(int32_t(remainder));
            lt_c = event.c > 1 ? abs(int32_t(event.c)) : 1;
          } else {
            lt_b = remainder;
            lt_c = event.c > 1 ? event.c : 1;
          }

          alu_lt_events[0] = {
            .pc = event.pc,
            .opcode = Opcode::SLTU,
            .a = 1,
            .b = lt_b,
            .c = lt_c,
          };
        }
      }

      // Add byte range checks for all values.
      {
        const array_t<uint8_t, WORD_SIZE> quotient_bytes = u32_to_le_bytes(quotient);
        add_u8_range_checks<F>(byte_trace, &quotient_bytes[0], WORD_SIZE);

        const array_t<uint8_t, WORD_SIZE> remainder_bytes = u32_to_le_bytes(remainder);
        add_u8_range_checks<F>(byte_trace, &remainder_bytes[0], WORD_SIZE);

        add_u8_range_checks<F>(byte_trace, &c_times_quotient[0], LONG_WORD_SIZE);
      }
    }
  }

  /// Kernel to convert division/remainder events to trace.
  template <class F>
  __global__ void events_to_trace_kernel(
    const size_t event_size,
    const size_t trace_size,
    const AluEvent* events,
    F* trace_matrix,
    const size_t num_cols,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    AluEvent* alu_mul_events,
    const size_t alu_mul_events_start_index,
    AluEvent* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * num_cols < trace_size) {
      DivRemCols<F>* cols = reinterpret_cast<DivRemCols<F>*>(&trace_matrix[idx * num_cols]);

      if (idx < event_size) {
        event_to_row<F>(
          events[idx], *cols, byte_trace,
          &alu_add_sub_events[alu_add_sub_events_start_index + extra_event_indices[idx].add],
          &alu_mul_events[alu_mul_events_start_index + (NUM_MUL_EVENTS_PER_DIVREM * idx)],
          &alu_lt_events[alu_lt_events_start_index + extra_event_indices[idx].lt]);
      } else {
        assign_padding_row<F>(*cols);
      }
    }
  }

  /// Generate trace from division/remainder events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const AluEvent* events,
    F* trace,
    cudaStream_t stream,
    const DivremExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    AluEvent* alu_mul_events,
    const size_t alu_mul_events_start_index,
    AluEvent* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const int block_size = 256;
      const int num_rows = trace_size / num_cols;
      const int grid_size = (num_rows + block_size - 1) / block_size;

      events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        event_size, trace_size, events, trace, num_cols, extra_event_indices, byte_trace, alu_add_sub_events,
        alu_add_sub_events_start_index, alu_mul_events, alu_mul_events_start_index, alu_lt_events,
        alu_lt_events_start_index);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::divrem
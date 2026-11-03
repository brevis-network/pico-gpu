// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cub/cub.cuh>
#include "field/ff_config.hpp"
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/common/is_zero.cuh"
#include "tracegen/common/field_word_range_checker.cuh"

namespace pico_gpu::ecall_u64 {

  // Write a Word<F> from a u64 as four 16-bit limbs (mirrors Rust Word::from(u64)).
  template <class F>
  __device__ inline void write_word_from_u64(Word<F>& word, const uint64_t v)
  {
    word._0[0] = F::from_canonical_u16(static_cast<uint16_t>(v & 0xFFFF));
    word._0[1] = F::from_canonical_u16(static_cast<uint16_t>((v >> 16) & 0xFFFF));
    word._0[2] = F::from_canonical_u16(static_cast<uint16_t>((v >> 32) & 0xFFFF));
    word._0[3] = F::from_canonical_u16(static_cast<uint16_t>((v >> 48) & 0xFFFF));
  }

  // The previous value of op_a's memory access - the syscall code on ECALL rows:
  // a read keeps its value, a write exposes its prev_value, an absent record is 0.
  __device__ inline uint64_t op_a_prev_value(const FfiCpuEventU64& event)
  {
    if (event.a_record.tag != FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      return 0;
    }
    const MemoryRecordEnumU64& r = event.a_record.some._0;
    if (r.tag == MemoryRecordEnumU64::Tag::Read) {
      return r.read._0.value;
    }
    return r.write._0.prev_value;
  }

  // The committed value of op_a: the memory record's value overwrites the event value
  // when a record exists.
  __device__ inline uint64_t op_a_committed_value(const FfiCpuEventU64& event)
  {
    if (event.a_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some) {
      return event.a_record.some._0.value();
    }
    return event.a;
  }

  // The committed value of op_b: a READ record's value overwrites the event value;
  // anything else keeps the event value.
  __device__ inline uint64_t op_b_committed_value(const FfiCpuEventU64& event)
  {
    if (event.b_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.b_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      return event.b_record.some._0.read._0.value;
    }
    return event.b;
  }

  // The committed value of op_c (see op_b_committed_value).
  __device__ inline uint64_t op_c_committed_value(const FfiCpuEventU64& event)
  {
    if (event.c_record.tag == FfiOption<MemoryRecordEnumU64>::Tag::Some &&
        event.c_record.some._0.tag == MemoryRecordEnumU64::Tag::Read) {
      return event.c_record.some._0.read._0.value;
    }
    return event.c;
  }

  template <class F>
  __device__ void event_to_row(
    const FfiCpuEventU64& event, EcallChipColsU64<F>& cols, F* byte_trace)
  {
    cols.chunk = F::from_canonical_u32(event.chunk);
    cols.clk = F::from_canonical_u32(static_cast<uint32_t>(event.clk));

    const uint64_t syscall_code_u64 = op_a_prev_value(event);
    const uint64_t op_b_value = op_b_committed_value(event);
    const uint64_t op_c_value = op_c_committed_value(event);

    write_word_from_u64<F>(cols.op_a_prev_value, syscall_code_u64);
    write_word_from_u64<F>(cols.op_a_value, op_a_committed_value(event));
    write_word_from_u64<F>(cols.op_b_value, op_b_value);
    write_word_from_u64<F>(cols.op_c_value, op_c_value);

    // Populate the u16 -> u8 decomposition gadget for the syscall code word and emit
    // the U16ToU8 u8-range byte lookups (4 low/high pairs from the 4 u16 limbs).
    for (int i = 0; i < WORD_SIZE; i++) {
      const uint32_t limb = static_cast<uint32_t>((syscall_code_u64 >> (16 * i)) & 0xFFFF);
      const uint8_t low_byte = static_cast<uint8_t>(limb & 0xFF);
      const uint8_t high_byte = static_cast<uint8_t>((limb >> 8) & 0xFF);
      cols.ecall.syscall_code_bytes.low_bytes[i] = F::from_canonical_u8(low_byte);
      if (byte_trace != nullptr) {
        byte::add_u8_range_check<F>(byte_trace, low_byte, high_byte);
      }
    }

    const uint8_t syscall_id_byte = static_cast<uint8_t>(syscall_code_u64 & 0xFF);
    const uint8_t send_to_table_byte = static_cast<uint8_t>((syscall_code_u64 >> 8) & 0xFF);
    const uint8_t num_extra_clk_byte = static_cast<uint8_t>((syscall_code_u64 >> 16) & 0xFF);
    const F syscall_id = F::from_canonical_u8(syscall_id_byte);

    cols.num_extra_clk = F::from_canonical_u8(num_extra_clk_byte);
    // The BitRange(NUM_EXTRA_CLK_BITS) cap the AIR looks up on num_extra_clk (mirrors Rust
    // riscv_ecall/traces.rs extra_record): one event per ecall row.
    if (byte_trace != nullptr) {
      byte::add_bit_range_check<F>(
        byte_trace, static_cast<uint16_t>(num_extra_clk_byte), NUM_EXTRA_CLK_BITS);
    }
    cols.send_to_table = F::from_canonical_u8(send_to_table_byte);

    // Populate the IsZero decode gadgets.
    is_zero::populate_from_field_element<F>(
      cols.ecall.is_enter_unconstrained,
      syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::ENTER_UNCONSTRAINED)));
    is_zero::populate_from_field_element<F>(
      cols.ecall.is_hint_len,
      syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::HINT_LEN)));
    // The HINT_LEN return-value range-check gate and its four U16Range events (mirrors Rust
    // riscv_ecall/traces.rs: event_to_row sets the gate, extra_record emits the events). HINT_LEN
    // is the one ecall whose written register word is pinned to nothing else, and the Cpu chip no
    // longer range-checks every written word.
    const bool is_hint_len =
      syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::HINT_LEN));
    cols.hint_len_mul = F::from_bool(is_hint_len);
    if (is_hint_len && byte_trace != nullptr) {
      const uint64_t op_a_value = op_a_committed_value(event);
      for (int i = 0; i < WORD_SIZE; i++) {
        byte::add_u16_range_check<F>(
          byte_trace, static_cast<uint16_t>((op_a_value >> (16 * i)) & 0xFFFF));
      }
    }
    is_zero::populate_from_field_element<F>(
      cols.ecall.is_halt,
      syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::HALT)));
    is_zero::populate_from_field_element<F>(
      cols.ecall.is_commit,
      syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT)));
    is_zero::populate_from_field_element<F>(
      cols.ecall.is_commit_deferred_proofs,
      syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS)));

    // If the syscall is COMMIT or COMMIT_DEFERRED_PROOFS, set the index bitmap.
    if (syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::COMMIT)) ||
        syscall_id_byte ==
          static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS))) {
      const uint32_t digest_idx = static_cast<uint32_t>(op_b_value & 0xFFFFFFFF);
      // The Rust twin panics on digest_idx >= PV_DIGEST_NUM_WORDS (malformed guest input);
      // in device code guard the write instead of corrupting neighboring columns.
      if (digest_idx < PV_DIGEST_NUM_WORDS) {
        cols.ecall.index_bitmap[digest_idx] = F::one();
      }
    }

    // Mirror the selected public-value digest bytes into the main trace and emit the two u8
    // range checks the AIR looks up for them. Mirrors Rust riscv_ecall/traces.rs::event_to_row.
    //
    // On an honest trace these are the little-endian bytes of the u32 the guest passed in a1:
    // CommitSyscall stores decode_u32_abi_word(arg2) into
    // public_values.committed_value_digest[word_idx], PublicValues::from splits that u32 into 4
    // LE bytes, and arg2 is op_c -- which the fat message delivers here as op_c_value. The
    // 32-bit truncation is required, not incidental: the guest ABI allows a sign-extended
    // 32-bit word in the RV64 register.
    //
    // COMMIT only, unlike the index-bitmap branch above which also fires for
    // COMMIT_DEFERRED_PROOFS: the AIR constrains commit_mul == is_real * is_commit
    // unconditionally, so a stray 1 on a COMMIT_DEFERRED_PROOFS row fails the debug check.
    //
    // Multiplicity: slice_range_check_u8 chunks the 4 limbs in pairs => exactly 2 U8Range
    // events per COMMIT row. On the host path they come from extra_record; this kernel is the
    // sole producer on the GPU path, so they are emitted here.
    if (syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::COMMIT))) {
      const uint32_t digest_u32 = static_cast<uint32_t>(op_c_value & 0xFFFFFFFF);
      uint8_t digest_bytes[WORD_SIZE];
      for (int i = 0; i < WORD_SIZE; i++) {
        digest_bytes[i] = static_cast<uint8_t>((digest_u32 >> (8 * i)) & 0xFF);
        cols.commit_digest_bytes._0[i] = F::from_canonical_u8(digest_bytes[i]);
      }
      cols.commit_mul = F::one();
      if (byte_trace != nullptr) {
        byte::add_u8_range_check<F>(byte_trace, digest_bytes[0], digest_bytes[1]);
        byte::add_u8_range_check<F>(byte_trace, digest_bytes[2], digest_bytes[3]);
      }
    }

    const bool is_halt =
      syscall_id_byte == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::HALT));
    cols.is_halt = F::from_bool(is_halt);

    // For halt and commit_deferred_proofs syscalls, range check one of the operands.
    if (is_halt) {
      write_word_from_u64<F>(cols.ecall.operand_to_check, op_b_value);
      field_word_range_checker::populate<F>(
        cols.ecall.operand_range_check_cols, byte_trace, op_b_value);
      cols.range_check_operand = F::one();
    }
    if (syscall_id_byte ==
        static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS))) {
      // This row's own share of the tally; the tally scan turns the two into running sums.
      // The mask share is the index bitmap's weight, so an out-of-range word index adds 0,
      // matching the host trace.
      const uint32_t digest_idx = static_cast<uint32_t>(op_b_value & 0xFFFFFFFF);
      cols.deferred_commit_mul = F::one();
      cols.deferred_commits = F::one();
      cols.deferred_commit_mask =
        digest_idx < PV_DIGEST_NUM_WORDS ? F::from_canonical_u32(1u << digest_idx) : F::zero();
      write_word_from_u64<F>(cols.ecall.operand_to_check, op_c_value);
      field_word_range_checker::populate<F>(
        cols.ecall.operand_range_check_cols, byte_trace, op_c_value);
      cols.range_check_operand = F::one();
    }

    cols.is_real = F::one();
  }

  /// The COMMIT_DEFERRED_PROOFS tally of one row: (count, mask).
  template <class F>
  struct DeferredTally {
    F count;
    F mask;
  };

  template <class F>
  struct DeferredTallyAdd {
    __device__ __forceinline__ DeferredTally<F>
    operator()(const DeferredTally<F>& a, const DeferredTally<F>& b) const
    {
      return DeferredTally<F>{a.count + b.count, a.mask + b.mask};
    }
  };

  constexpr int DEFERRED_TALLY_BLOCK = 256;
} // namespace pico_gpu::ecall_u64

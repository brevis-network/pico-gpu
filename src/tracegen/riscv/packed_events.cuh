// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Packed FFI event formats (MLE_PACKED_EVENTS=1) — struct mirrors + unpack helpers.
//
// Field-for-field mirror of pico's vm/src/instances/preparer/packed_events.rs. The Rust side
// verifies (per event, with fat-path fallback) every redundancy this format drops:
//   record.chunk == event.chunk          -> chunk becomes a kernel scalar argument
//   record.timestamp == clk + {3,2,1,0}  -> timestamps are recomputed on device
//   clk/pc/next_pc fit u32, op_b/op_c fit i32 (sign-extended)
//   memory / exit_code                   -> read by no kernel, dropped
//
// One field per operand, carrying the value the kernels commit: where an operand has a read
// record, a/b/c hold that RECORD's value, not the raw event operand. That is what the fat path
// ends up writing too (populate_memory_access_u64 overwrites the access column with it, and
// ecall.cuh prefers it), and the two differ in practice — a syscall whose precompile reuses the B
// slot leaves event.b holding the original operand. Some rows read the raw operand as well: branch
// compares and jump targets here, and the memory address (b + c) in memory_read_write. One field
// cannot hold both, so on those rows the Rust side requires the two to be equal.
//
// Every *_packed FFI entry — they live in pico's gpu-api/cuda/chips_group{1,2}.cu, not in
// this repo — validates PICO_PACKED_EVENT_FORMAT_V1, so an unpaired pico / pico-gpu
// deploy fails loudly instead of misreading bytes. The constant is duplicated by hand on both
// sides and covers layout only: it says nothing about whether the two agree on semantics.

#pragma once

#include <cstdint>
#include <cstddef>
#include "tracegen/common/types.cuh"

namespace pico_gpu {

  // `..02`: LtEventPacked byte 25 became `is_cmp` (merged branch comparison).
  constexpr size_t PICO_PACKED_EVENT_FORMAT_V1 = 0x50E1B302;

  // Flag bits in FfiCpuEventPacked::flags (mirror of the Rust PK_* consts).
  constexpr uint16_t PK_IMM_B = 1u << 0;
  constexpr uint16_t PK_IMM_C = 1u << 1;
  constexpr uint16_t PK_A_SOME = 1u << 2;
  constexpr uint16_t PK_A_WRITE = 1u << 3;
  constexpr uint16_t PK_B_SOME = 1u << 4;
  constexpr uint16_t PK_C_SOME = 1u << 5;
  constexpr uint16_t PK_M_SOME = 1u << 6;
  constexpr uint16_t PK_M_WRITE = 1u << 7;

  // Sentinel opcode byte used to detect which scratch ALU slots an unmodified event_to_row
  // actually wrote (valid opcodes are <= 52).
  constexpr uint8_t PACKED_ALU_SENTINEL = 0xFF;

  /// 80-byte packed CpuEvent (Rust CpuEventPacked).
  struct FfiCpuEventPacked {
    uint32_t clk;
    uint32_t pc;
    uint32_t next_pc;
    uint8_t opcode;
    uint8_t op_a;
    uint16_t flags;
    int32_t op_b;
    int32_t op_c;
    uint64_t a; // committed op_a value (= a_record.value() when a record exists)
    uint64_t b;
    uint64_t c;
    uint64_t a_prev_value; // writes: previous value; reads: == a
    uint32_t a_prev_chunk;
    uint32_t a_prev_ts;
    uint32_t b_prev_chunk;
    uint32_t b_prev_ts;
    uint32_t c_prev_chunk;
    uint32_t c_prev_ts;
  };

  /// 24-byte packed memory_record side-stream entry (dense, ordered exactly like the
  /// memory_read_write_event_indices array; entry i belongs to dense memrw row i).
  struct FfiMemRecordPacked {
    uint64_t value;
    uint64_t prev_value; // reads: == value
    uint32_t prev_chunk;
    uint32_t prev_ts;
  };

  /// 24-byte packed AluEvent for the add_sub family (kernels read only opcode/b/c).
  struct AddSubEventPacked {
    uint64_t b;
    uint64_t c;
    uint8_t opcode;
    uint8_t _pad[7];
  };

  /// 32-byte packed AluEvent for the lt family (kernel reads opcode/a/b/c/is_cmp).
  struct LtEventPacked {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint8_t opcode;
    uint8_t is_cmp;
    uint8_t _pad[6];
  };

  static_assert(sizeof(FfiCpuEventPacked) == 80, "ABI drift vs Rust CpuEventPacked");
  static_assert(sizeof(FfiMemRecordPacked) == 24, "ABI drift vs Rust MemRecordPacked");
  static_assert(sizeof(AddSubEventPacked) == 24, "ABI drift vs Rust AddSubEventPacked");
  static_assert(sizeof(LtEventPacked) == 32, "ABI drift vs Rust LtEventPacked");
  // Every field, not a sample: a size check cannot see two same-width fields swapped (b/c, or
  // opcode/op_a), and such a reorder would pass the format-version handshake and show up only as a
  // failed proof. The Rust mirror carries the same list, so a reorder on either side is caught.
  static_assert(offsetof(FfiCpuEventPacked, clk) == 0, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, pc) == 4, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, next_pc) == 8, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, opcode) == 12, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, op_a) == 13, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, flags) == 14, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, op_b) == 16, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, op_c) == 20, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, a) == 24, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, b) == 32, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, c) == 40, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, a_prev_value) == 48, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, a_prev_chunk) == 56, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, a_prev_ts) == 60, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, b_prev_chunk) == 64, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, b_prev_ts) == 68, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, c_prev_chunk) == 72, "ABI drift");
  static_assert(offsetof(FfiCpuEventPacked, c_prev_ts) == 76, "ABI drift");
  static_assert(offsetof(FfiMemRecordPacked, value) == 0, "ABI drift");
  static_assert(offsetof(FfiMemRecordPacked, prev_value) == 8, "ABI drift");
  static_assert(offsetof(FfiMemRecordPacked, prev_chunk) == 16, "ABI drift");
  static_assert(offsetof(FfiMemRecordPacked, prev_ts) == 20, "ABI drift");
  static_assert(offsetof(AddSubEventPacked, b) == 0, "ABI drift");
  static_assert(offsetof(AddSubEventPacked, c) == 8, "ABI drift");
  static_assert(offsetof(AddSubEventPacked, opcode) == 16, "ABI drift");
  static_assert(offsetof(LtEventPacked, a) == 0, "ABI drift");
  static_assert(offsetof(LtEventPacked, b) == 8, "ABI drift");
  static_assert(offsetof(LtEventPacked, c) == 16, "ABI drift");
  static_assert(offsetof(LtEventPacked, opcode) == 24, "ABI drift");
  static_assert(offsetof(LtEventPacked, is_cmp) == 25, "ABI drift");
  static_assert(sizeof(FfiCpuEventU64) == 304, "fat CpuEvent ABI drift");
  static_assert(sizeof(AluEventU64) == 48, "fat AluEvent ABI drift");
  static_assert(offsetof(AluEventU64, is_imm) == 40, "fat AluEvent ABI drift");
  static_assert(offsetof(AluEventU64, is_cmp) == 41, "fat AluEvent ABI drift");

  /// Reconstruct the fat FfiCpuEventU64 a packed event stands for, exactly, by the packing
  /// invariants listed above. Its one caller is the ecall kernel (packed_ecall.cuh), which hands
  /// the result to the unmodified ecall body; memory_record stays None because nothing that
  /// consumes this reconstruction reads it.
  __host__ __device__ inline FfiCpuEventU64 unpack_cpu_event(
    const FfiCpuEventPacked& p, const uint32_t chunk)
  {
    FfiCpuEventU64 e = {};
    e.chunk = chunk;
    e.clk = p.clk;
    e.pc = p.pc;
    e.next_pc = p.next_pc;
    e.instruction.opcode = static_cast<Opcode>(p.opcode);
    e.instruction.op_a = p.op_a;
    e.instruction.op_b = static_cast<uint64_t>(static_cast<int64_t>(p.op_b));
    e.instruction.op_c = static_cast<uint64_t>(static_cast<int64_t>(p.op_c));
    e.instruction.imm_b = (p.flags & PK_IMM_B) != 0;
    e.instruction.imm_c = (p.flags & PK_IMM_C) != 0;
    e.a = p.a;
    e.b = p.b;
    e.c = p.c;

    if (p.flags & PK_A_SOME) {
      e.a_record.tag = FfiOption<MemoryRecordEnumU64>::Tag::Some;
      auto& r = e.a_record.some._0;
      if (p.flags & PK_A_WRITE) {
        r.tag = MemoryRecordEnumU64::Tag::Write;
        r.write._0.value = p.a;
        r.write._0.chunk = chunk;
        r.write._0.timestamp = p.clk + 3;
        r.write._0.prev_value = p.a_prev_value;
        r.write._0.prev_chunk = p.a_prev_chunk;
        r.write._0.prev_timestamp = p.a_prev_ts;
      } else {
        r.tag = MemoryRecordEnumU64::Tag::Read;
        r.read._0.value = p.a;
        r.read._0.chunk = chunk;
        r.read._0.timestamp = p.clk + 3;
        r.read._0.prev_chunk = p.a_prev_chunk;
        r.read._0.prev_timestamp = p.a_prev_ts;
      }
    }
    if (p.flags & PK_B_SOME) {
      e.b_record.tag = FfiOption<MemoryRecordEnumU64>::Tag::Some;
      auto& r = e.b_record.some._0;
      r.tag = MemoryRecordEnumU64::Tag::Read;
      r.read._0.value = p.b;
      r.read._0.chunk = chunk;
      r.read._0.timestamp = p.clk + 2;
      r.read._0.prev_chunk = p.b_prev_chunk;
      r.read._0.prev_timestamp = p.b_prev_ts;
    }
    if (p.flags & PK_C_SOME) {
      e.c_record.tag = FfiOption<MemoryRecordEnumU64>::Tag::Some;
      auto& r = e.c_record.some._0;
      r.tag = MemoryRecordEnumU64::Tag::Read;
      r.read._0.value = p.c;
      r.read._0.chunk = chunk;
      r.read._0.timestamp = p.clk + 1;
      r.read._0.prev_chunk = p.c_prev_chunk;
      r.read._0.prev_timestamp = p.c_prev_ts;
    }
    // e.memory / e.memory_record stay None (memory is read by no kernel; the MemoryReadWrite
    // kernel takes memory_record from its own side stream, filling columns from it directly
    // rather than through this reconstruction). exit_code stays 0 (read by no kernel).
    return e;
  }

  /// Fat AluEventU64 for an add_sub-family packed event (clk/a/is_imm are read by no consumer
  /// of this buffer; zeros keep the reconstruction total).
  __host__ __device__ inline AluEventU64 unpack_add_sub(const AddSubEventPacked& p)
  {
    AluEventU64 e = {};
    e.opcode = static_cast<Opcode>(p.opcode);
    e.b = p.b;
    e.c = p.c;
    return e;
  }

  /// Fat AluEventU64 for an lt-family packed event (clk/is_imm are read by no consumer).
  __host__ __device__ inline AluEventU64 unpack_lt(const LtEventPacked& p)
  {
    AluEventU64 e = {};
    e.opcode = static_cast<Opcode>(p.opcode);
    e.a = p.a;
    e.b = p.b;
    e.c = p.c;
    e.is_cmp = p.is_cmp != 0;
    return e;
  }

  __host__ __device__ inline AddSubEventPacked pack_add_sub(const AluEventU64& e)
  {
    AddSubEventPacked p = {};
    p.b = e.b;
    p.c = e.c;
    p.opcode = static_cast<uint8_t>(e.opcode);
    return p;
  }

} // namespace pico_gpu

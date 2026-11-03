// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Every event stream the emulator writes, rebuilt from the execution history through history.cuh's
// loaders and compared element for element with the stream a dump holds
// (pico's vm/src/emulator/riscv/history_dump.rs). One row at a time, so the host's loader tests
// walk the rows in a loop and the device self-test runs one thread per row.
//
// The streams are the emulator's, before the augment pass appends to any; the comparison behind
// each branch, which the less-than chip proves after its own rows, is a stream of its own in
// execution order, the order of the branch list. The add and sub streams share one buffer in step
// order, so their rows are the merge of the two lists.
#pragma once

#include "tracegen/riscv/history.cuh"

namespace pico_gpu::history_check {

  // A memory record's canonical row in a dump; `kind` is 0 for none, 1 for a read, 2 for a write,
  // and a read's `prev_value` is 0.
  struct RecordRow {
    uint32_t kind;
    uint32_t chunk;
    uint32_t timestamp;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
    uint32_t zero;
    uint64_t value;
    uint64_t prev_value;
  };
  static_assert(sizeof(RecordRow) == 40, "history_dump.rs RECORD_ROW");
  static_assert(offsetof(RecordRow, prev_timestamp) == 16, "history_dump.rs RECORD_ROW");
  static_assert(offsetof(RecordRow, value) == 24, "history_dump.rs RECORD_ROW");
  static_assert(offsetof(RecordRow, prev_value) == 32, "history_dump.rs RECORD_ROW");

  // A CPU event's canonical row; its records are `a`, `b`, `c`, then memory.
  struct CpuRow {
    uint32_t chunk;
    uint32_t exit_code;
    uint64_t clk;
    uint64_t pc;
    uint64_t next_pc;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint8_t opcode;
    uint8_t op_a;
    uint8_t imm_b;
    uint8_t imm_c;
    uint32_t zero;
    uint64_t op_b;
    uint64_t op_c;
    RecordRow records[4];
    uint32_t memory_some;
    uint32_t zero2;
    uint64_t memory;
  };
  static_assert(sizeof(CpuRow) == 256, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, exit_code) == 4, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, clk) == 8, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, c) == 48, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, opcode) == 56, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, imm_c) == 59, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, op_b) == 64, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, op_c) == 72, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, records) == 80, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, memory_some) == 240, "history_dump.rs CPU_ROW");
  static_assert(offsetof(CpuRow, memory) == 248, "history_dump.rs CPU_ROW");

  // An ALU event's canonical row.
  struct AluRow {
    uint64_t clk;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint8_t opcode;
    uint8_t is_imm;
    uint8_t is_cmp;
    uint8_t zero[5];
  };
  static_assert(sizeof(AluRow) == 40, "history_dump.rs ALU_ROW");
  static_assert(offsetof(AluRow, opcode) == 32, "history_dump.rs ALU_ROW");
  static_assert(offsetof(AluRow, is_cmp) == 34, "history_dump.rs ALU_ROW");

  enum Stream : uint32_t {
    STREAM_CPU = 0,
    STREAM_MEMORY = 1,
    STREAM_ADD_SUB = 2,
    STREAM_MUL = 3,
    STREAM_BITWISE = 4,
    STREAM_SHIFT_LEFT = 5,
    STREAM_SHIFT_RIGHT = 6,
    STREAM_DIVREM = 7,
    STREAM_LT = 8,
    STREAM_SYSCALL = 9,
    STREAM_BRANCH_COMPARE = 10,
    STREAMS = 11,
  };

  // The dump's rows of each stream: CpuRow, uint64_t indices, AluRow, or SyscallEventU64.
  struct Expected {
    const void* rows[STREAMS];
    uint64_t counts[STREAMS];
  };

  // The list each one-list ALU stream is rebuilt from.
  REPLAY_HD inline uint32_t list_of(uint32_t stream)
  {
    switch (stream) {
    case STREAM_MUL:
      return history::LIST_MUL;
    case STREAM_BITWISE:
      return history::LIST_BITWISE;
    case STREAM_SHIFT_LEFT:
      return history::LIST_SHIFT_LEFT;
    case STREAM_SHIFT_RIGHT:
      return history::LIST_SHIFT_RIGHT;
    case STREAM_DIVREM:
      return history::LIST_DIVREM;
    case STREAM_LT:
      return history::LIST_LT;
    default:
      return history::LISTS;
    }
  }

  // How many rows of `stream` the history rebuilds.
  REPLAY_HD inline uint64_t rebuilt_rows(const history::History& h, uint32_t stream)
  {
    switch (stream) {
    case STREAM_CPU:
      return h.step_count;
    case STREAM_MEMORY:
      return history::list_length(h, history::LIST_MEMORY);
    case STREAM_ADD_SUB:
      return uint64_t(history::list_length(h, history::LIST_ADD)) + history::list_length(h, history::LIST_SUB);
    case STREAM_SYSCALL:
      return h.syscall_count;
    case STREAM_BRANCH_COMPARE:
      return history::list_length(h, history::LIST_BRANCH);
    default:
      return history::list_length(h, list_of(stream));
    }
  }

  // Row `k` of the step-ordered merge of the lists `first` and `second`, which share no step: the
  // list it comes from and its row there. Of the merge's first k rows, the number from `first` is
  // the least i whose next row of `second` precedes row i of `first`, if either remains.
  REPLAY_HD inline bool merged_entry(
    const history::History& h, uint32_t first, uint32_t second, uint32_t k, uint32_t& list, uint32_t& row)
  {
    const uint32_t n = history::list_length(h, first), m = history::list_length(h, second);
    if (uint64_t(k) >= uint64_t(n) + m) {
      history::fail(h.error, history::ERROR_LIST);
      return false;
    }
    const uint32_t* a = h.entries + h.list_start[first];
    const uint32_t* b = h.entries + h.list_start[second];
    uint32_t low = k > m ? k - m : 0, high = k < n ? k : n;
    while (low < high) {
      const uint32_t i = low + (high - low) / 2;
      if (b[k - i - 1] < a[i]) {
        high = i;
      } else {
        low = i + 1;
      }
    }
    const uint32_t i = low, j = k - low;
    if (i < n && (j == m || a[i] < b[j])) {
      list = first;
      row = i;
    } else {
      list = second;
      row = j;
    }
    return true;
  }

  REPLAY_HD inline bool same_record(const FfiOption<MemoryRecordEnumU64>& option, const RecordRow& row)
  {
    if (option.tag == FfiOption<MemoryRecordEnumU64>::Tag::None) return row.kind == 0;
    const MemoryRecordEnumU64& record = option.some._0;
    if (record.tag == MemoryRecordEnumU64::Tag::Read) {
      const MemoryReadRecordU64& read = record.read._0;
      return row.kind == 1 && read.value == row.value && read.chunk == row.chunk &&
             read.timestamp == row.timestamp && read.prev_chunk == row.prev_chunk &&
             read.prev_timestamp == row.prev_timestamp;
    }
    const MemoryWriteRecordU64& write = record.write._0;
    return row.kind == 2 && write.value == row.value && write.chunk == row.chunk &&
           write.timestamp == row.timestamp && write.prev_value == row.prev_value &&
           write.prev_chunk == row.prev_chunk && write.prev_timestamp == row.prev_timestamp;
  }

  // The exit code is left out: no trace reads a CPU event's (the exit code public value is the
  // chunk boundary's), and the loaders write 0.
  REPLAY_HD inline bool same_cpu(const FfiCpuEventU64& event, const CpuRow& row)
  {
    const InstructionU64& instruction = event.instruction;
    const bool memory_some = event.memory.tag == FfiOption<uint64_t>::Tag::Some;
    return event.chunk == row.chunk && event.clk == row.clk &&
           event.pc == row.pc && event.next_pc == row.next_pc && event.a == row.a && event.b == row.b &&
           event.c == row.c && static_cast<uint8_t>(instruction.opcode) == row.opcode &&
           instruction.op_a == row.op_a && uint8_t(instruction.imm_b) == row.imm_b &&
           uint8_t(instruction.imm_c) == row.imm_c && instruction.op_b == row.op_b && instruction.op_c == row.op_c &&
           same_record(event.a_record, row.records[0]) && same_record(event.b_record, row.records[1]) &&
           same_record(event.c_record, row.records[2]) && same_record(event.memory_record, row.records[3]) &&
           uint32_t(memory_some) == row.memory_some && (!memory_some || event.memory.some._0 == row.memory);
  }

  REPLAY_HD inline bool same_alu(const AluEventU64& event, const AluRow& row)
  {
    return event.clk == row.clk && event.a == row.a && event.b == row.b && event.c == row.c &&
           static_cast<uint8_t>(event.opcode) == row.opcode && uint8_t(event.is_imm) == row.is_imm &&
           uint8_t(event.is_cmp) == row.is_cmp;
  }

  REPLAY_HD inline bool same_syscall(const SyscallEventU64& event, const SyscallEventU64& row)
  {
    return event.chunk == row.chunk && event.clk == row.clk && event.syscall_id == row.syscall_id &&
           event.arg1 == row.arg1 && event.arg2 == row.arg2;
  }

  // Whether row `row` of `stream`, rebuilt from the history, equals the dump's; the caller has
  // checked the row is inside both.
  template <history::ExceptionLookup Lookup = history::DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline bool row_matches(const history::History& h, const Expected& expected, uint32_t stream, uint32_t row)
  {
    const void* rows = expected.rows[stream];
    switch (stream) {
    case STREAM_CPU:
      return same_cpu(history::cpu_event<Lookup>(h, row), static_cast<const CpuRow*>(rows)[row]);
    case STREAM_MEMORY: {
      uint32_t step = 0;
      return history::entry_at(h, history::LIST_MEMORY, row, step) &&
             step == static_cast<const uint64_t*>(rows)[row];
    }
    case STREAM_ADD_SUB: {
      uint32_t list = 0, at = 0;
      return merged_entry(h, history::LIST_ADD, history::LIST_SUB, row, list, at) &&
             same_alu(history::alu_event<Lookup>(h, list, at), static_cast<const AluRow*>(rows)[row]);
    }
    case STREAM_SYSCALL:
      return same_syscall(history::syscall_event(h, row), static_cast<const SyscallEventU64*>(rows)[row]);
    case STREAM_BRANCH_COMPARE:
      return same_alu(
        history::compare_event<Lookup>(h, history::list_length(h, history::LIST_LT) + row),
        static_cast<const AluRow*>(rows)[row]);
    default:
      return same_alu(history::alu_event<Lookup>(h, list_of(stream), row), static_cast<const AluRow*>(rows)[row]);
    }
  }

} // namespace pico_gpu::history_check

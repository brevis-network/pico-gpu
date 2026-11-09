// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The loaders against corrupt fills: each case breaks one fact of an intact fill, places it in an
// allocation of exactly the bytes the launch names, so the sanitizers see any read past it, and
// requires the loaders to set exactly the error bit that fact is checked by. Exits 0 when every
// case does.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "tracegen/riscv/history_check.cuh"
#include "tracegen/riscv/history_cpu_memory_lt.cuh"
#include "tracegen/riscv/history_ecall_syscall.cuh"

using namespace pico_gpu;
using namespace pico_gpu::history;

namespace {

  constexpr uint64_t NONCE = 9;
  constexpr uint64_t PROGRAM_KEY = 42;
  constexpr uint64_t PC_BASE = 0x1000;

  InstructionU64 instruction(Opcode opcode, uint8_t a, uint64_t b, uint64_t c, bool imm_b, bool imm_c)
  {
    InstructionU64 row = {};
    row.opcode = opcode;
    row.op_a = a;
    row.op_b = b;
    row.op_c = c;
    row.imm_b = imm_b;
    row.imm_c = imm_c;
    return row;
  }

  // An ADD to x0 with its excepted `a`, a load, a store, a branch and an ECALL with a syscall
  // row whose `b` and `c` are excepted too, each step's rows as its instruction's slots make them.
  const std::vector<InstructionU64> PROGRAM = {
    instruction(Opcode::ADD, 0, 1, 2, false, false), instruction(Opcode::LD, 4, 1, 0x10, false, true),
    instruction(Opcode::SD, 2, 1, 8, false, true),   instruction(Opcode::BEQ, 1, 2, 8, false, true),
    instruction(Opcode::ECALL, 5, 10, 11, false, false),
  };

  // The intact fill, word-backed so every table is 8-aligned.
  std::vector<uint64_t> intact_fill()
  {
    std::vector<Step> steps;
    std::vector<ReadAccess> reads;
    std::vector<Access> writes;
    for (uint32_t index = 0; index < PROGRAM.size(); ++index) {
      steps.push_back(Step{8 + 4 * index, uint32_t(PC_BASE) + 4 * index, uint32_t(PC_BASE) + 4 * index + 4,
                           uint32_t(reads.size()), uint32_t(writes.size())});
      const Slots slots = slots_of(PROGRAM[index]);
      for (uint32_t position = POSITIONS; position-- > 0;) {
        if (!slots.contains(position)) continue;
        const uint64_t value = 100 * index + position;
        if (slots.writes(position)) {
          writes.push_back(Access{value, value + 1, 1, 3});
        } else {
          reads.push_back(ReadAccess{value, 1, 2});
        }
      }
    }
    const std::vector<OperandException> exceptions = {{0, SLOT_A, 77}, {4, SLOT_B, 5}, {4, SLOT_C, 6}};
    // add, memory, memory, ecall, branch, in list order.
    const std::vector<uint32_t> lists[LISTS] = {{}, {}, {}, {}, {}, {}, {}, {}, {1, 2}, {4}, {3}};
    const SyscallEventU64 syscall = {0, 0, 24, 0x10, 0, 5, 6};

    FillDescriptor d = {};
    d.layout = HISTORY_LAYOUT;
    d.descriptor_bytes = sizeof(FillDescriptor);
    d.identity = FillIdentity{2, uint32_t(steps.size()), NONCE};
    d.program_key = PROGRAM_KEY;
    d.reads = uint32_t(reads.size());
    d.writes = uint32_t(writes.size());
    d.exceptions = uint32_t(exceptions.size());
    d.syscalls = 1;
    for (uint32_t list = 0; list < LISTS; ++list)
      d.list_start[list + 1] = d.list_start[list] + uint32_t(lists[list].size());
    d.cpu_slice_steps = 2;
    uint64_t at = sizeof(FillDescriptor);
    auto place = [&](uint64_t& table_at, uint64_t bytes) {
      table_at = (at + 7) / 8 * 8;
      at = table_at + bytes;
    };
    place(d.exceptions_at, exceptions.size() * sizeof(OperandException));
    place(d.steps_at, steps.size() * sizeof(Step));
    place(d.reads_at, reads.size() * sizeof(ReadAccess));
    place(d.writes_at, writes.size() * sizeof(Access));
    place(d.entries_at, d.list_start[LISTS] * sizeof(uint32_t));
    place(d.syscalls_at, sizeof(SyscallEventU64));
    place(d.completion_at, sizeof(uint64_t));
    d.buffer_bytes = at;

    std::vector<uint64_t> words((at + 7) / 8, 0);
    uint8_t* bytes = reinterpret_cast<uint8_t*>(words.data());
    std::memcpy(bytes, &d, sizeof d);
    std::memcpy(bytes + d.exceptions_at, exceptions.data(), exceptions.size() * sizeof(OperandException));
    std::memcpy(bytes + d.steps_at, steps.data(), steps.size() * sizeof(Step));
    std::memcpy(bytes + d.reads_at, reads.data(), reads.size() * sizeof(ReadAccess));
    std::memcpy(bytes + d.writes_at, writes.data(), writes.size() * sizeof(Access));
    for (uint32_t list = 0; list < LISTS; ++list)
      if (!lists[list].empty())
        std::memcpy(bytes + d.entries_at + 4 * d.list_start[list], lists[list].data(), 4 * lists[list].size());
    std::memcpy(bytes + d.syscalls_at, &syscall, sizeof syscall);
    std::memcpy(bytes + d.completion_at, &NONCE, sizeof NONCE);
    return words;
  }

  FillDescriptor& descriptor_of(std::vector<uint64_t>& fill) { return *reinterpret_cast<FillDescriptor*>(fill.data()); }

  bool same_record(const FfiOption<MemoryRecordEnumU64>& x, const FfiOption<MemoryRecordEnumU64>& y)
  {
    if (x.tag != y.tag) return false;
    if (x.tag == FfiOption<MemoryRecordEnumU64>::Tag::None) return true;
    const MemoryRecordEnumU64 &r = x.some._0, &q = y.some._0;
    if (r.tag != q.tag) return false;
    if (r.tag == MemoryRecordEnumU64::Tag::Read) {
      const MemoryReadRecordU64 &a = r.read._0, &b = q.read._0;
      return a.value == b.value && a.chunk == b.chunk && a.timestamp == b.timestamp && a.prev_chunk == b.prev_chunk &&
             a.prev_timestamp == b.prev_timestamp;
    }
    const MemoryWriteRecordU64 &a = r.write._0, &b = q.write._0;
    return a.value == b.value && a.chunk == b.chunk && a.timestamp == b.timestamp && a.prev_value == b.prev_value &&
           a.prev_chunk == b.prev_chunk && a.prev_timestamp == b.prev_timestamp;
  }

  // Whether two CPU events agree on all but their memory record and memory value.
  bool same_cpu_registers(const FfiCpuEventU64& x, const FfiCpuEventU64& y)
  {
    const InstructionU64 &xi = x.instruction, &yi = y.instruction;
    return x.chunk == y.chunk && x.clk == y.clk && x.pc == y.pc && x.next_pc == y.next_pc && xi.opcode == yi.opcode &&
           xi.op_a == yi.op_a && xi.op_b == yi.op_b && xi.op_c == yi.op_c && xi.imm_b == yi.imm_b &&
           xi.imm_c == yi.imm_c && x.a == y.a && x.b == y.b && x.c == y.c && same_record(x.a_record, y.a_record) &&
           same_record(x.b_record, y.b_record) && same_record(x.c_record, y.c_record) && x.exit_code == y.exit_code;
  }

  bool same_cpu(const FfiCpuEventU64& x, const FfiCpuEventU64& y)
  {
    return same_cpu_registers(x, y) && same_record(x.memory_record, y.memory_record) && x.memory.tag == y.memory.tag &&
           (x.memory.tag == FfiOption<uint64_t>::Tag::None || x.memory.some._0 == y.memory.some._0);
  }

  bool same_alu(const AluEventU64& x, const AluEventU64& y)
  {
    return x.clk == y.clk && x.opcode == y.opcode && x.a == y.a && x.b == y.b && x.c == y.c && x.is_imm == y.is_imm &&
           x.is_cmp == y.is_cmp;
  }

  // The list chips' row functions on the intact fill, its first memory-list row pointed at the ADD
  // step when `misfiled`, whose accesses are then broken too: the error word, or ~0 when a row's
  // event is not the one its step rebuilds, or a row is not of the kind expected.
  uint32_t list_rows(bool misfiled)
  {
    std::vector<uint64_t> fill = intact_fill();
    FillDescriptor& d = descriptor_of(fill);
    if (misfiled) {
      uint8_t* bytes = reinterpret_cast<uint8_t*>(fill.data());
      reinterpret_cast<uint32_t*>(bytes + d.entries_at)[d.list_start[LIST_MEMORY]] = 0;
      // A write row past the next step's first: reading the step's accesses would set the write bit.
      reinterpret_cast<Step*>(bytes + d.steps_at)[0].first_write = 2;
    }
    uint32_t error = 0;
    HistoryArgs args = {};
    args.fill = reinterpret_cast<const uint8_t*>(fill.data());
    args.fill_bytes = d.buffer_bytes;
    args.expected = d.identity;
    args.program = ProgramTable{PROGRAM.data(), PROGRAM.size(), PC_BASE, PROGRAM_KEY};
    args.error = &error;
    History h;
    if (!open(args, Readiness::Complete, h)) return error;
    bool held = true;
    const uint32_t rows = 4;
    for (uint32_t row = 0; row < rows; ++row) {
      uint32_t count = 0, step = 0;
      FfiCpuEventU64 event = {};
      const Row memory = cpu_memory_lt_history::memory_row_of(h, rows, row, row == 0, count, event);
      if (row >= count)
        held = held && memory == Row::Padding;
      else if (misfiled && row == 0)
        held = held && memory == Row::Nothing;
      else
        held =
          held && memory == Row::Event && entry_at(h, LIST_MEMORY, row, step) && same_cpu(event, cpu_event(h, step));
      event = {};
      const Row ecall = ecall_syscall_history::ecall_row_of(h, rows, row, row == 0, count, event);
      if (row >= count)
        held = held && ecall == Row::Padding;
      else
        held = held && ecall == Row::Event && entry_at(h, LIST_ECALL, row, step) && same_cpu(event, cpu_event(h, step));
      AluEventU64 compare = {};
      const Row lt = cpu_memory_lt_history::lt_row_of(h, rows, row, row == 0, count, compare);
      if (row >= count)
        held = held && lt == Row::Padding;
      else
        held = held && lt == Row::Event && same_alu(compare, history_check::compare_event(h, row));
    }
    return held ? error : ~0u;
  }

  struct Case {
    const char* name;
    // Breaks the fill, and the launch arguments' identity and allocation size where the case is
    // about them.
    std::function<void(std::vector<uint64_t>& fill, FillIdentity& expected, uint64_t& fill_bytes)> corrupt;
    Readiness readiness;
    uint32_t expected_bits;
  };

  // Opens the fill as the launch names it and reads every row through the loaders; the error word.
  uint32_t run(
    const std::vector<uint64_t>& fill, const FillIdentity& expected, uint64_t fill_bytes, Readiness readiness)
  {
    // Exactly the bytes the launch names, so a read past them is the sanitizers' to see.
    std::unique_ptr<uint8_t[]> allocation(new uint8_t[fill_bytes]);
    std::memcpy(allocation.get(), fill.data(), std::min<uint64_t>(fill_bytes, fill.size() * 8));
    uint32_t error = 0;
    HistoryArgs args = {};
    args.fill = allocation.get();
    args.fill_bytes = fill_bytes;
    args.expected = expected;
    args.program = ProgramTable{PROGRAM.data(), PROGRAM.size(), PC_BASE, PROGRAM_KEY};
    args.error = &error;
    History h;
    if (!open(args, readiness, h)) return error;
    for (uint32_t exception = 0; exception < h.exception_count; ++exception)
      exception_in_order(h, exception);
    for (uint32_t entry = 0; entry < h.list_start[LISTS]; ++entry)
      entry_in_order(h, entry);
    // The bitmap lookup's index, in arrays of exactly the words it holds.
    const uint32_t words = exception_words(h.step_count);
    std::unique_ptr<uint32_t[]> bits(new uint32_t[words]), first(new uint32_t[words]);
    for (uint32_t word = 0; word < words; ++word)
      build_exception_word(h, word, bits.get(), first.get());
    History indexed = h;
    indexed.exception_bits = bits.get();
    indexed.exception_first = first.get();
    if (readiness == Readiness::Complete) {
      for (uint32_t step = 0; step < h.step_count; ++step) {
        cpu_event<ExceptionLookup::Search>(h, step);
        cpu_event<ExceptionLookup::Bitmap>(indexed, step);
        Fetched fetched;
        fetch<ExceptionLookup::Search, MemorySlot::Skip>(h, step, fetched);
        fetch<ExceptionLookup::Bitmap, MemorySlot::Skip>(indexed, step, fetched);
      }
      for (uint32_t row = 0; row < list_length(h, LIST_MEMORY); ++row) {
        uint32_t index = 0;
        if (entry_at(h, LIST_MEMORY, row, index)) cpu_event(h, index);
      }
      for (uint32_t row = 0; row < list_length(h, LIST_BRANCH); ++row)
        history_check::compare_event(h, list_length(h, LIST_LT) + row);
      for (uint32_t row = 0; row < h.syscall_count; ++row)
        syscall_event(h, row);
    }
    return error;
  }

} // namespace

int main()
{
  const std::vector<Case> cases = {
    {"intact", [](auto&, auto&, auto&) {}, Readiness::Complete, 0},
    {"completion word of another fill",
     [](auto& fill, auto&, auto&) {
       const uint64_t at = descriptor_of(fill).completion_at;
       reinterpret_cast<uint8_t*>(fill.data())[at] ^= 1;
     },
     Readiness::Complete, ERROR_INCOMPLETE},
    {"completion word of another fill, read by a CPU slice",
     [](auto& fill, auto&, auto&) {
       const uint64_t at = descriptor_of(fill).completion_at;
       reinterpret_cast<uint8_t*>(fill.data())[at] ^= 1;
     },
     Readiness::Descriptor, 0},
    {"launched for another nonce", [](auto&, auto& expected, auto&) { expected.nonce ^= 1; }, Readiness::Complete,
     ERROR_IDENTITY},
    {"launched for another chunk", [](auto&, auto& expected, auto&) { expected.chunk += 1; }, Readiness::Complete,
     ERROR_IDENTITY},
    {"fill made against another program table",
     [](auto& fill, auto&, auto&) { descriptor_of(fill).program_key ^= 1; }, Readiness::Complete, ERROR_PROGRAM},
    {"allocation short of the fill by one word", [](auto&, auto&, auto& fill_bytes) { fill_bytes -= 8; },
     Readiness::Complete, ERROR_LAYOUT},
    {"allocation short of a descriptor", [](auto&, auto&, auto& fill_bytes) { fill_bytes = 100; },
     Readiness::Descriptor, ERROR_LAYOUT},
    {"descriptor claiming a buffer past its allocation",
     [](auto& fill, auto&, auto&) {
       FillDescriptor& d = descriptor_of(fill);
       d.buffer_bytes += 1 << 20;
       d.completion_at += 1 << 19;
     },
     Readiness::Complete, ERROR_LAYOUT},
    {"list entries past the buffer", [](auto& fill, auto&, auto&) { descriptor_of(fill).list_start[LISTS] = 1u << 30; },
     Readiness::Complete, ERROR_LAYOUT},
    {"list starts out of order",
     [](auto& fill, auto&, auto&) {
       FillDescriptor& d = descriptor_of(fill);
       d.list_start[3] = d.list_start[LISTS];
     },
     Readiness::Complete, ERROR_LAYOUT},
    {"exceptions out of order",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       auto* rows = reinterpret_cast<OperandException*>(reinterpret_cast<uint8_t*>(fill.data()) + d.exceptions_at);
       std::swap(rows[0], rows[2]);
     },
     Readiness::Complete, ERROR_EXCEPTION},
    {"exception slots out of order within a step",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       auto* rows = reinterpret_cast<OperandException*>(reinterpret_cast<uint8_t*>(fill.data()) + d.exceptions_at);
       std::swap(rows[1].slot, rows[2].slot);
     },
     Readiness::Complete, ERROR_EXCEPTION},
    {"exception on no step",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       reinterpret_cast<OperandException*>(reinterpret_cast<uint8_t*>(fill.data()) + d.exceptions_at)[2].step = 99;
     },
     Readiness::Complete, ERROR_EXCEPTION},
    {"exception naming memory, no operand",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       reinterpret_cast<OperandException*>(reinterpret_cast<uint8_t*>(fill.data()) + d.exceptions_at)[0].slot = MEMORY;
     },
     Readiness::Complete, ERROR_EXCEPTION},
    {"memory list naming no step",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(fill.data()) + d.entries_at)[d.list_start[LIST_MEMORY]] =
         1000;
     },
     Readiness::Complete, ERROR_LIST | ERROR_STEP},
    {"memory list out of step order",
     [](auto& fill, auto&, auto&) {
       const FillDescriptor& d = descriptor_of(fill);
       auto* entries = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(fill.data()) + d.entries_at);
       std::swap(entries[d.list_start[LIST_MEMORY]], entries[d.list_start[LIST_MEMORY] + 1]);
     },
     Readiness::Complete, ERROR_LIST},
  };
  bool passed = true;
  {
    // A word past the bitmap index's extent is refused, not written.
    std::vector<uint64_t> fill = intact_fill();
    uint32_t error = 0;
    HistoryArgs args = {};
    args.fill = reinterpret_cast<const uint8_t*>(fill.data());
    args.fill_bytes = descriptor_of(fill).buffer_bytes;
    args.expected = descriptor_of(fill).identity;
    args.program = ProgramTable{PROGRAM.data(), PROGRAM.size(), PC_BASE, PROGRAM_KEY};
    args.error = &error;
    History h;
    const uint32_t words = exception_words(descriptor_of(fill).identity.steps);
    std::unique_ptr<uint32_t[]> bits(new uint32_t[words]), first(new uint32_t[words]);
    const bool opened = open(args, Readiness::Complete, h);
    build_exception_word(h, words, bits.get(), first.get());
    const bool same = opened && error == ERROR_EXCEPTION;
    passed = passed && same;
    std::printf(
      "  %-55s error word %#06x, expected %#06x%s\n", "bitmap word past the index", error, ERROR_EXCEPTION,
      same ? "" : "  FAIL");
  }
  {
    // A fetch skipping the memory slot skips nothing else: on every step of the intact fill, the
    // loads and stores among them, it reads what the full fetch reads but that access.
    std::vector<uint64_t> fill = intact_fill();
    uint32_t error = 0;
    HistoryArgs args = {};
    args.fill = reinterpret_cast<const uint8_t*>(fill.data());
    args.fill_bytes = descriptor_of(fill).buffer_bytes;
    args.expected = descriptor_of(fill).identity;
    args.program = ProgramTable{PROGRAM.data(), PROGRAM.size(), PC_BASE, PROGRAM_KEY};
    args.error = &error;
    History h;
    bool same = open(args, Readiness::Complete, h);
    uint32_t memory_steps = 0;
    for (uint32_t step = 0; same && step < h.step_count; ++step) {
      Fetched skipped, loaded;
      same =
        fetch<ExceptionLookup::Search, MemorySlot::Skip>(h, step, skipped) &&
        fetch<ExceptionLookup::Search>(h, step, loaded) && history_check::same_but_memory(skipped, loaded);
      memory_steps += loaded.accesses.slots.contains(MEMORY);
    }
    same = same && error == 0 && memory_steps == 2;
    passed = passed && same;
    std::printf(
      "  %-55s error word %#06x, expected %#06x%s\n", "a fetch skipping the memory slot", error, 0u,
      same ? "" : "  FAIL");
  }
  for (const bool misfiled : {false, true}) {
    // A list chip fetches a row once and tests its opcode before reading its accesses: a misfiled
    // row sets the list bit alone.
    const uint32_t error = list_rows(misfiled), expected = misfiled ? uint32_t(ERROR_LIST) : 0u;
    const bool same = error == expected;
    passed = passed && same;
    std::printf(
      "  %-55s error word %#06x, expected %#06x%s\n",
      misfiled ? "list rows, a memory row naming the ADD step" : "list rows from one fetch", error, expected,
      same ? "" : "  FAIL");
  }
  {
    // A trace of no rows: the memory, less-than and CPU padding kernels still run over it, and the
    // memory list and the steps it cannot hold set their bits of the error word, never dropped.
    std::vector<uint64_t> fill = intact_fill();
    uint32_t error = 0;
    HistoryArgs args = {};
    args.fill = reinterpret_cast<const uint8_t*>(fill.data());
    args.fill_bytes = descriptor_of(fill).buffer_bytes;
    args.expected = descriptor_of(fill).identity;
    args.program = ProgramTable{PROGRAM.data(), PROGRAM.size(), PC_BASE, PROGRAM_KEY};
    args.error = &error;
    History h;
    bool same = open(args, Readiness::Complete, h) && list_length(h, LIST_MEMORY) > 0 && h.step_count > 0;
    uint32_t memory_count = 1, lt_count = 1;
    FfiCpuEventU64 event = {};
    AluEventU64 compare = {};
    same = same && cpu_memory_lt_history::memory_row_of(h, 0, 0, true, memory_count, event) == Row::Nothing &&
           memory_count == 0 && cpu_memory_lt_history::lt_row_of(h, 0, 0, true, lt_count, compare) == Row::Nothing &&
           lt_count == 0 && cpu_memory_lt_history::cpu_pad_row_of(h, 0, 0, true) == Row::Nothing;
    const uint32_t expected = uint32_t(ERROR_LIST) | uint32_t(ERROR_STEP);
    same = same && error == expected;
    passed = passed && same;
    std::printf(
      "  %-55s error word %#06x, expected %#06x%s\n", "a trace of no rows under a non-empty fill", error, expected,
      same ? "" : "  FAIL");
  }
  for (const Case& c : cases) {
    std::vector<uint64_t> fill = intact_fill();
    FillIdentity expected = descriptor_of(fill).identity;
    uint64_t fill_bytes = descriptor_of(fill).buffer_bytes;
    c.corrupt(fill, expected, fill_bytes);
    const uint32_t error = run(fill, expected, fill_bytes, c.readiness);
    const bool same = error == c.expected_bits;
    passed = passed && same;
    std::printf("  %-55s error word %#06x, expected %#06x%s\n", c.name, error, c.expected_bits, same ? "" : "  FAIL");
  }
  std::printf("%s\n", passed ? "every corrupt fill caught" : "FAILED");
  return passed ? 0 : 1;
}

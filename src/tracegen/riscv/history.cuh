// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// One chunk's execution history on the device, and the loaders every RISC-V chip kernel reads it
// through: the device port of pico's vm/src/emulator/riscv/trace_source.rs over the rows of
// vm/src/emulator/riscv/history.rs, whose layout every row here mirrors field for field.
//
// A fill is one device buffer. A FillDescriptor opens it and the operand-exception table follows,
// so that one copy carries both ahead of the rest; the steps, reads, writes, chip-list entries and
// syscall rows sit wherever the descriptor places them; the completion word lands last. A kernel
// is handed the identity of the fill it was launched for and opens the buffer against it on the
// device, since a host-side comparison says nothing about what the buffer holds. A kernel that
// reads past one CPU slice also requires the completion word to name that fill.
//
// Every index is checked against the descriptor. A failed check sets a bit of the error word and
// the loader returns a zero row, so the kernel runs to its end and the chunk fails when the host
// reads the word after the last reader of the fill: wrong rows are never silent.
//
// A step stores no instruction and no operand: the instruction is the program table's at its pc,
// the slots it accessed follow from the instruction (slots_of), and its operands from its accesses
// (operands_of), the exception table holding each operand that rule gets wrong. Both rules must be
// history.rs's `Slots::of` and `ruled_operands` exactly, because the host sink writes an exception
// only where its own copy of the rule misses.
//
// Plain data, inline functions and templates only: the header compiles into any number of
// translation units, and for the host with no CUDA header.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
  #ifndef REPLAY_HD
    #define REPLAY_HD __host__ __device__
  #endif
#else
  #ifndef REPLAY_HD
    #define REPLAY_HD
  #endif
  // types.cuh marks its helpers for both sides unconditionally.
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __forceinline__
    #define __forceinline__ inline
  #endif
#endif

#include <cstddef>
#include "tracegen/common/types.cuh"

// The operand-exception lookup operands_of uses when none is named. Set it alike for every
// translation unit of one library.
#ifndef HISTORY_EXCEPTION_BITMAP
  #define HISTORY_EXCEPTION_BITMAP 0
#endif

namespace pico_gpu::history {

  struct Step {
    uint32_t clk;
    uint32_t pc;
    uint32_t next_pc;
    // A step's reads run from its `first_read` to the next step's, or to the end of the reads for
    // the last step; its write likewise.
    uint32_t first_read;
    uint32_t first_write;
  };
  static_assert(sizeof(Step) == 20, "history.rs Step");
  static_assert(offsetof(Step, clk) == 0, "history.rs Step");
  static_assert(offsetof(Step, pc) == 4, "history.rs Step");
  static_assert(offsetof(Step, next_pc) == 8, "history.rs Step");
  static_assert(offsetof(Step, first_read) == 12, "history.rs Step");
  static_assert(offsetof(Step, first_write) == 16, "history.rs Step");

  // A read leaves the value it found, so it carries no previous value.
  struct ReadAccess {
    uint64_t value;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };
  static_assert(sizeof(ReadAccess) == 16, "history.rs ReadAccess");
  static_assert(offsetof(ReadAccess, value) == 0, "history.rs ReadAccess");
  static_assert(offsetof(ReadAccess, prev_chunk) == 8, "history.rs ReadAccess");
  static_assert(offsetof(ReadAccess, prev_timestamp) == 12, "history.rs ReadAccess");

  // A write row; a read is promoted to one with `prev_value` its own value.
  struct Access {
    uint64_t value;
    uint64_t prev_value;
    uint32_t prev_chunk;
    uint32_t prev_timestamp;
  };
  static_assert(sizeof(Access) == 24, "history.rs Access");
  static_assert(offsetof(Access, value) == 0, "history.rs Access");
  static_assert(offsetof(Access, prev_value) == 8, "history.rs Access");
  static_assert(offsetof(Access, prev_chunk) == 16, "history.rs Access");
  static_assert(offsetof(Access, prev_timestamp) == 20, "history.rs Access");

  // The operand in `slot`, a Position, that the step at `step` saw. The table is sorted by step
  // and, within a step, in `a`, `b`, `c` order.
  struct OperandException {
    uint32_t step;
    uint32_t slot;
    uint64_t value;
  };
  static_assert(sizeof(OperandException) == 16, "history.rs OperandException");
  static_assert(offsetof(OperandException, step) == 0, "history.rs OperandException");
  static_assert(offsetof(OperandException, slot) == 4, "history.rs OperandException");
  static_assert(offsetof(OperandException, value) == 8, "history.rs OperandException");

  // The syscall rows are pico's `SyscallEvent` (#[repr(C)]), the type the syscall kernel reads.
  static_assert(sizeof(SyscallEventU64) == 40, "pico SyscallEvent");
  static_assert(offsetof(SyscallEventU64, chunk) == 0, "pico SyscallEvent");
  static_assert(offsetof(SyscallEventU64, clk) == 8, "pico SyscallEvent");
  static_assert(offsetof(SyscallEventU64, syscall_id) == 16, "pico SyscallEvent");
  static_assert(offsetof(SyscallEventU64, arg1) == 24, "pico SyscallEvent");
  static_assert(offsetof(SyscallEventU64, arg2) == 32, "pico SyscallEvent");

  // The program table is pico's `Instruction` (#[repr(C)]) as the program chip uploads it.
  static_assert(sizeof(InstructionU64) == 32, "pico Instruction");
  static_assert(offsetof(InstructionU64, opcode) == 0, "pico Instruction");
  static_assert(offsetof(InstructionU64, op_a) == 1, "pico Instruction");
  static_assert(offsetof(InstructionU64, op_b) == 8, "pico Instruction");
  static_assert(offsetof(InstructionU64, op_c) == 16, "pico Instruction");
  static_assert(offsetof(InstructionU64, imm_b) == 24, "pico Instruction");
  static_assert(offsetof(InstructionU64, imm_c) == 25, "pico Instruction");

  static_assert(sizeof(AluEventU64) == 48, "pico AluEvent");
  static_assert(offsetof(AluEventU64, clk) == 0, "pico AluEvent");
  static_assert(offsetof(AluEventU64, opcode) == 8, "pico AluEvent");
  static_assert(offsetof(AluEventU64, a) == 16, "pico AluEvent");
  static_assert(offsetof(AluEventU64, b) == 24, "pico AluEvent");
  static_assert(offsetof(AluEventU64, c) == 32, "pico AluEvent");
  static_assert(offsetof(AluEventU64, is_imm) == 40, "pico AluEvent");
  static_assert(offsetof(AluEventU64, is_cmp) == 41, "pico AluEvent");

  // Operand slots, valued as pico's `MemoryAccessPosition`: the timestamp offset of the slot's
  // access from the step's clock.
  enum Position : uint32_t { MEMORY = 0, SLOT_C = 1, SLOT_B = 2, SLOT_A = 3, POSITIONS = 4 };

  // The chip lists in the order history.rs's `ChipSteps` declares them.
  enum List : uint32_t {
    LIST_ADD = 0,
    LIST_SUB = 1,
    LIST_MUL = 2,
    LIST_BITWISE = 3,
    LIST_SHIFT_LEFT = 4,
    LIST_SHIFT_RIGHT = 5,
    LIST_LT = 6,
    LIST_DIVREM = 7,
    LIST_MEMORY = 8,
    LIST_ECALL = 9,
    LIST_BRANCH = 10,
    LISTS = 11,
  };

  // Names the row layout, the list order and the launch arguments. Bumped with any of them; every
  // fill and every dump carries it, so a reader of another layout fails on the first word, and every
  // entry point taking HistoryArgs from the host checks it before launching.
  constexpr uint32_t HISTORY_LAYOUT = 0x48530000u | (LISTS << 8) | 2u;

  enum Error : uint32_t {
    // The descriptor is of another layout, or places a table outside the buffer or the buffer
    // outside its allocation.
    ERROR_LAYOUT = 1u << 0,
    // The buffer holds another fill than the kernel was launched for.
    ERROR_IDENTITY = 1u << 1,
    // The program table is not the one the fill was made against.
    ERROR_PROGRAM = 1u << 2,
    // The completion word does not name the fill: the kernel ran before the whole history landed.
    ERROR_INCOMPLETE = 1u << 3,
    ERROR_STEP = 1u << 4,
    ERROR_READ = 1u << 5,
    ERROR_WRITE = 1u << 6,
    // A step's rows are not the ones its instruction's slots make.
    ERROR_ROWS = 1u << 7,
    // A list row past its list, or a list entry that names no step or does not follow the one
    // before it.
    ERROR_LIST = 1u << 8,
    ERROR_PC = 1u << 9,
    // An exception names no operand or no step, is out of order, or the bitmap lookup has no index.
    ERROR_EXCEPTION = 1u << 10,
    ERROR_SYSCALL = 1u << 11,
    ERROR_SLICE = 1u << 12,
  };

  struct FillIdentity {
    uint32_t chunk;
    uint32_t steps;
    // Never zero and never reused on a device: the completion word is this value, so a word left
    // by an earlier fill of the same memory cannot pass for this one.
    uint64_t nonce;
  };
  static_assert(sizeof(FillIdentity) == 16, "fill identity");
  static_assert(offsetof(FillIdentity, chunk) == 0, "fill identity");
  static_assert(offsetof(FillIdentity, steps) == 4, "fill identity");
  static_assert(offsetof(FillIdentity, nonce) == 8, "fill identity");

  // The immutable part of a fill's header. It and the exception table land before the CPU chip's
  // first slice, the completion word after the whole history.
  struct FillDescriptor {
    uint32_t layout;
    uint32_t descriptor_bytes;
    FillIdentity identity;
    // The key the program table is cached under on the device.
    uint64_t program_key;
    // Every table and the completion word end inside the buffer.
    uint64_t buffer_bytes;
    uint32_t reads;
    uint32_t writes;
    uint32_t exceptions;
    uint32_t syscalls;
    // List l's entries are entries[list_start[l], list_start[l + 1]); list_start[LISTS] is their count.
    uint32_t list_start[LISTS + 1];
    // Rows no list holds, so that every chip's height is known before any kernel runs: the ADD rows
    // div/rem appends after the add chip's own, and the coarse add and sub lists split by opcode.
    uint32_t divrem_add_rows;
    uint32_t add_rows;
    uint32_t addw_rows;
    uint32_t sub_rows;
    uint32_t subw_rows;
    // Slice k of the CPU chip holds steps [k * cpu_slice_steps, (k + 1) * cpu_slice_steps), the
    // last one cut at the step count. A slice's copy carries the step after it too: that step's
    // first read and write end the rows of the slice's last step.
    uint32_t cpu_slice_steps;
    // Byte offsets from the start of the buffer, each a multiple of 8.
    uint64_t steps_at;
    uint64_t reads_at;
    uint64_t writes_at;
    uint64_t exceptions_at;
    uint64_t entries_at;
    uint64_t syscalls_at;
    uint64_t completion_at;
  };
  static_assert(sizeof(FillDescriptor) == 184, "fill descriptor");
  static_assert(offsetof(FillDescriptor, layout) == 0, "fill descriptor");
  static_assert(offsetof(FillDescriptor, descriptor_bytes) == 4, "fill descriptor");
  static_assert(offsetof(FillDescriptor, identity) == 8, "fill descriptor");
  static_assert(offsetof(FillDescriptor, program_key) == 24, "fill descriptor");
  static_assert(offsetof(FillDescriptor, buffer_bytes) == 32, "fill descriptor");
  static_assert(offsetof(FillDescriptor, reads) == 40, "fill descriptor");
  static_assert(offsetof(FillDescriptor, writes) == 44, "fill descriptor");
  static_assert(offsetof(FillDescriptor, exceptions) == 48, "fill descriptor");
  static_assert(offsetof(FillDescriptor, syscalls) == 52, "fill descriptor");
  static_assert(offsetof(FillDescriptor, list_start) == 56, "fill descriptor");
  static_assert(offsetof(FillDescriptor, divrem_add_rows) == 104, "fill descriptor");
  static_assert(offsetof(FillDescriptor, add_rows) == 108, "fill descriptor");
  static_assert(offsetof(FillDescriptor, addw_rows) == 112, "fill descriptor");
  static_assert(offsetof(FillDescriptor, sub_rows) == 116, "fill descriptor");
  static_assert(offsetof(FillDescriptor, subw_rows) == 120, "fill descriptor");
  static_assert(offsetof(FillDescriptor, cpu_slice_steps) == 124, "fill descriptor");
  static_assert(offsetof(FillDescriptor, steps_at) == 128, "fill descriptor");
  static_assert(offsetof(FillDescriptor, reads_at) == 136, "fill descriptor");
  static_assert(offsetof(FillDescriptor, writes_at) == 144, "fill descriptor");
  static_assert(offsetof(FillDescriptor, exceptions_at) == 152, "fill descriptor");
  static_assert(offsetof(FillDescriptor, entries_at) == 160, "fill descriptor");
  static_assert(offsetof(FillDescriptor, syscalls_at) == 168, "fill descriptor");
  static_assert(offsetof(FillDescriptor, completion_at) == 176, "fill descriptor");

  struct ProgramTable {
    const InstructionU64* instructions;
    uint64_t count;
    uint64_t pc_base;
    // Compared with the fill's `program_key`.
    uint64_t key;
  };
  static_assert(sizeof(ProgramTable) == 32, "program table");
  static_assert(offsetof(ProgramTable, instructions) == 0, "program table");
  static_assert(offsetof(ProgramTable, count) == 8, "program table");
  static_assert(offsetof(ProgramTable, pc_base) == 16, "program table");
  static_assert(offsetof(ProgramTable, key) == 24, "program table");

  // The exception table is sparse (writes to x0, and the ECALL closing an unconstrained block) and
  // lands with the descriptor, so a binary search over it needs nothing built on the device and no
  // ordering of such a build against the CPU chip's first slice. The bitmap trades that build, and
  // two words per 32 steps, for one load per step that has no exception.
  enum class ExceptionLookup : uint32_t { Search, Bitmap };
  constexpr ExceptionLookup DEFAULT_EXCEPTION_LOOKUP =
    HISTORY_EXCEPTION_BITMAP ? ExceptionLookup::Bitmap : ExceptionLookup::Search;

  // What a kernel reading the history is launched with, by value. The host builds it, so its layout
  // crosses the FFI as the rows do.
  struct HistoryArgs {
    // The fill's buffer, its descriptor at offset 0.
    const uint8_t* fill;
    FillIdentity expected;
    ProgramTable program;
    // The bitmap lookup's index (build_exception_word), `exception_words(steps)` words each; unread
    // by the search.
    const uint32_t* exception_bits;
    const uint32_t* exception_first;
    uint32_t* error;
    // The bytes allocated at `fill`. The descriptor that sizes every table is part of what is
    // checked, so its own `buffer_bytes` cannot bound the reads.
    uint64_t fill_bytes;
  };
  static_assert(sizeof(HistoryArgs) == 88, "history launch arguments");
  static_assert(offsetof(HistoryArgs, fill) == 0, "history launch arguments");
  static_assert(offsetof(HistoryArgs, expected) == 8, "history launch arguments");
  static_assert(offsetof(HistoryArgs, program) == 24, "history launch arguments");
  static_assert(offsetof(HistoryArgs, exception_bits) == 56, "history launch arguments");
  static_assert(offsetof(HistoryArgs, exception_first) == 64, "history launch arguments");
  static_assert(offsetof(HistoryArgs, error) == 72, "history launch arguments");
  static_assert(offsetof(HistoryArgs, fill_bytes) == 80, "history launch arguments");

  // What a kernel may read of a fill: the CPU chip's slice kernels start once the descriptor and the
  // exceptions have landed and read only their own slice; every other kernel needs the whole history.
  enum class Readiness : uint32_t { Descriptor, Complete };

  // An opened fill: its tables and their counts, as the descriptor on the device gives them.
  struct History {
    const Step* steps;
    const ReadAccess* reads;
    const Access* writes;
    const OperandException* exceptions;
    const uint32_t* entries;
    const SyscallEventU64* syscalls;
    const uint32_t* exception_bits;
    const uint32_t* exception_first;
    ProgramTable program;
    uint32_t* error;
    uint32_t chunk;
    uint32_t step_count;
    uint32_t read_count;
    uint32_t write_count;
    uint32_t exception_count;
    uint32_t syscall_count;
    uint32_t cpu_slice_steps;
    uint32_t list_start[LISTS + 1];
  };

  // The slots a traced step of an instruction accesses, one bit per Position, and which of them it
  // writes.
  struct Slots {
    uint8_t present;
    uint8_t written;

    REPLAY_HD bool contains(uint32_t position) const { return (present >> position) & 1u; }
    REPLAY_HD bool writes(uint32_t position) const { return (written >> position) & 1u; }
    REPLAY_HD uint32_t count() const { return bits(present); }
    REPLAY_HD uint32_t write_count() const { return bits(written); }

    static REPLAY_HD uint32_t bits(uint8_t mask)
    {
      return (mask & 1u) + (mask >> 1 & 1u) + (mask >> 2 & 1u) + (mask >> 3 & 1u);
    }
  };

  // A step's accesses by Position, a read promoted to an Access; only the slots present are set.
  struct StepAccesses {
    Access at[POSITIONS];
    Slots slots;
  };

  struct Operands {
    uint64_t a;
    uint64_t b;
    uint64_t c;
  };

  // Everything the chips read of one step.
  struct Fetched {
    Step step;
    InstructionU64 instruction;
    StepAccesses accesses;
    Operands operands;
  };

  REPLAY_HD inline void fail(uint32_t* error, uint32_t bits)
  {
#if defined(__CUDA_ARCH__)
    atomicOr(error, bits);
#else
    *error |= bits;
#endif
  }

  REPLAY_HD inline bool table_fits(uint64_t at, uint64_t rows, uint64_t row_bytes, uint64_t buffer_bytes)
  {
    return at % 8 == 0 && at >= sizeof(FillDescriptor) && at <= buffer_bytes &&
           rows * row_bytes <= buffer_bytes - at;
  }

  // Opens the fill `args` names. False, with the error word set and every count zero, when the
  // buffer is not that fill, is malformed, or has not landed as far as `readiness` needs.
  REPLAY_HD inline bool open(const HistoryArgs& args, Readiness readiness, History& h)
  {
    h = History{};
    h.program = args.program;
    h.exception_bits = args.exception_bits;
    h.exception_first = args.exception_first;
    h.error = args.error;
    if (args.fill_bytes < sizeof(FillDescriptor)) {
      fail(args.error, ERROR_LAYOUT);
      return false;
    }
    const FillDescriptor& d = *reinterpret_cast<const FillDescriptor*>(args.fill);
    if (d.layout != HISTORY_LAYOUT || d.descriptor_bytes != sizeof(FillDescriptor)) {
      fail(args.error, ERROR_LAYOUT);
      return false;
    }
    if (d.identity.chunk != args.expected.chunk || d.identity.steps != args.expected.steps ||
        d.identity.nonce != args.expected.nonce) {
      fail(args.error, ERROR_IDENTITY);
      return false;
    }
    if (d.program_key != args.program.key) {
      fail(args.error, ERROR_PROGRAM);
      return false;
    }
    bool lists_ordered = d.list_start[0] == 0;
    for (uint32_t list = 0; list < LISTS; ++list)
      lists_ordered = lists_ordered && d.list_start[list] <= d.list_start[list + 1];
    const uint64_t bytes = d.buffer_bytes;
    if (bytes > args.fill_bytes || !lists_ordered ||
        !table_fits(d.steps_at, d.identity.steps, sizeof(Step), bytes) ||
        !table_fits(d.reads_at, d.reads, sizeof(ReadAccess), bytes) ||
        !table_fits(d.writes_at, d.writes, sizeof(Access), bytes) ||
        !table_fits(d.exceptions_at, d.exceptions, sizeof(OperandException), bytes) ||
        !table_fits(d.entries_at, d.list_start[LISTS], sizeof(uint32_t), bytes) ||
        !table_fits(d.syscalls_at, d.syscalls, sizeof(SyscallEventU64), bytes) ||
        !table_fits(d.completion_at, 1, sizeof(uint64_t), bytes)) {
      fail(args.error, ERROR_LAYOUT);
      return false;
    }
    if (readiness == Readiness::Complete &&
        *reinterpret_cast<const uint64_t*>(args.fill + d.completion_at) != d.identity.nonce) {
      fail(args.error, ERROR_INCOMPLETE);
      return false;
    }
    h.steps = reinterpret_cast<const Step*>(args.fill + d.steps_at);
    h.reads = reinterpret_cast<const ReadAccess*>(args.fill + d.reads_at);
    h.writes = reinterpret_cast<const Access*>(args.fill + d.writes_at);
    h.exceptions = reinterpret_cast<const OperandException*>(args.fill + d.exceptions_at);
    h.entries = reinterpret_cast<const uint32_t*>(args.fill + d.entries_at);
    h.syscalls = reinterpret_cast<const SyscallEventU64*>(args.fill + d.syscalls_at);
    h.chunk = d.identity.chunk;
    h.step_count = d.identity.steps;
    h.read_count = d.reads;
    h.write_count = d.writes;
    h.exception_count = d.exceptions;
    h.syscall_count = d.syscalls;
    h.cpu_slice_steps = d.cpu_slice_steps;
    for (uint32_t list = 0; list <= LISTS; ++list)
      h.list_start[list] = d.list_start[list];
    return true;
  }

  REPLAY_HD inline uint32_t cpu_slices(const History& h)
  {
    if (h.cpu_slice_steps == 0) return 0;
    return static_cast<uint32_t>((uint64_t(h.step_count) + h.cpu_slice_steps - 1) / h.cpu_slice_steps);
  }

  // The steps [first, end) of the CPU chip's slice `slice`.
  REPLAY_HD inline bool cpu_slice(const History& h, uint32_t slice, uint32_t& first, uint32_t& end)
  {
    const uint64_t from = uint64_t(slice) * h.cpu_slice_steps;
    if (h.cpu_slice_steps == 0 || from >= h.step_count) {
      fail(h.error, ERROR_SLICE);
      return false;
    }
    const uint64_t to = from + h.cpu_slice_steps;
    first = static_cast<uint32_t>(from);
    end = to < h.step_count ? static_cast<uint32_t>(to) : h.step_count;
    return true;
  }

  REPLAY_HD inline uint32_t list_length(const History& h, uint32_t list)
  {
    return list < LISTS ? h.list_start[list + 1] - h.list_start[list] : 0;
  }

  // The step at row `row` of `list`.
  REPLAY_HD inline bool entry_at(const History& h, uint32_t list, uint32_t row, uint32_t& step)
  {
    if (list >= LISTS || row >= list_length(h, list)) {
      fail(h.error, ERROR_LIST);
      return false;
    }
    step = h.entries[h.list_start[list] + row];
    return true;
  }

  REPLAY_HD inline bool step_at(const History& h, uint32_t index, Step& step)
  {
    if (index >= h.step_count) {
      fail(h.error, ERROR_STEP);
      return false;
    }
    step = h.steps[index];
    return true;
  }

  // pico's `Program::fetch`.
  REPLAY_HD inline bool instruction_at(const History& h, uint32_t pc, InstructionU64& instruction)
  {
    const uint64_t at = pc;
    if (at < h.program.pc_base || (at - h.program.pc_base) / 4 >= h.program.count) {
      fail(h.error, ERROR_PC);
      return false;
    }
    instruction = h.program.instructions[(at - h.program.pc_base) / 4];
    return true;
  }

  REPLAY_HD inline uint32_t timestamp(const Step& step, uint32_t position) { return step.clk + position; }

  REPLAY_HD inline bool is_alu(Opcode opcode)
  {
    switch (opcode) {
    case Opcode::ADD:
    case Opcode::SUB:
    case Opcode::XOR:
    case Opcode::OR:
    case Opcode::AND:
    case Opcode::SLL:
    case Opcode::SRL:
    case Opcode::SRA:
    case Opcode::SLT:
    case Opcode::SLTU:
    case Opcode::MUL:
    case Opcode::MULH:
    case Opcode::MULHU:
    case Opcode::MULHSU:
    case Opcode::DIV:
    case Opcode::DIVU:
    case Opcode::REM:
    case Opcode::REMU:
    case Opcode::ADDW:
    case Opcode::SUBW:
    case Opcode::SLLW:
    case Opcode::SRLW:
    case Opcode::SRAW:
    case Opcode::MULW:
    case Opcode::DIVW:
    case Opcode::DIVUW:
    case Opcode::REMW:
    case Opcode::REMUW:
      return true;
    default:
      return false;
    }
  }

  REPLAY_HD inline bool is_store(Opcode opcode)
  {
    return opcode == Opcode::SB || opcode == Opcode::SH || opcode == Opcode::SW || opcode == Opcode::SD;
  }

  REPLAY_HD inline bool is_memory(Opcode opcode)
  {
    switch (opcode) {
    case Opcode::LB:
    case Opcode::LH:
    case Opcode::LW:
    case Opcode::LBU:
    case Opcode::LHU:
    case Opcode::LWU:
    case Opcode::LD:
      return true;
    default:
      return is_store(opcode);
    }
  }

  REPLAY_HD inline bool is_branch(Opcode opcode)
  {
    switch (opcode) {
    case Opcode::BEQ:
    case Opcode::BNE:
    case Opcode::BLT:
    case Opcode::BGE:
    case Opcode::BLTU:
    case Opcode::BGEU:
      return true;
    default:
      return false;
    }
  }

  // history.rs's `Slots::of`: `a` is read by stores and branches and written by everything else, x0
  // included; `b` and `c` are register reads where present.
  REPLAY_HD inline Slots slots_of(const InstructionU64& instruction)
  {
    const Opcode opcode = instruction.opcode;
    bool b = true, c = false;
    if (is_alu(opcode)) {
      b = !(instruction.imm_b && instruction.imm_c);
      c = !instruction.imm_c;
    } else if (opcode == Opcode::ECALL) {
      c = true;
    } else if (opcode == Opcode::JAL || opcode == Opcode::AUIPC) {
      b = false;
    }
    const bool store = is_store(opcode);
    Slots slots;
    slots.present = static_cast<uint8_t>(1u << SLOT_A | uint32_t(b) << SLOT_B | uint32_t(c) << SLOT_C |
                                         uint32_t(is_memory(opcode)) << MEMORY);
    slots.written = static_cast<uint8_t>(uint32_t(!store && !is_branch(opcode)) << SLOT_A | uint32_t(store) << MEMORY);
    return slots;
  }

  // Whether a loader reads a step's memory-slot access. A chip that proves only the register
  // accesses may skip it: the slot is the last of a step's rows, so skipping it moves no other slot's
  // row, and the row counts are checked against every slot either way.
  enum class MemorySlot : uint32_t { Load, Skip };

  // The accesses of the step at `index`, `step` being its row, in its rows' order: `a`, `b`, `c`,
  // memory, each slot present taken from the writes if it writes and from the reads otherwise. A
  // slice kernel's last step reads the next step's row, which its slice's copy carries. Under
  // MemorySlot::Skip, `at[MEMORY]` is left as it was.
  template <MemorySlot Memory = MemorySlot::Load>
  REPLAY_HD inline bool accesses_of(
    const History& h, uint32_t index, const Step& step, Slots slots, StepAccesses& accesses)
  {
    accesses.slots = slots;
    uint32_t read_end = h.read_count, write_end = h.write_count;
    if (index + 1 < h.step_count) {
      read_end = h.steps[index + 1].first_read;
      write_end = h.steps[index + 1].first_write;
    }
    if (step.first_read > read_end || read_end > h.read_count) {
      fail(h.error, ERROR_READ);
      return false;
    }
    if (step.first_write > write_end || write_end > h.write_count) {
      fail(h.error, ERROR_WRITE);
      return false;
    }
    if (read_end - step.first_read != slots.count() - slots.write_count() ||
        write_end - step.first_write != slots.write_count()) {
      fail(h.error, ERROR_ROWS);
      return false;
    }
    uint32_t read = step.first_read;
    for (uint32_t position = POSITIONS; position-- > 0;) {
      if (!slots.contains(position)) continue;
      if constexpr (Memory == MemorySlot::Skip) {
        if (position == MEMORY) break;
      }
      if (slots.writes(position)) {
        accesses.at[position] = h.writes[step.first_write];
      } else {
        const ReadAccess& row = h.reads[read++];
        accesses.at[position] = Access{row.value, row.value, row.prev_chunk, row.prev_timestamp};
      }
    }
    return true;
  }

  // history.rs's `ruled_operands`: a jump's `a` is its pc + 4, a present slot's operand is its
  // access's value, and an absent `b` or `c` is the instruction's immediate.
  REPLAY_HD inline Operands ruled_operands(const InstructionU64& instruction, uint32_t pc, const StepAccesses& accesses)
  {
    const Opcode opcode = instruction.opcode;
    uint64_t imm_c = instruction.op_c;
    if (opcode == Opcode::JAL) imm_c = 0;
    if (opcode == Opcode::AUIPC) imm_c = instruction.op_b;
    const Slots slots = accesses.slots;
    Operands operands;
    if (opcode == Opcode::JAL || opcode == Opcode::JALR) {
      operands.a = uint64_t(pc) + 4;
    } else {
      operands.a = slots.contains(SLOT_A) ? accesses.at[SLOT_A].value : 0;
    }
    operands.b = slots.contains(SLOT_B) ? accesses.at[SLOT_B].value : instruction.op_b;
    operands.c = slots.contains(SLOT_C) ? accesses.at[SLOT_C].value : imm_c;
    return operands;
  }

  // The first exception at or past step `index`.
  REPLAY_HD inline uint32_t first_exception_from(const History& h, uint64_t index)
  {
    uint32_t low = 0, high = h.exception_count;
    while (low < high) {
      const uint32_t middle = low + (high - low) / 2;
      if (h.exceptions[middle].step < index) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    return low;
  }

  // Whether exception `exception` is where the lookups look for it: on a step of the fill, in an
  // operand slot, and after the one before it by step and, within a step, in `a`, `b`, `c` order.
  // Both lookups search the table as sorted and would read an operand of the rule's in place of
  // one out of order, so a fill's exceptions are checked once, one call per row, before any lookup.
  REPLAY_HD inline bool exception_in_order(const History& h, uint32_t exception)
  {
    if (exception >= h.exception_count) {
      fail(h.error, ERROR_EXCEPTION);
      return false;
    }
    const OperandException& row = h.exceptions[exception];
    bool in_order = row.step < h.step_count && row.slot >= SLOT_C && row.slot <= SLOT_A;
    if (exception > 0) {
      const OperandException& before = h.exceptions[exception - 1];
      in_order = in_order && (before.step < row.step || (before.step == row.step && before.slot > row.slot));
    }
    if (!in_order) fail(h.error, ERROR_EXCEPTION);
    return in_order;
  }

  // Whether entry `entry` of the chip lists, counted over every list in turn, names a step of the
  // fill after its list's entry before it. The lists are in execution order, which the step-ordered
  // merge of the add and sub lists and every chip's row order rely on; one call per entry, once per
  // fill, checks that they are.
  REPLAY_HD inline bool entry_in_order(const History& h, uint32_t entry)
  {
    if (entry >= h.list_start[LISTS]) {
      fail(h.error, ERROR_LIST);
      return false;
    }
    uint32_t list = 0;
    while (entry >= h.list_start[list + 1])
      ++list;
    const uint32_t step = h.entries[entry];
    const bool in_order = step < h.step_count && (entry == h.list_start[list] || h.entries[entry - 1] < step);
    if (!in_order) fail(h.error, ERROR_LIST);
    return in_order;
  }

  REPLAY_HD inline uint32_t exception_words(uint32_t steps) { return (steps + 31) / 32; }

  // Word `word` of the bitmap lookup's index: bit s % 32 of `bits[s / 32]` is set when step s holds
  // an exception, and `first[w]` is the first exception at or past step 32 w. One call per word,
  // after the exceptions have landed and before any kernel looks one up. An exception before the
  // word, which only an unsorted table puts there, is skipped and flagged.
  REPLAY_HD inline void build_exception_word(const History& h, uint32_t word, uint32_t* bits, uint32_t* first)
  {
    if (word >= exception_words(h.step_count)) {
      fail(h.error, ERROR_EXCEPTION);
      return;
    }
    const uint64_t from = uint64_t(word) * 32;
    uint32_t exception = first_exception_from(h, from);
    first[word] = exception;
    uint32_t mask = 0;
    for (; exception < h.exception_count && h.exceptions[exception].step < from + 32; ++exception) {
      if (h.exceptions[exception].step < from) {
        fail(h.error, ERROR_EXCEPTION);
        continue;
      }
      mask |= 1u << (h.exceptions[exception].step - from);
    }
    bits[word] = mask;
  }

  // The operands of the step at `index`: the rule's, overridden by the step's exceptions.
  template <ExceptionLookup Lookup = DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline Operands operands_of(
    const History& h, uint32_t index, const Step& step, const InstructionU64& instruction,
    const StepAccesses& accesses)
  {
    Operands operands = ruled_operands(instruction, step.pc, accesses);
    uint32_t exception = 0;
    if constexpr (Lookup == ExceptionLookup::Bitmap) {
      if (h.exception_bits == nullptr || h.exception_first == nullptr ||
          index / 32 >= exception_words(h.step_count)) {
        fail(h.error, ERROR_EXCEPTION);
        return operands;
      }
      if (!((h.exception_bits[index / 32] >> (index % 32)) & 1u)) return operands;
      exception = h.exception_first[index / 32];
      while (exception < h.exception_count && h.exceptions[exception].step < index)
        ++exception;
    } else {
      exception = first_exception_from(h, index);
    }
    uint64_t* by_operand[3] = {&operands.a, &operands.b, &operands.c};
    for (; exception < h.exception_count && h.exceptions[exception].step == index; ++exception) {
      const OperandException& row = h.exceptions[exception];
      if (row.slot < SLOT_C || row.slot > SLOT_A) {
        fail(h.error, ERROR_EXCEPTION);
        continue;
      }
      *by_operand[SLOT_A - row.slot] = row.value;
    }
    return operands;
  }

  // fetch in two halves, so that a chip can test the opcode between them on the one fetch: the
  // step's row and instruction, then its accesses and operands. Each half zeroes only what it
  // fills, so a row refused on its opcode zeroes no access, and nothing past the instruction stays
  // live between the halves.
  REPLAY_HD inline bool fetch_instruction(const History& h, uint32_t index, Fetched& fetched)
  {
    fetched.step = Step{};
    fetched.instruction = InstructionU64{};
    return step_at(h, index, fetched.step) && instruction_at(h, fetched.step.pc, fetched.instruction);
  }

  template <ExceptionLookup Lookup = DEFAULT_EXCEPTION_LOOKUP, MemorySlot Memory = MemorySlot::Load>
  REPLAY_HD inline bool fetch_accesses(const History& h, uint32_t index, Fetched& fetched)
  {
    fetched.accesses = StepAccesses{};
    fetched.operands = Operands{};
    if (!accesses_of<Memory>(h, index, fetched.step, slots_of(fetched.instruction), fetched.accesses)) return false;
    fetched.operands = operands_of<Lookup>(h, index, fetched.step, fetched.instruction, fetched.accesses);
    return true;
  }

  // Under MemorySlot::Skip the memory slot's access stays zero.
  template <ExceptionLookup Lookup = DEFAULT_EXCEPTION_LOOKUP, MemorySlot Memory = MemorySlot::Load>
  REPLAY_HD inline bool fetch(const History& h, uint32_t index, Fetched& fetched)
  {
    return fetch_instruction(h, index, fetched) && fetch_accesses<Lookup, Memory>(h, index, fetched);
  }

  // The first half of the fetch of row `row` of `list`, whose step is at `index`: what a list chip
  // classifies the row by, `fetched.instruction.opcode`, before fetch_accesses reads the rest into the
  // same `fetched`. The entry, its step and its instruction are read once, and a row the chip refuses
  // on its opcode reads no access.
  REPLAY_HD inline bool fetch_entry(const History& h, uint32_t list, uint32_t row, uint32_t& index, Fetched& fetched)
  {
    return entry_at(h, list, row, index) && fetch_instruction(h, index, fetched);
  }

  // The ALU event the emulator writes for a fetched step: its clock and operands, the instruction's
  // opcode, and its `imm_c` as `is_imm`.
  REPLAY_HD inline AluEventU64 alu_event_of(const Fetched& fetched)
  {
    AluEventU64 event = {};
    event.clk = fetched.step.clk;
    event.opcode = fetched.instruction.opcode;
    event.a = fetched.operands.a;
    event.b = fetched.operands.b;
    event.c = fetched.operands.c;
    event.is_imm = fetched.instruction.imm_c;
    return event;
  }

  // The less-than chip's row for a fetched branch step: its comparison, built as pico's
  // `branch_compare` builds it. The host orders the comparisons by its own run boundaries; the chip's
  // rows are checked as a multiset, so list order proves the same trace.
  REPLAY_HD inline AluEventU64 compare_event_of(const Fetched& fetched)
  {
    AluEventU64 event = {};
    const Opcode opcode = fetched.instruction.opcode;
    const uint64_t a = fetched.operands.a, b = fetched.operands.b;
    const bool is_signed = opcode == Opcode::BLT || opcode == Opcode::BGE;
    const bool lt = is_signed ? static_cast<int64_t>(a) < static_cast<int64_t>(b) : a < b;
    const bool gt = is_signed ? static_cast<int64_t>(a) > static_cast<int64_t>(b) : a > b;
    event.clk = fetched.step.clk;
    event.opcode = is_signed ? Opcode::SLT : Opcode::SLTU;
    event.a = uint64_t(lt) | uint64_t(gt) << 16;
    event.b = a;
    event.c = b;
    event.is_cmp = true;
    return event;
  }

  REPLAY_HD inline FfiOption<MemoryRecordEnumU64> record_of(const History& h, const Fetched& fetched, uint32_t position)
  {
    FfiOption<MemoryRecordEnumU64> option = {};
    const Slots slots = fetched.accesses.slots;
    if (!slots.contains(position)) return option;
    option.tag = FfiOption<MemoryRecordEnumU64>::Tag::Some;
    MemoryRecordEnumU64& record = option.some._0;
    const Access& access = fetched.accesses.at[position];
    const uint32_t at = timestamp(fetched.step, position);
    if (slots.writes(position)) {
      record.tag = MemoryRecordEnumU64::Tag::Write;
      record.write._0 =
        MemoryWriteRecordU64{access.value, h.chunk, at, access.prev_value, access.prev_chunk, access.prev_timestamp};
    } else {
      record.tag = MemoryRecordEnumU64::Tag::Read;
      record.read._0 = MemoryReadRecordU64{access.value, h.chunk, at, access.prev_chunk, access.prev_timestamp};
    }
    return option;
  }

  // The CPU event the emulator writes for a fetched step. Its `memory` is the memory access's value,
  // of which only the presence is read, and its `exit_code`, which no chip reads, is 0. The CPU
  // kernel writes its rows from the fetch instead (cpu_u64::fetched_to_row).
  REPLAY_HD inline FfiCpuEventU64 cpu_event_of(const History& h, const Fetched& fetched)
  {
    FfiCpuEventU64 event = {};
    event.chunk = h.chunk;
    event.clk = fetched.step.clk;
    event.pc = fetched.step.pc;
    event.next_pc = fetched.step.next_pc;
    event.instruction = fetched.instruction;
    event.a = fetched.operands.a;
    event.a_record = record_of(h, fetched, SLOT_A);
    event.b = fetched.operands.b;
    event.b_record = record_of(h, fetched, SLOT_B);
    event.c = fetched.operands.c;
    event.c_record = record_of(h, fetched, SLOT_C);
    event.memory_record = record_of(h, fetched, MEMORY);
    if (fetched.accesses.slots.contains(MEMORY)) {
      event.memory.tag = FfiOption<uint64_t>::Tag::Some;
      event.memory.some._0 = fetched.accesses.at[MEMORY].value;
    }
    return event;
  }

  // The CPU event of the step at `index`, or the zero event when the step cannot be fetched.
  template <ExceptionLookup Lookup = DEFAULT_EXCEPTION_LOOKUP>
  REPLAY_HD inline FfiCpuEventU64 cpu_event(const History& h, uint32_t index)
  {
    Fetched fetched;
    if (!fetch<Lookup>(h, index, fetched)) return FfiCpuEventU64{};
    return cpu_event_of(h, fetched);
  }

  REPLAY_HD inline SyscallEventU64 syscall_event(const History& h, uint32_t row)
  {
    if (row >= h.syscall_count) {
      fail(h.error, ERROR_SYSCALL);
      return SyscallEventU64{};
    }
    return h.syscalls[row];
  }

} // namespace pico_gpu::history

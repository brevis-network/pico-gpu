// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>
#include "common/rusterror.h"
#include "tracegen/common/field_word_range_checker.cuh"
#include "tracegen/riscv/instruction.cuh"
#include "tracegen/riscv/memory_read_write.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"
#include "tracegen/riscv/byte.cuh"
#include "tracegen/riscv/divrem.cuh"
#include "tracegen/riscv/history.cuh"


// ─── U64 CPU chip ──────────────────────────────────────────────────────
namespace pico_gpu::cpu_u64 {
  using namespace byte;
  using divrem::write_word_from_u64;

  // Two limbs (Rust `Pc::try_from`, which rejects a value at or above 2^32; the disassembler's
  // guest address bound rules that out, and this side does not check).
  template <class F>
  __device__ inline void write_pc_from_u64(Pc<F>& pc, uint64_t value)
  {
    pc._0[0] = F::from_canonical_u16(static_cast<uint16_t>(value & 0xFFFF));
    pc._0[1] = F::from_canonical_u16(static_cast<uint16_t>((value >> 16) & 0xFFFF));
  }

  template <class F>
  __device__ inline void populate_opcode_selector(
    const InstructionU64& instruction, OpcodeSelectorColsU64<F>& cols)
  {
    cols.imm_b = F::from_bool(instruction.imm_b);
    cols.imm_c = F::from_bool(instruction.imm_c);

    // The ALU class is derived in the constraints (one-hot complement), so this branch
    // simply leaves every class selector at zero instead of committing is_alu.
    if (instruction::is_alu_instruction_u64(instruction)) {
      // no-op: ALU class is derived, not committed.
    } else if (instruction::is_ecall_instruction_u64(instruction)) {
      cols.is_ecall = F::one();
    } else if (instruction::is_memory_instruction_u64(instruction)) {
      // per-variant flags live in the MemoryReadWriteChip now.
      cols.is_memory = F::one();
      switch (instruction.opcode) {
      case Opcode::SB:
      case Opcode::SH:
      case Opcode::SW:
      case Opcode::SD:
        cols.is_store = F::one();
        break;
      default: break;
      }
    } else if (instruction::is_branch_instruction_u64(instruction)) {
      switch (instruction.opcode) {
      case Opcode::BEQ:  cols.is_beq = F::one(); break;
      case Opcode::BNE:  cols.is_bne = F::one(); break;
      case Opcode::BLT:  cols.is_blt = F::one(); break;
      case Opcode::BGE:  cols.is_bge = F::one(); break;
      case Opcode::BLTU: cols.is_bltu = F::one(); break;
      case Opcode::BGEU: cols.is_bgeu = F::one(); break;
      default: break;
      }
    } else if (instruction.opcode == Opcode::JAL) {
      cols.is_jal = F::ONE;
    } else if (instruction.opcode == Opcode::JALR) {
      cols.is_jalr = F::ONE;
    }
    // A trap instruction (UNIMP / EBREAK, Rust `Opcode::is_trap`) sets no class selector; the
    // emulator never emits an event for one.
  }

  template <class F>
  __device__ inline void populate_memory_access_u64(
    MemoryAccessLiteCols<F>& col,
    uint64_t value, uint32_t rec_chunk, uint32_t rec_timestamp,
    uint32_t rec_prev_chunk, uint32_t rec_prev_timestamp,
    F* byte_trace)
  {
    write_word_from_u64(col.value, value);
    // The predecessor is always in this chunk. When the address was last touched in an earlier
    // chunk there is no in-chunk timestamp to name, so the sentinel 0 stands in -- the same label
    // MemoryLocalChip puts on the state the chunk inherits. The emulator starts each chunk's clk
    // at 4, which keeps every real access above the sentinel. Against the sentinel the diff spans
    // the whole clk range, so it fits 16+9 only because the Rust split config caps a chunk's clk
    // at 2^25 - 1; `& 0x1FF` below would truncate silently past that.
    // The predecessor clk is not a column (Rust `MemoryAccessLiteCols`): the AIR rebuilds
    // it from the two committed difference limbs, so it is computed here only to difference from.
    const uint32_t prev_clk = (rec_prev_chunk == rec_chunk) ? rec_prev_timestamp : 0u;

    const uint32_t diff_minus_one = rec_timestamp - prev_clk - 1;

    col.diff_16bit_limb = F::from_canonical_u16(static_cast<uint16_t>(diff_minus_one & 0xFFFF));
    // Single 9-bit high limb (bits 16..25), BitRange(9)-checked.
    col.diff_hi9 = F::from_canonical_u16(static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF));

    add_u16_range_check<F>(byte_trace, static_cast<uint16_t>(diff_minus_one & 0xFFFF));
    add_bit_range_check<F>(byte_trace, static_cast<uint16_t>((diff_minus_one >> 16) & 0x1FF), 9);
  }

  template <class F>
  __device__ inline void assign_padding_row(CpuColsU64<F>& cols)
  {
    cols.opcode_selector.imm_b = F::one();
    cols.opcode_selector.imm_c = F::one();
  }

  // The CPU row of the step `fetched` holds, written from the history's rows: no CPU event is built.
  // A present register slot's access columns are filled where the AIR reads them (`a` always, `b`
  // and `c` when read); `a`, `b` and `c` are the operands, exceptions applied, which is what a branch
  // compares and a JALR target sums. No column reads the memory slot, so the step may be fetched with
  // MemorySlot::Skip. A branch's comparison row is the less-than chip's, built from the branch list,
  // so only its two bits are written here. The row must start zeroed: absent fields stay 0.
  template <class F>
  __device__ inline void fetched_to_row(
    const history::History& h, const history::Fetched& fetched, CpuColsU64<F>& cols, F* byte_trace)
  {
    const history::Step& step = fetched.step;
    const InstructionU64& instruction = fetched.instruction;
    const history::Slots slots = fetched.accesses.slots;
    const history::Operands& operands = fetched.operands;

    // No chunk column: the AIR's messages carry the `execution_chunk` public value. One `clk`
    // column; the Rust side asserts clk < 2^25 (the chunk-split budget) and this side does not. No
    // byte event for clk: no such lookup exists (clk is bounded by the row chain, riscv_cpu/chunk_clk).
    cols.clk = F::from_canonical_u32(step.clk);

    // No pc or next_pc range-check events: pc's two limbs are pinned by the Program lookup to the
    // preprocessed table's canonical Pc(pc_base + 4i), and next_pc is bound to the successor row's
    // pc or to a public value the recursion chain pins. Mirrors riscv_cpu/traces.rs.
    write_pc_from_u64(cols.pc, step.pc);
    write_pc_from_u64(cols.next_pc, step.next_pc);

    instruction::populate_instruction_u64<F>(instruction, cols.instruction);
    populate_opcode_selector<F>(instruction, cols.opcode_selector);

    // A present slot's access value overwrites its operand below; an absent one keeps the operand.
    write_word_from_u64(cols.op_a_access.value_mut(), operands.a);
    write_word_from_u64(cols.op_b_access.value_mut(), operands.b);
    write_word_from_u64(cols.op_c_access.value_mut(), operands.c);

    // `a`, read or written. A read's `prev_value` is its own value (history::accesses_of), so one
    // field serves both kinds. No byte event for a jump row's return address: the AIR has no range
    // check on op_a there (pc_carry below is pinned to the Program table's [pc0 >= 0xFFFC]).
    const history::Access& a = fetched.accesses.at[history::SLOT_A];
    if (slots.contains(history::SLOT_A)) {
      write_word_from_u64(cols.op_a_access.prev_value, a.prev_value);
      populate_memory_access_u64<F>(
        cols.op_a_access.access, a.value, h.chunk, history::timestamp(step, history::SLOT_A), a.prev_chunk,
        a.prev_timestamp, byte_trace);
    }
    if (slots.contains(history::SLOT_B) && !slots.writes(history::SLOT_B)) {
      const history::Access& b = fetched.accesses.at[history::SLOT_B];
      populate_memory_access_u64<F>(
        cols.op_b_access.access, b.value, h.chunk, history::timestamp(step, history::SLOT_B), b.prev_chunk,
        b.prev_timestamp, byte_trace);
    }
    if (slots.contains(history::SLOT_C) && !slots.writes(history::SLOT_C)) {
      const history::Access& c = fetched.accesses.at[history::SLOT_C];
      populate_memory_access_u64<F>(
        cols.op_c_access.access, c.value, h.chunk, history::timestamp(step, history::SLOT_C), c.prev_chunk,
        c.prev_timestamp, byte_trace);
    }

    // A branch commits a_lt_b and a_gt_b; `a_eq_b` and the taken predicate are derived in the AIR
    // (CpuChip::branch_taken_expr). Its comparison row is the less-than chip's, from the branch
    // list. A taken branch emits no ADD event: the AIR proves next_pc = pc + c as one field
    // equation. No Cpu row emits an ADD (a JAL target is a field equation, a JALR target four carry
    // bits in-chip, and AUIPC arrives re-encoded as an ALU ADD).
    if (instruction::is_branch_instruction_u64(instruction)) {
      const bool is_signed = instruction.opcode == Opcode::BLT || instruction.opcode == Opcode::BGE;
      const bool a_lt_b =
        is_signed ? static_cast<int64_t>(operands.a) < static_cast<int64_t>(operands.b) : operands.a < operands.b;
      const bool a_gt_b =
        is_signed ? static_cast<int64_t>(operands.a) > static_cast<int64_t>(operands.b) : operands.a > operands.b;
      cols.a_lt_b = F::from_bool(a_lt_b);
      cols.a_gt_b = F::from_bool(a_gt_b);
    }

    // An ECALL's syscall code is what `a` held before the call wrote it (the prev_value limbs
    // above): its low byte names the syscall, so the row commits is_halt (certified by the
    // Cpu->Ecall lookup), and its third byte is the extra cycles. The decode gadget and its byte
    // lookups live on the Ecall chip.
    if (instruction::is_ecall_instruction_u64(instruction)) {
      const uint64_t code = slots.contains(history::SLOT_A) ? a.prev_value : 0;
      cols.is_halt = F::from_bool(
        static_cast<uint8_t>(code & 0xFF) == static_cast<uint8_t>(static_cast<uint32_t>(SyscallCode::HALT)));
      cols.num_extra_clk = F::from_canonical_u8(static_cast<uint8_t>((code >> 16) & 0xFF));
    }

    // pc_carry = [pc0 >= 0xFFFC] on every real row: the Program message carries it, so it must
    // equal the preprocessed table's value for this pc (program.cuh writes the same function). On
    // a jump row with rd != x0 it is the carry out of limb 0 in pc+4 = op_a. Padding rows leave it 0.
    cols.pc_carry = F::from_bool((step.pc & 0xFFFF) >= 0xFFFC);

    cols.is_real = F::one();

    // The ALU dispatch multiplicity, zero when the destination is `x0` (riscv_cpu/traces.rs): the
    // AIR gates the ALU lookup on it.
    const F not_x0 = F::from_bool(instruction.op_a != 0);
    cols.is_alu_not_x0 = F::from_bool(instruction::is_alu_instruction_u64(instruction)) * not_x0;

    // Bit 0 of the unmasked JALR target, which the emulator masked out of `next_pc`.
    cols.jalr_lsb = (instruction.opcode == Opcode::JALR)
                      ? F::from_canonical_u32(static_cast<uint32_t>((operands.b + operands.c) & 1ull))
                      : F::zero();
  }

} // namespace pico_gpu::cpu_u64

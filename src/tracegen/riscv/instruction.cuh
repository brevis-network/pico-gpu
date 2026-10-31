// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::instruction {


  // ─── U64 variants using InstructionU64 ──────────────────────────────

  __PICO_HOSTDEV__ inline bool is_alu_instruction_u64(const InstructionU64& instruction)
  {
    switch (instruction.opcode) {
    case Opcode::ADD:  case Opcode::SUB:  case Opcode::XOR:
    case Opcode::OR:   case Opcode::AND:  case Opcode::SLL:
    case Opcode::SRL:  case Opcode::SRA:  case Opcode::SLT:
    case Opcode::SLTU: case Opcode::MUL:  case Opcode::MULH:
    case Opcode::MULHU:  case Opcode::MULHSU:
    case Opcode::DIV:  case Opcode::DIVU: case Opcode::REM:
    case Opcode::REMU:
    case Opcode::ADDW: case Opcode::SUBW: case Opcode::SLLW:
    case Opcode::SRLW: case Opcode::SRAW: case Opcode::MULW:
    case Opcode::DIVW: case Opcode::DIVUW:
    case Opcode::REMW: case Opcode::REMUW:
      return true;
    default:
      return false;
    }
  }

  __PICO_HOSTDEV__ inline bool is_branch_instruction_u64(const InstructionU64& instruction)
  {
    switch (instruction.opcode) {
    case Opcode::BEQ: case Opcode::BNE: case Opcode::BLT:
    case Opcode::BGE: case Opcode::BLTU: case Opcode::BGEU:
      return true;
    default:
      return false;
    }
  }

  __PICO_HOSTDEV__ inline bool is_ecall_instruction_u64(const InstructionU64& instruction)
  {
    return instruction.opcode == Opcode::ECALL;
  }

  __PICO_HOSTDEV__ inline bool is_jump_instruction_u64(const InstructionU64& instruction)
  {
    return instruction.opcode == Opcode::JAL || instruction.opcode == Opcode::JALR;
  }

  __PICO_HOSTDEV__ inline bool is_memory_instruction_u64(const InstructionU64& instruction)
  {
    switch (instruction.opcode) {
    case Opcode::LB:  case Opcode::LH:  case Opcode::LW:
    case Opcode::LBU: case Opcode::LHU: case Opcode::LWU:
    case Opcode::LD:  case Opcode::SB:  case Opcode::SH:
    case Opcode::SW:  case Opcode::SD:
      return true;
    default:
      return false;
    }
  }

  template <class F>
  __PICO_HOSTDEV__ inline void populate_instruction_u64(const InstructionU64& instruction, InstructionCols<F>& cols)
  {
    cols.opcode = F::from_canonical_u8(static_cast<uint8_t>(instruction.opcode));
    // op_a is a u8 register index, a single limb
    cols.op_a = F::from_canonical_u16(static_cast<uint16_t>(instruction.op_a));
    // op_b and op_c: the two low u16 limbs plus the sign fill of bits 32..64 (Rust
    // `SignExtImm::from_u64`, order limb[0], limb[1], sign). The Rust populate asserts bits
    // 32..64 are all zero or all one; this side takes bit 63 as the sign and does not assert.
    cols.op_b.limb[0] = F::from_canonical_u16(static_cast<uint16_t>(instruction.op_b & 0xFFFF));
    cols.op_b.limb[1] = F::from_canonical_u16(static_cast<uint16_t>((instruction.op_b >> 16) & 0xFFFF));
    cols.op_b.sign = F::from_bool((instruction.op_b >> 63) != 0);
    cols.op_c.limb[0] = F::from_canonical_u16(static_cast<uint16_t>(instruction.op_c & 0xFFFF));
    cols.op_c.limb[1] = F::from_canonical_u16(static_cast<uint16_t>((instruction.op_c >> 16) & 0xFFFF));
    cols.op_c.sign = F::from_bool((instruction.op_c >> 63) != 0);
    cols.op_a_0 = F::from_bool(instruction.op_a == REGISTER_X0);
  }

} // namespace pico_gpu::instruction

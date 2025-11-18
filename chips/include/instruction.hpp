#pragma once

#include <cstdint>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace pico_gpu::instruction {

  template <class F>
  __PICO_HOSTDEV__ inline void populate_instruction(const Instruction& instruction, InstructionCols<F>& cols)
  {
    cols.opcode = F::from_canonical_u8(static_cast<uint8_t>(instruction.opcode));
    write_word_from_u32_v2<F>(cols.op_a, instruction.op_a);
    write_word_from_u32_v2<F>(cols.op_b, instruction.op_b);
    write_word_from_u32_v2<F>(cols.op_c, instruction.op_c);
    cols.op_a_0 = F::from_bool(instruction.op_a == REGISTER_X0);
  } // populate

  __PICO_HOSTDEV__ inline bool is_alu_instruction(const Instruction& instruction)
  {
    switch (instruction.opcode) {
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
      return true;
    default:
      return false;
    }
  } // is_alu_instruction

  __PICO_HOSTDEV__ inline bool is_branch_instruction(const Instruction& instruction)
  {
    switch (instruction.opcode) {
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
  } // is_branch_instruction

  __PICO_HOSTDEV__ inline bool is_ecall_instruction(const Instruction& instruction)
  {
    return instruction.opcode == Opcode::ECALL;
  } // is_ecall_instruction

  __PICO_HOSTDEV__ inline bool is_jump_instruction(const Instruction& instruction)
  {
    switch (instruction.opcode) {
    case Opcode::JAL:
    case Opcode::JALR:
      return true;
    default:
      return false;
    }
  } // is_jump_instruction

  __PICO_HOSTDEV__ inline bool is_memory_instruction(const Instruction& instruction)
  {
    switch (instruction.opcode) {
    case Opcode::LB:
    case Opcode::LH:
    case Opcode::LW:
    case Opcode::LBU:
    case Opcode::LHU:
    case Opcode::SB:
    case Opcode::SH:
    case Opcode::SW:
      return true;
    default:
      return false;
    }
  } // is_memory_instruction

} // namespace pico_gpu::instruction

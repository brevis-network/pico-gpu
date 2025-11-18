#pragma once

#include <cstdint>
#include <iostream>
#include <util/rusterror.h>
#include "field_word_range_checker.hpp"
#include "instruction.hpp"
#include "memory_read_write.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::cpu {
  using namespace byte;
  using namespace memory_read_write;

  /// Populate chunk and clock columns with range checks.
  template <class F>
  __device__ inline void populate_chunk_clk(const FfiCpuEvent& event, CpuCols<F>& cols, F* byte_trace)
  {
    cols.chunk = F::from_canonical_u32(event.chunk);
    cols.clk = F::from_canonical_u32(event.clk);

    // Decompose clock into limbs for range checking.
    const uint16_t clk_16bit_limb = event.clk & 0xffff;
    const uint8_t clk_8bit_limb = (event.clk >> 16) & 0xff;
    cols.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
    cols.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);

    add_u16_range_check<F>(byte_trace, event.chunk);
    add_u16_range_check<F>(byte_trace, clk_16bit_limb);
    add_u8_range_check<F>(byte_trace, clk_8bit_limb, 0);
  }

  /// Populate opcode selector flags for the instruction.
  template <class F>
  __device__ inline void populate_opcode_selector(const Instruction& instruction, OpcodeSelectorCols<F>& cols)
  {
    cols.imm_b = F::from_bool(instruction.imm_b);
    cols.imm_c = F::from_bool(instruction.imm_c);

    if (instruction::is_alu_instruction(instruction)) {
      cols.is_alu = F::one();
    } else if (instruction::is_ecall_instruction(instruction)) {
      cols.is_ecall = F::one();
    } else if (instruction::is_memory_instruction(instruction)) {
      switch (instruction.opcode) {
      case Opcode::LB:
        cols.is_lb = F::one();
        break;
      case Opcode::LBU:
        cols.is_lbu = F::one();
        break;
      case Opcode::LHU:
        cols.is_lhu = F::one();
        break;
      case Opcode::LH:
        cols.is_lh = F::one();
        break;
      case Opcode::LW:
        cols.is_lw = F::one();
        break;
      case Opcode::SB:
        cols.is_sb = F::one();
        break;
      case Opcode::SH:
        cols.is_sh = F::one();
        break;
      case Opcode::SW:
        cols.is_sw = F::one();
        break;
      default:
        printf("unknown memory instruction: %d\n", instruction.opcode);
        break;
      }
    } else if (instruction::is_branch_instruction(instruction)) {
      switch (instruction.opcode) {
      case Opcode::BEQ:
        cols.is_beq = F::one();
        break;
      case Opcode::BNE:
        cols.is_bne = F::one();
        break;
      case Opcode::BLT:
        cols.is_blt = F::one();
        break;
      case Opcode::BGE:
        cols.is_bge = F::one();
        break;
      case Opcode::BLTU:
        cols.is_bltu = F::one();
        break;
      case Opcode::BGEU:
        cols.is_bgeu = F::one();
        break;
      default:
        printf("unknown branch instruction: %d\n", instruction.opcode);
        break;
      }
    } else if (instruction.opcode == Opcode::JAL) {
      cols.is_jal = F::ONE;
    } else if (instruction.opcode == Opcode::JALR) {
      cols.is_jalr = F::ONE;
    } else if (instruction.opcode == Opcode::AUIPC) {
      cols.is_auipc = F::ONE;
    } else if (instruction.opcode == Opcode::UNIMP) {
      cols.is_unimpl = F::ONE;
    }
  }

  /// Populate branch instruction columns and generate ALU events.
  template <class F>
  __device__ inline void populate_branch(
    const FfiCpuEvent& event,
    CpuCols<F>& cols,
    AluEvent* alu_add_sub_events,
    AluEvent* alu_lt_events,
    size_t& alu_add_sub_index,
    size_t& alu_lt_index)
  {
    if (instruction::is_branch_instruction(event.instruction)) {
      BranchCols<F>& branch_columns = cols.opcode_specific.branch;

      const bool a_eq_b = event.a == event.b;

      const bool use_signed_comparison =
        event.instruction.opcode == Opcode::BLT || event.instruction.opcode == Opcode::BGE;

      bool a_lt_b = false;
      bool a_gt_b = false;
      if (use_signed_comparison) {
        a_lt_b = (int32_t)event.a < (int32_t)event.b;
        a_gt_b = (int32_t)event.a > (int32_t)event.b;
      } else {
        a_lt_b = event.a < event.b;
        a_gt_b = event.a > event.b;
      }

      // Generate ALU comparison events.
      const Opcode alu_op_code = use_signed_comparison ? Opcode::SLT : Opcode::SLTU;

      AluEvent lt_comp_event = {
        .pc = event.clk,
        .opcode = alu_op_code,
        .a = static_cast<uint32_t>(a_lt_b),
        .b = event.a,
        .c = event.b,
      };
      alu_lt_events[alu_lt_index] = lt_comp_event;
      alu_lt_index++;

      AluEvent gt_comp_event = {
        .pc = event.clk,
        .opcode = alu_op_code,
        .a = static_cast<uint32_t>(a_gt_b),
        .b = event.b,
        .c = event.a,
      };
      alu_lt_events[alu_lt_index] = gt_comp_event;
      alu_lt_index++;

      branch_columns.a_eq_b = F::from_bool(a_eq_b);
      branch_columns.a_lt_b = F::from_bool(a_lt_b);
      branch_columns.a_gt_b = F::from_bool(a_gt_b);

      // Determine if branch is taken.
      bool branching = false;
      switch (event.instruction.opcode) {
      case Opcode::BEQ:
        branching = a_eq_b;
        break;
      case Opcode::BNE:
        branching = !a_eq_b;
        break;
      case Opcode::BLT:
      case Opcode::BLTU:
        branching = a_lt_b;
        break;
      case Opcode::BGE:
      case Opcode::BGEU:
        branching = a_eq_b || a_gt_b;
        break;
      default:
        printf("unknown branch instruction: %d\n", event.instruction.opcode);
        break;
      }

      const uint32_t next_pc = event.pc + event.c;
      write_word_from_u32_v2<F>(branch_columns.pc, event.pc);
      write_word_from_u32_v2<F>(branch_columns.next_pc, next_pc);
      field_word_range_checker::populate(branch_columns.pc_range_checker, event.pc);
      field_word_range_checker::populate(branch_columns.next_pc_range_checker, next_pc);

      if (branching) {
        cols.branching = F::one();

        AluEvent add_event = {
          .pc = event.clk,
          .opcode = Opcode::ADD,
          .a = next_pc,
          .b = event.pc,
          .c = event.c,
        };
        alu_add_sub_events[alu_add_sub_index] = add_event;
        alu_add_sub_index++;
      } else {
        cols.not_branching = F::one();
      }
    }
  }

  /// Populate jump instruction columns (JAL/JALR).
  template <class F>
  __device__ inline void
  populate_jump(const FfiCpuEvent& event, CpuCols<F>& cols, AluEvent* alu_add_sub_events, size_t& alu_add_sub_index)
  {
    if (instruction::is_jump_instruction(event.instruction)) {
      JumpCols<F>& jump_columns = cols.opcode_specific.jump;

      switch (event.instruction.opcode) {
      case Opcode::JAL: {
        const uint32_t next_pc = event.pc + event.b;
        field_word_range_checker::populate(jump_columns.op_a_range_checker, event.a);
        write_word_from_u32_v2<F>(jump_columns.pc, event.pc);
        field_word_range_checker::populate(jump_columns.pc_range_checker, event.pc);
        write_word_from_u32_v2<F>(jump_columns.next_pc, next_pc);
        field_word_range_checker::populate(jump_columns.next_pc_range_checker, next_pc);

        AluEvent add_event = {
          .pc = event.clk,
          .opcode = Opcode::ADD,
          .a = next_pc,
          .b = event.pc,
          .c = event.b,
        };
        alu_add_sub_events[alu_add_sub_index] = add_event;
        alu_add_sub_index++;

        break;
      }
      case Opcode::JALR: {
        const uint32_t next_pc = event.b + event.c;
        field_word_range_checker::populate(jump_columns.op_a_range_checker, event.a);
        write_word_from_u32_v2<F>(jump_columns.next_pc, next_pc);
        field_word_range_checker::populate(jump_columns.next_pc_range_checker, next_pc);

        AluEvent add_event = {
          .pc = event.clk,
          .opcode = Opcode::ADD,
          .a = next_pc,
          .b = event.b,
          .c = event.c,
        };
        alu_add_sub_events[alu_add_sub_index] = add_event;
        alu_add_sub_index++;

        break;
      }
      default:
        printf("unknown jump instruction: %d\n", event.instruction.opcode);
        break;
      }
    }
  }

  /// Populate AUIPC instruction columns.
  template <class F>
  __device__ inline void
  populate_auipc(const FfiCpuEvent& event, CpuCols<F>& cols, AluEvent* alu_add_sub_events, size_t& alu_add_sub_index)
  {
    if (event.instruction.opcode == Opcode::AUIPC) {
      AuipcCols<F>& auipc_columns = cols.opcode_specific.auipc;

      write_word_from_u32_v2<F>(auipc_columns.pc, event.pc);
      field_word_range_checker::populate(auipc_columns.pc_range_checker, event.pc);

      AluEvent add_event = {
        .pc = event.clk,
        .opcode = Opcode::ADD,
        .a = event.a,
        .b = event.pc,
        .c = event.b,
      };
      alu_add_sub_events[alu_add_sub_index] = add_event;
      alu_add_sub_index++;
    }
  }

  /// Populate ECALL instruction columns.
  /// Returns true if this is a HALT syscall.
  template <class F>
  __device__ inline bool populate_ecall(const FfiCpuEvent& event, CpuCols<F>& cols)
  {
    bool is_halt = false;

    if (cols.opcode_selector.is_ecall == F::one()) {
      EcallCols<F>& ecall_cols = cols.opcode_specific.ecall;

      cols.ecall_mul_send_to_table = cols.opcode_selector.is_ecall * cols.op_a_access.prev_value._0[1];

      F syscall_id = cols.op_a_access.prev_value._0[0];

      // Populate syscall type indicators.
      is_zero::populate_from_field_element(
        ecall_cols.is_enter_unconstrained,
        syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::ENTER_UNCONSTRAINED)));

      is_zero::populate_from_field_element(
        ecall_cols.is_hint_len, syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::HINT_LEN)));

      is_zero::populate_from_field_element(
        ecall_cols.is_halt, syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::HALT)));

      is_zero::populate_from_field_element(
        ecall_cols.is_commit, syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT)));

      is_zero::populate_from_field_element(
        ecall_cols.is_commit_deferred_proofs,
        syscall_id - F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS)));

      // Handle COMMIT and COMMIT_DEFERRED_PROOFS syscalls.
      if (
        (syscall_id == F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT))) ||
        (syscall_id == F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS)))) {
        const uint32_t digest_idx = word_to_u32(cols.op_b_access.access.value);
        ecall_cols.index_bitmap[digest_idx] = F::one();
      }

      is_halt = syscall_id == F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::HALT));

      // Range check operands for HALT and COMMIT_DEFERRED_PROOFS.
      if (is_halt) {
        write_word_from_u32_v2<F>(ecall_cols.operand_to_check, event.b);
        field_word_range_checker::populate(ecall_cols.operand_range_check_cols, event.b);
        cols.ecall_range_check_operand = F::one();
      }

      if (syscall_id == F::from_canonical_u32(static_cast<uint32_t>(SyscallCode::COMMIT_DEFERRED_PROOFS))) {
        write_word_from_u32_v2<F>(ecall_cols.operand_to_check, event.c);
        field_word_range_checker::populate(ecall_cols.operand_range_check_cols, event.c);
        cols.ecall_range_check_operand = F::one();
      }
    }

    return is_halt;
  }

  /// Initialize padding row with dummy values.
  template <class F>
  __device__ inline void assign_padding_row(CpuCols<F>& cols)
  {
    cols.opcode_selector.imm_b = F::one();
    cols.opcode_selector.imm_c = F::one();
  }

  /// Convert CPU event to trace row.
  template <class F>
  __device__ void event_to_row(
    const FfiCpuEvent& event, CpuCols<F>& cols, F* byte_trace, AluEvent* alu_add_sub_events, AluEvent* alu_lt_events)
  {
    populate_chunk_clk<F>(event, cols, byte_trace);

    // Populate basic fields.
    cols.pc = F::from_canonical_u32(event.pc);
    cols.next_pc = F::from_canonical_u32(event.next_pc);

    instruction::populate_instruction<F>(event.instruction, cols.instruction);
    populate_opcode_selector<F>(event.instruction, cols.opcode_selector);

    write_word_from_u32_v2(cols.op_a_access.value_mut(), event.a);
    write_word_from_u32_v2(cols.op_b_access.value_mut(), event.b);
    write_word_from_u32_v2(cols.op_c_access.value_mut(), event.c);

    // Populate memory accesses for operands.
    if (event.a_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some) {
      populate(cols.op_a_access, event.a_record.some._0, byte_trace);
    }
    if (
      event.b_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some &&
      event.b_record.some._0.tag == MemoryRecordEnum::Tag::Read) {
      populate(cols.op_b_access, event.b_record.some._0.read._0, byte_trace);
    }
    if (
      event.c_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some &&
      event.c_record.some._0.tag == MemoryRecordEnum::Tag::Read) {
      populate(cols.op_c_access, event.c_record.some._0.read._0, byte_trace);
    }

    // Add byte range checks for operand A.
    for (int i = 0; i < WORD_SIZE; i += 2) {
      add_u8_range_check<F>(
        byte_trace, static_cast<uint8_t>(cols.op_a_access.access.value._0[i].as_canonical_u32() & 0xFF),
        static_cast<uint8_t>(cols.op_a_access.access.value._0[i + 1].as_canonical_u32() & 0xFF));
    }

    size_t alu_add_sub_index = 0;
    size_t alu_lt_index = 0;

    populate_branch<F>(event, cols, alu_add_sub_events, alu_lt_events, alu_add_sub_index, alu_lt_index);
    populate_jump<F>(event, cols, alu_add_sub_events, alu_add_sub_index);
    populate_auipc<F>(event, cols, alu_add_sub_events, alu_add_sub_index);
    const bool is_halt = populate_ecall(event, cols);

    cols.is_sequential_instr = F::from_bool(
      !instruction::is_branch_instruction(event.instruction) && !instruction::is_jump_instruction(event.instruction) &&
      !is_halt);

    cols.is_real = F::one();
  }

  /// Kernel to convert CPU events to trace rows.
  template <class F>
  __global__ void cpu_events_to_trace_kernel(
    const size_t event_size,
    const FfiCpuEvent* events,
    F* __restrict__ trace_matrix,
    const size_t num_cols,
    const CpuExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    AluEvent* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < event_size) {
      CpuCols<F>* cols = reinterpret_cast<CpuCols<F>*>(&trace_matrix[idx * num_cols]);
      event_to_row<F>(
        events[idx], *cols, byte_trace,
        &alu_add_sub_events[alu_add_sub_events_start_index + extra_event_indices[idx].add],
        &alu_lt_events[alu_lt_events_start_index + extra_event_indices[idx].lt]);
    }
  }

  /// Kernel to pad trace with dummy rows.
  template <class F>
  __global__ void cpu_pad_trace_kernel(
    const size_t num_cols, const size_t num_rows_to_pad, F* __restrict__ trace_matrix_start_of_padding)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows_to_pad) {
      CpuCols<F>* cols = reinterpret_cast<CpuCols<F>*>(&trace_matrix_start_of_padding[idx * num_cols]);
      assign_padding_row<F>(*cols);
    }
  }

  /// Pad trace with dummy rows.
  template <class F>
  inline RustError
  pad_trace(const size_t num_cols, const size_t start_row, const size_t end_row, F* trace, cudaStream_t stream)
  {
    try {
      if (start_row >= end_row) return RustError{cudaSuccess};

      const size_t num_rows_to_pad = end_row - start_row;
      F* trace_ptr_start_of_padding = &trace[start_row * num_cols];

      // Initialize padding area with zeros.
      CUDA_OK(cudaMemsetAsync(&trace_ptr_start_of_padding[0], 0, num_rows_to_pad * num_cols * sizeof(F), stream));

      const int block_size = 256;
      const int grid_size = (num_rows_to_pad + block_size - 1) / block_size;

      cpu_pad_trace_kernel<F>
        <<<grid_size, block_size, 0, stream>>>(num_cols, num_rows_to_pad, trace_ptr_start_of_padding);

    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }

  /// Generate trace from CPU events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const FfiCpuEvent* events,
    F* trace,
    cudaStream_t stream,
    const CpuExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index,
    AluEvent* alu_lt_events,
    const size_t alu_lt_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, event_size * num_cols * sizeof(F), stream));

      // Launch kernel to process events.
      if (event_size > 0) {
        const int block_size = 256;
        const int grid_size = (event_size + block_size - 1) / block_size;

        cpu_events_to_trace_kernel<F><<<grid_size, block_size, 0, stream>>>(
          event_size, events, trace, num_cols, extra_event_indices, byte_trace, alu_add_sub_events,
          alu_add_sub_events_start_index, alu_lt_events, alu_lt_events_start_index);
        CUDA_OK(cudaGetLastError());
      }
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::cpu
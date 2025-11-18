#pragma once

#include <cstddef>
#include <iostream>
#include <util/rusterror.h>
#include "field_word_range_checker.hpp"
#include "prelude.hpp"
#include "utils.hpp"
#include "byte.hpp"

namespace pico_gpu::memory_read_write {
  using namespace byte;

  /// Populate memory-related columns for load/store instructions.
  template <class F>
  __device__ void
  populate_memory(MemoryChipValueCols<F>& col, const FfiCpuEvent& event, F* byte_trace, AluEvent* alu_add_sub_events)
  {
    // Calculate memory address and alignment.
    const uint32_t memory_addr = event.b + event.c;
    const uint32_t aligned_addr = memory_addr - memory_addr % WORD_SIZE;
    write_word_from_u32_v2(col.addr_word, memory_addr);
    field_word_range_checker::populate(col.addr_word_range_checker, memory_addr);
    col.addr_aligned = F::from_canonical_u32(aligned_addr);

    // Decompose aligned address least significant byte.
    assert(aligned_addr % 4 == 0);
    for (int i = 0; i < 6; ++i)
      col.aa_least_sig_byte_decomp[i] = F::from_bool((aligned_addr >> (i + 2)) & 1);

    // Generate ALU ADD event for address calculation.
    size_t add_sub_index = 0;
    AluEvent add_event = {
      .pc = event.clk,
      .opcode = Opcode::ADD,
      .a = memory_addr,
      .b = event.b,
      .c = event.c,
    };
    alu_add_sub_events[add_sub_index] = add_event;
    add_sub_index++;

    // Populate memory offset within word.
    const uint8_t addr_offset = memory_addr % WORD_SIZE;
    col.addr_offset = F::from_canonical_u8(addr_offset);
    col.offset_is_one = F::from_bool(addr_offset == 1);
    col.offset_is_two = F::from_bool(addr_offset == 2);
    col.offset_is_three = F::from_bool(addr_offset == 3);

    // Extract memory value from record.
    uint32_t mem_value;
    if (event.memory_record.some._0.tag == MemoryRecordEnum::Tag::Read) {
      mem_value = event.memory_record.some._0.read._0.value;
    } else {
      mem_value = event.memory_record.some._0.write._0.value;
    }

    // Handle load instructions.
    if (
      event.instruction.opcode == Opcode::LB || event.instruction.opcode == Opcode::LBU ||
      event.instruction.opcode == Opcode::LH || event.instruction.opcode == Opcode::LHU ||
      event.instruction.opcode == Opcode::LW) {
      switch (event.instruction.opcode) {
      case Opcode::LB:
      case Opcode::LBU:
        write_word_from_u32_v2(col.unsigned_mem_val, u32_to_le_bytes(mem_value)[addr_offset]);
        break;
      case Opcode::LH:
      case Opcode::LHU: {
        uint32_t value;
        if ((addr_offset >> 1) % 2 == 0) {
          value = mem_value & 0x0000FFFF;
        } else {
          value = (mem_value & 0xFFFF0000) >> 16;
        }
        write_word_from_u32_v2(col.unsigned_mem_val, value);
        break;
      }
      case Opcode::LW:
        write_word_from_u32_v2(col.unsigned_mem_val, mem_value);
        break;
      default:
        break;
      }

      // Handle signed load instructions (LB, LH).
      if (event.instruction.opcode == Opcode::LB || event.instruction.opcode == Opcode::LH) {
        uint8_t most_sig_mem_value_byte;
        uint32_t sign_value;

        if (event.instruction.opcode == Opcode::LB) {
          sign_value = 256;
          most_sig_mem_value_byte = u32_to_le_bytes(word_to_u32(col.unsigned_mem_val))[0];
        } else {
          sign_value = 65536;
          most_sig_mem_value_byte = u32_to_le_bytes(word_to_u32(col.unsigned_mem_val))[1];
        }

        // Decompose most significant byte.
        for (int i = 7; i >= 0; --i) {
          col.most_sig_byte_decomp[i] = F::from_canonical_u8(most_sig_mem_value_byte >> i & 0x01);
        }

        // Handle negative values (sign extension).
        if (col.most_sig_byte_decomp[7] == F::one()) {
          col.mem_value_is_neg_not_x0 = F::from_bool(event.instruction.op_a != static_cast<uint8_t>(Register::X0));

          AluEvent sub_event = {
            .pc = event.clk,
            .opcode = Opcode::SUB,
            .a = event.a,
            .b = word_to_u32(col.unsigned_mem_val),
            .c = sign_value,
          };
          alu_add_sub_events[add_sub_index] = sub_event;
        }
      }

      // Set positive value flag.
      col.mem_value_is_pos_not_x0 = F::from_bool(
        (((event.instruction.opcode == Opcode::LB || event.instruction.opcode == Opcode::LH) &&
          (col.most_sig_byte_decomp[7] == F::zero())) ||
         (event.instruction.opcode == Opcode::LBU || event.instruction.opcode == Opcode::LHU ||
          event.instruction.opcode == Opcode::LW)) &&
        event.instruction.op_a != static_cast<uint8_t>(Register::X0));
    }

    // Add byte range checks for address.
    const array_t<uint8_t, WORD_SIZE> addr_bytes = u32_to_le_bytes(memory_addr);
    for (int i = 0; i < WORD_SIZE; i += 2) {
      add_u8_range_check(byte_trace, addr_bytes[i], addr_bytes[i + 1]);
    }
  }

  /// Populate memory access columns with timing verification.
  template <class F>
  __device__ void populate_access(
    MemoryAccessCols<F>& col, const MemoryRecord& current_record, const MemoryRecord& prev_record, F* byte_trace)
  {
    write_word_from_u32_v2(col.value, current_record.value);
    col.prev_chunk = F::from_canonical_u32(prev_record.chunk);
    col.prev_clk = F::from_canonical_u32(prev_record.timestamp);

    // Verify current access time > previous access time.
    const bool use_clk_comparison = prev_record.chunk == current_record.chunk;
    col.compare_clk = F::from_bool(use_clk_comparison);

    const uint32_t prev_time_value = use_clk_comparison ? prev_record.timestamp : prev_record.chunk;
    const uint32_t current_time_value = use_clk_comparison ? current_record.timestamp : current_record.chunk;

    const uint32_t diff_minus_one = current_time_value - prev_time_value - 1;
    const uint16_t diff_16bit_limb = (diff_minus_one & 0xffff);
    col.diff_16bit_limb = F::from_canonical_u16(diff_16bit_limb);
    const uint32_t diff_8bit_limb = (diff_minus_one >> 16) & 0xff;
    col.diff_8bit_limb = F::from_canonical_u32(diff_8bit_limb);

    add_u16_range_check(byte_trace, diff_16bit_limb);
    add_u8_range_check(byte_trace, static_cast<uint8_t>(diff_8bit_limb), 0);
  }

  /// Populate read-only memory columns.
  template <class F>
  __device__ void populate(MemoryReadCols<F>& col, const MemoryReadRecord& record, F* byte_trace)
  {
    const MemoryRecord current_record = {
      record.chunk,
      record.timestamp,
      record.value,
    };
    const MemoryRecord prev_record = {
      record.prev_chunk,
      record.prev_timestamp,
      record.value,
    };
    populate_access(col.access, current_record, prev_record, byte_trace);
  }

  /// Populate write memory columns with previous value.
  template <class F>
  __device__ void populate(MemoryWriteCols<F>& col, const MemoryWriteRecord& record, F* byte_trace)
  {
    const MemoryRecord current_record = {
      record.chunk,
      record.timestamp,
      record.value,
    };
    const MemoryRecord prev_record = {
      record.prev_chunk,
      record.prev_timestamp,
      record.prev_value,
    };
    write_word_from_u32_v2(col.prev_value, prev_record.value);
    populate_access(col.access, current_record, prev_record, byte_trace);
  }

  /// Populate read/write columns for write operation.
  template <class F>
  __device__ void populate_write(MemoryReadWriteCols<F>& col, const MemoryWriteRecord& record, F* byte_trace)
  {
    const MemoryRecord current_record = {
      record.chunk,
      record.timestamp,
      record.value,
    };
    const MemoryRecord prev_record = {
      record.prev_chunk,
      record.prev_timestamp,
      record.prev_value,
    };
    write_word_from_u32_v2(col.prev_value, prev_record.value);
    populate_access(col.access, current_record, prev_record, byte_trace);
  }

  /// Populate read/write columns for read operation.
  template <class F>
  __device__ void populate_read(MemoryReadWriteCols<F>& col, const MemoryReadRecord& record, F* byte_trace)
  {
    const MemoryRecord current_record = {
      record.chunk,
      record.timestamp,
      record.value,
    };
    const MemoryRecord prev_record = {
      record.prev_chunk,
      record.prev_timestamp,
      record.value,
    };
    write_word_from_u32_v2(col.prev_value, prev_record.value);
    populate_access(col.access, current_record, prev_record, byte_trace);
  }

  /// Populate memory instruction columns.
  template <class F>
  __device__ void mem_inst_populate(MemoryInstructionCols<F>& col, const FfiCpuEvent& event)
  {
    const auto opcode = event.instruction.opcode;
    col.opcode = F::from_canonical_u8(static_cast<uint8_t>(opcode));
    col.op_a_0 = F::from_bool(event.instruction.op_a == static_cast<uint8_t>(Register::X0));

    // Set instruction type flags.
    switch (opcode) {
    case Opcode::LB:
      col.is_lb = F::one();
      break;
    case Opcode::LBU:
      col.is_lbu = F::one();
      break;
    case Opcode::LHU:
      col.is_lhu = F::one();
      break;
    case Opcode::LH:
      col.is_lh = F::one();
      break;
    case Opcode::LW:
      col.is_lw = F::one();
      break;
    case Opcode::SB:
      col.is_sb = F::one();
      break;
    case Opcode::SH:
      col.is_sh = F::one();
      break;
    case Opcode::SW:
      col.is_sw = F::one();
      break;
    default:
      break;
    }

    // Populate operand access columns.
    write_word_from_u32_v2(col.op_a_access.value_mut(), event.a);
    write_word_from_u32_v2(col.op_b_access.value_mut(), event.b);
    write_word_from_u32_v2(col.op_c_access.value_mut(), event.c);

    // Override with actual memory record values if available.
    if (event.a_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some) {
      write_word_from_u32_v2(col.op_a_access.value_mut(), event.a_record.some._0.value());
    }
    if (
      event.b_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some &&
      event.b_record.some._0.tag == MemoryRecordEnum::Tag::Read) {
      write_word_from_u32_v2(col.op_b_access.value_mut(), event.b_record.some._0.read._0.value);
    }
    if (
      event.c_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some &&
      event.c_record.some._0.tag == MemoryRecordEnum::Tag::Read) {
      write_word_from_u32_v2(col.op_c_access.value_mut(), event.c_record.some._0.read._0.value);
    }
  }

  /// Populate read/write columns based on record type.
  template <class F>
  __device__ void populate(MemoryReadWriteCols<F>& col, const MemoryRecordEnum& record, F* byte_trace)
  {
    switch (record.tag) {
    case MemoryRecordEnum::Tag::Read:
      populate_read(col, record.read._0, byte_trace);
      break;
    case MemoryRecordEnum::Tag::Write:
      populate_write(col, record.write._0, byte_trace);
      break;
    }
  }

  /// Convert memory read/write event to trace row.
  template <class F>
  __device__ void
  event_to_row(const FfiCpuEvent& event, MemoryChipValueCols<F>& col, F* byte_trace, AluEvent* alu_add_sub_events)
  {
    col.chunk = F::from_canonical_u32(event.chunk);
    col.clk = F::from_canonical_u32(event.clk);

    // Populate memory access if record exists.
    if (event.memory_record.tag == FfiOption<MemoryRecordEnum>::Tag::Some) {
      populate(col.memory_access, event.memory_record.some._0, byte_trace);
    }

    mem_inst_populate(col.instruction, event);
    populate_memory(col, event, byte_trace, alu_add_sub_events);
  }

  /// Kernel to convert memory read/write events to trace.
  template <class F>
  __global__ void memory_read_write_events_to_trace_kernel(
    const FfiCpuEvent* events,
    const size_t* event_indices,
    const size_t count,
    F* trace_matrix,
    const size_t num_cols,
    const MemoryReadWriteExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MemoryChipValueCols<F>* col = reinterpret_cast<MemoryChipValueCols<F>*>(&trace_matrix[idx * num_cols]);
    event_to_row<F>(
      events[event_indices[idx]], *col, byte_trace,
      &alu_add_sub_events[alu_add_sub_events_start_index + extra_event_indices[idx].add_sub]);
  }

  /// Generate trace from memory read/write events.
  template <class F>
  inline RustError generate_extra_record_and_main(
    const size_t num_cols,
    const size_t event_size,
    const size_t trace_size,
    const FfiCpuEvent* events,
    const size_t* event_indices,
    F* trace,
    cudaStream_t stream,
    const MemoryReadWriteExtraEventIndices* extra_event_indices,
    F* byte_trace,
    AluEvent* alu_add_sub_events,
    const size_t alu_add_sub_events_start_index)
  {
    try {
      if (event_size == 0) return RustError{cudaSuccess};

      // Initialize trace matrix with zeros.
      CUDA_OK(cudaMemsetAsync(trace, 0, trace_size * sizeof(F), stream));

      // Launch kernel.
      const int block_size = 256;
      const int grid_size = (event_size + block_size - 1) / block_size;

      memory_read_write_events_to_trace_kernel<<<grid_size, block_size, 0, stream>>>(
        events, event_indices, event_size, trace, num_cols, extra_event_indices, byte_trace, alu_add_sub_events,
        alu_add_sub_events_start_index);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }

    return RustError{cudaSuccess};
  }
} // namespace pico_gpu::memory_read_write
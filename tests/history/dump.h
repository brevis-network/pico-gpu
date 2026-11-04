// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Reads a history dump (pico's vm/src/emulator/riscv/history_dump.rs) on the host: a header, then
// tagged sections of rows, little-endian, each padded to a multiple of 8.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "tracegen/riscv/history_check.cuh"

namespace history_dump {

  constexpr char MAGIC[8] = {'P', 'I', 'C', 'O', 'H', 'D', 'M', 'P'};
  constexpr uint32_t VERSION = 1;

  enum Section : uint32_t {
    FILL = 1,
    PROGRAM = 2,
    PROGRAM_BASE = 3,
    CPU_EVENTS = 10,
    MEMORY_INDICES = 11,
    ADD_SUB = 12,
    MUL = 13,
    BITWISE = 14,
    SHIFT_LEFT = 15,
    SHIFT_RIGHT = 16,
    DIVREM = 17,
    LT = 18,
    SYSCALLS = 19,
    BRANCH_COMPARES = 20,
  };

  struct Rows {
    uint32_t row_bytes = 0;
    uint64_t rows = 0;
    // Words, so that every section is 8-aligned in memory as the fill's tables need.
    std::vector<uint64_t> words;
    bool present = false;

    const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(words.data()); }
  };

  struct Dump {
    uint32_t chunk = 0;
    Rows sections[21];

    const Rows& section(Section tag) const { return sections[tag]; }
  };

  // Reads the dump at `path` into `dump`; an empty string, or what is wrong with the file.
  inline std::string read(const char* path, Dump& dump)
  {
    FILE* file = std::fopen(path, "rb");
    if (!file) return "cannot open the file";
    auto take = [&](void* out, size_t bytes) { return std::fread(out, 1, bytes, file) == bytes; };
    char magic[8];
    uint32_t header[4];
    std::string error;
    if (!take(magic, 8) || std::memcmp(magic, MAGIC, 8) != 0) {
      error = "not a history dump";
    } else if (!take(header, sizeof header)) {
      error = "a header cut short";
    } else if (header[0] != VERSION || header[1] != pico_gpu::history::HISTORY_LAYOUT) {
      error = "a dump of version " + std::to_string(header[0]) + ", layout " + std::to_string(header[1]) +
              "; this reader takes version " + std::to_string(VERSION) + ", layout " +
              std::to_string(pico_gpu::history::HISTORY_LAYOUT);
    } else {
      dump.chunk = header[2];
      for (uint32_t s = 0; s < header[3] && error.empty(); ++s) {
        uint32_t tag_and_size[2];
        uint64_t rows = 0;
        if (!take(tag_and_size, sizeof tag_and_size) || !take(&rows, sizeof rows)) {
          error = "a section header cut short";
          break;
        }
        const uint32_t tag = tag_and_size[0];
        if (tag >= sizeof dump.sections / sizeof dump.sections[0] || dump.sections[tag].present) {
          error = "section " + std::to_string(tag) + " unknown or repeated";
          break;
        }
        Rows& section = dump.sections[tag];
        section.present = true;
        section.row_bytes = tag_and_size[1];
        section.rows = rows;
        const uint64_t bytes = rows * section.row_bytes;
        section.words.assign((bytes + 7) / 8, 0);
        if (!take(section.words.data(), section.words.size() * 8))
          error = "section " + std::to_string(tag) + " cut short";
      }
    }
    std::fclose(file);
    return error;
  }

  // The dump's stream rows, in history_check's stream order; empty if a section is missing or its
  // rows are not the size history_check reads.
  inline std::string expected(const Dump& dump, pico_gpu::history_check::Expected& expected)
  {
    using namespace pico_gpu::history_check;
    const struct {
      Section tag;
      uint32_t row_bytes;
    } streams[STREAMS] = {
      {CPU_EVENTS, sizeof(CpuRow)},
      {MEMORY_INDICES, 8},
      {ADD_SUB, sizeof(AluRow)},
      {MUL, sizeof(AluRow)},
      {BITWISE, sizeof(AluRow)},
      {SHIFT_LEFT, sizeof(AluRow)},
      {SHIFT_RIGHT, sizeof(AluRow)},
      {DIVREM, sizeof(AluRow)},
      {LT, sizeof(AluRow)},
      {SYSCALLS, sizeof(pico_gpu::SyscallEventU64)},
      {BRANCH_COMPARES, sizeof(AluRow)},
    };
    for (uint32_t stream = 0; stream < STREAMS; ++stream) {
      const Rows& rows = dump.section(streams[stream].tag);
      if (!rows.present || rows.row_bytes != streams[stream].row_bytes)
        return "section " + std::to_string(streams[stream].tag) + " missing or of another row size";
      expected.rows[stream] = rows.words.data();
      expected.counts[stream] = rows.rows;
    }
    return "";
  }

} // namespace history_dump

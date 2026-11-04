// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The loaders on the host: for each dump named, every stream the emulator wrote is rebuilt from the
// history through history.cuh, under both exception lookups, and compared element for element with
// the dump's. Around that, the paths no stream takes: the CPU chip's slices, the opening of a fill
// whose completion word has not landed, and launches naming another fill or program table, each of
// which must set its own error bit alone. Exits 0 when every row of every dump is equal, no loader
// set an error bit, and every probe holds.
#include <cstdio>
#include <cstring>
#include <vector>

#include "dump.h"

using namespace pico_gpu;

namespace {

  const char* const STREAM_NAMES[history_check::STREAMS] = {
    "cpu", "memory", "add_sub", "mul", "bitwise", "shift_left",
    "shift_right", "divrem", "lt", "syscall", "branch_compare",
  };

  template <history::ExceptionLookup Lookup>
  bool check_streams(const history::History& h, const history_check::Expected& expected, const char* lookup)
  {
    bool equal = true;
    for (uint32_t stream = 0; stream < history_check::STREAMS; ++stream) {
      const uint64_t rebuilt = history_check::rebuilt_rows(h, stream), held = expected.counts[stream];
      const uint64_t rows = rebuilt < held ? rebuilt : held;
      uint64_t mismatched = 0, first = 0;
      for (uint64_t row = 0; row < rows; ++row) {
        if (history_check::row_matches<Lookup>(h, expected, stream, static_cast<uint32_t>(row))) continue;
        if (mismatched++ == 0) first = row;
      }
      const bool same = rebuilt == held && mismatched == 0;
      equal = equal && same;
      std::printf(
        "  %-6s %-14s %10llu rows", lookup, STREAM_NAMES[stream], static_cast<unsigned long long>(held));
      if (rebuilt != held) std::printf(", %llu rebuilt", static_cast<unsigned long long>(rebuilt));
      if (mismatched)
        std::printf(
          ", %llu differ, the first at %llu", static_cast<unsigned long long>(mismatched),
          static_cast<unsigned long long>(first));
      std::printf("%s\n", same ? "" : "  MISMATCH");
    }
    return equal;
  }

  // The CPU chip's slices: consecutive, every step in one, none past the last, each as the
  // descriptor cuts them, and the slice after the last refused with the slice bit alone.
  bool check_slices(history::History h)
  {
    uint32_t error = 0;
    h.error = &error;
    const uint32_t slices = history::cpu_slices(h);
    bool held = h.step_count == 0 || slices > 0;
    uint32_t next = 0;
    for (uint32_t slice = 0; slice < slices && held; ++slice) {
      uint32_t first = 0, end = 0;
      held = history::cpu_slice(h, slice, first, end) && first == next && end > first &&
             end - first <= h.cpu_slice_steps && (end - first == h.cpu_slice_steps || end == h.step_count);
      next = end;
    }
    held = held && next == h.step_count && error == 0;
    uint32_t first = 0, end = 0;
    held = held && !history::cpu_slice(h, slices, first, end) && error == history::ERROR_SLICE;
    std::printf("  %u slice(s) of %u steps%s\n", slices, h.cpu_slice_steps, held ? "" : "  FAIL");
    return held;
  }

  // Opens `fill` as `args` names it, but for `change`; true when the opening ends as `opens` says
  // with exactly `bits` set.
  bool probe(
    const char* name,
    history::HistoryArgs args,
    const std::vector<uint64_t>& fill,
    history::Readiness readiness,
    bool opens,
    uint32_t bits)
  {
    uint32_t error = 0;
    args.fill = reinterpret_cast<const uint8_t*>(fill.data());
    args.error = &error;
    history::History h;
    const bool opened = history::open(args, readiness, h);
    const bool held = opened == opens && error == bits;
    std::printf(
      "  %-44s %s, error word %#06x%s\n", name, opened ? "opens" : "refused", error, held ? "" : "  FAIL");
    return held;
  }

  bool check_openings(const history::HistoryArgs& args, const history_dump::Rows& fill)
  {
    using history::Readiness;
    const auto& descriptor = *reinterpret_cast<const history::FillDescriptor*>(fill.bytes());
    std::vector<uint64_t> landed = fill.words, unlanded = fill.words;
    reinterpret_cast<uint8_t*>(unlanded.data())[descriptor.completion_at] ^= 1;
    history::HistoryArgs other_nonce = args, other_chunk = args, other_steps = args, other_program = args;
    other_nonce.expected.nonce ^= 1;
    other_chunk.expected.chunk += 1;
    other_steps.expected.steps += 1;
    other_program.program.key ^= 1;
    const struct {
      const char* name;
      const history::HistoryArgs& args;
      const std::vector<uint64_t>& fill;
      Readiness readiness;
      bool opens;
      uint32_t bits;
    } probes[] = {
      {"landed, for a CPU slice", args, landed, Readiness::Descriptor, true, 0},
      {"completion word unlanded", args, unlanded, Readiness::Complete, false, history::ERROR_INCOMPLETE},
      {"completion word unlanded, for a CPU slice", args, unlanded, Readiness::Descriptor, true, 0},
      {"launched for another nonce", other_nonce, landed, Readiness::Complete, false, history::ERROR_IDENTITY},
      {"launched for another chunk", other_chunk, landed, Readiness::Descriptor, false, history::ERROR_IDENTITY},
      {"launched for another step count", other_steps, landed, Readiness::Complete, false, history::ERROR_IDENTITY},
      {"launched against another program table", other_program, landed, Readiness::Complete, false,
       history::ERROR_PROGRAM},
    };
    bool held = true;
    for (const auto& p : probes)
      held = probe(p.name, p.args, p.fill, p.readiness, p.opens, p.bits) && held;
    return held;
  }

  bool check_dump(const char* path)
  {
    history_dump::Dump dump;
    std::string error = history_dump::read(path, dump);
    history_check::Expected expected = {};
    if (error.empty()) error = history_dump::expected(dump, expected);
    const history_dump::Rows& fill = dump.section(history_dump::FILL);
    const history_dump::Rows& program = dump.section(history_dump::PROGRAM);
    const history_dump::Rows& base = dump.section(history_dump::PROGRAM_BASE);
    if (error.empty() && (!fill.present || !program.present || program.row_bytes != sizeof(InstructionU64) ||
                          !base.present || base.rows != 2))
      error = "the fill or the program table is missing";
    if (!error.empty()) {
      std::printf("%s: %s\n", path, error.c_str());
      return false;
    }
    std::vector<InstructionU64> instructions(program.rows);
    if (program.rows) std::memcpy(instructions.data(), program.bytes(), program.rows * sizeof(InstructionU64));
    const history::FillDescriptor& descriptor = *reinterpret_cast<const history::FillDescriptor*>(fill.bytes());
    uint32_t error_word = 0;
    history::HistoryArgs args = {};
    args.fill = fill.bytes();
    args.fill_bytes = fill.rows;
    args.expected = descriptor.identity;
    args.program = history::ProgramTable{instructions.data(), program.rows, base.words[0], base.words[1]};
    args.error = &error_word;
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) {
      std::printf("%s: the fill does not open, error word %#x\n", path, error_word);
      return false;
    }
    std::printf(
      "%s: chunk %u, %u steps, %u exceptions\n", path, dump.chunk, h.step_count, h.exception_count);
    for (uint32_t exception = 0; exception < h.exception_count; ++exception)
      history::exception_in_order(h, exception);
    for (uint32_t entry = 0; entry < h.list_start[history::LISTS]; ++entry)
      history::entry_in_order(h, entry);
    bool equal = check_slices(h);
    equal = check_openings(args, fill) && equal;
    equal = check_streams<history::ExceptionLookup::Search>(h, expected, "search") && equal;
    const uint32_t words = history::exception_words(h.step_count);
    std::vector<uint32_t> bits(words), first(words);
    for (uint32_t word = 0; word < words; ++word)
      history::build_exception_word(h, word, bits.data(), first.data());
    h.exception_bits = bits.data();
    h.exception_first = first.data();
    equal = check_streams<history::ExceptionLookup::Bitmap>(h, expected, "bitmap") && equal;
    if (error_word) std::printf("  the loaders set error word %#x\n", error_word);
    return equal && error_word == 0;
  }

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::printf("usage: %s <dump>...\n", argv[0]);
    return 2;
  }
  bool equal = true;
  for (int arg = 1; arg < argc; ++arg)
    equal = check_dump(argv[arg]) && equal;
  std::printf("%s\n", equal ? "every stream equal" : "FAILED");
  return equal ? 0 : 1;
}

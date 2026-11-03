// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The loaders on a card: every stream the emulator wrote, rebuilt from a history dump's fill on the
// device through history.cuh, one thread per row, and compared element for element with the
// dump's rows (history_check.cuh). Pico's vm/examples/history_self_test.rs reads a dump and calls
// the entry; the host's loader tests are the same comparison in a loop.
#include <cstdint>
#include <vector>

#include "common/exception.cuh"
#include "common/rusterror.h"
#include "tracegen/common/prelude.cuh"
#include "tracegen/riscv/history_check.cuh"
#include "tracegen/riscv/history_validate.cuh"

namespace pico_gpu::history_self_test {

  // Rows of every stream, each stream starting at its `start`.
  struct Plan {
    uint64_t start[history_check::STREAMS + 1];
  };

  // Per stream: the rows the history rebuilds, the rows that differ, and the first that does.
  struct Results {
    unsigned long long rebuilt[history_check::STREAMS];
    unsigned long long mismatched[history_check::STREAMS];
    unsigned long long first[history_check::STREAMS];
  };

  __global__ void count_kernel(history::HistoryArgs args, Results* results)
  {
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) return;
    for (uint32_t stream = threadIdx.x; stream < history_check::STREAMS; stream += blockDim.x)
      results->rebuilt[stream] = history_check::rebuilt_rows(h, stream);
  }

  __global__ void index_kernel(history::HistoryArgs args, uint32_t* bits, uint32_t* first)
  {
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) return;
    const uint32_t words = history::exception_words(h.step_count);
    for (uint32_t word = blockIdx.x * blockDim.x + threadIdx.x; word < words; word += gridDim.x * blockDim.x)
      history::build_exception_word(h, word, bits, first);
  }

  template <history::ExceptionLookup Lookup>
  __global__ void check_kernel(history::HistoryArgs args, history_check::Expected expected, Plan plan, Results* results)
  {
    history::History h;
    if (!history::open(args, history::Readiness::Complete, h)) return;
    const uint64_t rows = plan.start[history_check::STREAMS];
    for (uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < rows;
         at += uint64_t(gridDim.x) * blockDim.x) {
      uint32_t stream = 0;
      while (at >= plan.start[stream + 1])
        ++stream;
      const uint64_t row = at - plan.start[stream];
      if (row >= history_check::rebuilt_rows(h, stream)) continue;
      if (history_check::row_matches<Lookup>(h, expected, stream, static_cast<uint32_t>(row))) continue;
      atomicAdd(&results->mismatched[stream], 1ull);
      atomicMin(&results->first[stream], static_cast<unsigned long long>(row));
    }
  }

  // Device allocations freed together, on every path out.
  struct Allocations {
    std::vector<void*> pointers;

    template <class T>
    T* copy_of(const void* host, size_t bytes)
    {
      void* device = nullptr;
      CUDA_OK(cudaMalloc(&device, bytes ? bytes : 1));
      pointers.push_back(device);
      if (bytes) CUDA_OK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
      return static_cast<T*>(device);
    }

    ~Allocations()
    {
      for (void* pointer : pointers)
        cudaFree(pointer);
    }
  };

} // namespace pico_gpu::history_self_test

// Uploads a dump's fill and program table and its stream rows (`rows[s]`, `counts[s]` rows of
// stream s, in history_check's stream order and row layouts), validates the fill as an upload does,
// rebuilds every stream on the device under the exception lookup `bitmap` names, and writes per
// stream the rows rebuilt, the rows that differ and the first that does to `rebuilt`, `mismatched`
// and `first`, and the loaders' error word to `error_word`. The kernels are launched for
// `*expected`, or for the fill's own identity when it is null, so a launch naming another fill can
// be tested. Synchronous on the current device.
PICO_FFI RustError::by_value cuda_history_self_test(
  uint32_t layout,
  const uint8_t* fill,
  uint64_t fill_bytes,
  const pico_gpu::InstructionU64* program,
  uint64_t program_rows,
  uint64_t pc_base,
  uint64_t program_key,
  const void* const* rows,
  const uint64_t* counts,
  uint32_t bitmap,
  const pico_gpu::history::FillIdentity* expected_identity,
  uint64_t* rebuilt,
  uint64_t* mismatched,
  uint64_t* first,
  uint32_t* error_word)
{
  using namespace pico_gpu;
  using namespace pico_gpu::history_self_test;
  if (layout != history::HISTORY_LAYOUT || fill_bytes < sizeof(history::FillDescriptor))
    return RustError{cudaErrorInvalidValue};
  const uint64_t row_bytes[history_check::STREAMS] = {
    sizeof(history_check::CpuRow), 8, sizeof(history_check::AluRow), sizeof(history_check::AluRow),
    sizeof(history_check::AluRow), sizeof(history_check::AluRow), sizeof(history_check::AluRow),
    sizeof(history_check::AluRow), sizeof(history_check::AluRow), sizeof(SyscallEventU64),
    sizeof(history_check::AluRow),
  };
  try {
    Allocations allocations;
    history::HistoryArgs args = {};
    args.fill = allocations.copy_of<uint8_t>(fill, fill_bytes);
    args.fill_bytes = fill_bytes;
    args.expected = expected_identity ? *expected_identity
                                      : reinterpret_cast<const history::FillDescriptor*>(fill)->identity;
    args.program = history::ProgramTable{
      allocations.copy_of<InstructionU64>(program, program_rows * sizeof(InstructionU64)), program_rows, pc_base,
      program_key};
    const uint32_t zero = 0;
    args.error = allocations.copy_of<uint32_t>(&zero, sizeof zero);
    history_check::Expected expected = {};
    Plan plan = {};
    for (uint32_t stream = 0; stream < history_check::STREAMS; ++stream) {
      expected.rows[stream] = allocations.copy_of<void>(rows[stream], counts[stream] * row_bytes[stream]);
      expected.counts[stream] = counts[stream];
      plan.start[stream + 1] = plan.start[stream] + counts[stream];
    }
    Results initial = {};
    for (uint32_t stream = 0; stream < history_check::STREAMS; ++stream)
      initial.first[stream] = ~0ull;
    Results* results = allocations.copy_of<Results>(&initial, sizeof initial);

    count_kernel<<<1, 32>>>(args, results);
    CUDA_OK(cudaGetLastError());
    const int block = 256, grid = 1024;
    const RustError validated = history_validate::validate(args, 0);
    if (validated.code != cudaSuccess) return validated;
    if (bitmap) {
      const auto& descriptor = *reinterpret_cast<const history::FillDescriptor*>(fill);
      const size_t words = history::exception_words(descriptor.identity.steps);
      std::vector<uint32_t> none(words ? words : 1, 0);
      uint32_t* bits = allocations.copy_of<uint32_t>(none.data(), none.size() * sizeof(uint32_t));
      uint32_t* first_exception = allocations.copy_of<uint32_t>(none.data(), none.size() * sizeof(uint32_t));
      index_kernel<<<grid, block>>>(args, bits, first_exception);
      CUDA_OK(cudaGetLastError());
      args.exception_bits = bits;
      args.exception_first = first_exception;
      check_kernel<history::ExceptionLookup::Bitmap><<<grid, block>>>(args, expected, plan, results);
    } else {
      check_kernel<history::ExceptionLookup::Search><<<grid, block>>>(args, expected, plan, results);
    }
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    Results done = {};
    CUDA_OK(cudaMemcpy(&done, results, sizeof done, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(error_word, args.error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t stream = 0; stream < history_check::STREAMS; ++stream) {
      rebuilt[stream] = done.rebuilt[stream];
      mismatched[stream] = done.mismatched[stream];
      first[stream] = done.first[stream];
    }
  } catch (const cuda_error& e) {
    return RustError{e.code()};
  }
  return RustError{cudaSuccess};
}

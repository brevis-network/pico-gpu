// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The checks of a whole fill that no loader can make from the rows it reads: the order of the
// exception table, which both lookups search as sorted, and of every chip list, on which the merge
// of the add and sub lists and each chip's row order rely. One launch per fill, after every table
// has landed and before the completion word, so every reader that waits for the completion word
// reads a checked fill; a failure sets the fill's error word, read with the others after the
// chunk's last reader.
#pragma once

#include "common/exception.cuh"
#include "common/rusterror.h"
#include "tracegen/riscv/history.cuh"

namespace pico_gpu::history_validate {

  static __global__ void validate_kernel(history::HistoryArgs args)
  {
    history::History h;
    if (!history::open(args, history::Readiness::Descriptor, h)) return;
    const uint32_t entries = h.list_start[history::LISTS];
    const uint32_t rows = h.exception_count > entries ? h.exception_count : entries;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x) {
      if (row < h.exception_count) history::exception_in_order(h, row);
      if (row < entries) history::entry_in_order(h, row);
    }
  }

  // Launches the checks of the fill `args` names on `stream`.
  inline RustError validate(const history::HistoryArgs& args, cudaStream_t stream)
  {
    try {
      validate_kernel<<<1024, 256, 0, stream>>>(args);
      CUDA_OK(cudaGetLastError());
    } catch (const cuda_error& e) {
      return RustError{e.code()};
    }
    return RustError{cudaSuccess};
  }

} // namespace pico_gpu::history_validate

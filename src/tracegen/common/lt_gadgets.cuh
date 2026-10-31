// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cassert>
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

namespace pico_gpu::lt_gadgets {

  template <class F, size_t N>
  __PICO_HOSTDEV__ inline void populate(AssertLtColsBits<F, N>& gadget, const uint32_t* a, const uint32_t* b)
  {
    uint8_t bit_flags[N];
    for (int i = 0; i < N; ++i)
      bit_flags[i] = 0;

    for (int i = N - 1; i >= 0; --i) {
      if (a[i] < b[i]) {
        bit_flags[i] = 1;
        break;
      }
    }

    for (int i = 0; i < N; ++i) {
      gadget.bit_flags[i] = F::from_canonical_u8(bit_flags[i]);
    }
  }

} // namespace pico_gpu::lt_gadgets
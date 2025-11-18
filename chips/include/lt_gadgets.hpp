#pragma once

#include <cassert>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

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
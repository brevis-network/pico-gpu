#pragma once

#include <cstdint>
#include <util/rusterror.h>

#include "is_zero.hpp"
#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace is_zero = pico_gpu::is_zero;

namespace pico_gpu::is_zero_word {

  template <class F>
  __PICO_HOSTDEV__ inline bool populate_from_field_element(IsZeroWordOperation<F>& gadget, Word<F>& a)
  {
    bool is_zero = true;
    for (uintptr_t i = 0; i < WORD_SIZE; ++i) {
      is_zero &= is_zero::populate_from_field_element(gadget.is_zero_byte[i], a._0[i]) == 1;
    }
    gadget.is_lower_half_zero = gadget.is_zero_byte[0].result * gadget.is_zero_byte[1].result;
    gadget.is_upper_half_zero = gadget.is_zero_byte[2].result * gadget.is_zero_byte[3].result;
    gadget.result = F::from_bool(is_zero);
    return is_zero;
  } // populate_from_field_element

  template <class F>
  __PICO_HOSTDEV__ inline bool populate(IsZeroWordOperation<F>& gadget, const uint32_t a_u32)
  {
    Word<F> a{F::zero()};
    write_word_from_u32_v2(a, a_u32);

    return populate_from_field_element(gadget, a);
  } // populate

} // namespace pico_gpu::is_zero_word

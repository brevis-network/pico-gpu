#pragma once

#include <util/rusterror.h>

#include "is_zero_word.hpp"
#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace is_zero_word = pico_gpu::is_zero_word;

namespace pico_gpu::is_equal_word {

  template <class F>
  __PICO_HOSTDEV__ inline bool populate(IsEqualWordOperation<F>& gadget, const uint32_t a_u32, const uint32_t b_u32)
  {
    array_t<uint8_t, 4> a = u32_to_le_bytes(a_u32);
    array_t<uint8_t, 4> b = u32_to_le_bytes(b_u32);

    Word<F> diff{F::zero()};
    diff._0[0] = F::from_canonical_u8(a[0]) - F::from_canonical_u8(b[0]);
    diff._0[1] = F::from_canonical_u8(a[1]) - F::from_canonical_u8(b[1]);
    diff._0[2] = F::from_canonical_u8(a[2]) - F::from_canonical_u8(b[2]);
    diff._0[3] = F::from_canonical_u8(a[3]) - F::from_canonical_u8(b[3]);

    is_zero_word::populate_from_field_element(gadget.is_diff_zero, diff);

    return a_u32 == b_u32;
  } // populate

} // namespace pico_gpu::is_equal_word

// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

#include <cstdint>
#include "field/ff_config.hpp"
#include "hash/poseidon2/constants.cuh"
#include "hash/poseidon2/kernels.cuh"
#include "common/rusterror.h"

#include "tracegen/common/prelude.cuh"
#include "tracegen/common/types.cuh"
#include "tracegen/common/utils.cuh"

using namespace poseidon2;

namespace pico_gpu::global_interaction {
  __PICO_HOSTDEV__ inline void populate_dummy(GlobalInteractionOperation<field_t>* col)
  {
    // 1. offset = 0
    col->offset = field_t::zero();

    // 2. x_coordinate = CURVE_WITNESS_DUMMY_POINT_X
    for (int i = 0; i < 7; i++) {
      col->x_coordinate._0[i] = septic_curve_t::dummy_x[i];
    }

    // 3. y_coordinate = CURVE_WITNESS_DUMMY_POINT_Y
    for (int i = 0; i < 7; i++) {
      col->y_coordinate._0[i] = septic_curve_t::dummy_y[i];
    }

    // 4. y6_limbs = 0
    for (int i = 0; i < 3; i++) {
      col->y6_limbs[i] = field_t::zero();
    }
  }

  __PICO_HOSTDEV__ inline FfiOption<Poseidon2Event> populate(
    GlobalInteractionOperation<field_t>* col,
    const uint32_t values[8],
    bool is_receive,
    bool is_real,
    uint8_t kind,
    const Poseidon2Constants<field_t>& poseidon2_constants)
  {
    if (!is_real) {
      populate_dummy(col);
      FfiOption<Poseidon2Event> op;
      op.tag = FfiOption<Poseidon2Event>::Tag::None;
      return op;
    }

    septic_curve_t point;
    uint8_t offset = 0;
    field_t m_trial[16];
    field_t m_hash[16];
    {
      // kind packed above the 25-bit values[0] budget; mirrors
      // Rust get_digest. Capacity: (2^25-1) + 31*2^25 = 2^30-1 < p.
      m_trial[0] = field_t::from_canonical_u32(values[0]) + field_t::from_canonical_u32(uint32_t(kind) << GLOBAL_KIND_SHIFT);
      for (int i = 1; i < 7; i++) {
        m_trial[i] = field_t::from_canonical_u32(values[i]);
      }
      field_t base7 = field_t::from_canonical_u32(values[7]);
      for (int i = 8; i < 16; i++) {
        m_trial[i] = field_t::zero();
      }

      bool found = false;
      for (int trial = 0; trial <= 255; ++trial) {
        offset = static_cast<uint8_t>(trial);
        m_trial[7] = base7 + field_t::from_canonical_u32(1u << 16) * field_t::from_canonical_u8(offset);
        for (int i = 0; i < 16; i++) {
          m_hash[i] = m_trial[i];
        }
        permute_state<field_t, 16>(m_hash, poseidon2_constants);

        auto x_trial = septic_extension_t(m_hash);
        auto y_sq = x_trial.curve_formula();
        auto y = y_sq.sqrt(y_sq.pow_r());
        if (!(y == septic_extension_t::zero())) {
          if (y.is_exception()) continue;
          if (y.is_send()) {
            point.x = x_trial;
            point.y = -y;
          } else {
            point.x = x_trial;
            point.y = y;
          }
          found = true;
          break;
        }
      }
      assert(found && "curve point couldn't be found after 256 attempts");

      if (!is_receive) { point = point.neg(); }
    }

    col->offset = field_t::from_canonical_u8(offset);

    for (int i = 0; i < 7; i++) {
      col->x_coordinate._0[i] = point.x.value[i];
      col->y_coordinate._0[i] = point.y.value[i];
    }

    uint32_t y6_u32 = point.y.value[6].as_canonical_u32();
    uint32_t range_check_value = is_receive ? (y6_u32 - 1) : (y6_u32 - ((field_t::MOD + 1) >> 1));

    // Compact limb decomposition [lo16, mid, top] of range_check_value (see types.cuh).
    const uint32_t split = Y6_RANGE_BITS - field_t::TOP_BITS; // 23
    const uint32_t mid_mask = (1u << (split - Y6_LO_LIMB_BITS)) - 1; // 0x7f
    const uint32_t top_mask = (1u << field_t::TOP_BITS) - 1; // 0x7f
    col->y6_limbs[0] = field_t::from_canonical_u32(range_check_value & 0xFFFF);
    col->y6_limbs[1] = field_t::from_canonical_u32((range_check_value >> Y6_LO_LIMB_BITS) & mid_mask);
    col->y6_limbs[2] = field_t::from_canonical_u32((range_check_value >> split) & top_mask);

    assert(col->x_coordinate._0[0] == m_hash[0]);

    FfiOption<Poseidon2Event> op;
    op.tag = FfiOption<Poseidon2Event>::Tag::Some;
    for (int i = 0; i < 16; i++) {
      op.some._0.input[i] = m_trial[i].as_canonical_u32();
      op.some._0.output[i] = m_hash[i].as_canonical_u32();
    }
    return op;
  }
} // namespace pico_gpu::global_interaction

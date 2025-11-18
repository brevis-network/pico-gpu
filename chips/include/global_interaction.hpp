#pragma once

#include <cstdint>
#include <ff/ff_config.hpp>
#include <poseidon2/constants.cuh>
#include <util/rusterror.h>

#include "prelude.hpp"
#include "types.hpp"
#include "utils.hpp"

using namespace poseidon2;

namespace pico_gpu::global_interaction {
  __PICO_HOSTDEV__ inline void populate_dummy(GlobalInteractionOperation<field_t>* col)
  {
    // 1. offset_bits = 0
    for (int i = 0; i < 8; i++) {
      col->offset_bits[i] = field_t::zero();
    }

    // 2. x_coordinate = CURVE_WITNESS_DUMMY_POINT_X
    for (int i = 0; i < 7; i++) {
      col->x_coordinate._0[i] = septic_curve_t::dummy_x[i];
    }

    // 3. y_coordinate = CURVE_WITNESS_DUMMY_POINT_Y
    for (int i = 0; i < 7; i++) {
      col->y_coordinate._0[i] = septic_curve_t::dummy_y[i];
    }

    // 4. y6_bit_decomp = 0
    for (int i = 0; i < 30; i++) {
      col->y6_bit_decomp[i] = field_t::zero();
    }

    // 5. range_check_witness = 0
    col->range_check_witness = field_t::zero();

    // 6. poseidon2_input/output = [0; 16]
    for (int i = 0; i < PERMUTATION_WIDTH; i++) {
      col->poseidon2_input[i] = field_t::zero();
      col->poseidon2_output[i] = field_t::zero();
    }
  }

  __PICO_HOSTDEV__ inline FfiOption<Poseidon2Event> populate(
    GlobalInteractionOperation<field_t>* col,
    const uint32_t values[7],
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

    // get_digest
    septic_curve_t point;
    uint8_t offset;
    field_t m_trial[16];
    field_t m_hash[16];
    {
      septic_extension_t x_start;
      for (int i = 0; i < 7; i++) {
        x_start.value[i] = field_t::from_canonical_u32(values[i]);
      }
      field_t kind_shifted = field_t::from_canonical_u32(uint32_t(kind) << 16);
      x_start.value[0] += kind_shifted;
      septic_curve_t::lift_x(x_start, point, offset, m_trial, m_hash, poseidon2_constants);

      if (!is_receive) { point = point.neg(); }
    }

    for (int i = 0; i < 8; i++) {
      col->offset_bits[i] = field_t::from_canonical_u8((offset >> i) & 1);
    }

    for (int i = 0; i < 7; i++) {
      col->x_coordinate._0[i] = point.x.value[i];
      col->y_coordinate._0[i] = point.y.value[i];
    }

    uint32_t y6_u32 = point.y.value[6].as_canonical_u32();
    uint32_t range_check_value = is_receive ? (y6_u32 - 1) : (y6_u32 - ((field_t::MOD + 1) >> 1));

    field_t top_field_bits = field_t::zero();
    for (int i = 0; i < 30; i++) {
      field_t bit = field_t::from_canonical_u32((range_check_value >> i) & 1);
      col->y6_bit_decomp[i] = bit;
      if (i >= 30 - field_t::TOP_BITS) { top_field_bits += bit; }
    }

    top_field_bits -= field_t::from_canonical_u32(field_t::TOP_BITS);
    col->range_check_witness = top_field_bits.reciprocal();

    assert(col->x_coordinate._0[0] == m_hash[0]);

    for (int i = 0; i < 16; i++) {
      col->poseidon2_input[i] = m_trial[i];
      col->poseidon2_output[i] = m_hash[i];
    }

    FfiOption<Poseidon2Event> op;
    op.tag = FfiOption<Poseidon2Event>::Tag::Some;
    for (int i = 0; i < 16; i++) {
      op.some._0.input[i] = col->poseidon2_input[i].as_canonical_u32();
      op.some._0.output[i] = col->poseidon2_output[i].as_canonical_u32();
    }
    return op;
  }
} // namespace pico_gpu::global_interaction

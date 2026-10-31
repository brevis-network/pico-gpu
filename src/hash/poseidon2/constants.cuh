// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef POSEIDON2_CONSTANTS_CUH
#define POSEIDON2_CONSTANTS_CUH

#include "hash/poseidon2/params.cuh"

namespace poseidon2 {
  /// A Poseidon2 instance as the permutation kernels in `kernels.cuh` see it. `round_constants`
  /// is a caller-owned device buffer in the layout `params.cuh` describes.
  template <typename S>
  struct Poseidon2Constants {
    int width;
    int alpha;
    int internal_rounds;
    int external_rounds;
    const S* round_constants = nullptr;
  };
} // namespace poseidon2

#endif

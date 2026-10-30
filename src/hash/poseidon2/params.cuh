// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

#pragma once

// Shape of the KoalaBear width-16 Poseidon2 permutation, shared by both implementations in this
// directory: kernels.cuh (trace generation, every round's state exposed) and poseidon2.cuh
// (the prover's hashing, tuned for throughput).
//
// Round structure of the KoalaBear width-16 Poseidon2 as Plonky3 defines it: R_F = 8 external
// rounds (4 + 4), R_P = 20 internal rounds, S-box x^3. The round constants are not compiled in:
// every kernel receives them from its caller as a device buffer holding one WIDTH-wide row per
// round -- the four initial external rounds, the 20 internal rounds with the constant in lane 0
// and zeros elsewhere, then the four terminal external rounds -- which is the layout the Rust
// side's `RoundConstants::flatten_for_gpu` produces.
namespace poseidon2 {
  constexpr int state_width = 16;
  constexpr int half_external_rounds = 4;
  constexpr int internal_rounds = 20;
  constexpr int alpha = 3;
  constexpr int EXTERNAL_ROUNDS = 8;
} // namespace poseidon2

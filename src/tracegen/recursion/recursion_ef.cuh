// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared extension-field multiply for recursion chips: KoalaBear^5 = F[X] / (X^5 + X^2 - 1).

#pragma once

namespace pico_gpu {

  // out = a * b in the quintic trinomial extension; matches Plonky3's `trinomial_quintic_mul`,
  // which the Rust tracegens call. Every product is formed before `out` is written, so `out` may
  // alias an input.
  template <class F>
  __device__ __forceinline__ void
  ef_mul(const F* a, const F* b, F* out)
  {
    // Convolution c_k = sum_{i+j=k} a_i b_j.
    F c0 = a[0] * b[0];
    F c1 = a[0] * b[1] + a[1] * b[0];
    F c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];
    F c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
    F c4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];
    F c5 = a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];
    F c6 = a[2] * b[4] + a[3] * b[3] + a[4] * b[2];
    F c7 = a[3] * b[4] + a[4] * b[3];
    F c8 = a[4] * b[4];
    // X^5 = 1 - X^2, X^6 = X - X^3, X^7 = X^2 - X^4, X^8 = X^3 + X^2 - 1.
    F c5_minus_c8 = c5 - c8;
    out[0] = c0 + c5_minus_c8;
    out[1] = c1 + c6;
    out[2] = c2 - c5_minus_c8 + c7;
    out[3] = c3 - c6 + c8;
    out[4] = c4 - c7;
  }

} // namespace pico_gpu

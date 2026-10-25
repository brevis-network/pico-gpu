// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.
//
// Shared degree-4 binomial-extension (X^4 - W) multiply for recursion chips.

#pragma once

namespace pico_gpu {

  // out = a * b in the degree-4 binomial extension with constant `w`; matches the Rust `ef_mul`.
  template <class F>
  __device__ __forceinline__ void
  ef4_mul(const F* a, const F* b, F* out, const F& w)
  {
    F a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    F b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = a0 * b0 + w * (a1 * b3 + a2 * b2 + a3 * b1);
    out[1] = a0 * b1 + a1 * b0 + w * (a2 * b3 + a3 * b2);
    out[2] = a0 * b2 + a1 * b1 + a2 * b0 + w * (a3 * b3);
    out[3] = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
  }

} // namespace pico_gpu

// Copyright (c) 2025-2026 Brevis. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BUSL-1.1).
// See LICENSE.md for details.

#pragma once

#include "tracegen/common/types.cuh"
#include <cstddef>
#include <cstdint>
#include "common/device_ptr.hpp"

#ifndef __CUDACC__
  #define __PICO_HOSTDEV__
  #define __PICO_NOINLINE__
  #include <array>
  #include <tuple>
  #include <utility>

namespace pico_gpu {
  template <class T, std::size_t N>
  using array_t = std::array<T, N>;

  template <class T1, class T2>
  using tuple_t = std::tuple<T1, T2>;

  template <class T1, class T2>
  using pair_t = std::pair<T1, T2>;
} // namespace pico_gpu
#else
  #define __PICO_HOSTDEV__ __host__ __device__
  #define __PICO_NOINLINE__ __noinline__
  #include <cuda/std/array>
  #include <cuda/std/tuple>
  #include <cuda/std/utility>

namespace pico_gpu {
  template <class T, std::size_t N>
  using array_t = cuda::std::array<T, N>;

  template <class T1, class T2>
  using tuple_t = cuda::std::tuple<T1, T2>;

  template <class T1, class T2>
  using pair_t = cuda::std::pair<T1, T2>;
} // namespace pico_gpu
#endif

#define PICO_FFI extern "C" __attribute__((visibility("default")))

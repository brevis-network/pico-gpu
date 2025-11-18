#pragma once

#include "prelude.hpp"

namespace pico_gpu::warmup {
  __global__ void warmup_kernel()
  {
    // do nothing, but cannot be empty, otherwise it may be optimized
    volatile int _temp = 1;
  }

  inline void warmup()
  {
    warmup_kernel<<<1, 32>>>();
    warmup_kernel<<<1, 64>>>();
    warmup_kernel<<<1, 128>>>();
    warmup_kernel<<<1, 256>>>();
    warmup_kernel<<<1, 512>>>();

    cudaError_t _err = cudaDeviceSynchronize();
  }
} // namespace pico_gpu::warmup

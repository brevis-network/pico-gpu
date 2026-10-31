# pico-gpu

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**pico-gpu** holds the CUDA kernels of the [Pico zkVM](https://github.com/brevis-network/pico): GPU
trace generation for Pico's RISC-V, precompile and recursion chips, and the GPU side of Pico's
multilinear prover (LogUp-GKR, zerocheck, stacked reduction, WHIR).

> **⚠️ Disclaimer**: This project is still a work in progress and is evolving quickly. It has not
> been audited and is **not** recommended for production use; use it at your own risk.

## How it is built

pico-gpu is a source library for Pico and is not built on its own. The Rust crate in `rust/`
contains no code: through its `links` key it hands the location of this repository
(`DEP_PICO_GPU_ROOT`) to the Pico crates that depend on it, and those crates compile the kernels
when Pico is built with its GPU feature:

| Pico crate | Compiles | Into |
|---|---|---|
| `cuda-mle` | `src/prover/` and the NTT sources in `third_party/sppark/` | the static libraries `cuda_kb_all` and `mle-kernels` |
| `gpu-api` | its own compilation units, which include the headers in `src/tracegen/` | the trace-generation library |

Every source includes the others by its path from `src/` (`"field/fp.h"`, `"tracegen/riscv/cpu.cuh"`),
so `src/` is the only include root besides the sppark directories.

To build Pico with GPU proving, follow Pico's documentation; the GPU feature is `cuda-mle`, for
example:

```bash
cargo build --release --bin mle-single-node --features "cuda-mle bigint-rug tcmalloc mmap-memory"
```

GPU targets:

- `cuda-mle` compiles for the compute capability of the first GPU `nvidia-smi` reports, plus PTX.
  Set `CUDA_ARCH` (e.g. `CUDA_ARCH=120`, comma-separated for several) to choose the targets
  yourself, for instance on a build machine without a GPU.
- `gpu-api` embeds PTX that the driver compiles when the library loads. Set `GPU_TARGET_ARCHS`
  (e.g. `GPU_TARGET_ARCHS=120`) to compile it ahead of time instead.

## Requirements

- Linux, an NVIDIA GPU and its driver (`nvidia-smi` on `PATH`)
- CUDA Toolkit with `nvcc` on `PATH`, recent enough for your GPU
- The Rust toolchain pinned by Pico's `rust-toolchain`

Tested with CUDA 13 on GeForce RTX 5090 (compute capability 12.0).

## Repository layout

| Path | Contents |
|---|---|
| `src/common/` | Launch helpers, error handling, out-of-memory recovery, shared device utilities |
| `src/field/` | KoalaBear field and extensions: `kb31_t` (trace generation) and `fp.h` / `fpext.h` (prover) |
| `src/hash/poseidon2/` | KoalaBear width-16 Poseidon2: the trace-generation kernels and the prover's hashing, sharing one parameter header |
| `src/prover/` | Multilinear prover kernels: sumcheck, WHIR, stacked reduction, Merkle commitment, sponge |
| `src/prover/dag/` | Encoding and evaluation of the constraint DAG |
| `src/prover/logup_zerocheck/` | LogUp-GKR and zerocheck kernels |
| `src/prover/ntt/` | Small in-kernel NTTs and the Reed-Solomon encoding fast path |
| `src/tracegen/` | Trace generation, grouped into `common/`, `riscv/`, `precompiles/` and `recursion/` |
| `third_party/sppark/` | Code derived from sppark (Apache-2.0), see its [README](third_party/sppark/README.md) |
| `rust/` | The `pico-gpu` crate that exports this repository's location to Pico |

## Security

If you discover a security vulnerability, please see [SECURITY.md](SECURITY.md) for responsible
disclosure instructions.

## Acknowledgements

pico-gpu builds on prior work in GPU-accelerated zero-knowledge proving:

- [sppark](https://github.com/supranational/sppark): Portions of this library, especially in
  `third_party/sppark/`, `src/common/` and `src/field/kb31_t.hpp`, are ported and adapted from
  the sppark project, licensed under the Apache License 2.0 (see
  [third_party/sppark/LICENSE](third_party/sppark/LICENSE)).
- [SP1](https://github.com/succinctlabs/sp1): Parts of pico-gpu's trace-generation design are
  inspired by SP1.
- [Plonky3](https://github.com/Plonky3/Plonky3): pico-gpu's Poseidon2 kernels use Plonky3's
  KoalaBear parameters and match its permutation bit for bit.
- [OpenVM](https://github.com/openvm-org/openvm): Parts of pico-gpu's multilinear prover kernels are
  adapted from OpenVM's [stark-backend](https://github.com/openvm-org/stark-backend), licensed under
  MIT or Apache-2.0.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option, except for the code derived from third-party projects,
which keeps its original licence:

- [sppark](https://github.com/supranational/sppark) (Apache-2.0, see
  [third_party/sppark/LICENSE](third_party/sppark/LICENSE)): `third_party/sppark/`,
  `src/common/device_ptr.hpp`, `src/common/exception.hpp`, `src/common/exception.cuh`,
  `src/common/rusterror.h` and `src/field/kb31_t.hpp`.
- [RISC Zero](https://github.com/risc0/risc0) (Apache-2.0), by way of OpenVM: `src/field/fp.h` and
  `src/field/fpext.h`.
- [OpenVM stark-backend](https://github.com/openvm-org/stark-backend) (MIT or Apache-2.0,
  Copyright (c) 2025 The OpenVM Authors): parts of `src/prover/`, `src/common/launcher.cuh` and
  `src/common/utils.cuh`.

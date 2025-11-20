# pico-gpu

**pico-gpu** is the GPU proving engine for the Pico zkVM. It currently contains high-performance CUDA/C++ kernels for finite-field arithmetic, hashing, elliptic-curve operations, and GPU trace generation.

This project is still a work in progress and is evolving quickly. It has not been audited and is **not** recommended for production use; use it at your own risk.

## Repository Structure

The repository is organized into the following components:

- **`chips/`**: GPU trace generators for Pico’s chips, covering RISC-V, recursion, and precompile chips.
- **`ff/`**: Finite-field implementations, including base fields and extension fields for the KoalaBear and BabyBear fields.
- **`poseidon2/`**: CUDA kernels and parameter tables for the Poseidon2 permutation over KoalaBear and BabyBear fields.
- **`util/`**: Shared CUDA utilities such as device buffers and error-handling helpers adapted from the `sppark` project.
- **`rust/`**: The Rust crate exposing `pico-gpu`’s CUDA entry points for integration with the Pico zkVM.

## Acknowledgements

pico-gpu builds on prior work in GPU-accelerated zero-knowledge proving:

- [sppark](https://github.com/supranational/sppark): Portions of this library, especially in `util/`, are ported and adapted from the `sppark` project.
- [SP1](https://github.com/succinctlabs/sp1): The BabyBear field implementation under `ff/` is borrowed from SP1, and parts of pico-gpu’s trace-generation design are inspired by SP1.

## License

This project is licensed under the terms described in [LICENSE.md](LICENSE.md).

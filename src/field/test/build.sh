#!/usr/bin/env bash
# Copyright (c) 2025-2026 Brevis.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Build the extension-field benchmark and test. Same include paths and nvcc flags as the prover
# kernels (src and third_party/sppark; c++17, relaxed constexpr, per-thread default
# stream), for the RTX 5090 (sm_120). -Xptxas -v prints registers/spills.
#   ext_bench  : extension-field cost (needs vectors/ from e3_ext_vectors)
#   ext5_test  : kb31_5_t differential test (needs vectors/kb5_mul.bin from e3_ext_vectors)
# vectors/: `cargo run --release -p pico-mle-core --example e3_ext_vectors -- <this dir>/vectors`
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../.."
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
flags=(-O3 -std=c++17 --expt-relaxed-constexpr --extended-lambda --default-stream=per-thread
       -gencode arch=compute_120,code=sm_120 -Xptxas -v)
for b in ${@:-ext_bench ext5_test}; do
    "$NVCC" "${flags[@]}" -I "$root/src" -I "$root/third_party/sppark" \
        -I "$here/vectors" -o "$here/$b" "$here/$b.cu" 2> "$here/ptxas_$b.log"
    grep -E "Compiling entry|registers|spill" "$here/ptxas_$b.log" | paste - - - |
        sed -E 's/ptxas info *: //g' > "$here/regs_$b.txt" || true
    echo "built $here/$b (registers: $here/regs_$b.txt)"
done

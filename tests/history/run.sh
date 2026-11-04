#!/usr/bin/env bash
# Copyright (c) 2025-2026 Brevis.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Builds the host loader tests under the address and undefined-behaviour sanitizers, runs the
# corrupt-fill cases, then the loaders on the history dumps named, if any.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
flags=(-std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  -Wall -Wextra -Werror -I "$here/../../src")
"${CXX:-c++}" "${flags[@]}" "$here/corrupt.cpp" -o "$build/corrupt"
"${CXX:-c++}" "${flags[@]}" "$here/loaders.cpp" -o "$build/loaders"
"$build/corrupt"
if [ "$#" -gt 0 ]; then
  "$build/loaders" "$@"
fi

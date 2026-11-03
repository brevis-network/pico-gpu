// Copyright (c) 2025-2026 Brevis.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{env, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut base_dir = manifest_dir.join("pico-gpu");
    if !base_dir.exists() {
        base_dir = manifest_dir
            .parent()
            .expect("can't access parent of current directory")
            .into();
    }

    // Export DEP_PICO_GPU_ROOT for gpu-api build.rs to use
    println!("cargo:ROOT={}", base_dir.to_string_lossy());

    // Set up rerun conditions for directories that contain headers
    // gpu-api/build.rs scans these directories for headers, so we need to
    // trigger rebuilds when headers change.
    //
    // Only the four directories this crate still carries. A rerun-if-changed naming a
    // path that does not exist makes cargo re-run this script on every build, so the
    // list has to track what is actually here: quotient/, air/, matrix/ and fri_open/
    // held the classic STARK prover and are gone.
    println!(
        "cargo:rerun-if-changed={}",
        base_dir.join("third_party").to_string_lossy()
    );
    // src/ is watched entry by entry, all but tracegen: that directory is compiled by pico-gpu-api,
    // which fingerprints what it reads by content, and a watch on it here would rerun this script
    // (and through its `links` key every dependent script) on each edit under it.
    let mut src_entries: Vec<PathBuf> = std::fs::read_dir(base_dir.join("src"))
        .expect("can not list pico-gpu src")
        .map(|entry| entry.expect("can not read an entry of pico-gpu src").path())
        .filter(|path| path.file_name().is_some_and(|name| name != "tracegen"))
        .collect();
    src_entries.sort();
    for entry in src_entries {
        println!("cargo:rerun-if-changed={}", entry.to_string_lossy());
    }

    // This build.rs only exports the root path and sets up rerun conditions
}

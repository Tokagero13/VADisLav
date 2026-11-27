#!/bin/sh
# Builds dist/vad.bin (Linux x86-64), dist/vad-arm64.bin (Linux ARM64) and, on Windows, dist/vad.exe.
set -e
cd "$(dirname "$0")"
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
mkdir -p dist
cargo build --release --target x86_64-unknown-linux-musl
cargo build --release --target aarch64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/vad dist/vad.bin
cp target/aarch64-unknown-linux-musl/release/vad dist/vad-arm64.bin
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cargo build --release && cp target/release/vad.exe dist/ ;; esac
ls -l dist

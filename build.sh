#!/bin/sh
# Builds into dist/: vad.bin (Linux x86-64), vad-arm64.bin (Linux ARM64) and, on Windows, vad.exe.
# After `sh models/fetch.sh` it builds the noise tool the same way. The Linux builds of noise need
# clang and llvm-ar, because its model runner has parts in assembly; without them they are skipped.
set -e
cd "$(dirname "$0")"
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
mkdir -p dist

noise=""
[ -f models/ced-tiny.onnx ] && noise="--features noise"
linux_noise="$noise"
if [ -n "$noise" ] && ! { command -v clang && command -v llvm-ar; } >/dev/null; then
  echo "clang or llvm-ar not found: the Linux builds of noise are skipped"
  linux_noise=""
fi
export CC_x86_64_unknown_linux_musl=clang AR_x86_64_unknown_linux_musl=llvm-ar
export CC_aarch64_unknown_linux_musl=clang AR_aarch64_unknown_linux_musl=llvm-ar

for target in x86_64-unknown-linux-musl aarch64-unknown-linux-musl; do
  case $target in aarch64*) suffix=-arm64 ;; *) suffix= ;; esac
  cargo build --release --target $target $linux_noise
  cp target/$target/release/vad dist/vad$suffix.bin
  [ -z "$linux_noise" ] || cp target/$target/release/noise dist/noise$suffix.bin
done
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*)
  cargo build --release $noise
  cp target/release/vad.exe dist/
  [ -z "$noise" ] || cp target/release/noise.exe dist/
esac
ls -l dist

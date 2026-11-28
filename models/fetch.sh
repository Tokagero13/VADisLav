#!/bin/sh
# Downloads what the noise tool is built with: the sound-naming model CED-tiny (22 MB, Apache-2.0)
# and its list of 527 sound names, into this folder; and the model's own test recordings into
# tests/samples/ced/. Re-running skips what is already here.
set -eu
cd "$(dirname "$0")"

NAME=sherpa-onnx-ced-tiny-audio-tagging-2024-04-19
SUM="84baf315b57d61aa69480c4fee878dab54cbc7be3e877db334e65d8b087e23c3  ced.tar.bz2"

if [ ! -f ced-tiny.onnx ] || [ ! -f ced-labels.csv ] || [ ! -d ../tests/samples/ced ]; then
  curl -fL --retry 3 -o ced.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/$NAME.tar.bz2"
  echo "$SUM" | sha256sum -c 2>/dev/null || echo "$SUM" | shasum -a 256 -c # macOS has only the second
  tar xjf ced.tar.bz2
  mv "$NAME/model.onnx" ced-tiny.onnx
  mv "$NAME/class_labels_indices.csv" ced-labels.csv
  mkdir -p ../tests/samples/ced
  mv "$NAME"/test_wavs/*.wav ../tests/samples/ced/
  rm -r ced.tar.bz2 "$NAME"
fi
ls -l ced-tiny.onnx ced-labels.csv

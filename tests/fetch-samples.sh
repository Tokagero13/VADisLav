#!/bin/sh
# Downloads the recordings the tests use into tests/samples/ (about 60 MB, not kept in git) and
# checks each against its SHA-256. Then: cargo test --release -- --include-ignored
#
# Sources: OpenAI Whisper and whisper.cpp (jfk), Silero VAD (en, ru, silero), Hugging Face asr_dummy
# (mlk), Wikimedia Commons (example), W3C media server (Sintel and Big Buck Bunny trailers by the
# Blender Foundation, movie), pyannote-audio test data (ami: pieces of the AMI Meeting Corpus, CC BY 4.0).
set -e
mkdir -p "$(dirname "$0")/samples/ami"
cd "$(dirname "$0")/samples"
PYANNOTE=https://github.com/pyannote/pyannote-audio/raw/main/tests/data

get() { # <sha256> <file> <url>
    [ -f "$2" ] || curl -sSL --fail -o "$2" "$3"
    echo "$1  $2" | sha256sum -c --quiet -
}

get 63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715 jfk.flac https://github.com/openai/whisper/raw/main/tests/jfk.flac
get 20d3323a2bcce6f25498b8911a397503a0a99fa92b6ba58d62788cb42b6e5459 jfk.mp3 https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.mp3
get 89f17d9c94c4b31eb320f424628bcbc920abaddbee6e2760fd868bfb1d9a2e47 en.wav https://models.silero.ai/vad_models/en.wav
get 129086dcc52430ec46a094c7629d0be54d07eba751f4e74a64f14cea3f31521b ru.wav https://models.silero.ai/vad_models/ru.wav
get 8c009ea434e0323f453dc311f03d980929406af6c44c34897503f0b01e84265e silero.opus https://github.com/snakers4/silero-vad/raw/master/tests/data/test.opus
get 65a63b8ab116e80078cfc700ebec6d496b769c33bde72018ea41d535d4cfea0e silero.mp3 https://github.com/snakers4/silero-vad/raw/master/tests/data/test.mp3
get 88a3befa3c8ae57f515dedae5c23071acf1a29377694c310eee3205d299db51b mlk.flac https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac
get f57b56d8aae4c847cf01224fb45293610d801cfdac43d932b5eeab1cd318182a example.ogg https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg
get b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254 sintel.mp4 https://media.w3.org/2010/05/sintel/trailer.mp4
get 498d1588f72423cb0460e527a32056d0e3cfc53bd275204511c42ba25f6d46b3 sintel.webm https://media.w3.org/2010/05/sintel/trailer.webm
get b2ded9ae5a20fa36ca8cec49ef923bffb5e1a51d9e8c1f8336273d2fa9d35ff0 bunny.mp4 https://media.w3.org/2010/05/bunny/trailer.mp4
get dcbe6f2ea404a52d4b2f6fc91f274874e69be1671a1097eb3dc19f28f7c6e40f bunny.ogv https://media.w3.org/2010/05/bunny/trailer.ogv
get 80c548058688a577ce9ca501cf9807311b95cc526cc82d292ec7e138e42257de movie.mp4 https://media.w3.org/2010/05/video/movie_300.mp4
get 5f213644274ec8758dadd72466cd3925ba67aeccf00234b4e2dfd42f44d13112 movie.webm https://media.w3.org/2010/05/video/movie_300.webm
get 187386cda67317574da5b30068a8b853e52cc6df0d13abfec381912dfa9e7461 ami/dev00.wav $PYANNOTE/dev00.wav
get 13834217e1bd46a581af1b8040018bfba1e65ce1e0b9ee46cdcefa9334b9ddea ami/dev01.wav $PYANNOTE/dev01.wav
get b5e8d91105604d25cb05cc3840d015d88bd6689a2acba35506dd0d7dcb00668a ami/trn01.wav $PYANNOTE/trn01.wav
get 2e4ce968332e68cac5ed74cd00f96b63ed6c034667583be34a32a7c2aa260803 ami/trn02.wav $PYANNOTE/trn02.wav
get d12d132a6d4788f85efaa8e18563e575746e6a6fcc928e919c7c0e6a3e0238b2 ami/trn03.wav $PYANNOTE/trn03.wav
get afa4bd13ee05be34953beb594d0d755ba83284c5a742ae27e388952d41c4d0f1 ami/trn04.wav $PYANNOTE/trn04.wav
get ec694aa5c9eea7453263eaf90d38ed921c1b6f8cc4f52897d9a4457a504d668d ami/trn05.wav $PYANNOTE/trn05.wav
get e70cbed4f13c7971fdf9d424b079d87a25ed1ce3928da0bccd18f143eb4ac359 ami/trn06.wav $PYANNOTE/trn06.wav
get 77628cd91dcc27c61f9b709e670f003cb860d597fda56e7ccb0ccbbc42d5a26f ami/trn07.wav $PYANNOTE/trn07.wav
get a7d3397e82cb1025307f08b0fe8459a9d43fe61f3d9ce674f17819c7450bfac5 ami/trn08.wav $PYANNOTE/trn08.wav
get 8e18bc410e6d54a166bbba2ff0b46a222626706788fa222add4f97bca842f822 ami/trn09.wav $PYANNOTE/trn09.wav
get 0af781fb566501f7950479fd8bd92329276782c7ac930a03497d553b21c33397 ami/tst00.wav $PYANNOTE/tst00.wav
get a40925742870d05363eb781bd2cf1501fc37883cf003389175a39b24a5b2c7c9 ami/tst01.wav $PYANNOTE/tst01.wav

# where people marked speech in the ami recordings: the three lists joined into one file
[ -f ami/speech.rttm ] || for part in train development test; do curl -sSL --fail "$PYANNOTE/debug.$part.rttm"; done > ami/speech.rttm
echo "fc442f35a0570ce7faf6492616fb18b2e887783c6dad8757cfb4c71892c206b1  ami/speech.rttm" | sha256sum -c --quiet -
echo "samples ready: $(du -sh . | cut -f1)"

#!/bin/sh
# Downloads the recordings the tests use into tests/samples/ (about 90 MB, not kept in git) and
# checks each against its SHA-256. Then: cargo test --release -- --include-ignored
#
# Sources: OpenAI Whisper and whisper.cpp (jfk), Silero VAD (en, ru, silero), Hugging Face asr_dummy
# (mlk), Wikimedia Commons (example), W3C media server (Sintel and Big Buck Bunny trailers by the
# Blender Foundation, movie), pyannote-audio test data (ami: pieces of the AMI Meeting Corpus, CC BY 4.0), ESC-50 (esc: labelled everyday sounds).
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

# labelled sounds for the noise tool: 5 clips each of 12 kinds from ESC-50 (K. Piczak, CC BY-NC 3.0)
ESC=https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/audio
mkdir -p esc
get 2e28b25e14611723cdcc0becfa36468a8d0329a0c141e4f1d894bd6bb5773f34 esc/clock_alarm.1-13613-A-37.wav $ESC/1-13613-A-37.wav
get 220c3d127df2f0d8979a2edde5318aaaa3aed19d8c5d3b4a40d529f979bba453 esc/clock_alarm.1-14262-A-37.wav $ESC/1-14262-A-37.wav
get 87c0ead33b915845a8d86437b78755278408aedf8ed19e4880674a3679122cf4 esc/clock_alarm.1-34853-A-37.wav $ESC/1-34853-A-37.wav
get e00a8c065c7797e35ac9857dfdd8ae4272a22cc1a41af0ed245068774a8b9cea esc/clock_alarm.1-67033-A-37.wav $ESC/1-67033-A-37.wav
get d4bf2130de0d593abb7cd7c96867e4f26a3ae266cf07413a3a7804781d221d02 esc/clock_alarm.1-72195-A-37.wav $ESC/1-72195-A-37.wav
get ab2afe6d0ee0de5edc1f0107448077919c6ac94dcf59243497e6fcb333685438 esc/coughing.1-19111-A-24.wav $ESC/1-19111-A-24.wav
get 4fb6197f72791e50c465de8f1f715a7a3f93f7afab62dfa130f28d500303d8d9 esc/coughing.1-19118-A-24.wav $ESC/1-19118-A-24.wav
get e0374e3191067ad9a63df3ae9157e0ca44a36f4f1fa7586982609565683f6a13 esc/coughing.1-30830-A-24.wav $ESC/1-30830-A-24.wav
get dbf47513fb47c7cac8eb44110871b57ac9c51daf874d9185250fea1081e8917f esc/coughing.1-52266-A-24.wav $ESC/1-52266-A-24.wav
get 74d2b6a06d461e16b869c4ed5d3a83c5af698fe3650f1ac395976de974638d80 esc/coughing.1-52323-A-24.wav $ESC/1-52323-A-24.wav
get ddf5bf45f73bc73d1838bf0cb6af530584f324aef513112ee15c210e83ecbd6b esc/crying_baby.1-187207-A-20.wav $ESC/1-187207-A-20.wav
get 3cbcd59a0db21f0ed8c252c74ea1bc193117d2c13ed99821885c46c564010b3e esc/crying_baby.1-211527-A-20.wav $ESC/1-211527-A-20.wav
get 935be6bfdd4e24f767f5af092bb88c9a6ac000b926f1fe9075b66ebed907c8c0 esc/crying_baby.1-211527-B-20.wav $ESC/1-211527-B-20.wav
get 165fa383fbfd24ea4455a84802a78970f02c3461677c23fabecd647c3c675dbc esc/crying_baby.1-211527-C-20.wav $ESC/1-211527-C-20.wav
get e97893bf8a391f769576ee70497ba51640d3bc646cbf20e719f6d22cc033b2e3 esc/crying_baby.1-22694-A-20.wav $ESC/1-22694-A-20.wav
get f40a849a2375c8c63312a73dd2dd6c74007301fcc21b4be2ece29a642831e3d8 esc/dog.1-100032-A-0.wav $ESC/1-100032-A-0.wav
get 4a18f3f834eb184b9f02595ac74c7de37ef8048a3da92f58f27018fec580b36a esc/dog.1-110389-A-0.wav $ESC/1-110389-A-0.wav
get ca2626d2b49c7c20a3e0f9f8d7aa56f4497643c6bc264ff37764392be2b15753 esc/dog.1-30226-A-0.wav $ESC/1-30226-A-0.wav
get 52a876cff473fe9b6deb4feea9545a2b976e63e03744921afc72f9800758ce30 esc/dog.1-30344-A-0.wav $ESC/1-30344-A-0.wav
get 0e96f0bab8bbba81c98a8d7741cf258c8381c6481bac36961b8603a600a6c321 esc/dog.1-32318-A-0.wav $ESC/1-32318-A-0.wav
get e738266e395e378102f74e0920b05ff5937543788420d5b2a6b8d3df8bdb3910 esc/door_wood_knock.1-101336-A-30.wav $ESC/1-101336-A-30.wav
get 1c1d3e65b7a58658766180d2faeeb9bdd1fe7405119bf100d9cd2f671a6e2b11 esc/door_wood_knock.1-103995-A-30.wav $ESC/1-103995-A-30.wav
get f33db9ec0b036d0dd17c911d456c33a59c0fc85974245a85fed56b8d2568724e esc/door_wood_knock.1-103999-A-30.wav $ESC/1-103999-A-30.wav
get 655277d3f1c75ddc67aada3ab350bd6029fea04d288b9c747e10bc84cc802de1 esc/door_wood_knock.1-26188-A-30.wav $ESC/1-26188-A-30.wav
get 09082f42b1ee2bc467eab41e2251419eb29fbf0b8bb7ab72a4b04812641c5dc6 esc/door_wood_knock.1-52290-A-30.wav $ESC/1-52290-A-30.wav
get cb90f52c7fe7d0d994941b2e6de555cb36e898fb49b52cc31f08401ac464e7c6 esc/footsteps.1-155858-A-25.wav $ESC/1-155858-A-25.wav
get 234358f787b98c5903fcf56f7fc4c97fda2b4f0ad5311bd35f9c6c6c9a4b2277 esc/footsteps.1-155858-B-25.wav $ESC/1-155858-B-25.wav
get 47c7c305285136b8a3673f0143e42d832eec4ab1c907d2492396c4aa13943426 esc/footsteps.1-155858-C-25.wav $ESC/1-155858-C-25.wav
get 1842d2cc95f18f873b159b41ade87620b9e3232098947528d80af1e60380d25a esc/footsteps.1-155858-D-25.wav $ESC/1-155858-D-25.wav
get f20bcbe29484e399bda52e2e2dba9de478ae0147a7b588242b0d6acc1534b9d9 esc/footsteps.1-155858-E-25.wav $ESC/1-155858-E-25.wav
get d68e4716f74f073be8f78f5fe8e45daf7c09b0ec0575b21f57989d803536abb3 esc/keyboard_typing.1-137-A-32.wav $ESC/1-137-A-32.wav
get 7edf3a6b856987de4fc2da27906d70abb00c40fba3352275eec840917daf2e50 esc/keyboard_typing.1-53501-A-32.wav $ESC/1-53501-A-32.wav
get 655cc6c53397673f39fb8f125124e65fafc87480e1d18cc53f78f4404464826d esc/keyboard_typing.1-61212-A-32.wav $ESC/1-61212-A-32.wav
get a8adb317bb75afae3395204f0d0e19fb0230539d0a2cd02d6e2cc424cb3efe05 esc/keyboard_typing.1-62594-A-32.wav $ESC/1-62594-A-32.wav
get f3fd05d3bfed74942257ef036a83c7775c00a4213fd471c5ad6868e9487db525 esc/keyboard_typing.1-79711-A-32.wav $ESC/1-79711-A-32.wav
get dc6cd665f3d20eed6089d9593a2b776104d8671c57ce6c48e6be030c75a53643 esc/laughing.1-1791-A-26.wav $ESC/1-1791-A-26.wav
get dcb1b2c6686b090bf63775eb79a64d96c243987e6ae66cbb96a4a66610b37809 esc/laughing.1-30039-A-26.wav $ESC/1-30039-A-26.wav
get 1f7b0a945875fa57a2c213e8e09afb4898675f87aafd59b2bdbfcaae9f750723 esc/laughing.1-30043-A-26.wav $ESC/1-30043-A-26.wav
get 372c161a4c9720a4ff4f14499dcfe763de12b6ab428c430dbed22bb46d1e1528 esc/laughing.1-33658-A-26.wav $ESC/1-33658-A-26.wav
get 1e458a12c75bae2b27e719449f4a8036e315da58c4c1871318f39170039bc193 esc/laughing.1-36164-A-26.wav $ESC/1-36164-A-26.wav
get 4461fa26f119749f99a4762be421d0d64e774960be7eef9780812d48580e48fb esc/mouse_click.1-118206-A-31.wav $ESC/1-118206-A-31.wav
get fe84ff9cb5831e21fb8cbcaa56f6e3a477daad8ffe93999e490d442bc0150e45 esc/mouse_click.1-18655-A-31.wav $ESC/1-18655-A-31.wav
get 0f109c4e08634da4b4683c050bcfcf5ad6c2b6dc73a908be654711a8c7d6f078 esc/mouse_click.1-45645-A-31.wav $ESC/1-45645-A-31.wav
get 3e81a17a5c1b196b3ad300ad5eee300eddafa97f45edb9a38ca189767d647f04 esc/mouse_click.1-81851-A-31.wav $ESC/1-81851-A-31.wav
get 8e66ea2d3177ae19620834686b6bb1a1f9a91dae3d0a924a24e88e09284afcc2 esc/mouse_click.1-85123-A-31.wav $ESC/1-85123-A-31.wav
get e4c640a001c2abe808389c90b90ba7084fb1922d58235e3ec95c18c4afb4ef5c esc/siren.1-31482-A-42.wav $ESC/1-31482-A-42.wav
get 2446a3eb69758de0ddde8962ab8966790a19960c71ab649cff52aae3d5d93a54 esc/siren.1-31482-B-42.wav $ESC/1-31482-B-42.wav
get b0ce84a3c012d144a1f97394cab2830512078d660e809eecb8abc4fa8ad0a9e1 esc/siren.1-54084-A-42.wav $ESC/1-54084-A-42.wav
get 4b3084efd71e12121f459df54c21b24e4a4710f93dd4f1585d2c2d507b6ec365 esc/siren.1-76831-A-42.wav $ESC/1-76831-A-42.wav
get 5e91fb8adabdc48483e968357f839d801e29054c7fdb1338f8781ececbd9e76c esc/siren.1-76831-B-42.wav $ESC/1-76831-B-42.wav
get 8e32762c8c48e5414dbecc2979833748cb838ba476bac75d2a512262ef842eb0 esc/sneezing.1-26143-A-21.wav $ESC/1-26143-A-21.wav
get 4c1cdb30db5fd43eb62c17c051efaf7dde38938c0449e6bfcfb998dba92cc443 esc/sneezing.1-29680-A-21.wav $ESC/1-29680-A-21.wav
get 865652b27db628cd353379f222c01a48711f383d16b3d80bcd8493e6e8432a9e esc/sneezing.1-31748-A-21.wav $ESC/1-31748-A-21.wav
get a77b4c35c9d4a66bdbe2a677a43b8f646579ce5773c59cf721bcb9db419d1b2d esc/sneezing.1-47273-A-21.wav $ESC/1-47273-A-21.wav
get c7114b4b84b83598dfacbf7d1774f34fb61ba87af426944db4ad8ade82c7779b esc/sneezing.1-47274-A-21.wav $ESC/1-47274-A-21.wav
get f97e97b0fe0afce323d858503b6fa3ffd6c8cb1162c43cf3b24461300fb9abf6 esc/vacuum_cleaner.1-100210-A-36.wav $ESC/1-100210-A-36.wav
get acfdccbb58647655b9b63ab5574c178783b969f56e05933f115ae4dd1878a524 esc/vacuum_cleaner.1-100210-B-36.wav $ESC/1-100210-B-36.wav
get d6d58e48005b976207cd6030de5d8976e9e7f4377eab86f790c5473c95ad2dd1 esc/vacuum_cleaner.1-19840-A-36.wav $ESC/1-19840-A-36.wav
get 6f15bb8c669aaf930dfd44ef95920141a2b8d43148f7e29c0f1a7a52f91260d6 esc/vacuum_cleaner.1-19872-A-36.wav $ESC/1-19872-A-36.wav
get 610ae9ac76627cf8c1d655190fffa635979a24d25c5925cbbcf1dc659e323146 esc/vacuum_cleaner.1-19872-B-36.wav $ESC/1-19872-B-36.wav

# where people marked speech in the ami recordings: the three lists joined into one file
[ -f ami/speech.rttm ] || for part in train development test; do curl -sSL --fail "$PYANNOTE/debug.$part.rttm"; done > ami/speech.rttm
echo "fc442f35a0570ce7faf6492616fb18b2e887783c6dad8757cfb4c71892c206b1  ami/speech.rttm" | sha256sum -c --quiet -
echo "samples ready: $(du -sh . | cut -f1)"

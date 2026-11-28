# VADisLav

A tool for one task: prepare a recording for speech recognition. `vad` takes an audio or video
file, finds the speech, drops the gaps and writes the speech as one WAV file, with a report of
where every kept piece came from.

Why do this before transcribing:

- **Fewer invented words.** Whisper and similar models tend to make up text on long silence, noise
  and hold music ("Thank you for watching", the last phrase repeated). Audio that is not sent
  cannot be transcribed wrongly.
- **Less to process.** A meeting is often one third silence: the upload is smaller and the
  transcription faster and cheaper.

One file of 2–3 MB, nothing to install: no ffmpeg, no model download, no network.

## Use

```sh
vad meeting.mp4                  # writes meeting.vad.wav next to the input
vad meeting.webm -o speech.wav   # choose where to write
vad meeting.mp4 --json           # also report where every kept piece is, see below
```

```
$ vad jfk.flac
vad: kept 10.3 s of 11.0 s in 0.02 s (decode 0.02, detect 0.00, write 0.00)
jfk.vad.wav
```

The summary goes to stderr and the path of the result to stdout, so a script can do
`out=$(vad input.mp4)`.

| | |
|---|---|
| Input | mp4, m4a, mov, mkv, webm, mp3, wav, flac, ogg, opus, aac, aiff, caf. For video, the audio track is used. |
| Output | 16 kHz mono 16-bit WAV (what speech recognition expects), about 1.9 MB per minute of speech. |
| Pauses | A pause up to 0.64 s stays as it is; a longer one is cut down to 0.64 s. |
| Exit code | `0` done, `1` error, `2` no speech found (nothing is written). |

A flag such as `--mp4` before the file name is accepted and ignored: the format is detected from the
file itself.

## With Whisper

```sh
vad meeting.mp4 --json > meeting.json    # writes meeting.vad.wav, the report goes to meeting.json

whisper-cli -m ggml-large-v3-turbo.bin -f meeting.vad.wav          # whisper.cpp, or
curl "$SERVER/v1/audio/transcriptions" -H "Authorization: Bearer $KEY" \
     -F model=whisper-1 -F response_format=verbose_json -F file=@meeting.vad.wav
```

Whisper's times now refer to the trimmed file. The report says how to get back to the original:

```json
{
  "version": "0.2.0",
  "input": "mlk.flac",
  "output": "mlk.vad.wav",
  "duration": "00:00:12:992",
  "kept": "00:00:10:016",
  "segments": [
    {"from": "00:00:00:848", "to": "00:00:02:784", "at": "00:00:00:000"},
    {"from": "00:00:03:392", "to": "00:00:05:056", "at": "00:00:01:936"},
    {"from": "00:00:06:048", "to": "00:00:09:312", "at": "00:00:03:600"},
    {"from": "00:00:09:840", "to": "00:00:12:992", "at": "00:00:06:864"}
  ]
}
```

Times are `hours:minutes:seconds:milliseconds`. Each piece was taken from `from`–`to` in the input
and starts at `at` in the output. To place a time from Whisper in the original recording, take the
last piece whose `at` is not after it and add the difference to `from`: 4.0 s in the trimmed file
falls in the third piece (`at` 3.6 s), so it is 6.048 + (4.0 − 3.6) = 6.448 s in the original.

With `--json` the report replaces the path on stdout. When no speech is found, `output` is `null`,
`segments` is empty and the exit code is still `2`.

## Get the binary

Download it from the [releases page](../../releases):

| File | Runs on |
|---|---|
| `vad.bin` | Linux x86-64. Fully static: any distro, also an empty container. |
| `vad-arm64.bin` | Linux ARM64. Fully static; tested in an emulator only, not on real hardware. |
| `vad.exe` | Windows x86-64. |
| `vad-macos-arm64` | macOS on Apple Silicon. |

On Linux and macOS make it executable first: `chmod +x vad.bin`. `SHA256SUMS` in the release lists
the checksums.

Or build it with [Rust](https://rustup.rs): `sh build.sh` puts the Linux binaries (and, on Windows,
`vad.exe`) into `dist/`. The Linux binaries build from any OS, no C toolchain needed. For the
machine you are on, `cargo build --release` is enough.

A new release is made by pushing a tag: `git tag v0.2.0 && git push origin v0.2.0`
(`.github/workflows/release.yml` tests, builds and attaches the files).

## Speed and accuracy

Measured on an Intel i5-13600KF, one thread:

- One hour of video takes about 4 s (roughly 1000× realtime). Most of that is decoding the file.
- Memory: about 134 MB per hour of audio, because the whole recording is held in memory.
- On meeting recordings with human-marked speech (6.5 minutes of the AMI corpus) it keeps 95.9% of
  the speech and removes 81.4% of the rest. The same recordings made 30 dB quieter: 92.2% kept.

## Limits

- AAC with more than 2 channels (5.1 film sound) cannot be decoded: convert it to stereo first.
- Music without speech is about half kept. The tool is meant for speech recordings.
- The output is always WAV, and there are no options for thresholds.
- Whether it reduces invented words on your recordings was not measured here; it removes the
  silence they usually appear in.

## How it works

1. Decode the audio track ([symphonia](https://github.com/pdeljanov/Symphonia); Opus through
   [unsafe-libopus](https://github.com/DCNick3/unsafe-libopus)), mix it to mono, resample to 16 kHz.
2. Score every 16 ms frame with [Earshot](https://github.com/pykeio/earshot), a 40 KB neural
   network compiled into the binary. A quiet recording is turned up first, for the detector only.
3. Speech starts at a score of 0.6 and lasts until the score falls under 0.4. Runs shorter than
   0.1 s are dropped as clicks; 0.32 s is kept on each side of the rest.
4. Write the kept pieces one after another.

Everything is in `src/main.rs`.

## Tests and benchmark

```sh
cargo test --release                        # unit tests
sh tests/fetch-samples.sh                   # downloads 59 MB of public recordings, checksummed
cargo test --release -- --include-ignored   # also decodes every sample and checks the accuracy
pwsh bench.ps1 a.webm b.mp4                 # time, CPU and peak memory per file (Windows)
pwsh bench.ps1 a.webm -Hour                 # also one-hour versions in 5 formats (needs ffmpeg)
```

## Licence

[MIT](LICENSE). The binaries also contain Symphonia (MPL-2.0), libopus (BSD-3-Clause) and Earshot
(MIT or Apache-2.0).

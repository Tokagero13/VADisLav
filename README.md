# VADisLav

Two small tools that cut a recording down to the part you need:

- **`vad`** keeps the speech and drops the rest. Its task: prepare a recording for speech recognition.
- **[`noise`](#noise-find-and-name-the-other-sounds)** does the opposite: it keeps the sounds that
  are not speech (a dog, typing, a phone, music) and names them.

`vad` takes an audio or video file, finds the speech, drops the gaps and writes the speech as one
WAV file, with a report of where every kept piece came from.

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
  "version": "0.3.0",
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

## noise: find and name the other sounds

```sh
noise exam.mp4                              # writes exam.noise.wav: every sound that is not speech
noise exam.mp4 --json                       # also say what each sound is
noise exam.mp4 --only whispering,telephone  # look only for these
noise --labels                              # the 527 sounds it can name
```

A separate program of about 41 MB, because it carries a sound-naming model
([CED-tiny](https://github.com/RicherMans/CED), 527 kinds of sound). Input formats, output format,
`-o` and exit codes are the same as for `vad`. The report has the same fields plus the names heard
in each piece, best first, with a score from 0 to 1. This is a test recording made of speech, a dog
(11–16 s), typing (20–25 s), a siren (28–33 s) and more speech:

```json
{
  "version": "0.3.0",
  "input": "mixed.wav",
  "output": "mixed.noise.wav",
  "duration": "00:00:42:992",
  "kept": "00:00:19:456",
  "segments": [
    {"from": "00:00:10:240", "to": "00:00:16:384", "at": "00:00:00:000", "labels": [{"name": "Dog", "score": 0.70}, {"name": "Animal", "score": 0.70}, {"name": "Domestic animals, pets", "score": 0.62}, {"name": "Bow-wow", "score": 0.54}, {"name": "Yip", "score": 0.32}]},
    {"from": "00:00:19:456", "to": "00:00:25:600", "at": "00:00:06:144", "labels": [{"name": "Computer keyboard", "score": 0.47}, {"name": "Typing", "score": 0.44}, {"name": "Owl", "score": 0.20}]},
    {"from": "00:00:27:648", "to": "00:00:34:816", "at": "00:00:12:288", "labels": [{"name": "Owl", "score": 0.53}, {"name": "Hoot", "score": 0.42}, {"name": "Siren", "score": 0.40}]}
  ]
}
```

The example shows both what it does and where it is weak: the three sounds are found and the speech
is left out, but the siren is taken for an owl first and a siren third.

- **Without `--only`** every sound counts except plain speech and silence. With it, only the named
  sounds count, and speech can be one of them: `--only speech,whispering`. A name may be part of a
  label (`dog` also matches `Whimper (dog)`).
- **Times are exact to about 2 s.** The model has to hear 5 s at once to name a sound. Sounds with
  no more than about a second between them become one piece.
- **It names sounds, not people.** It can say that someone speaks or whispers, not who.

How well it names sounds, measured on 60 labelled clips of 12 everyday sounds (ESC-50): 86% of the
clips are found, 75% get a right name. Clean speech (four recordings) is left out completely. A
clip with one short beep per second is missed. One hour of audio takes about 13 s on all cores of
an Intel i5-13600KF and 190 MB of memory.

## Get the binaries

Download them from the [releases page](../../releases):

| File | Runs on |
|---|---|
| `vad.bin`, `noise.bin` | Linux x86-64. Fully static: any distro, also an empty container. |
| `vad-arm64.bin`, `noise-arm64.bin` | Linux ARM64. Fully static; tested in an emulator only, not on real hardware. |
| `vad.exe`, `noise.exe` | Windows x86-64. |
| `vad-macos-arm64`, `noise-macos-arm64` | macOS on Apple Silicon. |

On Linux and macOS make them executable first: `chmod +x vad.bin`. `SHA256SUMS` in the release
lists the checksums.

Or build them with [Rust](https://rustup.rs):

- `vad` alone: `cargo build --release` for the machine you are on, or `sh build.sh` for the Linux
  binaries (and, on Windows, `vad.exe`) in `dist/`. Pure Rust: the Linux binaries build from any
  OS, no C toolchain needed.
- `noise` too: run `sh models/fetch.sh` first (downloads the 22 MB model, which is built into the
  binary), then `cargo build --release --features noise` or `sh build.sh`. Its Linux builds need
  clang and llvm-ar.

A new release is made by pushing a tag: `git tag v0.3.0 && git push origin v0.3.0`
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

`noise` decodes the same way, then lets the CED-tiny model (run by
[tract](https://github.com/sonos/tract)) score 527 kinds of sound on a 5-second window starting at
every second. A second counts when it is not silent and a wanted sound scores 0.2 or more on
average over the windows that contain it.

`vad` is `src/main.rs`, `noise` is `src/bin/noise.rs`, and what they share (decoding, WAV, report)
is `src/lib.rs`.

## Tests and benchmark

```sh
cargo test --release                        # unit tests of vad
sh tests/fetch-samples.sh                   # downloads 90 MB of public recordings, checksummed
cargo test --release -- --include-ignored   # also decodes every sample and checks the accuracy

sh models/fetch.sh                          # the model and its test recordings, needed for noise
cargo test --release --features noise -- --include-ignored   # everything, noise included

pwsh bench.ps1 a.webm b.mp4                 # time, CPU and peak memory per file (Windows)
pwsh bench.ps1 a.webm -Hour                 # also one-hour versions in 5 formats (needs ffmpeg)
pwsh bench.ps1 a.webm -Vad dist/noise.exe   # the same for noise
```

## Licence

[MIT](LICENSE). The binaries also contain Symphonia (MPL-2.0), libopus (BSD-3-Clause) and Earshot
(MIT or Apache-2.0); `noise` also contains tract (MIT or Apache-2.0) and the CED-tiny model
(Apache-2.0), whose sound names come from AudioSet (CC BY 4.0).

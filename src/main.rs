//! vad: removes the parts of a recording where nobody speaks.
//! Reads any audio or video file, keeps only the speech and writes it as one 16 kHz mono WAV.
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

use earshot::Detector;
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use unsafe_libopus::{opus_decode, opus_decoder_create, opus_decoder_destroy};

const RATE: u32 = 16000; // what the speech detector needs, and what speech-to-text models take
const FRAME: usize = 256; // the detector works on 16 ms frames
// The numbers below were chosen by measuring against human-marked speech in meeting recordings
// (the meeting_accuracy test): they keep 96% of the marked speech and remove 81% of the rest.
const ON: f32 = 0.6; // the detector's score at which speech starts
const OFF: f32 = 0.4; // ...and the score below which it ends: word endings fade out, so they score lower
const PAD: usize = 20; // frames (0.32 s) kept on each side of speech: pauses up to 0.64 s stay whole
const MIN_SPEECH: usize = 6; // frames (0.1 s): shorter bursts are clicks, not words
const QUIET_DB: f32 = -35.0; // recordings quieter than this are turned up to it for the detector

const HELP: &str = "vad: cut the silence out of a recording

usage: vad [--mp4] <input> [-o <output.wav>] [--json]

  <input>   any audio or video file: mp4, m4a, mov, mkv, webm, mp3, wav, flac, ogg, opus, aac, aiff, caf
  -o        where to write the result (default: <input>.vad.wav next to the input)
  --json    print a report instead of the path: the files, the lengths and, for every kept piece,
            where it is in the input (from, to) and in the output (at), as hours:minutes:seconds:milliseconds

Writes 16 kHz mono 16-bit WAV with only the speech and prints its path.
Pauses up to 0.64 s stay as they are, longer ones are cut down to that.
Not supported: surround (more than 2 channels) AAC audio; convert it to stereo first.
Exit code: 0 done, 1 error, 2 no speech found (nothing is written).";

/// Converts a mono stream of any sample rate to 16 kHz, a block at a time.
/// A windowed-sinc low-pass filter, precomputed for every position an output sample can fall on
/// between two input samples, so high tones do not fold back into the speech band.
struct Resampler {
    up: usize, // 16000 / from, reduced: output samples per `down` input samples
    down: usize,
    taps: usize,
    bank: Vec<f32>, // `up` filters of `taps` coefficients
    buf: Vec<f32>,  // input not yet consumed
    phase: usize,   // position of the next output sample between two input samples, in 1/up
}

impl Resampler {
    fn new(from: u32) -> Self {
        let gcd = |mut a: u32, mut b: u32| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let g = gcd(from, RATE);
        let (up, down) = ((RATE / g) as usize, (from / g) as usize);
        // pass band, as a share of the input's: 6% below the lower of the two limits
        let cutoff = 0.94 * (up as f64 / down as f64).min(1.0);
        let half = ((16.0 / cutoff).ceil() as usize).next_multiple_of(4); // 16 ripples of the sinc each side
        let taps = 2 * half;
        let mut bank = Vec::with_capacity(up * taps);
        for phase in 0..up {
            let filter: Vec<f64> = (0..taps)
                .map(|k| {
                    let x = phase as f64 / up as f64 + half as f64 - k as f64; // distance to the input sample
                    if x.abs() >= half as f64 {
                        return 0.0;
                    }
                    let px = std::f64::consts::PI * x;
                    let sinc = if x == 0.0 { 1.0 } else { (cutoff * px).sin() / (cutoff * px) };
                    sinc * (0.5 + 0.5 * (px / half as f64).cos())
                })
                .collect();
            let sum: f64 = filter.iter().sum(); // each filter adds up to 1: the volume does not change
            bank.extend(filter.iter().map(|c| (c / sum) as f32));
        }
        Resampler { up, down, taps, bank, buf: vec![0.0; half], phase: 0 }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<i16>) {
        self.buf.extend_from_slice(input);
        let mut i = 0;
        while i + self.taps <= self.buf.len() {
            let filter = &self.bank[self.phase * self.taps..][..self.taps];
            // 8 separate sums so the compiler can use SIMD (taps is a multiple of 8)
            let mut acc = [0f32; 8];
            for (x, c) in self.buf[i..i + self.taps].chunks_exact(8).zip(filter.chunks_exact(8)) {
                for k in 0..8 {
                    acc[k] += x[k] * c[k];
                }
            }
            out.push(to_i16(acc.iter().sum()));
            self.phase += self.down;
            i += self.phase / self.up;
            self.phase %= self.up;
        }
        self.buf.drain(..i);
    }

    /// Call once after the last block: the filter looks half its length ahead.
    fn finish(&mut self, out: &mut Vec<i16>) {
        self.process(&vec![0.0; self.taps / 2 - 1], out);
    }
}

fn to_i16(v: f32) -> i16 {
    (v * 32767.0).round().clamp(-32768.0, 32767.0) as i16
}

/// The file's audio as 16 kHz mono.
/// shortcut: the whole result is held in memory, 115 MB per hour of input; decode, detect and
/// write in one pass if day-long files ever matter.
fn decode(path: &Path) -> Result<Vec<i16>, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, MediaSourceStream::new(Box::new(file), Default::default()), Default::default(), Default::default())
        .map_err(|e| format!("{}: not a media file this tool can read ({e})", path.display()))?;
    let track = format.default_track(TrackType::Audio).ok_or("the file has no audio track")?;
    let id = track.id;
    let params = track.codec_params.as_ref().and_then(|p| p.audio()).ok_or("the file has no audio track")?.clone();

    // a recording cut off mid-write (a crashed recorder, a closed tab) still gives what it has
    let next = |format: &mut Box<dyn FormatReader>| loop {
        match format.next_packet() {
            Ok(Some(p)) if p.track_id != id => continue,
            Ok(p) => return p,
            // how WebM written live (browser recordings) ends: its length is never filled in
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => {
                eprintln!("vad: the file ends early ({e}); using what could be read");
                return None;
            }
        }
    };

    let mut pcm = Vec::new();
    if params.codec == CODEC_ID_OPUS {
        // symphonia cannot decode Opus (WebM, Ogg, voice messages); libopus can, straight to 16 kHz mono
        if params.channels.as_ref().is_some_and(|c| c.count() > 2) {
            return Err("Opus audio with more than 2 channels is not supported".into());
        }
        let dec = unsafe { opus_decoder_create(RATE as i32, 1, &mut 0) };
        if dec.is_null() {
            return Err("cannot start the Opus decoder".into());
        }
        let mut buf = [0i16; 1920]; // 120 ms, the longest Opus packet
        while let Some(p) = next(&mut format) {
            let n = unsafe { opus_decode(dec, p.data.as_ptr(), p.data.len() as i32, buf.as_mut_ptr(), buf.len() as i32, 0) };
            if n > 0 {
                pcm.extend_from_slice(&buf[..n as usize]); // n < 0 is a damaged packet: skipped
            }
        }
        unsafe { opus_decoder_destroy(dec) };
        return Ok(pcm);
    }

    let mut dec = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &Default::default())
        .map_err(|e| match params.channels.as_ref().map_or(0, |c| c.count()) {
            // seen with 5.1 AAC in film trailers; the decoder handles AAC in mono and stereo only
            n if n > 2 => format!("cannot decode this file's {n}-channel (surround) audio ({e}); convert it to stereo first"),
            _ => format!("cannot decode this file's audio ({e})"),
        })?;
    let mut resampler = None;
    let (mut interleaved, mut mono) = (Vec::<f32>::new(), Vec::<f32>::new());
    while let Some(p) = next(&mut format) {
        let buf = match dec.decode(&p) {
            Ok(buf) => buf,
            Err(Error::DecodeError(_)) => continue, // a damaged packet
            Err(e) => {
                eprintln!("vad: decoding stopped early ({e}); using what could be read");
                break;
            }
        };
        let (rate, channels) = (buf.spec().rate(), buf.spec().channels().count());
        interleaved.resize(buf.samples_interleaved(), f32::MID);
        buf.copy_to_slice_interleaved(&mut interleaved);
        mono.clear();
        mono.extend(interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32));
        if rate == RATE {
            pcm.extend(mono.iter().map(|&v| to_i16(v)));
        } else {
            resampler.get_or_insert_with(|| Resampler::new(rate)).process(&mono, &mut pcm);
        }
    }
    if let Some(r) = &mut resampler {
        r.finish(&mut pcm);
    }
    Ok(pcm)
}

/// How loud the recording's sound is, in dB below full scale, ignoring its silent parts:
/// the average power of the frames within 25 dB of the loudest ones.
fn loudness_db(pcm: &[i16]) -> f32 {
    let mut power: Vec<f32> =
        pcm.chunks_exact(FRAME).map(|f| f.iter().map(|&s| (s as f32 / 32768.0).powi(2)).sum::<f32>() / FRAME as f32).collect();
    if power.is_empty() {
        return 0.0;
    }
    power.sort_by(|a, b| a.total_cmp(b));
    let floor = power[power.len() * 99 / 100] / 316.0; // 25 dB under the 99th percentile: one click does not count
    let loud = &power[power.partition_point(|&p| p <= floor)..];
    10.0 * (loud.iter().sum::<f32>() / loud.len().max(1) as f32 + 1e-12).log10()
}

/// How sure the detector is that each 16 ms frame is speech, 0 to 1.
fn scores(pcm: &[i16]) -> Vec<f32> {
    // The detector's scores sink with the volume: meetings recorded 30 dB quieter lost 80% of their
    // speech. So a quiet recording is turned up for the detector; the output keeps its own volume.
    let gain = 10f32.powf((QUIET_DB - loudness_db(pcm)).max(0.0) / 20.0) / 32768.0;
    let mut detector = Detector::default();
    let mut frame = [0f32; FRAME];
    pcm.chunks_exact(FRAME)
        .map(|f| {
            for (out, &s) in frame.iter_mut().zip(f) {
                *out = (s as f32 * gain).clamp(-1.0, 1.0);
            }
            detector.predict_f32(&frame)
        })
        .collect()
}

/// Which frames go into the output: every stretch of speech at least MIN_SPEECH frames long,
/// widened by PAD frames on both sides. Speech starts at a score of ON and lasts until it drops under OFF.
fn keep(scores: &[f32]) -> Vec<bool> {
    let mut keep = vec![false; scores.len()];
    let mut i = 0;
    while i < scores.len() {
        if scores[i] < ON {
            i += 1;
            continue;
        }
        let end = i + scores[i..].iter().take_while(|&&s| s >= OFF).count();
        if end - i >= MIN_SPEECH {
            keep[i.saturating_sub(PAD)..(end + PAD).min(scores.len())].fill(true);
        }
        i = end;
    }
    keep
}

fn write_wav(path: &Path, pcm: &[i16], keep: &[bool]) -> std::io::Result<()> {
    let bytes = (keep.iter().filter(|&&k| k).count() * FRAME * 2) as u32;
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + bytes).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // size of the format block
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&1u16.to_le_bytes())?; // mono
    w.write_all(&RATE.to_le_bytes())?;
    w.write_all(&(RATE * 2).to_le_bytes())?; // bytes per second
    w.write_all(&2u16.to_le_bytes())?; // bytes per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample
    w.write_all(b"data")?;
    w.write_all(&bytes.to_le_bytes())?;
    for (frame, _) in pcm.chunks_exact(FRAME).zip(keep).filter(|(_, k)| **k) {
        for s in frame {
            w.write_all(&s.to_le_bytes())?;
        }
    }
    w.flush()
}

/// The kept pieces, each as [first frame, frame after its last, its first frame in the output].
fn segments(keep: &[bool]) -> Vec<[usize; 3]> {
    let (mut out, mut pos, mut at) = (Vec::new(), 0, 0);
    for run in keep.chunk_by(|a, b| a == b) {
        if run[0] {
            out.push([pos, pos + run.len(), at]);
            at += run.len();
        }
        pos += run.len();
    }
    out
}

/// A position as hours:minutes:seconds:milliseconds. A frame is exactly 16 ms, so nothing is rounded.
fn clock(frames: usize) -> String {
    let ms = frames * FRAME * 1000 / RATE as usize;
    format!("{:02}:{:02}:{:02}:{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// A JSON string. Windows paths are full of backslashes.
fn quote(s: &str) -> String {
    let mut out = String::from('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => out.extend(['\\', c]),
            c if c < ' ' => out += &format!("\\u{:04x}", c as u32),
            c => out.push(c),
        }
    }
    out + "\""
}

/// The --json report. `output` is None when no speech was found and nothing was written.
fn report(input: &Path, output: Option<&Path>, keep: &[bool]) -> String {
    let segments = segments(keep);
    let lines: Vec<String> = segments
        .iter()
        .map(|&[from, to, at]| format!("\n    {{\"from\": \"{}\", \"to\": \"{}\", \"at\": \"{}\"}}", clock(from), clock(to), clock(at)))
        .collect();
    format!(
        "{{\n  \"version\": \"{}\",\n  \"input\": {},\n  \"output\": {},\n  \"duration\": \"{}\",\n  \"kept\": \"{}\",\n  \"segments\": [{}{}]\n}}",
        env!("CARGO_PKG_VERSION"),
        quote(&input.display().to_string()),
        output.map_or("null".into(), |o| quote(&o.display().to_string())),
        clock(keep.len()),
        clock(segments.iter().map(|s| s[1] - s[0]).sum()),
        lines.join(","),
        if lines.is_empty() { "" } else { "\n  " }
    )
}

fn run() -> Result<i32, String> {
    let (mut input, mut output, mut json) = (None::<PathBuf>, None::<PathBuf>, false);
    let mut args = env::args_os().skip(1);
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("-h" | "--help") => {
                println!("{HELP}");
                return Ok(0);
            }
            Some("-o" | "--out") => output = Some(args.next().ok_or("-o needs a file name")?.into()),
            Some("--json") => json = true,
            // `--mp4 file.mp4`, `--audio file.mp3`: the flag only says what follows, the format is detected
            Some(flag) if flag.starts_with('-') && flag.len() > 1 => {}
            _ if input.is_none() => input = Some(a.into()),
            _ => return Err(format!("unexpected argument {a:?}\n\n{HELP}")),
        }
    }
    let input = input.ok_or(HELP)?;
    let output = output.unwrap_or_else(|| input.with_extension("vad.wav"));
    // the input is read completely before the output is created, so this would silently replace it
    if output.canonicalize().is_ok_and(|o| Some(o) == input.canonicalize().ok()) {
        return Err("the output would overwrite the input: give another name with -o".into());
    }

    let start = Instant::now();
    let pcm = decode(&input)?;
    let decoded = Instant::now();
    let keep = keep(&scores(&pcm));
    let detected = Instant::now();
    let secs = |frames: usize| (frames * FRAME) as f32 / RATE as f32;
    let kept = keep.iter().filter(|&&k| k).count();
    if kept == 0 {
        eprintln!("vad: no speech found in {:.1} s of audio; nothing written", secs(keep.len()));
        if json {
            println!("{}", report(&input, None, &keep));
        }
        return Ok(2);
    }
    write_wav(&output, &pcm, &keep).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    eprintln!(
        "vad: kept {:.1} s of {:.1} s in {:.2} s (decode {:.2}, detect {:.2}, write {:.2})",
        secs(kept),
        secs(keep.len()),
        start.elapsed().as_secs_f32(),
        (decoded - start).as_secs_f32(),
        (detected - decoded).as_secs_f32(),
        detected.elapsed().as_secs_f32()
    );
    if json {
        println!("{}", report(&input, Some(&output), &keep));
    } else {
        println!("{}", output.display());
    }
    Ok(0)
}

fn main() {
    exit(run().unwrap_or_else(|e| {
        eprintln!("vad: {e}");
        1
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_speech_with_padding_and_drops_clicks() {
        let mut scores = vec![0.1; 200];
        scores[50..60].fill(0.9); // a word
        scores[60..64].fill(0.5); // ...fading out: under ON but over OFF, still part of it
        scores[120] = 0.9; // a click
        scores[140..170].fill(0.5); // never reaches ON: not speech
        scores[195..200].fill(0.9); // too short, at the very end
        let k = keep(&scores);
        let kept: Vec<usize> = (0..200).filter(|&i| k[i]).collect();
        assert_eq!(kept, (50 - PAD..64 + PAD).collect::<Vec<_>>());

        // speech at both edges and a pause shorter than 2 * PAD: one piece, nothing out of range
        let mut scores = vec![0.9; 80];
        scores[20..20 + 2 * PAD - 1].fill(0.0);
        assert!(keep(&scores).iter().all(|&k| k));
        assert!(keep(&[]).is_empty());
    }

    #[test]
    fn report_says_where_each_kept_piece_is_in_the_input_and_in_the_output() {
        let mut keep = vec![false; 4100];
        keep[0..50].fill(true);
        keep[3971..4000].fill(true);
        assert_eq!(segments(&keep), [[0, 50, 0], [3971, 4000, 50]]);
        assert_eq!(clock(3971), "00:01:03:536");
        assert_eq!(clock(225_000), "01:00:00:000");
        assert_eq!(quote("C:\\a \"b\"\n"), r#""C:\\a \"b\"\u000a""#);

        let r = report(Path::new("C:\\in.mp4"), Some(Path::new("out.wav")), &keep);
        assert!(r.contains(r#""input": "C:\\in.mp4""#) && r.contains(r#""output": "out.wav""#), "{r}");
        assert!(r.contains(r#""duration": "00:01:05:600""#) && r.contains(r#""kept": "00:00:01:264""#), "{r}");
        assert!(r.contains(r#"{"from": "00:01:03:536", "to": "00:01:04:000", "at": "00:00:00:800"}"#), "{r}");
        let none = report(Path::new("in.mp4"), None, &[false; 9]);
        assert!(none.contains(r#""output": null"#) && none.contains(r#""segments": []"#), "{none}");
    }

    #[test]
    fn loudness_is_that_of_the_sound_not_of_the_silence_around_it() {
        // 1 s of a tone at 0.1 of full scale (-23 dB) in 10 s of silence
        let mut pcm: Vec<i16> = tone(440.0, RATE).iter().map(|&v| to_i16(v / 5.0)).collect();
        pcm.resize(10 * RATE as usize, 0);
        assert!((loudness_db(&pcm) + 23.0).abs() < 1.0, "{}", loudness_db(&pcm));
        assert_eq!(loudness_db(&[]), 0.0);
        assert!(scores(&vec![0; 4 * FRAME]).iter().all(|s| s.is_finite())); // pure silence: no division by zero
    }

    fn tone(hz: f32, rate: u32) -> Vec<f32> {
        (0..rate).map(|n| 0.5 * (2.0 * std::f32::consts::PI * hz * n as f32 / rate as f32).sin()).collect()
    }

    fn resampled_rms(hz: f32, rate: u32) -> f32 {
        let mut r = Resampler::new(rate);
        let mut out = Vec::new();
        for block in tone(hz, rate).chunks(1000) {
            r.process(block, &mut out);
        }
        r.finish(&mut out);
        assert_eq!(out.len(), RATE as usize, "1 s in at {rate}");
        let mid = &out[2000..14000]; // away from the edges, where the filter sees silence
        (mid.iter().map(|&s| (s as f32 / 32767.0).powi(2)).sum::<f32>() / mid.len() as f32).sqrt()
    }

    #[test]
    fn resampler_keeps_speech_tones_and_removes_what_16khz_cannot_hold() {
        let full = 0.5 / 2f32.sqrt(); // loudness of the test tone
        for rate in [8000, 11025, 22050, 44100, 48000, 96000] {
            assert!((resampled_rms(1000.0, rate) - full).abs() < 0.01 * full, "1 kHz at {rate}");
        }
        for rate in [44100, 48000] {
            assert!((resampled_rms(6000.0, rate) - full).abs() < 0.02 * full, "6 kHz at {rate}");
            assert!(resampled_rms(10000.0, rate) < 0.002 * full, "10 kHz at {rate} must be gone");
        }
    }

    #[test]
    fn resampler_keeps_the_pitch() {
        // 1 kHz at 16 kHz crosses zero upwards once every 16 samples
        let mut r = Resampler::new(44100);
        let mut out = Vec::new();
        r.process(&tone(1000.0, 44100), &mut out);
        let ups = out[2004..10004].windows(2).filter(|w| w[0] < 0 && w[1] >= 0).count();
        assert_eq!(ups, 500);
    }

    // ---- tests on real recordings: `sh tests/fetch-samples.sh`, then `cargo test --release -- --include-ignored`

    fn sample(name: &str) -> Result<Vec<i16>, String> {
        decode(&Path::new("tests/samples").join(name))
    }

    fn kept_secs(pcm: &[i16]) -> f32 {
        keep(&scores(pcm)).iter().filter(|&&k| k).count() as f32 * FRAME as f32 / RATE as f32
    }

    #[test]
    #[ignore = "needs the recordings: sh tests/fetch-samples.sh"]
    fn real_files_decode_fully_and_keep_their_speech() {
        // file, what it is, seconds of audio as ffmpeg decodes it, least and most seconds to keep
        for (name, what, secs, min, max) in [
            ("en.wav", "English speech with pauses, 16 kHz WAV", 60.0, 50.0, 58.0),
            ("ru.wav", "Russian speech with pauses and background sounds", 60.0, 42.0, 52.0),
            ("jfk.flac", "one speaker, 44.1 kHz stereo FLAC", 11.0, 9.0, 11.0),
            ("jfk.mp3", "the same as 16 kHz MP3", 11.0, 9.0, 11.0),
            ("mlk.flac", "a speech before a crowd, 22 kHz FLAC", 13.0, 8.0, 12.0),
            ("example.ogg", "one sentence, Ogg Vorbis", 6.1, 5.5, 6.2),
            ("silero.mp3", "a short phrase, MP3 with a cover picture inside", 2.53, 2.0, 2.6),
            ("silero.opus", "quiet speech, Ogg Opus", 8.06, 2.0, 8.1),
            ("ami/dev00.wav", "a meeting, 32-bit float WAV", 30.0, 20.0, 28.0),
            ("sintel.mp4", "film trailer, voices over music, H.264 + AAC", 51.94, 15.0, 48.0),
            ("sintel.webm", "the same as VP8 + Vorbis", 52.68, 15.0, 48.0),
            // known weakness: music with no speech at all should give 0, about half is kept
            ("bunny.ogv", "cartoon trailer, music only, Theora + Vorbis", 32.99, 0.0, 20.0),
            ("movie.mp4", "a beep every second, H.264 + AAC", 300.14, 0.0, 0.0),
            ("movie.webm", "the same as VP8 + Vorbis", 300.02, 0.0, 0.0),
        ] {
            let pcm = sample(name).unwrap();
            let (len, kept) = (pcm.len() as f32 / RATE as f32, kept_secs(&pcm));
            println!("{name:<14} {len:>6.1} s, kept {kept:>5.1} s  ({what})");
            assert!((len - secs).abs() < 0.1, "{name}: decoded {len} s, the file holds {secs} s");
            assert!((min..=max).contains(&kept), "{name}: kept {kept} s, expected {min} to {max}");
        }
        let surround = sample("bunny.mp4").unwrap_err(); // 5.1 AAC: must be refused with a reason, not garbled
        assert!(surround.contains("6-channel"), "{surround}");
    }

    /// Thirteen 30 s pieces of real meetings (AMI corpus) in which people marked where someone speaks.
    #[test]
    #[ignore = "needs the recordings: sh tests/fetch-samples.sh"]
    fn meeting_accuracy() {
        let marks = std::fs::read_to_string("tests/samples/ami/speech.rttm").unwrap();
        // % of the marked speech kept and % of everything else removed, with the volume multiplied by `gain`
        let accuracy = |gain: f32| {
            let (mut speech, mut speech_kept, mut other, mut other_removed) = (0, 0, 0, 0);
            for name in ["dev00", "dev01", "trn01", "trn02", "trn03", "trn04", "trn05", "trn06", "trn07", "trn08", "trn09", "tst00", "tst01"] {
                let pcm: Vec<i16> = sample(&format!("ami/{name}.wav")).unwrap().iter().map(|&s| (s as f32 * gain).round() as i16).collect();
                let keep = keep(&scores(&pcm));
                let mut marked = vec![false; keep.len()];
                // a line: SPEAKER <file> 1 <start s> <length s> ...
                for f in marks.lines().map(|l| l.split(' ').collect::<Vec<_>>()).filter(|f| f[1] == name) {
                    let (start, len): (f32, f32) = (f[3].parse().unwrap(), f[4].parse().unwrap());
                    let frames = keep.len();
                    let frame = |s: f32| ((s * RATE as f32 / FRAME as f32) as usize).min(frames);
                    marked[frame(start)..frame(start + len)].fill(true);
                }
                for (&m, &k) in marked.iter().zip(&keep) {
                    if m {
                        speech += 1;
                        speech_kept += k as usize;
                    } else {
                        other += 1;
                        other_removed += !k as usize;
                    }
                }
            }
            (100.0 * speech_kept as f32 / speech as f32, 100.0 * other_removed as f32 / other as f32)
        };
        let (kept, removed) = accuracy(1.0);
        let (quiet_kept, quiet_removed) = accuracy(0.03);
        println!("meetings as recorded: {kept:.1}% of speech kept, {removed:.1}% of the rest removed");
        println!("30 dB quieter:        {quiet_kept:.1}% of speech kept, {quiet_removed:.1}% of the rest removed");
        assert!(kept > 95.0 && removed > 79.0, "{kept} {removed}");
        assert!(quiet_kept > 90.0 && quiet_removed > 79.0, "{quiet_kept} {quiet_removed}");
    }
}

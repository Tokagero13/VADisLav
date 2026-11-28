//! What the vad and noise tools share: reading any audio or video file as 16 kHz mono, writing the
//! kept pieces as WAV, and the --json report.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use unsafe_libopus::{opus_decode, opus_decoder_create, opus_decoder_destroy};

pub const RATE: u32 = 16000; // what the speech detector needs, and what speech-to-text models take
pub const FRAME: usize = 256; // the detector works on 16 ms frames

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
pub fn decode(path: &Path) -> Result<Vec<i16>, String> {
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
                eprintln!("warning:the file ends early ({e}); using what could be read");
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
            // shortcut: a damaged packet is dropped, so everything after it is reported about 20 ms early
            // per packet; put silence of the packet's length in its place if damaged files matter
            Err(Error::DecodeError(_)) => continue,
            Err(e) => {
                eprintln!("warning:decoding stopped early ({e}); using what could be read");
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
pub fn loudness_db(pcm: &[i16]) -> f32 {
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

pub fn write_wav(path: &Path, pcm: &[i16], keep: &[bool]) -> std::io::Result<()> {
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
pub fn segments(keep: &[bool]) -> Vec<[usize; 3]> {
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
pub fn clock(frames: usize) -> String {
    let ms = frames * (FRAME * 1000 / RATE as usize); // in this order it also fits 32-bit systems
    format!("{:02}:{:02}:{:02}:{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// A JSON string. Windows paths are full of backslashes.
pub fn quote(s: &str) -> String {
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
pub fn report(input: &Path, output: Option<&Path>, keep: &[bool]) -> String {
    report_with(input, output, keep, |_| String::new())
}

/// The same with more fields per piece: `extra(i)` gives them for piece `i`, as `, "name": value`.
pub fn report_with(input: &Path, output: Option<&Path>, keep: &[bool], extra: impl Fn(usize) -> String) -> String {
    let segments = segments(keep);
    let lines: Vec<String> = segments
        .iter()
        .enumerate()
        .map(|(i, &[from, to, at])| format!("\n    {{\"from\": \"{}\", \"to\": \"{}\", \"at\": \"{}\"{}}}", clock(from), clock(to), clock(at), extra(i)))
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

#[cfg(test)]
mod tests {
    use super::*;

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

}

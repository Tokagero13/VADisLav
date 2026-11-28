//! vad: removes the parts of a recording where nobody speaks.
//! Reads any audio or video file, keeps only the speech and writes it as one 16 kHz mono WAV.
use std::env;
use std::path::PathBuf;
use std::process::exit;
use std::time::Instant;

use earshot::Detector;
use vad::{FRAME, RATE, decode, loudness_db, report, write_wav};

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

fn run() -> Result<i32, String> {
    let (mut input, mut output, mut json) = (None::<PathBuf>, None::<PathBuf>, false);
    let mut args = env::args_os().skip(1);
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("-h" | "--help") => {
                println!("{HELP}");
                return Ok(0);
            }
            // `-o --json` would otherwise write to a file called --json
            Some("-o" | "--out") => output = Some(args.next().filter(|v| !v.to_string_lossy().starts_with('-')).ok_or("-o needs a file name")?.into()),
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
    use std::path::Path;

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
    fn silence_gets_finite_scores() {
        assert!(scores(&vec![0; 4 * FRAME]).iter().all(|s| s.is_finite())); // pure silence: no division by zero
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

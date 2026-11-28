//! noise: finds the sounds of a recording that are not speech and names them.
//! The counterpart of vad: reads any audio or video file, keeps only the stretches with a sound in them
//! and writes those as one 16 kHz mono WAV; the --json report says what each sound is.
use std::env;
use std::path::PathBuf;
use std::process::exit;
use std::sync::Arc;
use std::time::Instant;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use tract_onnx::prelude::*;
use vad::{FRAME, RATE, decode, quote, report_with, segments, write_wav};

/// CED-tiny (Apache-2.0): 527 scores from 0 to 1, one per kind of sound, for a stretch of audio.
static MODEL: &[u8] = include_bytes!("../../models/ced-tiny.onnx");
/// The names of the 527 sounds (AudioSet), as lines of `index,id,"name"`.
static NAMES: &str = include_str!("../../models/ced-labels.csv");

const STEP: usize = 64; // frames (1.024 s): how finely a sound is placed in time
// The model was trained on 10 s clips. On 60 labelled clips its best name was right for 65% of whole
// 5 s clips, 54% of 2 s windows and 34% of 1 s windows, so it always hears about 5 s at once.
const WINDOW: usize = 5; // steps (5.12 s)
// A sound counts when its score reaches this. Measured on 60 labelled clips and four recordings of clean
// speech (the finds_and_names_labelled_sounds test): at 0.2, 86% of the clips are found, 75% get a right
// name and none of the speech is kept; at 0.3 it is 78%, 68% and none; at 0.1, 100%, 81% and 5% of the speech.
const THRESHOLD: f32 = 0.2;
const GAP: usize = 1; // steps: sounds with no more than this between them are one piece (vad joins speech across 0.64 s)
const QUIET: f32 = 1e-6; // a step with less power than this (-60 dB) is silence, whatever the model says
const MOST: usize = 5; // names listed per piece

// What the model's input must look like (as sherpa-onnx and the model's authors compute it):
const FFT: usize = 512; // 32 ms window
const HOP: usize = 160; // 10 ms step
const MELS: usize = 64; // frequency bands, 0 to 8000 Hz
const SAMPLES: usize = WINDOW * STEP * FRAME;
const COLUMNS: usize = (SAMPLES + HOP / 2) / HOP;

/// Not reported unless asked for with --only: speech is vad's job, and the rest describes the room, not a sound.
const SKIPPED: [&str; 12] = [
    "Speech",
    "Speech synthesizer", // what the model calls old or processed recordings of speech
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Silence",
    "Inside, small room",
    "Inside, large room or hall",
    "Inside, public space",
    "Outside, urban or manmade",
];

const HELP: &str = "noise: find the sounds in a recording that are not speech, and name them

usage: noise <input> [-o <output.wav>] [--json] [--only <name,name,...>]
       noise --labels

  <input>   any audio or video file: mp4, m4a, mov, mkv, webm, mp3, wav, flac, ogg, opus, aac, aiff, caf
  -o        where to write the result (default: <input>.noise.wav next to the input)
  --json    print a report instead of the path: the files, the lengths and, for every kept piece,
            where it is in the input (from, to) and in the output (at), as hours:minutes:seconds:milliseconds,
            and the names of the sounds heard in it with their scores (0 to 1)
  --only    look only for these sounds, for example: --only whispering,speech,telephone
            A name may be part of a label; upper and lower case are the same.
  --labels  print the 527 sounds that can be named

Writes 16 kHz mono 16-bit WAV with only the stretches that hold a sound, and prints its path.
Without --only every sound counts except plain speech and silence.
Times are exact to about 2 s: the model has to hear 5 s at once to name a sound.
Exit code: 0 done, 1 error, 2 no sound found (nothing is written).";

fn names() -> Vec<&'static str> {
    NAMES.lines().skip(1).filter_map(|l| l.splitn(3, ',').nth(2)).map(|n| n.trim().trim_matches('"')).collect()
}

/// Which of the sounds to look for: those whose name contains one of `only`, or all but SKIPPED.
fn wanted(names: &[&str], only: Option<&str>) -> Result<Vec<bool>, String> {
    let Some(only) = only else {
        return Ok(names.iter().map(|n| !SKIPPED.contains(n)).collect());
    };
    let mut wanted = vec![false; names.len()];
    for part in only.split(',').map(|p| p.trim().to_lowercase()) {
        let mut found = false;
        for (w, name) in wanted.iter_mut().zip(names) {
            if !part.is_empty() && name.to_lowercase().contains(&part) {
                (*w, found) = (true, true);
            }
        }
        if !found {
            return Err(format!("--only: no sound is called {part:?}; `noise --labels` lists the names"));
        }
    }
    Ok(wanted)
}

/// Turns a window of sound into the picture the model reads: how much energy each of 64 frequency
/// bands holds every 10 ms, in dB. Must match the model's training exactly, see `matches_the_reference`.
struct Mel {
    window: Vec<f32>,
    bands: Vec<Vec<(usize, f32)>>, // per band: the FFT bins it sums and their weights
    fft: Arc<dyn Fft<f32>>,
}

impl Mel {
    fn new() -> Self {
        let window = (0..FFT).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos()) as f32).collect();
        // triangles of equal width on the mel scale, which is finer for low tones, like hearing
        let mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
        let width = mel(RATE as f32 / 2.0) / (MELS + 1) as f32;
        let bands = (0..MELS)
            .map(|band| {
                let (left, center, right) = (band as f32 * width, (band + 1) as f32 * width, (band + 2) as f32 * width);
                (0..FFT / 2)
                    .filter_map(|bin| {
                        let m = mel(RATE as f32 / FFT as f32 * bin as f32);
                        let weight = if m <= center { (m - left) / (center - left) } else { (right - m) / (right - center) };
                        (m > left && m < right).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        Mel { window, bands, fft: FftPlanner::new().plan_fft_forward(FFT) }
    }

    /// `wave` holds SAMPLES values from -1 to 1; the result is MELS rows of COLUMNS values.
    fn picture(&self, wave: &[f32]) -> Vec<f32> {
        let n = wave.len() as isize;
        let mut out = vec![0f32; MELS * COLUMNS];
        let mut buf = vec![Complex32::default(); FFT];
        for column in 0..COLUMNS {
            let start = (column * HOP + HOP / 2) as isize - (FFT / 2) as isize;
            for (k, b) in buf.iter_mut().enumerate() {
                let mut i = start + k as isize;
                if i < 0 || i >= n {
                    i = if i < 0 { -i - 1 } else { 2 * n - 1 - i }; // past an edge the sound is mirrored
                }
                *b = Complex32::new(wave[i as usize] * self.window[k], 0.0);
            }
            self.fft.process(&mut buf);
            for (band, bins) in self.bands.iter().enumerate() {
                out[band * COLUMNS + column] = bins.iter().map(|&(bin, weight)| weight * buf[bin].norm_sqr()).sum();
            }
        }
        let mut loudest = f32::MIN;
        for v in &mut out {
            *v = 10.0 * v.max(1e-10).log10();
            loudest = loudest.max(*v);
        }
        out.iter_mut().for_each(|v| *v = v.max(loudest - 120.0));
        out
    }
}

type Model = Arc<TypedRunnableModel>;

fn model() -> TractResult<Model> {
    tract_onnx::onnx()
        .model_for_read(&mut &MODEL[..])?
        .with_input_fact(0, f32::fact([1, MELS, COLUMNS]).into())?
        .into_optimized()?
        .into_runnable()
}

/// The model's 527 scores for one window of sound.
fn hear(model: &Model, mel: &Mel, wave: &[f32]) -> TractResult<Vec<f32>> {
    let picture = Tensor::from_shape(&[1, MELS, COLUMNS], &mel.picture(wave))?;
    let out = model.run(tvec!(picture.into()))?;
    Ok(out[0].to_plain_array_view::<f32>()?.iter().copied().collect())
}

fn power(pcm: &[i16]) -> f32 {
    pcm.iter().map(|&s| (s as f32 / 32768.0).powi(2)).sum::<f32>() / pcm.len().max(1) as f32
}

/// For every step of the recording, the score of each of the 527 sounds: the average over the
/// windows that contain the step (up to WINDOW of them, one starting at every step).
fn scores(pcm: &[i16], model: &Model) -> TractResult<Vec<Vec<f32>>> {
    let steps = (pcm.len() / FRAME).div_ceil(STEP);
    let starts: Vec<usize> = (0..=steps.saturating_sub(WINDOW)).collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(starts.len().max(1));
    // each window is independent of the others, so every processor core takes its share of them
    let heard: Vec<TractResult<Vec<(usize, Vec<f32>)>>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                let starts = &starts;
                scope.spawn(move || {
                    let mel = Mel::new();
                    let mut wave = vec![0f32; SAMPLES];
                    let mut out = Vec::new();
                    for &start in starts.iter().skip(t).step_by(threads) {
                        let from = (start * STEP * FRAME).min(pcm.len());
                        let window = &pcm[from..(from + SAMPLES).min(pcm.len())];
                        if power(window) < QUIET {
                            continue; // silence: every score stays 0
                        }
                        wave.fill(0.0); // a recording shorter than the window is followed by silence
                        for (w, &s) in wave.iter_mut().zip(window) {
                            *w = s as f32 / 32768.0;
                        }
                        out.push((start, hear(model, &mel, &wave)?));
                    }
                    Ok(out)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("a worker thread failed")).collect()
    });
    let mut sums = vec![vec![0f32; 527]; steps];
    for part in heard {
        for (start, scores) in part? {
            for sum in &mut sums[start..(start + WINDOW).min(steps)] {
                sum.iter_mut().zip(&scores).for_each(|(a, s)| *a += s);
            }
        }
    }
    for (step, sum) in sums.iter_mut().enumerate() {
        // windows start at 0..=steps-WINDOW, so a step near either end lies in fewer of them
        let windows = (step + 1).min(WINDOW).min(steps - step).min(starts.len());
        sum.iter_mut().for_each(|s| *s /= windows as f32);
    }
    Ok(sums)
}

/// Which frames go into the output: every step that is not silent and has a wanted sound scoring
/// THRESHOLD or more, and the gaps of up to GAP steps between such steps.
fn keep(scores: &[Vec<f32>], wanted: &[bool], pcm: &[i16]) -> Vec<bool> {
    let mut steps: Vec<bool> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let from = (i * STEP * FRAME).min(pcm.len());
            power(&pcm[from..(from + STEP * FRAME).min(pcm.len())]) >= QUIET && s.iter().zip(wanted).any(|(&score, &w)| w && score >= THRESHOLD)
        })
        .collect();
    let mut last = None;
    for i in 0..steps.len() {
        if steps[i] {
            if let Some(l) = last.filter(|&l| i - l - 1 <= GAP) {
                steps[l..i].fill(true);
            }
            last = Some(i);
        }
    }
    (0..pcm.len() / FRAME).map(|frame| steps[frame / STEP]).collect()
}

/// The report's extra field for a piece from frame `from` to `to`: the wanted sounds heard in it,
/// each with the highest score it reaches there, best first.
fn heard_in(from: usize, to: usize, scores: &[Vec<f32>], wanted: &[bool], names: &[&str]) -> String {
    let steps = &scores[from / STEP..to.div_ceil(STEP).min(scores.len())];
    let mut heard: Vec<(f32, &str)> = (0..names.len())
        .filter(|&i| wanted[i])
        .map(|i| (steps.iter().map(|s| s[i]).fold(0.0, f32::max), names[i]))
        .filter(|&(score, _)| score >= THRESHOLD)
        .collect();
    heard.sort_by(|a, b| b.0.total_cmp(&a.0));
    let list: Vec<String> = heard.iter().take(MOST).map(|(score, name)| format!("{{\"name\": {}, \"score\": {score:.2}}}", quote(name))).collect();
    format!(", \"labels\": [{}]", list.join(", "))
}

fn run() -> Result<i32, String> {
    let (mut input, mut output, mut json, mut only) = (None::<PathBuf>, None::<PathBuf>, false, None::<String>);
    let mut args = env::args_os().skip(1);
    // a value that starts with a dash is the next flag: `-o --json` must not write to a file called --json
    let value = |args: &mut dyn Iterator<Item = std::ffi::OsString>, flag: &str, what: &str| {
        args.next().filter(|v| !v.to_string_lossy().starts_with('-')).ok_or(format!("{flag} needs {what}"))
    };
    while let Some(a) = args.next() {
        match a.to_str() {
            Some("-h" | "--help") => {
                println!("{HELP}");
                return Ok(0);
            }
            Some("--labels") => {
                names().iter().for_each(|n| println!("{n}"));
                return Ok(0);
            }
            Some("-o" | "--out") => output = Some(value(&mut args, "-o", "a file name")?.into()),
            Some("--only") => only = Some(value(&mut args, "--only", "a list of sounds")?.to_string_lossy().into_owned()),
            Some("--json") => json = true,
            Some(flag) if flag.starts_with('-') && flag.len() > 1 => return Err(format!("unknown option {flag}\n\n{HELP}")),
            _ if input.is_none() => input = Some(a.into()),
            _ => return Err(format!("unexpected argument {a:?}\n\n{HELP}")),
        }
    }
    let input = input.ok_or(HELP)?;
    let output = output.unwrap_or_else(|| input.with_extension("noise.wav"));
    // the input is read completely before the output is created, so this would silently replace it
    if output.canonicalize().is_ok_and(|o| Some(o) == input.canonicalize().ok()) {
        return Err("the output would overwrite the input: give another name with -o".into());
    }
    let names = names();
    let wanted = wanted(&names, only.as_deref())?;

    let start = Instant::now();
    let pcm = decode(&input)?;
    let decoded = Instant::now();
    let scores = model().and_then(|model| scores(&pcm, &model)).map_err(|e| format!("the sound model failed: {e}"))?;
    let keep = keep(&scores, &wanted, &pcm);
    let detected = Instant::now();
    let secs = |frames: usize| (frames * FRAME) as f32 / RATE as f32;
    let kept = keep.iter().filter(|&&k| k).count();
    let pieces = segments(&keep);
    let report = |output| report_with(&input, output, &keep, |i| heard_in(pieces[i][0], pieces[i][1], &scores, &wanted, &names));
    if kept == 0 {
        eprintln!("noise: no sound found in {:.1} s of audio; nothing written", secs(keep.len()));
        if json {
            println!("{}", report(None));
        }
        return Ok(2);
    }
    write_wav(&output, &pcm, &keep).map_err(|e| format!("cannot write {}: {e}", output.display()))?;
    eprintln!(
        "noise: kept {:.1} s of {:.1} s in {:.2} s (decode {:.2}, detect {:.2}, write {:.2})",
        secs(kept),
        secs(keep.len()),
        start.elapsed().as_secs_f32(),
        (decoded - start).as_secs_f32(),
        (detected - decoded).as_secs_f32(),
        detected.elapsed().as_secs_f32()
    );
    if json {
        println!("{}", report(Some(&output)));
    } else {
        println!("{}", output.display());
    }
    Ok(0)
}

fn main() {
    exit(run().unwrap_or_else(|e| {
        eprintln!("noise: {e}");
        1
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn only_picks_sounds_by_part_of_their_name() {
        let names = names();
        assert_eq!(names.len(), 527);
        assert!(names.contains(&"Male speech, man speaking") && names.contains(&"Whispering"));
        assert!(SKIPPED.iter().all(|s| names.contains(s)), "a skipped name is not in the model's list");

        let picked = |only| -> Vec<&str> { wanted(&names, only).unwrap().iter().zip(&names).filter(|(w, _)| **w).map(|(_, n)| *n).collect() };
        assert_eq!(picked(None).len(), 527 - SKIPPED.len());
        assert_eq!(picked(Some("WHISPER, typing")), ["Whispering", "Typing"]);
        assert!(picked(Some("speech")).contains(&"Child speech, kid speaking"));
        assert!(wanted(&names, Some("dog,nosuchsound")).unwrap_err().contains("nosuchsound"));
    }

    #[test]
    fn keeps_steps_with_a_wanted_sound_and_joins_near_ones() {
        let wanted = [false, true];
        let mut scores = vec![vec![0.9, 0.0]; 12]; // only the sound that is not wanted
        let (first, near, far) = (1, 2 + GAP, 4 + 2 * GAP); // GAP steps between the first two, one more before the third
        for step in [first, near, far] {
            scores[step][1] = THRESHOLD;
        }
        scores[11][1] = 0.9; // a silent step: the model's opinion does not count
        let mut pcm = vec![1000i16; 12 * STEP * FRAME + 5];
        pcm[11 * STEP * FRAME..].fill(0);
        let keep = keep(&scores, &wanted, &pcm);
        assert_eq!(keep.len(), 12 * STEP);
        let steps: Vec<usize> = (0..12).filter(|&s| keep[s * STEP]).collect();
        let joined: Vec<usize> = (first..=near).chain([far]).collect();
        assert_eq!(steps, joined); // the gap between the first two is filled, the third stands alone

        let names = ["a", "b"];
        assert_eq!(heard_in(STEP, (near + 1) * STEP, &scores, &wanted, &names), format!(", \"labels\": [{{\"name\": \"b\", \"score\": {THRESHOLD:.2}}}]"));
    }

    // ---- tests with the model: `sh models/fetch.sh` (also needed to build), `sh tests/fetch-samples.sh`

    fn top(file: &str, model: &Model) -> (String, f32) {
        let mut wave: Vec<f32> = decode(Path::new(file)).unwrap().iter().map(|&s| s as f32 / 32768.0).collect();
        wave.resize(SAMPLES, 0.0);
        let scores = hear(model, &Mel::new(), &wave).unwrap();
        let best = (0..scores.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap();
        (names()[best].to_string(), scores[best])
    }

    /// The model's own test recordings, cut to the first 5.12 s. The expected names and scores are
    /// what the reference implementation (sherpa-onnx 2.0.3) gives for the same cut.
    #[test]
    #[ignore = "needs the recordings: sh models/fetch.sh"]
    fn matches_the_reference() {
        let model = model().unwrap();
        for (file, name, score) in REFERENCE {
            let (got, got_score) = top(&format!("tests/samples/ced/{file}"), &model);
            assert_eq!(got, name, "{file}");
            assert!((got_score - score).abs() < 0.01, "{file}: {name} scored {got_score}, the reference gives {score}");
        }
    }
    /// 60 five-second clips of 12 kinds of everyday sound (ESC-50), each with the names that are right for it.
    const KINDS: [(&str, &[&str]); 12] = [
        ("clock_alarm", &["Alarm", "Beep", "Buzzer", "Telephone bell ringing", "Ringtone"]),
        ("coughing", &["Cough"]),
        ("crying_baby", &["Baby cry", "Crying"]),
        ("dog", &["Dog", "Bark", "Bow-wow", "Yip", "Howl", "Growling"]),
        ("door_wood_knock", &["Knock"]),
        ("footsteps", &["Walk, footsteps", "Run", "Shuffle"]),
        ("keyboard_typing", &["Typing", "Computer keyboard", "Typewriter"]),
        ("laughing", &["Laughter", "Giggle", "Chuckle", "Snicker", "laugh"]),
        ("mouse_click", &["Click", "Computer keyboard", "Typing", "Tick"]),
        ("siren", &["Siren", "siren"]),
        ("sneezing", &["Sneeze"]),
        ("vacuum_cleaner", &["Vacuum cleaner"]),
    ];

    /// How the choice of THRESHOLD was made, and the guard on it: the labelled clips must be found and
    /// named, and recordings of nothing but clean speech must stay nearly empty.
    #[test]
    #[ignore = "needs the recordings: sh models/fetch.sh && sh tests/fetch-samples.sh"]
    fn finds_and_names_labelled_sounds() {
        let model = model().unwrap();
        let names = names();
        let wanted = wanted(&names, None).unwrap();
        let mut clips: Vec<(Vec<bool>, Vec<Vec<f32>>, Vec<i16>)> = Vec::new(); // which names are right, scores, sound
        for entry in std::fs::read_dir("tests/samples/esc").unwrap() {
            let path = entry.unwrap().path();
            let kind = path.file_name().unwrap().to_str().unwrap().split('.').next().unwrap().to_string();
            let right = KINDS.iter().find(|k| k.0 == kind).unwrap().1;
            let pcm = decode(&path).unwrap();
            clips.push((names.iter().map(|n| right.iter().any(|r| n.contains(r))).collect(), scores(&pcm, &model).unwrap(), pcm));
        }
        assert_eq!(clips.len(), 60);
        let speech: Vec<(Vec<Vec<f32>>, Vec<i16>)> = ["en.wav", "jfk.flac", "example.ogg", "silero.mp3"]
            .iter()
            .map(|f| decode(&Path::new("tests/samples").join(f)).unwrap())
            .map(|pcm| (scores(&pcm, &model).unwrap(), pcm))
            .collect();

        // for a threshold: % of clips with a sound found, % with a right name reaching it, % of clean speech kept
        let measure = |threshold: f32| {
            let reaches = |scores: &[Vec<f32>], which: &[bool]| scores.iter().any(|s| s.iter().zip(which).any(|(&v, &w)| w && v >= threshold));
            let found = clips.iter().filter(|c| reaches(&c.1, &wanted)).count();
            let named = clips.iter().filter(|c| reaches(&c.1, &c.0)).count();
            let (flagged, steps) = speech.iter().fold((0, 0), |(f, n), (scores, _)| {
                (f + scores.iter().filter(|s| s.iter().zip(&wanted).any(|(&v, &w)| w && v >= threshold)).count(), n + scores.len())
            });
            (100 * found / 60, 100 * named / 60, 100.0 * flagged as f32 / steps as f32)
        };
        for threshold in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5] {
            let (found, named, flagged) = measure(threshold);
            println!("threshold {threshold:.2}: found {found:3}% of the clips, right name for {named:3}%, {flagged:4.1}% of clean speech kept");
        }
        let (found, named, flagged) = measure(THRESHOLD);
        assert!(found >= 85 && named >= 70 && flagged <= 5.0, "found {found}%, named {named}%, speech kept {flagged}%");

        // and through the real path: a dog clip gives one piece that says so
        let dog = clips.iter().find(|c| reaches_any(&c.1, &c.0, 0.6) && c.0[names.iter().position(|n| *n == "Dog").unwrap()]).unwrap();
        let keep = keep(&dog.1, &wanted, &dog.2);
        let pieces = segments(&keep);
        assert_eq!(pieces.len(), 1);
        assert!(heard_in(pieces[0][0], pieces[0][1], &dog.1, &wanted, &names).contains("\"Dog\""));
    }

    #[test]
    #[ignore = "needs the recordings: sh models/fetch.sh && sh tests/fetch-samples.sh"]
    fn music_is_music_and_speech_only_counts_when_asked_for() {
        let model = model().unwrap();
        let names = names();
        // share of the recording kept, and the names heard in its first piece
        let run = |file: &str, only: Option<&str>| {
            let pcm = decode(&Path::new("tests/samples").join(file)).unwrap();
            let wanted = wanted(&names, only).unwrap();
            let scores = scores(&pcm, &model).unwrap();
            let keep = keep(&scores, &wanted, &pcm);
            let heard = segments(&keep).first().map_or(String::new(), |p| heard_in(p[0], p[1], &scores, &wanted, &names));
            (keep.iter().filter(|&&k| k).count() as f32 / keep.len() as f32, heard)
        };
        let (kept, heard) = run("bunny.ogv", None); // a cartoon trailer: music from start to end
        assert!(kept > 0.9 && heard.starts_with(", \"labels\": [{\"name\": \"Music\""), "{kept} {heard}");
        assert_eq!(run("en.wav", None).0, 0.0); // a minute of speech and nothing else
        let (kept, heard) = run("en.wav", Some("speech"));
        assert!(kept > 0.9 && heard.contains("\"Speech\""), "{kept} {heard}");
        assert_eq!(run("en.wav", Some("dog,siren")).0, 0.0);
    }

    fn reaches_any(scores: &[Vec<f32>], which: &[bool], threshold: f32) -> bool {
        scores.iter().any(|s| s.iter().zip(which).any(|(&v, &w)| w && v >= threshold))
    }

    const REFERENCE: [(&str, &str, f32); 13] = [
        ("1.wav", "Animal", 0.910),
        ("2.wav", "Whistling", 0.762),
        ("3.wav", "A capella", 0.666),
        ("4.wav", "Snicker", 0.539),
        ("5.wav", "Finger snapping", 0.721),
        ("6.wav", "Baby cry, infant cry", 0.699),
        ("7.wav", "Smoke detector, smoke alarm", 0.742),
        ("8.wav", "Civil defense siren", 0.771),
        ("9.wav", "Civil defense siren", 0.830),
        ("10.wav", "Trickle, dribble", 0.385),
        ("11.wav", "Cat", 0.907),
        ("12.wav", "Bark", 0.388),
        ("13.wav", "Pig", 0.647),
    ];
}

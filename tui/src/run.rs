//! The live run as the files tell it: the coach's log (results/live.log) parsed into the current run's header, its
//! reports (loss, rate, speed, the ring), the policy's windows, its events, the last sample and the end, and the
//! checkpoints the coach collected (data/live). The log is read whole once and then by its appended bytes only (a
//! line cut by a read is kept and completed by the next), so a run's record never leaves the window however long the
//! log grows (a fixed 512 KiB tail lost the opening line of a six-hour run, 2026-10-27). The part's live words are
//! live.rs.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

/// The run's non-report lines kept, newest last.
pub const EVENTS_KEEP: usize = 12;
const FILES_POLL: Duration = Duration::from_secs(1);

/// A "report chunk" line of the coach: the loss over its window, the rate in effect, the speed, the ring, and for
/// a MoE run the experts updated so far and the home bonus in force.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub chunk: u64,
    pub tokens: u64,
    pub nats: f64,
    pub bits: f64,
    pub lr: f64,
    pub scale: f64,
    pub tok_s: f64,
    pub ahead: u64,
    pub experts: Option<u64>,
    pub bias: Option<f64>,
}

/// The coach's opening line: the stream, the run's chunks, the schedule and the policy's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub started: String,
    pub file: String,
    pub first: u64,
    pub last: u64,
    pub chunks: u64,
    pub tokens: u64,
    pub lr: f64,
    pub every: u64,
    pub window: u64,
    pub min_gain: f64,
    pub halvings: u64,
    pub sample_every: u64,
    pub sample_tokens: u64,
    /// The n-gram memory's note of the opening line: "FILE: row lr 0.1, gates ones, tables every 50000".
    pub memory: Option<String>,
    /// The MoE note: "loaded: bias 100 to 0 over 100000 chunks, expert lr 0.05" (or "knobs: ..." for the knobs alone).
    pub moe: Option<String>,
}

/// A policy window: the mean loss of the last W reports and its gain over the previous window.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub mean: f64,
    pub gain: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub chunk: u64,
    pub seed: u64,
    pub metrics: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ckpt {
    pub name: String,
    pub bytes: u64,
    pub modified: SystemTime,
}

#[derive(Debug, Default)]
pub struct Run {
    pub header: Option<Header>,
    pub reports: Vec<Report>,
    pub windows: Vec<Window>,
    /// "HH:MM:SS text" of the run's commands, policy decisions, checkpoints and coach notes, oldest first.
    pub events: Vec<String>,
    pub sample: Option<Sample>,
    /// The time of the last line read, for the run's elapsed time.
    pub last_time: String,
    /// The "run ended" line's text, once the coach wrote it.
    pub ended: Option<String>,
    pub ckpts: Vec<Ckpt>,
    /// Bytes of the log taken so far, and the unterminated tail after them (bytes, so a cut UTF-8 sequence completes).
    log_len: u64,
    fragment: Vec<u8>,
    polled: Option<Instant>,
}

/// A number with the punctuation the log puts around it stripped ("1.000e-03," "(18069" "10:").
fn num(s: &str) -> Option<f64> {
    s.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')).parse().ok()
}

fn int(s: &str) -> Option<u64> {
    num(s).map(|v| v.max(0.0) as u64)
}

/// "HH:MM:SS text" split; a line without the stamp has an empty time.
fn stamped(line: &str) -> (&str, &str) {
    let b = line.as_bytes();
    if b.len() > 9 && b[2] == b':' && b[5] == b':' && b[8] == b' ' {
        (&line[..8], &line[9..])
    } else {
        ("", line)
    }
}

/// "report chunk 7100  tokens 227200  loss 4.2315 nats = 6.1045 bits/token  lr 7.500e-04  scale 0.2500  1291 tok/s
/// ring 960 ahead" (host/coach.c logline), with "  moe 120 experts, bias 90.9" (the bias while it decays) for a
/// MoE run.
pub fn report_of(rest: &str) -> Option<Report> {
    let w: Vec<&str> = rest.split_whitespace().collect();
    if w.len() < 20 || w[0] != "report" || w[1] != "chunk" {
        return None;
    }
    let moe = w.len() >= 23 && w[20] == "moe";
    Some(Report {
        chunk: int(w[2])?,
        tokens: int(w[4])?,
        nats: num(w[6])?,
        bits: num(w[9])?,
        lr: num(w[12])?,
        scale: num(w[14])?,
        tok_s: num(w[15])?,
        ahead: int(w[18])?,
        experts: if moe { int(w[21]) } else { None },
        bias: if moe && w.len() >= 25 && w[23] == "bias" { num(w[24]) } else { None },
    })
}

/// "NAME (inside)" after `key` in the opening line, as "NAME: inside".
fn note_after<'a>(rest: &'a str, key: &str) -> Option<String> {
    let (_, after) = rest.split_once(key)?;
    let (name, after) = after.split_once(" (")?;
    let (inside, _) = after.split_once(')')?;
    Some(format!("{name}: {inside}"))
}

/// "coach: FILE chunks 648..18716 (18069 chunks = 578208 tokens), lr 1.000e-03, report every 100, window 10, min
/// gain 0.010, halvings 3, sample every 1000 x 128 tokens, ..., checkpoints DIR/[, n-gram memory FILE (...)][, MoE
/// loaded (...)]"
pub fn header_of(time: &str, rest: &str) -> Option<Header> {
    let w: Vec<&str> = rest.split_whitespace().collect();
    if w.len() < 27 || w[0] != "coach:" || w[2] != "chunks" {
        return None;
    }
    let (first, last) = w[3].split_once("..")?;
    Some(Header {
        started: time.to_string(),
        file: w[1].to_string(),
        first: int(first)?,
        last: int(last)?,
        chunks: int(w[4])?,
        tokens: int(w[7])?,
        lr: num(w[10])?,
        every: int(w[13])?,
        window: int(w[15])?,
        min_gain: num(w[18])?,
        halvings: int(w[20])?,
        sample_every: int(w[23])?,
        sample_tokens: int(w[25])?,
        memory: note_after(rest, ", n-gram memory "),
        moe: note_after(rest, ", MoE "),
    })
}

/// "window: mean 7.1886 bits/token, gain 0.0141 over the previous window"
pub fn window_of(rest: &str) -> Option<Window> {
    let w: Vec<&str> = rest.split_whitespace().collect();
    if w.len() < 6 || w[0] != "window:" {
        return None;
    }
    Some(Window { mean: num(w[2])?, gain: num(w[5])? })
}

/// "sample chunk 10000 seed 10: hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112 | TEXT"
pub fn sample_of(rest: &str) -> Option<Sample> {
    let rest = rest.strip_prefix("sample chunk ")?;
    let (chunk, rest) = rest.split_once(" seed ")?;
    let (seed, rest) = rest.split_once(": ")?;
    let (metrics, text) = rest.split_once(" | ").unwrap_or((rest, ""));
    Some(Sample { chunk: int(chunk)?, seed: int(seed)?, metrics: metrics.to_string(), text: text.to_string() })
}

/// The file's bytes from `from` to its end.
fn read_from(path: &Path, from: u64) -> Option<Vec<u8>> {
    let mut f = File::open(path).ok()?;
    f.seek(SeekFrom::Start(from)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    Some(buf)
}

impl Run {
    fn reset(&mut self) {
        self.header = None;
        self.reports.clear();
        self.windows.clear();
        self.events.clear();
        self.sample = None;
        self.ended = None;
        self.last_time.clear();
    }

    fn push_event(&mut self, time: &str, rest: &str) {
        self.events.push(format!("{time} {rest}"));
        if self.events.len() > EVENTS_KEEP {
            self.events.remove(0);
        }
    }

    /// One line of the log, in order: a coach opening line starts a new record, the lines after it fill it, lines
    /// before any opening line are not a run's.
    pub fn take(&mut self, line: &str) {
        let (time, rest) = stamped(line);
        if let Some(h) = header_of(time, rest) {
            self.reset();
            self.last_time = time.to_string();
            self.header = Some(h);
            return;
        }
        if self.header.is_none() {
            return;
        }
        if !time.is_empty() {
            self.last_time = time.to_string();
        }
        if let Some(r) = report_of(rest) {
            self.reports.push(r);
        } else if let Some(w) = window_of(rest) {
            self.windows.push(w);
        } else if let Some(s) = sample_of(rest) {
            self.sample = Some(s);
        } else if rest.starts_with("coach: run ended") {
            self.ended = Some(rest.to_string());
            self.push_event(time, rest);
        } else if rest.starts_with("coach: ") || rest.starts_with("policy: ") || rest.starts_with("command: ") || rest.starts_with("checkpoint ") || rest.starts_with("sample:") || rest.starts_with("tables:") || rest.starts_with("memory:") || rest.starts_with("pool:") {
            self.push_event(time, rest);
        }
    }

    /// The current run's record from these lines (the last opening line on).
    pub fn parse(&mut self, lines: &[String]) {
        self.reset();
        for l in lines {
            self.take(l);
        }
    }

    /// The log's new bytes as whole lines (the unterminated tail kept for the next time).
    fn take_bytes(&mut self, bytes: &[u8]) {
        self.fragment.extend_from_slice(bytes);
        let cut = match self.fragment.iter().rposition(|&b| b == b'\n') {
            Some(i) => i + 1,
            None => return,
        };
        let rest = self.fragment.split_off(cut);
        let text = String::from_utf8_lossy(&std::mem::replace(&mut self.fragment, rest)).into_owned();
        for line in text.lines() {
            self.take(line);
        }
    }

    /// The log's new bytes and the checkpoint directory when due; whether anything shown changed.
    pub fn poll(&mut self, log: &Path, ckdir: &Path) -> bool {
        if self.polled.is_some_and(|t| t.elapsed() < FILES_POLL) {
            return false;
        }
        self.polled = Some(Instant::now());
        let mut changed = false;
        let len = fs::metadata(log).map_or(0, |m| m.len());
        if len < self.log_len {
            // truncated or replaced: start over
            self.log_len = 0;
            self.fragment.clear();
            self.reset();
            changed = true;
        }
        if len > self.log_len {
            if let Some(bytes) = read_from(log, self.log_len) {
                self.log_len += bytes.len() as u64;
                self.take_bytes(&bytes);
                changed = true;
            }
        }
        let mut ckpts: Vec<Ckpt> = fs::read_dir(ckdir)
            .map(|d| {
                d.filter_map(Result::ok)
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        // ckpt-N.bin, and the files a run writes beside them: pool-N.f32, tables-N.f32, ckpt-N-gates.f32
                        if !(name.ends_with(".bin") || name.ends_with(".f32")) {
                            return None;
                        }
                        let m = e.metadata().ok()?;
                        Some(Ckpt { name, bytes: m.len(), modified: m.modified().ok()? })
                    })
                    .collect()
            })
            .unwrap_or_default();
        ckpts.sort_by(|a, b| a.modified.cmp(&b.modified));
        if ckpts != self.ckpts {
            self.ckpts = ckpts;
            changed = true;
        }
        changed
    }

    /// Seconds from the run's start to its last line, by the log's own clock (no local clock needed).
    pub fn elapsed_s(&self) -> Option<i64> {
        let start = clock_s(&self.header.as_ref()?.started)?;
        let end = clock_s(&self.last_time)?;
        Some(if end >= start { end - start } else { end + 86_400 - start })
    }
}

/// "HH:MM:SS" as seconds of the day.
pub fn clock_s(t: &str) -> Option<i64> {
    let mut it = t.split(':');
    let h: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next()?.parse().ok()?;
    let s: i64 = it.next()?.parse().ok()?;
    Some(h * 3600 + m * 60 + s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const LOG: &str = "19:20:26 coach: run ended at chunk 160 (5120 tokens), 0 halvings, final scale 1.0000
20:06:29 coach: data/mix/chat-train.u16 chunks 648..18716 (18069 chunks = 578208 tokens), lr 1.000e-03, report every 100, window 10, min gain 0.010, halvings 3, sample every 1000 x 128 tokens, orchestra data/stories-10m.ngc, cmd results/live.cmd, checkpoints data/live/
20:06:29 coach: live (magic set, 960 chunks in the ring)
20:06:45 report chunk 600  tokens 19200  loss 5.5812 nats = 8.0520 bits/token  lr 1.000e-03  scale 1.0000  1224 tok/s  ring 960 ahead
20:06:48 report chunk 700  tokens 22400  loss 4.9579 nats = 7.1527 bits/token  lr 1.000e-03  scale 1.0000  1224 tok/s  ring 960 ahead
20:10:00 window: mean 7.1886 bits/token, gain 0.0141 over the previous window
20:10:53 sample chunk 10000 seed 10: hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112 | Once upon a time, there was a little boy named Tim. <|endoftext|>
20:11:00 command: scale 0.5000
20:11:01 policy: plateau -> scale 0.2500 (halving 1), checkpoint requested
20:11:02 checkpoint (requested): data/live/ckpt-3001.bin, 23463424 bytes, chunk 3001, push 2, copied in 25 ms";

    fn check_record(r: &Run) {
        let h = r.header.as_ref().unwrap();
        assert_eq!((h.started.as_str(), h.file.as_str(), h.first, h.last, h.chunks, h.tokens), ("20:06:29", "data/mix/chat-train.u16", 648, 18716, 18069, 578208));
        assert_eq!((h.lr, h.every, h.window, h.min_gain, h.halvings, h.sample_every, h.sample_tokens), (1e-3, 100, 10, 0.010, 3, 1000, 128));
        assert_eq!(r.reports.len(), 2);
        assert_eq!(r.reports[1], Report { chunk: 700, tokens: 22400, nats: 4.9579, bits: 7.1527, lr: 1e-3, scale: 1.0, tok_s: 1224.0, ahead: 960, experts: None, bias: None });
        assert_eq!((h.memory.as_deref(), h.moe.as_deref()), (None, None));
        assert_eq!(r.windows, vec![Window { mean: 7.1886, gain: 0.0141 }]);
        let s = r.sample.as_ref().unwrap();
        assert_eq!((s.chunk, s.seed, s.metrics.as_str()), (10000, 10, "hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112"));
        assert!(s.text.starts_with("Once upon a time"));
        assert_eq!(r.events.len(), 4, "{:?}", r.events);
        assert!(r.events[1].starts_with("20:11:00 command: scale"));
        assert_eq!(r.ended, None);
        assert_eq!(r.elapsed_s(), Some(4 * 60 + 33));
    }

    #[test]
    fn the_log_parses_into_the_current_run() {
        let mut r = Run::default();
        r.parse(&LOG.lines().map(String::from).collect::<Vec<_>>());
        check_record(&r);
    }

    #[test]
    fn a_new_opening_line_starts_a_new_record() {
        let mut r = Run::default();
        r.parse(&LOG.lines().map(String::from).collect::<Vec<_>>());
        r.take("21:00:00 coach: data/x.u16 chunks 0..9 (10 chunks = 320 tokens), lr 3.000e-03, report every 100, window 300, min gain 0.010, halvings 3, sample every 2000 x 128 tokens, save every 5000, no stop, cmd results/live.cmd, checkpoints data/live-x/, n-gram memory /dev/shm/t.f32 (row lr 0.1, gates ones, tables every 50000), MoE loaded (bias 100 to 0 over 100000 chunks, expert lr 0.05)");
        let h = r.header.as_ref().unwrap();
        assert_eq!(h.file, "data/x.u16");
        assert_eq!(h.memory.as_deref(), Some("/dev/shm/t.f32: row lr 0.1, gates ones, tables every 50000"));
        assert_eq!(h.moe.as_deref(), Some("loaded: bias 100 to 0 over 100000 chunks, expert lr 0.05"));
        assert!(r.reports.is_empty() && r.windows.is_empty() && r.events.is_empty() && r.sample.is_none());
        r.take("21:00:01 report chunk 9100  tokens 291200  loss 3.7231 nats = 5.3713 bits/token  lr 5.000e-04  scale 1.0000  813 tok/s  ring 960 ahead  moe 120 experts, bias 90.9");
        r.take("21:00:01 report chunk 9200  tokens 294400  loss 3.7231 nats = 5.3713 bits/token  lr 5.000e-04  scale 1.0000  813 tok/s  ring 960 ahead  moe 8804 experts");
        assert_eq!((r.reports[0].experts, r.reports[0].bias), (Some(120), Some(90.9)));
        assert_eq!((r.reports[1].experts, r.reports[1].bias), (Some(8804), None));
        r.take("21:00:01 pool: data/live-x/pool-10.f32, 4293918720 bytes in 187 rounds, 34.0 s");
        assert_eq!(r.events.len(), 1);
        r.take("21:00:02 coach: run ended at chunk 10 (320 tokens), 0 halvings, final scale 1.0000");
        assert!(r.ended.is_some());
        assert_eq!(r.elapsed_s(), Some(2));
    }

    #[test]
    fn the_log_is_read_by_its_appended_bytes_with_a_cut_line_completed() {
        let path = std::env::temp_dir().join(format!("cortex-tui-run-test-{}.log", std::process::id()));
        let ckdir = std::env::temp_dir();
        let text = LOG.to_string() + "\n";
        let (head, tail) = text.split_at(text.len() - 40);
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(head.as_bytes()).unwrap();
        }
        let mut r = Run::default();
        assert!(r.poll(&path, &ckdir));
        assert_eq!(r.events.len(), 3, "the last line is cut: not taken yet: {:?}", r.events);
        {
            let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(tail.as_bytes()).unwrap();
        }
        r.polled = None;
        assert!(r.poll(&path, &ckdir));
        check_record(&r);
        fs::remove_file(&path).unwrap();
    }
}

//! The live run as the files tell it: the coach's log (results/live.log) parsed into the current run's header, its
//! reports (loss, rate, speed, the ring), the policy's windows, its events, the last sample and the end, and the
//! checkpoints the coach collected (data/live). Read when due; the part's live words are live.rs.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

const LOG_TAIL_BYTES: u64 = 512 << 10;
/// The run's non-report lines kept, newest last.
pub const EVENTS_KEEP: usize = 12;
const FILES_POLL: Duration = Duration::from_secs(1);

/// A "report chunk" line of the coach: the loss over its window, the rate in effect, the speed, the ring.
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
    log_len: u64,
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
/// ring 960 ahead" (host/coach.c logline).
pub fn report_of(rest: &str) -> Option<Report> {
    let w: Vec<&str> = rest.split_whitespace().collect();
    if w.len() < 20 || w[0] != "report" || w[1] != "chunk" {
        return None;
    }
    Some(Report {
        chunk: int(w[2])?,
        tokens: int(w[4])?,
        nats: num(w[6])?,
        bits: num(w[9])?,
        lr: num(w[12])?,
        scale: num(w[14])?,
        tok_s: num(w[15])?,
        ahead: int(w[18])?,
    })
}

/// "coach: FILE chunks 648..18716 (18069 chunks = 578208 tokens), lr 1.000e-03, report every 100, window 10, min
/// gain 0.010, halvings 3, sample every 1000 x 128 tokens, ..."
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

impl Run {
    /// The current run's record from the log's lines: everything after the last coach opening line.
    pub fn parse(&mut self, lines: &[String]) {
        let start = lines.iter().rposition(|l| header_of(stamped(l).0, stamped(l).1).is_some());
        self.header = None;
        self.reports.clear();
        self.windows.clear();
        self.events.clear();
        self.sample = None;
        self.ended = None;
        self.last_time.clear();
        let Some(start) = start else { return };
        for line in &lines[start..] {
            let (time, rest) = stamped(line);
            if !time.is_empty() {
                self.last_time = time.to_string();
            }
            if self.header.is_none() {
                self.header = header_of(time, rest);
                continue;
            }
            if let Some(r) = report_of(rest) {
                self.reports.push(r);
            } else if let Some(w) = window_of(rest) {
                self.windows.push(w);
            } else if let Some(s) = sample_of(rest) {
                self.sample = Some(s);
            } else if rest.starts_with("coach: run ended") {
                self.ended = Some(rest.to_string());
                self.events.push(format!("{time} {rest}"));
            } else if rest.starts_with("coach: ") || rest.starts_with("policy: ") || rest.starts_with("command: ") || rest.starts_with("checkpoint ") || rest.starts_with("sample:") {
                self.events.push(format!("{time} {rest}"));
            }
        }
        let n = self.events.len();
        self.events.drain(..n.saturating_sub(EVENTS_KEEP));
    }

    /// The log's tail and the checkpoint directory when due; whether anything shown changed.
    pub fn poll(&mut self, log: &Path, ckdir: &Path) -> bool {
        if self.polled.is_some_and(|t| t.elapsed() < FILES_POLL) {
            return false;
        }
        self.polled = Some(Instant::now());
        let mut changed = false;
        let len = fs::metadata(log).map_or(0, |m| m.len());
        if len != self.log_len {
            self.log_len = len;
            self.parse(&tail_lines(log, LOG_TAIL_BYTES));
            changed = true;
        }
        let mut ckpts: Vec<Ckpt> = fs::read_dir(ckdir)
            .map(|d| {
                d.filter_map(Result::ok)
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if !name.ends_with(".bin") {
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

/// The whole lines within the file's last `bytes`.
pub fn tail_lines(path: &Path, bytes: u64) -> Vec<String> {
    let Ok(mut f) = File::open(path) else { return Vec::new() };
    let len = f.metadata().map_or(0, |m| m.len());
    let start = len.saturating_sub(bytes);
    if f.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    if start > 0 && !lines.is_empty() {
        lines.remove(0); // a line cut at the start of the window
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn the_log_parses_into_the_current_run() {
        let mut r = Run::default();
        r.parse(&LOG.lines().map(String::from).collect::<Vec<_>>());
        let h = r.header.as_ref().unwrap();
        assert_eq!((h.started.as_str(), h.file.as_str(), h.first, h.last, h.chunks, h.tokens), ("20:06:29", "data/mix/chat-train.u16", 648, 18716, 18069, 578208));
        assert_eq!((h.lr, h.every, h.window, h.min_gain, h.halvings, h.sample_every, h.sample_tokens), (1e-3, 100, 10, 0.010, 3, 1000, 128));
        assert_eq!(r.reports.len(), 2);
        assert_eq!(r.reports[1], Report { chunk: 700, tokens: 22400, nats: 4.9579, bits: 7.1527, lr: 1e-3, scale: 1.0, tok_s: 1224.0, ahead: 960 });
        assert_eq!(r.windows, vec![Window { mean: 7.1886, gain: 0.0141 }]);
        let s = r.sample.as_ref().unwrap();
        assert_eq!((s.chunk, s.seed, s.metrics.as_str()), (10000, 10, "hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112"));
        assert!(s.text.starts_with("Once upon a time"));
        assert_eq!(r.events.len(), 4, "{:?}", r.events);
        assert!(r.events[1].starts_with("20:11:00 command: scale"));
        assert_eq!(r.ended, None);
        assert_eq!(r.elapsed_s(), Some(4 * 60 + 33));
    }
}

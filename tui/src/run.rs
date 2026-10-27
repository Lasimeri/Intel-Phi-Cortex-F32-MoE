//! The live run as the harness watches it beside the chat: the coach's log tail (results/live.log), one loss point
//! per report line in it, and the checkpoints the coach collected (data/live). Read from the files when due; the
//! live words of the part are live.rs.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

const LOG_TAIL_BYTES: u64 = 64 << 10;
const LOG_KEEP: usize = 60;
pub const LOSS_KEEP: usize = 12;
const FILES_POLL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, PartialEq)]
pub struct LossPoint {
    pub chunk: u64,
    pub nats: f64,
    pub bits: f64,
    pub tok_s: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ckpt {
    pub name: String,
    pub bytes: u64,
    pub modified: SystemTime,
}

#[derive(Debug, Default)]
pub struct Run {
    /// The log's last lines, oldest first.
    pub log: Vec<String>,
    log_len: u64,
    /// The last reports' losses, oldest first (from the log's `report chunk` lines).
    pub losses: Vec<LossPoint>,
    pub ckpts: Vec<Ckpt>,
    polled: Option<Instant>,
}

impl Run {
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
            let lines = tail_lines(log, LOG_TAIL_BYTES);
            self.losses = lines.iter().filter_map(|l| report_of(l)).collect();
            let n = self.losses.len();
            self.losses.drain(..n.saturating_sub(LOSS_KEEP));
            let n = lines.len();
            self.log = lines[n.saturating_sub(LOG_KEEP)..].to_vec();
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

/// A coach report line (host/coach.c logline): "17:36:40 report chunk 7100  tokens 227200  loss 4.2315 nats =
/// 6.1045 bits/token  lr 7.500e-04  scale 0.2500  1291 tok/s  ring 960 ahead".
pub fn report_of(line: &str) -> Option<LossPoint> {
    let rest = line.split_once("report chunk ")?.1;
    let (chunk, rest) = number(rest)?;
    let (nats, rest) = number(rest.split_once("loss ")?.1)?;
    let (bits, rest) = number(rest.split_once("= ")?.1)?;
    let (tok_s, _) = number(rest.split_once("scale ")?.1.split_once("  ")?.1.trim_start())?;
    Some(LossPoint { chunk: chunk as u64, nats, bits, tok_s })
}

fn number(s: &str) -> Option<(f64, &str)> {
    let s = s.trim_start();
    let end = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+')).unwrap_or(s.len());
    Some((s[..end].parse().ok()?, &s[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_report_line() {
        let p = report_of("17:36:40 report chunk 7100  tokens 227200  loss 4.2315 nats = 6.1045 bits/token  lr 7.500e-04  scale 0.2500  1291 tok/s  ring 960 ahead").unwrap();
        assert_eq!(p, LossPoint { chunk: 7100, nats: 4.2315, bits: 6.1045, tok_s: 1291.0 });
        assert!(report_of("17:38:13 coach: run ended at chunk 10021 (320672 tokens)").is_none());
    }
}

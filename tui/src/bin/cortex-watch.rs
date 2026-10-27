//! cortex-watch: the training monitor of the Intel Phi Cortex, a standalone read-only screen in the harness's style.
//! It reads the part's live words every 250 ms (lib/cx-proto.h CX_T_*, the daemon's heartbeat) and the coach's log
//! and checkpoint directory every second, and draws the run: its progress and time left, its numbers, the loss per
//! report and the speed as column charts, the policy's window, the last sample, the events, the checkpoints.
//!
//!   cortex-watch [--tree DIR] [--log FILE] [--ckdir DIR] [--part FILE]
//!
//! Keys: tab shows both charts, the loss alone or the speed alone; q (also esc, ctrl+q, ctrl+c) quits.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

use cortex_tui::live::{Live, Reader};
use cortex_tui::run::Run;
use cortex_tui::term::{theme, Frame, Span, Term};
use cortex_tui::view::{bar, chars, clip, column_chart, dim, faint, file_name, footer_with, group, header_with, pad, rule, state_word, text, CHART_LABEL, LN_2, MIN};
use cortex_tui::wrap::wrap;

const USAGE: &str = "usage: cortex-watch [--tree DIR] [--log FILE] [--ckdir DIR] [--part FILE]
  --tree DIR     the Cortex tree the paths are relative to (default: the current directory)
  --log FILE     the coach's log (results/live.log)
  --ckdir DIR    where the coach puts the checkpoints (data/live)
  --part FILE    the shared host-memory region (/dev/shm/phi-hostmem)";
const LIVE_POLL: Duration = Duration::from_millis(250);
const HEART_STALE: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
struct Config {
    tree: String,
    log: String,
    ckdir: String,
    part: String,
}

impl Config {
    fn from_args<I: Iterator<Item = String>>(mut args: I) -> Result<Self, String> {
        let mut c = Config { tree: ".".into(), log: "results/live.log".into(), ckdir: "data/live".into(), part: "/dev/shm/phi-hostmem".into() };
        while let Some(a) = args.next() {
            let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
            match a.as_str() {
                "--tree" => c.tree = value("--tree")?,
                "--log" => c.log = value("--log")?,
                "--ckdir" => c.ckdir = value("--ckdir")?,
                "--part" => c.part = value("--part")?,
                "-h" | "--help" => return Err(USAGE.to_string()),
                other => return Err(format!("unknown option {other}\n{USAGE}")),
            }
        }
        Ok(c)
    }

    fn in_tree(&self, p: &str) -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() { p.to_path_buf() } else { Path::new(&self.tree).join(p) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Charts {
    Both,
    Loss,
    Speed,
}

struct Watch {
    cfg: Config,
    live: Option<Live>,
    reader: Option<Reader>,
    live_error: Option<String>,
    live_polled: Option<Instant>,
    heart_seen: (u64, Instant),
    alive: bool,
    run: Run,
    charts: Charts,
    quit: bool,
}

impl Watch {
    fn new(cfg: Config) -> Self {
        let now = Instant::now();
        Self { cfg, live: None, reader: None, live_error: None, live_polled: None, heart_seen: (0, now.checked_sub(HEART_STALE).unwrap_or(now)), alive: false, run: Run::default(), charts: Charts::Both, quit: false }
    }

    fn key(&mut self, code: KeyCode, ctrl: bool) {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Tab => self.charts = match self.charts { Charts::Both => Charts::Loss, Charts::Loss => Charts::Speed, Charts::Speed => Charts::Both },
            _ => {}
        }
    }

    /// The files and the part when due; whether a redraw is due.
    fn tick(&mut self) -> bool {
        let mut changed = self.poll_live();
        let log = self.cfg.in_tree(&self.cfg.log);
        let ckdir = self.cfg.in_tree(&self.cfg.ckdir);
        changed |= self.run.poll(&log, &ckdir);
        changed
    }

    fn poll_live(&mut self) -> bool {
        if self.live_polled.is_some_and(|t| t.elapsed() < LIVE_POLL) {
            return false;
        }
        self.live_polled = Some(Instant::now());
        if self.reader.is_none() {
            match Reader::open(&self.cfg.part) {
                Ok(r) => self.reader = Some(r),
                Err(e) => {
                    let msg = format!("{}: {e}", self.cfg.part);
                    let changed = self.live_error.as_deref() != Some(&msg);
                    self.live_error = Some(msg);
                    return changed;
                }
            }
        }
        let Some(r) = &self.reader else { return false };
        match r.read() {
            Ok(l) => {
                if l.heart != self.heart_seen.0 {
                    self.heart_seen = (l.heart, Instant::now());
                }
                let alive = self.heart_seen.1.elapsed() < HEART_STALE;
                let changed = self.live.as_ref() != Some(&l) || alive != self.alive || self.live_error.is_some();
                self.alive = alive;
                self.live = Some(l);
                self.live_error = None;
                changed
            }
            Err(e) => {
                self.live_error = Some(format!("{}: {e}", self.cfg.part));
                self.live = None;
                self.alive = false;
                true
            }
        }
    }
}

fn minutes(s: f64) -> String {
    if s < 90.0 {
        format!("{s:.0} s")
    } else if s < 5400.0 {
        format!("{:.1} min", s / 60.0)
    } else {
        format!("{:.1} h", s / 3600.0)
    }
}

/// The card as a few words for the header: the run's state and speed, else the daemon.
fn card_word(w: &Watch) -> Vec<Span> {
    match &w.live {
        Some(l) if l.run => vec![text(format!("● {}", state_word(l.train_state))), dim(format!(" {} tok/s", group(l.tok_s.round().max(0.0) as u64)))],
        Some(_) if w.run.ended.is_some() => vec![dim("○ run ended"), dim(if w.alive { ", daemon alive" } else { ", daemon absent" })],
        Some(_) if w.alive => vec![text("● daemon alive"), dim(", no run")],
        _ => vec![dim("○ no daemon")],
    }
}

/// The run's four lines: the progress, the position and time, the loss and the window, the rate and the policy.
fn run_rows(w: &Watch, cw: usize) -> Vec<Vec<Span>> {
    let mut rows = Vec::new();
    let Some(h) = &w.run.header else {
        rows.push(vec![dim(format!("no run in {} yet: the coach's opening line is what starts the record", w.cfg.log))]);
        return rows;
    };
    let last = w.run.reports.last();
    let live = w.live.as_ref().filter(|l| l.run);
    let done = live.map_or(last.map_or(0, |r| r.chunk), |l| l.chunks).min(h.chunks);
    let tok_s = live.map_or(last.map_or(0.0, |r| r.tok_s), |l| l.tok_s);
    let frac = if h.chunks > 0 { done as f64 / h.chunks as f64 } else { 0.0 };
    let left = if w.run.ended.is_some() { "ended".to_string() } else if tok_s > 0.0 { format!("{} left at {} tok/s", minutes((h.chunks - done) as f64 * 32.0 / tok_s), group(tok_s.round() as u64)) } else { "no speed yet".to_string() };
    let head = format!("{} chunks {}..{}: ", file_name(&h.file), group(h.first), group(h.last));
    let tail = format!(" {} of {} chunks ({:.0} %), {left}", group(done), group(h.chunks), frac * 100.0);
    let bw = cw.saturating_sub(chars(&head) + chars(&tail)).clamp(4, 40);
    let mut row = vec![dim(head)];
    row.extend(bar(frac, bw));
    row.push(text(tail));
    rows.push(row);
    let state = live.map_or(if w.run.ended.is_some() { "ended" } else { "not live" }, |l| state_word(l.train_state));
    let elapsed = w.run.elapsed_s().map_or("-".to_string(), |s| minutes(s as f64));
    let ahead = live.map_or(last.map_or(0, |r| r.ahead), |l| l.chunks_fed.saturating_sub(l.chunks));
    rows.push(vec![dim("state "), text(state), dim("   chunk "), text(format!("{} ({} tokens)", group(done), group(done * 32))), dim("   ring "), text(format!("{} ahead", group(ahead))), dim("   elapsed "), text(elapsed), dim(format!("   started {}", h.started))]);
    let (nats, bits) = live.map_or(last.map_or((0.0, 0.0), |r| (r.nats, r.bits)), |l| (l.loss_nats, l.loss_nats / LN_2));
    let mut row = vec![dim("loss "), text(format!("{nats:.4} nats = {bits:.4} bits/token")).bold()];
    match w.run.windows.last() {
        Some(win) => row.push(dim(format!("   window of {}: mean {:.4}, gain {:+.4} over the previous (a plateau is under {:.3})", h.window, win.mean, win.gain, h.min_gain))),
        None => row.push(dim(format!("   window of {} reports: none complete yet", h.window))),
    }
    rows.push(row);
    let (lr_eff, scale) = live.map_or(last.map_or((h.lr, 1.0), |r| (r.lr, r.scale)), |l| (l.lr_eff, l.scale));
    let halvings = if scale > 0.0 && scale < 1.0 { (-scale.log2()).round() as u64 } else { 0 };
    let (saves, ckpt_chunk) = live.map_or((0, 0), |l| (l.saves, l.ckpt_chunk));
    rows.push(vec![dim("rate "), text(format!("{lr_eff:.3e}")), dim(format!(" in effect = base {:.3e} x scale {:.4}   halvings {} of {}   checkpoints {} pushed, last at chunk {}   sample every {} chunks", h.lr, scale, halvings, h.halvings, saves, group(ckpt_chunk), group(h.sample_every)))]);
    rows
}

fn render(w: &Watch, width: u16, height: u16) -> Frame {
    let mut f = Frame::new(width, height);
    if width < MIN.0 || height < MIN.1 {
        f.set(0, vec![text("the terminal is too small")], theme::BG);
        f.set(1, vec![dim(format!("{} x {} at least", MIN.0, MIN.1))], theme::BG);
        return f;
    }
    header_with(&mut f, "training", card_word(w));
    let cw = width as usize - 2;
    let h = height as usize;
    let mut rows: Vec<Vec<Span>> = vec![rule("the run", cw)];
    rows.extend(run_rows(w, cw));
    rows.push(vec![]);
    // the charts share the rows left after everything else
    let sample_rows = 1 + 3;
    let events_rows = 1 + w.run.events.len().clamp(1, 6);
    let ckpt_rows = 1 + w.run.ckpts.len().clamp(1, 4);
    let fixed = 1 + rows.len() + sample_rows + 1 + events_rows + 1 + ckpt_rows + 1 + 2;
    let avail = h.saturating_sub(fixed);
    let every = w.run.header.as_ref().map_or(100, |x| x.every);
    let reports = &w.run.reports;
    let x_axis = |first: u64, last: u64| -> Vec<Span> { vec![pad(CHART_LABEL), dim(format!("chunk {}", group(first))), pad(1), faint(clip(&format!("to {}", group(last)), cw.saturating_sub(CHART_LABEL + 10)))] };
    let chart_w = cw.saturating_sub(CHART_LABEL).max(8);
    let (loss_h, speed_h) = match w.charts {
        Charts::Both if avail >= 12 => (avail * 3 / 5 - 2, avail - avail * 3 / 5 - 2),
        Charts::Both => (avail.saturating_sub(3).max(3), 0),
        Charts::Loss => (avail.saturating_sub(3).max(3), 0),
        Charts::Speed => (0, avail.saturating_sub(3).max(3)),
    };
    if loss_h > 0 {
        rows.push(rule(&format!("loss, bits per token, one column per report of {} chunks (tab: the charts)", group(every)), cw));
        if reports.is_empty() {
            rows.push(vec![dim("no report yet")]);
        } else {
            let values: Vec<f64> = reports.iter().map(|r| r.bits).collect();
            rows.extend(column_chart(&values, chart_w, loss_h, &|v| format!("{v:.2}")));
            let n = reports.len().min(chart_w);
            rows.push(x_axis(reports[reports.len() - n].chunk, reports[reports.len() - 1].chunk));
        }
        rows.push(vec![]);
    }
    if speed_h > 0 {
        rows.push(rule("speed, tokens per second, idle time included", cw));
        if reports.is_empty() {
            rows.push(vec![dim("no report yet")]);
        } else {
            let values: Vec<f64> = reports.iter().map(|r| r.tok_s).collect();
            rows.extend(column_chart(&values, chart_w, speed_h, &|v| group(v.round().max(0.0) as u64)));
            let n = reports.len().min(chart_w);
            rows.push(x_axis(reports[reports.len() - n].chunk, reports[reports.len() - 1].chunk));
        }
        rows.push(vec![]);
    }
    match &w.run.sample {
        Some(s) => {
            rows.push(rule(&format!("the last sample, chunk {} seed {}: {}", group(s.chunk), s.seed, s.metrics), cw));
            let lines = wrap(&s.text.replace("<|endoftext|>", " ∎"), cw.saturating_sub(1));
            for l in lines.iter().take(3) {
                rows.push(vec![pad(1), text(l.clone())]);
            }
        }
        None => {
            rows.push(rule("the last sample", cw));
            rows.push(vec![dim("none yet")]);
        }
    }
    rows.push(vec![]);
    rows.push(rule("events", cw));
    if w.run.events.is_empty() {
        rows.push(vec![dim("none yet")]);
    }
    let n = w.run.events.len();
    for e in &w.run.events[n.saturating_sub(6)..] {
        rows.push(vec![faint(clip(e, cw))]);
    }
    rows.push(vec![]);
    rows.push(rule(&format!("checkpoints in {}", w.cfg.ckdir), cw));
    if w.run.ckpts.is_empty() {
        rows.push(vec![dim("none yet")]);
    }
    for c in w.run.ckpts.iter().rev().take(4) {
        let age = c.modified.elapsed().map_or(0, |d| d.as_secs());
        rows.push(vec![text(format!("{:<20}", c.name)), dim(format!("{:>12} B  {} ago", group(c.bytes), minutes(age as f64)))]);
    }
    for (r, row) in rows.into_iter().take(h.saturating_sub(3)).enumerate() {
        let mut full = vec![pad(1)];
        full.extend(row);
        f.set((r + 1) as u16, full, theme::BG);
    }
    let status = match &w.live_error {
        Some(e) => Some((e.as_str(), true)),
        None => None,
    };
    let note = format!("{} · {} · the part every 250 ms, the log every second", w.cfg.log, w.cfg.ckdir);
    let status = status.or(Some((note.as_str(), false)));
    footer_with(&mut f, status, &[("tab", "the charts")], ("q", "quit"));
    f
}

fn main() -> ExitCode {
    let mut cfg = match Config::from_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("cortex-watch: {msg}");
            return ExitCode::from(2);
        }
    };
    match std::fs::canonicalize(&cfg.tree) {
        Ok(t) => cfg.tree = t.to_string_lossy().into_owned(),
        Err(e) => {
            eprintln!("cortex-watch: --tree {}: {e}", cfg.tree);
            return ExitCode::from(1);
        }
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!("cortex-watch: needs a terminal on stdin and stdout (tools/watch.sh opens one on the desktop)");
        return ExitCode::from(1);
    }
    let mut term = match Term::open() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cortex-watch: the terminal: {e}");
            return ExitCode::from(1);
        }
    };
    let mut watch = Watch::new(cfg);
    let mut dirty = true;
    let mut last_draw = Instant::now();
    let outcome = loop {
        if dirty {
            let (cols, lines) = Term::size();
            let frame = render(&watch, cols, lines);
            if let Err(e) = term.draw(&frame) {
                break Err(e);
            }
            dirty = false;
            last_draw = Instant::now();
        }
        match event::poll(Duration::from_millis(100)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(k)) => {
                    if k.kind != KeyEventKind::Release {
                        watch.key(k.code, k.modifiers.contains(KeyModifiers::CONTROL));
                        dirty = true;
                    }
                }
                Ok(Event::Resize(..)) => dirty = true,
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => {}
            Err(e) => break Err(e),
        }
        if watch.tick() || last_draw.elapsed() > Duration::from_secs(5) {
            dirty = true;
        }
        if watch.quit {
            break Ok(());
        }
    };
    drop(term);
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cortex-watch: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_size_fits_with_and_without_a_run() {
        let log = "20:06:29 coach: data/mix/chat-train.u16 chunks 648..18716 (18069 chunks = 578208 tokens), lr 1.000e-03, report every 100, window 10, min gain 0.010, halvings 3, sample every 1000 x 128 tokens, orchestra x, cmd y, checkpoints z/
20:06:45 report chunk 600  tokens 19200  loss 5.5812 nats = 8.0520 bits/token  lr 1.000e-03  scale 1.0000  1224 tok/s  ring 960 ahead
20:06:48 report chunk 700  tokens 22400  loss 4.9579 nats = 7.1527 bits/token  lr 1.000e-03  scale 1.0000  1225 tok/s  ring 960 ahead
20:10:00 window: mean 7.1886 bits/token, gain 0.0141 over the previous window
20:10:53 sample chunk 1000 seed 10: hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112 | Once upon a time, there was a little boy named Tim. <|endoftext|>
20:11:00 command: scale 0.5000";
        for with_run in [false, true] {
            for charts in [Charts::Both, Charts::Loss, Charts::Speed] {
                for (cols, lines) in [(40u16, 10u16), (80, 24), (120, 40), (200, 60)] {
                    let mut w = Watch::new(Config { tree: ".".into(), log: "l".into(), ckdir: "c".into(), part: "p".into() });
                    w.charts = charts;
                    if with_run {
                        w.run.parse(&log.lines().map(String::from).collect::<Vec<_>>());
                    }
                    let f = render(&w, cols, lines);
                    for r in 0..lines as usize {
                        assert!(f.row_width(r) <= cols as usize, "{cols}x{lines} run {with_run}: row {r} is {} wide", f.row_width(r));
                    }
                    if with_run && cols >= 80 && lines >= 24 {
                        assert!((0..lines as usize).any(|r| f.row_text(r).contains("loss ")), "{cols}x{lines}: no loss line");
                    }
                }
            }
        }
    }
}

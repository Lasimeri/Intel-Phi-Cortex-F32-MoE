//! cortex-watch: the training monitor of the Intel Phi Cortex, in the harness's style, with a say line. The run is
//! watched (the part's live words every 250 ms: lib/cx-proto.h CX_T_* and the daemon's heartbeat; the coach's log
//! and checkpoint directory every second) and drawn: its progress and time left, its numbers, the loss per report
//! and the speed as column charts, the policy's window, the last sample, the events, the checkpoints. The run is
//! steered from the same screen (lines for the coach's command file: pause, resume, save, sample, the scale, the
//! base rate, the report cadence, stop), and the model is talked to on its live weights between chunks (one
//! `cortex --live` per turn, framed by tui/cortex.jinja over the system prompt, as the harness does; chat.rs).
//! Nothing typed here and nothing the model writes is training data.
//!
//!   cortex-watch [--tree DIR] [--log FILE] [--ckdir DIR] [--part FILE] [--cmd FILE] [--cortex PATH]
//!                [--orchestra NGC|off] [--words LIST] [--system FILE|off] [--template FILE] [--name NAME]
//!                [--ngen N] [--ctx CHARS]
//!
//! Keys: enter says the line (a `/command` steers the coach or sets a knob; f1 lists them), f2 pause or resume, f3
//! save, f4 sample, f5 and f6 the rate's scale doubled or halved, tab the charts, pgup/pgdn the talk, ctrl+q quit
//! (a reply in flight finishes first: the client is never killed).

use std::io::IsTerminal;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use cortex_tui::chat::{self, Chat, Config};
use cortex_tui::editor::Editor;
use cortex_tui::live::LiveState;
use cortex_tui::run::Run;
use cortex_tui::term::{self, theme, Frame, Span, Term};
use cortex_tui::view::{bar, chars, clip, column_chart, conversation, dim, faint, file_name, footer_with, group, header_with, help_rows, input_row, moe_pairs, pad, rule, state_word, text, CHART_LABEL, LN_2, MIN, SPIN};
use cortex_tui::wrap::wrap;

const INFO_FOR: Duration = Duration::from_secs(8);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charts {
    Both,
    Loss,
    Speed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Screen {
    Main,
    Help,
}

struct Watch {
    cfg: Config,
    /// Why no turn can run here (the tree lacks the client or the frame); the run is watched all the same.
    talk_error: Option<String>,
    chat: Chat,
    input: Editor,
    /// Rows scrolled up from the bottom of the talk; 0 follows the newest.
    back: usize,
    /// Talk rows in the last frame: a page.
    page: usize,
    help_back: usize,
    spin: usize,
    live: LiveState,
    run: Run,
    charts: Charts,
    screen: Screen,
    /// The status row: the message, whether it is an error (kept until the next message; an info clears after 8 s),
    /// and when it was set.
    status: Option<(String, bool, Instant)>,
    quit: bool,
}

impl Watch {
    fn new(cfg: Config, talk_error: Option<String>) -> Self {
        let chat = Chat::new(&cfg);
        let mut w = Self {
            chat,
            cfg,
            talk_error,
            input: Editor::new(),
            back: 0,
            page: 0,
            help_back: 0,
            spin: 0,
            live: LiveState::new(),
            run: Run::default(),
            charts: Charts::Both,
            screen: Screen::Main,
            status: None,
            quit: false,
        };
        if let Some(e) = w.talk_error.clone() {
            w.info(format!("the run is watched; no turn can run here: {e}"));
        }
        w
    }

    fn done(&self) -> bool {
        self.quit && !self.chat.in_flight
    }

    fn info(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), false, Instant::now()));
    }

    fn error(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), true, Instant::now()));
    }

    fn report(&mut self, r: Result<String, String>) {
        match r {
            Ok(m) => self.info(m),
            Err(e) => self.error(e),
        }
    }

    fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        match k.code {
            KeyCode::Char('q') | KeyCode::Char('c') if ctrl => return self.request_quit(),
            KeyCode::F(1) => {
                self.screen = if self.screen == Screen::Help { Screen::Main } else { Screen::Help };
                return;
            }
            _ => {}
        }
        match self.screen {
            Screen::Main => self.main_key(k, ctrl, alt),
            Screen::Help => match k.code {
                KeyCode::Esc => self.screen = Screen::Main,
                KeyCode::Up | KeyCode::PageUp => self.help_back = self.help_back.saturating_sub(1),
                KeyCode::Down | KeyCode::PageDown => self.help_back += 1,
                _ => {}
            },
        }
    }

    fn main_key(&mut self, k: KeyEvent, ctrl: bool, alt: bool) {
        match k.code {
            KeyCode::Enter => self.say(),
            KeyCode::Esc => self.input.set_text(""),
            KeyCode::Tab => self.cycle_charts(),
            KeyCode::F(2) => self.pause_or_resume(),
            KeyCode::F(3) => self.coach("save".into()),
            KeyCode::F(4) => self.coach("sample".into()),
            KeyCode::F(5) => self.rescale(2.0),
            KeyCode::F(6) => self.rescale(0.5),
            KeyCode::PageUp => self.back += self.page.max(1),
            KeyCode::PageDown => self.back = self.back.saturating_sub(self.page.max(1)),
            KeyCode::Up => self.back += 1,
            KeyCode::Down => self.back = self.back.saturating_sub(1),
            KeyCode::Char('z') if ctrl => {
                if !self.input.undo() {
                    self.info("nothing to undo");
                }
            }
            _ => {
                chat::edit_key(&mut self.input, k.code, ctrl, alt);
            }
        }
    }

    fn cycle_charts(&mut self) {
        self.charts = match self.charts {
            Charts::Both => Charts::Loss,
            Charts::Loss => Charts::Speed,
            Charts::Speed => Charts::Both,
        };
    }

    /// A bracketed paste, into the line being typed.
    fn paste(&mut self, s: &str) {
        if self.screen == Screen::Main {
            self.input.insert_str(s);
        }
    }

    fn request_quit(&mut self) {
        self.quit = true;
        if self.chat.in_flight {
            self.info("quitting after the reply in flight (the client is never killed: its QUIT closes the daemon's session)");
        }
    }

    fn say(&mut self) {
        let line = self.input.text().trim().to_string();
        if line.is_empty() {
            return;
        }
        if let Some(cmd) = line.strip_prefix('/') {
            self.input.set_text("");
            self.command(cmd);
            return;
        }
        if let Some(e) = &self.talk_error {
            let e = e.clone();
            return self.error(format!("no turn can run here: {e}"));
        }
        if self.chat.in_flight {
            return self.error("a reply is streaming; press enter again when it has ended");
        }
        self.input.set_text("");
        match self.chat.say(&self.cfg, line) {
            Ok(()) => self.back = 0,
            Err(e) => self.error(e),
        }
    }

    /// A `/command`: the screen, the charts and quitting here; the knobs and the coach's lines in chat.rs.
    fn command(&mut self, cmd: &str) {
        let mut it = cmd.split_whitespace();
        let name = it.next().unwrap_or("");
        let arg = it.next();
        match name {
            "help" | "?" => self.screen = Screen::Help,
            "charts" => self.cycle_charts(),
            "quit" | "exit" | "q" => self.request_quit(),
            "" => {}
            _ => {
                if let Some(r) = self.chat.setting(&mut self.cfg, name, arg) {
                    if r.is_ok() && matches!(name, "clear" | "new" | "retry" | "again") {
                        self.back = 0;
                    }
                    self.report(r);
                } else if let Some(r) = chat::coach_line(name, arg) {
                    match r {
                        Ok(line) => self.coach(line),
                        Err(e) => self.error(e),
                    }
                } else {
                    self.error(format!("unknown command /{name}; f1 lists them"));
                }
            }
        }
    }

    /// `pause` while the run trains, `resume` while it is paused.
    fn pause_or_resume(&mut self) {
        let line = if self.live.paused() { "resume" } else { "pause" };
        self.coach(line.into());
    }

    /// The scale knob multiplied, from the value the card reports.
    fn rescale(&mut self, by: f64) {
        let Some(scale) = self.live.scale() else { return self.error("no live run is feeding the daemon") };
        self.coach(format!("scale {}", scale * by));
    }

    /// A line for the coach's command file.
    fn coach(&mut self, line: String) {
        let r = chat::coach_send(&self.cfg, self.live.running(), &line);
        self.report(r);
    }

    /// The turn in flight, the part, the run's files, the status row's age; whether a redraw is due.
    fn tick(&mut self) -> bool {
        let (mut changed, failed) = self.chat.pump();
        if let Some(e) = failed {
            self.error(e);
        }
        if self.chat.in_flight {
            self.spin = self.spin.wrapping_add(1);
            changed = true;
        }
        changed |= self.live.poll(&self.cfg.part);
        let log = self.cfg.in_tree(&self.cfg.log_file);
        let ckdir = self.cfg.in_tree(&self.cfg.ckdir);
        changed |= self.run.poll(&log, &ckdir);
        if self.status.as_ref().is_some_and(|(_, err, t)| !err && t.elapsed() > INFO_FOR) {
            self.status = None;
            changed = true;
        }
        changed
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

/// The card as a few words for the header: the turn in flight, else the run's state and speed, else the daemon.
fn card_word(w: &Watch) -> Vec<Span> {
    if w.chat.in_flight {
        let s = SPIN[w.spin % SPIN.len()];
        return vec![text(format!("{s} generating {:.1} s", w.chat.turn_started.elapsed().as_secs_f64()))];
    }
    match &w.live.live {
        Some(l) if l.run => vec![text(format!("● {}", state_word(l.train_state))), dim(format!(" {} tok/s", group(l.tok_s.round().max(0.0) as u64)))],
        Some(_) if w.run.ended.is_some() => vec![dim("○ run ended"), dim(if w.live.alive { ", daemon alive" } else { ", daemon absent" })],
        Some(_) if w.live.alive => vec![text("● daemon alive"), dim(", no run")],
        _ => vec![dim("○ no daemon")],
    }
}

/// The run's lines: the progress, the position and time, the loss and the window, the rate and the policy, and for
/// a run with the memory or the pool a line of their words.
fn run_rows(w: &Watch, cw: usize) -> Vec<Vec<Span>> {
    let mut rows = Vec::new();
    let Some(h) = &w.run.header else {
        rows.push(vec![dim(format!("no run in {} yet: the coach's opening line is what starts the record", w.cfg.log_file))]);
        return rows;
    };
    let last = w.run.reports.last();
    let live = w.live.live.as_ref().filter(|l| l.run);
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
    // the pool: the part's words while the run is live, else the last report's; the opening line's note after them
    let moe_report = last.and_then(|r| r.experts.map(|e| (e, r.bias)));
    if live.is_some_and(|l| l.moe) || moe_report.is_some() || h.moe.is_some() {
        let mut row = vec![dim("moe  ")];
        match live.filter(|l| l.moe) {
            Some(l) => row.push(text(moe_pairs(l).iter().enumerate().map(|(i, (k, v))| if i == 0 { v.clone() } else { format!("{k} {v}") }).collect::<Vec<_>>().join("   "))),
            None => match moe_report {
                Some((e, b)) => row.push(text(format!("{} experts updated{}", group(e), b.map_or(String::new(), |b| format!(", bias {b}"))))),
                None => row.push(text("routing through the pool")),
            },
        }
        if let Some(m) = &h.moe {
            row.push(dim(format!("   {m}")));
        }
        rows.push(row);
    }
    if live.is_some_and(|l| l.memory) || h.memory.is_some() {
        rows.push(vec![dim("memory "), text(if live.is_some_and(|l| l.memory) { "trained with the run" } else { "in the opening line" }), dim(h.memory.as_ref().map_or(String::new(), |m| format!("   {m}")))]);
    }
    rows
}

const HELP: &[(&str, &str)] = &[
    ("enter", "say it: the model continues the talk on its live weights, framed by tui/cortex.jinja over the system prompt"),
    ("esc", "clear the line (on this screen: back)"),
    ("f2", "pause the run, or resume it while it is paused"),
    ("f3 f4", "a checkpoint now; a sample now (the coach's log shows it)"),
    ("f5 f6", "the rate's scale doubled, halved (from the value the card reports)"),
    ("tab", "both charts, the loss alone, the speed alone"),
    ("pgup pgdn ↑ ↓", "scroll the talk"),
    ("ctrl+a ctrl+e ctrl+u ctrl+k", "the line: its start, its end, erase to the start, to the end"),
    ("ctrl+w ctrl+←/→ ctrl+z", "erase a word, move by words, a step back"),
    ("ctrl+q", "quit (a reply in flight finishes first)"),
    ("", ""),
    ("/pause /resume /save /sample", "lines for the coach's command file (a running coach reads them; as f2, f3, f4)"),
    ("/scale X /lr X /every N", "the learning-rate multiplier, the base rate, the report cadence"),
    ("/stop-run", "the coach ends the run after the current chunk, with a final checkpoint"),
    ("", ""),
    ("/n N", "tokens per reply (1 to 2048)"),
    ("/ctx CHARS", "transcript characters re-fed as the prompt after the system prompt"),
    ("/temp /topk /topp /minp /rp X", "the sampler (llama.cpp's order; defaults 0.5, 75, 0.95, 0.05, 1.2)"),
    ("/greedy /orchestra [on|off]", "the argmax token; the host n-gram mix, word filter and repeat block"),
    ("/think [on|off]", "the model reasons in its think block first (on), or is given an empty block and answers at once"),
    ("/system [on|off|FILE]", "the system message: ~/CLAUDE.md unless told otherwise, read at every turn"),
    ("/seed N | auto", "a fixed seed (the same prompt gives the same reply), or fresh each turn"),
    ("/retry /clear", "the last reply again; a new talk"),
    ("/charts /help /quit", "as tab, f1, ctrl+q"),
];

/// The help screen: the keys and the commands, scrolling when they outnumber the rows.
fn help(f: &mut Frame, w: &mut Watch) {
    header_with(f, "help", card_word(w));
    let cw = f.width as usize - 2;
    let h = f.height as usize;
    let rows = help_rows(HELP, cw);
    let room = h.saturating_sub(4);
    let back = w.help_back.min(rows.len().saturating_sub(room));
    w.help_back = back;
    for (r, row) in rows.into_iter().skip(back).take(room).enumerate() {
        let mut full = vec![pad(1)];
        full.extend(row);
        f.set((r + 1) as u16, full, theme::BG);
    }
    let status = w.status.as_ref().map(|(s, err, _)| (s.as_str(), *err));
    footer_with(f, status, &[("↑/↓", "scroll")], ("esc", "back"));
}

/// The main screen: the run, the charts, the sample, the events, the checkpoints, then the talk above the say line.
/// Each chart block costs its height + 4 rows (the rule, the axis, the x axis, a blank); the talk keeps its rows
/// whatever the terminal's height, the sections above it are cut when they do not fit.
fn main_screen(f: &mut Frame, w: &mut Watch) {
    header_with(f, "training", card_word(w));
    let cw = f.width as usize - 2;
    let h = f.height as usize;
    let mut rows: Vec<Vec<Span>> = vec![rule("the run", cw)];
    rows.extend(run_rows(w, cw));
    rows.push(vec![]);
    let talk_h = (h / 6).clamp(3, 10);
    let sample_rows = 1 + 3;
    let events_rows = 1 + w.run.events.len().clamp(1, 5);
    let ckpt_rows = 1 + w.run.ckpts.len().clamp(1, 3);
    let fixed = 1 + rows.len() + sample_rows + 1 + events_rows + 1 + ckpt_rows + 1 + (1 + talk_h) + 1 + 2;
    let avail = h.saturating_sub(fixed);
    let every = w.run.header.as_ref().map_or(100, |x| x.every);
    let reports = &w.run.reports;
    let x_axis = |first: u64, last: u64| -> Vec<Span> { vec![pad(CHART_LABEL), dim(format!("chunk {}", group(first))), pad(1), faint(clip(&format!("to {}", group(last)), cw.saturating_sub(CHART_LABEL + 10)))] };
    let chart_w = cw.saturating_sub(CHART_LABEL).max(8);
    let (loss_h, speed_h) = match w.charts {
        Charts::Both if avail >= 14 => {
            let l = (avail - 8) * 3 / 5;
            (l, avail - 8 - l)
        }
        Charts::Both | Charts::Loss => (avail.saturating_sub(4).max(3), 0),
        Charts::Speed => (0, avail.saturating_sub(4).max(3)),
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
    for e in &w.run.events[n.saturating_sub(5)..] {
        rows.push(vec![faint(clip(e, cw))]);
    }
    rows.push(vec![]);
    rows.push(rule(&format!("checkpoints in {}", w.cfg.ckdir), cw));
    if w.run.ckpts.is_empty() {
        rows.push(vec![dim("none yet")]);
    }
    for c in w.run.ckpts.iter().rev().take(3) {
        let age = c.modified.elapsed().map_or(0, |d| d.as_secs());
        rows.push(vec![text(format!("{:<20}", c.name)), dim(format!("{:>14} B  {} ago", group(c.bytes), minutes(age as f64)))]);
    }
    rows.push(vec![]);
    // the talk keeps its rows: the sections above are cut when the terminal is short
    rows.truncate(h.saturating_sub(4 + 1 + talk_h));
    rows.push(rule("talk: the model on its live weights, between chunks", cw));
    let conv = conversation(&w.chat.turns, w.spin, cw);
    let back = w.back.min(conv.len().saturating_sub(talk_h));
    w.back = back;
    w.page = talk_h;
    let end = conv.len() - back;
    let start = end.saturating_sub(talk_h);
    rows.extend(conv[start..end].iter().cloned());
    for (r, row) in rows.into_iter().take(h.saturating_sub(4)).enumerate() {
        let mut full = vec![pad(1)];
        full.extend(row);
        f.set((r + 1) as u16, full, theme::BG);
    }
    input_row(f, (h - 3) as u16, " say › ", &mut w.input);
    let note = format!("{} · {} · the part every 250 ms, the log every second · f1 help", w.cfg.log_file, w.cfg.ckdir);
    let status = match (&w.status, &w.live.error) {
        (Some((s, err, _)), _) => Some((s.as_str(), *err)),
        (None, Some(e)) => Some((e.as_str(), true)),
        (None, None) => Some((note.as_str(), false)),
    };
    let pr = if w.live.paused() { "resume" } else { "pause" };
    footer_with(f, status, &[("enter", "say"), ("f2", pr), ("f3", "save"), ("f4", "sample"), ("f5", "scale x2"), ("f6", "scale /2"), ("tab", "charts"), ("f1", "help")], ("ctrl+q", "quit"));
}

fn render(w: &mut Watch, width: u16, height: u16) -> Frame {
    let mut f = Frame::new(width, height);
    if width < MIN.0 || height < MIN.1 {
        f.set(0, vec![text("the terminal is too small")], theme::BG);
        f.set(1, vec![dim(format!("{} x {} at least", MIN.0, MIN.1))], theme::BG);
        return f;
    }
    match w.screen {
        Screen::Main => main_screen(&mut f, w),
        Screen::Help => help(&mut f, w),
    }
    f
}

fn main() -> ExitCode {
    let mut cfg = match Config::from_args("cortex-watch", std::env::args().skip(1)) {
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
    // the tree is there: a missing client or frame leaves the run watched and the say line refused
    let talk_error = cfg.check().err();
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
    let mut watch = Watch::new(cfg, talk_error);
    let mut dirty = true;
    let mut last_draw = Instant::now();
    let outcome = loop {
        if dirty {
            let (cols, lines) = Term::size();
            let frame = render(&mut watch, cols, lines);
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
                        watch.key(k);
                        dirty = true;
                    }
                }
                Ok(Event::Paste(s)) => {
                    watch.paste(&s);
                    dirty = true;
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
        if let Some(p) = term::take_worker_panic() {
            watch.error(format!("a turn's reader failed: {p}"));
            dirty = true;
        }
        if watch.done() {
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
    use cortex_tui::chat::{Turn, Who, EOT};

    fn config() -> Config {
        Config {
            tree: ".".into(),
            cortex: "c".into(),
            orchestra: None,
            orchestra_given: false,
            words: "w".into(),
            vocab: None,
            ngram: None,
            system: None,
            template: "t".into(),
            name: None,
            cmd_file: "cmd".into(),
            log_file: "l".into(),
            ckdir: "c".into(),
            part: "p".into(),
            ngen: 8,
            ctx_chars: 4000,
        }
    }

    #[test]
    fn every_size_fits_with_and_without_a_run_and_shows_the_talk() {
        let log = "20:06:29 coach: data/mix/chat-train.u16 chunks 648..18716 (18069 chunks = 578208 tokens), lr 1.000e-03, report every 100, window 10, min gain 0.010, halvings 3, sample every 1000 x 128 tokens, orchestra x, cmd y, checkpoints z/
20:06:45 report chunk 600  tokens 19200  loss 5.5812 nats = 8.0520 bits/token  lr 1.000e-03  scale 1.0000  1224 tok/s  ring 960 ahead
20:06:48 report chunk 700  tokens 22400  loss 4.9579 nats = 7.1527 bits/token  lr 1.000e-03  scale 1.0000  1225 tok/s  ring 960 ahead
20:10:00 window: mean 7.1886 bits/token, gain 0.0141 over the previous window
20:10:53 sample chunk 1000 seed 10: hit 1.000 hit5 1.000 big 1.000 rep4 0.009 sent 12.4 words 112 | Once upon a time, there was a little boy named Tim. <|endoftext|>
20:11:00 command: scale 0.5000";
        for with_run in [false, true] {
            for charts in [Charts::Both, Charts::Loss, Charts::Speed] {
                for (cols, lines) in [(40u16, 10u16), (80, 24), (120, 42), (200, 60)] {
                    let mut w = Watch::new(config(), None);
                    w.charts = charts;
                    if with_run {
                        w.run.parse(&log.lines().map(String::from).collect::<Vec<_>>());
                    }
                    w.chat.turns.push(Turn::new(Who::You, "Once upon a time there was a very long sentence that needs wrapping at narrow widths to be drawn whole".into()));
                    let mut c = Turn::new(Who::Cortex, String::new());
                    c.append(b" there was a cat who liked to sit in the sun.");
                    c.append(EOT.as_bytes());
                    c.finish(&[], None, false);
                    w.chat.turns.push(c);
                    for screen in [Screen::Main, Screen::Help] {
                        w.screen = screen;
                        let f = render(&mut w, cols, lines);
                        for r in 0..lines as usize {
                            assert!(f.row_width(r) <= cols as usize, "{screen:?} {charts:?} {cols}x{lines} run {with_run}: row {r} is {} wide", f.row_width(r));
                        }
                        if let Some((x, y)) = f.cursor {
                            assert!(x < cols && y < lines, "{cols}x{lines}: the cursor at {x},{y}");
                        }
                        if screen == Screen::Main {
                            assert!(f.row_text(lines as usize - 3).starts_with(" say ›"), "{cols}x{lines}: no say line");
                            if cols >= 80 && lines >= 24 {
                                assert!((0..lines as usize).any(|r| f.row_text(r).contains("cortex there was a cat")), "{charts:?} {cols}x{lines} run {with_run}: the talk is not shown");
                                if with_run {
                                    assert!((0..lines as usize).any(|r| f.row_text(r).contains("loss ")), "{cols}x{lines}: no loss line");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_say_line_refuses_turns_without_a_client_and_routes_commands() {
        let mut w = Watch::new(config(), Some("no client".into()));
        w.input.set_text("hello");
        w.say();
        assert!(w.status.as_ref().is_some_and(|s| s.1 && s.0.starts_with("no turn can run here")));
        assert_eq!(w.input.text(), "hello", "the line is kept");
        w.input.set_text("/pause");
        w.say();
        assert!(w.status.as_ref().is_some_and(|s| s.1 && s.0.starts_with("not sent")), "{:?}", w.status);
        w.input.set_text("/temp 0.7");
        w.say();
        assert_eq!(w.chat.sampling.temp, 0.7);
        w.input.set_text("/charts");
        w.say();
        assert_eq!(w.charts, Charts::Loss);
    }
}

//! The harness's state: what is on screen and what each key does, the turns, the turn in flight, the live words,
//! the run's files. Drawing is view.rs. The visual style is MechJev's (seaof.glass's palette, term.rs); the harness
//! is the Phi Cortex's: a chat on the daemon's live weights with the card's and the live run's numbers beside it, a
//! run screen that steers the coach, and ~/CLAUDE.md as the system prompt of every turn.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::cortex::{self, Request, Sampling, Stats};
use crate::editor::Editor;
use crate::frame::{self, Message, Rendered};
use crate::live::{Live, Reader};
use crate::run::Run;

/// The story separator of the BPE (tools/bpe.c): one token; its literal in a prompt tokenises back to it.
pub const EOT: &str = "<|endoftext|>";
pub const NGEN_MAX: u32 = 2048;
const LIVE_POLL: Duration = Duration::from_millis(250);
const HEART_STALE: Duration = Duration::from_secs(2);
const INFO_FOR: Duration = Duration::from_secs(8);

pub const USAGE: &str = "usage: cortex-tui [--tree DIR] [--cortex PATH] [--orchestra NGC|off] [--words LIST] [--system FILE|off]
                  [--template FILE] [--name NAME] [--cmd FILE] [--log FILE] [--ckdir DIR] [--part FILE]
                  [--ngen N] [--ctx CHARS]
  --tree DIR        the Cortex tree the client runs in (default: the current directory)
  --cortex PATH     the host client (build/host/cortex)
  --orchestra NGC   the orchestrator's n-gram count model (data/stories-10m.ngc), or off for the plain model
  --words LIST      the orchestrator's word list (data/wordlist-stories.txt)
  --system FILE     the system message of every turn (~/CLAUDE.md when it exists), or off
  --template FILE   the frame of every turn, a Jinja template (tui/cortex.jinja): the identity, the labels, the stop
  --name NAME       the identity's name in the frame (the template's own default when not given)
  --cmd FILE        the coach's command file, where the run commands go (results/live.cmd)
  --log FILE        the coach's log, read for the reports (results/live.log)
  --ckdir DIR       where the coach puts the checkpoints (data/live)
  --part FILE       the shared host-memory region (/dev/shm/phi-hostmem)
  --ngen N          tokens per reply (128, at most 2048)
  --ctx CHARS       dialogue characters kept in a turn's prompt; the oldest exchanges are dropped beyond it (4000)";

/// Command-line configuration; every value defaults to the tree's layout.
#[derive(Clone, Debug)]
pub struct Config {
    pub tree: String,
    pub cortex: String,
    pub orchestra: Option<String>,
    pub words: String,
    pub system: Option<PathBuf>,
    pub template: String,
    pub name: Option<String>,
    pub cmd_file: String,
    pub log_file: String,
    pub ckdir: String,
    pub part: String,
    pub ngen: u32,
    pub ctx_chars: usize,
}

/// `~/x` as a path under the home directory.
pub fn expand_home(p: &str) -> String {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => format!("{}/{rest}", home.to_string_lossy()),
        _ => p.to_string(),
    }
}

impl Config {
    pub fn from_args<I: Iterator<Item = String>>(mut args: I) -> Result<Self, String> {
        let home_claude = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("CLAUDE.md")).filter(|p| p.is_file());
        let mut c = Config {
            tree: ".".into(),
            cortex: "build/host/cortex".into(),
            orchestra: Some("data/stories-10m.ngc".into()),
            words: "data/wordlist-stories.txt".into(),
            system: home_claude,
            template: "tui/cortex.jinja".into(),
            name: None,
            cmd_file: "results/live.cmd".into(),
            log_file: "results/live.log".into(),
            ckdir: "data/live".into(),
            part: "/dev/shm/phi-hostmem".into(),
            ngen: 128,
            ctx_chars: 4000,
        };
        while let Some(a) = args.next() {
            let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
            match a.as_str() {
                "--tree" => c.tree = value("--tree")?,
                "--cortex" => c.cortex = value("--cortex")?,
                "--orchestra" => {
                    let v = value("--orchestra")?;
                    c.orchestra = if v == "off" { None } else { Some(v) };
                }
                "--words" => c.words = value("--words")?,
                "--system" => {
                    let v = value("--system")?;
                    c.system = if v == "off" { None } else { Some(PathBuf::from(expand_home(&v))) };
                }
                "--template" => c.template = value("--template")?,
                "--name" => c.name = Some(value("--name")?),
                "--cmd" => c.cmd_file = value("--cmd")?,
                "--log" => c.log_file = value("--log")?,
                "--ckdir" => c.ckdir = value("--ckdir")?,
                "--part" => c.part = value("--part")?,
                "--ngen" => c.ngen = value("--ngen")?.parse().map_err(|_| format!("--ngen needs a token count\n{USAGE}"))?,
                "--ctx" => c.ctx_chars = value("--ctx")?.parse().map_err(|_| format!("--ctx needs a character count\n{USAGE}"))?,
                "-h" | "--help" => return Err(USAGE.to_string()),
                other => return Err(format!("unknown option {other}\n{USAGE}")),
            }
        }
        Ok(c)
    }

    /// Resolves the tree and checks what a turn needs; the message says what to pass when something is missing.
    pub fn check(&mut self) -> Result<(), String> {
        let tree = fs::canonicalize(&self.tree).map_err(|e| format!("--tree {}: {e}", self.tree))?;
        self.tree = tree.to_string_lossy().into_owned();
        let bin = self.in_tree(&self.cortex);
        if !bin.is_file() {
            return Err(format!("{} not found: run ./build.sh in the tree, or pass --tree /mnt/Phi-Cortex-F32-MoE", bin.display()));
        }
        self.cortex = bin.to_string_lossy().into_owned();
        if let Some(ngc) = &self.orchestra {
            if !self.in_tree(ngc).is_file() {
                return Err(format!("orchestra file {ngc} not found in {}: pass --orchestra off or another count model", self.tree));
            }
            if !self.in_tree(&self.words).is_file() {
                return Err(format!("word list {} not found in {}: pass --words LIST", self.words, self.tree));
            }
        }
        if let Some(s) = &self.system {
            if !s.is_file() {
                return Err(format!("system prompt {} not found: pass --system FILE or --system off", s.display()));
            }
        }
        if !self.in_tree(&self.template).is_file() {
            return Err(format!("the frame {} not found in {}: pass --template FILE", self.template, self.tree));
        }
        if self.ngen == 0 || self.ngen > NGEN_MAX {
            return Err(format!("--ngen must be 1 to {NGEN_MAX}"));
        }
        Ok(())
    }

    pub fn in_tree(&self, p: &str) -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() { p.to_path_buf() } else { Path::new(&self.tree).join(p) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Run,
    Help,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    You,
    Cortex,
}

#[derive(Clone, Debug)]
pub struct Turn {
    pub who: Who,
    /// What is shown and re-fed; a Cortex turn's stream decoded (lossy only for a piece split inside a UTF-8 char).
    pub text: String,
    bytes: Vec<u8>,
    pub stats: Option<Stats>,
    pub done: bool,
    pub seed: u64,
    /// The token limit the reply was asked with.
    pub limit: u32,
    /// How the reply ended, once done.
    pub ended: Option<Ended>,
}

/// How a reply ended: at the story's end (`<|endoftext|>`), before the next speaker's label (the frame's stop), or
/// at the token limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Story,
    Label,
    Limit,
}

impl Turn {
    pub fn new(who: Who, text: String) -> Self {
        Self { who, text, bytes: Vec::new(), stats: None, done: who != Who::Cortex, seed: 0, limit: 0, ended: None }
    }

    pub fn append(&mut self, b: &[u8]) {
        self.bytes.extend_from_slice(b);
        self.text = String::from_utf8_lossy(&self.bytes).into_owned();
    }

    /// The reply as kept: the client's own trailing newline dropped, a partial stop label at the end cut (the client
    /// prints the pieces before the one completing the label), trailing whitespace trimmed; and how it ended.
    pub fn finish(&mut self, stop: Option<&str>, limit_hit: bool) {
        if self.bytes.last() == Some(&b'\n') {
            self.bytes.pop();
        }
        let mut text = String::from_utf8_lossy(&self.bytes).into_owned();
        let mut ended = if text.ends_with(EOT) { Ended::Story } else if limit_hit { Ended::Limit } else { Ended::Label };
        if let Some(stop) = stop.filter(|s| !s.is_empty()) {
            let trimmed = text.trim_end();
            if let Some(cut) = (1..=stop.len()).rev().find(|&k| stop.is_char_boundary(k) && trimmed.ends_with(&stop[..k])) {
                text.truncate(trimmed.len() - cut);
                ended = Ended::Label;
            }
        }
        self.text = text.trim_end().to_string();
        self.done = true;
        self.ended = Some(ended);
    }
}

pub struct App {
    pub cfg: Config,
    pub screen: Screen,
    pub turns: Vec<Turn>,
    pub input: Editor,
    /// Rows scrolled up from the bottom of the conversation; 0 follows the newest.
    pub back: usize,
    /// Conversation rows in the last frame (view.rs sets it): a page.
    pub page: usize,
    pub help_back: usize,
    pub sampling: Sampling,
    pub ngen: u32,
    pub ctx_chars: usize,
    pub orchestra_on: bool,
    pub system_on: bool,
    pub system_chars: usize,
    pub fixed_seed: Option<u64>,
    seed_next: u64,
    /// The frame's stop of the turn in flight.
    stop: Option<String>,
    rx: Option<Receiver<cortex::Event>>,
    pub in_flight: bool,
    pub turn_started: Instant,
    pub spin: usize,
    pub last_stats: Option<Stats>,
    pub live: Option<Live>,
    live_reader: Option<Reader>,
    pub live_error: Option<String>,
    live_polled: Option<Instant>,
    heart_seen: (u64, Instant),
    pub daemon_alive: bool,
    pub run: Run,
    /// The status row: the message, whether it is an error (kept until the next message; an info clears after 8 s),
    /// and when it was set.
    pub status: Option<(String, bool, Instant)>,
    quit: bool,
}

impl App {
    pub fn new(cfg: Config) -> Self {
        let now = Instant::now();
        let seed_base = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(1) * 1000;
        let system_chars = cfg.system.as_ref().and_then(|p| fs::read_to_string(p).ok()).map_or(0, |t| t.chars().count());
        let mut app = Self {
            orchestra_on: cfg.orchestra.is_some(),
            system_on: cfg.system.is_some(),
            system_chars,
            ngen: cfg.ngen,
            ctx_chars: cfg.ctx_chars,
            cfg,
            screen: Screen::Chat,
            turns: Vec::new(),
            input: Editor::new(),
            back: 0,
            page: 0,
            help_back: 0,
            sampling: Sampling::default(),
            fixed_seed: None,
            seed_next: seed_base,
            stop: None,
            rx: None,
            in_flight: false,
            turn_started: now,
            spin: 0,
            last_stats: None,
            live: None,
            live_reader: None,
            live_error: None,
            live_polled: None,
            heart_seen: (0, now.checked_sub(HEART_STALE).unwrap_or(now)),
            daemon_alive: false,
            run: Run::default(),
            status: None,
            quit: false,
        };
        if let Some(p) = app.cfg.system.clone() {
            app.info(format!("the system prompt {} ({} chars) prefixes every turn's prompt", p.display(), app.system_chars));
        }
        app
    }

    pub fn done(&self) -> bool {
        self.quit && !self.in_flight
    }

    pub fn info(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), false, Instant::now()));
    }

    pub fn error(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), true, Instant::now()));
    }

    // ---- keys ----

    pub fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        match k.code {
            KeyCode::Char('q') | KeyCode::Char('c') if ctrl => return self.request_quit(),
            KeyCode::F(1) => {
                self.screen = if self.screen == Screen::Help { Screen::Chat } else { Screen::Help };
                return;
            }
            KeyCode::F(2) => {
                self.screen = if self.screen == Screen::Run { Screen::Chat } else { Screen::Run };
                return;
            }
            _ => {}
        }
        match self.screen {
            Screen::Chat => self.chat_key(k, ctrl, alt),
            Screen::Run => self.run_key(k),
            Screen::Help => self.help_key(k),
        }
    }

    fn chat_key(&mut self, k: KeyEvent, ctrl: bool, alt: bool) {
        match k.code {
            KeyCode::Enter => self.say(),
            KeyCode::Esc => self.input.set_text(""),
            KeyCode::F(3) => self.toggle_orchestra(),
            KeyCode::F(4) => self.toggle_greedy(),
            KeyCode::PageUp => self.back += self.page.max(1),
            KeyCode::PageDown => self.back = self.back.saturating_sub(self.page.max(1)),
            KeyCode::Up => self.back += 1,
            KeyCode::Down => self.back = self.back.saturating_sub(1),
            KeyCode::Left if ctrl || alt => self.input.word_left(),
            KeyCode::Right if ctrl || alt => self.input.word_right(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Backspace if ctrl || alt => self.input.delete_word_back(),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('k') if ctrl => self.input.kill_to_end(),
            // Ctrl+W, and Ctrl+Backspace where the terminal sends ^H for it
            KeyCode::Char('w') | KeyCode::Char('h') if ctrl => self.input.delete_word_back(),
            KeyCode::Char('z') if ctrl => {
                if !self.input.undo() {
                    self.info("nothing to undo");
                }
            }
            KeyCode::Char('b') if alt => self.input.word_left(),
            KeyCode::Char('f') if alt => self.input.word_right(),
            KeyCode::Char(c) if !ctrl && !alt => self.input.insert_char(c),
            _ => {}
        }
    }

    fn run_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc => self.screen = Screen::Chat,
            KeyCode::Char('p') => self.coach("pause".into()),
            KeyCode::Char('r') => self.coach("resume".into()),
            KeyCode::Char('s') => self.coach("save".into()),
            KeyCode::Char('m') => self.coach("sample".into()),
            KeyCode::Char('+') | KeyCode::Char('=') => self.rescale(2.0),
            KeyCode::Char('-') => self.rescale(0.5),
            _ => {}
        }
    }

    fn help_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::Esc => self.screen = Screen::Chat,
            KeyCode::Up | KeyCode::PageUp => self.help_back = self.help_back.saturating_sub(1),
            KeyCode::Down | KeyCode::PageDown => self.help_back += 1,
            _ => {}
        }
    }

    /// A bracketed paste, into the line being typed.
    pub fn paste(&mut self, s: &str) {
        if self.screen == Screen::Chat {
            self.input.insert_str(s);
        }
    }

    fn request_quit(&mut self) {
        self.quit = true;
        if self.in_flight {
            self.info("quitting after the reply in flight (the client is never killed: its QUIT closes the daemon's session)");
        }
    }

    fn toggle_orchestra(&mut self) {
        if self.cfg.orchestra.is_none() {
            return self.error("no orchestra count model was given (--orchestra NGC): the plain model answers");
        }
        self.orchestra_on = !self.orchestra_on;
        let m = if self.orchestra_on { "orchestra on: the host n-gram mix, word filter and repeat block" } else { "orchestra off: the plain model's logits" };
        self.info(m);
    }

    fn toggle_greedy(&mut self) {
        self.sampling.greedy = !self.sampling.greedy;
        let m = if self.sampling.greedy { "greedy on: the argmax token every step" } else { "greedy off: sampling" };
        self.info(m);
    }

    // ---- turns ----

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
        if self.in_flight {
            return self.error("a reply is streaming; press enter again when it has ended");
        }
        self.input.set_text("");
        self.turns.push(Turn::new(Who::You, line));
        self.generate();
    }

    fn generate(&mut self) {
        let Rendered { prompt, stop } = match self.prompt() {
            Ok(r) => r,
            Err(e) => return self.error(e),
        };
        self.stop = stop.clone();
        let seed = match self.fixed_seed {
            Some(s) => s,
            None => {
                self.seed_next += 1;
                self.seed_next
            }
        };
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.in_flight = true;
        self.turn_started = Instant::now();
        self.back = 0;
        let mut turn = Turn::new(Who::Cortex, String::new());
        turn.seed = seed;
        turn.limit = self.ngen;
        self.turns.push(turn);
        let orchestra = if self.orchestra_on { self.cfg.orchestra.clone().map(|n| (n, self.cfg.words.clone())) } else { None };
        cortex::start(
            Request { tree: self.cfg.tree.clone(), cortex: self.cfg.cortex.clone(), orchestra, prompt, stop, ngen: self.ngen, seed, sampling: self.sampling.clone() },
            tx,
        );
    }

    /// The system prompt as it is now (the file is read at every turn: a hook restamps its date line).
    pub fn system_text(&self) -> Option<String> {
        if !self.system_on {
            return None;
        }
        let text = fs::read_to_string(self.cfg.system.as_ref()?).ok()?;
        if text.trim().is_empty() { None } else { Some(text) }
    }

    /// The prompt of a turn: the frame rendered with the system message (the system prompt file as it is now), the
    /// dialogue and the model's label last; the oldest exchanges are dropped until the dialogue's characters fit
    /// `ctx_chars`.
    pub fn prompt(&mut self) -> Result<Rendered, String> {
        let path = self.cfg.in_tree(&self.cfg.template);
        let src = fs::read_to_string(&path).map_err(|e| format!("the frame {}: {e}", path.display()))?;
        let mut head = Vec::new();
        if let Some(sys) = self.system_text() {
            self.system_chars = sys.chars().count();
            head.push(Message { role: "system", content: sys });
        }
        let dialogue: Vec<Message> = self
            .turns
            .iter()
            .map(|t| Message { role: if t.who == Who::You { "user" } else { "assistant" }, content: t.text.clone() })
            .collect();
        let name = self.cfg.name.clone();
        let base = frame::render(&src, &head, name.as_deref())?.prompt.chars().count();
        let mut skip = 0;
        loop {
            let all: Vec<Message> = head.iter().chain(&dialogue[skip..]).cloned().collect();
            let r = frame::render(&src, &all, name.as_deref())?;
            if r.prompt.chars().count().saturating_sub(base) <= self.ctx_chars || dialogue.len() - skip <= 1 {
                return Ok(r);
            }
            skip += 1;
            while skip < dialogue.len() && dialogue[skip].role == "assistant" {
                skip += 1;
            }
        }
    }

    /// The turn in flight's events; whether anything shown changed.
    fn pump(&mut self) -> bool {
        let Some(rx) = self.rx.take() else { return false };
        let mut changed = false;
        let mut done = None;
        while let Ok(ev) = rx.try_recv() {
            changed = true;
            match ev {
                cortex::Event::Bytes(b) => {
                    if let Some(t) = self.turns.last_mut() {
                        t.append(&b);
                    }
                }
                cortex::Event::Done { ok, stats, stderr } => {
                    done = Some((ok, stats, stderr));
                    break;
                }
            }
        }
        let Some((ok, stats, stderr)) = done else {
            self.rx = Some(rx);
            return changed;
        };
        self.in_flight = false;
        let stop = self.stop.take();
        if let Some(t) = self.turns.last_mut() {
            let limit_hit = stats.as_ref().is_some_and(|s| s.generated >= t.limit as u64);
            t.finish(stop.as_deref(), limit_hit);
            t.stats = stats.clone();
        }
        if ok {
            self.last_stats = stats;
            return true;
        }
        if self.turns.last().is_some_and(|t| t.who == Who::Cortex && t.text.is_empty()) {
            self.turns.pop();
        }
        let why = stderr
            .iter()
            .rev()
            .find(|l| l.starts_with("cortex:") && !l.contains("tok/s"))
            .cloned()
            .unwrap_or_else(|| "the client failed without a message".into());
        self.error(format!("no reply: {why}"));
        true
    }

    fn retry(&mut self) {
        if self.in_flight {
            return self.error("a reply is streaming; /retry when it has ended");
        }
        let Some(last) = self.turns.last() else { return self.error("nothing to retry yet") };
        if last.who == Who::Cortex {
            self.turns.pop();
        }
        if self.fixed_seed.is_some() {
            self.info("the seed is fixed: the retry repeats the same reply (/seed auto for a new one)");
        }
        self.generate();
    }

    // ---- commands ----

    fn command(&mut self, cmd: &str) {
        let mut it = cmd.split_whitespace();
        let name = it.next().unwrap_or("");
        let arg = it.next();
        match name {
            "help" | "?" => self.screen = Screen::Help,
            "run" | "status" => self.screen = Screen::Run,
            "quit" | "exit" | "q" => self.request_quit(),
            "clear" | "new" => {
                self.turns.clear();
                self.back = 0;
                self.info("a new story: the transcript is empty");
            }
            "retry" | "again" => self.retry(),
            "n" | "tokens" => self.set_u32(arg, 1, NGEN_MAX, "tokens per reply", |a, v| a.ngen = v),
            "ctx" => self.set_usize(arg, 100, 100_000, "prompt characters", |a, v| a.ctx_chars = v),
            "temp" => self.set_f32(arg, 0.05, 5.0, "temperature", |a, v| a.sampling.temp = v),
            "topk" => self.set_u32(arg, 1, 4096, "top-k", |a, v| a.sampling.topk = v),
            "topp" => self.set_f32(arg, 0.01, 1.0, "top-p", |a, v| a.sampling.topp = v),
            "minp" => self.set_f32(arg, 0.0, 1.0, "min-p", |a, v| a.sampling.minp = v),
            "rp" => self.set_f32(arg, 1.0, 3.0, "repeat penalty", |a, v| a.sampling.rp = v),
            "greedy" => match on_off(arg) {
                Some(v) if v == self.sampling.greedy => self.info(format!("greedy is {}", if v { "on" } else { "off" })),
                _ => self.toggle_greedy(),
            },
            "orchestra" => match on_off(arg) {
                Some(v) if v == self.orchestra_on => self.info(format!("the orchestra is {}", if v { "on" } else { "off" })),
                _ => self.toggle_orchestra(),
            },
            "system" => self.system_command(arg),
            "seed" => match arg {
                None | Some("auto") => {
                    self.fixed_seed = None;
                    self.info("seed: a fresh one each turn");
                }
                Some(v) => match v.parse::<u64>() {
                    Ok(s) => {
                        self.fixed_seed = Some(s);
                        self.info(format!("seed fixed at {s}: the same prompt gives the same reply"));
                    }
                    Err(_) => self.error("usage: /seed N or /seed auto"),
                },
            },
            "pause" | "resume" | "save" | "sample" => self.coach(name.to_string()),
            "stop-run" => self.coach("stop".to_string()),
            "scale" | "lr" | "every" => match arg {
                Some(v) if v.parse::<f64>().is_ok() => self.coach(format!("{name} {v}")),
                _ => self.error(format!("usage: /{name} VALUE")),
            },
            "" => {}
            other => self.error(format!("unknown command /{other}; f1 lists them")),
        }
    }

    fn system_command(&mut self, arg: Option<&str>) {
        match arg {
            None => {
                if self.cfg.system.is_none() {
                    return self.error("no system prompt file: /system FILE sets one");
                }
                self.system_on = !self.system_on;
                let m = if self.system_on { "system prompt on" } else { "system prompt off" };
                self.info(m);
            }
            Some("on") | Some("off") => {
                if self.cfg.system.is_none() {
                    return self.error("no system prompt file: /system FILE sets one");
                }
                self.system_on = arg == Some("on");
                let m = if self.system_on { "system prompt on" } else { "system prompt off" };
                self.info(m);
            }
            Some(file) => {
                let p = PathBuf::from(expand_home(file));
                match fs::read_to_string(&p) {
                    Ok(t) => {
                        self.system_chars = t.chars().count();
                        self.cfg.system = Some(p.clone());
                        self.system_on = true;
                        self.info(format!("system prompt {} ({} chars) from the next turn", p.display(), self.system_chars));
                    }
                    Err(e) => self.error(format!("{}: {e}", p.display())),
                }
            }
        }
    }

    fn set_f32(&mut self, arg: Option<&str>, lo: f32, hi: f32, what: &str, apply: impl FnOnce(&mut Self, f32)) {
        match arg.and_then(|v| v.parse::<f32>().ok()) {
            Some(v) if (lo..=hi).contains(&v) => {
                apply(self, v);
                self.info(format!("{what} = {v} from the next turn"));
            }
            _ => self.error(format!("{what}: a value from {lo} to {hi}")),
        }
    }

    fn set_u32(&mut self, arg: Option<&str>, lo: u32, hi: u32, what: &str, apply: impl FnOnce(&mut Self, u32)) {
        match arg.and_then(|v| v.parse::<u32>().ok()) {
            Some(v) if (lo..=hi).contains(&v) => {
                apply(self, v);
                self.info(format!("{what} = {v} from the next turn"));
            }
            _ => self.error(format!("{what}: a count from {lo} to {hi}")),
        }
    }

    fn set_usize(&mut self, arg: Option<&str>, lo: usize, hi: usize, what: &str, apply: impl FnOnce(&mut Self, usize)) {
        match arg.and_then(|v| v.parse::<usize>().ok()) {
            Some(v) if (lo..=hi).contains(&v) => {
                apply(self, v);
                self.info(format!("{what} = {v} from the next turn"));
            }
            _ => self.error(format!("{what}: a count from {lo} to {hi}")),
        }
    }

    /// The scale knob multiplied, from the value the card reports.
    fn rescale(&mut self, by: f64) {
        let Some(scale) = self.live.as_ref().filter(|l| l.run).map(|l| l.scale) else {
            return self.error("no live run is feeding the daemon");
        };
        self.coach(format!("scale {}", scale * by));
    }

    /// A line for the coach's command file; refused when no run is feeding (the coach reads only lines appended
    /// after it started, so a line written now would be lost).
    fn coach(&mut self, line: String) {
        if !self.live.as_ref().is_some_and(|l| l.run) {
            return self.error(format!("not sent: no live run is feeding the daemon (start build/host/coach ... --cmd {} first)", self.cfg.cmd_file));
        }
        let path = self.cfg.in_tree(&self.cfg.cmd_file);
        match OpenOptions::new().append(true).create(true).open(&path).and_then(|mut f| writeln!(f, "{line}")) {
            Ok(()) => self.info(format!("sent to the coach: {line} (acts at the next chunk or report)")),
            Err(e) => self.error(format!("cannot append to {}: {e}", path.display())),
        }
    }

    // ---- what the loop does between keys ----

    /// The turn in flight, the live words, the run's files, the status row's age; whether a redraw is due.
    pub fn tick(&mut self) -> bool {
        let mut changed = self.pump();
        if self.in_flight {
            self.spin = self.spin.wrapping_add(1);
            changed = true;
        }
        changed |= self.poll_live();
        let log = self.cfg.in_tree(&self.cfg.log_file);
        let ckdir = self.cfg.in_tree(&self.cfg.ckdir);
        changed |= self.run.poll(&log, &ckdir);
        if self.status.as_ref().is_some_and(|(_, err, t)| !err && t.elapsed() > INFO_FOR) {
            self.status = None;
            changed = true;
        }
        changed
    }

    fn poll_live(&mut self) -> bool {
        if self.live_polled.is_some_and(|t| t.elapsed() < LIVE_POLL) {
            return false;
        }
        self.live_polled = Some(Instant::now());
        if self.live_reader.is_none() {
            match Reader::open(&self.cfg.part) {
                Ok(r) => self.live_reader = Some(r),
                Err(e) => {
                    let msg = format!("{}: {e}", self.cfg.part);
                    let changed = self.live_error.as_deref() != Some(&msg);
                    self.live_error = Some(msg);
                    return changed;
                }
            }
        }
        let Some(r) = &self.live_reader else { return false };
        match r.read() {
            Ok(l) => {
                if l.heart != self.heart_seen.0 {
                    self.heart_seen = (l.heart, Instant::now());
                }
                let alive = self.heart_seen.1.elapsed() < HEART_STALE;
                let changed = self.live.as_ref() != Some(&l) || alive != self.daemon_alive || self.live_error.is_some();
                self.daemon_alive = alive;
                self.live = Some(l);
                self.live_error = None;
                changed
            }
            Err(e) => {
                self.live_error = Some(format!("{}: {e}", self.cfg.part));
                self.live = None;
                self.daemon_alive = false;
                true
            }
        }
    }
}

fn on_off(arg: Option<&str>) -> Option<bool> {
    match arg {
        Some("on") => Some(true),
        Some("off") => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn test_config() -> Config {
        Config {
            tree: ".".into(),
            cortex: "c".into(),
            orchestra: None,
            words: "w".into(),
            system: None,
            template: "cortex.jinja".into(),
            name: None,
            cmd_file: "cmd".into(),
            log_file: "log".into(),
            ckdir: "ck".into(),
            part: "p".into(),
            ngen: 8,
            ctx_chars: 4000,
        }
    }

    #[test]
    fn a_reply_ends_at_the_story_the_label_or_the_limit() {
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" The sun was warm.\nHum");
        c.finish(Some("\nHuman:"), false);
        assert_eq!((c.text.as_str(), c.ended), (" The sun was warm.", Some(Ended::Label)));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" The end.");
        c.append(EOT.as_bytes());
        c.append(b"\n");
        c.finish(Some("\nHuman:"), false);
        assert_eq!((c.text.as_str(), c.ended), (" The end.<|endoftext|>", Some(Ended::Story)));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" and then\n");
        c.finish(None, true);
        assert_eq!((c.text.as_str(), c.ended), (" and then", Some(Ended::Limit)));
    }

    #[test]
    fn the_prompt_is_the_frame_over_the_system_message_and_the_dialogue() {
        let dir = std::env::temp_dir().join(format!("cortex-tui-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tpl = dir.join("frame.jinja");
        fs::write(&tpl, "{% set stop = \"\\nU:\" %}{% for m in messages %}{{ m.role }}: {{ m.content | trim }}\n{% endfor %}A:").unwrap();
        let sys = dir.join("system.md");
        fs::write(&sys, "Be brief.\n").unwrap();
        let mut cfg = test_config();
        cfg.template = tpl.to_string_lossy().into_owned();
        let mut a = App::new(cfg);
        a.turns.push(Turn::new(Who::You, "hi".into()));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" Hello there.\n");
        c.finish(None, true);
        a.turns.push(c);
        a.turns.push(Turn::new(Who::You, "and then?".into()));
        let r = a.prompt().unwrap();
        assert_eq!(r.prompt, "user: hi\nassistant: Hello there.\nuser: and then?\nA:");
        assert_eq!(r.stop.as_deref(), Some("\nU:"));
        a.cfg.system = Some(sys);
        a.system_on = true;
        assert_eq!(a.prompt().unwrap().prompt, "system: Be brief.\nuser: hi\nassistant: Hello there.\nuser: and then?\nA:");
        assert_eq!(a.system_chars, 10);
        a.ctx_chars = 20; // the oldest exchange is dropped: the dialogue is then "user: and then?\n"
        assert_eq!(a.prompt().unwrap().prompt, "system: Be brief.\nuser: and then?\nA:");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn commands_set_the_knobs_and_refuse_bad_values() {
        let mut a = App::new(test_config());
        a.command("n 64");
        assert_eq!(a.ngen, 64);
        a.command("n 0");
        assert_eq!(a.ngen, 64);
        assert!(a.status.as_ref().is_some_and(|s| s.1));
        a.command("temp 0.8");
        assert_eq!(a.sampling.temp, 0.8);
        a.command("seed 42");
        assert_eq!(a.fixed_seed, Some(42));
        a.command("seed auto");
        assert_eq!(a.fixed_seed, None);
        a.command("pause");
        assert!(a.status.as_ref().is_some_and(|s| s.1 && s.0.starts_with("not sent")));
    }
}

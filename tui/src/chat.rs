//! The chat core shared by the harness (src/main.rs, app.rs) and the monitor (src/bin/cortex-watch.rs): the
//! configuration both take, the turns, the prompt of a turn through the frame, the turn in flight, the settings
//! commands, the lines for the coach's command file, and the line editor's keys.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::KeyCode;

use crate::cortex::{self, Request, Sampling, Stats};
use crate::editor::Editor;
use crate::frame::{self, Message, Rendered};

/// The story separator of the BPE (tools/bpe.c): one token; its literal in a prompt tokenises back to it.
pub const EOT: &str = "<|endoftext|>";
/// The think block's markers, one token each in the chat vocabulary: a reply is `<|think|>REASONING<|/think|>ANSWER`.
pub const THINK_OPEN: &str = "<|think|>";
pub const THINK_CLOSE: &str = "<|/think|>";
pub const NGEN_MAX: u32 = 2048;

/// A reply split into its thinking and its answer: the text between the think markers (to the end when the block
/// is still open) and what follows the close; a reply without the markers is all answer.
pub fn split_think(text: &str) -> (&str, &str) {
    let Some(open) = text.find(THINK_OPEN) else { return ("", text) };
    let body = &text[open + THINK_OPEN.len()..];
    match body.find(THINK_CLOSE) {
        Some(close) => (body[..close].trim(), body[close + THINK_CLOSE.len()..].trim_start()),
        None => (body.trim_start(), ""),
    }
}

const OPTIONS: &str = "  --tree DIR        the Cortex tree the paths are relative to (default: the current directory)
  --cortex PATH     the host client (build/host/cortex)
  --orchestra NGC   the orchestrator's n-gram count model (data/stories-10m.ngc), or off for the plain model
  --words LIST      the orchestrator's word list (data/wordlist-stories.txt)
  --vocab FILE      the BPE vocabulary the client encodes and decodes with (its own default: data/bpe/vocab4096.txt)
  --ngram FILE      the n-gram memory's tables the client reads each step's rows from (the live run's file in
                    /dev/shm, or a saved tables-N.f32), for a model trained with the memory; none: the plain model
  --system FILE     the system message of every turn (~/CLAUDE.md when it exists), or off
  --template FILE   the frame of every turn, a Jinja template (tui/cortex.jinja): the identity, the labels, the stop
  --name NAME       the identity's name in the frame (the template's own default when not given)
  --cmd FILE        the coach's command file, where the run commands go (results/live.cmd)
  --log FILE        the coach's log, read for the reports (results/live.log)
  --ckdir DIR       where the coach puts the checkpoints (data/live)
  --part FILE       the shared host-memory region (/dev/shm/phi-hostmem)
  --ngen N          tokens per reply (128, at most 2048)
  --ctx CHARS       dialogue characters kept in a turn's prompt; the oldest exchanges are dropped beyond it (4000)";

pub fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--tree DIR] [--cortex PATH] [--orchestra NGC|off] [--words LIST] [--vocab FILE] [--ngram FILE]
                  [--system FILE|off] [--template FILE] [--name NAME] [--cmd FILE] [--log FILE] [--ckdir DIR]
                  [--part FILE] [--ngen N] [--ctx CHARS]\n{OPTIONS}"
    )
}

/// Command-line configuration; every value defaults to the tree's layout.
#[derive(Clone, Debug)]
pub struct Config {
    pub tree: String,
    pub cortex: String,
    pub orchestra: Option<String>,
    /// `--orchestra` was given: the count model is meant for this model (the default one belongs to the stories'
    /// vocabulary and is switched off for a model with its own).
    pub orchestra_given: bool,
    pub words: String,
    /// The client's BPE vocabulary, when not its default (a model trained with its own tokenizer).
    pub vocab: Option<String>,
    /// The n-gram memory's tables file the client reads a step's rows from, for a model trained with the memory.
    pub ngram: Option<String>,
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
    pub fn from_args<I: Iterator<Item = String>>(program: &str, mut args: I) -> Result<Self, String> {
        let home_claude = std::env::var_os("HOME").map(|h| PathBuf::from(h).join("CLAUDE.md")).filter(|p| p.is_file());
        let mut c = Config {
            tree: ".".into(),
            cortex: "build/host/cortex".into(),
            orchestra: Some("data/stories-10m.ngc".into()),
            orchestra_given: false,
            words: "data/wordlist-stories.txt".into(),
            vocab: None,
            ngram: None,
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
            let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value\n{}", usage(program)));
            match a.as_str() {
                "--tree" => c.tree = value("--tree")?,
                "--cortex" => c.cortex = value("--cortex")?,
                "--orchestra" => {
                    let v = value("--orchestra")?;
                    c.orchestra = if v == "off" { None } else { Some(v) };
                    c.orchestra_given = true;
                }
                "--words" => c.words = value("--words")?,
                "--vocab" => c.vocab = Some(value("--vocab")?),
                "--ngram" => c.ngram = Some(value("--ngram")?),
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
                "--ngen" => c.ngen = value("--ngen")?.parse().map_err(|_| format!("--ngen needs a token count\n{}", usage(program)))?,
                "--ctx" => c.ctx_chars = value("--ctx")?.parse().map_err(|_| format!("--ctx needs a character count\n{}", usage(program)))?,
                "-h" | "--help" => return Err(usage(program)),
                other => return Err(format!("unknown option {other}\n{}", usage(program))),
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
        if self.vocab.is_some() && !self.orchestra_given {
            // the default count model and word list are the stories' vocabulary's: on other ids they mix and filter
            // the wrong tokens
            self.orchestra = None;
        }
        if let Some(ngc) = &self.orchestra {
            if !self.in_tree(ngc).is_file() {
                return Err(format!("orchestra file {ngc} not found in {}: pass --orchestra off or another count model", self.tree));
            }
            if !self.in_tree(&self.words).is_file() {
                return Err(format!("word list {} not found in {}: pass --words LIST", self.words, self.tree));
            }
        }
        if let Some(v) = &self.vocab {
            if !self.in_tree(v).is_file() {
                return Err(format!("vocabulary {v} not found in {}: pass --vocab FILE", self.tree));
            }
        }
        if let Some(n) = &self.ngram {
            if !self.in_tree(n).is_file() {
                return Err(format!("n-gram tables {n} not found in {}: pass --ngram FILE (the live run's file in /dev/shm) or leave it out", self.tree));
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
pub enum Who {
    You,
    Cortex,
}

/// How a reply ended: at the story's end (`<|endoftext|>`), before one of the frame's stops (the end marker, or a
/// role marker the model began the next turn with; the string is the stop, empty when the client did not say), or
/// at the token limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ended {
    Story,
    Label(String),
    Limit,
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

impl Turn {
    pub fn new(who: Who, text: String) -> Self {
        Self { who, text, bytes: Vec::new(), stats: None, done: who != Who::Cortex, seed: 0, limit: 0, ended: None }
    }

    pub fn append(&mut self, b: &[u8]) {
        self.bytes.extend_from_slice(b);
        self.text = String::from_utf8_lossy(&self.bytes).into_owned();
    }

    /// The reply as kept: the client's own trailing newline dropped, a partial stop label at the end cut (the client
    /// prints the pieces before the one completing the label), trailing whitespace trimmed; and how it ended: at the
    /// story's end, at the token limit, or at one of the frame's `stops` (`stopped` as the client reported it, else
    /// the stop whose partial label ended the text, else unnamed).
    pub fn finish(&mut self, stops: &[String], stopped: Option<&str>, limit_hit: bool) {
        if self.bytes.last() == Some(&b'\n') {
            self.bytes.pop();
        }
        let mut text = String::from_utf8_lossy(&self.bytes).into_owned();
        let mut ended = if text.ends_with(EOT) {
            Ended::Story
        } else if limit_hit {
            Ended::Limit
        } else {
            Ended::Label(stopped.unwrap_or_default().to_string())
        };
        let cut = {
            let trimmed = text.trim_end();
            stops
                .iter()
                .filter(|s| !s.is_empty())
                .filter_map(|s| (1..=s.len()).rev().find(|&k| s.is_char_boundary(k) && trimmed.ends_with(&s[..k])).map(|k| (k, s)))
                .max_by_key(|&(k, _)| k)
                .map(|(k, s)| (trimmed.len() - k, s.clone()))
        };
        if let Some((len, stop)) = cut {
            text.truncate(len);
            ended = Ended::Label(stopped.map_or(stop, String::from));
        }
        self.text = text.trim_end().to_string();
        self.done = true;
        self.ended = Some(ended);
    }
}

/// The conversation with the live weights and the settings of its turns.
pub struct Chat {
    pub turns: Vec<Turn>,
    pub sampling: Sampling,
    pub ngen: u32,
    pub ctx_chars: usize,
    pub orchestra_on: bool,
    pub system_on: bool,
    pub system_chars: usize,
    /// The model opens its think block itself (true), or is given an empty one and answers at once (false).
    pub think: bool,
    pub fixed_seed: Option<u64>,
    seed_next: u64,
    /// The frame's stops of the turn in flight.
    stop: Vec<String>,
    rx: Option<Receiver<cortex::Event>>,
    pub in_flight: bool,
    pub turn_started: Instant,
    pub last_stats: Option<Stats>,
}

impl Chat {
    pub fn new(cfg: &Config) -> Self {
        let seed_base = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(1) * 1000;
        let system_chars = cfg.system.as_ref().and_then(|p| fs::read_to_string(p).ok()).map_or(0, |t| t.chars().count());
        Self {
            turns: Vec::new(),
            sampling: Sampling::default(),
            ngen: cfg.ngen,
            ctx_chars: cfg.ctx_chars,
            orchestra_on: cfg.orchestra.is_some(),
            system_on: cfg.system.is_some(),
            system_chars,
            think: true,
            fixed_seed: None,
            seed_next: seed_base,
            stop: Vec::new(),
            rx: None,
            in_flight: false,
            turn_started: Instant::now(),
            last_stats: None,
        }
    }

    /// Your line as a turn, the reply started.
    pub fn say(&mut self, cfg: &Config, line: String) -> Result<(), String> {
        if self.in_flight {
            return Err("a reply is streaming; press enter again when it has ended".into());
        }
        self.turns.push(Turn::new(Who::You, line));
        self.generate(cfg)
    }

    fn generate(&mut self, cfg: &Config) -> Result<(), String> {
        let Rendered { prompt, stop } = self.prompt(cfg)?;
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
        let mut turn = Turn::new(Who::Cortex, String::new());
        turn.seed = seed;
        turn.limit = self.ngen;
        self.turns.push(turn);
        let orchestra = if self.orchestra_on { cfg.orchestra.clone().map(|n| (n, cfg.words.clone())) } else { None };
        cortex::start(
            Request { tree: cfg.tree.clone(), cortex: cfg.cortex.clone(), orchestra, vocab: cfg.vocab.clone(), ngram: cfg.ngram.clone(), prompt, stop, ngen: self.ngen, seed, sampling: self.sampling.clone() },
            tx,
        );
        Ok(())
    }

    /// The last reply again (a fresh seed unless one is fixed); the note to show, if any.
    pub fn retry(&mut self, cfg: &Config) -> Result<Option<String>, String> {
        if self.in_flight {
            return Err("a reply is streaming; /retry when it has ended".into());
        }
        let Some(last) = self.turns.last() else { return Err("nothing to retry yet".into()) };
        if last.who == Who::Cortex {
            self.turns.pop();
        }
        let note = self.fixed_seed.map(|_| "the seed is fixed: the retry repeats the same reply (/seed auto for a new one)".to_string());
        self.generate(cfg)?;
        Ok(note)
    }

    /// The system prompt as it is now (the file is read at every turn: a hook restamps its date line).
    pub fn system_text(&self, cfg: &Config) -> Option<String> {
        if !self.system_on {
            return None;
        }
        let text = fs::read_to_string(cfg.system.as_ref()?).ok()?;
        if text.trim().is_empty() { None } else { Some(text) }
    }

    /// The prompt of a turn: the frame rendered with the system message (the system prompt file as it is now), the
    /// dialogue (earlier replies without their think blocks) and the assistant marker last; the oldest exchanges are
    /// dropped until the dialogue's characters fit `ctx_chars`.
    pub fn prompt(&mut self, cfg: &Config) -> Result<Rendered, String> {
        let path = cfg.in_tree(&cfg.template);
        let src = fs::read_to_string(&path).map_err(|e| format!("the frame {}: {e}", path.display()))?;
        let mut head = Vec::new();
        if let Some(sys) = self.system_text(cfg) {
            self.system_chars = sys.chars().count();
            head.push(Message { role: "system", content: sys });
        }
        let dialogue: Vec<Message> = self
            .turns
            .iter()
            .map(|t| match t.who {
                Who::You => Message { role: "user", content: t.text.clone() },
                Who::Cortex => Message { role: "assistant", content: split_think(&t.text).1.to_string() },
            })
            .collect();
        let name = cfg.name.clone();
        let base = frame::render(&src, &head, name.as_deref(), self.think)?.prompt.chars().count();
        let mut skip = 0;
        loop {
            let all: Vec<Message> = head.iter().chain(&dialogue[skip..]).cloned().collect();
            let r = frame::render(&src, &all, name.as_deref(), self.think)?;
            if r.prompt.chars().count().saturating_sub(base) <= self.ctx_chars || dialogue.len() - skip <= 1 {
                return Ok(r);
            }
            skip += 1;
            while skip < dialogue.len() && dialogue[skip].role == "assistant" {
                skip += 1;
            }
        }
    }

    /// The turn in flight's events: whether anything shown changed, and the client's error when it failed.
    pub fn pump(&mut self) -> (bool, Option<String>) {
        let Some(rx) = self.rx.take() else { return (false, None) };
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
            return (changed, None);
        };
        self.in_flight = false;
        let stops = std::mem::take(&mut self.stop);
        if let Some(t) = self.turns.last_mut() {
            let limit_hit = stats.as_ref().is_some_and(|s| s.generated >= t.limit as u64);
            let stopped = stderr.iter().find_map(|l| l.strip_prefix("cortex: stopped at "));
            t.finish(&stops, stopped, limit_hit);
            t.stats = stats.clone();
        }
        if ok {
            self.last_stats = stats;
            return (true, None);
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
        (true, Some(format!("no reply: {why}")))
    }

    /// A settings command (`/n`, `/ctx`, `/temp`, `/topk`, `/topp`, `/minp`, `/rp`, `/greedy`, `/orchestra`,
    /// `/system`, `/seed`, `/clear`, `/retry`): the message to show, or None when `name` is not one of them.
    pub fn setting(&mut self, cfg: &mut Config, name: &str, arg: Option<&str>) -> Option<Result<String, String>> {
        Some(match name {
            "n" | "tokens" => set_u32(arg, 1, NGEN_MAX, "tokens per reply").map(|v| { self.ngen = v; format!("tokens per reply = {v} from the next turn") }),
            "ctx" => set_u32(arg, 100, 100_000, "prompt characters").map(|v| { self.ctx_chars = v as usize; format!("prompt characters = {v} from the next turn") }),
            "temp" => set_f32(arg, 0.05, 5.0, "temperature").map(|v| { self.sampling.temp = v; format!("temperature = {v} from the next turn") }),
            "topk" => set_u32(arg, 1, 4096, "top-k").map(|v| { self.sampling.topk = v; format!("top-k = {v} from the next turn") }),
            "topp" => set_f32(arg, 0.01, 1.0, "top-p").map(|v| { self.sampling.topp = v; format!("top-p = {v} from the next turn") }),
            "minp" => set_f32(arg, 0.0, 1.0, "min-p").map(|v| { self.sampling.minp = v; format!("min-p = {v} from the next turn") }),
            "rp" => set_f32(arg, 1.0, 3.0, "repeat penalty").map(|v| { self.sampling.rp = v; format!("repeat penalty = {v} from the next turn") }),
            "greedy" => {
                self.sampling.greedy = on_off(arg).unwrap_or(!self.sampling.greedy);
                Ok(if self.sampling.greedy { "greedy on: the argmax token every step" } else { "greedy off: sampling" }.to_string())
            }
            "think" => {
                self.think = on_off(arg).unwrap_or(!self.think);
                Ok(if self.think { "think on: the model reasons in its think block before the answer" } else { "think off: an empty think block is given, the model answers at once" }.to_string())
            }
            "orchestra" => {
                if cfg.orchestra.is_none() {
                    Err("no orchestra count model was given (--orchestra NGC): the plain model answers".into())
                } else {
                    self.orchestra_on = on_off(arg).unwrap_or(!self.orchestra_on);
                    Ok(if self.orchestra_on { "orchestra on: the host n-gram mix, word filter and repeat block" } else { "orchestra off: the plain model's logits" }.to_string())
                }
            }
            "system" => self.system_command(cfg, arg),
            "seed" => match arg {
                None | Some("auto") => {
                    self.fixed_seed = None;
                    Ok("seed: a fresh one each turn".into())
                }
                Some(v) => match v.parse::<u64>() {
                    Ok(s) => {
                        self.fixed_seed = Some(s);
                        Ok(format!("seed fixed at {s}: the same prompt gives the same reply"))
                    }
                    Err(_) => Err("usage: /seed N or /seed auto".into()),
                },
            },
            "clear" | "new" => {
                self.turns.clear();
                Ok("a new story: the transcript is empty".into())
            }
            "retry" | "again" => self.retry(cfg).map(|note| note.unwrap_or_else(|| "the last reply again".into())),
            _ => return None,
        })
    }

    fn system_command(&mut self, cfg: &mut Config, arg: Option<&str>) -> Result<String, String> {
        match arg {
            None | Some("on") | Some("off") => {
                if cfg.system.is_none() {
                    return Err("no system prompt file: /system FILE sets one".into());
                }
                self.system_on = on_off(arg).unwrap_or(!self.system_on);
                Ok(if self.system_on { "system prompt on" } else { "system prompt off" }.to_string())
            }
            Some(file) => {
                let p = PathBuf::from(expand_home(file));
                let t = fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                self.system_chars = t.chars().count();
                cfg.system = Some(p.clone());
                self.system_on = true;
                Ok(format!("system prompt {} ({} chars) from the next turn", p.display(), self.system_chars))
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

fn set_f32(arg: Option<&str>, lo: f32, hi: f32, what: &str) -> Result<f32, String> {
    match arg.and_then(|v| v.parse::<f32>().ok()) {
        Some(v) if (lo..=hi).contains(&v) => Ok(v),
        _ => Err(format!("{what}: a value from {lo} to {hi}")),
    }
}

fn set_u32(arg: Option<&str>, lo: u32, hi: u32, what: &str) -> Result<u32, String> {
    match arg.and_then(|v| v.parse::<u32>().ok()) {
        Some(v) if (lo..=hi).contains(&v) => Ok(v),
        _ => Err(format!("{what}: a count from {lo} to {hi}")),
    }
}

/// A run command (`/pause`, `/resume`, `/save`, `/sample`, `/stop-run`, `/scale X`, `/lr X`, `/every N`) as the
/// line the coach reads, or its usage; None when `name` is not one of them.
pub fn coach_line(name: &str, arg: Option<&str>) -> Option<Result<String, String>> {
    Some(match name {
        "pause" | "resume" | "save" | "sample" => Ok(name.to_string()),
        "stop-run" => Ok("stop".to_string()),
        "scale" | "lr" | "every" => match arg {
            Some(v) if v.parse::<f64>().is_ok() => Ok(format!("{name} {v}")),
            _ => Err(format!("usage: /{name} VALUE")),
        },
        _ => return None,
    })
}

/// A line appended to the coach's command file; refused when no run is feeding (the coach reads only lines appended
/// after it started, so a line written now would be lost).
pub fn coach_send(cfg: &Config, live_run: bool, line: &str) -> Result<String, String> {
    if !live_run {
        return Err(format!("not sent: no live run is feeding the daemon (start build/host/coach ... --cmd {} first)", cfg.cmd_file));
    }
    let path = cfg.in_tree(&cfg.cmd_file);
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{line}"))
        .map(|()| format!("sent to the coach: {line} (acts at the next chunk or report)"))
        .map_err(|e| format!("cannot append to {}: {e}", path.display()))
}

/// The line editor's keys (readline's and the arrows); whether the key was one of them.
pub fn edit_key(e: &mut Editor, code: KeyCode, ctrl: bool, alt: bool) -> bool {
    match code {
        KeyCode::Left if ctrl || alt => e.word_left(),
        KeyCode::Right if ctrl || alt => e.word_right(),
        KeyCode::Left => e.left(),
        KeyCode::Right => e.right(),
        KeyCode::Home => e.home(),
        KeyCode::End => e.end(),
        KeyCode::Backspace if ctrl || alt => e.delete_word_back(),
        KeyCode::Backspace => e.backspace(),
        KeyCode::Delete => e.delete(),
        KeyCode::Char('a') if ctrl => e.home(),
        KeyCode::Char('e') if ctrl => e.end(),
        KeyCode::Char('u') if ctrl => e.kill_to_start(),
        KeyCode::Char('k') if ctrl => e.kill_to_end(),
        // Ctrl+W, and Ctrl+Backspace where the terminal sends ^H for it
        KeyCode::Char('w') | KeyCode::Char('h') if ctrl => e.delete_word_back(),
        KeyCode::Char('z') if ctrl => {
            e.undo();
        }
        KeyCode::Char('b') if alt => e.word_left(),
        KeyCode::Char('f') if alt => e.word_right(),
        KeyCode::Char(c) if !ctrl && !alt => e.insert_char(c),
        _ => return false,
    }
    true
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn test_config() -> Config {
        Config {
            tree: ".".into(),
            cortex: "c".into(),
            orchestra: None,
            orchestra_given: false,
            words: "w".into(),
            vocab: None,
            ngram: None,
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
        let human = vec!["\nHuman:".to_string()];
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" The sun was warm.\nHum");
        c.finish(&human, None, false);
        assert_eq!((c.text.as_str(), c.ended.clone()), (" The sun was warm.", Some(Ended::Label("\nHuman:".into()))));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" The end.");
        c.append(EOT.as_bytes());
        c.append(b"\n");
        c.finish(&human, None, false);
        assert_eq!((c.text.as_str(), c.ended.clone()), (" The end.<|endoftext|>", Some(Ended::Story)));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" and then\n");
        c.finish(&[], None, true);
        assert_eq!((c.text.as_str(), c.ended.clone()), (" and then", Some(Ended::Limit)));
        // the chat markers are single tokens: nothing of them is printed, the client names the one that ended the turn
        let markers = vec!["<|end|>".to_string(), "<|user|>".to_string()];
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b"Fine.\n");
        c.finish(&markers, Some("<|user|>"), false);
        assert_eq!((c.text.as_str(), c.ended.clone()), ("Fine.", Some(Ended::Label("<|user|>".into()))));
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
        let mut chat = Chat::new(&cfg);
        chat.turns.push(Turn::new(Who::You, "hi".into()));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" Hello there.\n");
        c.finish(&[], None, true);
        chat.turns.push(c);
        chat.turns.push(Turn::new(Who::You, "and then?".into()));
        let r = chat.prompt(&cfg).unwrap();
        assert_eq!(r.prompt, "user: hi\nassistant: Hello there.\nuser: and then?\nA:");
        assert_eq!(r.stop, ["\nU:"]);
        cfg.system = Some(sys);
        chat.system_on = true;
        assert_eq!(chat.prompt(&cfg).unwrap().prompt, "system: Be brief.\nuser: hi\nassistant: Hello there.\nuser: and then?\nA:");
        assert_eq!(chat.system_chars, 10);
        chat.ctx_chars = 20; // the oldest exchange is dropped: the dialogue is then "user: and then?\n"
        assert_eq!(chat.prompt(&cfg).unwrap().prompt, "system: Be brief.\nuser: and then?\nA:");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reply_splits_into_its_thinking_and_its_answer() {
        assert_eq!(split_think("<|think|>Let me see.<|/think|>Thirty."), ("Let me see.", "Thirty."));
        assert_eq!(split_think("<|think|><|/think|>Thirty."), ("", "Thirty."));
        assert_eq!(split_think("<|think|>Let me"), ("Let me", ""));
        assert_eq!(split_think("Thirty."), ("", "Thirty."));
        let mut cfg = test_config();
        let dir = std::env::temp_dir().join(format!("cortex-tui-think-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let tpl = dir.join("frame.jinja");
        fs::write(&tpl, "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}{% if not think %}[no think]{% endif %}").unwrap();
        cfg.template = tpl.to_string_lossy().into_owned();
        let mut chat = Chat::new(&cfg);
        chat.turns.push(Turn::new(Who::You, "hi".into()));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b"<|think|>A greeting.<|/think|>Hello.");
        c.finish(&[], None, false);
        chat.turns.push(c);
        chat.turns.push(Turn::new(Who::You, "and then?".into()));
        assert_eq!(chat.prompt(&cfg).unwrap().prompt, "user: hi\nassistant: Hello.\nuser: and then?\n", "an earlier reply is re-fed without its think block");
        assert_eq!(chat.setting(&mut cfg, "think", Some("off")).unwrap().is_ok(), true);
        assert!(chat.prompt(&cfg).unwrap().prompt.ends_with("[no think]"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_vocabulary_of_its_own_switches_the_default_orchestra_off() {
        let dir = std::env::temp_dir().join(format!("cortex-tui-vocab-{}", std::process::id()));
        fs::create_dir_all(dir.join("build/host")).unwrap();
        fs::write(dir.join("build/host/cortex"), "").unwrap();
        fs::write(dir.join("v.txt"), "cxbpe 4096 specials 7\n").unwrap();
        fs::write(dir.join("t.jinja"), "").unwrap();
        fs::write(dir.join("n.ngc"), "").unwrap();
        let args = |extra: &[&str]| {
            let mut a = vec!["--tree".to_string(), dir.to_string_lossy().into_owned(), "--template".into(), "t.jinja".into(), "--vocab".into(), "v.txt".into(), "--system".into(), "off".into()];
            a.extend(extra.iter().map(|s| s.to_string()));
            a.into_iter()
        };
        let mut c = Config::from_args("t", args(&[])).unwrap();
        c.check().unwrap();
        assert_eq!(c.orchestra, None, "the stories' count model is off for a model with its own vocabulary");
        let mut c = Config::from_args("t", args(&["--orchestra", "n.ngc", "--words", "v.txt"])).unwrap();
        c.check().unwrap();
        assert_eq!(c.orchestra.as_deref(), Some("n.ngc"), "an orchestra named explicitly stays");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn settings_and_coach_lines() {
        let mut cfg = test_config();
        let mut chat = Chat::new(&cfg);
        assert_eq!(chat.setting(&mut cfg, "n", Some("64")), Some(Ok("tokens per reply = 64 from the next turn".into())));
        assert_eq!(chat.ngen, 64);
        assert!(chat.setting(&mut cfg, "n", Some("0")).unwrap().is_err());
        assert_eq!(chat.setting(&mut cfg, "temp", Some("0.8")).unwrap().is_ok(), true);
        assert_eq!(chat.sampling.temp, 0.8);
        assert!(chat.setting(&mut cfg, "pause", None).is_none());
        assert_eq!(coach_line("pause", None), Some(Ok("pause".into())));
        assert_eq!(coach_line("stop-run", None), Some(Ok("stop".into())));
        assert_eq!(coach_line("scale", Some("0.5")), Some(Ok("scale 0.5".into())));
        assert!(coach_line("scale", None).unwrap().is_err());
        assert_eq!(coach_line("retry", None), None);
        assert!(coach_send(&cfg, false, "pause").is_err());
    }
}

//! One turn of generation: `build/host/cortex --live PROMPT N ...` run in the tree, its stdout (one BPE piece per
//! write, flushed by the client) streamed as bytes, its stderr kept for the statistics line and for the error text
//! when it fails. The child is never killed: a client that dies after its first command leaves the daemon's session
//! open and the next client cannot be taken (card/cortexd.c poll_session); a turn is at most 2,048 tokens of about
//! 0.4 ms each.

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub prompt_tokens: u64,
    pub generated: u64,
    pub seconds: f64,
    pub tok_s: f64,
    pub card_us: f64,
    pub host_us: f64,
}

pub enum Event {
    Bytes(Vec<u8>),
    Done { ok: bool, stats: Option<Stats>, stderr: Vec<String> },
}

/// The sampler's settings, in the client's defaults (llama.cpp's order: repeat penalty, top-k, top-p, min-p,
/// temperature).
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub greedy: bool,
    pub temp: f32,
    pub topk: u32,
    pub topp: f32,
    pub minp: f32,
    pub rp: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { greedy: false, temp: 0.5, topk: 75, topp: 0.95, minp: 0.05, rp: 1.2 }
    }
}

pub struct Request {
    pub tree: String,
    pub cortex: String,
    /// The orchestrator's count model and word list, both relative to the tree, or none for the plain model.
    pub orchestra: Option<(String, String)>,
    /// The BPE vocabulary file, relative to the tree, when not the client's default.
    pub vocab: Option<String>,
    pub prompt: String,
    /// The reply ends before any of these strings (the frame's end marker and role markers); each goes to the
    /// client as its own `--stop`.
    pub stop: Vec<String>,
    pub ngen: u32,
    pub seed: u64,
    pub sampling: Sampling,
}

pub fn args_of(r: &Request) -> Vec<String> {
    let mut a = vec![
        "--live".to_string(),
        r.prompt.clone(),
        r.ngen.to_string(),
        "--no-echo".into(),
        "--stop-eot".into(),
        "--seed".into(),
        r.seed.to_string(),
    ];
    if r.sampling.greedy {
        a.push("--greedy".into());
    } else {
        let s = &r.sampling;
        for (flag, value) in [("--temp", s.temp.to_string()), ("--topk", s.topk.to_string()), ("--topp", s.topp.to_string()), ("--minp", s.minp.to_string()), ("--rp", s.rp.to_string())] {
            a.push(flag.into());
            a.push(value);
        }
    }
    if let Some((ngc, words)) = &r.orchestra {
        a.extend(["--orchestra".into(), ngc.clone(), "--words".into(), words.clone()]);
    }
    for stop in &r.stop {
        a.extend(["--stop".into(), stop.clone()]);
    }
    if let Some(vocab) = &r.vocab {
        a.extend(["--vocab".into(), vocab.clone()]);
    }
    a
}

/// Runs the turn on its own thread; every stdout read lands as `Event::Bytes`, the exit as `Event::Done`.
pub fn start(r: Request, tx: Sender<Event>) {
    thread::spawn(move || {
        let args = args_of(&r);
        let mut child = match Command::new(&r.cortex)
            .args(&args)
            .current_dir(&r.tree)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(Event::Done { ok: false, stats: None, stderr: vec![format!("cortex: cannot run {}: {e}", r.cortex)] });
                return;
            }
        };
        let mut out = child.stdout.take().expect("piped stdout");
        let err = child.stderr.take().expect("piped stderr");
        let err_lines = thread::spawn(move || BufReader::new(err).lines().map_while(Result::ok).collect::<Vec<String>>());
        let mut buf = [0u8; 512];
        loop {
            match out.read(&mut buf) {
                Ok(0) | Err(_) => break,
                // the pipe is drained even when nobody listens any more, so the client always reaches its QUIT
                Ok(n) => { let _ = tx.send(Event::Bytes(buf[..n].to_vec())); }
            }
        }
        let stderr = err_lines.join().unwrap_or_default();
        let ok = child.wait().map(|s| s.success()).unwrap_or(false);
        let stats = stderr.iter().find_map(|l| parse_stats(l));
        let _ = tx.send(Event::Done { ok, stats, stderr });
    });
}

/// The client's closing line: "cortex: 12 prompt + 128 generated tokens in 0.052 s: 2680.1 tok/s overall; per token
/// card 364.0 us (mailbox + forward + DMA), host 15.2 us (orchestra + sampling + print)".
pub fn parse_stats(line: &str) -> Option<Stats> {
    let rest = line.strip_prefix("cortex: ")?;
    let (prompt_tokens, rest) = number(rest)?;
    let (generated, rest) = number(rest.strip_prefix(" prompt + ")?)?;
    let (seconds, rest) = number(rest.strip_prefix(" generated tokens in ")?)?;
    let (tok_s, rest) = number(rest.strip_prefix(" s: ")?)?;
    let (card_us, rest) = number(rest.strip_prefix(" tok/s overall; per token card ")?)?;
    let (host_us, _) = number(rest.split_once("host ")?.1)?;
    Some(Stats { prompt_tokens: prompt_tokens as u64, generated: generated as u64, seconds, tok_s, card_us, host_us })
}

fn number(s: &str) -> Option<(f64, &str)> {
    let end = s.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).unwrap_or(s.len());
    Some((s[..end].parse().ok()?, &s[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_closing_line() {
        let s = parse_stats("cortex: 12 prompt + 128 generated tokens in 0.052 s: 2680.1 tok/s overall; per token card 364.0 us (mailbox + forward + DMA), host 15.2 us (orchestra + sampling + print)").unwrap();
        assert_eq!(s, Stats { prompt_tokens: 12, generated: 128, seconds: 0.052, tok_s: 2680.1, card_us: 364.0, host_us: 15.2 });
        assert!(parse_stats("cortex: the daemon's live weights; prompt 12 tokens; sampling; bus off").is_none());
    }

    #[test]
    fn builds_the_client_arguments() {
        let r = Request { tree: ".".into(), cortex: "c".into(), orchestra: Some(("n.ngc".into(), "w.txt".into())), vocab: Some("v.txt".into()), prompt: "Once".into(), stop: vec!["\nHuman:".into()], ngen: 64, seed: 5, sampling: Sampling::default() };
        assert_eq!(args_of(&r).join(" "), "--live Once 64 --no-echo --stop-eot --seed 5 --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --orchestra n.ngc --words w.txt --stop \nHuman: --vocab v.txt");
    }
}

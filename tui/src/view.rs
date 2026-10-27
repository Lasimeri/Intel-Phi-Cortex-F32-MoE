//! Drawing the harness: each screen as a term::Frame in seaof.glass's colours and lowercase (the visual style of
//! MechJev's TUI). What is drawn is the Phi Cortex's: the chat on the live weights with the card's and the live
//! run's numbers beside it, the run screen, the help. Nothing here changes what the app holds but the scroll
//! positions and the page size. The pieces the monitor draws with too (the conversation, the say line, a help list,
//! the header and footer, the charts) are public.

use crate::app::{App, Screen};
use crate::chat::{split_think, Ended, Turn, Who, EOT};
use crate::editor::Editor;
use crate::live::{Live, EXPERTS, POOL_ROUNDS};
use crate::term::{theme, Frame, Span};
use crate::wrap::wrap;

pub const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
/// The smallest terminal drawn; anything smaller is told so.
pub const MIN: (u16, u16) = (40, 10);
/// "you    " and "cortex ": the label column of the conversation.
const LABEL: usize = 7;
/// The side panel: its width, and the terminal width it appears from.
const SIDE: usize = 46;
const SIDE_FROM: usize = 100;
/// The loss bars span this range of bits per token (the lineage's losses sit between).
const BAR_LO: f64 = 4.0;
const BAR_HI: f64 = 8.0;
pub const LN_2: f64 = std::f64::consts::LN_2;

pub fn text(t: impl Into<String>) -> Span {
    Span::new(t, theme::TEXT)
}

pub fn dim(t: impl Into<String>) -> Span {
    Span::new(t, theme::DIM)
}

pub fn faint(t: impl Into<String>) -> Span {
    Span::new(t, theme::BORDER)
}

pub fn pad(n: usize) -> Span {
    text(" ".repeat(n))
}

pub fn chars(t: &str) -> usize {
    t.chars().count()
}

/// `t` cut to `w` characters, `…` where it cut.
pub fn clip(t: &str, w: usize) -> String {
    if chars(t) <= w {
        return t.to_string();
    }
    let mut s: String = t.chars().take(w.saturating_sub(1)).collect();
    if w > 0 {
        s.push('…');
    }
    s
}

/// A section rule `── label ────`, `w` wide.
pub fn rule(label: &str, w: usize) -> Vec<Span> {
    let lab = format!("{label} ");
    let rest = w.saturating_sub(3 + chars(&lab));
    vec![faint("── "), text(lab).bold(), faint("─".repeat(rest))]
}

/// A fraction as a bar, `w` cells.
pub fn bar(frac: f64, w: usize) -> Vec<Span> {
    let n = ((frac.clamp(0.0, 1.0) * w as f64).round() as usize).min(w);
    vec![Span::new("█".repeat(n), theme::ACCENT_DIM), faint("─".repeat(w - n))]
}

/// The label column of a column chart: `%8s┤`.
pub const CHART_LABEL: usize = 9;
const EIGHTHS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// A column chart of the last `width` of `values`, `height` rows: each column fills from the bottom in eighths of a
/// row, between the smallest value (the bottom row's label) and the largest (the top row's); the middle row is
/// labelled too; the last column is lit, the others dim. Then an axis row `└───`. Every row is
/// CHART_LABEL + width characters.
pub fn column_chart(values: &[f64], width: usize, height: usize, label: &dyn Fn(f64) -> String) -> Vec<Vec<Span>> {
    let height = height.max(1);
    let shown = &values[values.len().saturating_sub(width)..];
    let lo = shown.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = shown.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (lo, hi) = if shown.is_empty() { (0.0, 1.0) } else if hi - lo < 1e-9 { (lo, lo + 1.0) } else { (lo, hi) };
    let eighths: Vec<usize> = shown.iter().map(|v| (((v - lo) / (hi - lo) * (height * 8) as f64).round() as usize).clamp(1, height * 8)).collect();
    let mut rows = Vec::with_capacity(height + 1);
    for r in 0..height {
        let lab = if r == 0 { label(hi) } else if r + 1 == height { label(lo) } else if r == height / 2 && height > 2 { label((hi + lo) / 2.0) } else { String::new() };
        let mut row = vec![dim(format!("{lab:>8}")), faint("┤")];
        let base = (height - 1 - r) * 8;
        let mut dimmed = String::new();
        let mut lit = String::new();
        for (i, &e) in eighths.iter().enumerate() {
            let k = e.saturating_sub(base).min(8);
            let c = EIGHTHS[k];
            if i + 1 == eighths.len() { lit.push(c); } else { dimmed.push(c); }
        }
        row.push(Span::new(dimmed, theme::ACCENT_DIM));
        row.push(Span::new(lit, theme::TEXT));
        row.push(pad(width.saturating_sub(eighths.len())));
        rows.push(row);
    }
    rows.push(vec![pad(8), faint(format!("└{}", "─".repeat(width)))]);
    rows
}

/// 1234567 as "1,234,567".
pub fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `spans` cut or padded to exactly `w` characters.
pub fn fit(spans: Vec<Span>, w: usize) -> Vec<Span> {
    let mut used = 0;
    let mut out = Vec::new();
    for s in spans {
        let n = chars(&s.text);
        if used + n > w {
            let keep: String = s.text.chars().take(w - used).collect();
            used += chars(&keep);
            out.push(Span { text: keep, ..s });
            break;
        }
        used += n;
        out.push(s);
    }
    out.push(pad(w - used));
    out
}

/// One row of a key-value list, the value cut to the room.
pub fn kv(label: &str, value: impl Into<String>, w: usize) -> Vec<Span> {
    vec![dim(format!("{label:<15}")), text(clip(&value.into(), w.saturating_sub(15)))]
}

pub fn file_name(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

pub fn state_word(state: u32) -> &'static str {
    match state {
        1 => "training",
        2 => "paused",
        3 => "waiting for chunks",
        4 => "stopped",
        _ => "starting",
    }
}

/// The card as a few words for the header: the turn in flight, else the run, else the daemon.
fn card_word(a: &App) -> Vec<Span> {
    if a.chat.in_flight {
        let s = SPIN[a.spin % SPIN.len()];
        return vec![text(format!("{s} generating {:.1} s", a.chat.turn_started.elapsed().as_secs_f64()))];
    }
    match &a.live.live {
        Some(l) if l.run => vec![text(format!("● {}", state_word(l.train_state))), dim(format!(" {} tok/s", group(l.tok_s.round().max(0.0) as u64)))],
        Some(_) if a.live.alive => vec![text("● live"), dim(" no run")],
        _ => vec![dim("○ no daemon")],
    }
}

/// Row 0: the name, where we are, and `right` at the right edge when it fits.
pub fn header_with(f: &mut Frame, title: &str, right: Vec<Span>) {
    let w = f.width as usize;
    let mut left = vec![pad(1), text("intel phi cortex").bold(), dim(format!(" / {title}"))];
    let used: usize = left.iter().chain(&right).map(|s| chars(&s.text)).sum();
    if used + 2 <= w {
        left.push(pad(w - used - 1));
        left.extend(right);
    }
    f.set(0, left, theme::BG);
}

/// Row 0: the name, where we are, and the card on the right.
fn header(f: &mut Frame, a: &App, title: &str) {
    header_with(f, title, card_word(a));
}

/// The last two rows: the status (a message, bold with `!` when it is an error) and the keys; `last` is the key
/// kept at the row's end whatever the width.
pub fn footer_with(f: &mut Frame, status: Option<(&str, bool)>, keys: &[(&str, &str)], last: (&str, &str)) {
    let h = f.height;
    let w = f.width as usize;
    let status = match status {
        Some((s, true)) => vec![pad(1), text(format!("! {}", clip(s, w.saturating_sub(3)))).bold()],
        Some((s, false)) => vec![pad(1), dim(clip(s, w.saturating_sub(1)))],
        None => vec![],
    };
    f.set(h - 2, status, theme::BG);
    f.set(h - 1, keys_row(keys, w, last), theme::BG);
}

/// The last two rows: the status and the keys.
fn footer(f: &mut Frame, a: &App, keys: &[(&str, &str)]) {
    footer_with(f, a.status.as_ref().map(|(s, err, _)| (s.as_str(), *err)), keys, ("f1", "help"));
}

/// The keys row: `(key, what)` pairs in order while they fit, `last` kept at the end, so a narrow terminal loses the
/// least used first.
pub fn keys_row(keys: &[(&str, &str)], w: usize, last: (&str, &str)) -> Vec<Span> {
    let tail = format!(" · {} {}", last.0, last.1);
    let mut out = vec![pad(1)];
    let mut used = 1;
    for (i, (k, what)) in keys.iter().enumerate() {
        let sep = if i == 0 { "" } else { " · " };
        let n = chars(sep) + chars(k) + 1 + chars(what);
        if used + n + chars(&tail) > w {
            break;
        }
        out.push(faint(sep));
        out.push(text(*k));
        out.push(dim(format!(" {what}")));
        used += n;
    }
    let sep = if used > 1 { " · " } else { "" };
    out.extend([faint(sep), text(last.0), dim(format!(" {}", last.1))]);
    out
}

/// The line to say at row `y`, on the surface colour as a hovered row of the site, the cursor where it is typed.
pub fn input_row(f: &mut Frame, y: u16, lead: &str, input: &mut Editor) {
    let w = f.width as usize;
    let shown = input.visible(w.saturating_sub(chars(lead) + 1));
    let cx = input.cursor_on_screen();
    f.set(y, vec![text(lead).bold(), text(shown)], theme::SURFACE);
    f.cursor = Some(((chars(lead) + cx) as u16, y));
}

/// A help list as rows: the keys in a column as wide as the longest (at most half the width), the text wrapped
/// beside it; an empty key is a blank row.
pub fn help_rows(help: &[(&str, &str)], cw: usize) -> Vec<Vec<Span>> {
    let kw = help.iter().map(|(k, _)| chars(k)).max().unwrap_or(10).min(cw / 2);
    let mut rows: Vec<Vec<Span>> = Vec::new();
    for (k, what) in help {
        if k.is_empty() {
            rows.push(vec![]);
            continue;
        }
        for (i, l) in wrap(what, cw.saturating_sub(kw + 2)).into_iter().enumerate() {
            let key = if i == 0 { clip(k, kw) } else { String::new() };
            rows.push(vec![text(format!("{key:<kw$}")), dim(format!("  {l}"))]);
        }
    }
    rows
}

pub fn render(a: &mut App, w: u16, h: u16) -> Frame {
    let mut f = Frame::new(w, h);
    if w < MIN.0 || h < MIN.1 {
        f.set(0, vec![text("the terminal is too small")], theme::BG);
        f.set(1, vec![dim(format!("{} x {} at least", MIN.0, MIN.1))], theme::BG);
        return f;
    }
    match a.screen {
        Screen::Chat => chat(&mut f, a),
        Screen::Run => run(&mut f, a),
        Screen::Help => help(&mut f, a),
    }
    f
}

/// A reply's note: its numbers, in one line under it.
fn turn_note(t: &Turn) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &t.stats {
        parts.push(format!("{} tokens in {:.2} s", group(s.generated), s.seconds));
        parts.push(format!("{} tok/s", group(s.tok_s.round().max(0.0) as u64)));
        parts.push(format!("card {:.0} us, host {:.0} us a token", s.card_us, s.host_us));
        parts.push(format!("prompt {} tokens", group(s.prompt_tokens)));
    }
    parts.push(format!("seed {}", t.seed));
    parts.push(match &t.ended {
        Some(Ended::Story) => "the story's end".to_string(),
        Some(Ended::Label(m)) if m.is_empty() || m == "<|end|>" => "the end marker".to_string(),
        Some(Ended::Label(m)) => format!("stopped at {}", m.replace('\n', "\\n")),
        _ => "the token limit".to_string(),
    });
    parts.join("  · ")
}

/// The conversation as rows of spans, `cw` wide: your lines; the model's thinking (faint, under "think") and its
/// answer (bold, under "cortex") with their note under them, the reply in flight with the spinner after its last
/// row; a hint when there is nothing yet.
pub fn conversation(turns: &[Turn], spin: usize, cw: usize) -> Vec<Vec<Span>> {
    let tw = cw.saturating_sub(LABEL).max(1);
    let mut conv: Vec<Vec<Span>> = Vec::new();
    for t in turns {
        match t.who {
            Who::You => {
                for (i, l) in wrap(&t.text, tw).into_iter().enumerate() {
                    conv.push(vec![faint(if i == 0 { "you    " } else { "       " }), text(l)]);
                }
            }
            Who::Cortex => {
                let (thinking, answer) = split_think(&t.text);
                let mut rows: Vec<Vec<Span>> = wrap(thinking, tw)
                    .into_iter()
                    .filter(|l| !l.is_empty() || !thinking.is_empty())
                    .enumerate()
                    .map(|(i, l)| vec![faint(if i == 0 { "think  " } else { "       " }), faint(l).italic()])
                    .collect();
                let answer = answer.replace(EOT, " ∎");
                let answer = if answer.is_empty() && thinking.is_empty() && t.done { "(no words)".to_string() } else { answer };
                if !answer.is_empty() || t.text.contains(crate::chat::THINK_CLOSE) {
                    rows.extend(wrap(&answer, tw).into_iter().enumerate().map(|(i, l)| vec![dim(if i == 0 { "cortex " } else { "       " }), text(l).bold()]));
                }
                if rows.is_empty() {
                    rows.push(vec![dim("cortex "), text("")]);
                }
                if !t.done {
                    if let Some(last) = rows.last_mut() {
                        last.push(dim(format!(" {}", SPIN[spin % SPIN.len()])));
                    }
                }
                conv.extend(rows);
                if t.done {
                    for l in wrap(&turn_note(t), tw) {
                        if !l.is_empty() {
                            conv.push(vec![pad(LABEL), faint(l).italic()]);
                        }
                    }
                }
            }
        }
        conv.push(vec![]);
    }
    if conv.is_empty() {
        for l in wrap("say something below: the model answers on the card's live weights", cw) {
            conv.push(vec![dim(l)]);
        }
    }
    conv
}

/// The MoE words of the part while the loaded model routes through the pool (lib/cx-proto.h CX_M_BIAS, CX_M_LR,
/// CX_T_MOE_DISTINCT, CX_T_POOL): label and value pairs for a panel or a row.
pub fn moe_pairs(l: &Live) -> Vec<(&'static str, String)> {
    let mut v = vec![("moe", format!("on: bias {:.1}, expert lr {}", l.moe_bias, l.expert_lr)), ("experts", format!("{} of {} updated", group(l.moe_distinct as u64), group(EXPERTS as u64)))];
    if let Some((dir, round)) = l.pool_transfer() {
        v.push(("pool", format!("{dir} under way, round {round} of {POOL_ROUNDS}")));
    }
    v
}

/// The side panel: the card, the live run, the loss by report, this chat's settings.
fn side_panel(a: &App, w: usize) -> Vec<Vec<Span>> {
    let mut p: Vec<Vec<Span>> = vec![rule("the card", w)];
    match (&a.live.live, &a.live.error) {
        (Some(l), _) => {
            p.push(kv("daemon", if a.live.alive { "● alive" } else { "○ absent" }, w));
            p.push(kv("session", if l.session_open { "open" } else { "idle" }, w));
            p.push(vec![]);
            p.push(rule("the live run", w));
            if l.run {
                p.push(kv("state", state_word(l.train_state), w));
                p.push(kv("chunk", format!("{} ({} tokens)", group(l.chunks), group(l.chunks * 32)), w));
                p.push(kv("loss", format!("{:.4} nats = {:.3} b/t", l.loss_nats, l.loss_nats / LN_2), w));
                p.push(kv("rate", format!("{:.2e} (scale {:.3})", l.lr_eff, l.scale), w));
                p.push(kv("speed", format!("{} tok/s", group(l.tok_s.round().max(0.0) as u64)), w));
                p.push(kv("ring", format!("{} chunks ahead", group(l.chunks_fed.saturating_sub(l.chunks))), w));
                p.push(kv("checkpoint", format!("chunk {} ({} pushed)", group(l.ckpt_chunk), l.saves), w));
                if l.memory {
                    p.push(kv("memory", "trained with the run", w));
                }
                if l.moe {
                    for (k, v) in moe_pairs(l) {
                        p.push(kv(k, v, w));
                    }
                }
            } else {
                if l.moe {
                    p.push(kv("moe", "the loaded model routes through the pool", w));
                }
                for l in wrap("no run: the weights are as the last run left them (f2: the run screen)", w) {
                    p.push(vec![dim(l)]);
                }
            }
        }
        (None, Some(e)) => {
            p.push(kv("part", "unreadable", w));
            for l in wrap(e, w) {
                p.push(vec![faint(l)]);
            }
        }
        (None, None) => p.push(kv("part", "reading", w)),
    }
    if !a.run.reports.is_empty() {
        p.push(vec![]);
        p.push(rule(&format!("loss by report, bits (bar: {BAR_LO:.0} to {BAR_HI:.0})"), w));
        let bw = w.saturating_sub(17).max(4);
        let n = a.run.reports.len();
        for pt in &a.run.reports[n.saturating_sub(12)..] {
            let mut row = vec![dim(format!("{:>9} ", group(pt.chunk)))];
            row.extend(bar((pt.bits - BAR_LO) / (BAR_HI - BAR_LO), bw));
            row.push(dim(format!(" {:.3}", pt.bits)));
            p.push(row);
        }
    }
    p.push(vec![]);
    p.push(rule("this chat", w));
    let c = &a.chat;
    if c.sampling.greedy {
        p.push(kv("sampler", "greedy: the argmax token", w));
    } else {
        let s = &c.sampling;
        p.push(kv("sampler", format!("temp {}  top-k {}", s.temp, s.topk), w));
        p.push(kv("", format!("top-p {}  min-p {}  rp {}", s.topp, s.minp, s.rp), w));
    }
    p.push(kv("reply", format!("{} tokens at most", c.ngen), w));
    p.push(kv("context", format!("the last {} chars", group(c.ctx_chars as u64)), w));
    let orchestra = match &a.cfg.orchestra {
        Some(n) if c.orchestra_on => format!("on, {}", file_name(n)),
        _ => "off, the plain model".to_string(),
    };
    p.push(kv("orchestra", orchestra, w));
    let system = match (&a.cfg.system, c.system_on) {
        (Some(path), true) => format!("{} ({} chars)", file_name(&path.to_string_lossy()), group(c.system_chars as u64)),
        (Some(_), false) => "off".to_string(),
        (None, _) => "none".to_string(),
    };
    p.push(kv("system prompt", system, w));
    let frame = match &a.cfg.name {
        Some(n) => format!("{}, as {n}", file_name(&a.cfg.template)),
        None => file_name(&a.cfg.template).to_string(),
    };
    p.push(kv("frame", frame, w));
    p.push(kv("vocabulary", a.cfg.vocab.as_deref().map_or("the client's default".to_string(), |v| file_name(v).to_string()), w));
    p.push(kv("memory", a.cfg.ngram.as_deref().map_or("off, the plain model".to_string(), |n| format!("rows from {}", file_name(n))), w));
    p.push(kv("seed", c.fixed_seed.map_or("fresh each turn".to_string(), |s| s.to_string()), w));
    if let Some(s) = &c.last_stats {
        p.push(kv("last reply", format!("{} tok/s, card {:.0} us", group(s.tok_s.round().max(0.0) as u64), s.card_us), w));
        p.push(kv("its prompt", format!("{} tokens", group(s.prompt_tokens)), w));
    }
    p
}

/// The chat screen: the conversation on the left, the card's and the run's numbers on the right (from 100
/// columns), the line to say at the bottom.
fn chat(f: &mut Frame, a: &mut App) {
    header(f, a, "chat");
    let w = f.width as usize;
    let h = f.height as usize;
    let side = if w >= SIDE_FROM { SIDE } else { 0 };
    let cw = if side > 0 { w.saturating_sub(side + 4) } else { w.saturating_sub(2) }.max(10);
    let rows = h.saturating_sub(5); // header, a gap, the input, the status, the keys
    let conv = conversation(&a.chat.turns, a.spin, cw);
    let back = a.back.min(conv.len().saturating_sub(rows));
    a.back = back;
    a.page = rows;
    let end = conv.len() - back;
    let start = end.saturating_sub(rows);
    let shown = &conv[start..end];
    let sidep = if side > 0 { side_panel(a, side - 1) } else { Vec::new() };
    for r in 0..rows {
        let mut row = vec![pad(1)];
        row.extend(fit(shown.get(r).cloned().unwrap_or_default(), cw));
        if side > 0 {
            row.push(pad(1));
            row.push(faint("│ "));
            row.extend(sidep.get(r).cloned().unwrap_or_default());
        }
        f.set((r + 1) as u16, row, theme::BG);
    }
    input_row(f, (h - 3) as u16, " say › ", &mut a.input);
    let orchestra = if a.chat.orchestra_on && a.cfg.orchestra.is_some() { "orchestra off" } else { "orchestra on" };
    let greedy = if a.chat.sampling.greedy { "sampling" } else { "greedy" };
    footer(f, a, &[("enter", "say it"), ("f2", "the run"), ("f3", orchestra), ("f4", greedy), ("pgup/pgdn", "scroll"), ("ctrl+q", "quit")]);
}

/// Seconds as "12 s", "4 min", "2 h".
fn age(since: std::time::SystemTime) -> String {
    let s = since.elapsed().map_or(0, |d| d.as_secs());
    if s < 90 {
        format!("{s} s ago")
    } else if s < 5400 {
        format!("{} min ago", s / 60)
    } else {
        format!("{} h ago", s / 3600)
    }
}

/// The run screen: the live run in full, the checkpoints, the coach's log tail; the keys steer the coach.
fn run(f: &mut Frame, a: &mut App) {
    header(f, a, "run");
    let w = f.width as usize;
    let h = f.height as usize;
    let cw = w.saturating_sub(2);
    let mut rows: Vec<Vec<Span>> = vec![rule("the live run", cw)];
    match &a.live.live {
        Some(l) if l.run => {
            rows.push(kv("state", state_word(l.train_state), cw));
            rows.push(kv("chunks", format!("{} trained of {} fed ({} tokens trained)", group(l.chunks), group(l.chunks_fed), group(l.chunks * 32)), cw));
            rows.push(kv("reports", format!("{} (every {} chunks)", group(l.reports), l.every), cw));
            rows.push(kv("loss", format!("{:.4} nats = {:.4} bits/token over the last report window", l.loss_nats, l.loss_nats / LN_2), cw));
            rows.push(kv("rate", format!("{:.3e} in effect = base {:.3e} x scale {:.4}", l.lr_eff, l.lr, l.scale), cw));
            rows.push(kv("speed", format!("{} tok/s, idle time included", group(l.tok_s.round().max(0.0) as u64)), cw));
            rows.push(kv("flags", format!("asked {:#x}, acknowledged {:#x} (bit 0 save, 1 stop, 2 pause)", l.flags, l.ack), cw));
            rows.push(kv("checkpoint", format!("chunk {}, {} pushed into the stage", group(l.ckpt_chunk), l.saves), cw));
            if l.memory {
                rows.push(kv("memory", "the n-gram memory trains with the run (its tables in the coach's file)", cw));
            }
            if l.moe {
                for (k, v) in moe_pairs(l) {
                    rows.push(kv(k, v, cw));
                }
            }
        }
        Some(l) if l.moe => rows.push(vec![dim(format!("no run is feeding the daemon; the loaded model routes through the pool; a coach started with --cmd {} takes the keys below", a.cfg.cmd_file))]),
        Some(_) => rows.push(vec![dim(format!("no run is feeding the daemon; a coach started with --cmd {} takes the keys below", a.cfg.cmd_file))]),
        None => rows.push(vec![dim(a.live.error.clone().unwrap_or_else(|| "the part is not read yet".into()))]),
    }
    rows.push(vec![]);
    rows.push(rule(&format!("checkpoints in {}", a.cfg.ckdir), cw));
    if a.run.ckpts.is_empty() {
        rows.push(vec![dim("none yet")]);
    }
    for c in a.run.ckpts.iter().rev().take(6) {
        rows.push(vec![text(format!("{:<20}", c.name)), dim(format!("{:>14} B  {}", group(c.bytes), age(c.modified)))]);
    }
    rows.push(vec![]);
    if a.run.teacher_prompt.is_some() || !a.run.teacher.is_empty() {
        let title = match (&a.run.teacher_prompt, &a.run.teacher_done) {
            (Some(p), Some(d)) => format!("the teacher and the student: \"{p}\": {d}"),
            (Some(p), None) => format!("the teacher and the student: \"{p}\""),
            (None, _) => "the teacher and the student".to_string(),
        };
        rows.push(rule(&clip(&title, cw.saturating_sub(4)), cw));
        let n = a.run.teacher.len();
        if n == 0 {
            rows.push(vec![dim("no test yet")]);
        }
        for i in &a.run.teacher[n.saturating_sub(3)..] {
            let when = if i.chunk == 0 { format!("round {} final", i.round) } else { format!("round {} chunk {}", i.round, group(i.chunk)) };
            let verdict = if i.verdict == "PASS" { text(format!("{} ", i.verdict)).bold() } else { dim(format!("{} ", i.verdict)) };
            rows.push(vec![dim(format!("{when:<22}")), verdict, faint(clip(&i.note, cw.saturating_sub(30)))]);
            rows.push(vec![pad(2), dim("student  "), text(clip(&i.reply, cw.saturating_sub(11)))]);
        }
        rows.push(vec![]);
    }
    rows.push(rule(&format!("the run's events, {}", a.cfg.log_file), cw));
    let room = h.saturating_sub(4).saturating_sub(rows.len());
    let mut log_rows: Vec<Vec<Span>> = Vec::new();
    for line in &a.run.events {
        for l in wrap(line, cw) {
            log_rows.push(vec![faint(l)]);
        }
    }
    if log_rows.is_empty() {
        log_rows.push(vec![dim("no events yet")]);
    }
    let n = log_rows.len();
    rows.extend(log_rows.into_iter().skip(n.saturating_sub(room)));
    for (r, row) in rows.into_iter().take(h.saturating_sub(4)).enumerate() {
        let mut full = vec![pad(1)];
        full.extend(row);
        f.set((r + 1) as u16, full, theme::BG);
    }
    footer(f, a, &[("p", "pause"), ("r", "resume"), ("s", "save"), ("m", "sample"), ("+", "scale x2"), ("-", "scale /2"), ("esc", "chat")]);
}

const HELP: &[(&str, &str)] = &[
    ("enter", "say it: the model continues your text on the live weights"),
    ("esc", "clear the line (on the run and help screens: back to the chat)"),
    ("f2", "the run screen: the live run in full, the checkpoints, the coach's log"),
    ("f3", "the orchestra on or off (the host n-gram mix, word filter, repeat block)"),
    ("f4", "greedy or sampling"),
    ("pgup pgdn ↑ ↓", "scroll the conversation"),
    ("ctrl+a ctrl+e", "the start, the end of the line"),
    ("ctrl+u ctrl+k", "erase to the start, to the end"),
    ("ctrl+w ctrl+←/→", "erase a word, move by words; ctrl+z takes a step back"),
    ("ctrl+q", "quit (a reply in flight finishes first)"),
    ("", ""),
    ("/n N", "tokens per reply (1 to 2048)"),
    ("/ctx CHARS", "transcript characters re-fed as the prompt after the system prompt"),
    ("/temp /topk /topp /minp /rp X", "the sampler (llama.cpp's order; defaults 0.5, 75, 0.95, 0.05, 1.2)"),
    ("/greedy /orchestra [on|off]", "as f4, f3"),
    ("/system [on|off|FILE]", "the system message: ~/CLAUDE.md unless told otherwise, read at every turn"),
    ("the frame", "tui/cortex.jinja frames every turn in the trained format (the markers, the end marker as the stop); an edit applies to the next turn"),
    ("/seed N | auto", "a fixed seed (the same prompt gives the same reply), or fresh each turn"),
    ("/retry", "the last reply again"),
    ("/clear", "a new story: the transcript emptied"),
    ("/run /help /quit", "the screens, and out"),
    ("", ""),
    ("/pause /resume /save /sample", "lines for the coach's command file (a running coach reads them)"),
    ("/scale X /lr X /every N", "the learning-rate multiplier, the base rate, the report cadence"),
    ("/stop-run", "the coach ends the run after the current chunk, with a final checkpoint"),
];

/// The help screen: the keys and the commands, scrolling when they outnumber the rows.
fn help(f: &mut Frame, a: &mut App) {
    header(f, a, "help");
    let w = f.width as usize;
    let h = f.height as usize;
    let rows = help_rows(HELP, w.saturating_sub(2));
    let room = h.saturating_sub(4);
    let back = a.help_back.min(rows.len().saturating_sub(room));
    a.help_back = back;
    for (r, row) in rows.into_iter().skip(back).take(room).enumerate() {
        let mut full = vec![pad(1)];
        full.extend(row);
        f.set((r + 1) as u16, full, theme::BG);
    }
    footer(f, a, &[("↑/↓", "scroll"), ("esc", "chat")]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tests::test_config;

    #[test]
    fn groups_thousands() {
        assert_eq!(group(0), "0");
        assert_eq!(group(999), "999");
        assert_eq!(group(1000), "1,000");
        assert_eq!(group(1234567), "1,234,567");
    }

    #[test]
    fn every_screen_fits_the_terminal() {
        for screen in [Screen::Chat, Screen::Run, Screen::Help] {
            for (w, h) in [(40u16, 10u16), (80, 24), (100, 30), (200, 60)] {
                let mut a = App::new(test_config());
                a.screen = screen;
                a.chat.turns.push(Turn::new(Who::You, "Once upon a time there was a very long sentence that needs wrapping at narrow widths to be drawn whole".into()));
                let mut c = Turn::new(Who::Cortex, String::new());
                c.append(b" there was a cat who liked to sit in the sun all day long and never once got up.");
                c.append(EOT.as_bytes());
                c.finish(&[], None, false);
                a.chat.turns.push(c);
                let f = render(&mut a, w, h);
                for r in 0..h as usize {
                    assert!(f.row_width(r) <= w as usize, "{screen:?} at {w}x{h}: row {r} is {} wide", f.row_width(r));
                }
                if let Some((x, y)) = f.cursor {
                    assert!(x < w && y < h, "{screen:?} at {w}x{h}: the cursor at {x},{y}");
                }
            }
        }
        let mut a = App::new(test_config());
        let f = render(&mut a, 30, 5);
        assert_eq!(f.row_text(0), "the terminal is too small");
    }

    #[test]
    fn the_conversation_labels_and_notes() {
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b" the end.");
        c.append(EOT.as_bytes());
        c.finish(&[], None, false);
        let rows = conversation(&[Turn::new(Who::You, "Once".into()), c], 0, 60);
        let texts: Vec<String> = rows.iter().map(|r| r.iter().map(|s| s.text.as_str()).collect()).collect();
        assert_eq!(texts[0], "you    Once");
        assert_eq!(texts[2], "cortex the end. ∎"); // a piece's leading space is dropped at a row's start
        assert!(texts[3].contains("seed 0") && texts[3].contains("the story's end"));
        let mut c = Turn::new(Who::Cortex, String::new());
        c.append(b"<|think|>A right triangle: 5-12-13.<|/think|>Yes, and its area is 30.");
        c.finish(&["<|end|>".to_string()], Some("<|end|>"), false);
        let rows = conversation(&[Turn::new(Who::You, "Is 5, 12, 13 a right triangle?".into()), c], 0, 60);
        let texts: Vec<String> = rows.iter().map(|r| r.iter().map(|s| s.text.as_str()).collect()).collect();
        assert_eq!(texts[2], "think  A right triangle: 5-12-13.");
        assert_eq!(texts[3], "cortex Yes, and its area is 30.");
        assert!(texts[4].contains("the end marker"), "{}", texts[4]);
        let mut open = Turn::new(Who::Cortex, String::new());
        open.append(b"<|think|>Let me");
        let rows = conversation(&[open], 0, 60);
        assert_eq!(rows[0].iter().map(|s| s.text.as_str()).collect::<String>(), "think  Let me ⠋", "an open think block streams as thinking");
    }
}

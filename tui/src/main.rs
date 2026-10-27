//! cortex-tui: the Intel Phi Cortex's harness, a chat terminal on the card's live weights while the daemon trains
//! and serves (card/cortexd.c), in the visual style of MechJev's TUI (seaof.glass's colours, crossterm only).
//!
//! Each turn runs the gated host client `build/host/cortex --live` and streams its tokens into the conversation as
//! they are sampled; the live-training words of the shared part and the coach's log stand beside it; the coach is
//! steered from the run screen; every turn is framed by a Jinja template (tui/cortex.jinja: the identity, the
//! speaker labels, the stop) with ~/CLAUDE.md as the system message. Nothing typed here and nothing the model writes
//! is training data. The talk itself is chat.rs, shared with the monitor (cortex-watch).
//!
//!   cortex-tui [--tree DIR] [--cortex PATH] [--orchestra NGC|off] [--words LIST] [--system FILE|off]
//!              [--template FILE] [--name NAME] [--cmd FILE] [--log FILE] [--ckdir DIR] [--part FILE]
//!              [--ngen N] [--ctx CHARS]
//!
//! The loop draws when something changed, waits up to 100 ms for a key, and ends on ctrl+q once no reply is in
//! flight: the client ends the daemon's session with a QUIT, and a client killed after its first command leaves the
//! session open for good (card/cortexd.c poll_session), so the client is never killed.
use std::io::IsTerminal;
use std::process::ExitCode;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};

use cortex_tui::chat::Config;
use cortex_tui::term::{self, Term};
use cortex_tui::{app, view};

fn main() -> ExitCode {
    let mut cfg = match Config::from_args("cortex-tui", std::env::args().skip(1)) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("cortex-tui: {msg}");
            return ExitCode::from(2);
        }
    };
    if let Err(msg) = cfg.check() {
        eprintln!("cortex-tui: {msg}");
        return ExitCode::from(1);
    }
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!("cortex-tui: needs a terminal on stdin and stdout (tools/chat.sh opens one on the desktop)");
        return ExitCode::from(1);
    }
    let mut term = match Term::open() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cortex-tui: the terminal: {e}");
            return ExitCode::from(1);
        }
    };
    let mut app = app::App::new(cfg);
    let mut dirty = true;
    let outcome = loop {
        if dirty {
            let (w, h) = Term::size();
            let frame = view::render(&mut app, w, h);
            if let Err(e) = term.draw(&frame) {
                break Err(e);
            }
            dirty = false;
        }
        match event::poll(Duration::from_millis(100)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(k)) => {
                    if k.kind != KeyEventKind::Release {
                        app.key(k);
                        dirty = true;
                    }
                }
                Ok(Event::Paste(s)) => {
                    app.paste(&s);
                    dirty = true;
                }
                Ok(Event::Resize(..)) => dirty = true,
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => {}
            Err(e) => break Err(e),
        }
        if app.tick() {
            dirty = true;
        }
        if let Some(p) = term::take_worker_panic() {
            app.error(format!("a turn's reader failed: {p}"));
            dirty = true;
        }
        if app.done() {
            break Ok(());
        }
    };
    drop(term);
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cortex-tui: {e}");
            ExitCode::from(1)
        }
    }
}

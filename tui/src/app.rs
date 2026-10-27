//! The harness's state: what is on screen and what each key does, over the talk (chat.rs), the live words (live.rs)
//! and the run's files (run.rs). Drawing is view.rs. The visual style is MechJev's (seaof.glass's palette, term.rs);
//! the harness is the Phi Cortex's: a chat on the daemon's live weights with the card's and the live run's numbers
//! beside it, a run screen that steers the coach, and ~/CLAUDE.md as the system prompt of every turn.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::chat::{self, Chat, Config};
use crate::editor::Editor;
use crate::live::LiveState;
use crate::run::Run;

const INFO_FOR: Duration = Duration::from_secs(8);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Run,
    Help,
}

pub struct App {
    pub cfg: Config,
    pub screen: Screen,
    pub chat: Chat,
    pub input: Editor,
    /// Rows scrolled up from the bottom of the conversation; 0 follows the newest.
    pub back: usize,
    /// Conversation rows in the last frame (view.rs sets it): a page.
    pub page: usize,
    pub help_back: usize,
    pub spin: usize,
    pub live: LiveState,
    pub run: Run,
    /// The status row: the message, whether it is an error (kept until the next message; an info clears after 8 s),
    /// and when it was set.
    pub status: Option<(String, bool, Instant)>,
    quit: bool,
}

impl App {
    pub fn new(cfg: Config) -> Self {
        let chat = Chat::new(&cfg);
        let mut app = Self {
            chat,
            cfg,
            screen: Screen::Chat,
            input: Editor::new(),
            back: 0,
            page: 0,
            help_back: 0,
            spin: 0,
            live: LiveState::new(),
            run: Run::default(),
            status: None,
            quit: false,
        };
        if let Some(p) = app.cfg.system.clone() {
            app.info(format!("the system prompt {} ({} chars) prefixes every turn's prompt", p.display(), app.chat.system_chars));
        }
        app
    }

    pub fn done(&self) -> bool {
        self.quit && !self.chat.in_flight
    }

    pub fn info(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), false, Instant::now()));
    }

    pub fn error(&mut self, m: impl Into<String>) {
        self.status = Some((m.into(), true, Instant::now()));
    }

    fn report(&mut self, r: Result<String, String>) {
        match r {
            Ok(m) => self.info(m),
            Err(e) => self.error(e),
        }
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
            KeyCode::F(3) => self.toggle("orchestra"),
            KeyCode::F(4) => self.toggle("greedy"),
            KeyCode::F(5) => self.toggle("think"),
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

    /// A knob flipped (`orchestra`, `greedy`) and the status told.
    fn toggle(&mut self, knob: &str) {
        if let Some(r) = self.chat.setting(&mut self.cfg, knob, None) {
            self.report(r);
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
        if self.chat.in_flight {
            self.info("quitting after the reply in flight (the client is never killed: its QUIT closes the daemon's session)");
        }
    }

    // ---- the line said ----

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
        if self.chat.in_flight {
            return self.error("a reply is streaming; press enter again when it has ended");
        }
        self.input.set_text("");
        match self.chat.say(&self.cfg, line) {
            Ok(()) => self.back = 0,
            Err(e) => self.error(e),
        }
    }

    /// A `/command`: the screens and quitting here; the knobs and the coach's lines in chat.rs.
    fn command(&mut self, cmd: &str) {
        let mut it = cmd.split_whitespace();
        let name = it.next().unwrap_or("");
        let arg = it.next();
        match name {
            "help" | "?" => self.screen = Screen::Help,
            "run" | "status" => self.screen = Screen::Run,
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

    // ---- what the loop does between keys ----

    /// The turn in flight, the live words, the run's files, the status row's age; whether a redraw is due.
    pub fn tick(&mut self) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tests::test_config;

    #[test]
    fn commands_set_the_knobs_and_refuse_bad_values() {
        let mut a = App::new(test_config());
        a.command("n 64");
        assert_eq!(a.chat.ngen, 64);
        a.command("n 0");
        assert_eq!(a.chat.ngen, 64);
        assert!(a.status.as_ref().is_some_and(|s| s.1));
        a.command("temp 0.8");
        assert_eq!(a.chat.sampling.temp, 0.8);
        a.command("seed 42");
        assert_eq!(a.chat.fixed_seed, Some(42));
        a.command("seed auto");
        assert_eq!(a.chat.fixed_seed, None);
        a.command("pause");
        assert!(a.status.as_ref().is_some_and(|s| s.1 && s.0.starts_with("not sent")));
        a.command("nonsense");
        assert!(a.status.as_ref().is_some_and(|s| s.1 && s.0.starts_with("unknown command")));
        a.command("help");
        assert_eq!(a.screen, Screen::Help);
    }
}

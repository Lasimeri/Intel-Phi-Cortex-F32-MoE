//! The Intel Phi Cortex's harness, as a library shared by its two binaries: `cortex-tui` (the chat on the live
//! weights, src/main.rs) and `cortex-watch` (the training monitor with a say line, src/bin/cortex-watch.rs). The
//! visual style is MechJev's TUI (seaof.glass's palette, term.rs); everything drawn is this project's. chat.rs is
//! the talk both share: the turns, the frame, the client runs, the knobs, the lines for the coach.
pub mod app;
pub mod chat;
pub mod cortex;
pub mod editor;
pub mod frame;
pub mod live;
pub mod run;
pub mod term;
pub mod view;
pub mod wrap;

//! The Intel Phi Cortex's harness, as a library shared by its two binaries: `cortex-tui` (the chat on the live
//! weights, src/main.rs) and `cortex-watch` (the training monitor, src/bin/cortex-watch.rs). The visual style is
//! MechJev's TUI (seaof.glass's palette, term.rs); everything drawn is this project's.
pub mod app;
pub mod cortex;
pub mod editor;
pub mod frame;
pub mod live;
pub mod run;
pub mod term;
pub mod view;
pub mod wrap;

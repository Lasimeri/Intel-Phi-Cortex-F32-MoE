//! The line to say: one line of characters and a cursor, with the operations a terminal's keys map to, and a view
//! that scrolls sideways to keep the cursor on screen. No terminal here: view.rs draws what `visible` returns and
//! puts the cursor where `cursor_on_screen` says. The single-line half of MechJev-v2's tui/editor.rs (the same
//! author, MIT OR Apache-2.0): a pasted newline becomes a space; a run of typing up to a space, a run of erasing, a
//! paste, a word erased or a kill each undo as one step.

#[derive(Debug, Clone)]
pub struct Editor {
    chars: Vec<char>,
    /// The cursor, in characters.
    col: usize,
    /// The first column shown, from the last `visible`.
    left: usize,
    /// The text and cursor before each undoable step, newest last.
    history: Vec<(Vec<char>, usize)>,
    /// The kind of the step in progress: a run of one kind is one step.
    step: Option<Step>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Type,
    Erase,
    Other,
}

/// The most steps `undo` goes back.
const HISTORY: usize = 100;

impl Editor {
    pub fn new() -> Self {
        Self { chars: Vec::new(), col: 0, left: 0, history: Vec::new(), step: None }
    }

    /// Replace the text; the cursor goes to its end and the history starts afresh.
    pub fn set_text(&mut self, text: &str) {
        self.chars = text.replace(['\r', '\n'], " ").chars().collect();
        self.col = self.chars.len();
        self.left = 0;
        self.history.clear();
        self.step = None;
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    /// Before an edit: a snapshot, unless it continues the step in progress.
    fn record(&mut self, step: Step) {
        if step == Step::Other || self.step != Some(step) {
            self.history.push((self.chars.clone(), self.col));
            if self.history.len() > HISTORY {
                self.history.remove(0);
            }
        }
        self.step = Some(step);
    }

    /// The last step taken back; whether there was one.
    pub fn undo(&mut self) -> bool {
        let Some((chars, col)) = self.history.pop() else { return false };
        self.chars = chars;
        self.col = col;
        self.step = None;
        true
    }

    /// A move ends the step in progress.
    fn moved(&mut self) {
        self.step = None;
    }

    fn put(&mut self, c: char) {
        let c = if matches!(c, '\t' | '\n' | '\r') { ' ' } else { c };
        self.chars.insert(self.col, c);
        self.col += 1;
    }

    pub fn insert_char(&mut self, c: char) {
        self.record(Step::Type);
        self.put(c);
        if c == ' ' {
            // a word typed is a step
            self.step = None;
        }
    }

    /// A paste: one step, however long; its line breaks become spaces.
    pub fn insert_str(&mut self, s: &str) {
        self.record(Step::Other);
        for c in s.replace("\r\n", "\n").chars() {
            self.put(c);
        }
        self.step = None;
    }

    pub fn backspace(&mut self) {
        if self.col == 0 {
            return;
        }
        self.record(Step::Erase);
        self.col -= 1;
        self.chars.remove(self.col);
    }

    pub fn delete(&mut self) {
        if self.col == self.chars.len() {
            return;
        }
        self.record(Step::Erase);
        self.chars.remove(self.col);
    }

    /// Where the word before the cursor starts: back over spaces, then over the word.
    fn word_start(&self) -> usize {
        let mut c = self.col;
        while c > 0 && self.chars[c - 1] == ' ' {
            c -= 1;
        }
        while c > 0 && self.chars[c - 1] != ' ' {
            c -= 1;
        }
        c
    }

    /// To the start of this word or the one before.
    pub fn word_left(&mut self) {
        self.col = self.word_start();
        self.moved();
    }

    /// Past the end of this word or the next.
    pub fn word_right(&mut self) {
        let mut c = self.col;
        while c < self.chars.len() && self.chars[c] == ' ' {
            c += 1;
        }
        while c < self.chars.len() && self.chars[c] != ' ' {
            c += 1;
        }
        self.col = c;
        self.moved();
    }

    /// The word before the cursor erased.
    pub fn delete_word_back(&mut self) {
        if self.col == 0 {
            return;
        }
        self.record(Step::Other);
        let s = self.word_start();
        self.chars.drain(s..self.col);
        self.col = s;
    }

    /// The rest of the line erased.
    pub fn kill_to_end(&mut self) {
        if self.col == self.chars.len() {
            return;
        }
        self.record(Step::Other);
        self.chars.truncate(self.col);
    }

    /// The line before the cursor erased.
    pub fn kill_to_start(&mut self) {
        if self.col == 0 {
            return;
        }
        self.record(Step::Other);
        self.chars.drain(..self.col);
        self.col = 0;
    }

    pub fn left(&mut self) {
        self.col = self.col.saturating_sub(1);
        self.moved();
    }

    pub fn right(&mut self) {
        self.col = (self.col + 1).min(self.chars.len());
        self.moved();
    }

    pub fn home(&mut self) {
        self.col = 0;
        self.moved();
    }

    pub fn end(&mut self) {
        self.col = self.chars.len();
        self.moved();
    }

    /// Scroll so the cursor is inside a view `width` wide, and return what to draw.
    pub fn visible(&mut self, width: usize) -> String {
        let width = width.max(1);
        if self.col < self.left {
            self.left = self.col;
        } else if self.col >= self.left + width {
            self.left = self.col + 1 - width;
        }
        self.chars.iter().skip(self.left).take(width).collect()
    }

    /// The cursor's column inside the last `visible` view.
    pub fn cursor_on_screen(&self) -> usize {
        self.col - self.left
    }
}

#[cfg(test)]
mod tests {
    use super::Editor;

    #[test]
    fn one_line_that_scrolls_sideways() {
        let mut e = Editor::new();
        e.insert_str("first\r\nsecond");
        assert_eq!(e.text(), "first second");
        assert_eq!(e.visible(4), "ond");
        assert_eq!(e.cursor_on_screen(), 3);
        e.home();
        assert_eq!(e.visible(4), "firs");
        assert_eq!(e.cursor_on_screen(), 0);
        e.insert_char('\n');
        assert_eq!(e.text(), " first second");
    }

    #[test]
    fn words_move_and_erase_like_readline() {
        let mut e = Editor::new();
        e.set_text("one  two three");
        e.word_left();
        assert_eq!(e.col, 9);
        e.word_left();
        assert_eq!(e.col, 5);
        e.word_right();
        assert_eq!(e.col, 8);
        e.delete_word_back();
        assert_eq!(e.text(), "one   three");
        e.kill_to_end();
        assert_eq!(e.text(), "one  ");
        e.kill_to_start();
        assert_eq!((e.text().as_str(), e.col), ("", 0));
        e.backspace();
        e.delete();
        assert_eq!(e.text(), "");
    }

    #[test]
    fn undo_takes_back_a_word_a_run_of_erasing_or_a_paste_at_a_time() {
        let mut e = Editor::new();
        for c in "hello world".chars() {
            e.insert_char(c);
        }
        e.backspace();
        e.backspace();
        assert_eq!(e.text(), "hello wor");
        e.undo(); // the two backspaces
        assert_eq!(e.text(), "hello world");
        e.undo(); // "world"
        assert_eq!(e.text(), "hello ");
        e.insert_str("pasted\ntext");
        e.kill_to_start();
        assert_eq!(e.text(), "");
        e.undo();
        assert_eq!(e.text(), "hello pasted text");
        e.undo(); // the paste
        assert_eq!(e.text(), "hello ");
        e.undo();
        assert_eq!(e.text(), "");
        assert!(!e.undo());
        e.set_text("loaded");
        assert!(!e.undo(), "a load starts a fresh history");
    }
}

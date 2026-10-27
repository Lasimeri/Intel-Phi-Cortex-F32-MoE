//! Word wrap for the transcript: hard breaks at '\n', soft breaks at spaces, a word longer than the width split at
//! the width. Widths count chars (the model's text is ASCII; a wide glyph would only shorten a line).

pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        let mut len = 0usize;
        for word in para.split(' ') {
            let wlen = word.chars().count();
            if len == 0 {
                open_word(&mut lines, &mut line, &mut len, word, width);
            } else if len + 1 + wlen <= width {
                line.push(' ');
                line.push_str(word);
                len += 1 + wlen;
            } else {
                lines.push(std::mem::take(&mut line));
                open_word(&mut lines, &mut line, &mut len, word, width);
            }
        }
        lines.push(line);
    }
    lines
}

/// A word at the start of a line: whole when it fits, else full-width pieces with the last piece left open.
fn open_word(lines: &mut Vec<String>, line: &mut String, len: &mut usize, word: &str, width: usize) {
    let chars: Vec<char> = word.chars().collect();
    let mut i = 0;
    while chars.len() - i > width {
        lines.push(chars[i..i + width].iter().collect());
        i += width;
    }
    *line = chars[i..].iter().collect();
    *len = chars.len() - i;
}

#[cfg(test)]
mod tests {
    use super::wrap;

    #[test]
    fn breaks_at_spaces_and_newlines() {
        assert_eq!(wrap("hello world foo", 11), vec!["hello world", "foo"]);
        assert_eq!(wrap("a\nb", 10), vec!["a", "b"]);
        assert_eq!(wrap("", 10), vec![""]);
    }

    #[test]
    fn splits_a_long_word() {
        assert_eq!(wrap("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        assert_eq!(wrap("x abcdefghij y", 4), vec!["x", "abcd", "efgh", "ij y"]);
    }
}

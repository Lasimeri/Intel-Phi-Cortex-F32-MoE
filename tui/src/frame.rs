//! The frame of a turn: the template (tui/cortex.jinja unless told otherwise) rendered by minijinja with the
//! messages (role system | user | assistant), the identity, the think mode and add_generation_prompt, as a chat
//! template is; the template's `stop` export is the string a reply ends before (the turn's end marker). The file is
//! read at every turn, so an edit to it applies to the next reply.

use minijinja::{context, value::Value, Environment};

/// The BPE's story separator, offered to the template as `eos_token`.
pub const EOS: &str = "<|endoftext|>";

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    pub prompt: String,
    pub stop: Option<String>,
}

/// `think`: the model opens its think block itself (true), or is given an empty one and answers at once (false).
pub fn render(src: &str, messages: &[Message], name: Option<&str>, think: bool) -> Result<Rendered, String> {
    let env = Environment::new();
    let tmpl = env.template_from_str(src).map_err(|e| format!("the frame: {e}"))?;
    let msgs: Vec<Value> = messages.iter().map(|m| context! { role => m.role, content => m.content.as_str() }).collect();
    let identity = match name {
        Some(n) => context! { name => n },
        None => context! {},
    };
    let captured = tmpl
        .render_captured(context! { messages => msgs, identity => identity, think => think, add_generation_prompt => true, eos_token => EOS })
        .map_err(|e| format!("the frame: {e}"))?;
    let stop = captured.state().lookup("stop").and_then(|v| v.as_str().map(String::from)).filter(|s| !s.is_empty());
    Ok(Rendered { prompt: captured.output().to_string(), stop })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: &str = include_str!("../cortex.jinja");

    fn dialogue() -> Vec<Message> {
        vec![
            Message { role: "system", content: "Be brief.\n".into() },
            Message { role: "user", content: "hi".into() },
            Message { role: "assistant", content: " Hello there.".into() },
            Message { role: "user", content: "and then?".into() },
        ]
    }

    #[test]
    fn the_frame_renders_the_dialogue_in_the_trained_format_and_exports_the_stop() {
        let r = render(FRAME, &dialogue(), None, true).unwrap();
        assert_eq!(r.prompt, "<|system|>Be brief.<|end|>\n<|user|>hi<|end|>\n<|assistant|>Hello there.<|end|>\n<|user|>and then?<|end|>\n<|assistant|>");
        assert_eq!(r.stop.as_deref(), Some("<|end|>"));
        let r = render(FRAME, &dialogue(), None, false).unwrap();
        assert!(r.prompt.ends_with("<|user|>and then?<|end|>\n<|assistant|><|think|><|/think|>"), "{}", r.prompt);
        let r = render(FRAME, &dialogue()[1..], Some("Phi"), true).unwrap();
        assert!(r.prompt.starts_with("<|system|>You are Phi, the Intel Phi Cortex"), "{}", r.prompt);
        assert!(r.prompt.ends_with("<|end|>\n<|user|>hi<|end|>\n<|assistant|>Hello there.<|end|>\n<|user|>and then?<|end|>\n<|assistant|>"), "{}", r.prompt);
    }

    #[test]
    fn a_broken_template_is_an_error_not_a_panic() {
        assert!(render("{% for m in messages %}", &dialogue(), None, true).is_err());
    }
}

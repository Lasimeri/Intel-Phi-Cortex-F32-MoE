//! The frame of a turn: the template (tui/cortex.jinja unless told otherwise) rendered by minijinja with the
//! messages (role system | user | assistant), the identity and add_generation_prompt, as a chat template is; the
//! template's `stop` export is the string a reply ends before (the next speaker's label). The file is read at every
//! turn, so an edit to the identity applies to the next reply.

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

pub fn render(src: &str, messages: &[Message], name: Option<&str>) -> Result<Rendered, String> {
    let env = Environment::new();
    let tmpl = env.template_from_str(src).map_err(|e| format!("the frame: {e}"))?;
    let msgs: Vec<Value> = messages.iter().map(|m| context! { role => m.role, content => m.content.as_str() }).collect();
    let identity = match name {
        Some(n) => context! { name => n },
        None => context! {},
    };
    let captured = tmpl
        .render_captured(context! { messages => msgs, identity => identity, add_generation_prompt => true, eos_token => EOS })
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
    fn the_frame_renders_the_dialogue_with_the_identity_and_exports_the_stop() {
        let r = render(FRAME, &dialogue(), None).unwrap();
        assert!(r.prompt.starts_with("Be brief.\n\nCortex is the Intel Phi Cortex"), "{}", r.prompt);
        assert!(r.prompt.contains("\nHuman: hi\nCortex: Hello there.\nHuman: and then?\nCortex:"), "{}", r.prompt);
        assert!(r.prompt.ends_with("Cortex:"));
        assert_eq!(r.stop.as_deref(), Some("\nHuman:"));
        let r = render(FRAME, &dialogue()[1..], Some("Phi")).unwrap();
        assert!(r.prompt.starts_with("Phi is the Intel Phi Cortex"), "{}", r.prompt);
        assert!(r.prompt.ends_with("\nPhi:"));
    }

    #[test]
    fn a_broken_template_is_an_error_not_a_panic() {
        assert!(render("{% for m in messages %}", &dialogue(), None).is_err());
    }
}

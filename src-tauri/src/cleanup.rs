//! Clean-up of raw transcripts via an in-process Apple MLX model.
//!
//! The raw whisper transcript (full of "um/uh/like") is fed to a small LLM that runs **in the
//! zipscribe process** on Apple Silicon via MLX (`crate::mlx`) — no external service or daemon.
//! The LLM removes fillers, fixes grammar/punctuation, and preserves meaning without fabricating
//! content.
//!
//! **No-block guarantee:** on any MLX load/generate failure (model missing/broken, out of memory,
//! empty output, …) the cleaner falls back to the deterministic local `light_cleanse` (filler
//! strip + light punctuation) rather than blocking — dictation must never hang because the model
//! is unavailable.

/// Where the cleansed text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanSource {
    /// Produced by the in-process MLX model.
    Mlx,
    /// Produced by the deterministic `light_cleanse` fallback.
    Fallback,
}

/// A cleansed transcript plus its provenance.
pub struct CleanResult {
    pub text: String,
    pub source: CleanSource,
}

/// Carries the timeout for the no-block path. The resident `crate::mlx::MlxModel` handle is
/// owned by the clean-up thread in `crate::service` and passed on demand.
pub struct Cleaner {
    /// Per-inference timeout, seconds.
    pub timeout_secs: u64,
}

impl Cleaner {
    pub fn new() -> Self {
        Self { timeout_secs: 30 }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Deterministic no-LLM fallback used when cleanup is disabled or MLX is unavailable.
    pub fn fallback_clean(&self, raw: &str) -> CleanResult {
        let raw = raw.trim();
        if raw.is_empty() {
            return CleanResult {
                text: String::new(),
                source: CleanSource::Fallback,
            };
        }
        CleanResult {
            text: light_cleanse(raw),
            source: CleanSource::Fallback,
        }
    }

    /// MLX clean-up of `raw` via the resident `model_handle`. On any failure or empty output it
    /// returns `fallback_clean`, so the pipeline never blocks and never swallows the dictation.
    pub fn clean_mlx(
        &self,
        model_handle: &crate::mlx::MlxModel,
        system_prompt: &str,
        raw: &str,
    ) -> CleanResult {
        let raw = raw.trim();
        if raw.is_empty() {
            return CleanResult {
                text: String::new(),
                source: CleanSource::Fallback,
            };
        }
        let deadline = std::time::Duration::from_secs(self.timeout_secs.max(1));
        // Pure disfluencies are removed up front so a small model only has to fix grammar, and
        // its output gets the same deterministic finish (capitals, "I", end punctuation).
        let pre = strip_fillers(raw);
        match model_handle.generate(system_prompt, &pre, 256, deadline) {
            Ok(out) => {
                let cleaned = finish_sentence(&strip_fillers(&normalize_spaces(&scrub_markdown(
                    &extract_after_thinking(out.trim()),
                ))));
                // A model that returns nothing must never swallow the user's dictation.
                if cleaned.trim().is_empty() {
                    log::warn!("[cleanup] mlx returned empty output; using light_cleanse fallback");
                    self.fallback_clean(raw)
                } else {
                    log::info!("[cleanup] mlx source -> {cleaned:?}");
                    CleanResult {
                        text: cleaned,
                        source: CleanSource::Mlx,
                    }
                }
            }
            Err(e) => {
                log::warn!("[cleanup] mlx unavailable ({e}); using light_cleanse fallback");
                self.fallback_clean(raw)
            }
        }
    }
}

/// Deterministic local fallback: strip common fillers, collapse whitespace, capitalize, and
/// ensure a trailing period.  No LLM, never blocks, faithful (doesn't invent content).
pub fn light_cleanse(raw: &str) -> String {
    finish_sentence(&strip_fillers(raw))
}

/// Drop pure disfluencies (case-insensitive). Words like "like", "so" or "but" carry meaning
/// too often to drop without a language model, so they are left alone.
fn strip_fillers(raw: &str) -> String {
    const FILLERS: [&str; 8] = ["um", "umm", "uh", "uhh", "er", "erm", "hmm", "hmmm"];
    let mut out: Vec<String> = Vec::new();
    for tok in raw.split_whitespace() {
        let lower = strip_punct(tok).to_lowercase();
        if FILLERS.iter().any(|f| f == &lower) {
            continue;
        }
        // "you know" mid-sentence: drop both words, not just "know".
        if lower == "know" && out.last().is_some_and(|p| strip_punct(p).eq_ignore_ascii_case("you")) {
            out.pop();
            continue;
        }
        out.push(tok.to_string());
    }
    out.join(" ").trim().to_string()
}

/// Capitalize each sentence start and the pronoun "I" (incl. I'm/I'll/I've/I'd), and end with
/// terminal punctuation.
fn finish_sentence(s: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut sentence_start = true;
    for tok in s.split_whitespace() {
        let bare = strip_punct(tok).to_lowercase();
        let is_i = bare == "i" || bare.starts_with("i'") || bare.starts_with("i\u{2019}");
        let t = if sentence_start || is_i { capitalize_first(tok) } else { tok.to_string() };
        sentence_start = t.ends_with(['.', '!', '?']);
        out.push(t);
    }
    let mut s = out.join(" ");
    if !s.is_empty() && !s.ends_with(['.', '!', '?', ':']) {
        s.push('.');
    }
    s
}

/// Trim leading/trailing non-alphanumeric punctuation (commas, quotes, parens, etc.) so a
/// token like "um," or "(like)" still matches the bare filler list.
fn strip_punct(s: &str) -> &str {
    let s = s.trim();
    let start = s.find(|c: char| c.is_alphanumeric()).unwrap_or(0);
    let end = s
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    if start >= end {
        ""
    } else {
        &s[start..end]
    }
}

/// Capitalize the first Unicode scalar, leave the rest intact.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut out = String::new();
            for c in first.to_uppercase() {
                out.push(c);
            }
            out.extend(chars);
            out
        }
    }
}

/// Collapse runs of whitespace/tabs/newlines to single spaces.
pub fn normalize_spaces(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Drop any inline reasoning block from the model output. Text after the last closing think
/// tag is the answer; a dangling open tag (truncated reasoning) yields empty so the caller
/// falls back to `light_cleanse`. Output with no think tags is returned unchanged.
pub fn extract_after_thinking(s: &str) -> String {
    if let Some(i) = s.rfind("</think>") {
        return s[i + "</think>".len()..].trim().to_string();
    }
    if s.contains("<think>") {
        return String::new();
    }
    s.trim().to_string()
}

/// Defensive guard for LLM output that, even after the strong "output ONLY plain text" instruction,
/// sometimes carries a stray line of commentary, markdown markers, or surrounding quotes. Conservative
/// and deterministic: it strips wrappers but never invents content. With qwen3 thinking ON the
/// `content` field is already clean, so this is a safety net, not the main cleaning step.
pub fn scrub_markdown(s: &str) -> String {
    let t = s.trim();
    // 1. Keep every substantive content line (skip headings/bullets/"here is"-style lines) so a
    //    multi-sentence answer split across lines is not truncated.
    let kept: Vec<&str> = t.lines().map(str::trim).filter(|l| is_content_line(l)).collect();
    let joined = kept.join(" ");
    let line = if kept.is_empty() { t } else { joined.as_str() };
    // 2. Strip markdown emphasis markers into an owned string, then a surrounding quote pair.
    let cleaned_str: String = line.replace("**", "").replace('*', "").replace('`', "");
    let s = cleaned_str.trim();
    let s = s
        .strip_prefix('"')
        .and_then(|x| x.trim_end().strip_suffix('"'))
        .or_else(|| s.strip_prefix('\u{201c}').and_then(|x| x.trim_end().strip_suffix('\u{201d}')))
        .unwrap_or(s);
    s.trim().to_string()
}

/// A line counts as content if it's non-empty and not obvious commentary/heading/bullet markup.
fn is_content_line(l: &str) -> bool {
    if l.is_empty() {
        return false;
    }
    let lower = l.to_lowercase();
    !l.starts_with('#')
        && !l.starts_with("- ")
        && !l.starts_with("* ")
        && !lower.starts_with("here is")
        && !lower.starts_with("here's")
        && !lower.contains("why this")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_strips_bullets_and_marks() {
        assert_eq!(scrub_markdown("**We should reschedule the meeting.**"), "We should reschedule the meeting.");
        assert_eq!(scrub_markdown("# Title\n- foo\nWe should reschedule the meeting."), "We should reschedule the meeting.");
        assert_eq!(scrub_markdown("\u{201c}Reschedule for Friday.\u{201d}"), "Reschedule for Friday.");
        assert_eq!(scrub_markdown("Here is the cleaned text:\nWe should reschedule."), "We should reschedule.");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_spaces("a\nb\t  c\n\nd"), "a b c d");
    }

    #[test]
    fn light_cleanse_strips_fillers() {
        let out = light_cleanse("um so i think we should um like go to the store and get some milk");
        assert!(!out.to_lowercase().contains(" um "), "fillers remain: {out}");
        assert!(out.ends_with('.'), "no trailing period: {out}");
        assert!(out.chars().next().map_or(false, |c| c.is_uppercase()), "not capitalized: {out}");
        assert!(out.contains("go"), "meaning lost: {out}");
        assert!(out.contains("milk"), "meaning lost: {out}");
    }

    #[test]
    fn finish_sentence_capitalizes_sentences_and_pronoun_i() {
        assert_eq!(
            finish_sentence("i think so. i'm sure it works? yes i do"),
            "I think so. I'm sure it works? Yes I do."
        );
        assert_eq!(finish_sentence("in it"), "In it.");
    }

    #[test]
    fn light_cleanse_drops_you_know_and_keeps_meaningful_words() {
        let out = light_cleanse("move it to friday you know but i like the plan");
        assert_eq!(out, "Move it to friday but I like the plan.");
    }

    #[test]
    fn extract_strips_think_blocks_and_keeps_multiline_answers() {
        assert_eq!(extract_after_thinking("<think>\n\n</think>\n\nHello there."), "Hello there.");
        assert_eq!(extract_after_thinking("<think>still reasoning"), "");
        assert_eq!(scrub_markdown("First sentence.\nSecond sentence."), "First sentence. Second sentence.");
    }

    #[test]
    fn light_cleanse_preserves_meaning() {
        let out = light_cleanse("the quick brown fox jumps over the lazy dog");
        assert_eq!(out, "The quick brown fox jumps over the lazy dog.");
    }
}

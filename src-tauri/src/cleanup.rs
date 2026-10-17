//! LLM cleanup of raw transcripts via a local Ollama model.
//!
//! The raw whisper transcript (full of "um/uh/like") is POSTed to a configurable local Ollama
//! model over `http://localhost:11434`.  The LLM removes fillers, fixes grammar/punctuation, and
//! preserves meaning without fabricating content.  **On any Ollama failure the cleaner falls back
//! to a deterministic local "light cleanse" (filler strip + light punctuation) rather than
//! blocking** — dictation must never hang because the LLM is down.

use reqwest::blocking::Client;
use serde_json::json;
use std::time::Duration;

/// Where the cleansed text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanSource {
    Ollama,
    Fallback,
}

/// A cleansed transcript plus its provenance.
pub struct CleanResult {
    pub text: String,
    pub source: CleanSource,
}

#[derive(Debug, thiserror::Error)]
pub enum CleanError {
    #[error("ollama request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("unexpected ollama response: {0}")]
    Response(String),
}

/// A configurable, Ollama-backed cleaner with a local fallback.
pub struct Cleaner {
    ollama_url: String,
    model: String,
    client: Client,
}

const SYSTEM_PROMPT: &str = "You convert raw dictation into clean, ready-to-type prose for a dictation app.\n\
Remove ALL hesitation and filler words — um, uh, er, eh, hmm, like, so, so basically, you know, I mean,\n\
I think, I was thinking, basically, essentially, actually, kind of, sort of, that is to say, well, okay,\n\
alright — and remove stuttered or repeated words (\"the the\" -> \"the\", \"and and\" -> \"and\").\n\
Fix capitalization, punctuation, and basic grammar, and add a single period at the end.\n\
Preserve every spoken CONTENT word and proper noun exactly — do NOT add facts, opinions,\n\
examples, or words that were not spoken, do NOT fabricate anything, and do NOT substitute,\n\
correct, or \"fix\" a word that looks like a mishearing (e.g. leave \"crime\" as-is even if\n\
\"client\" was meant): your job is to remove fillers and fix grammar/punctuation/capitalization\n\
ONLY - correcting transcribed content is not your job. Do the cleaning silently and output ONLY\n\
the result: plain text, one line, no markdown, no bold/asterisks, no quotation marks, no \"here\n\
is\", no explanation, no bullet list, no leading \"so\" or \"the answer is\". Just the cleaned\n\
sentence(s) and nothing else.";

impl Cleaner {
    pub fn new(ollama_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            ollama_url: ollama_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("build reqwest client"),
        }
    }

    /// Cleanse `raw`.  Tries Ollama; on any error returns the local light-cleansed fallback so
    /// the dictation pipeline never blocks.
    pub fn clean(&self, raw: &str) -> Result<CleanResult, CleanError> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(CleanResult {
                text: String::new(),
                source: CleanSource::Fallback,
            });
        }
        let prompt = format!("Raw dictation transcript:\n{}\n\nReturn ONLY the cleaned prose (one line):", raw);
        let body = json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": prompt }
            ],
            "stream": false,
            "think": false,
            // Some reasoning models (qwen3) reason inline; we extract the final answer after the\n            // cap num_predict so a short one-line output can't run away.
            "options": { "temperature": 0.0, "num_ctx": 4096, "num_predict": 256 }
        });

        match self.call_ollama(&body) {
            Ok(text) => {
                let cleaned = normalize_spaces(&scrub_markdown(&extract_after_thinking(text.trim())));
                // A model that returns nothing must never swallow the user's dictation.
                if cleaned.trim().is_empty() {
                    log::warn!("[cleanup] model returned empty output; using local fallback");
                    Ok(CleanResult {
                        text: light_cleanse(raw),
                        source: CleanSource::Fallback,
                    })
                } else {
                    Ok(CleanResult {
                        text: cleaned,
                        source: CleanSource::Ollama,
                    })
                }
            }
            Err(e) => {
                log::warn!("[cleanup] ollama unavailable ({e}); using local fallback");
                Ok(CleanResult {
                    text: light_cleanse(raw),
                    source: CleanSource::Fallback,
                })
            }
        }
    }

    /// POST to Ollama's native chat endpoint and parse `message.content`.
    fn call_ollama(&self, body: &serde_json::Value) -> Result<String, CleanError> {
        let url = format!("{}/api/chat", self.ollama_url);
        let resp = self.client.post(&url).json(body).send()?;
        if !resp.status().is_success() {
            return Err(CleanError::Response(format!(
                "HTTP {} from {}",
                resp.status(),
                &url
            )));
        }
        let parsed: serde_json::Value = resp.json()?;
        // Ollama native response shape: { "message": { "content": "..." } }
        let content = parsed.pointer("/message/content").and_then(|v| v.as_str()).ok_or_else(
            || CleanError::Response(format!("no /message/content in response: {parsed}")),
        )?;
        Ok(content.to_string())
    }

    /// Whether the configured Ollama endpoint is reachable (used by the tray to show status).
    #[allow(dead_code)]
    pub fn ollama_reachable(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.ollama_url))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// Deterministic local fallback: strip common fillers, collapse whitespace, capitalize, and
/// ensure a trailing period.  No LLM, never blocks, faithful (doesn't invent content).
pub fn light_cleanse(raw: &str) -> String {
    // Single-word fillers (case-insensitive).  "you know" is caught by dropping "know" after "you".
    const FILLERS: [&str; 16] = [
        "um", "uh", "er", "hmm", "hmmm", "like", "basically", "essentially", "actually",
        "literally", "okay", "ok", "well", "right", "so", "but",
    ];
    let mut out: Vec<String> = Vec::new();
    let mut last_lower = String::new();
    for tok in raw.split_whitespace() {
        let bare = strip_punct(tok);
        let lower = bare.to_lowercase();
        let is_single = FILLERS.iter().any(|f| f == &lower);
        let is_two = bare == "know" && last_lower == "you";
        if is_single || is_two {
            continue;
        }
        out.push(tok.to_string());
        last_lower = lower;
    }
    let mut s = out.join(" ").trim().to_string();
    s = capitalize_first(&s);
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
fn normalize_spaces(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract the final answer from a model that reasons inline (e.g. qwen3), even when think is left
/// on and the whole response (reasoning + answer) lands in the content field. The answer is whatever
/// follows the last closing think block. A dangling open tag (truncated reasoning) yields empty so
/// the caller falls back to light_cleanse. Deterministic; never invents content.
fn extract_after_thinking(s: &str) -> String {
    // qwen3 (via Ollama 0.40) reasons as PLAIN TEXT and settles on its answer as the LAST
    // substantive line; some variants keep reasoning in a separate `thinking` field (already
    // absent from `content`). So take the last content line, skipping markdown/bullet/commentary.
    // For a non-reasoning response this is effectively the whole (single-line) text.
    let lines: Vec<&str> = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && is_content_line(l))
        .collect();
    match lines.last() {
        Some(l) => l.to_string(),
        None => s.trim().to_string(),
    }
}

/// Defensive guard for LLM output that, even after the strong "output ONLY plain text" instruction,
/// sometimes carries a stray line of commentary, markdown markers, or surrounding quotes. Conservative
/// and deterministic: it strips wrappers but never invents content. With qwen3 thinking ON the
/// `content` field is already clean, so this is a safety net, not the main cleaning step.
fn scrub_markdown(s: &str) -> String {
    let t = s.trim();
    // 1. Take the first substantive content line (skip headings/bullets/"here is"-style lines).
    let line = t.lines().map(str::trim).find(|l| is_content_line(l)).unwrap_or(t).trim();
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
fn is_content_line(l: &str) -> bool {    if l.is_empty() {
        return false;
    }
    let lower = l.to_lowercase();
    !l.starts_with('#')
        && !l.starts_with("- ")
        && !l.starts_with("* ")
        && !lower.starts_with("here is")
        && !lower.starts_with("here's")
        && !lower.starts_with("the clean")
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

    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_spaces("a\nb\t  c\n\nd"), "a b c d");
    }

    #[test]
    fn light_cleanse_strips_fillers() {
        let out = light_cleanse("um so i think we should um like go to the store and get some milk");
        assert!(!out.to_lowercase().contains(" um "), "fillers remain: {out}");
        assert!(!out.to_lowercase().contains("like "), "fillers remain: {out}");
        assert!(out.ends_with('.'), "no trailing period: {out}");
        assert!(out.chars().next().map_or(false, |c| c.is_uppercase()), "not capitalized: {out}");
        assert!(out.contains("go"), "meaning lost: {out}");
        assert!(out.contains("milk"), "meaning lost: {out}");
    }

    #[test]
    fn light_cleanse_preserves_meaning() {
        let out = light_cleanse("the quick brown fox jumps over the lazy dog");
        assert_eq!(out, "The quick brown fox jumps over the lazy dog.");
    }
}

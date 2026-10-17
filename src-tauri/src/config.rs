//! Config schema: load/save to `~/.config/zipscribe/config.toml`.
//!
//! Design rules:
//!   * Missing file → use `Default`.
//!   * Missing/unknown fields → per-field `#[serde(default)]` so an old/partial file still loads.
//!   * Round-trippable: a `Config` serializes to TOML and deserializes back identically.
//!
//! The config file is the user-facing knob set (also surfaced in the tray menu):
//! `hotkey`, `cleanup_*` (Ollama endpoint + model + toggle), `output_mode`, and `enabled`.
//! The nested sections described in docs/SPEC.md are flattened here for v1 ergonomics.

use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::PathBuf;

/// Master user config. Flat for v1; nested sections were a spec suggestion only.
/// `Default` is hand-written (see below) because the v1 defaults are non-trivial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Config {
    // ---- general ----
    /// Master enable/disable for the dictation service.
    pub enabled: bool,
    /// Push-to-Talk hotkey chord, e.g. `"ctrl+space"`.
    pub hotkey: String,
    /// Where cleansed text goes: `"type"` (key simulation), `"paste"` (clipboard + ⌘V) or `"clipboard"` (copy only).
    pub output_mode: String,
    /// Whether to append a trailing newline after typing.
    pub type_trailing_newline: bool,
    /// Set once the guided first-run setup has been completed (or dismissed).
    pub setup_complete: bool,

    // ---- STT (whisper) ----
    /// Whisper model identifier (must be `whisper-medium.en` per the goal).
    pub stt_model: String,
    /// Optional override filesystem path to the GGML model file.
    pub stt_model_path: String,
    /// Whisper worker thread count (0 => auto).
    pub stt_threads: u32,

    // ---- cleanup (Ollama LLM) ----
    /// Whether to run LLM cleanup at all.
    pub cleanup_enabled: bool,
    /// `"ollama"` runs the local LLM; `"local_fallback"` skips it.
    pub cleanup_mode: String,
    /// Local Ollama base URL.
    pub ollama_url: String,
    /// Ollama model name to use for cleanup (small/fast by default).
    pub cleanup_model: String,
    /// Per-request timeout for the Ollama call, in seconds.
    pub cleanup_timeout_secs: u64,
    /// System prompt governing filler removal / no-fabrication.
    pub cleanup_system_prompt: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            hotkey: "ctrl+space".to_string(),
            output_mode: "type".to_string(),
            type_trailing_newline: false,
            setup_complete: false,
            stt_model: "whisper-medium.en".to_string(),
            stt_model_path: String::new(),
            stt_threads: 0,
            cleanup_enabled: true,
            cleanup_mode: "ollama".to_string(),
            ollama_url: "http://localhost:11434".to_string(),
            // Default to a small/fast model for low-latency cleanup. Must be pulled
            // locally before cleanup works (see cleanup step / `cargo run -- install-model`).
            cleanup_model: "qwen2.5:7b".to_string(),
            cleanup_timeout_secs: 20,
            cleanup_system_prompt: default_system_prompt().to_string(),
        }
    }
}

/// v1 default cleanup prompt: removes fillers, fixes grammar/punctuation, never fabricates.
/// Built with `.join` so the literal newlines are unambiguous (avoids fragile adja\xadcent-literal parsing).
fn default_system_prompt() -> String {
    [
        "You are a dictation post-processor. Given a raw speech transcript, return cleaned plain text that is ready to type into a field. Rules:",
        "1. Remove filler/disfluency words: um, uh, er, like, you know, basically, I mean, that's, kind of.",
        "2. Fix grammar and spelling, and add/correct sentence punctuation and capitalization.",
        "3. PRESERVE the speaker's meaning exactly. Do NOT add facts, opinions, or length not spoken. Do NOT fabricate.",
        "4. Keep proper nouns and entities faithful.",
        "5. Output plain text only: no quotes, no markdown, no commentary, no preamble.",
        "6. Do not add a trailing newline.",
    ]
    .join("\n")
}

impl Config {
    /// Resolve the config file path: honor `ZIPSCRIBE_CONFIG` for testing, else
    /// `~/.config/zipscribe/config.toml`.
    ///
    /// `ZIPSCRIBE_CONFIG` is only used under `-A`/test harnesses; on real macOS we
    /// default to the XDG-ish `~/.config` location.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("ZIPSCRIBE_CONFIG") {
            return PathBuf::from(p);
        }
        let mut p = std::env::var("HOME")
            .map(|h| PathBuf::from(h))
            .expect("HOME must be set on macOS");
        p.push(".config/zipscribe/config.toml");
        p
    }

    /// Load config, falling back to fully-default when the file is missing/corrupt.
    ///
    /// Always returns a valid `Config` so the service can start with sane defaults.
    pub fn load_or_default() -> Config {
        Self::load().unwrap_or_default()
    }

    /// Load config from the default path.
    ///
    /// Returns `Ok(cfg)` when the file exists, or an error when it is absent or fails to
    /// parse. The caller decides whether absence is fatal (service: not, tests: yes).
    pub fn load() -> Result<Config, ConfigError> {
        let path = Self::default_path();
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ConfigError::NotFound(path));
            }
            Err(e) => return Err(ConfigError::Io(path, e)),
        };
        let cfg = toml::from_str::<Config>(&raw).map_err(|e| ConfigError::Parse(e.to_string()))?;
        Ok(cfg)
    }

    /// Persist this config atomically (write temp + rename) creating parent dirs.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::default_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ConfigError::Io(parent.to_path_buf(), e))?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| ConfigError::Parse(e.to_string()))?;
        let tmp = path.with_extension("toml.tmp");
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| ConfigError::Io(tmp.clone(), e))?;
            f.write_all(text.as_bytes())
                .map_err(|e| ConfigError::Io(tmp.clone(), e))?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| ConfigError::Io(path, e))?;
        Ok(())
    }
}

/// Errors from loading/saving config. Dependency-free (manual `Display`/`Error`).
#[derive(Debug)]
pub enum ConfigError {
    NotFound(PathBuf),
    Io(PathBuf, std::io::Error),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::NotFound(p) => write!(f, "config file not found at {}", p.display()),
            ConfigError::Io(p, e) => write!(f, "io error for {}: {e}", p.display()),
            ConfigError::Parse(msg) => write!(f, "failed to parse config: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io(_, e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_round_trip(cfg: &Config) {
        // Serialize to TOML and back; must be identical.
        let text = toml::to_string_pretty(cfg).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(&back, cfg, "round-trip mismatch");
    }

    #[test]
    fn defaults_match_spec_values() {
        let d = Config::default();
        assert!(
            d.enabled,
            "default enabled should be true per scaffold contract"
        );
        assert_eq!(d.hotkey, "ctrl+space");
        assert_eq!(d.output_mode, "type");
        assert_eq!(d.stt_model, "whisper-medium.en");
        assert_eq!(d.cleanup_mode, "ollama");
        assert_eq!(d.ollama_url, "http://localhost:11434");
        assert!(d.cleanup_enabled);
        assert!(
            !d.cleanup_model.is_empty(),
            "default cleanup model must be set (small/fast)"
        );
        assert!(d.cleanup_timeout_secs > 0);
        assert!(
            d.cleanup_system_prompt.contains("fabricat"),
            "system prompt must forbid fabrication"
        );
        assert_round_trip(&d);
    }

    #[test]
    fn full_example_round_trips() {
        let mut c = Config::default();
        c.enabled = false;
        c.hotkey = "cmd+alt+z".to_string();
        c.output_mode = "type".to_string();
        c.stt_threads = 4;
        c.cleanup_model = "llama3.2:1b".to_string();
        c.ollama_url = "http://127.0.0.1:11434".to_string();
        c.cleanup_timeout_secs = 12;
        c.type_trailing_newline = true;
        assert_round_trip(&c);
    }

    #[test]
    fn missing_file_returns_not_found() {
        let old = std::env::var("ZIPSCRIBE_CONFIG").ok();
        unsafe {
            std::env::set_var(
                "ZIPSCRIBE_CONFIG",
                "/nonexistent/zipscribe/nope.toml",
            );
        }
        let err = Config::load().unwrap_err();
        assert!(matches!(err, ConfigError::NotFound(_)), "got: {err:?}");
        match old {
            Some(v) => unsafe { std::env::set_var("ZIPSCRIBE_CONFIG", v) },
            None => unsafe { std::env::remove_var("ZIPSCRIBE_CONFIG") },
        }
    }

    #[test]
    fn missing_and_unknown_fields_fall_back_to_defaults() {
        use std::io::Write;
        let mut dir = std::env::temp_dir();
        dir.push(format!("zipscribe-cfgtest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("config.toml");
        let old = std::env::var("ZIPSCRIBE_CONFIG").ok();
        unsafe {
            std::env::set_var("ZIPSCRIBE_CONFIG", f.clone());
        }

        // A file that sets only `hotkey` under a `[general]` section must load,
        // ignoring unknown sections/fields and defaulting the rest.
        let mut fp = std::fs::File::create(&f).unwrap();
        fp.write_all(b"hotkey = \"alt+q\"\ntotally_unknown = 42\n")
            .unwrap();

        let cfg = Config::load().unwrap_or_else(|e| panic!("load should succeed, got {e}"));
        assert_eq!(cfg.hotkey, "alt+q");
        // Unknown fields/sections silently ignored; missing ones defaulted:
        assert!(cfg.enabled);
        assert_eq!(cfg.stt_model, "whisper-medium.en");
        assert_eq!(cfg.output_mode, "type");
        assert!(!cfg.cleanup_model.is_empty());

        std::fs::remove_dir_all(&dir).ok();
        match old {
            Some(v) => unsafe { std::env::set_var("ZIPSCRIBE_CONFIG", v) },
            None => unsafe { std::env::remove_var("ZIPSCRIBE_CONFIG") },
        }
    }
}

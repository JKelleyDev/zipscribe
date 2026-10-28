//! In-process, on-device LLM clean-up via Apple MLX (native Rust `mlxrs` binding).
//!
//! The raw whisper transcript (full of "um/uh/like") is fed to a small LLM that runs
//! **inside the zipscribe process** — no external service or daemon or helper process — on
//! Apple Silicon, MLX's native home. The model is a standard MLX-format directory
//! (safetensors + tokenizer + config.json), which makes it a drop-in for any `mlx-community`
//! model downloaded from HuggingFace.
//!
//! **No-block guarantee:** any load/generate failure surfaces as `Err` so the caller falls
//! back to the deterministic `crate::cleanup::light_cleanse`. Dictation must never hang
//! because the model is missing, broken, or out of memory.
//!
//! Device: MLX prefers the GPU (Metal). When the Metal backend is not built (this machine
//! ships only the Xcode Command Line Tools, so we compile MLX CPU-only — see the
//! `mlx-vendor/mlxrs-sys/build.rs` `MLX_BUILD_METAL=OFF` define) we force the default
//! device + stream to CPU so the same in-process path still runs. On a machine built with
//! Metal, `Device::gpu()` / the default (already-GPU) stream take over automatically.

use std::path::Path;
use std::time::{Duration, Instant};

use mlxrs::lm::factory::ModelContext;
use mlxrs::lm::generate::GenConfig;
use mlxrs::lm::load;
use mlxrs::lm::models::qwen3::{Qwen3, Qwen3Config};

/// A loaded MLX cleanup model, kept resident for reuse across dictation cycles.
///
/// Not `Send`/`Sync`-safe by construction here; the single dictation worker owns it.
pub struct MlxModel {
    ctx: ModelContext,
}

/// Load an MLX model directory (safetensors + tokenizer + config.json).
///
/// Forces the default device to GPU when available, else CPU, so the same in-process
/// path runs on both. The patched MLP op layer's CPU-stream fallback covers the
/// fully-CPU build. Returns `Err` (→ caller falls back to `light_cleanse`) on any
/// missing file, wrong format, or backend failure.
pub fn load_model(model_dir: &Path) -> Result<MlxModel, String> {
    // Prefer the GPU device (Metal) when present; otherwise force CPU so the in-process
    // forward pass + sampling run without a GPU. The op layer's CPU-stream fallback keeps
    // this working on machines without full Xcode (where MLX is built CPU-only).
    match mlxrs::Device::gpu() {
        Ok(gpu) => {
            let _ = gpu.set_default();
        }
        Err(_) => {
            if let Ok(cpu) = mlxrs::Device::cpu() {
                let _ = cpu.set_default();
            }
        }
    }

    // Single in-process load: typed config + owned weights + tokenizer.
    let (config, weights, tokenizer) =
        load::load(model_dir).map_err(|e| format!("mlx load failed: {e}"))?;
    let cfg_json = std::fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let arch = Qwen3Config::from_json(&cfg_json).map_err(|e| format!("Qwen3 config: {e}"))?;
    let model = Qwen3::from_weights(arch, weights).map_err(|e| format!("Qwen3 build: {e}"))?;
    Ok(MlxModel {
        ctx: ModelContext::new(Box::new(model), tokenizer, config),
    })
}

/// The chat messages handed to the model: the system prompt as the system turn and the raw
/// transcript as the user turn. Rendered through the model's own chat template so instruct
/// models (e.g. Qwen3) see the format they were trained on.
fn build_messages(system_prompt: &str, raw: &str) -> serde_json::Value {
    // One worked example: without it small Qwen3 models tend to echo the transcript unchanged.
    serde_json::json!([
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "Transcript:\nso i talked to john yesterday and he said the the launch is moved to march which honestly i dont love" },
        { "role": "assistant", "content": "So I talked to John yesterday, and he said the launch is moved to March, which, honestly, I don't love." },
        { "role": "user", "content": format!("Transcript:\n{raw}") },
    ])
}

/// Output budget for a transcript of `input_tokens`: cleaned text is never much longer than
/// what was spoken, so cap generation near the input length instead of a fixed large ceiling.
pub fn token_budget(input_tokens: usize, max_tokens: usize) -> usize {
    (input_tokens + input_tokens / 2 + 16).min(max_tokens).max(1)
}

impl MlxModel {
    /// Generate cleaned prose for `raw` under `system_prompt`, bounded to `max_tokens` and to a
    /// wall-clock `deadline`.
    ///
    /// Thinking is disabled in the chat template (reasoning tokens would only add latency), and
    /// tokens are streamed so the deadline is checked after every step: a slow or stuck model
    /// returns `Err` and the pipeline falls back instead of hanging dictation.
    pub fn generate(
        &self,
        system_prompt: &str,
        raw: &str,
        max_tokens: usize,
        deadline: Duration,
    ) -> Result<String, String> {
        let start = Instant::now();
        let raw = raw.trim();
        let messages = build_messages(system_prompt, raw);
        let extra = serde_json::json!({ "enable_thinking": false });
        let tokens = self
            .ctx
            .apply_chat_template_ids(&messages, None, true, false, Some(&extra))
            .map_err(|e| format!("chat template failed: {e}"))?;
        let debug = std::env::var_os("ZIPSCRIBE_MLX_DEBUG").is_some();
        if debug {
            let prompt = self.ctx.decode(&tokens, false).unwrap_or_default();
            eprintln!("[mlx-debug] prompt:\n{prompt}");
        }
        let input_tokens = self.ctx.encode(raw, false).map(|t| t.len()).unwrap_or(max_tokens);
        let cfg = GenConfig::default().with_max_tokens(token_budget(input_tokens, max_tokens));
        let mut out = String::new();
        for step in self.ctx.stream_generate(&tokens, cfg) {
            let step = step.map_err(|e| format!("generate failed: {e}"))?;
            out.push_str(&step.text);
            if start.elapsed() > deadline {
                return Err(format!(
                    "timed out after {:.1}s ({} tokens)",
                    start.elapsed().as_secs_f32(),
                    step.generation_tokens
                ));
            }
        }
        if debug {
            eprintln!("[mlx-debug] raw output: {out:?}");
        }
        log::info!("[mlx] generated in {:.2}s", start.elapsed().as_secs_f32());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{build_messages, token_budget};

    #[test]
    fn messages_carry_system_and_raw() {
        let m = build_messages("CLEAN ONLY", "um hello there");
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[0]["content"], "CLEAN ONLY");
        assert_eq!(m[3]["role"], "user");
        assert!(m[3]["content"].as_str().unwrap().contains("um hello there"));
    }

    #[test]
    fn token_budget_scales_with_input_and_respects_cap() {
        assert_eq!(token_budget(20, 256), 46);
        assert_eq!(token_budget(1000, 256), 256);
        assert!(token_budget(0, 256) >= 1);
    }
}

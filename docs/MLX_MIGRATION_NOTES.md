# MLX migration — feasibility proof + open decisions

**STATUS: COMPLETED.** The zipscribe text-clean-up step now runs on-device, in-process Apple MLX
(inference via the `mlx-rs` binding), with the former Ollama HTTP daemon fully removed. See
`docs/EVIDENCE.md` for the end-to-end proof and `README.md` / `docs/SPEC.md` for the current design.

## task-1 feasibility gate: PASS (proven)
On this Apple-Silicon Mac, a native Rust MLX binding loads an MLX-format LLM
(safetensors + tokenizer) and generates NON-EMPTY clean text, IN-PROCESS, with NO
external daemon/helper process. This gate unblocked the migration.

- Binding: `mlxrs` v0.2.0 + `mlxrs-sys` v0.2.0 (crates.io; `mlxrs-sys` compiles Apple's
  MLX-C C++ sources via CMake), originally vendored with a `cargo patch` (since removed; see below).
- Model proven: `mlx-community/Qwen3-0.6B-4bit` (335 MB, safetensors + tokenizer, MLX format).
  Downloaded into /tmp/mlxmodel for the proof.
- Proof binary: /tmp/mlxproof (cargo project). Generation output (non-empty, coherent):
  "So, the first thing that comes to my mind is that there's a lot of things that are
  happening in the world right now, like, the economy, the environment, the social
  issues, the tech..."
- Proof build time: ~59s to compile MLX; generation ~150s for 48 tokens ON CPU.
- No external process is spawned. Generation is in `ctx.generate(prompt, GenConfig)`.

### mlxrs API surface (v0.2.0) for the in-project module
- `mlxrs::lm::load::load(dir: &Path) -> Result<(Config, Weights, Tokenizer)>`
  (reads config.json + weights + tokenizer; needs features safetensors, tokenizer-chat, lm).
- Model build (qwen3): `Qwen3Config::from_json(cfg_json)` then `Qwen3::from_weights(&cfg, weights)`
  → boxed into `Box<dyn mlxrs::lm::BaseModel>`; `ModelContext::new(model, tokenizer, config)`.
- Generation: `ctx.encode(text, /*bos*/true)` → `ctx.generate(prompt, GenConfig::default()
  .with_max_tokens(n))` → `(String, GenerationStats)`.
- Chat template: `ctx.apply_template("user", msg, /*add_gen*/true)` or
  `tokenizer.encode(&[Prompt{role,content}], false)`.
- CPU vs Metal: `mlxrs::Device::cpu().set_default()` + `mlxrs::stream::set_default_stream()`.

## DECISION — RESOLVED: Metal only (revised 2026-10-28)
The first cut was a hybrid Metal/CPU build so the project compiled with only the Xcode Command
Line Tools, using patched copies of `mlxrs`/`mlxrs-sys` vendored under `src-tauri/mlx-vendor/`.
CPU-only clean-up proved far too slow to be usable, so the app is now Metal-only: it uses the
stock crates.io `mlxrs` (Metal on by default), requires full Xcode with the Metal toolchain to
build, and bundles `mlx.metallib` into the app. The vendored copies are gone; macOS 14 is pinned
via `src-tauri/.cargo/config.toml`. On any MLX failure the cleaner still falls back to the
deterministic `light_cleanse`, so dictation never blocks.

## Migration plan — COMPLETED (ordered tasks in the goal)
1. DONE — in-process MLX generation core (`src/mlx.rs`).
2. DONE — model catalog + HF download: `src/cleanup_models.rs` presets (below); HF repo id/link
   download into `~/Library/Application Support/Zipscribe/models/cleanup/<id>/` mirroring the whisper
   `models::download` flow (progress + cancel + byte-count) via `cleanup_install`.
3. DONE — config: dropped the previous `ollama_url` + `cleanup_mode`; added active-MLX-model identity
   (`cleanup_model`) + install path (`cleanup_model_path`); kept deterministic `light_cleanse`. (The
   references to the old `ollama_url`/`cleanup_mode` are the fields that were REMOVED.)
4. DONE — service wiring: MLX cleaner in `service.rs` + `main.rs`; reload on model change; graceful
   fallback to `light_cleanse` on any load/generate failure or timeout.
5. DONE — Ollama fully removed: the previous `ollama_url` config field, the `ollama_models` Tauri
   command, and the HTTP `/api/chat` clean path are all gone; `grep -rin ollama src-tauri/src` is empty.
6. DONE — UI: setup installs a chosen preset (smallest default); settings selects the active model and
   adds a model from an arbitrary HuggingFace repo id/link.
7. DONE — verify + evidence: `cargo build`/`cargo test` clean (34 tests); `grep -rin ollama
   src-tauri/src` empty; in-process MLX load+generate proof in `docs/EVIDENCE.md`.

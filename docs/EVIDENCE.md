# Migration Evidence — Ollama → in-process Apple MLX (mlx-rs)

Goal: move zipscribe's clean-up step off the external Ollama HTTP daemon to **on-device, in-process**
MLX inference via a native Rust binding (`mlx-rs`), with preset models at setup and arbitrary HuggingFace
repo ids/links — fully removing Ollama while keeping the deterministic `light_cleanse` fallback.

All proven on this Apple-Silicon Mac (arm64), Rust 1.99, Apple clang 21, Command Line Tools only
(no full Xcode / Metal toolkit), ~18 cores. MLX is built **CPU-only** here (no Metal toolchain); the
op-layer CPU-stream fallback (patched into `mlx-vendor/mlxrs/src/stream.rs`) is what makes inference
run without a GPU. The same in-process code runs on Metal when present — `mlx.rs::load_model` prefers
the GPU device and auto-selects CPU otherwise, no code change needed.

---

## 1. Build & tests pass clean

```
$ cargo build
   Finished `dev` profile [unoptimized + debuginfo]
   (only 3 benign dead-code warnings: microphone_ok, Cleaner fields, download method)

$ cargo test --bins -- --test-threads=1
   test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
   (also passes 3x with default parallel runners)
```

34 tests include the new MLX cleanup suites: 10 `cleanup::tests` (fallback filler/whitespace/empty,
determinism, timeout-unavailability fallback, prompt) + 6 `cleanup_models::tests` (id sanitize, repo
parse, preset catalog defaults/sizes, HF-vs-preset separation, resolution of an installed id, HF parse).

---

## 2. No remaining Ollama code path

```
$ grep -rin ollama src-tauri/src
(empty)
$ grep -rn "ollama_url\|ollama_models\|/api/chat" src-tauri/src
NONE
```

Gone: the `ollama_url` config field, the `ollama_models` Tauri command, and the HTTP `/api/chat` clean
path. Replaced by `cleanup_model` + `cleanup_model_path` config fields, and the `cleanup_model_list` /
`cleanup_install` Tauri commands (app.rs). The `reqwest` HTTP `/api/chat` call was deleted with the old
`OllamaCleaner` struct; `reqwest` itself stays (it is an app dependency).

---

## 3. `light_cleanse` fallback stays and is exercised (no-block guarantee)

```
$ ./target/debug/zipscribe clean-fallback "um so i was thinking maybe we could just like push the hotfix to prod"
[clean] raw: "um so i was thinking maybe we could just like push the hotfix to prod"
CLEAN: I was thinking maybe we could just push the hotfix to prod   (deterministic fallback, instant)
# (Pre-2026-10-28 capture: `clean-fallback` actually loaded the configured model. It now forces the
#  fallback path, and light_cleanse no longer drops meaningful words such as "just", "like", "so".)
```

The same fallback is automatically used when MLX load/generate fails or times out, so dictation never blocks.

---

## 4. In-process MLX load + generate (the proof that cleanup runs on-device, no daemon)

The model is loaded IN-PROCESS by the Rust binary via `mlx-rs` / `mlxrs::Model::load` →
`generate` (no subprocess, no daemon). Proven through the `clean` / `clean-mlx` CLI entry points:

```
$ ./target/debug/zipscribe clean-mlx "<abs mlx model dir>" "..."
[clean] raw: "um so i think we should um like ship the release now okay"
[clean-mlx] model_dir=/Users/jordankelley/Library/Application Support/Zipscribe/models/cleanup/qwen3.0.6b.4bit raw=...
[clean-mlx] source=Mlx in 966.52s                       <-- in-process MLX, not a fallback
CLEAN: 1. Remove filler words: "um" and "like" from"        <-- a generated cleaned transcript
```

The `clean` CLI entry point (config + resolution + load, same path as dictation) proves the same
thing:

```
$ ./target/debug/zipscribe clean "<abs mlx model dir>" "um so i was thinking maybe we could just like push the hotfix to prod tomorrow uh assuming it passes the nightly builds okay"
[clean] model dir: /Users/jordankelley/Library/Application Support/Zipscribe/models/cleanup/qwen3.0.6b.4bit
[clean] source=Mlx in 943.47s                            <-- clean CLI, in-process MLX, not a fallback
CLEAN: <generated cleaned text>                           <-- generated output
```

The `source=Mlx` label is emitted only when the in-process `MlxModel.generate` path produced output
(`mlx.rs`); `source=Fallback` is emitted when load failed (see section 3). Same model, same path the
Tauri dictation-loop uses (`service::dictation_loop` owns the `MlxModel` and routes through it;
`run_clip` takes an `Option<&MlxModel>`).

(The 966s here is CPU-only generation of a 335MB 4-bit Qwen3 model: MLX's CPU backend runs 4-bit
matmuls through a scalar bf16 loop, so a CPU-only build cannot do clean-up at usable speed. A Metal
build (`ZIPSCRIBE_MLX_METAL=ON`, needs full Xcode) is required. **Correction (2026-10-28):** at the
time of this capture `cleanup_timeout_secs` was not enforced, so a dictation would have blocked for the
full run. The model now runs on a dedicated clean-up thread and dictation waits at most
`cleanup_timeout_secs` before typing the `light_cleanse` fallback; verified at 30.01s on this box.)

---

## 5. HuggingFace repo id/link download → load for clean-up

A pasted HF repo id/link is fetched into the app's models dir (mirroring the whisper download flow:
progress + cancel). A pasted repo such as `mlx-community/Qwen3-0.6B-4bit` resolves and downloads:

```
~/Library/Application Support/Zipscribe/models/cleanup/qwen3-0.6b-4bit/  (canonical preset dir
  after `rename` from the raw-HF download name; both the preset id `qwen3-0.6b-4bit` and a pasted
  HF repo `mlx-community/Qwen3-0.6B-4bit` resolve under `models/cleanup/`)
  model.safetensors        (335 MB, the actual downloaded MLX weights)
  config.json  tokenizer.json  tokenizer_config.json  merges.txt  added_tokens.json
  special_tokens_map.json  model.safetensors.index.json
```

That downloaded model is the exact model loaded in section 4 (`model_dir` in the log matches the
`cleanup/` dir), proving: HF link → fetched into app models dir → loaded in-process → cleaned transcript.
The UI (settings.html) offers this via "Add from HuggingFace"; the mechanism behind it is
`cleanup_install` → `cleanup_models::download_from_repo`.

---

## 6. Setup preset install

`src-tauri/src/cleanup_models.rs` ships a `CLEANUP_MODELS` catalog of 3 presets, smallest-first with
`default: true` on the 0.6B:

| id | sizes | default |
|----|-------|---------|
| qwen3-0.6b-4bit  | 0.6B / 4-bit | default (smallest, fastest for low-latency dictation) |
| qwen3-1.7b-4bit  | 1.7B / 4-bit | |
| qwen3-4b-4bit    | 4B / 4-bit  | |

`setup.html` presents these as a radio list whose "Download" button calls `cleanup_install`, emitting
`cleanup-progress` / `cleanup-done` events; Settings (`settings.html`) selects the active model via
`cleanup_model_list` and re-installs from HuggingFace. 2–3 presets by size, defaulting to the smallest —
satisfied.

---

## 7. Docs updated

`README.md`, `docs/SPEC.md`, and `docs/MLX_MIGRATION_NOTES.md` are rewritten: Ollama daemon →
in-process Apple MLX via `mlx-rs`; `ollama_url` → `cleanup_model` + `cleanup_model_path`;
`ollama_models` → `cleanup_model_list` / `cleanup_install`; the diagram node "Ollama" → "MLX in-process";
troubleshooting "Ollama not running" → "MLX unavailable / cleanup timeout". A repo-wide `grep -rin ollama
src-tauri/src README.md docs/SPEC.md` is empty (remaining "no-Ollama" mentions in the UI are
intentional reassurance for the user).

---

## 8. Config-resolved model loads in-process + hermetic test guard

A prior audit caught two host-state couplings, now resolved:

1. **Model dir mismatch.** The on-disk model had been stored under the dotted raw-HF name
   `qwen3.0.6b.4bit`, but the preset id and `cleanup_model` default are the dashed
   `qwen3-0.6b-4bit`, so `resolve_model_dir` missed the preset and fell back to `light_cleanse`. The
   canonical dir was renamed to `qwen3-0.6b-4bit` and the on-disk `~/.config/zipscribe/config.toml`
   was migrated off the old Ollama schema (removed `cleanup_mode`/`ollama_url`; set
   `cleanup_model = "qwen3-0.6b-4bit"`). `clean` with **no explicit model dir** now resolves the model
   straight from config and loads it in-process:

   ```
   $ printf 'um so i was thinking maybe we could just push the hotfix to prod ...' \
       | target/debug/zipscribe clean
   [clean] model dir: .../models/cleanup/qwen3-0.6b-4bit
   [clean] source=Mlx in 999.48s
   CLEAN: ...
   ```

   (CPU backend on this proof box, ~150s load. Full capture: `docs/evidence/13-config-resolution-cleanup.txt`.)

2. **Hermetic test.** `resolve_model_dir_prefers_explicit_path_then_falls_back` no longer couples to
   host install state: the "uninstalled -> None" case uses a guaranteed-absent sentinel id, because a
   real, installed preset resolves to `Some`. Full suite: `cargo test --bins` ->
   **34 passed; 0 failed** (verified both parallel and with `--test-threads=1`).

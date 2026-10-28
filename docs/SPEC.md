# zipscribe — Design Spec (v1)

> Local, on-device, macOS dictation app in Rust. On Push-to-Talk, transcribe held
> audio with in-process `openai/whisper-medium.en` (via whisper-rs / whisper.cpp),
> clean it with a configurable **in-process Apple MLX** LLM (a small language model loaded
> and run inside the binary, no external daemon), and type the cleansed text into the
> currently focused window. Runs at startup as a LaunchAgent.
>
> **Note on research:** At spec time, live web research against the Zipscribe repo
> was unavailable (web-research tools were restricted then). This spec is derived from
> the well-known Zipscribe design pattern plus the user's confirmed v1 choices, and
> is the source of truth for implementation. Cross-checking against the live repo is a
> v2 task if desired.

---

## 1. Goals & scope

### In scope (v1)
- Platform: macOS only (Apple Silicon M5 Pro).
- Activation: **Push-to-Talk (P2T)** only — hold a hotkey to record, release to process.
- STT engine: `openai/whisper-medium.en` loaded **in-process** via `whisper-rs`
  (whisper.cpp bindings), kept resident.
- Cleanup: a single configurable **in-process Apple MLX** LLM, defaulting to the smallest
  preset; the user can switch the model or add any Qwen3-based MLX model from Hugging Face.
- Output: **type** the cleansed text into the currently focused, non-app window via
  key simulation.
- UI: a macOS **menu-bar / tray** indicator with enable/disable toggle + config access.
- Startup: register as a **LaunchAgent / Login Item** that starts on login.
- **Fully local / on-device.** Nothing leaves the machine: STT and clean-up both run
  in-process on this Mac (Whisper + MLX).

### Out of scope (v2 — do NOT start)
- Continuous voice-activity/silence auto-mode.
- Wake-word trigger.
- Windows / Linux builds.
- Full preferences GUI window (beyond the tray menu).
- Cloud fallback / remote STT.
- Non-English / multilingual models (v1 = `whisper-medium.en`).
- Speaker diarization.
- A transcript history / undo UI.

### Non-goals we must not break
- Do not modify `~/AI-Projects/investment-calculator` or other global user config
  outside this project and its own LaunchAgent.
- All implementation remains **Rust**.

---

## 2. Core pipeline (the Zipscribe logic we need)

```
            ┌──────────────┐
 P2T key ──▶│  1. Capture   │──► raw PCM audio (16 kHz, mono, f32/i16)
 (hold)     └──────────────┘
            ┌──────────────┐
 P2T key ──▶│  2. Transcribe│──► raw transcript (Whisper)
 (release)  └──────────────┘
            ┌──────────────┐
            │  3. Clean     │──► cleansed text (in-process MLX LLM, or local fallback)
            └──────────────┘
            ┌──────────────┐
            │  4. Type      │──► input injected into the focused window
            └──────────────┘
```

### 2.1 Capture
- On **P2T key-down**: start capturing the default macOS microphone input into an
  in-memory audio buffer.
- Sample format: **16 kHz, mono, 16-bit PCM** (Whisper's native requirement).
- On **P2T key-up**: stop capturing, freeze the buffer, hand off to transcribe.
- Capture must be cancellation-safe: releasing the key mid-capture must not deadlock.
- Latency goal: capture overhead ~0 (the "cost" is recording while the user talks,
  which is invisible). No buffering delay introduced by a VAD in v1.

### 2.2 Transcribe (Whisper)
- Engine: `openai/whisper-medium.en` GGML, loaded via `whisper-rs` (wraps
  whisper.cpp). Loaded **once at startup and kept resident** in memory to avoid
  reload latency on every utterance.
- Input: the captured PCM buffer (downsample/converted to 16 kHz mono if needed).
- Output: a raw transcript string.
- Settings: single-segment, greedy/beam small, English (`.en`), no timestamps
  (dictation, not meeting transcription), no prompt/history carried in v1.
- If transcription fails or yields empty/whitespace: do **not** type anything,
  log a warning, and return to idle. Never type a blank line.

### 2.3 Clean (in-process MLX LLM)
- Load the configured MLX-format model (safetensors + tokenizer) **in-process** via the
  `mlx-rs` binding and generate the cleaned transcript there — no external daemon and no
  network at generation time. The model is a small/medium language model (default: the
  smallest preset; the user can switch, or add an MLX-format model from Hugging Face by
  repo id or link). It is loaded **once and kept resident** so repeat dictations stay fast.
- **Cleanup rules the prompt must enforce:**
  1. Remove filler / disfluency words ("um", "uh", "like", "you know", "actually",
     "basically", repeated self-corrections, trailing "and...").
  2. Fix grammar and spelling.
  3. Add/repair **punctuation and capitalization** suitable for typing into a field.
  4. **Do NOT fabricate content.** Preserve the speaker's intended meaning; do not add
     new facts, opinions, or length the user did not speak.
  5. Preserve proper nouns / entities faithfully.
  6. Do not wrap output in quotes, markdown fences, or commentary — return plain text
     only.
  7. Trim surrounding whitespace; do not add a trailing newline unless the user
     clearly ended a sentence that warrants it (configurable later).
- **Graceful fallback (no-block guarantee):** if MLX inference fails, is unavailable, or
  times out, fall back to a lightweight **local** cleanup (filler-word strip + minimal
  punctuation/ capitalization, no fabrication) and type that, rather than blocking or
  typing raw "um/uh" speech. Log that fallback was used.
- **Special commands (v1 minimal, optional):** recognize phrases like "insert time",
  "insert date", "new line", "stop dictation" as literal command expansions rather
  than transcribed text. Keep minimal and documented.

### 2.4 Output (type into focused window)
- **Key simulation** into the currently focused application (CGEvent / `enigo`-style
  on macOS). The typed text is the cleansed output.
- **Restore focus:** P2T must capture which window/app was focused *before* the user
  grabbed the hotkey, and the typed text goes there (the tray menu / app itself must
  not be the target).
- **Accessibility permission** is required to inject keystrokes. Detect + guide the
  user if missing (open System Settings, or use the standard `AXIsProcessTrusted`
  prompt). Degrade gracefully — never crash.
- **Type speed / batching:** use a bulk text-injection path (e.g. insert an entire
  string at once where the platform supports it) to minimize per-character latency;
  fall back to per-character only if necessary.
- Trailing behavior: type exactly the cleansed text; do not auto-press "enter" by
  default (configurable flag for v2).

### 2.5 Latency target
- **End-to-end (release -> typed) target: a few seconds or less** for typical utterances
  (say 3–10 words). Driven by: resident whisper model, resident small cleanup model,
  worker-thread processing, minimal I/O, bulk text injection.
- Measure and record actual latencies in the repo during the integration step.
- While the user holds the key, capture is instantaneous; work happens after release.

### 2.6 Startup
- Register as a **LaunchAgent** (`~/Library/LaunchAgents/<bundle>.plist`) so the app
  starts on login. Provide an **install / uninstall** command (launchctl load/unload,
  or `osascript` login-item toggle).
- v1 form factor: tray app + background service; headless (no separate window).

### 2.7 Permissions
- **Microphone** (input capture) — detected + guided; degrade gracefully if denied.
- **Accessibility** (key injection) — detected + guided; degrade gracefully if denied.
- App should clearly surface the missing-permission state in the tray instead of
  crashing silently.

---

## 3. Config schema (config file, e.g. `~/.config/zipscribe/config.toml`)
```toml
[general]
enabled = true                 # master enable/disable toggle
hotkey = "ctrl+shift+space"    # P2T key chord (hold)
output_mode = "type"           # "type" (v1 only; "clipboard" is a v2 option)
type_trailing_newline = false

[stt]
model = "whisper-medium.en"    # GGML path resolved relative to model dir
model_path = ""                # optional override path
threads = 8                    # whisper worker threads

[cleanup]
enabled = true
model = "qwen3-0.6b-4bit"      # preset id, installed path, or HF repo; default = smallest preset
model_path = ""                # optional explicit override to a downloaded .mlx dir
timeout_secs = 30
system_prompt = "..."          # the filler-removal / no-fabrication prompt

[permissions]
# flags surfaced from checks; not user-edited except "open_mic" / "open_a11y" toggles
```
- Missing/unknown fields fall back to documented defaults. Round-trip tested.
- Hotkey, cleanup `model`, `output_mode`, and `enabled` are the user-facing knobs that
  also appear in the tray menu / config access.

---

## 4. Module layout (implementation guide, step 2 onward)
```
zipscribe/
  Cargo.toml             # workspace root
  rust-toolchain.toml    # pinned
  src/
    main.rs              # app entry, tray lifecycle, orchestration
    config.rs            # schema (step 2 / scaffold-config)
    capture.rs           # microphone capture via cpal (step 3)
    hotkey.rs            # global P2T detection (step 3)
    transcribe.rs        # whisper-rs wrapper, resident model
    cleanup.rs           # in-process MLX clean-up client + local fallback
    cleanup_models.rs    # clean-up model catalog (presets) + HF-repo download
    mlx.rs               # mlx-rs binding: load + generate the resident MLX model
    output.rs            # key-sim typing + permission checks
    tray.rs              # menu-bar UI + enable/disable + config access
    launch.rs            # LaunchAgent install/uninstall helpers
  models/                # whisper-medium.en GGML (downloaded/generated, not committed)
  README.md              # build, install-as-login-item, permissions, config
  docs/SPEC.md           # this file
  docs/latency.txt       # measured latencies (integration step)
```

### Suggested Rust crates (decide/lock in step 2; keep light)
- Audio capture: `cpal` (CoreAudio backend).
- Global hotkey: `global-hotkey` (or `enigo`/`cocoa` for raw key events).
- Key simulation (typing into focused app): `enigo` and/or CoreGraphics `CGEvent`.
- Whisper: `whisper-rs` (whisper.cpp). Need `whisper-medium.en` GGML model file.
- In-process LLM: `mlx-rs` (Apple MLX via Rust) for clean-up; no HTTP call at runtime.
- Config: `serde` + `toml` + `dirs`/`dirs-next`.
- Tray / menu bar (lightweight, no heavy GUI framework): `tao` + `tray-icon`
  (or `tao`'s menu-bar). Avoid Tauri/tao-with-webview to keep the binary small.
- Threading: `crossbeam`/`std::thread` + channels for pipeline handoff.

---

## 5. Acceptance checklist (maps to success criteria)
- [ ] `cargo build --release` → `zipscribe` binary runs on the M5 Pro.
- [ ] P2T: hold hotkey → record mic → release → cleansed text types into the
      previously focused window with "um/uh" removed (demo in a text editor).
- [ ] STT is `whisper-medium.en`, loaded in-process + resident; raw transcript verified.
- [ ] Cleanup via configurable in-process MLX model; graceful local fallback when MLX unavailable.
- [ ] Tray indicator with enable/disable + config.
- [ ] LaunchAgent auto-starts on login (verified via `launchctl list`).
- [ ] Mic + Accessibility detected/guided; no crash when missing.
- [ ] README documents build / install / permissions / config.
- [ ] End-to-end latency measured and recorded.

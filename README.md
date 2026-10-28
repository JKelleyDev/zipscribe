# Zipscribe

A fast, **on-device** macOS menu-bar dictation app. Hold a hotkey, speak, release — within a
couple of seconds your words are typed into whatever window you're using. Your voice never leaves
your Mac.

- **Speech-to-text**: OpenAI Whisper (`whisper.cpp`, Metal GPU accelerated), run inside the app and
  kept loaded so repeat dictations are fast.
- **Clean-up (optional)**: a small language model, running **in-process via Apple MLX** and kept
  resident, removes "um"s, fixes grammar and punctuation, and keeps your meaning without inventing
  anything. If no clean-up model is installed (or inference fails), you still get a lightly
  tidied transcript — dictation is never blocked.
- **Output**: types into the focused window (default), or copies to the clipboard and pastes.
- **Menu bar**: a 🎙 icon with an on/off toggle and Settings. It shows 🔴 while listening and ⏳
  while working.

## Requirements

- A Mac with Apple Silicon (M1 or later) running macOS 14 (Sonoma) or newer
- About 150 MB – 1.6 GB of disk for the speech model (you choose the size during setup)
- *Optional:* a clean-up language model — pick and download a size in the app (on-device, Apple MLX)

## Install

### From a release (most people)

1. Download `Zipscribe.dmg` from the [Releases](https://github.com/JKelleyDev/zipscribe/releases) page and open it.
2. Drag **Zipscribe** onto **Applications**.
3. **First launch only:** the app isn't notarized by Apple, so macOS will refuse a normal
   double-click. In Applications, **right-click Zipscribe → Open → Open**. (Or run
   `xattr -cr /Applications/Zipscribe.app` once, then open it normally.)
4. A 🎙 appears in the menu bar and the setup guide opens.

### Setup guide

The guide walks through six steps:

1. **Welcome**
2. **Microphone** — click *Allow microphone* and approve the macOS prompt.
3. **Accessibility** — needed so Zipscribe can press keys to type for you. Click *Grant access*,
   then switch Zipscribe on in the list that opens. Or skip it and use clipboard mode instead.
4. **Speech model** — pick a size and download it (one time). Medium is the most accurate; Base is
   the fastest.
5. **Text clean-up** *(optional)* — see below.
6. **Hotkey** — the default is **Ctrl + Space**. Try it in the box.

Change anything later from **🎙 → Settings…**.

### Optional: AI clean-up (on-device, in-process)

A small language model runs inside Zipscribe via Apple MLX — no separate server and no internet
call at runtime. In the **Text clean-up** setup step pick a size and download it once:

- **Small (0.6B)** — fast, a little less polished. *The default.*
- **Medium (1.7B)** — more accurate clean-up, a bit slower.
- **Large (4B)** — the most accurate, the slowest to load.

You can also use your own model, either during setup (**Use your own model**) or in **Settings →
Text clean-up → Add a model from HuggingFace**: paste a repo id or link (e.g.
`mlx-community/Qwen3-8B-4bit`) and it is downloaded into your model folder with progress and cancel.
The model must be a **Qwen3-based MLX** model; other architectures are rejected before anything is
downloaded. Models you add can be removed again from Settings.

Skipping this is fine — Zipscribe falls back to a simple, deterministic clean-up automatically.

## Using it

Click into any text field, **hold the hotkey**, speak, and **release**. The text appears at your
cursor. Settings let you change the hotkey, switch between typing and clipboard-paste output, pick
the clean-up model, edit the clean-up instructions, and turn on **Start at login**.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Zipscribe can't be opened" / "damaged" on first launch | Right-click the app → **Open** → **Open**, or run `xattr -cr /Applications/Zipscribe.app`. |
| Nothing is typed, but no error | Accessibility isn't granted. System Settings → Privacy & Security → Accessibility → turn Zipscribe on. |
| Permissions look granted but don't work after updating | The app is ad-hoc signed, so macOS ties permissions to the exact build. In Privacy & Security, **remove** the old Zipscribe entry (the `−` button) from Accessibility and Microphone, then re-add or re-grant it. |
| Microphone step won't turn green | Quit and reopen Zipscribe. If it's still stuck, check System Settings → Privacy & Security → Microphone. |
| Text isn't being cleaned up | Open Settings → Text clean-up and download a model (the smallest is enough); the active model shows *ready on-device* once installed. If it still isn't cleaning, check the log. |
| "No speech model is loaded" | Finish the *Speech model* download in Settings. |
| Something else | Check the log at `~/Library/Logs/Zipscribe/zipscribe.log`. |

## Where things live

| What | Where |
|---|---|
| Config | `~/.config/zipscribe/config.toml` |
| Speech models | `~/Library/Application Support/Zipscribe/models/` |
| Log | `~/Library/Logs/Zipscribe/zipscribe.log` |
| Login item | `~/Library/LaunchAgents/com.jordankelley.zipscribe.plist` |

## Uninstall

Open **🎙 → Settings… → Help → Uninstall…**. Zipscribe quits, moves itself to the Trash, removes
the start-at-login item, and resets its microphone and Accessibility permissions. Two checkboxes
let you keep the downloaded speech model (up to 1.5 GB) or your settings if you plan to reinstall.

There is no separate clean-up service to remove — the model lives in your model folder and is
taken out by the *Delete the downloaded models* checkbox above, or by deleting that folder.

<details>
<summary>Uninstall by hand</summary>

```sh
rm -rf /Applications/Zipscribe.app \
       ~/.config/zipscribe \
       ~/Library/Application\ Support/Zipscribe \
       ~/Library/Logs/Zipscribe \
       ~/Library/LaunchAgents/com.jordankelley.zipscribe.plist
tccutil reset All com.jordankelley.zipscribe
```

</details>

## Build from source

You need Apple Silicon, **full Xcode** (from the App Store) with its Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`), CMake (`brew install cmake`), [Rust](https://rustup.rs) (the pinned toolchain installs itself via
`rust-toolchain.toml`), and Node.js (only for the Tauri CLI).

```sh
git clone https://github.com/JKelleyDev/zipscribe.git && cd zipscribe
npm install
./scripts/make-dmg.sh   # builds the app, then -> src-tauri/target/release/bundle/Zipscribe.dmg
```

The clean-up model runs on the GPU through MLX's Metal backend, which is why full Xcode is required;
there is no CPU-only build.

The first build compiles whisper.cpp and takes several minutes; later builds are quick. Then open
the DMG and follow **Install** above.

### Project layout

| Path | Contents |
|---|---|
| `src-tauri/src/` | The app, in Rust (capture, whisper, hotkey, cleanup, output, tray) |
| `ui/` | The setup guide and Settings screens (plain HTML/CSS/JS bundled into the app) |
| `scripts/make-dmg.sh` | Packages the built app into a drag-to-Applications DMG |
| `docs/` | Design notes (`SPEC.md`) and captured test evidence |

### Debugging commands

The binary also has subcommands, run as
`/Applications/Zipscribe.app/Contents/MacOS/zipscribe <command>`:

| Command | What it does |
|---|---|
| *(none)* | Start the menu-bar app |
| `record-secs <n>` | Record `n` seconds from the mic to a WAV |
| `p2t` | Log each hotkey press and release |
| `transcribe <file.wav>...` | Transcribe audio files |
| `clean [model] <text…>` | Clean text via the in-process MLX model (falls back on failure; `model` is a dir, preset id, or HF repo) |
| `clean-fallback "" <text>` | Force the deterministic no-LLM clean-up path |

Set `ZIPSCRIBE_MLX_DEBUG=1` when running `clean` to print the exact prompt and raw model output.
| `perms` | Report Microphone and Accessibility status |
| `type-demo [text]` | Demonstrate typed output |
| `install` / `uninstall` | Add or remove the login item |

## How it works

```
hold hotkey  ──>  16 kHz mono mic capture
release      ──>  Whisper (resident in memory)      ──>  raw transcript
                  MLX model (or simple fallback) ──>  cleaned prose
                  key simulation (or ⌘V paste)      ──>  typed into the focused window
```

With the model kept in memory, a warm dictation takes roughly one second end to end; the first one
after launch can take a few seconds while models warm up. Design rationale is in
[`docs/SPEC.md`](docs/SPEC.md) and test captures are in `docs/evidence/`.

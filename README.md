# Zipscribe

A fast, **on-device** macOS menu-bar dictation app. Hold a hotkey, speak, release — within a
couple of seconds your words are typed into whatever window you're using. Your voice never leaves
your Mac.

- **Speech-to-text**: OpenAI Whisper (`whisper.cpp`, Metal GPU accelerated), run inside the app and
  kept loaded so repeat dictations are fast.
- **Clean-up (optional)**: a local [Ollama](https://ollama.com) model removes "um"s, fixes grammar
  and punctuation, and keeps your meaning without inventing anything. If Ollama isn't running, you
  still get a lightly tidied transcript.
- **Output**: types into the focused window (default), or copies to the clipboard and pastes.
- **Menu bar**: a 🎙 icon with an on/off toggle and Settings. It shows 🔴 while listening and ⏳
  while working.

## Requirements

- A Mac with Apple Silicon (M1 or later) running macOS 12 or newer
- About 150 MB – 1.6 GB of disk for the speech model (you choose the size during setup)
- *Optional:* [Ollama](https://ollama.com), if you want AI clean-up

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

### Optional: AI clean-up with Ollama

1. Install Ollama from [ollama.com](https://ollama.com) and make sure it's running.
2. Pull the default model:

   ```sh
   ollama pull qwen2.5:7b
   ```

   (A smaller model such as `qwen2.5:1.5b` is faster but less polished; choose it in
   Settings → Text clean-up.)

Skipping this is fine — Zipscribe falls back to simple clean-up automatically.

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
| Text isn't being cleaned up | Open Settings → Text clean-up. If it says *Ollama not detected*, start Ollama and confirm the model name matches one from `ollama list`. |
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

Ollama and its models are separate and are never touched; remove them with `ollama rm <model>` if
you want.

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

You need Apple Silicon, Xcode Command Line Tools (`xcode-select --install`), CMake
(`brew install cmake`), [Rust](https://rustup.rs) (the pinned toolchain installs itself via
`rust-toolchain.toml`), and Node.js (only for the Tauri CLI).

```sh
git clone https://github.com/JKelleyDev/zipscribe.git && cd zipscribe
npm install
npx tauri build              # -> src-tauri/target/release/bundle/macos/Zipscribe.app
./scripts/make-dmg.sh        # -> src-tauri/target/release/bundle/Zipscribe.dmg
```

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
| `clean <text> [model]` | Clean text via Ollama (falls back on failure) |
| `clean-fallback <text>` | Force the no-Ollama path |
| `perms` | Report Microphone and Accessibility status |
| `type-demo [text]` | Demonstrate typed output |
| `install` / `uninstall` | Add or remove the login item |

## How it works

```
hold hotkey  ──>  16 kHz mono mic capture
release      ──>  Whisper (resident in memory)      ──>  raw transcript
                  Ollama model (or simple fallback) ──>  cleaned prose
                  key simulation (or ⌘V paste)      ──>  typed into the focused window
```

With the model kept in memory, a warm dictation takes roughly one second end to end; the first one
after launch can take a few seconds while models warm up. Design rationale is in
[`docs/SPEC.md`](docs/SPEC.md) and test captures are in `docs/evidence/`.

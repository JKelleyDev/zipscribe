//! zipscribe — local, on-device macOS dictation app.
//!
//! Pipeline: Push-to-Talk → capture mic → transcribe (whisper-medium.en, in-process)
//! → clean (local Ollama LLM) → type cleansed text into the focused window.
//!
//! This module is the skeleton entry point. Modules are implemented in later steps:
//!   config   — schema + load/save  (step 2 / scaffold-config)
//!   capture  — microphone capture via cpal (step 3 / stt-capture)
//!   hotkey   — global Push-to-Talk detection (step 3 / stt-hotkey)
//!   transcribe — whisper-rs wrapper, resident model (step 3 / stt-whisper)
//!   cleanup  — Ollama client + local fallback (step 4 / cleanup)
//!   output   — key-sim typing + permission checks (step 5 / output)
//!   app      — Tauri menu-bar app, setup + settings windows
//!   service  — dictation worker loop
//!   launch   — LaunchAgent install/uninstall helpers (step 7 / startup)

mod capture;
mod config;
mod hotkey;
mod launch;
mod output;
mod app;
mod models;
mod service;
mod stt;
mod cleanup;

use std::process::ExitCode;

fn main() -> ExitCode {
    // Subcommand surface (v1 minimal): `install` / `uninstall` register the LaunchAgent;
    // anything else (or no args) launches the dictation service. Decided here so the
    // binary can be registered as a login item that, on launch, just runs the service.
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str);

    match cmd {
        Some("install") => match launch::install() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("install failed: {e}");
                ExitCode::FAILURE
            }
        },
        Some("uninstall") => match launch::uninstall() {
            Ok(msg) => {
                println!("{msg}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("uninstall failed: {e}");
                ExitCode::FAILURE
            }
        },
        Some("record-secs") => match args.get(2) {
            Some(n) => match n.parse() {
                Ok(secs) => match run_record(secs) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("record-secs failed: {e}");
                        ExitCode::FAILURE
                    }
                },
                Err(_) => {
                    eprintln!("usage: record-secs <seconds>");
                    ExitCode::FAILURE
                }
            },
            None => {
                eprintln!("usage: record-secs <seconds>");
                ExitCode::FAILURE
            }
        },
        Some("p2t") => match run_p2t_demo() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("p2t failed: {e}");
                ExitCode::FAILURE
            }
        },
        Some("transcribe") => match args.get(2) {
            Some(_pcm) => {
                // Multiple files prove residency: one model load, many transcriptions.
                let files: Vec<String> = args[2..].to_vec();
                match run_transcribe(&files) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("transcribe failed: {e}");
                        ExitCode::FAILURE
                    }
                }
            },
            None => {
                eprintln!("usage: transcribe <h1.pcm> [<h2.pcm ...] [model-path]");
                ExitCode::FAILURE
            }
        },
        Some("clean") => match run_clean(args.get(2).map(|s| s.as_str()), args.get(3).map(|s| s.as_str()), "http://localhost:11434") {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => { eprintln!("clean failed: {e}"); ExitCode::FAILURE }
        },
        Some("clean-fallback") => match run_clean(args.get(2).map(|s| s.as_str()), args.get(3).map(|s| s.as_str()), "http://127.0.0.1:99999") {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => { eprintln!("clean-fallback failed: {e}"); ExitCode::FAILURE }
        },
        Some("perms") => match run_perms() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => { eprintln!("perms check failed: {e}"); ExitCode::FAILURE }
        },
        Some("type-demo") => match run_type_demo(args.get(2).map(|s| s.as_str())) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => { eprintln!("type-demo failed: {e}"); ExitCode::FAILURE }
        },
        _ => app::run(),
    }
}

fn run_record(secs: f32) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Duration;
    let out = capture::record_seconds(Duration::from_secs_f32(secs))?;
    let dir = std::env::temp_dir().join(format!("zipscribe-record-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let wav = dir.join("clip.wav");
    let pcm = dir.join("clip.pcm");
    capture::write_wav(&wav, &out)?;
    capture::write_pcm(&pcm, &out)?;
    let level = if out.rms_db() == f32::NEG_INFINITY {
        "(silence)".to_string()
    } else {
        format!("{:.1} dB", out.rms_db())
    };
    eprintln!(
        "[record] {} samples @ {} Hz -> {:.2}s (rms {})",
        out.samples.len(),
        out.sample_rate,
        out.duration_secs(),
        level,
    );
    println!("wrote {} and {}", wav.display(), pcm.display());
    Ok(())
}

/// Demo: register the configured P2T hotkey and log each Press/Release with a wall-clock
/// timestamp. Proves key-down starts capture and key-up stops it. In production this loop
/// is infinite and each event feeds the capture -> transcribe -> clean -> type pipeline.
fn run_p2t_demo() -> Result<(), String> {
    let _ = env_logger::try_init();
    let cfg = config::Config::load_or_default();
    let p2t = hotkey::P2T::new(&cfg)?;
    eprintln!(
        "[p2t] holding \"{}\" records; release transcribes+cleans+types. idle exit after ~12s.",
        cfg.hotkey
    );
    let start = std::time::Instant::now();
    p2t.run(1200, |ev, _id| {
        match ev {
            hotkey::P2TEvent::Press => {
                eprintln!("[p2t] PRESSED at {:?} -> START capture", start.elapsed());
            }
            hotkey::P2TEvent::Release => {
                eprintln!("[p2t] RELEASED at {:?} -> STOP capture", start.elapsed());
            }
            hotkey::P2TEvent::None => {}
        }
        true
    });
    Ok(())
}

/// Transcribe a raw 16 kHz mono PCM (or WAV) file with the resident whisper model and print
/// the raw transcript + timing. Usage: `transcribe <pcm-or-wav> [model-path]`.
fn run_transcribe(files: &[std::string::String]) -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    // Model path: env WF_MODEL overrides, else the default in-repo model.
    let model = std::env::var("WF_MODEL").unwrap_or_else(|_| "models/ggml-medium.en.bin".into());
    eprintln!("[transcribe] loading model (resident): {model}");
    let t_load = std::time::Instant::now();
    let mut stt = stt::Transcriber::load(&model)?;
    eprintln!("[transcribe] model resident in {:.2}s", t_load.elapsed().as_secs_f32());

    for audio_path in files {
        // Accept raw .pcm (16k mono i16) or a .wav we produced earlier.
        let pcm = if audio_path.ends_with(".wav") {
            capture::read_wav(audio_path)?
        } else {
            capture::read_pcm(audio_path)?
        };
        let secs = pcm.len() as f32 / 16000.0;
        eprintln!("[transcribe] {:?}: {:.2}s audio, {} i16 samples", audio_path, secs, pcm.len());
        let r = stt.transcribe(&pcm)?;
        println!("RAW: {}", r.text);
        eprintln!("[transcribe] inference took {:.2}s", r.elapsed.as_secs_f32());
    }
    eprintln!("[transcribe] transcribed {} files from one resident model", files.len());
    Ok(())
}

/// Cleanse a raw transcript with a local Ollama model (or fall back locally).  Prints the raw and
/// cleansed text side by side.  Usage: `clean [raw-text] [model]` or `clean-fallback`.
fn run_clean(
    raw_arg: Option<&str>,
    model_arg: Option<&str>,
    ollama_url: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let cfg = config::Config::load().unwrap_or_default();
    // Raw transcript: from arg, else read a line from stdin.
    let raw = match raw_arg {
        Some(s) => s.to_string(),
        None => {
            eprintln!("[clean] reading raw transcript from stdin...");
            let mut s = String::new();
            std::io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        }
    };
    let model = model_arg
        .unwrap_or(cfg.cleanup_model.as_str());
    eprintln!("[clean] raw: {raw:?}");
    eprintln!("[clean] ollama_url={ollama_url} model={model}");
    let cleaner = cleanup::Cleaner::new(ollama_url.to_string(), model.to_string());
    let t0 = std::time::Instant::now();
    let res = cleaner.clean(&raw)?;
    eprintln!("[clean] source={:?} in {:.2}s", res.source, t0.elapsed().as_secs_f32());
    println!("CLEAN: {}", res.text);
    Ok(())
}

/// Print current Microphone + Accessibility permission status and any guided hints.
fn run_perms() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = config::Config::load().unwrap_or_default();
    let report = output::check_permissions();
    println!("Microphone:    {}", if report.microphone { "granted" } else { "MISSING" });
    println!("Accessibility:  {}", if report.accessibility { "granted" } else { "MISSING" });
    let ok = report.ok_for(&cfg.output_mode);
    println!("Operational for output_mode = \"{}\":  {}", cfg.output_mode, if ok { "YES" } else { "NO" });
    if ok {
        return Ok(());
    }
    println!("\nGuidance (grant these in System Settings ▸ Privacy & Security):");
    for h in report.guidance(&cfg.output_mode) {
        println!("  {h}");
    }
    Ok(())
}

/// Demonstrate key-simulation into the currently focused window by typing a short string.
/// Usage: `type-demo [text]` (defaults to a sample sentence). This is how the service
/// injects cleansed prose into whatever app the user last focused.
fn run_type_demo(text_arg: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let _ = env_logger::try_init();
    let cfg = config::Config::load().unwrap_or_default();
    let text = text_arg.unwrap_or("Hello from zipscribe!").to_string();
    eprintln!("[type-demo] output_mode={} — typing into the focused window: {text:?}", cfg.output_mode);
    match output::deliver(&text, &cfg) {
        Ok(()) => {
            eprintln!("[type-demo] done");
            Ok(())
        }
        Err(output::OutputError::Permission(msg)) => {
            eprintln!("[type-demo] {}", msg);
            // Degrade gracefully: print guidance rather than crashing.
            for h in output::check_permissions().guidance(&cfg.output_mode) {
                eprintln!("  guidance: {h}");
            }
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

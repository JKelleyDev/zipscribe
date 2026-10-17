//! The dictation service: shared state, the push-to-talk state machine, and the worker loop that
//! captures audio, transcribes it, cleans it up and types it.
//!
//! The worker owns the resident Whisper model. Everything the settings UI can change flows through
//! `Shared`: bumping `version` makes the worker re-read the config, rebuild the cleaner and reload
//! the model if its path changed, all without restarting the app.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use crate::config::Config;

/// What the menu bar should show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayStatus {
    Idle,
    Recording,
    Processing,
    Disabled,
}

impl TrayStatus {
    pub fn title(self) -> &'static str {
        match self {
            TrayStatus::Idle => "🎙",
            TrayStatus::Recording => "🔴 Listening",
            TrayStatus::Processing => "⏳",
            TrayStatus::Disabled => "🎙⃠",
        }
    }
}

/// Called by the worker whenever the visible status changes.
pub type StatusSink = Box<dyn Fn(TrayStatus) + Send>;

/// State shared between the UI thread, commands and the dictation worker.
pub struct Shared {
    pub cfg: RwLock<Config>,
    /// Bumped whenever `cfg` changes; the worker reloads on change.
    pub version: AtomicU64,
    pub enabled: AtomicBool,
    pub hotkey_id: AtomicU32,
    pub model_ready: AtomicBool,
    pub model_error: Mutex<Option<String>>,
}

impl Shared {
    pub fn new(cfg: Config) -> Arc<Shared> {
        Arc::new(Shared {
            enabled: AtomicBool::new(cfg.enabled),
            cfg: RwLock::new(cfg),
            version: AtomicU64::new(1),
            hotkey_id: AtomicU32::new(0),
            model_ready: AtomicBool::new(false),
            model_error: Mutex::new(None),
        })
    }

    pub fn config(&self) -> Config {
        self.cfg.read().unwrap().clone()
    }

    pub fn set_config(&self, cfg: Config) {
        self.enabled.store(cfg.enabled, Ordering::Relaxed);
        *self.cfg.write().unwrap() = cfg;
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    /// Ask the worker to re-check the model file (e.g. after a download finished).
    pub fn poke(&self) {
        self.version.fetch_add(1, Ordering::SeqCst);
    }
}

/// The dictation state machine's phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Not currently holding the P2T hotkey.
    Idle,
    /// Holding the hotkey; microphone is being captured.
    Recording,
}

/// The decision the driver should take for a single P2T event under `enabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Do nothing (disabled, or no state change).
    Idle,
    /// Begin capturing the microphone.
    StartRecording,
    /// Stop capture and run transcribe -> clean -> deliver on the buffered audio.
    StopAndDictate,
}

/// Pure transition function. Given the current phase, the incoming hotkey event, and whether the
/// dictation service is enabled, return the decision and the resulting phase.
///
/// Disabling while recording *stops* capture (discarding the buffer) so that toggling off cleanly
/// halts dictation; re-enabling resumes from `Idle`.
pub fn decide(phase: Phase, event: crate::hotkey::P2TEvent, enabled: bool) -> (Decision, Phase) {
    use crate::hotkey::P2TEvent;
    match (phase, event, enabled) {
        (Phase::Idle, P2TEvent::Press, true) => (Decision::StartRecording, Phase::Recording),
        (Phase::Recording, P2TEvent::Release, true) => {
            (Decision::StopAndDictate, Phase::Idle)
        }
        // Disabled: ignore Press, and if we were mid-recording, stop (discard) it.
        (_, P2TEvent::Press, false) => (Decision::Idle, Phase::Idle),
        (Phase::Recording, _, false) => (Decision::Idle, Phase::Idle),
        // Any no-op / already-idle / no-event cases.
        _ => (Decision::Idle, phase),
    }
}

/// Run the full dictation pipeline for one buffered clip: transcribe -> clean -> deliver.
fn run_clip(
    transcriber: &mut crate::stt::Transcriber,
    cleaner: &crate::cleanup::Cleaner,
    cfg: &Config,
    pcm_i16: &[i16],
) {
    let t0 = std::time::Instant::now();
    let transcript = match transcriber.transcribe(pcm_i16) {
        Ok(t) => t,
        Err(e) => {
            log::error!("[service] transcription failed: {e:?}");
            return;
        }
    };
    log::info!(
        "[service] RAW ({:.2}s inference, {} samples): {:?}",
        transcript.elapsed.as_secs_f32(),
        transcript.sample_count,
        transcript.text
    );
    if transcript.text.trim().is_empty() {
        return;
    }

    let cleaned = if cfg.cleanup_enabled {
        match cleaner.clean(&transcript.text) {
            Ok(r) => {
                log::info!("[service] CLEAN source={:?} -> {:?}", r.source, r.text);
                r.text
            }
            Err(e) => {
                log::warn!("[service] cleanup failed, using raw: {e:?}");
                transcript.text.clone()
            }
        }
    } else {
        transcript.text.clone()
    };

    if let Err(e) = crate::output::deliver(&cleaned, cfg) {
        log::warn!("[service] deliver failed: {e}");
        for h in crate::output::check_permissions().guidance(&cfg.output_mode) {
            log::warn!("[service] permission: {}", h);
        }
    }
    log::info!("[service] cycle total {:.2}s: {:?}", t0.elapsed().as_secs_f32(), cleaned);
}

fn load_model(shared: &Shared, cfg: &Config) -> Option<(PathBuf, crate::stt::Transcriber)> {
    let path = crate::models::resolve_path(cfg);
    if !path.exists() {
        *shared.model_error.lock().unwrap() =
            Some(format!("speech model not installed ({})", path.display()));
        shared.model_ready.store(false, Ordering::Relaxed);
        return None;
    }
    match crate::stt::Transcriber::load(&path) {
        Ok(t) => {
            *shared.model_error.lock().unwrap() = None;
            shared.model_ready.store(true, Ordering::Relaxed);
            Some((path, t))
        }
        Err(e) => {
            log::error!("[service] could not load whisper model at {}: {e:?}", path.display());
            *shared.model_error.lock().unwrap() = Some(format!("could not load model: {e:?}"));
            shared.model_ready.store(false, Ordering::Relaxed);
            None
        }
    }
}

/// The dictation worker. Runs until `stop` is set.
pub fn dictation_loop(shared: Arc<Shared>, stop: Arc<AtomicBool>, status: StatusSink) {
    log::info!("[service] starting dictation loop");

    let mut seen_version = 0u64;
    let mut cfg = shared.config();
    let mut cleaner = crate::cleanup::Cleaner::new(cfg.ollama_url.clone(), cfg.cleanup_model.clone());
    let mut model: Option<(PathBuf, crate::stt::Transcriber)> = None;

    let mut phase = Phase::Idle;
    let mut handle: Option<crate::capture::CaptureHandle> = None;
    let idle_status = |enabled: bool| if enabled { TrayStatus::Idle } else { TrayStatus::Disabled };

    loop {
        if stop.load(Ordering::Relaxed) {
            log::info!("[service] stop requested; tearing down");
            break;
        }

        // Pick up config / model changes made from the UI.
        let version = shared.version.load(Ordering::SeqCst);
        if version != seen_version {
            seen_version = version;
            cfg = shared.config();
            cleaner = crate::cleanup::Cleaner::new(cfg.ollama_url.clone(), cfg.cleanup_model.clone());
            let want = crate::models::resolve_path(&cfg);
            if model.as_ref().map(|(p, _)| p != &want).unwrap_or(true) {
                status(TrayStatus::Processing);
                model = load_model(&shared, &cfg);
            }
            status(idle_status(shared.enabled.load(Ordering::Relaxed)));
        }

        let event = crate::hotkey::poll_id(shared.hotkey_id.load(Ordering::Relaxed));
        let enabled = shared.enabled.load(Ordering::Relaxed);
        let (decision, next_phase) = decide(phase, event, enabled);
        phase = next_phase;
        match decision {
            Decision::StartRecording => {
                if model.is_none() {
                    log::warn!("[service] hotkey pressed but no speech model is loaded");
                    phase = Phase::Idle;
                } else {
                    match crate::capture::CaptureHandle::start() {
                        Ok(h) => {
                            log::info!("[service] capturing microphone (P2T held)");
                            status(TrayStatus::Recording);
                            handle = Some(h);
                        }
                        Err(e) => {
                            log::error!("[service] mic capture start failed: {e}");
                            phase = Phase::Idle;
                        }
                    }
                }
            }
            Decision::StopAndDictate => {
                status(TrayStatus::Processing);
                if let (Some(h), Some((_, transcriber))) = (handle.take(), model.as_mut()) {
                    match h.stop() {
                        Ok(audio) => {
                            log::info!(
                                "[service] captured {:.2}s, {} samples, {} dBFS",
                                audio.duration_secs(),
                                audio.samples.len(),
                                audio.rms_db()
                            );
                            if audio.rms_db() < -60.0 {
                                log::info!("[service] clip too quiet; discarding");
                            } else {
                                run_clip(transcriber, &cleaner, &cfg, &audio.samples);
                            }
                        }
                        Err(e) => log::warn!("[service] capture stop/timeout: {e}"),
                    }
                }
                status(idle_status(enabled));
            }
            Decision::Idle => {
                // Disabled mid-recording: drop the in-flight capture.
                if phase == Phase::Idle && handle.take().is_some() {
                    status(TrayStatus::Disabled);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hotkey::P2TEvent;

    #[test]
    fn press_when_enabled_starts_recording() {
        let (d, p) = decide(Phase::Idle, P2TEvent::Press, true);
        assert_eq!(d, Decision::StartRecording);
        assert_eq!(p, Phase::Recording);
    }

    #[test]
    fn release_when_enabled_stops_and_dictates() {
        let (d, p) = decide(Phase::Recording, P2TEvent::Release, true);
        assert_eq!(d, Decision::StopAndDictate);
        assert_eq!(p, Phase::Idle);
    }

    #[test]
    fn press_when_disabled_does_nothing() {
        let (d, p) = decide(Phase::Idle, P2TEvent::Press, false);
        assert_eq!(d, Decision::Idle);
        assert_eq!(p, Phase::Idle);
    }

    #[test]
    fn disabled_mid_recording_stops_cleanly() {
        // Enabled start, then disabled release -> must NOT dictate, and resets to Idle.
        let (d0, p0) = decide(Phase::Idle, P2TEvent::Press, true);
        assert_eq!(p0, Phase::Recording);
        let (d1, p1) = decide(Phase::Recording, P2TEvent::Release, false);
        assert_eq!(d1, Decision::Idle);
        assert_eq!(p1, Phase::Idle);
    }

    #[test]
    fn no_event_keeps_phase_idle() {
        let (d, p) = decide(Phase::Idle, P2TEvent::None, true);
        assert_eq!(d, Decision::Idle);
        assert_eq!(p, Phase::Idle);
    }
}

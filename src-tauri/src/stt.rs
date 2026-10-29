//! In-process STT via `whisper-rs` (whisper.cpp).
//!
//! Loads the `GGML` Whisper model (`large-v3-turbo`) once and keeps the **context** (the heavy weights)
//! resident in memory, so each transcription pays only inference cost (no reload). whisper.cpp
//! runs each `full` on a fresh, lightweight *state* — that's the correct pattern: the model
//! stays resident, the per-call state is cheap to recreate. `transcribe` takes 16 kHz mono i16
//! PCM (exactly what `crate::capture` produces) and returns the raw transcript + timing.

use std::path::Path;
use std::time::Instant;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

/// A resident Whisper model. Load once at startup; call `transcribe` many times.
///
/// The underlying context is thread-confined (not `Send`), which is fine because the dictation
/// service runs it on a single worker thread.
pub struct Transcriber {
    ctx: WhisperContext,
}

/// Result of a transcription: the raw text plus how long inference took.
pub struct Transcript {
    pub text: String,
    pub elapsed: std::time::Duration,
    pub sample_count: usize,
}

/// Cap transcription thread count. Whisper on Apple Silicon is memory/cache-bound;
/// oversubscribing 18 cores can slow it down, so we cap conservatively.
fn thread_count() -> i32 {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    n.clamp(1, 8) as i32
}

impl Transcriber {
    /// Load the GGML model at `model_path` in-process. Blocks while the model loads; keep the
    /// returned `Transcriber` alive to keep the model resident in memory.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Transcriber, WhisperError> {
        let t0 = Instant::now();
        let ctx = WhisperContext::new_with_params(
            model_path.as_ref(),
            WhisperContextParameters::default(),
        )?;
        log::info!(
            "[stt] model loaded: {} ({:.2}s)",
            model_path.as_ref().display(),
            t0.elapsed().as_secs_f32()
        );
        Ok(Transcriber { ctx })
    }

    /// Transcribe 16 kHz mono i16 PCM. The model context stays resident; a fresh state is
    /// created per call (whisper.cpp's intended usage).
    pub fn transcribe(&mut self, pcm_i16: &[i16]) -> Result<Transcript, WhisperError> {
        // i16 PCM in [-32768, 32767] -> f32 in [-1, 1].
        let audio: Vec<f32> = pcm_i16.iter().map(|&s| s as f32 / 32768.0).collect();
        let n = audio.len();

        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_n_threads(thread_count());
        // The model is multilingual, but Zipscribe is English-only for now: pin the language so
        // it never auto-detects (and mis-detects) another one. Silence all console prints so logs
        // stay clean.
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let t0 = Instant::now();
        state.full(params, &audio)?;
        let elapsed = t0.elapsed();

        // Join per-segment text into one raw transcript.
        let mut text = String::new();
        for seg in state.as_iter() {
            let s = seg.to_str_lossy()?;
            if !text.is_empty() && !s.is_empty() {
                text.push(' ');
            }
            text.push_str(&s);
        }
        let text = text.trim().to_string();
        log::info!(
            "[stt] {:.1}s audio -> {:.2}s inference, {} chars",
            n as f32 / 16000.0,
            elapsed.as_secs_f32(),
            text.chars().count()
        );
        Ok(Transcript {
            text,
            elapsed,
            sample_count: n,
        })
    }
}

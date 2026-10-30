//! The Whisper model file: which one we ship, where it lives, and downloading it.
//!
//! The speech model is fixed (not a user choice): `large-v3-turbo`, 5-bit quantized (`q5_0`). The
//! quantized weights use ~780 MB of memory while transcribing versus ~1.9 GB for the f16 original,
//! with identical transcripts in our tests. It is multilingual, but `crate::stt` pins decoding to
//! English for now.

use crate::config::Config;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// A downloadable GGML Whisper model.
pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub size_mb: u32,
}

/// The one speech model Zipscribe uses.
pub const MODEL: ModelInfo =
    ModelInfo { id: "large-v3-turbo-q5_0", label: "Whisper Large v3 Turbo", size_mb: 574 };

/// `~/Library/Application Support/Zipscribe/models`
pub fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Zipscribe")
        .join("models")
}

fn file_name(id: &str) -> String {
    format!("ggml-{id}.bin")
}

/// Where the speech model is installed (may not exist yet).
pub fn installed_path() -> PathBuf {
    models_dir().join(file_name(MODEL.id))
}

/// Resolve the model file to load: `WF_MODEL`, an explicit config path, the app-support copy, or
/// (for development checkouts) `models/` relative to the working directory.
pub fn resolve_path(cfg: &Config) -> PathBuf {
    if let Ok(p) = std::env::var("WF_MODEL") {
        return PathBuf::from(p);
    }
    if !cfg.stt_model_path.is_empty() {
        return PathBuf::from(&cfg.stt_model_path);
    }
    let installed = installed_path();
    if installed.exists() {
        return installed;
    }
    let dev = PathBuf::from("models").join(file_name(MODEL.id));
    if dev.exists() {
        return dev;
    }
    installed
}

pub fn is_installed() -> bool {
    installed_path().exists()
}

fn url(id: &str) -> String {
    format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        file_name(id)
    )
}

/// Download the speech model into `models_dir()`, calling `progress(done, total)` as bytes arrive.
/// Writes to a `.part` file and renames on success, so a partial download is never mistaken for a
/// model. Checks the byte count against the server's `Content-Length`. On success, removes
/// Whisper models from earlier versions (base/small/medium.en, f16 large-v3-turbo) to reclaim
/// their disk space.
pub fn download(
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, String> {
    let id = MODEL.id;
    let dir = models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let dest = installed_path();
    let part = dest.with_extension("bin.part");

    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client
        .get(url(id))
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("download failed: {e}"))?;
    let total = resp.content_length().unwrap_or(0);

    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = 0u64;
    let mut last = std::time::Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&part);
            return Err("cancelled".into());
        }
        let n = resp.read(&mut buf).map_err(|e| format!("download interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        if last.elapsed().as_millis() > 150 {
            progress(done, total);
            last = std::time::Instant::now();
        }
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if total > 0 && done != total {
        let _ = std::fs::remove_file(&part);
        return Err(format!("incomplete download ({done} of {total} bytes)"));
    }
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    progress(done, total.max(done));
    remove_legacy_models();
    Ok(dest)
}

/// Whisper models that earlier versions offered and this one no longer uses.
const LEGACY_IDS: &[&str] = &["base.en", "small.en", "medium.en", "large-v3-turbo"];

fn remove_legacy_models() {
    for id in LEGACY_IDS {
        let p = models_dir().join(file_name(id));
        if p.exists() {
            match std::fs::remove_file(&p) {
                Ok(()) => log::info!("[models] removed old speech model {}", p.display()),
                Err(e) => log::warn!("[models] could not remove {}: {e}", p.display()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_model_is_large_v3_turbo_q5() {
        assert_eq!(file_name(MODEL.id), "ggml-large-v3-turbo-q5_0.bin");
        assert_eq!(
            url(MODEL.id),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
        );
        assert!(!LEGACY_IDS.contains(&MODEL.id));
    }
}

//! Whisper model files: where they live, which ones we offer, and downloading them.

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

pub const MODELS: &[ModelInfo] = &[
    ModelInfo { id: "base.en", label: "Base (fastest, basic accuracy)", size_mb: 148 },
    ModelInfo { id: "small.en", label: "Small (fast, good accuracy)", size_mb: 488 },
    ModelInfo { id: "medium.en", label: "Medium (best accuracy)", size_mb: 1533 },
];

/// `~/Library/Application Support/Zipscribe/models`
pub fn models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Zipscribe")
        .join("models")
}

/// Model id for the config value (`"whisper-medium.en"` -> `"medium.en"`).
pub fn model_id(cfg: &Config) -> &str {
    cfg.stt_model.trim_start_matches("whisper-")
}

fn file_name(id: &str) -> String {
    format!("ggml-{id}.bin")
}

/// Where the model for `id` is installed (may not exist yet).
pub fn installed_path(id: &str) -> PathBuf {
    models_dir().join(file_name(id))
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
    let id = model_id(cfg);
    let installed = installed_path(id);
    if installed.exists() {
        return installed;
    }
    let dev = PathBuf::from("models").join(file_name(id));
    if dev.exists() {
        return dev;
    }
    installed
}

pub fn is_installed(id: &str) -> bool {
    installed_path(id).exists()
}

fn url(id: &str) -> String {
    format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        file_name(id)
    )
}

/// Download model `id` into `models_dir()`, calling `progress(done, total)` as bytes arrive.
/// Writes to a `.part` file and renames on success, so a partial download is never mistaken for a
/// model. Checks the byte count against the server's `Content-Length`.
pub fn download(
    id: &str,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, String> {
    if !MODELS.iter().any(|m| m.id == id) {
        return Err(format!("unknown model '{id}'"));
    }
    let dir = models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let dest = installed_path(id);
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
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_strips_prefix() {
        let cfg = Config::default();
        assert_eq!(model_id(&cfg), "medium.en");
        assert_eq!(file_name("small.en"), "ggml-small.en.bin");
    }

    #[test]
    fn unknown_model_is_rejected() {
        let cancel = AtomicBool::new(false);
        assert!(download("../evil", &cancel, |_, _| {}).is_err());
    }
}

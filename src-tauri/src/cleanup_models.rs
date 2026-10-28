//! Preset MLX clean-up models + HuggingFace download of arbitrary MLX repos.
//!
//! The clean-up LLM is a standard MLX-format model directory (safetensors + tokenizer +
//! config.json) that loads in-process via `crate::mlx` (task-1). Presets are small
//! `mlx-community` Qwen3 models curated by size (the smallest is the default install target).
//! Any user-pasted HuggingFace repo id or link is also supported: we enumerate the repo's
//! file tree via the HF API and fetch the MLX-relevant files (safetensors + tokenizer +
//! config) into the app models dir.
//!
//! The download mirrors the whisper `crate::models::download` flow: writes to a `.part`
//! scratch dir, reports byte-count progress, supports cancellation, and atomically finalizes
//! only on a complete transfer so a partial download is never mistaken for a model.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::models::models_dir;

/// A preset clean-up model. `repo_id` is the HuggingFace `owner/name`; files are enumerated
/// lazily via the HF tree API so the same path serves sharded and single-file layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanupModel {
    /// Stable, filesystem-safe local identifier, e.g. `"qwen3-0.6b-4bit"`.
    pub id: &'static str,
    pub label: &'static str,
    /// HuggingFace `owner/name` (case-sensitive as published).
    pub repo_id: &'static str,
    pub size_mb: u32,
    /// Marks the smallest/fastest preset as the default install target.
    pub is_default: bool,
}

/// Curated preset list. Smallest (and default) first, then larger/higher-quality.
pub const CLEANUP_MODELS: &[CleanupModel] = &[
    CleanupModel {
        id: "qwen3-0.6b-4bit",
        label: "Qwen3 0.6B · 4-bit — fastest, smallest",
        repo_id: "mlx-community/Qwen3-0.6B-4bit",
        size_mb: 335,
        is_default: true,
    },
    CleanupModel {
        id: "qwen3-1.7b-4bit",
        label: "Qwen3 1.7B · 4-bit — balanced",
        repo_id: "mlx-community/Qwen3-1.7B-4bit",
        size_mb: 1200,
        is_default: false,
    },
    CleanupModel {
        id: "qwen3-4b-4bit",
        label: "Qwen3 4B · 4-bit — best quality",
        repo_id: "mlx-community/Qwen3-4B-4bit",
        size_mb: 2600,
        is_default: false,
    },
];

/// Whether `id` is a curated preset id.
pub fn is_preset_id(id: &str) -> bool {
    CLEANUP_MODELS.iter().any(|m| m.id == id)
}

/// The default (smallest) preset.
pub fn default_model() -> &'static CleanupModel {
    CLEANUP_MODELS
        .iter()
        .find(|m| m.is_default)
        .expect("at least one preset is marked default")
}

/// The root dir for all clean-up models: `<app>/models/cleanup/`.
pub fn cleanup_models_dir() -> PathBuf {
    models_dir().join("cleanup")
}

/// Where a model with a (filesystem-safe) `id` is installed; may not exist yet.
///
/// The `id` is sanitized so a pasted HF link cannot break into the models dir.
pub fn installed_dir(local_id: &str) -> PathBuf {
    cleanup_models_dir().join(sanitize_id(local_id))
}

/// Is the clean-up model with `local_id` installed (a non-empty, configured dir)?
pub fn is_installed(local_id: &str) -> bool {
    let d = installed_dir(local_id);
    d.join("config.json").exists() && (d.join("model.safetensors").exists()
        || has_shards(&d))
}

/// Resolve the on-disk MLX model directory for the active cleanup model.
///
/// An explicit `cleanup_model_path`, when set and present, wins. Otherwise the `cleanup_model`
/// identity is resolved: a preset id maps to its local dir; an arbitrary HF repo id/link is
/// sanitized to a local dir. Returns `None` if nothing is installed (the caller then falls back
/// to `light_cleanse` — dictation must never block).
pub fn resolve_model_dir(cleanup_model: &str, cleanup_model_path: &str) -> Option<PathBuf> {
    // 1. Explicit path wins when it exists.
    if !cleanup_model_path.trim().is_empty() {
        let p = std::path::Path::new(cleanup_model_path.trim());
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    // 2. Direct local id already on disk (e.g. a previously-added HF model stored in
    //    `cleanup_model`, or an explicit `cleanup_model` id). Checked before the preset /
    //    HF-parse branches so an installed id resolves even when it isn't an owner/name
    //    string.
    if is_installed(cleanup_model) {
        return Some(installed_dir(cleanup_model));
    }
    // 3. Preset id -> its local dir (if installed).
    if is_preset_id(cleanup_model) {
        return is_installed(cleanup_model).then_some(installed_dir(cleanup_model));
    }
    // 4. Arbitrary HF repo id/link -> sanitized local dir (if installed).
    match parse_repo(cleanup_model) {
        Ok(repo) => {
            let local = sanitize_id(&repo);
            is_installed(&local).then_some(installed_dir(&local))
        }
        Err(_) => None,
    }
}

fn has_shards(d: &Path) -> bool {
    std::fs::read_dir(d)
        .map(|mut e| e.any(|f| {
            f.as_ref().is_ok_and(|f| {
                f.file_name()
                    .to_string_lossy()
                    .starts_with("model.safetensors-")
            })
        }))
        .unwrap_or(false)
}

/// Filesystem-safe identifier for a HF repo id: lowercased, keep
/// `[a-z0-9._-]`, collapse runs of other separators to a single dot, and trim a leading
/// or trailing separator. `mlx-community/Qwen3-0.6B-4bit` → `mlx-community.qwen3-0.6b-4bit`.
pub fn sanitize_id(repo_id: &str) -> String {
    let mut out = String::new();
    let mut in_sep = false;
    for c in repo_id.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            in_sep = false;
        } else if c == '.' || c == '-' {
            // A kept separator char; collapse runs of any separator to one.
            if !in_sep {
                out.push(c);
            }
            in_sep = true;
        } else {
            // Any other separator (slash, space, …) collapses to a single dot.
            if !in_sep {
                out.push('.');
            }
            in_sep = true;
        }
    }
    out.trim_matches(|c| c == '.' || c == '-').to_string()
}

/// Resolve a user-supplied HuggingFace reference to a plain `owner/name` repo id.
///
/// Accepts a bare `owner/name`, an `mlx-community/...` id, or a pasted link such as
/// `https://huggingface.co/owner/name`, `https://huggingface.co/owner/name/tree/main/...`,
/// or `www.huggingface.co/owner/name`. Returns `Err` when no `owner/name` can be extracted.
pub fn parse_repo(input: &str) -> Result<String, String> {
    let s = input.trim();
    // Strip a known scheme + host to reach "owner/name/...". Bind the owned `replace`
    // result so it outlives the borrowed segments.
    let no_host = s.replace("www.", "");
    let after_host = no_host
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/");
    // The repo id is the first two non-empty path segments (owner/name).
    let segs: Vec<&str> = after_host
        .split(['/', '?', '#'])
        .filter(|p| !p.is_empty())
        .collect();
    let valid = |seg: &str| {
        !seg.is_empty()
            && seg != "."
            && seg != ".."
            && seg.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if segs.len() >= 2 && valid(segs[0]) && valid(segs[1]) {
        Ok(format!("{}/{}", segs[0], segs[1]))
    } else {
        Err(format!("expected 'owner/name' or a HuggingFace link, got {s:?}"))
    }
}

/// Architectures `crate::mlx` can build (`config.json` `model_type`).
pub const SUPPORTED_MODEL_TYPES: &[&str] = &["qwen3"];

/// Records which HuggingFace repo a user-added model came from, for its label in the UI.
const SOURCE_FILE: &str = "zipscribe-source.txt";

/// A clean-up model installed from an arbitrary HuggingFace repo (not a preset).
pub struct CustomModel {
    pub id: String,
    pub repo: String,
    pub size_bytes: u64,
}

/// Every installed non-preset model under the clean-up models dir.
pub fn installed_custom_models() -> Vec<CustomModel> {
    let Ok(entries) = std::fs::read_dir(cleanup_models_dir()) else {
        return Vec::new();
    };
    let mut out: Vec<CustomModel> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let id = e.file_name().to_string_lossy().to_string();
            if is_preset_id(&id) || id.ends_with(".part") || !is_installed(&id) {
                return None;
            }
            let dir = e.path();
            let repo = std::fs::read_to_string(dir.join(SOURCE_FILE))
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| id.clone());
            let size_bytes = std::fs::read_dir(&dir)
                .map(|fs| fs.filter_map(|f| f.ok()?.metadata().ok()).map(|m| m.len()).sum())
                .unwrap_or(0);
            Some(CustomModel { id, repo, size_bytes })
        })
        .collect();
    out.sort_by(|a, b| a.repo.cmp(&b.repo));
    out
}

/// Delete an installed user-added model. Presets are left alone (they can be re-selected
/// without a download), and the id is sanitized so it cannot escape the models dir.
pub fn remove_custom(id: &str) -> Result<(), String> {
    if is_preset_id(id) {
        return Err("built-in models can't be removed here".into());
    }
    let dir = installed_dir(id);
    if !dir.starts_with(cleanup_models_dir()) || dir == cleanup_models_dir() {
        return Err("invalid model id".into());
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("remove {}: {e}", dir.display()))
}

/// The HF model files we care about (MLX loader reads these). Everything else in the
/// repo (notebooks, demos, README, GGUF variants, …) is skipped.
fn is_mlx_file(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    p == "config.json"
        || p == "generation_config.json"
        || p == "special_tokens_map.json"
        || p == "added_tokens.json"
        || p.starts_with("model.safetensors")
        || p == "tokenizer.json"
        || p == "tokenizer_config.json"
        || p == "vocab.json"
        || p == "merges.txt"
}

fn resolve_download_url(repo_id: &str, file: &str, revision: &str) -> String {
    format!(
        "https://huggingface.co/{repo_id}/resolve/{revision}/{file}"
    )
}

/// Download an arbitrary HuggingFace MLX repo into the app models dir, with byte-count
/// progress and cancellation, mirroring whisper's `crate::models::download`.
///
/// `revision` is the ref (e.g. `main`). Returns the model dir. Writes files to a `.part`
/// scratch dir and renames to the final dir only on a complete transfer, so a partial
/// download never loads.
pub fn download_from_repo(
    local_id: &str,
    repo_id: &str,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .connect_timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;

    let revision = "main";
    // Enumerate the repo tree; collect the MLX-relevant files + their sizes.
    let files = enumerate_files(&client, repo_id, revision, cancel)?;
    if files.is_empty() {
        return Err(format!(
            "no MLX files found in {repo_id} (expected config.json + safetensors + tokenizer)"
        ));
    }
    if !files.iter().any(|f| f.remote.starts_with("model") && f.remote.ends_with(".safetensors")) {
        return Err(format!(
            "{repo_id} has no MLX safetensors weights — pick an MLX-format repo (e.g. from mlx-community)"
        ));
    }
    check_architecture(&client, repo_id, revision)?;
    let total: u64 = files.iter().map(|f| f.size).sum();

    let dest_dir = installed_dir(local_id);
    let part_dir = dest_dir.with_extension("part");
    let _ = std::fs::remove_dir_all(&part_dir);
    std::fs::create_dir_all(&part_dir)
        .map_err(|e| format!("create {}: {e}", part_dir.display()))?;

    let mut done: u64 = 0;
    for f in &files {
        let base = done;
        if cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_dir_all(&part_dir);
            return Err("cancelled".into());
        }
        let url = resolve_download_url(repo_id, &f.remote, revision);
        let got = download_one_file(&client, &url, part_dir.join(&f.local), f.size, cancel, |got| {
            progress(base + got, total);
        })?;
        done = base + got;
        if cancel.load(Ordering::Relaxed) {
            let _ = std::fs::remove_dir_all(&part_dir);
            return Err("cancelled".into());
        }
    }
    if !is_preset_id(local_id) {
        let _ = std::fs::write(part_dir.join(SOURCE_FILE), repo_id);
    }
    // Finalize atomically.
    let _ = std::fs::remove_dir_all(&dest_dir);
    std::fs::rename(&part_dir, &dest_dir)
        .map_err(|e| format!("finalize {}: {e}", dest_dir.display()))?;
    progress(done, total.max(done));
    Ok(dest_dir)
}

/// Reject a repo whose `config.json` names an architecture `crate::mlx` cannot build, before
/// any weights are downloaded.
fn check_architecture(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    revision: &str,
) -> Result<(), String> {
    let url = resolve_download_url(repo_id, "config.json", revision);
    let v: serde_json::Value = client
        .get(&url)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.json())
        .map_err(|e| format!("read {repo_id} config.json: {e}"))?;
    let model_type = v.get("model_type").and_then(|t| t.as_str()).unwrap_or("unknown");
    if SUPPORTED_MODEL_TYPES.contains(&model_type) {
        Ok(())
    } else {
        Err(format!(
            "{repo_id} is a \"{model_type}\" model; Zipscribe can currently run Qwen3-based MLX models \
             (e.g. mlx-community/Qwen3-1.7B-4bit)"
        ))
    }
}

/// Enumerate the MLX-relevant files of a repo via the HuggingFace tree API.
fn enumerate_files(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    revision: &str,
    cancel: &AtomicBool,
) -> Result<Vec<ModelFile>, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let url = format!(
        "https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"
    );
    let resp = client.get(&url).send().map_err(|e| format!("couldn't reach HuggingFace: {e}"))?;
    match resp.status().as_u16() {
        401 | 403 | 404 => {
            return Err(format!("{repo_id} wasn't found on HuggingFace (check the name, or it may be private/gated)"))
        }
        c if c >= 400 => return Err(format!("HuggingFace returned HTTP {c} for {repo_id}")),
        _ => {}
    }
    let v: serde_json::Value = resp.json().map_err(|e| format!("parse tree: {e}"))?;
    let arr = v.as_array().ok_or_else(|| "unexpected tree payload".to_string())?;
    let mut out = Vec::new();
    for e in arr {
        if e.get("type").and_then(|t| t.as_str()) == Some("file")
            && e.get("path").and_then(|p| p.as_str()).map(|p| is_mlx_file(p)).unwrap_or(false)
        {
            let path = e.get("path").and_then(|p| p.as_str()).unwrap_or("").to_string();
            let size = e.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
            out.push(ModelFile {
                remote: path.clone(),
                // Local name mirrors the repo path (config.json / model.safetensors / tokensizer.json).
                local: path,
                size,
            });
        }
    }
    Ok(out)
}

/// One HuggingFace file to fetch.
struct ModelFile {
    remote: String,
    local: String,
    size: u64,
}

/// Download a single file with byte-count + cancellation into its local path.
fn download_one_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: PathBuf,
    expected_size: u64,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<u64, String> {
    let mut resp = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("download {url}: {e}"))?;
    // Prefer the server's Content-Length; fall back to the tree-enumerated size.
    let total = resp.content_length().unwrap_or(expected_size);
    let mut file = std::fs::File::create(&dest).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done = 0u64;
    let mut last = std::time::Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&dest);
            return Err("cancelled".into());
        }
        let n = resp.read(&mut buf).map_err(|e| format!("read {url}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write {}: {e}", dest.display()))?;
        done += n as u64;
        if last.elapsed().as_millis() > 100 {
            progress(done);
            last = std::time::Instant::now();
        }
    }
    file.flush().map_err(|e| e.to_string())?;
    drop(file);
    if total > 0 && done != total {
        let _ = std::fs::remove_file(&dest);
        return Err(format!("incomplete download of {url} ({done} of {total} bytes)"));
    }
    progress(done);
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_id_is_fs_safe_and_lower() {
        assert_eq!(
            sanitize_id("mlx-community/Qwen3-0.6B-4bit"),
            "mlx-community.qwen3-0.6b-4bit"
        );
        // Underscores collapse to a single dot; hyphens are kept.
        assert_eq!(
            sanitize_id("mlx_community/Qwen3-0.6B-4bit"),
            "mlx.community.qwen3-0.6b-4bit"
        );
        // No path breakouts.
        assert_eq!(sanitize_id("../../evil"), "evil");
        assert_eq!(sanitize_id("a//b/../c"), "a.b.c");
    }

    #[test]
    fn parse_repo_accepts_bare_and_links() {
        assert_eq!(parse_repo("mlx-community/Qwen3-0.6B-4bit").unwrap(), "mlx-community/Qwen3-0.6B-4bit");
        assert_eq!(
            parse_repo("https://huggingface.co/owner/my-model").unwrap(),
            "owner/my-model"
        );
        assert_eq!(
            parse_repo("https://www.huggingface.co/owner/my-model/tree/main/model.safetensors")
                .unwrap(),
            "owner/my-model"
        );
        assert!(parse_repo("justone").is_err());
        assert!(parse_repo("owner/..").is_err());
        assert!(parse_repo("owner/name with spaces").is_err());
    }

    #[test]
    fn is_mlx_file_filters_correctly() {
        assert!(is_mlx_file("config.json"));
        assert!(is_mlx_file("model.safetensors"));
        assert!(is_mlx_file("model.safetensors-00001-of-00003.safetensors"));
        assert!(is_mlx_file("tokenizer.json"));
        assert!(!is_mlx_file("README.md"));
        assert!(!is_mlx_file("model.gguf"));
        assert!(!is_mlx_file("notebook.ipynb"));
    }

    #[test]
    fn resolve_directly_installed_id() {
        // A `cleanup_model` value naming an already-installed local dir (e.g. a previously
        // downloaded HF model) must resolve to its on-disk location even when it is not a
        // preset id or an owner/name repo string. Uses a unique, self-cleaning id so it never
        // collides with a real model.
        let id = "zipscribe-resolve-self-test".to_string();
        let _ = std::fs::remove_dir_all(cleanup_models_dir().join(&id));
        let d = cleanup_models_dir().join(id.clone());
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), b"{}".to_vec()).unwrap();
        std::fs::write(d.join("model.safetensors"), b"w".to_vec()).unwrap();
        assert!(is_installed(&id));
        assert_eq!(resolve_model_dir(&id, ""), Some(d.clone()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn default_preset_is_smallest() {
        assert!(default_model().is_default);
        assert_eq!(default_model().id, "qwen3-0.6b-4bit");
        // At least two sizes for the size choice.
        assert!(CLEANUP_MODELS.len() >= 2);
        assert_eq!(
            CLEANUP_MODELS.iter().filter(|m| m.is_default).count(),
            1,
            "exactly one default"
        );
        // Smallest by size_mb is the default.
        let smallest = CLEANUP_MODELS
            .iter()
            .min_by_key(|m| m.size_mb)
            .unwrap();
        assert!(smallest.is_default, "smallest preset should be default");
    }

    #[test]
    fn resolve_model_dir_prefers_explicit_path_then_falls_back() {
        let dir = std::env::temp_dir().join(format!("zpresolve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), [0u8; 8]).unwrap();
        // Explicit, present path wins over an (uninstalled) unrelated preset id.
        assert_eq!(
            resolve_model_dir("mlx-community/Nonexistent-Model", dir.to_str().unwrap()),
            Some(dir)
        );
        // Absent explicit path + a model that is not installed anywhere -> None
        // (caller uses light_cleanse). Use a hermetic sentinel rather than a real preset
        // id: a real preset (e.g. "qwen3-0.6b-4bit") resolves to Some *when installed* on
        // this host, but the test must not couple to host install state.
        let sentinel = "sentinel-uninstalled-zipscribe-model-xyz";
        assert_eq!(resolve_model_dir(sentinel, "/nonexistent/zipscribe/none"), None);
        assert_eq!(resolve_model_dir(sentinel, ""), None);
        // A syntactically bad model id resolves to None, not a panic.
        assert_eq!(resolve_model_dir("   ", ""), None);
    }
}

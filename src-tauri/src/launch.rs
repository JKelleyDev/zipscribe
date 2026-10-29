//! LaunchAgent install/uninstall for "runs on startup" behavior.
//!
//! On login, macOS starts LaunchAgents in `~/Library/LaunchAgents/*.plist`. We generate a
//! plist whose `ProgramArguments` point at the exact release binary (resolved via
//! `current_exe` when you run `zipscribe install`), with `RunAtLoad = true`, and load
//! it via `launchctl load -w`. This makes the dictation service auto-start at login.
//!
//! The same binary doubles as a login "app" that, with no subcommand, runs the service.
//! (Step 6, `app`, wires the tray; for the skeleton this is a clean no-op exit.)

use std::path::{Path, PathBuf};

const LABEL: &str = "com.jordankelley.zipscribe";

/// `~/Library/LaunchAgents`
fn launch_agents_dir() -> Result<PathBuf, LaunchError> {
    let home = std::env::var("HOME").map_err(|e| LaunchError::Home(e))?;
    Ok(PathBuf::from(home).join("Library").join("LaunchAgents"))
}

/// The plist path for our agent.
fn plist_path() -> Result<PathBuf, LaunchError> {
    Ok(launch_agents_dir()?.join(format!("{LABEL}.plist")))
}

/// Generate the plist text for a given binary path. `RunAtLoad` ensures it starts at login.
fn generate_plist(binary: &Path, log_dir: &Path, working_dir: &Path) -> String {
    let out = log_dir.join("zipscribe.out.log");
    let err = log_dir.join("zipscribe.err.log");
    // LaunchAgent plists require a well-formed dictionary; values are absolute paths.
    // `WorkingDirectory` is the repo root (resolved at install time) so relative model paths
    // like `models/ggml-large-v3-turbo.bin` resolve for the auto-started service, which otherwise
    // inherits launchd's default cwd of $HOME.
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{out}</string>
    <key>StandardErrorPath</key>
    <string>{err_file}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
"#,
        label = LABEL,
        binary = binary.display(),
        work_dir = working_dir.display(),
        out = out.display(),
        err_file = err.display(),
    )
}

/// Resolve the binary path for the LaunchAgent. Prefer the path passed by the caller; fall
/// back to `current_exe` (the running `install` invocation) so the agent tracks a release.
fn binary_path_for_install() -> Result<PathBuf, LaunchError> {
    // An override lets you point at a specific binary (e.g. a symlink in /usr/local/bin).
    if let Ok(p) = std::env::var("ZIPSCRIBE_BIN") {
        return Ok(PathBuf::from(p));
    }
    std::env::current_exe().map_err(|e| LaunchError::CurrentExe(e))
}

/// Install the LaunchAgent: create the dir, write the plist, and `launchctl load -w` it.
/// Idempotent — unloads first if already present.
pub fn install() -> Result<String, LaunchError> {
    let plist = plist_path()?;
    let log_dir = launch_agents_dir().map(|d| d.join("logs"))?;
    std::fs::create_dir_all(&log_dir)?;

    // Ensure only one agent is loaded: unload any existing copy first (ignore errors).
    if is_registered() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w", &plist.display().to_string()])
            .status();
    }

    let binary = binary_path_for_install()?;
    std::fs::create_dir_all(binary.parent().unwrap()).ok();
    // The repo root at install time is where relative model paths resolve from.
    let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    let text = generate_plist(&binary, &log_dir, &working_dir);
    std::fs::write(&plist, text)?;

    // On a bare machine, materialize a default config.toml so the service starts cleanly.
    // Never overwrite a user's existing config file.
    let cfg_path = crate::config::Config::default_path();
    let config_written = {
        if !cfg_path.exists() {
            crate::config::Config::default().save().is_ok()
        } else {
            false
        }
    };

    let status = std::process::Command::new("launchctl")
        .args(["load", "-w", &plist.display().to_string()])
        .status()
        .map_err(|e| LaunchError::Launchctl("load".into(), e))?;
    if !status.success() {
        return Err(LaunchError::LaunchctlFailed("load"));
    }
    Ok(format!(
        "Installed LaunchAgent {} for binary {}{}",
        plist.display(),
        binary.display(),
        if config_written {
            format!(" (created default config at {})", crate::config::Config::default_path().display())
        } else {
            String::new()
        }
    ))
}

/// Enable/disable start-at-login for the running app by writing/removing the LaunchAgent plist
/// only. Unlike `install`/`uninstall` it never calls `launchctl`, so toggling it can neither start
/// a second copy nor kill the copy that is running now.
pub fn set_login_item(on: bool) -> Result<(), LaunchError> {
    let plist = plist_path()?;
    if !on {
        if plist.exists() {
            std::fs::remove_file(&plist)?;
        }
        return Ok(());
    }
    let home = std::env::var("HOME").map_err(LaunchError::Home)?;
    let log_dir = PathBuf::from(home).join("Library/Logs/Zipscribe");
    std::fs::create_dir_all(&log_dir)?;
    std::fs::create_dir_all(launch_agents_dir()?)?;
    let binary = binary_path_for_install()?;
    let text = generate_plist(&binary, &log_dir, Path::new("/"));
    std::fs::write(&plist, text)?;
    Ok(())
}

/// Whether start-at-login is enabled (the plist exists).
pub fn login_item_enabled() -> bool {
    plist_path().map(|p| p.exists()).unwrap_or(false)
}

/// Uninstall the LaunchAgent: `launchctl unload` then remove the plist. Idempotent.
pub fn uninstall() -> Result<String, LaunchError> {
    let plist = plist_path()?;
    // Unload regardless of whether it's loaded; ignore "not loaded" errors.
    let _ = std::process::Command::new("launchctl")
        .args(["unload", "-w", &plist.display().to_string()])
        .status();
    if plist.exists() {
        std::fs::remove_file(&plist)?;
    }
    Ok(format!("Uninstalled LaunchAgent ({})", plist.display()))
}

/// Whether the agent is currently registered in `launchctl list`.
pub fn is_registered() -> bool {
    let out = std::process::Command::new("launchctl")
        .arg("list")
        .output()
        .ok();
    match out {
        Some(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).contains(LABEL),
        _ => false,
    }
}

#[derive(Debug)]
pub enum LaunchError {
    Home(std::env::VarError),
    CurrentExe(std::io::Error),
    Io(std::io::Error),
    Launchctl(String, std::io::Error),
    LaunchctlFailed(&'static str),
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::Home(e) => write!(f, "could not resolve HOME: {e}"),
            LaunchError::CurrentExe(e) => write!(f, "could not resolve current binary: {e}"),
            LaunchError::Io(e) => write!(f, "io error: {e}"),
            LaunchError::Launchctl(op, e) => write!(f, "launchctl {op} failed to spawn: {e}"),
            LaunchError::LaunchctlFailed(op) => write!(f, "launchctl {op} returned non-zero"),
        }
    }
}

impl std::error::Error for LaunchError {}

impl From<std::io::Error> for LaunchError {
    fn from(e: std::io::Error) -> Self {
        LaunchError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_has_runatload_and_label() {
        let t = generate_plist(
            Path::new("/opt/homebrew/bin/zipscribe"),
            Path::new("/tmp/wfr-logs"),
            Path::new("/Users/jordankelley/AI-Projects/whisperflow-rs"),
        );
        assert!(t.contains("<key>RunAtLoad</key>"));
        assert!(t.contains("<true/>"));
        assert!(t.contains(LABEL));
        assert!(t.contains("/opt/homebrew/bin/zipscribe"));
        assert!(t.contains("<key>Label</key>"));
        assert!(t.contains("<key>WorkingDirectory</key>"));
        assert!(t.contains("/Users/jordankelley/AI-Projects/whisperflow-rs"));
    }

    #[test]
    fn label_and_paths_resolve() {
        // These should not error on macOS in this repo context.
        assert!(plist_path().is_ok());
        assert!(launch_agents_dir().is_ok());
        // Do not actually install during tests.
    }
}

//! The Tauri application: menu-bar item, setup/settings windows and the commands they call.
//!
//! Threading: the global hotkey manager must live on the main thread (macOS delivers hotkeys
//! through its run loop), so it is created in `setup` and only touched via `run_on_main_thread`.
//! The dictation worker runs on its own thread and only reads hotkey events.

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};

use crate::config::Config;
use crate::hotkey::P2T;
use crate::service::{dictation_loop, Shared, TrayStatus};

/// The hotkey manager is not `Send`; it is only ever used on the main thread.
struct MainOnly(P2T);
// SAFETY: access is confined to closures run via `run_on_main_thread`.
unsafe impl Send for MainOnly {}

struct AppState {
    shared: Arc<Shared>,
    p2t: Mutex<Option<MainOnly>>,
    toggle_item: MenuItem<tauri::Wry>,
    status_item: MenuItem<tauri::Wry>,
    download_cancel: Arc<AtomicBool>,
    downloading: AtomicBool,
}

// ---------------------------------------------------------------- windows

fn open_window(app: &AppHandle, label: &str, page: &str, title: &str, w: f64, h: f64) {
    if let Some(win) = app.get_webview_window(label) {
        let _ = win.show();
        let _ = win.set_focus();
        return;
    }
    let built = WebviewWindowBuilder::new(app, label, WebviewUrl::App(page.into()))
        .title(title)
        .inner_size(w, h)
        .min_inner_size(w.min(520.0), 480.0)
        .center()
        .build();
    match built {
        Ok(win) => {
            let _ = win.set_focus();
        }
        Err(e) => log::error!("[app] could not open {label} window: {e}"),
    }
}

fn open_settings(app: &AppHandle) {
    open_window(app, "settings", "settings.html", "Zipscribe Settings", 640.0, 760.0);
}

fn open_setup(app: &AppHandle) {
    open_window(app, "setup", "setup.html", "Welcome to Zipscribe", 600.0, 640.0);
}

// ---------------------------------------------------------------- menu / status

fn set_tray_status(app: &AppHandle, st: TrayStatus) {
    if let Some(tray) = app.tray_by_id("main") {
        let _ = tray.set_title(Some(st.title()));
    }
    let state = app.state::<AppState>();
    let text = match st {
        TrayStatus::Idle => "Ready — hold your hotkey to dictate",
        TrayStatus::Recording => "Listening…",
        TrayStatus::Processing => "Working…",
        TrayStatus::Disabled => "Paused",
    };
    let _ = state.status_item.set_text(text);
}

fn sync_toggle_text(state: &AppState) {
    let on = state.shared.enabled.load(Ordering::Relaxed);
    let _ = state
        .toggle_item
        .set_text(if on { "Pause dictation" } else { "Resume dictation" });
}

// ---------------------------------------------------------------- commands

#[derive(Serialize)]
struct Permissions {
    microphone: &'static str,
    accessibility: bool,
}

#[derive(Serialize)]
struct ModelEntry {
    id: &'static str,
    label: &'static str,
    size_mb: u32,
    installed: bool,
}

#[derive(Serialize)]
struct ServiceStatus {
    model_ready: bool,
    model_error: Option<String>,
    downloading: bool,
    hotkey_registered: bool,
}

#[tauri::command]
fn get_config(state: State<'_, AppState>) -> Config {
    state.shared.config()
}

#[tauri::command]
fn default_prompt() -> String {
    Config::default().cleanup_system_prompt
}

#[tauri::command]
fn validate_hotkey(hotkey: String) -> Result<(), String> {
    let mut cfg = Config::default();
    cfg.hotkey = hotkey;
    crate::hotkey::parse_hotkey(&cfg).map(|_| ())
}

#[tauri::command]
async fn save_config(
    app: AppHandle,
    state: State<'_, AppState>,
    cfg: Config,
) -> Result<(), String> {
    crate::hotkey::parse_hotkey(&cfg)?;
    let old_hotkey = state.shared.config().hotkey;

    if cfg.hotkey != old_hotkey || state.p2t.lock().unwrap().is_none() {
        let (tx, rx) = mpsc::channel();
        let app2 = app.clone();
        let cfg2 = cfg.clone();
        app.run_on_main_thread(move || {
            let st = app2.state::<AppState>();
            let mut guard = st.p2t.lock().unwrap();
            let result = match guard.as_mut() {
                Some(p) => p.0.rebind(&cfg2),
                None => P2T::new(&cfg2).map(|p| {
                    let id = p.id();
                    *guard = Some(MainOnly(p));
                    id
                }),
            };
            if let Ok(id) = &result {
                st.shared.hotkey_id.store(*id, Ordering::Relaxed);
            }
            let _ = tx.send(result);
        })
        .map_err(|e| e.to_string())?;
        rx.recv_timeout(Duration::from_secs(3))
            .map_err(|_| "timed out registering the hotkey".to_string())??;
    }

    cfg.save().map_err(|e| e.to_string())?;
    state.shared.set_config(cfg.clone());
    sync_toggle_text(&state);
    let st = if cfg.enabled { TrayStatus::Idle } else { TrayStatus::Disabled };
    set_tray_status(&app, st);
    Ok(())
}

#[tauri::command]
async fn permissions() -> Permissions {
    Permissions {
        microphone: crate::output::mic_status().as_str(),
        accessibility: crate::output::check_permissions().accessibility,
    }
}

#[tauri::command]
async fn request_microphone() {
    // Opening an input stream is what makes macOS show its prompt.
    std::thread::spawn(crate::output::trigger_microphone_prompt);
}

#[tauri::command]
async fn request_accessibility() -> bool {
    crate::output::request_accessibility()
}

#[tauri::command]
fn open_system_settings(pane: String) {
    let anchor = match pane.as_str() {
        "microphone" => "Privacy_Microphone",
        _ => "Privacy_Accessibility",
    };
    let _ = std::process::Command::new("open")
        .arg(format!("x-apple.systempreferences:com.apple.preference.security?{anchor}"))
        .spawn();
}

/// (Cleanup no longer lists external LLM models: the clean-up LLM runs in-process via MLX, and the
/// set of *available* MLX models comes from `crate::cleanup_models` presets + any the user has
/// installed from HuggingFace — not from an external service.)

#[tauri::command]
fn model_list() -> Vec<ModelEntry> {
    crate::models::MODELS
        .iter()
        .map(|m| ModelEntry {
            id: m.id,
            label: m.label,
            size_mb: m.size_mb,
            installed: crate::models::is_installed(m.id),
        })
        .collect()
}

#[tauri::command]
fn service_status(state: State<'_, AppState>) -> ServiceStatus {
    ServiceStatus {
        model_ready: state.shared.model_ready.load(Ordering::Relaxed),
        model_error: state.shared.model_error.lock().unwrap().clone(),
        downloading: state.downloading.load(Ordering::Relaxed),
        hotkey_registered: state.p2t.lock().unwrap().is_some(),
    }
}

#[derive(Serialize, Clone)]
struct DownloadProgress {
    id: String,
    done: u64,
    total: u64,
}

#[derive(Serialize, Clone)]
struct DownloadResult {
    id: String,
    error: Option<String>,
}

#[tauri::command]
fn download_model(app: AppHandle, state: State<'_, AppState>, id: String) -> Result<(), String> {
    if state.downloading.swap(true, Ordering::SeqCst) {
        return Err("a download is already running".into());
    }
    state.download_cancel.store(false, Ordering::Relaxed);
    let cancel = state.download_cancel.clone();
    std::thread::spawn(move || {
        let progress_app = app.clone();
        let progress_id = id.clone();
        let result = crate::models::download(&id, &cancel, move |done, total| {
            let _ = progress_app.emit(
                "download-progress",
                DownloadProgress { id: progress_id.clone(), done, total },
            );
        });
        let st = app.state::<AppState>();
        st.downloading.store(false, Ordering::SeqCst);
        if result.is_ok() {
            st.shared.poke();
        }
        let _ = app.emit(
            "download-done",
            DownloadResult { id, error: result.err() },
        );
    });
    Ok(())
}

#[tauri::command]
fn cancel_download(state: State<'_, AppState>) {
    state.download_cancel.store(true, Ordering::Relaxed);
}

/// One clean-up model option for the UI: a preset or a user-installed HF model.
#[derive(Serialize, Clone)]
struct CleanupModelEntry {
    id: String,
    label: String,
    size_mb: u32,
    is_default: bool,
    installed: bool,
    repo: String,
    /// Added by the user from HuggingFace (removable), rather than a built-in preset.
    custom: bool,
}

/// The catalog of clean-up models: the presets plus any model the user added from HuggingFace.
/// The clean-up LLM runs in-process via Apple MLX — no external daemon.
#[tauri::command]
fn cleanup_model_list() -> Vec<CleanupModelEntry> {
    use crate::cleanup_models as cm;
    let presets = cm::CLEANUP_MODELS.iter().map(|m| CleanupModelEntry {
        id: m.id.to_string(),
        label: m.label.to_string(),
        size_mb: m.size_mb,
        is_default: m.is_default,
        installed: cm::is_installed(m.id),
        repo: m.repo_id.to_string(),
        custom: false,
    });
    let custom = cm::installed_custom_models().into_iter().map(|m| CleanupModelEntry {
        label: m.repo.clone(),
        size_mb: (m.size_bytes / 1_000_000) as u32,
        is_default: false,
        installed: true,
        repo: m.repo,
        id: m.id,
        custom: true,
    });
    presets.chain(custom).collect()
}

/// Install a clean-up model by preset id OR an arbitrary HuggingFace repo id/link (e.g.
/// `qwen3-0.6b-4bit`, `mlx-community/Qwen3-0.6B-4bit`, or a pasted URL). Validation errors
/// (bad input, unsupported architecture) are returned straight away; the download itself runs
/// in the background with `cleanup-progress` / `cleanup-done` events, and on success the
/// dictation worker picks the model up on its next config version bump — no restart needed.
/// Resolves to the local id the model is installed under, so the UI can select it.
#[tauri::command]
fn cleanup_install(app: AppHandle, state: State<'_, AppState>, repo: String) -> Result<String, String> {
    use crate::cleanup_models as cm;
    let (local_id, repo_id) = match cm::CLEANUP_MODELS.iter().find(|m| m.id == repo) {
        Some(m) => (m.id.to_string(), m.repo_id.to_string()),
        None => {
            let repo_id = cm::parse_repo(&repo)?;
            // A pasted link to a preset's repo installs as that preset, not a duplicate.
            match cm::CLEANUP_MODELS.iter().find(|m| m.repo_id.eq_ignore_ascii_case(&repo_id)) {
                Some(m) => (m.id.to_string(), m.repo_id.to_string()),
                None => (cm::sanitize_id(&repo_id), repo_id),
            }
        }
    };
    if state.downloading.swap(true, Ordering::SeqCst) {
        return Err("a download is already running".into());
    }
    state.download_cancel.store(false, Ordering::Relaxed);
    let cancel = state.download_cancel.clone();
    let id = local_id.clone();
    std::thread::spawn(move || {
        let progress_app = app.clone();
        let result = cm::download_from_repo(&id, &repo_id, &cancel, |done, total| {
            let _ = progress_app.emit("cleanup-progress", DownloadProgress { id: id.clone(), done, total });
        });
        let st = app.state::<AppState>();
        st.downloading.store(false, Ordering::SeqCst);
        if result.is_ok() {
            st.shared.poke();
        }
        let _ = app.emit("cleanup-done", DownloadResult { id, error: result.err() });
    });
    Ok(local_id)
}

/// Delete a model the user added from HuggingFace.
#[tauri::command]
fn cleanup_remove(state: State<'_, AppState>, id: String) -> Result<(), String> {
    crate::cleanup_models::remove_custom(&id)?;
    state.shared.poke();
    Ok(())
}

#[tauri::command]
fn login_item_enabled() -> bool {
    crate::launch::login_item_enabled()
}

#[tauri::command]
fn set_login_item(on: bool) -> Result<(), String> {
    crate::launch::set_login_item(on).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_settings_window(app: AppHandle) {
    open_settings(&app);
}

#[tauri::command]
fn open_setup_window(app: AppHandle) {
    open_setup(&app);
}

#[tauri::command]
fn close_window(window: tauri::WebviewWindow) {
    let _ = window.close();
}

/// Quote a path for a POSIX shell (single quotes, escaping embedded single quotes).
fn sh_quote(p: &std::path::Path) -> String {
    format!("'{}'", p.to_string_lossy().replace('\'', "'\\''"))
}

/// Remove Zipscribe from this Mac. The login item is dropped immediately; everything else is
/// done by a detached shell after the app quits (a running app can't delete its own bundle, and
/// the permission reset must happen after we stop touching the mic). The app bundle goes to the
/// Trash so it is recoverable. No external LLM service is touched (cleanup is in-process MLX).
#[tauri::command]
fn uninstall_app(app: AppHandle, delete_model: bool, delete_settings: bool) -> Result<(), String> {
    use std::path::PathBuf;
    let home = dirs::home_dir().ok_or("could not find your home folder")?;
    let _ = crate::launch::uninstall();

    let mut cmds: Vec<String> = vec!["sleep 1".into()];
    // Move the .app bundle (found from the running binary) to the Trash.
    if let Some(bundle) = std::env::current_exe().ok().and_then(|exe| {
        exe.ancestors()
            .find(|a| a.extension().is_some_and(|e| e == "app"))
            .map(PathBuf::from)
    }) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let dest = home.join(".Trash").join(format!("Zipscribe-{stamp}.app"));
        cmds.push(format!("mv {} {}", sh_quote(&bundle), sh_quote(&dest)));
    }
    let mut remove = vec![
        home.join("Library/Caches/com.jordankelley.zipscribe"),
        home.join("Library/WebKit/com.jordankelley.zipscribe"),
        home.join("Library/LaunchAgents/logs"),
    ];
    if delete_model {
        if let Some(dir) = crate::models::models_dir().parent() {
            remove.push(dir.to_path_buf());
        }
    }
    if delete_settings {
        if let Some(dir) = crate::config::Config::default_path().parent() {
            remove.push(dir.to_path_buf());
        }
        remove.push(home.join("Library/Logs/Zipscribe"));
    }
    for p in remove {
        cmds.push(format!("rm -rf {}", sh_quote(&p)));
    }
    cmds.push("tccutil reset All com.jordankelley.zipscribe".into());
    let script = format!("({}) >/dev/null 2>&1 &", cmds.join("; "));
    std::process::Command::new("/bin/sh")
        .args(["-c", &script])
        .spawn()
        .map_err(|e| e.to_string())?;
    log::info!("[app] uninstall started (model={delete_model}, settings={delete_settings})");
    app.exit(0);
    Ok(())
}

#[tauri::command]
fn log_path() -> String {
    log_file_path().display().to_string()
}

// ---------------------------------------------------------------- startup

fn log_file_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join("Library/Logs/Zipscribe/zipscribe.log")
}

fn init_logging() {
    let path = log_file_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    if let Ok(file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        builder.target(env_logger::Target::Pipe(Box::new(file)));
    }
    let _ = builder.try_init();
}

pub fn run() -> ExitCode {
    init_logging();
    let cfg = Config::load_or_default();
    log::info!("Zipscribe starting: hotkey={} setup_complete={}", cfg.hotkey, cfg.setup_complete);

    let result = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // Launching the app again opens its settings.
            open_settings(app);
        }))
        .invoke_handler(tauri::generate_handler![
            get_config,
            default_prompt,
            validate_hotkey,
            save_config,
            permissions,
            request_microphone,
            request_accessibility,
            open_system_settings,
            model_list,
            cleanup_model_list,
            cleanup_install,
            cleanup_remove,
            service_status,
            download_model,
            cancel_download,
            login_item_enabled,
            set_login_item,
            open_settings_window,
            open_setup_window,
            close_window,
            log_path,
            uninstall_app,
        ])
        .setup(move |app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let shared = Shared::new(cfg.clone());

            // Hotkey: main thread. A failure (e.g. combo taken) is not fatal; settings can fix it.
            let p2t = match P2T::new(&cfg) {
                Ok(p) => {
                    shared.hotkey_id.store(p.id(), Ordering::Relaxed);
                    Some(MainOnly(p))
                }
                Err(e) => {
                    log::error!("[app] could not register hotkey '{}': {e}", cfg.hotkey);
                    None
                }
            };

            let status_item = MenuItem::with_id(app, "status", "Starting…", false, None::<&str>)?;
            let toggle_item = MenuItem::with_id(
                app,
                "toggle",
                if cfg.enabled { "Pause dictation" } else { "Resume dictation" },
                true,
                None::<&str>,
            )?;
            let settings_item = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
            let setup_item = MenuItem::with_id(app, "setup", "Setup guide…", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit Zipscribe", true, None::<&str>)?;
            let sep1 = PredefinedMenuItem::separator(app)?;
            let sep2 = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(
                app,
                &[&status_item, &sep1, &toggle_item, &settings_item, &setup_item, &sep2, &quit_item],
            )?;

            app.manage(AppState {
                shared: shared.clone(),
                p2t: Mutex::new(p2t),
                toggle_item,
                status_item,
                download_cancel: Arc::new(AtomicBool::new(false)),
                downloading: AtomicBool::new(false),
            });

            TrayIconBuilder::with_id("main")
                .title(if cfg.enabled { TrayStatus::Idle } else { TrayStatus::Disabled }.title())
                .tooltip("Zipscribe")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "toggle" => {
                        let st = app.state::<AppState>();
                        let on = !st.shared.enabled.load(Ordering::Relaxed);
                        st.shared.enabled.store(on, Ordering::Relaxed);
                        let mut cfg = st.shared.config();
                        cfg.enabled = on;
                        let _ = cfg.save();
                        st.shared.set_config(cfg);
                        sync_toggle_text(&st);
                        set_tray_status(app, if on { TrayStatus::Idle } else { TrayStatus::Disabled });
                    }
                    "settings" => open_settings(app),
                    "setup" => open_setup(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            // Dictation worker.
            let sink_app = app.handle().clone();
            let stop = Arc::new(AtomicBool::new(false));
            std::thread::Builder::new()
                .name("dictation".into())
                .spawn(move || {
                    dictation_loop(
                        shared,
                        stop,
                        Box::new(move |st| set_tray_status(&sink_app, st)),
                    )
                })
                .expect("spawn dictation thread");

            if !cfg.setup_complete {
                open_setup(app.handle());
            }
            Ok(())
        })
        .build(tauri::generate_context!());

    match result {
        Ok(app) => {
            app.run(|_app, event| {
                // Closing the setup/settings windows must not quit the menu-bar app.
                if let tauri::RunEvent::ExitRequested { api, code, .. } = event {
                    if code.is_none() {
                        api.prevent_exit();
                    }
                }
            });
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to start Zipscribe: {e}");
            ExitCode::FAILURE
        }
    }
}

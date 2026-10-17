//! Output: inject cleansed text into the previously-focused window, plus macOS permission
//! detection so the app guides the user (rather than crashing) when Microphone or Accessibility
//! is missing.
//!
//! Two output strategies:
//!  * `type`  — key simulation of the full string via `enigo::Keyboard::text` (fast path).
//!  * `paste` — write to the macOS pasteboard and press ⌘V (reliable for arbitrary Unicode/newlines).
//!
//! The tray agent never takes key focus, so the *previously focused* window is always some other
//! app — we simply deliver to whatever currently has focus.
//!
//! Permissions:
//!  * **Accessibility** — required for CGEvent key simulation.  Detected via `AXIsProcessTrusted`;
//!    if missing we guide the user to enable it and degrade (no crash) to `paste`, which only needs
//!    the pasteboard, not full input-mirroring.
//!  * **Microphone** — probed by attempting a short capture; a denial surfaces as a device/IO
//!    error, which we report and guide without crashing.

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use cpal::traits::StreamTrait as _;

/// Whether accessibility is granted to this process.  Detects whether the current process may
/// simulate keyboard input (required by `enigo`).  On non-macOS we assume granted.
#[inline]
#[cfg(target_os = "macos")]
fn accessibility_trusted() -> bool {
    unsafe extern "C" {
        #[link_name = "AXIsProcessTrusted"]
        fn ax_is_process_trusted() -> std::os::raw::c_int;
    }
    // SAFETY: `AXIsProcessTrusted` is a no-arg C function that returns a non-zero int.
    unsafe { ax_is_process_trusted() != 0 }
}
/// Non-macOS stub for cross-platform builds (not used in v1).
#[cfg(not(target_os = "macos"))]
fn accessibility_trusted() -> bool {
    true
}

/// Microphone authorization as reported by AVFoundation (does not prompt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicStatus {
    NotDetermined,
    Restricted,
    Denied,
    Authorized,
}

impl MicStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MicStatus::NotDetermined => "not_determined",
            MicStatus::Restricted => "restricted",
            MicStatus::Denied => "denied",
            MicStatus::Authorized => "authorized",
        }
    }
}

#[cfg(target_os = "macos")]
pub fn mic_status() -> MicStatus {
    use std::ffi::{c_char, c_void};
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {
        static AVMediaTypeAudio: *const c_void;
    }
    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *const c_void;
        fn sel_registerName(name: *const c_char) -> *const c_void;
        fn objc_msgSend();
    }
    // SAFETY: standard Objective-C message send of `+[AVCaptureDevice authorizationStatusForMediaType:]`
    // (returns NSInteger). objc_msgSend is cast to the concrete signature of that call.
    let raw = unsafe {
        let class = objc_getClass(c"AVCaptureDevice".as_ptr());
        let sel = sel_registerName(c"authorizationStatusForMediaType:".as_ptr());
        let send: unsafe extern "C" fn(*const c_void, *const c_void, *const c_void) -> isize =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(class, sel, AVMediaTypeAudio)
    };
    match raw {
        0 => MicStatus::NotDetermined,
        1 => MicStatus::Restricted,
        2 => MicStatus::Denied,
        _ => MicStatus::Authorized,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn mic_status() -> MicStatus {
    MicStatus::Authorized
}

/// Show the system Accessibility prompt (once) for this process. Returns whether it is trusted now.
#[cfg(target_os = "macos")]
pub fn request_accessibility() -> bool {
    use std::ffi::c_void;
    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        static kAXTrustedCheckOptionPrompt: *const c_void;
        fn AXIsProcessTrustedWithOptions(options: *const c_void) -> u8;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        static kCFBooleanTrue: *const c_void;
        static kCFTypeDictionaryKeyCallBacks: c_void;
        static kCFTypeDictionaryValueCallBacks: c_void;
        fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            n: isize,
            key_cb: *const c_void,
            value_cb: *const c_void,
        ) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }
    // SAFETY: builds the documented {kAXTrustedCheckOptionPrompt: true} dictionary and releases it.
    unsafe {
        let keys = [kAXTrustedCheckOptionPrompt];
        let values = [kCFBooleanTrue];
        let dict = CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            1,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        );
        let trusted = AXIsProcessTrustedWithOptions(dict) != 0;
        CFRelease(dict);
        trusted
    }
}

#[cfg(not(target_os = "macos"))]
pub fn request_accessibility() -> bool {
    true
}

/// Ask macOS for microphone access via `+[AVCaptureDevice requestAccessForMediaType:completionHandler:]`.
/// Going through AVFoundation (rather than opening a CoreAudio stream) is what makes
/// `mic_status()` report the answer in this same process instead of staying "not determined".
#[cfg(target_os = "macos")]
pub fn trigger_microphone_prompt() {
    use block2::RcBlock;
    use std::ffi::{c_char, c_void};
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {
        static AVMediaTypeAudio: *const c_void;
    }
    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *const c_void;
        fn sel_registerName(name: *const c_char) -> *const c_void;
        fn objc_msgSend();
    }
    let handler = RcBlock::new(|granted: objc2::runtime::Bool| {
        log::info!("[output] microphone access request answered: granted={}", granted.as_bool());
    });
    // SAFETY: standard Objective-C message send; objc_msgSend is cast to the concrete signature
    // of `requestAccessForMediaType:completionHandler:` (returns void). The block is retained
    // by AVFoundation for the duration of the request.
    unsafe {
        let class = objc_getClass(c"AVCaptureDevice".as_ptr());
        let sel = sel_registerName(c"requestAccessForMediaType:completionHandler:".as_ptr());
        let send: unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *const c_void) =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send(class, sel, AVMediaTypeAudio, &*handler as *const _ as *const c_void);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn trigger_microphone_prompt() {
    let _ = microphone_ok();
}

/// Probe the microphone by attempting a short capture.  Returns `true` when we can open an input
/// device and get at least one buffer.  A denial surfaces as an error or no data — we translate that
/// to guidance rather than crashing.
fn microphone_ok() -> bool {
    use cpal::traits::{DeviceTrait, HostTrait};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let host = cpal::default_host();
    let dev = match host.input_devices().ok().and_then(|mut i| i.next()) {
        Some(d) => d,
        None => return false,
    };
    let supp = match dev.default_input_config() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let config = supp.config();
    let got = Arc::new(AtomicBool::new(false));
    let flag = got.clone();
    let err_fn = |e: cpal::Error| log::warn!("[output] mic probe error: {e:?}");
    // Keep the stream alive briefly so the audio thread can flush at least one buffer.
    let stream = match dev.build_input_stream(
        config,
        move |data: &[i16], _info: &cpal::InputCallbackInfo| {
            if !data.is_empty() {
                flag.store(true, Ordering::Relaxed);
            }
        },
        err_fn,
        None,
    ) {
        Ok(s) => s,
        Err(_) => return false,
    };
    // Must explicitly start the stream — CoreAudio will not deliver buffers until it
    // is in the played state.  capture.rs does the same.
    if let Err(e) = stream.play() {
        log::warn!("[output] mic probe play() failed: {e:?}");
        return false;
    }
    // A single CoreAudio buffer can take ~100-230 ms; wait half a second so the async audio
    // thread can flush at least one buffer before we drop the stream.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let got_any = got.load(Ordering::Relaxed);
    drop(stream);
    got_any
}

/// Per-permission status plus a guided hint for when it's missing.
#[derive(Debug, Default)]
pub struct PermissionReport {
    pub accessibility: bool,
    pub microphone: bool,
}

impl PermissionReport {
    /// Whether the app can operate at all.  Microphone is strictly required; Accessibility is
    /// required for `type` mode but not for `paste` mode.
    pub fn ok_for(&self, output_mode: &str) -> bool {
        if output_mode == "clipboard" {
            self.microphone
        } else {
            self.microphone && self.accessibility
        }
    }

    /// Human-guided hints for whatever is missing (empty when everything is fine).
    pub fn guidance(&self, output_mode: &str) -> Vec<String> {
        let mut hints = Vec::new();
        if !self.microphone {
            hints.push(
                "Microphone access is not available — open System Settings ▸ Privacy & Security ▸ \
                 Microphone and enable Zipscribe."
                    .to_string(),
            );
        }
        if output_mode != "clipboard" && !self.accessibility {
            hints.push(
                "Accessibility access is needed to type into other apps' windows — open System \
                 Settings ▸ Privacy & Security ▸ Accessibility and ensure this app is enabled; or set \
                 output_mode=\"clipboard\" in config."
                    .to_string(),
            );
        }
        hints
    }
}

/// Detect current macOS permissions for this process.
pub fn check_permissions() -> PermissionReport {
    PermissionReport {
        accessibility: accessibility_trusted(),
        microphone: mic_status() == MicStatus::Authorized,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("not permitted to deliver output: {0}")]
    Permission(String),
    #[error("output simulation failed: {0}")]
    Sim(#[from] enigo::InputError),
    #[error("could not create input simulation (permissions?): {0}")]
    NewCon(#[from] enigo::NewConError),
}

/// Deliver `text` into the currently focused window per `config.output_mode`.
///
/// On a permission failure we emit a guided message and **degrade gracefully** (for `type`, fall
/// back to a paste) instead of crashing.
pub fn deliver(text: &str, config: &crate::config::Config) -> Result<(), OutputError> {
    if text.trim().is_empty() {
        return Ok(());
    }
    let accessibility = accessibility_trusted();

    let out_text = format!(
        "{}{}",
        text,
        if config.type_trailing_newline { "\n" } else { "" }
    );

    if config.output_mode == "clipboard" || !accessibility {
        // Without Accessibility we cannot press keys: leave the text on the clipboard instead.
        log::info!("[output] copying to clipboard only (accessibility={accessibility})");
        return copy_to_clipboard(&out_text);
    }
    if config.output_mode == "paste" {
        log::info!("[output] deliver via paste");
        return write_and_paste(&out_text);
    }

    // Type path: pure key simulation.
    log::info!("[output] deliver via key simulation ({} chars)", out_text.chars().count());
    type_text(&out_text)
}

/// Pure key simulation of a full string via enigo.
fn type_text(text: &str) -> Result<(), OutputError> {
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.text(text)?;
    Ok(())
}

/// Put `text` on the macOS pasteboard via `pbcopy` (stdin avoids quoting issues).
fn copy_to_clipboard(text: &str) -> Result<(), OutputError> {
    use std::io::Write as _;
    let mut child = std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| OutputError::Permission(format!("pbcopy failed: {e}")))?;
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(text.as_bytes());
    }
    let _ = child.wait();
    Ok(())
}

/// Copy `text` to the pasteboard, then press ⌘V into the focused window.
fn write_and_paste(text: &str) -> Result<(), OutputError> {
    copy_to_clipboard(text)?;
    press_cmd_v()
}

/// Press ⌘ then V to paste into whatever window currently has focus.
fn press_cmd_v() -> Result<(), OutputError> {
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.key(Key::Meta, Direction::Press)?;
    enigo.key(Key::Unicode('v'), Direction::Click)?;
    enigo.key(Key::Meta, Direction::Release)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guidance_for_missing_mic_is_present() {
        let r = PermissionReport {
            accessibility: true,
            microphone: false,
        };
        let hints = r.guidance("type");
        assert!(
            hints.iter().any(|h| h.contains("Microphone")),
            "no mic guidance: {hints:?}"
        );
    }

    #[test]
    fn guidance_for_missing_accessibility_in_type_mode() {
        let r = PermissionReport {
            accessibility: false,
            microphone: true,
        };
        let hints = r.guidance("type");
        assert!(
            hints.iter().any(|h| h.contains("Accessibility")),
            "no ax guidance: {hints:?}"
        );
        // "clipboard" mode doesn't need accessibility.
        let hints2 = r.guidance("clipboard");
        assert!(
            !hints2.iter().any(|h| h.contains("Accessibility")),
            "clipboard mode shouldn't flag ax: {hints2:?}"
        );
    }

    #[test]
    fn ok_for_requires_mic_always() {
        assert!(!PermissionReport::default().ok_for("type"));
        assert!(!PermissionReport {
            accessibility: true,
            microphone: false
        }
        .ok_for("clipboard"));
    }
}

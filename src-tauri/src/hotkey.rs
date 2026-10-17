//! Global Push-to-Talk (P2T) hotkey detection over `global-hotkey`.
//!
//! `global-hotkey` registers a system-wide macOS hotkey on a background thread and
//! surfaces press/release on `GlobalHotKeyEvent::receiver()`. For P2T we register the
//! configured combo once and poll:
//!   - `HotKeyState::Pressed`  => begin mic capture
//!   - `HotKeyState::Released` => stop capture (and, in the full pipeline, transcribe/clean/type)
//!
//! The hotkey string comes from `Config` and is parsed with `global-hotkey`'s `FromStr`
//! (e.g. "ctrl+shift+space" -> CONTROL|SHIFT + Code::Space), so the hotkey is fully configurable.

use crate::config::Config;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// A parsed hotkey, validated before we spend a system resource on it.
pub fn parse_hotkey(cfg: &Config) -> Result<HotKey, String> {
    cfg.hotkey.parse().map_err(|e: global_hotkey::hotkey::HotKeyParseError| {
        format!("invalid hotkey '{}': {e}", cfg.hotkey)
    })
}

/// Drain the global event channel, returning the first Press/Release for hotkey `id`.
/// Safe to call from any thread; only the manager must live on the main thread.
pub fn poll_id(id: u32) -> P2TEvent {
    while let Ok(ev) = GlobalHotKeyEvent::receiver().try_recv() {
        if ev.id == id {
            return match ev.state {
                HotKeyState::Pressed => P2TEvent::Press,
                HotKeyState::Released => P2TEvent::Release,
            };
        }
    }
    P2TEvent::None
}

/// A P2T controller that owns the system-wide hotkey registration.
pub struct P2T {
    _manager: GlobalHotKeyManager,
    hotkey: HotKey,
}

/// The result of a single poll cycle.
#[derive(Debug, PartialEq, Eq)]
pub enum P2TEvent {
    /// Hotkey went from up -> down: begin recording.
    Press,
    /// Hotkey went from down -> up: stop recording.
    Release,
    /// No pending hotkey event this cycle.
    None,
}

impl P2T {
    /// Build the controller: create the manager, parse + register the configured combo.
    /// Requires the macOS hotkey subsystem to be running (true in a GUI login session,
    /// which a LaunchAgent provides).
    pub fn new(cfg: &Config) -> Result<Self, String> {
        let manager =
            GlobalHotKeyManager::new().map_err(|e| format!("hotkey manager init: {e}"))?;
        let hotkey = parse_hotkey(cfg)?;
        manager
            .register(hotkey)
            .map_err(|e| format!("register {}: {e}", cfg.hotkey))?;
        log::info!("[hotkey] P2T listening on '{}' (id {})", cfg.hotkey, hotkey.id());
        Ok(Self {
            _manager: manager,
            hotkey,
        })
    }

    /// Swap to the combo in `cfg`. On failure the previous hotkey stays registered.
    /// Must be called on the thread that created the manager (the main thread on macOS).
    pub fn rebind(&mut self, cfg: &Config) -> Result<u32, String> {
        let new = parse_hotkey(cfg)?;
        if new.id() == self.hotkey.id() {
            return Ok(new.id());
        }
        self._manager
            .register(new)
            .map_err(|e| format!("register {}: {e}", cfg.hotkey))?;
        let _ = self._manager.unregister(self.hotkey);
        self.hotkey = new;
        log::info!("[hotkey] P2T now listening on '{}' (id {})", cfg.hotkey, new.id());
        Ok(new.id())
    }

    /// The hotkey id this controller is watching.
    pub fn id(&self) -> u32 {
        self.hotkey.id()
    }

    /// Drain the event channel once, returning the first event matching our hotkey.
    /// Returns `P2TEvent::None` if nothing matching is pending right now.
    ///
    /// The channel is a `crossbeam` receiver, so `try_recv()` yields `Err(Empty)` when
    /// idle and `Err(_)` if disconnected; both map to `None` here.
    pub fn poll(&self) -> P2TEvent {
        poll_id(self.hotkey.id())
    }

    /// Run the P2T loop, calling `handle` for each matching Press/Release. Stops when a
    /// handler returns `false`, or after `max_idle_cycles` consecutive empty cycles
    /// (0 = run forever). Used by demos to exit cleanly when idle.
    pub fn run<F>(&self, max_idle_cycles: u32, mut handle: F)
    where
        F: FnMut(P2TEvent, u32) -> bool,
    {
        let cycle = std::time::Duration::from_millis(10);
        let mut idle = 0u32;
        loop {
            match self.poll() {
                P2TEvent::None => {
                    idle += 1;
                    if max_idle_cycles > 0 && idle >= max_idle_cycles {
                        log::info!(
                            "[hotkey] {} idle cycles with no P2T event; exiting demo loop",
                            max_idle_cycles
                        );
                        break;
                    }
                    std::thread::sleep(cycle);
                }
                ev => {
                    idle = 0;
                    if !handle(ev, self.hotkey.id()) {
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use global_hotkey::hotkey::{Code, Modifiers};

    #[test]
    fn parses_default_ctrl_shift_space() {
        let mut cfg = Config::default();
        cfg.hotkey = "ctrl+shift+space".into();
        let hk = parse_hotkey(&cfg).expect("should parse default combo");
        assert_eq!(hk.mods, Modifiers::CONTROL | Modifiers::SHIFT);
        assert_eq!(hk.key, Code::Space);
    }

    #[test]
    fn parses_command_shift_d() {
        let mut cfg = Config::default();
        cfg.hotkey = "cmd+shift+d".into();
        let hk = parse_hotkey(&cfg).expect("should parse");
        assert_eq!(hk.mods, Modifiers::META | Modifiers::SHIFT);
        assert_eq!(hk.key, Code::KeyD);
    }

    #[test]
    fn rejects_invalid_combo() {
        let mut cfg = Config::default();
        cfg.hotkey = "not a combo".into();
        assert!(parse_hotkey(&cfg).is_err());
    }

    #[test]
    fn hotkey_id_is_deterministic() {
        // Same combo => same id (deterministic key/mod encoding).
        let mut a = Config::default();
        a.hotkey = "cmd+shift+d".into();
        let mut b = Config::default();
        b.hotkey = "cmd+shift+d".into();
        assert_eq!(
            parse_hotkey(&a).unwrap().id(),
            parse_hotkey(&b).unwrap().id()
        );
    }
}

// Shared helpers for the setup and settings windows.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (sel, root = document) => root.querySelector(sel);

function toast(msg) {
  let t = $(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}

function chip(el, text, kind) {
  el.textContent = text;
  el.className = "chip " + (kind || "");
}

const KEY_NAMES = { Space: "space", Enter: "enter", Tab: "tab", Backspace: "backspace", Escape: "escape" };
// Turn a keydown event into a hotkey string ("ctrl+shift+space"), or null while only modifiers are held.
function hotkeyFromEvent(e) {
  if (["Control", "Shift", "Alt", "Meta"].includes(e.key)) return null;
  let key;
  if (e.code.startsWith("Key")) key = e.code.slice(3).toLowerCase();
  else if (e.code.startsWith("Digit")) key = e.code.slice(5);
  else if (/^F\d+$/.test(e.code)) key = e.code.toLowerCase();
  else key = KEY_NAMES[e.code] || e.code.toLowerCase();
  const mods = [];
  if (e.ctrlKey) mods.push("ctrl");
  if (e.altKey) mods.push("alt");
  if (e.shiftKey) mods.push("shift");
  if (e.metaKey) mods.push("cmd");
  return [...mods, key].join("+");
}

const PRETTY = { ctrl: "⌃", alt: "⌥", shift: "⇧", cmd: "⌘", space: "Space", enter: "Return", tab: "Tab", escape: "Esc", backspace: "⌫" };
function prettyHotkey(s) {
  return s.split("+").map((p) => PRETTY[p] || p.toUpperCase()).join(" ");
}

// Click-to-record hotkey field. `onChange(hotkeyString)` fires after the backend accepts the combo.
function hotkeyRecorder(el, initial, onChange) {
  let value = initial;
  const show = () => { el.textContent = prettyHotkey(value); el.classList.remove("rec"); };
  show();
  let recording = false;
  const stop = () => { recording = false; window.removeEventListener("keydown", onKey, true); show(); };
  async function onKey(e) {
    e.preventDefault(); e.stopPropagation();
    if (e.code === "Escape") return stop();
    const combo = hotkeyFromEvent(e);
    if (!combo) return;
    if (!/ctrl|alt|shift|cmd/.test(combo) && !/^f\d+$/.test(combo)) {
      toast("Include a modifier key (⌃ ⌥ ⇧ ⌘)");
      return;
    }
    try { await invoke("validate_hotkey", { hotkey: combo }); }
    catch (err) { toast(String(err)); return; }
    value = combo;
    stop();
    onChange(value);
  }
  el.addEventListener("click", () => {
    if (recording) return;
    recording = true;
    el.textContent = "Press new shortcut… (Esc cancels)";
    el.classList.add("rec");
    window.addEventListener("keydown", onKey, true);
  });
  return { set(v) { value = v; show(); } };
}

const fmtMB = (mb) => (mb >= 1000 ? (mb / 1000).toFixed(1) + " GB" : mb + " MB");
const fmtBytes = (b) => fmtMB(Math.round(b / 1e6));

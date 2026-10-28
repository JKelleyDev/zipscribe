#!/bin/sh
# Build a plain drag-to-Applications .dmg from the bundled app (no Finder scripting, so it works
# headless/CI). Always rebuilds the app bundle first so the DMG never ships a stale build.
set -e
cd "$(dirname "$0")/.."
npx tauri build
APP=src-tauri/target/release/bundle/macos/Zipscribe.app
# MLX needs its compiled GPU kernels (mlx.metallib) next to the binary. Without this the app only
# works on the build machine, via the absolute build path MLX bakes in as a fallback.
LIB=$(ls -t $(find src-tauri/target/release/build -path '*mlxrs-sys-*' -name mlx.metallib) 2>/dev/null | head -n 1)
[ -n "$LIB" ] || { echo "mlx.metallib not found — is full Xcode + MetalToolchain installed?" >&2; exit 1; }
cp "$LIB" "$APP/Contents/MacOS/mlx.metallib"
codesign --force --deep --sign - "$APP"
OUT=src-tauri/target/release/bundle/Zipscribe.dmg
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "Zipscribe" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null
rm -rf "$STAGE"
echo "$OUT"

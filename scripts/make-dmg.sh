#!/bin/sh
# Build a plain drag-to-Applications .dmg from the bundled app (no Finder scripting, so it works
# headless/CI). Run `npx tauri build` first.
set -e
cd "$(dirname "$0")/.."
APP=src-tauri/target/release/bundle/macos/Zipscribe.app
OUT=src-tauri/target/release/bundle/Zipscribe.dmg
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "Zipscribe" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null
rm -rf "$STAGE"
echo "$OUT"

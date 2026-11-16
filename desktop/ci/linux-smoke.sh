#!/usr/bin/env bash
set -euo pipefail

work=$(mktemp -d '/tmp/验证 Portable.XXXXXX')
tar -xzf /packages/NanoVoice-Linux-x86_64.tar.gz -C "$work"
export QT_QPA_PLATFORM="$DISPLAY_BACKEND"
export HF_HUB_OFFLINE=1
export XDG_RUNTIME_DIR="$work/runtime"
mkdir -m 700 "$XDG_RUNTIME_DIR"

if [[ "$DISPLAY_BACKEND" == wayland ]]; then
  export WAYLAND_DISPLAY=moss-test
  weston --backend=headless --socket="$WAYLAND_DISPLAY" --idle-time=0 > "$work/display.log" 2>&1 &
else
  Xvfb -displayfd 3 -screen 0 1280x800x24 -nolisten tcp 3> "$work/display-number" > "$work/display.log" 2>&1 &
fi
display_pid=$!
trap 'kill "$display_pid" 2>/dev/null || true' EXIT

for attempt in {1..50}; do
  if [[ -S "$XDG_RUNTIME_DIR/${WAYLAND_DISPLAY:-unused}" || -s "$work/display-number" ]]; then
    break
  fi
  if ! kill -0 "$display_pid" 2>/dev/null; then
    cat "$work/display.log"
    exit 1
  fi
  sleep 0.2
done

if [[ "$DISPLAY_BACKEND" == xcb ]]; then
  export DISPLAY=":$(cat "$work/display-number")"
fi
chmod +x "$work/NanoVoice/NanoVoice"
if ! timeout 300 "$work/NanoVoice/NanoVoice" --self-test || ! test -f "$work/NanoVoice/data/self-test.ok"; then
  cat "$work/display.log"
  cat "$work/NanoVoice/data/logs/desktop.log" 2>/dev/null || true
  exit 1
fi

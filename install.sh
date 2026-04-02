#!/usr/bin/env bash
# Install racing-telemetry as an mpv script directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detect mpv config directory
if [ -n "${MPV_HOME:-}" ]; then
    MPV_CONF="$MPV_HOME"
elif [ -n "${XDG_CONFIG_HOME:-}" ]; then
    MPV_CONF="$XDG_CONFIG_HOME/mpv"
elif [ -d "$HOME/.config/mpv" ] || [ "$(uname)" != "Darwin" ]; then
    MPV_CONF="$HOME/.config/mpv"
else
    MPV_CONF="$HOME/.config/mpv"
fi

TARGET="$MPV_CONF/scripts/racing-telemetry"

mkdir -p "$MPV_CONF/scripts"

# Remove old single-file installs if present
rm -f "$MPV_CONF/scripts/telemetry.lua"

# Symlink the whole repo as a script directory (mpv loads main.lua automatically)
if [ -L "$TARGET" ]; then
    rm "$TARGET"
elif [ -d "$TARGET" ]; then
    echo "Warning: $TARGET exists and is not a symlink. Remove it first."
    exit 1
fi

ln -s "$SCRIPT_DIR" "$TARGET"
echo "Installed → $TARGET"
echo "mpv will auto-load racing-telemetry on next launch."

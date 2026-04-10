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

# Configure OSC custom button for telemetry toggle
OSC_CONF="$MPV_CONF/script-opts/osc.conf"
mkdir -p "$MPV_CONF/script-opts"

add_osc_option() {
    local key="$1" val="$2"
    if [ -f "$OSC_CONF" ] && grep -q "^${key}=" "$OSC_CONF"; then
        # Already configured
        return
    fi
    echo "${key}=${val}" >> "$OSC_CONF"
}

# Find first unused custom button slot
SLOT=1
if [ -f "$OSC_CONF" ]; then
    while grep -q "^custom_button_${SLOT}_content=" "$OSC_CONF"; do
        # Skip if it's already our telemetry button
        if grep -q "^custom_button_${SLOT}_mbtn_left_command=script-binding.*toggle-telemetry" "$OSC_CONF"; then
            SLOT=0; break
        fi
        SLOT=$((SLOT + 1))
    done
fi

if [ "$SLOT" -gt 0 ]; then
    add_osc_option "custom_button_${SLOT}_content" "⏱"
    add_osc_option "custom_button_${SLOT}_mbtn_left_command" "script-binding toggle-telemetry"
    add_osc_option "custom_button_${SLOT}_mbtn_right_command" "script-binding toggle-calibration"
    echo "Added telemetry button to OSC (slot $SLOT) → $OSC_CONF"
fi

echo "mpv will auto-load racing-telemetry on next launch."
echo "  Left-click ⏱ button = toggle telemetry"
echo "  Right-click ⏱ button = calibration mode"

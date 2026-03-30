# Racing Telemetry — mpv Lua Script

Reads baked-in telemetry overlays from onboard racing camera videos by sampling raw pixel data in memory and re-renders them as a clean ASS overlay with scrolling traces, gear indicator, steering wheel, and pedal bars.

## How it works

1. **Calibrate** (Ctrl+C) — Pause the video, draw rectangles over each telemetry element (throttle bar, brake bar, gear digit, steering indicator, speed, fuel). The script auto-detects colors and measurement types.

2. **Display** — During playback, the script samples pixels from the calibrated regions via `screenshot-raw` and renders a smooth ASS overlay synced to the video position.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+T` | Toggle telemetry overlay |
| `Ctrl+C` | Enter/exit calibration mode |
| `Ctrl+G` | Cycle overlay position |
| `Ctrl+= / -` | Resize overlay |
| `Ctrl+N` | Cycle through saved calibrations |

### Calibration mode

| Key | Action |
|-----|--------|
| `1-6` | Select channel (1=throttle 2=brake 3=gear 4=steering 5=speed 6=fuel) |
| Click+drag | Draw rectangle for selected channel |
| `C` | Pick color (click on active/filled color) |
| `M` | Set center point (for steering) |
| `S` | Save calibration (prompts for name) |
| `Escape` / `Space` | Exit calibration |

## Installation

Copy or symlink `telemetry.lua` into your mpv scripts directory:

```bash
ln -s "$(pwd)/telemetry.lua" ~/.config/mpv/scripts/telemetry.lua
```

## Configuration

Calibrations are saved to `~/.config/mpv/telemetry-configs/`. Includes a built-in default for TDS Racing IMSA videos (1280×720).

## Requirements

- mpv

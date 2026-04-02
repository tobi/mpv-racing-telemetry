# Racing Telemetry for mpv

Reads baked-in telemetry overlays from onboard racing videos and re-renders them as a clean real-time overlay with scrolling traces, gear indicator, steering wheel, and pedal bars.

![screenshot placeholder]

## Install

### 1. Install mpv

**macOS (Homebrew):**
```bash
brew install mpv
```

**Arch Linux:**
```bash
sudo pacman -S mpv
```

**Ubuntu/Debian:**
```bash
sudo apt install mpv
```

### 2. Install the plugin

```bash
git clone https://github.com/tobi/racing-telemetry-mpv.git ~/.config/mpv/scripts/racing-telemetry
```

That's it — mpv auto-loads any `main.lua` it finds in `~/.config/mpv/scripts/*/`.

Alternatively, if you want to clone it elsewhere and symlink:

```bash
git clone https://github.com/tobi/racing-telemetry-mpv.git
cd racing-telemetry-mpv && ./install.sh
```

### Optional: digit recognition (OCR)

The gear/speed OCR uses ONNX Runtime. Install it if you want digit recognition:

**macOS:** `brew install onnxruntime`
**Arch:** `yay -S onnxruntime` (or `sudo pacman -S onnxruntime`)
**Ubuntu:** See [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)

Without it, everything works — gear is read via color matching instead of OCR.

## Usage

Open any racing onboard video in mpv:

```bash
mpv your-video.mp4
```

| Key | Action |
|-----|--------|
| `Ctrl+T` | Toggle telemetry overlay |
| `Ctrl+C` | Enter/exit calibration mode |
| `Ctrl+G` | Cycle overlay position |
| `Ctrl+= / -` | Resize overlay |
| `Ctrl+N` | Cycle through saved calibrations |

### Calibration

1. Pause the video, press `Ctrl+C`
2. Press `1`–`6` to select a channel (throttle, brake, gear, steering, speed, fuel)
3. Click+drag a rectangle over each telemetry element
4. Press `C` then click to pick the active color
5. Press `S` to save — calibrations persist across sessions

Comes with a built-in default for TDS Racing IMSA 1280×720 videos.

## Uninstall

```bash
rm ~/.config/mpv/scripts/racing-telemetry
```

## Development

```bash
just check    # run tests
just build    # build optional ONNX C bridge (not needed — pure FFI is default)
```

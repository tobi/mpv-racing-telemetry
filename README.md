# Racing Telemetry for mpv

Reads baked-in telemetry overlays from onboard racing videos and re-renders them as a clean real-time overlay with scrolling traces, gear indicator, steering wheel, and pedal bars.

![Racing telemetry overlay screenshot](./assets/screeenshot.png)

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

**Windows:** install a build of mpv that includes LuaJIT. The official/shinchiro
Windows builds generally work.

### 2. Install the plugin

Download the latest ready-to-install zip from [GitHub Releases](https://github.com/tobi/mpv-racing-telemetry/releases/latest), then extract the `racing-telemetry` folder into your mpv scripts directory.

**macOS/Linux git install:**
```bash
git clone https://github.com/tobi/mpv-racing-telemetry.git ~/.config/mpv/scripts/racing-telemetry
```

**Windows installed mpv git install:**
```powershell
git clone https://github.com/tobi/mpv-racing-telemetry.git "$env:APPDATA\mpv\scripts\racing-telemetry"
```

**Windows portable mpv:** extract the release zip, or clone this repo, into:
```text
<mpv folder>\portable_config\scripts\racing-telemetry
```

That's it — mpv auto-loads any `main.lua` it finds in the script directory.

Alternatively, on macOS/Linux, if you want to clone it elsewhere and symlink:

```bash
git clone https://github.com/tobi/racing-telemetry-mpv.git
cd racing-telemetry-mpv && ./install.sh
```

### Optional: digit recognition (OCR)

The gear/speed OCR uses ONNX Runtime. Install it if you want digit recognition:

**Windows:** CPU ONNX Runtime DLLs are bundled in `third_party/onnxruntime/win-x64`, so no separate install is needed. You still need a LuaJIT-enabled mpv build.

**macOS:** `brew install onnxruntime`
**Arch:** `yay -S onnxruntime` (or `sudo pacman -S onnxruntime`)
**Ubuntu:** See [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)

Without ONNX Runtime or LuaJIT FFI, everything except neural OCR still works; the script falls back to the older pattern matcher.

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

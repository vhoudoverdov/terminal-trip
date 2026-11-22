# terminal-trip

A retro 90s/2000s music visualizer that runs right inside your terminal—inspired by Winamp's **Geiss / MilkDrop**, Windows Media Player's **Bars and Waves / Ambience**, and classic **demoscene wormholes and plasmas**.

>Looking for the under-the-hood DSP formulas, feedback warp math, lookup-table equations, and ANSI rendering architecture? See **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 1. Download & Run

### Option A: One-Step Install (`pip` / `pipx` / `uv`)
Clone the repository and install it into a virtual environment (or globally with `pipx`):

```powershell
git clone https://github.com/vhoudoverdov/terminal-trip.git
cd terminal-trip

# Create a virtual environment and install
python -m venv .venv
.\.venv\Scripts\pip install .

# Launch!
.\.venv\Scripts\terminal-trip
```

*(On macOS / Linux, replace `.\.venv\Scripts\` with `./.venv/bin/`.)*

### Option B: Strict Hash-Verified Install
If you want cryptographic SHA-256 verification of every downloaded wheel:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install --require-hashes -r requirements.txt
.\.venv\Scripts\python main.py
```

---

## 2. Common Launch Modes

```powershell
# React to whatever is playing on your computer (Spotify, YouTube, etc. via Windows WASAPI loopback)
python main.py

# Play and visualize an audio file (.wav, .mp3, .flac, .ogg)
python main.py --file path/to/song.mp3

# Run the built-in 125 BPM acid-techno synth and auto-cycle presets + palettes
python main.py --synth --auto

# Start on a specific preset (1-4) and color palette (1-6)
python main.py --preset 2 --palette 3
```

> **Tip:** Use **Windows Terminal** (or any modern true-color terminal). Shrinking your terminal font size (`Ctrl` + `-`) increases the pixel resolution of the canvas!

---

## 3. Controls & Cycling Patterns / Colors

While `terminal-trip` is running, use your keyboard to switch visual patterns, cycle color palettes, change audio inputs, or take screenshots:

| Key | Action | Description |
| :--- | :--- | :--- |
| **`1` / `2` / `3` / `4`** | **Select Visual Preset** | Jump directly to Preset 1–4 with a smooth crossfade |
| **`←` / `→`** | **Previous / Next Preset** | Cycle backward or forward through the 4 visual patterns |
| **`Space`** | **Toggle Auto-Cycle** | Automatically crossfade to a new preset and palette every ~20 seconds on a beat |
| **`P`** | **Cycle Color Palette** | Smoothly morph to the next 256-color palette (6 built-in palettes) |
| **`A`** | **Switch Audio Source** | Cycle between *System Loopback* $\rightarrow$ *Demo Synth* $\rightarrow$ *Audio File* |
| **`+` / `-`** | **Audio Sensitivity** | Boost or attenuate visual reactivity (`0.25x` to `4.0x` gain) |
| **`D`** | **Color Detail Level** | Cycle `256` $\rightarrow$ `128` $\rightarrow$ `64` color levels (lower = fewer ANSI codes for slower terminals) |
| **`H`** | **Toggle Bottom HUD** | Show or hide the status bar (active preset, palette, FPS, live Bass/Mid/Treble VU meter) |
| **`S`** | **Save PNG Screenshot** | Save the current frame as a crisp 4× upscaled PNG (`terminal-trip-YYYYMMDD-HHMMSS.png`) |
| **`?`** | **Toggle Help Overlay** | Show or hide the on-screen hotkey cheat sheet |
| **`Q` / `Esc`** | **Quit** | Cleanly restore the terminal screen and exit |

---

## 4. Built-in Visual Presets & Palettes

### Visual Presets (`1`–`4` or `←`/`→`)
1. **`[1]` Geiss Vortex**: Polar feedback warp (zoom + swirl + petal modulation) that pulls previous frames into a vortex while drawing live oscillating rings, ribbons, Lissajous waves, and beat sparks.
2. **`[2]` LUT Wormhole**: Classic demoscene `1/r` lookup-table tunnel with a wandering look-at center, $N$-fold kaleidoscope symmetry, XOR/plasma wall textures, and audio spectrum rings rushing toward the camera.
3. **`[3]` Bars & Waves**: Tribute to Windows Media Player—48 logarithmic spectrum bars with gravity peak-hold caps, rising heat trails, and a glowing oscilloscope overlay.
4. **`[4]` Plasma & Fire**: Four-wave sine interference plasma with contour bands combined with a classic demoscene cellular fire buffer pumped by the frequency spectrum and bass beats.

### Color Palettes (`P`)
1. **Geiss Fire** *(Indigo $\rightarrow$ Magenta $\rightarrow$ Electric Orange $\rightarrow$ White-Hot Yellow)*
2. **WMP Cobalt** *(Midnight Blue $\rightarrow$ Electric Cyan $\rightarrow$ Violet $\rightarrow$ Hot Pink)*
3. **Acid Matrix** *(Deep Emerald $\rightarrow$ Neon Lime $\rightarrow$ Cyan)*
4. **Neon Sunset** *(Crimson $\rightarrow$ Tangerine $\rightarrow$ Gold $\rightarrow$ Deep Violet)*
5. **Thermal Cosine** *(Continuous Inigo Quilez periodic cosine rainbow)*
6. **Infrared** *(Dark Red $\rightarrow$ Lava Orange $\rightarrow$ Incandescent White)*

---

## 5. Command-Line Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--file PATH` | *None* | Audio file to play and visualize (`.wav`, `.mp3`, `.flac`, `.ogg`) |
| `--synth` | *Off* | Start immediately with the built-in procedural 125 BPM demo synth |
| `--preset {1..4}` | `1` | Starting visual preset |
| `--palette {1..6}` | `1` | Starting 256-color palette |
| `--auto` | *Off* | Enable automatic preset & palette cycling on startup |
| `--fps FPS` | `60.0` | Target frame rate (`1`–`240`) |
| `--detail {0,1,2}` | `1` | Bit-shift quantization (`0` = 256 levels, `1` = 128 levels, `2` = 64 levels) |
| `--no-hud` | *Off* | Hide the bottom HUD bar on startup |
| `--bench N` | `0` | Headless benchmark mode: render `N` frames per preset and print timing stats |
| `--size COLSxROWS` | `120x40` | Canvas size in terminal cells for `--bench` mode |
| `--snapshot DIR` | *None* | Save one PNG snapshot per preset into `DIR` during `--bench` |

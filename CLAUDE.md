# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MangaRelief Pro is a Windows-targeted PyQt6 desktop app that converts 2D manga panels / grayscale artwork into 3D-printable relief models (STL and Bambu Studio-flavored 3MF). Grayscale ink intensity is quantized into discrete "terraces" mapped to physical Z heights, designed for multi-color filament-swap printing (M600 layer pauses).

## Commands

```bash
pip install -r requirements.txt
python manga_to_3d.py             # run the app (GUI — needs a display)
python test_topo_colors.py <image>  # manual smoke test of the topographic pipeline (K-Means posterize + STL export, no GUI)
build_exe.bat                     # Windows PyInstaller build (uses --add-data for style.qss, icon.ico, assets/)
```

There is no automated test suite, linter, or CI. `test_topo_colors.py` is a manual debug script that writes `debug_posterized.png` and `debug_topo_mesh.stl`.

## Architecture

The app is a single-window controller/worker split:

- **`manga_to_3d.py`** — entry point and `Manga3DAppController`, which *subclasses* `MainWindowUI`. All UI event wiring, image loading (OpenCV with PIL/pillow-heif fallback for AVIF/HEIC), color-mode state, and auto-Z computation live here. Generation is delegated to a `MeshWorker` QThread.
- **`ui_main_window.py`** — pure UI construction (`MainWindowUI`) plus `ImageGraphicsView`, a QGraphicsView with wheel-zoom, right/middle-click pan, and a `pixelClicked(x, y)` signal used for gray-value sampling.
- **`worker.py`** — `MeshWorker(QThread)` runs the whole generation pipeline: image resize/blur → gray quantization → piecewise-linear Z mapping (`np.interp`) → heightmap → solid mesh → optional decimation (`fast_simplification`, triggered above 200k faces) → export. Communicates via `progress(int, str)`, `finished_ok(str, str)`, `finished_err(str)` signals; cancellation is a `cancel_requested` flag polled inside `run()`.
- **`mesh_utils.py`** — `create_solid_mesh` (vectorized watertight heightmap-to-solid, top/bottom/4 sealed sides), `process_mesh_topo` (color-terrace mesh via cKDTree nearest-color + median filter), `compute_topo_z_heights` (layer-height-snapped Z distribution), and `export_3mf`, which exports via trimesh then rewrites the 3MF ZIP to inject Bambu Studio `Metadata/custom_gcode_per_layer.xml` (color changes at Z heights) and `Metadata/slice_info.config`.
- **`color_utils.py`** — K-Means helpers: `extract_dominant_colors` (RGB, sorted by luminance) and `suggest_midtones` (grayscale K=4, returns the two middle centers).
- **`config.py`** — `DeckboxConfig` dimensional constants and `SLOT_COLORS_3MF` filament slot colors.
- **`utils.py`** — `resource_path()` for PyInstaller `_MEIPASS`-aware asset resolution; always use it when loading `style.qss` or files from `assets/`.

### Three generation modes

`mode_selector` index drives the pipeline branch in both the controller and the worker:

1. **Standard (index 0)** — grayscale relief. Operates in 2/3/4-color sub-modes (`color_mode_state`), chosen automatically from the image's midtone percentage vs. the halftone-threshold slider. The sub-mode changes the gray→Z piecewise mapping and which L1/L2 UI rows are visible.
2. **Topographic (index 1)** — full-color: K-Means dominant colors (user-reorderable list, top = base) mapped to evenly distributed, layer-snapped terraces via `process_mesh_topo`.
3. **Deckbox (index 2)** — debosses the relief into a TCG deckbox front wall and engraves a game logo on the lid, using STL templates and logo JPGs from `assets/` (`TCG_LOGO_MAP` in worker.py). Exports a combined front+lid plate.

### Key conventions

- Z heights are always snapped to the printing layer height (`spin_layer_height`) and rounded to 3 decimals; the 3 color-change Z values (`color_changes_z`) flow from the controller into both mesh geometry and 3MF metadata.
- Output paths are auto-computed next to the source image (`output/stl/`, `output/3mf/`) with anti-overwrite numbering — no save dialog.
- Sampled grays are ordered `[L0 white bg, L1 light, L2 dark, L3 black]` in `sampled_colors`; `white_clip`/`black_clip` hard-clamp extremes to flat base/top surfaces.
- UI strings and code comments are a mix of Italian and English; match the surrounding style.
- Styling comes from `style.qss` (loaded at startup); swatch/button state changes require the `unpolish`/`polish` refresh idiom already used throughout.

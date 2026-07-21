# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch scope

This is `feature/phone-cover`, which descends from `feature/spot-color` ← `feature/topo-color-mode`. It carries the full feature set: Standard, Topographic Color, Deckbox, Spot Color, and Phone Cover generation modes. Other branches (`main`, `feature/topo-color-mode`, `feature/spot-color`) represent earlier release points on the same project — do not assume they have these files.

## Project Overview

MangaRelief Pro is a PyQt6 desktop app that quantizes manga grayscale (or full-color art) into "terraced" 3D relief/engraved meshes for multi-color 3D printing (STL + Bambu Studio-flavored 3MF). Five generation modes share one geometry/color engine:

1. **Standard** — grayscale relief, 2/3/4-color sub-modes auto-selected from midtone %.
2. **Topographic Color** — K-Means dominant-color terraces from a full-color image.
3. **Deckbox** — debosses the relief into a TCG deckbox front wall + engraves a logo on the lid.
4. **Spot Color** — "silkscreen" mode: white base + 1-2 user-picked accent colors + black top, everything else binarized. Built for print accessibility (base + accent, not exact multi-color fidelity).
5. **Phone Cover Plate** — generates a decorative back plate (multi-color, engraved or raised) sized to a specific phone's camera-cutout geometry, optionally paired with a companion TPU bumper/case STL.

## Commands

```bash
pip install -r requirements.txt   # scipy/sklearn/fast-simplification/shapely/manifold3d/mapbox-earcut all required
python manga_to_3d.py             # run the app (GUI — needs a display)
python test_topo_colors.py <image>  # manual smoke test of K-Means posterize + STL export, no GUI
build_exe.bat                     # Windows PyInstaller build (hidden-imports kept in sync with requirements.txt)
```

No automated test suite or linter exists. When validating changes in this repo, write throwaway pytest-less scripts (`python - <<'EOF' ... EOF`) that exercise `MeshWorker.run()` end-to-end and assert on the resulting mesh (watertight, `Z` quantile set, face count) — that is the pattern used throughout this branch's commit history and catches the geometry regressions that matter (winding, terrace snapping, mask holes).

Qt widget tests must run with `QT_QPA_PLATFORM=offscreen` and **must call `win.show()`** before asserting `isVisible()` — without a shown top-level window, Qt reports every child widget as not visible regardless of `setVisible(True)`, which produced a false failure earlier in this branch's history.

## Architecture

- **`manga_to_3d.py`** — `Manga3DAppController(MainWindowUI)`. All UI event wiring, image loading, per-mode state (Spot accents, Cover composition/zoom/offset), and the export-success popup text (`_build_color_change_instructions`) live here. Generation itself is delegated to `MeshWorker`.
- **`ui_main_window.py`** — pure UI construction (`MainWindowUI`) + `ImageGraphicsView` (wheel-zoom/pan/`pixelClicked` signal). `_on_mode_changed` toggles per-mode group visibility. `self.lockable_widgets` is a flat registry of every widget that must disable during generation — **add new controls to this list, not to `toggle_ui_state`**, which just iterates the registry and restores mode-conditional states (auto-Z, auto-midtones, Deckbox-locked physical params, Cover levels selector) afterward.
- **`worker.py`** — `MeshWorker(QThread)`. One `run()` method branches by mode: prepares/composes the source image → classifies pixels into a palette → builds the heightmap/terrace mesh → optional decimation (`fast_simplification`, >200k faces) → export. Signals: `progress(int,str)`, `finished_ok(stl_path,3mf_path)`, `finished_err(str)`. Cancellation is cooperative via `self.cancel_requested` checked by `_check_cancel()` between stages.
- **`mesh_utils.py`** — geometry core:
  - `create_solid_mesh(X,Y,Z,bottom_z,mask=None)` — vectorized heightmap→watertight-solid. With `mask`, only cells fully inside the boolean mask get top/bottom faces, and **every** boundary edge (outer outline *and* interior holes) gets a sealed vertical wall — this is what makes the Phone Cover plate's camera cutouts and rounded silhouette printable.
  - `rounded_rect_mask(h,w,radius_px,holes=[(x,y,w,h,r),...])` — plate silhouette generator.
  - `process_mesh_topo(...)` — shared terrace-mesh pipeline used by Topographic, Spot Color, *and* Phone Cover (Cover composes onto a palette first, then calls this with its plate `mask`). Takes `min_feature_mm` (island/fringe cleanup threshold — default 0.5mm for panel-scale prints, lowered to 0.25–0.35mm by Cover mode since a small plate's engraved groove is missing material, not a printed wall, so it tolerates finer features) and `max_res_cap`.
  - `compute_topo_z_heights` / `compute_topo_switch_z` — terrace Z quantization and the filament-switch Z (which is `z[i-1] + layer_height`, **not** `z[i]` — the switch happens one layer above the previous terrace's top, a fix that took a full debugging pass; don't regress it).
  - `export_3mf(mesh, path, color_changes_z, slot_colors=None)` — trimesh-native 3MF + injected Bambu `custom_gcode_per_layer.xml`/`slice_info.config`. Filters `color_changes_z` to `>0` and dedupes by rounding — **be careful introducing new switch-Z values that could collide after rounding**, a dedup collision silently drops a color layer from the printed object (see git history around the Cover engraved-surface debugging).
- **`color_utils.py`** — all pixel classification, shared across modes:
  - `rgb_to_lab(..., chroma_weight=...)` — Lab conversion with amplified a/b channels; `CHROMA_MATCH_WEIGHT=2.5` prevents neutral grays from matching saturated palette colors (was the root cause of "red bleeding onto black/white edges" in Topographic mode).
  - `extract_dominant_colors` / `merge_lab_clusters` / `downsample_for_analysis` — Topographic's K-Means-in-Lab-with-cluster-merging, shared helper functions.
  - `suggest_spot_accents` / `classify_spot_pixels` / `build_spot_palette` — Spot Color engine (hue-distance matching + coverage-gated saturation threshold, not Lab distance — an earlier Lab-based attempt made the coverage slider nearly inert).
  - `quantize_grayscale_levels` / `grayscale_palette` — Cover mode's B/N finish. **Do not re-implement this with a fresh per-call K-Means(k=n_levels)** — it's unstable when the image's real tonal populations don't match `n_levels` (a small noise/anti-aliasing cluster can steal the middle slot from the real dominant midtone). The current design always runs the same K=4 clustering Standard mode uses (`suggest_midtones`'s approach) and picks the matching landmark subset per `n_levels`, mirroring how Standard mode's 2/3/4-color selector hides L1/L2.
- **`case_utils.py`** — Phone Cover mode, two independent ways to get a plate+companion pair:
  - **Parametric** (`build_bumper`, `compute_plate_dims`): a TPU bumper built from 4 stacked shapely ring cross-sections (back frame / plate groove / phone cavity / front screen-guard lip) via `trimesh.boolean` (needs the `manifold` engine — `manifold3d` package). Cutouts (`side_cutouts`, `top_cutouts`, `bottom_opening`) are expressed **looking at the phone's back** (matches the artwork's viewing orientation) and are mirrored internally to model space — see the docstring on `build_bumper` before changing cutout math.
  - **Real-seat retrofit** (`carve_plate_recess`): given any third-party case STL plus measured cavity/wall coordinates, cuts a plate-shaped window + undercut pocket into it, preserving named keep-zones (e.g. an existing camera-lens block). Presets with a `case_plate` key (see `assets/phone_presets.json`) use this real, ruler-verified seat instead of the parametric system — `build_case_plate_raster` fills the measured outline polygon directly.
  - `compose_plate_art` / `compose_cover_art` — fits the source image onto the plate raster (cover-fill with user zoom/offset in mm), optionally excluding the area under the camera holes (`avoid_camera`).
  - **Engraved vs Raised** is a single palette-order inversion (`palette[::-1]`, indices flipped) done once in `worker.py`'s Cover branch — everything downstream (terraces, 3MF slot colors, popup text) follows from that one flip. The convention: index 0 of the palette always gets the *smallest* Z (`base_h`) and the last index gets the *largest* Z (`max_h`), since `create_solid_mesh`'s `bottom_z` is uniformly flat — so whichever color is last in palette order becomes the flush outward-facing plane.
  - `assets/phone_presets.json` — per-phone geometry, coordinates measured **looking at the phone's back**, origin at the top-left corner of the phone body. `_note`/`_todo` keys inside are human documentation, filtered out at load time (`k.startswith('_')`).

### Key conventions carried from earlier modes (still apply)

- Z heights always snap to `layer_height` and round to 3 decimals.
- Standard-mode sampled grays: `[L0 white bg, L1 light, L2 dark, L3 black]`.
- Output paths are auto-computed next to the source image with anti-overwrite numbering; Phone Cover uses `cover_plate_<image>.stl/.3mf` + `cover_bumper_<image>.stl` (`MeshWorker.companion_path_for` derives the latter from the former — keep them in sync if you change the naming scheme).
- `resource_path()` (in `utils.py`) for any file under `assets/` — required for PyInstaller `_MEIPASS` compatibility.
- UI/code comments mix Italian and English; match the surrounding style in the file you're editing.

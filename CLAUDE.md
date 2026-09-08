# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch scope

This is `feature/keychain-cutout`, which descends from `feature/engine-extraction` ← `feature/phone-cover` ← `feature/spot-color` ← `feature/topo-color-mode`. It carries the full feature set: Standard, Topographic Color, Deckbox, Spot Color and Phone Cover generation modes, the generation logic extracted into a PyQt-free `engine/` package (see Architecture), and the Cutout / Keychain option on top. Other branches (`main`, `feature/topo-color-mode`, `feature/spot-color`, `feature/phone-cover`, `feature/engine-extraction`) represent earlier release points on the same project — do not assume they have these files.

## Project Overview

MangaRelief Pro is a PyQt6 desktop app that quantizes manga grayscale (or full-color art) into "terraced" 3D relief/engraved meshes for multi-color 3D printing (STL + Bambu Studio-flavored 3MF). Six generation modes share one geometry/color engine:

1. **Standard** — grayscale relief, 2/3/4-color sub-modes auto-selected from midtone %.
2. **Topographic Color** — K-Means dominant-color terraces from a full-color image.
3. **Deckbox** — debosses the relief into a TCG deckbox front wall + engraves a logo on the lid.
4. **Spot Color** — "silkscreen" mode: white base + 1-2 user-picked accent colors + black top, everything else binarized. Built for print accessibility (base + accent, not exact multi-color fidelity).
5. **Phone Cover Plate** — generates a decorative back plate (multi-color, engraved or raised) sized to a specific phone's camera-cutout geometry, optionally paired with a companion TPU bumper/case STL.

6. **Keychain / Cutout** — replaces the rectangular panel with the silhouette of the drawing (see `engine/cutout_utils.py`). Not a flag on the other modes: it is a mode, and like Phone Cover it carries a **Finish** selector (Spot Color / B&W) because the cutout decides the *shape*, not the colours.

## Commands

```bash
pip install -r requirements.txt   # scipy/sklearn/fast-simplification/shapely/manifold3d/mapbox-earcut all required
python manga_to_3d.py             # run the app (GUI — needs a display)
python test_topo_colors.py <image>  # manual smoke test of K-Means posterize + STL export, no GUI
python tests/copertura_2colori.py   # coverage slider + Standard mockup, engine and Qt (~2 min, no display needed)
python tests/ritaglio_sagoma.py     # cutout: regions, click, holes in the real mesh, Qt panel (~1 min)
python tests/tratto_minimo.py       # finest-stroke measure: calibration and per-mode mm/px (~20 s)
build_exe.bat                     # Windows PyInstaller build (hidden-imports kept in sync with requirements.txt)
```

No linter and no pytest. Checks are plain scripts that print `PASS`/`FAIL` and exit non-zero: they exercise `engine.generate()` end-to-end (and `MeshWorker` when the Qt plumbing itself is what changed) and assert on the resulting mesh — watertight, `Z` quantile set, face count. That is the pattern throughout this branch's history, and it catches the geometry regressions that matter (winding, terrace snapping, mask holes).

**`tests/` holds the ones worth keeping**; write a throwaway (`python - <<'EOF' ... EOF`) for a one-off, and move it into `tests/` the moment it covers something that could plausibly break again. The criterion is not size, it is whether re-running it in six months would tell you anything.

Three ways these assertions go wrong, each of which has happened here:

- **Measuring a proxy instead of the thing.** Counting vertices at the top Z is not the printed ink area: decimation crowds vertices where geometry is complicated, which is the opposite of area. Sum `area_faces` over the faces that lie at that Z.
- **Expecting the wrong shape.** A 2-colour relief has *three* Z values — the flat bottom at 0, the base, and the ink — not two.
- **A fixture that cannot tell the settings apart.** A test panel with one hatch density classifies the same at 25% and 70% coverage, so the check passes without checking anything. Build the fixture so the setting under test actually changes the answer.

Qt widget tests must run with `QT_QPA_PLATFORM=offscreen` and **must call `win.show()`** before asserting `isVisible()` — without a shown top-level window, Qt reports every child widget as not visible regardless of `setVisible(True)`, which produced a false failure earlier in this branch's history.

## Architecture

- **`manga_to_3d.py`** — `Manga3DAppController(MainWindowUI)`. All UI event wiring, image loading, per-mode state (Spot accents, Cover composition/zoom/offset), and the export-success popup text (`_build_color_change_instructions`) live here. Generation itself is delegated to `MeshWorker`.
- **Standard a 2 colori** — il pannello "Color Picking" diventa "Ink Coverage":
  gli swatch spariscono (L1/L2 non esistono, e quattro pulsanti di cui meta'
  inerti sono peggio di nessun pulsante) e al loro posto c'e' il cursore della
  copertura, che finisce in `GenerationParams.bw_coverage`. Lo scambio lo fa
  `_refresh_color_mode`, che e' gia' il punto in cui il pannello cambia forma
  con la modalita'; fuori dai 2 colori `_current_bw_coverage()` restituisce
  `None`, che nel motore e' la strada di sempre.
- **`btn_std_mockup`** — l'anteprima della classificazione Standard, gemella di
  quella Spot: stesso schema (pulsante bistabile, timer da 180 ms, `setImage`
  sul viewer). Esegue `prepare_source_image` + `standard_heightmap` e dipinge
  ogni pixel col tono campionato della sua banda — *non* posterizza e basta,
  perche' la banda dipende da `color_changes_z`, che con l'auto-Z spento e'
  scritto a mano. Senza anteprima il cursore della copertura sarebbe cieco.

- **Keychain panel** — `group_keychain` in `ui_main_window.py`, stato e click in `manga_to_3d.py` (`_cutout_*`). La finitura ne governa la forma esattamente come `combo_cover_finish` governa quella della cover: in Spot compare `group_spot`, in B/N tornano `group_swatch` e `group_z`. Tre cose non ovvie: (a) i click di modifica **accendono per forza l'anteprima**, perche' le coordinate vanno lette sul raster di segmentazione e l'unica immagine mostrata a quella risoluzione e' l'anteprima; (b) `_cutout_source()` deve restituire *la stessa immagine che ricevera' il motore* — e in questa modalita' dipende dalla finitura (RGB in Spot, grigio filtrato in B/N), quindi va tenuto allineato con la scelta di `input_img` in `generate_stl`, altrimenti i confini delle regioni mostrati non sono quelli che verranno generati; (c) il ramo ritaglio in `on_pixel_clicked` ha la **precedenza** su swatch e accenti Spot, perche' e' l'unico che si arma esplicitamente e che mostra un'immagine sua (campionare un colore dal giallo dell'anteprima leggerebbe l'anteprima, non il disegno).
- **Ordine dei riquadri** — il pannello di una modalita' sta **sopra** i pannelli che accende: `mode_selector` in cima, poi `group_cover`/`group_keychain` (che hanno il selettore Finish), poi `group_spot` e `group_swatch`/`group_z` che la finitura fa comparire. Con il selettore sotto, si sceglie una cosa guardando un riquadro gia' scorso. E' un'invariante che si rompe da sola aggiungendo un pannello in fondo a `initUI` (successo una volta), quindi `tests/ritaglio_sagoma.py` la verifica con `layout.indexOf`.
- **`ui_main_window.py`** — pure UI construction (`MainWindowUI`) + `ImageGraphicsView` (wheel-zoom/pan/`pixelClicked` signal). `_on_mode_changed` toggles per-mode group visibility. `self.lockable_widgets` is a flat registry of every widget that must disable during generation — **add new controls to this list, not to `toggle_ui_state`**, which just iterates the registry and restores mode-conditional states (auto-Z, auto-midtones, Deckbox-locked physical params, Cover levels selector) afterward.
- **`engine/`** — the whole generation pipeline, **importable without PyQt** (so the same code can serve a web backend). Nothing under `engine/` may import PyQt or touch the filesystem outside `engine.resources`.
  - `engine/pipeline.py` — `generate(image, params, progress=None, should_cancel=None) -> GenerationResult`. One function branches by mode: prepares/composes the source image → classifies pixels into a palette → builds the heightmap/terrace mesh → optional decimation (`fast_simplification`, >`DECIMATE_THRESHOLD` = 200k faces) → export. `progress(pct, msg)` and `should_cancel()` are plain callables; errors propagate as exceptions (the caller decides how to present them).
  - `engine/params.py` — `GenerationMode` (mode string constants, ordered like the UI selector), `GenerationParams` (every knob in one serializable dataclass — replaces the ~32 positional args the worker used to take) and `GenerationResult` (`stl_path`, `mf3_path`, `companion_path`, `elapsed_s`).
  - `engine/resources.py` — asset resolution for the engine: explicit `set_assets_dir()` override (for a web host) → PyInstaller `sys._MEIPASS/assets` → repo-root `assets/`. Use `asset_path(...)` inside `engine/`, **not** `utils.resource_path` (which is cwd-relative in dev and stays for desktop-only assets like `style.qss`/`icon.ico`).
- **`worker.py`** — thin `MeshWorker(QThread)` adapter over `engine.generate`: translates the engine callbacks into the Qt signals `progress(int,str)`, `finished_ok(stl_path,3mf_path)`, `finished_err(str)`, and keeps cooperative cancellation via `self.cancel_requested`. Deckbox mode emits the 3MF **first** in `finished_ok` — a UI quirk the adapter preserves. No generation logic belongs here.
- **`engine/mesh_utils.py`** — geometry core:
  - `create_solid_mesh(X,Y,Z,bottom_z,mask=None)` — vectorized heightmap→watertight-solid. With `mask`, only cells fully inside the boolean mask get top/bottom faces, and **every** boundary edge (outer outline *and* interior holes) gets a sealed vertical wall — this is what makes the Phone Cover plate's camera cutouts and rounded silhouette printable.
  - `rounded_rect_mask(h,w,radius_px,holes=[(x,y,w,h,r),...])` — plate silhouette generator.
  - `process_mesh_topo(...)` — shared terrace-mesh pipeline used by Topographic, Spot Color, *and* Phone Cover (Cover composes onto a palette first, then calls this with its plate `mask`). Takes `min_feature_mm` (island/fringe cleanup threshold — default 0.5mm for panel-scale prints, lowered to 0.25–0.35mm by Cover mode since a small plate's engraved groove is missing material, not a printed wall, so it tolerates finer features) and `max_res_cap`.
  - `compute_topo_z_heights` / `compute_topo_switch_z` — terrace Z quantization and the filament-switch Z (which is `z[i-1] + layer_height`, **not** `z[i]` — the switch happens one layer above the previous terrace's top, a fix that took a full debugging pass; don't regress it).
  - `export_3mf(mesh, path, color_changes_z, slot_colors=None)` — trimesh-native 3MF + injected Bambu `custom_gcode_per_layer.xml`/`slice_info.config`. Filters `color_changes_z` to `>0` and dedupes by rounding — **be careful introducing new switch-Z values that could collide after rounding**, a dedup collision silently drops a color layer from the printed object (see git history around the Cover engraved-surface debugging).
- **`engine/cutout_utils.py`** — **Cutout / Keychain**: sostituisce il pannello rettangolare con la sagoma del disegno. Sfrutta il `mask` che `create_solid_mesh`/`process_mesh_topo` avevano gia' per i fori fotocamera della cover, quindi la geometria non e' codice nuovo — quello che e' nuovo e' *da dove viene la maschera*.
  - Il problema non e' "togliere il bianco": due regioni bianche indistinguibili (il vuoto racchiuso fra due tratti, e l'interno di una forma chiusa) vanno una bucata e una tenuta, e nessuna proprieta' geometrica le separa. Quindi **non si indovina**: `segment_regions` divide la carta nelle regioni delimitate dal tratto (connettivita' **4**, non 8: con la 8 lo sfondo cola dentro attraverso ogni contorno inclinato), la regola automatica taglia solo cio' che tocca il bordo, e le eccezioni arrivano come **semi** — coppie `(x, y)` che nominano una regione. Non raster: i `GenerationParams` restano serializzabili e la scelta si rifa' identica in anteprima e in generazione.
  - I semi sono in coordinate del **raster di segmentazione** (`seg_shape_for`, cap `cutout_seg_res`/`SEG_MAX_RES=1600`), che dipende *solo* da forma sorgente e cap — mai da `white_clip`. E' cio' che rende i click sopravvissuti a una risegmentazione: muovere White Clip non cancella il lavoro fatto. Se cambi come si deriva quel raster, invalidi ogni seme gia' salvato.
  - `mask_from_regions` ritorna il **complemento delle sole regioni tagliate**, non l'unione di quelle tenute: cosi' ogni pixel di tratto resta materiale e diventa la parete verticale del pezzo. Per lo stesso motivo `seal_px` (dilatazione del tratto prima di segmentare, contro i contorni non perfettamente chiusi) non mangia il disegno — esce dalle regioni, non dalla maschera, quindi semmai aggiunge un filo di materiale sul bordo dei fori.
  - `mask_from_paint` e' la strada alternativa (opzione B, maschera dipinta a mano): li' non serve nessuna connettivita', e' materiale tutto cio' che non e' bianco. Resta come scorciatoia per i casi che il click non risolve.
  - `compute_cutout` fa **due giri** sul passo mm/pixel, e non e' una svista: `max_dim` misura il *pezzo ritagliato*, ma bordino e occhiello sono in millimetri e cambiano la sagoma — quindi il passo si conosce solo dopo aver saputo quanto e' grande la sagoma. Il terzo giro cambierebbe le cifre dopo la virgola.
  - La pipeline **ritaglia anche la sorgente** al riquadro della sagoma (`_crop_source_to`): senza, Max Dim continuerebbe a misurare il foglio e un disegno che ne occupa un quarto uscirebbe in scala 1:4.
  - E' una **modalita'**, non un flag trasversale: `GenerationMode.KEYCHAIN` accende il ritaglio da solo, e `keychain_finish_spot` sceglie se l'arte passa dalla classificazione Spot o dalla posterizzazione Standard. Le altre modalita' non hanno ritaglio — Deckbox e Phone Cover una sagoma ce l'hanno gia' (la scatola, la plate), e Standard/Topo/Spot restano i pannelli rettangolari che erano.
  - `n_pieces > 1` e `ring_attached == False` finiscono in `GenerationResult` e nel popup: sono scelte prese dal motore (tenere un troncone solo) o errori di posizionamento che nel file non si vedono.
- **`engine/color_utils.py`** — all pixel classification, shared across modes:
  - `rgb_to_lab(..., chroma_weight=...)` — Lab conversion with amplified a/b channels; `CHROMA_MATCH_WEIGHT=2.5` prevents neutral grays from matching saturated palette colors (was the root cause of "red bleeding onto black/white edges" in Topographic mode).
  - `extract_dominant_colors` / `merge_lab_clusters` / `downsample_for_analysis` — Topographic's K-Means-in-Lab-with-cluster-merging, shared helper functions.
  - `suggest_spot_accents` / `classify_spot_pixels` / `build_spot_palette` — Spot Color engine (hue-distance matching + coverage-gated saturation threshold, not Lab distance — an earlier Lab-based attempt made the coverage slider nearly inert).
  - `feature_scale` / `NOZZLE_MM` / `SOLID_MM` — quanto misura in **millimetri di stampa** il tratto piu' fine del disegno, e il vuoto piu' stretto fra due tratti. Non e' una proprieta' dell'immagine ne' della stampante: e' il prodotto delle due, e cambia a ogni tocco di Max Dim. Serve a rispondere *prima* alla domanda che altrimenti si scopre nello slicer. Due misure perche' i modi di fallire sono due (il tratto sparisce / i tratti si fondono), ed e' la stessa distance transform letta sull'inchiostro e sulla carta. La larghezza viene dalla cresta come `2·d − 1`, che sui tratti di larghezza pari sottostima di un pixel: sbaglia per prudenza, ed e' voluto. `max_dim_mm` va passato dal chiamante — dove la sorgente e' piu' larga del pezzo (il ritaglio) ricavarlo dal lato lungo dell'immagine gonfia il consiglio del rapporto fra foglio e pezzo. Il collegamento alla UI e' `_feature_scale_source()`, che per il portachiavi usa `CutoutResult.pitch_mm`. Prove in `tests/tratto_minimo.py`.
  - `quantize_grayscale_levels` / `grayscale_palette` — Cover mode's B/N finish. **Do not re-implement this with a fresh per-call K-Means(k=n_levels)** — it's unstable when the image's real tonal populations don't match `n_levels` (a small noise/anti-aliasing cluster can steal the middle slot from the real dominant midtone). The current design always runs the same K=4 clustering Standard mode uses (`suggest_midtones`'s approach) and picks the matching landmark subset per `n_levels`, mirroring how Standard mode's 2/3/4-color selector hides L1/L2.
- **`engine/case_utils.py`** — Phone Cover mode, two independent ways to get a plate+companion pair:
  - **Parametric** (`build_bumper`, `compute_plate_dims`): a TPU bumper built from 4 stacked shapely ring cross-sections (back frame / plate groove / phone cavity / front screen-guard lip) via `trimesh.boolean` (needs the `manifold` engine — `manifold3d` package). Cutouts (`side_cutouts`, `top_cutouts`, `bottom_opening`) are expressed **looking at the phone's back** (matches the artwork's viewing orientation) and are mirrored internally to model space — see the docstring on `build_bumper` before changing cutout math.
  - **Real-seat retrofit** (`carve_plate_recess`): given any third-party case STL plus measured cavity/wall coordinates, cuts a plate-shaped window + undercut pocket into it, preserving named keep-zones (e.g. an existing camera-lens block). Presets with a `case_plate` key (see `assets/phone_presets.json`) use this real, ruler-verified seat instead of the parametric system — `build_case_plate_raster` fills the measured outline polygon directly.
  - `compose_plate_art` / `compose_cover_art` — fits the source image onto the plate raster (cover-fill with user zoom/offset in mm), optionally excluding the area under the camera holes (`avoid_camera`).
  - **Engraved vs Raised** is a single palette-order inversion (`palette[::-1]`, indices flipped) done once in the Cover branch of `engine/pipeline.py` — everything downstream (terraces, 3MF slot colors, popup text) follows from that one flip. The convention: index 0 of the palette always gets the *smallest* Z (`base_h`) and the last index gets the *largest* Z (`max_h`), since `create_solid_mesh`'s `bottom_z` is uniformly flat — so whichever color is last in palette order becomes the flush outward-facing plane.
  - `assets/phone_presets.json` — per-phone geometry, coordinates measured **looking at the phone's back**, origin at the top-left corner of the phone body. `_note`/`_todo` keys inside are human documentation, filtered out at load time (`k.startswith('_')`).

### Key conventions carried from earlier modes (still apply)

- **`_PHYS_DEFAULTS`** (`manga_to_3d.py`) — ogni modalita' porta con se' la propria taglia (Max Dim / Base / Max Z / Layer), **in entrambe le direzioni**. Non e' cosmesi: la copertura a 2 colori misura una finestra di `BW_WINDOW_MM` = 0,7 mm *reali*, quindi Max Dim decide quanti pixel della sorgente ci finiscono dentro (4 px a 200 mm, 14 px a 60 mm) e con essi la classificazione. Quando i default si applicavano solo *entrando* in Cover e Keychain, dopo un giro nel portachiavi la modalita' Standard restava a 60 mm e binarizzava lo stesso pannello con una finestra tre volte piu' grossa, a cursore fermo. Per lo stesso motivo `spin_dim` e' connesso a `_refresh_std_mockup`.
- Z heights always snap to `layer_height` and round to 3 decimals.
- Standard-mode sampled grays: `[L0 white bg, L1 light, L2 dark, L3 black]`.
- Output paths are auto-computed next to the source image with anti-overwrite numbering. The stem says which mode made the file, because `output/` mixes them: `<image>_3D` for the panel modes, `<image>_keychain` for Keychain. A **prefix** instead of a suffix marks the modes that emit more than one file that belong together — `cover_plate_<image>` + `cover_bumper_<image>`, `full_deckbox_<image>` — where it groups them by role (`MeshWorker.companion_path_for` derives the bumper from the plate — keep them in sync if you change the naming scheme).
- `engine.resources.asset_path()` for any file under `assets/` read by the engine; `utils.resource_path()` stays for desktop-only resources (`style.qss`, `icon.ico`). Both are PyInstaller `_MEIPASS`-aware. Note that `MangaRelief_Pro.spec` bundles only `icon.ico` — a frozen build that needs `assets/` requires adding `('assets', 'assets')` to `datas` (pre-existing gap, unchanged by the engine extraction).
- UI/code comments mix Italian and English; match the surrounding style in the file you're editing. **User-visible strings are English** — labels, tooltips, status-bar messages, dialog and popup text, and the exception messages the engine raises for the UI to show. The mixed-language convention is about comments and docstrings, not about what the person using the app reads.

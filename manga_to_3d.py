import sys
import os
import time
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from PIL import Image
import pillow_heif
import ctypes

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer

from utils import resource_path
from ui_main_window import MainWindowUI
from engine import (GenerationMode, GenerationParams, ink_level,
                    prepare_source_image, standard_heightmap)
from engine.color_utils import (extract_dominant_colors, suggest_midtones, feature_scale, thicken_ink, thicken_applied_mm,
                         suggest_spot_accents, classify_spot_pixels,
                         downsample_for_analysis, build_spot_palette,
                         grayscale_palette)
from engine.cutout_utils import (SEG_MAX_RES, compute_cutout, overlay_preview,
                                  segment_regions, to_seg_raster)
from engine.mesh_utils import compute_topo_z_heights, compute_topo_switch_z
from engine.case_utils import (load_phone_presets, build_plate_raster,
                        build_case_plate_raster, compose_plate_art)
from worker import MeshWorker

# Abilitiamo i plugin HEIF e AVIF in caso di fallimento OpenCV
try:
    pillow_heif.register_heif_opener()
except AttributeError:
    pass
try:
    pillow_heif.register_avif_opener()
except AttributeError:
    pass

# Imposta l'AppUserModelID di Windows per mostrare l'icona nativa sulla taskbar
try:
    myappid = 'antigravity.mangareliefpro.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

class Manga3DAppController(MainWindowUI):
    def __init__(self):
        super().__init__()
        
        self.img_filtered_array = None
        self._ink_level = None
        self.active_swatch_index = None
        self.loaded_image_path = None
        self.last_opened_dir = ""
        self.sampled_colors = [250, 210, 150, 15]
        self.color_mode_state = 4
        self.last_midtone_pct = 100.0

        # Spot Color state
        self.spot_accents = [None, None]
        self.active_spot_swatch = None

        # Ritaglio sagoma. I semi sono coordinate del raster di segmentazione
        # (lo stesso su cui si disegna l'anteprima, quindi i click ci cadono
        # dentro senza conversioni), e sono l'unica forma in cui la scelta
        # dell'utente viaggia fino al motore.
        self.cutout_cut_seeds = []
        self.cutout_keep_seeds = []
        self.cutout_paint_mask = None
        self.cutout_ring_xy = None
        self._cutout_regions = None
        self._cutout_regions_key = None

        # Phone Cover state
        try:
            self.phone_presets = {k: v for k, v in load_phone_presets().items()
                                  if not k.startswith('_')}
        except Exception as e:
            print(f"Warning: phone presets not loaded ({e})")
            self.phone_presets = {}
        self.combo_phone_model.addItems(list(self.phone_presets.keys()))

        self.setup_connections()
        self.update_swatch_colors()
        self._refresh_auto_z_display()
        self._on_auto_z_toggled(True)

    def setup_connections(self):
        self.btn_load.clicked.connect(self.load_image)
        self.btn_generate.clicked.connect(self.generate_stl)
        self.viewer.pixelClicked.connect(self.on_pixel_clicked)
        
        for i, btn in enumerate(self.swatches):
            btn.clicked.connect(lambda checked, idx=i: self.set_active_swatch(idx))
            
        self.btn_auto_white.clicked.connect(self._apply_auto_white)
        self.btn_extract_topo.clicked.connect(self._extract_topo_colors)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self.chk_auto_z.toggled.connect(self._on_auto_z_toggled)
        self.chk_auto_midtones.toggled.connect(self._on_auto_midtones_toggled)
        self.mode_selector.currentIndexChanged.connect(self._update_viewport_mode)
        
        self.spin_base.valueChanged.connect(self._on_physical_param_changed)
        self.spin_maxh.valueChanged.connect(self._on_physical_param_changed)
        self.spin_layer_height.valueChanged.connect(self._on_physical_param_changed)

        # Phone Cover
        self.btn_cover_preview.toggled.connect(self._on_cover_preview_toggled)
        self.slider_cover_scale.valueChanged.connect(self._on_cover_param_changed)
        self.slider_cover_offx.valueChanged.connect(self._on_cover_param_changed)
        self.slider_cover_offy.valueChanged.connect(self._on_cover_param_changed)
        self.combo_phone_model.currentIndexChanged.connect(self._on_phone_model_changed)
        self.chk_cover_avoid_camera.toggled.connect(lambda _: self._refresh_cover_preview())
        self.mode_selector.currentIndexChanged.connect(self._on_mode_defaults)

        # Spot Color
        self.btn_spot_auto.clicked.connect(self._apply_spot_auto)
        for i, btn in enumerate(self.spot_swatches):
            btn.clicked.connect(lambda checked, idx=i: self.set_active_spot_swatch(idx))
        self.slider_spot_coverage.valueChanged.connect(self._on_spot_coverage_changed)
        self.combo_spot_naccents.currentIndexChanged.connect(lambda _: self._refresh_spot_mockup())
        # Questi ora influenzano anche la classificazione Spot: tengono vivo il mockup
        self.spin_white_clip.valueChanged.connect(lambda _: self._refresh_spot_mockup())
        self.spin_black_clip.valueChanged.connect(lambda _: self._refresh_spot_mockup())
        self.cmb_quality.currentIndexChanged.connect(lambda _: self._refresh_spot_mockup())
        self.btn_spot_mockup.toggled.connect(self._on_spot_mockup_toggled)

        # Keychain / Cutout
        self.combo_keychain_finish.currentIndexChanged.connect(
            self._on_keychain_finish_changed)
        self.combo_cutout_src.currentIndexChanged.connect(self._on_cutout_source_changed)
        self.btn_cutout_paint.clicked.connect(self._load_paint_mask)
        self.btn_cutout_edit.toggled.connect(self._on_cutout_edit_toggled)
        self.btn_cutout_reset.clicked.connect(self._reset_cutout)
        self.btn_cutout_preview.toggled.connect(self._on_cutout_preview_toggled)
        self.slider_cutout_border.valueChanged.connect(self._on_cutout_border_changed)
        self.chk_cutout_ring.toggled.connect(lambda _: self._refresh_cutout_preview())
        self.spin_ring_d.valueChanged.connect(lambda _: self._refresh_cutout_preview())
        # White Clip decide cosa e' tratto, quindi ridisegna i confini delle
        # regioni: la segmentazione in cache non vale piu'.
        self.spin_white_clip.valueChanged.connect(self._invalidate_cutout_regions)
        self.spin_dim.valueChanged.connect(lambda _: self._refresh_cutout_preview())

        # Standard: la copertura e l'anteprima della classificazione. Le stesse
        # cose che tengono vivo il mockup Spot lo tengono vivo anche qui —
        # ritaglio, qualita' e quote di cambio colore cambiano su quale bobina
        # finisce un pixel, quindi cambiano l'immagine che si sta guardando.
        self.slider_bw_coverage.valueChanged.connect(self._on_bw_coverage_changed)
        self.btn_std_mockup.toggled.connect(self._on_std_mockup_toggled)
        self.spin_white_clip.valueChanged.connect(lambda _: self._refresh_std_mockup())
        self.spin_black_clip.valueChanged.connect(lambda _: self._refresh_std_mockup())
        self.cmb_quality.currentIndexChanged.connect(lambda _: self._refresh_std_mockup())
        self.chk_auto_z.toggled.connect(lambda _: self._refresh_std_mockup())
        # Max Dim non e' solo la scala: a 2 colori la copertura si misura su
        # una finestra in millimetri, quindi cambiando la dimensione del pezzo
        # cambia quanti pixel entrano nella finestra e cambia la
        # classificazione. Senza questa riga il mockup mostrava quella vecchia.
        self.spin_dim.valueChanged.connect(lambda _: self._refresh_std_mockup())

        # Il tratto piu' fine in millimetri: dipende da quanto sara' grande il
        # pezzo e da cosa conta come inchiostro, quindi si rifa' quando cambia
        # una delle due — e al cambio di modalita', che le cambia entrambe.
        self.spin_dim.valueChanged.connect(lambda _: self._refresh_feature_scale())
        self.spin_white_clip.valueChanged.connect(lambda _: self._refresh_feature_scale())
        self.mode_selector.currentIndexChanged.connect(
            lambda _: self._refresh_feature_scale())
        self.combo_keychain_finish.currentIndexChanged.connect(
            lambda _: self._refresh_feature_scale())
        self.slider_cutout_border.valueChanged.connect(
            lambda _: self._refresh_feature_scale())

        # L'ingrossamento cambia la geometria che tutti guardano: la misura del
        # tratto, la classificazione Standard e quella Spot. Non il ritaglio,
        # che nel motore viene prima ed e' una domanda sulla sagoma.
        self.slider_line_thicken.valueChanged.connect(self._on_line_thicken_changed)
        for _sp in (self.spin_z1, self.spin_z2, self.spin_z3):
            _sp.valueChanged.connect(lambda _: self._refresh_std_mockup())

    def _update_viewport_mode(self, index):
        """Switch viewport display between Color and Grayscale based on selected mode."""
        # Cambiando modalità le anteprime toggle si spengono sempre
        if self.btn_spot_mockup.isChecked():
            self.btn_spot_mockup.setChecked(False)
            return  # il toggle handler richiama questo metodo col viewport giusto
        if self.btn_cover_preview.isChecked():
            self.btn_cover_preview.setChecked(False)
            return
        if self.btn_cutout_preview.isChecked():
            self.btn_cutout_preview.setChecked(False)
            return
        # Anche questo: era l'unico che restava acceso cambiando modalità, e
        # mostrava una classificazione calcolata con i parametri della
        # modalità precedente — che dopo un giro nel portachiavi sono altri.
        if self.btn_std_mockup.isChecked():
            self.btn_std_mockup.setChecked(False)
            return

        if getattr(self, 'img_filtered_array', None) is None:
            return

        if index == 1:        # Topographic Mode
            self.viewer.setImage(self._get_rgb_filtered())
        elif index in (3, 4): # Spot / Cover: serve l'originale a colori
            self.viewer.setImage(self.img_rgb_original)
        else:                 # Standard / Deckbox
            self.viewer.setImage(self.img_filtered_array)

    # ------------------------------------------------------------------
    # PHONE COVER — composizione artwork sulla plate
    # ------------------------------------------------------------------

    # Taglie di default per modalità: (Max Dim, Base, Max Z, Layer), in mm.
    #
    # Prima i default si applicavano solo ENTRANDO in Cover e Keychain, e non
    # si tornava mai indietro: dopo un giro nel portachiavi la modalità
    # Standard restava a 60 mm. Non è un dettaglio estetico — la copertura a
    # 2 colori misura una finestra di 0,7 mm REALI, quindi a 60 mm quella
    # finestra copre 14 px della sorgente invece di 4, e lo stesso pannello si
    # binarizza tre volte più grosso. Il cursore era rimasto a 35%, l'immagine
    # era la stessa, e il risultato cambiava senza che niente lo dicesse.
    #
    # Quindi la tabella copre TUTTE le modalità, non solo quelle che avevano
    # un default speciale: cambiare modalità porta con sé la sua taglia, in
    # entrambe le direzioni.
    _PHYS_DEFAULTS = {
        0: (200.0, 1.0, 2.4, 0.20),   # Standard
        1: (200.0, 1.0, 2.4, 0.20),   # Topographic
        2: (200.0, 4.0, 2.0, 0.20),   # Deckbox (Max Dim lo blocca _on_mode_changed)
        3: (200.0, 1.0, 2.4, 0.20),   # Spot Color
        4: (200.0, 0.3, 1.0, 0.10),   # Phone Cover: sede slim, layer fini
        # Keychain. I due numeri hanno ragioni diverse e non vanno scambiati:
        # la base regge lo strappo dell'anellino (8 layer bastano su un pezzo
        # da 60 mm), il rilievo serve solo al colore e deve ospitare
        # colori-1 bande da almeno 2 layer l'una — 1,2 mm e' il minimo che
        # regge anche il caso peggiore, due accenti (2+2+2). Sotto, una banda
        # scende a un layer e il colore sotto traspare.
        5: (60.0,  1.6, 2.8, 0.20),
    }

    def _on_mode_defaults(self, index):
        """Porta i parametri fisici alla taglia della modalità scelta."""
        dim, base, maxh, layer = self._PHYS_DEFAULTS.get(
            index, self._PHYS_DEFAULTS[0])
        self.spin_dim.setValue(dim)
        self.spin_base.setValue(base)
        self.spin_maxh.setValue(maxh)
        self.spin_layer_height.setValue(layer)

    def _on_phone_model_changed(self, _index):
        preset = self._current_phone_preset() or {}
        # Con la sede reale la fotocamera è già fuori sagoma: il toggle non serve
        self.chk_cover_avoid_camera.setEnabled('case_plate' not in preset)
        self._refresh_cover_preview()

    def _current_phone_preset(self):
        name = self.combo_phone_model.currentText()
        return self.phone_presets.get(name)

    def _compose_cover(self, max_res_cap=1200):
        """Applica sagoma + composizione correnti. Ritorna (art, mask, res)."""
        preset = self._current_phone_preset()
        if preset is None or getattr(self, 'img_rgb_original', None) is None:
            return None
        if 'case_plate' in preset:
            # sede reale ricavata dalla cover template: fotocamera già esclusa
            mask, res, _ = build_case_plate_raster(preset, max_res_cap=max_res_cap)
            avoid = False
        else:
            mask, res, _ = build_plate_raster(preset, max_res_cap=max_res_cap)
            avoid = self.chk_cover_avoid_camera.isChecked()
        art = compose_plate_art(
            self.img_rgb_original, preset, mask, res,
            user_scale=self.slider_cover_scale.value() / 100.0,
            off_x=float(self.slider_cover_offx.value()),
            off_y=float(self.slider_cover_offy.value()),
            avoid_camera=avoid)
        return art, mask, res

    def _on_cover_param_changed(self, _value):
        self.lbl_cover_scale.setText(f"Zoom: {self.slider_cover_scale.value()}%")
        self.lbl_cover_offx.setText(f"Offset X: {self.slider_cover_offx.value()} mm")
        self.lbl_cover_offy.setText(f"Offset Y: {self.slider_cover_offy.value()} mm")
        self._refresh_cover_preview()

    def _on_cover_preview_toggled(self, checked):
        if checked and getattr(self, 'img_rgb_original', None) is None:
            self.btn_cover_preview.setChecked(False)
            return
        if checked:
            self.btn_cover_preview.setText("👁 Back to Original")
            self._refresh_cover_preview()
        else:
            self.btn_cover_preview.setText("👁 Plate Preview")
            self._update_viewport_mode(self.mode_selector.currentIndex())

    def _refresh_cover_preview(self):
        """Anteprima: artwork composto con la sagoma della plate sovrapposta
        (esterno oscurato, fori camera evidenti). Solo se il toggle è attivo."""
        if not self.btn_cover_preview.isChecked():
            return
        composed = self._compose_cover(max_res_cap=800)  # leggera per la UI
        if composed is None:
            return
        art, mask, res = composed
        disp = art.copy()
        disp[~mask] = disp[~mask] // 4  # oscura fuori sagoma e nei fori
        self.cover_preview_array = np.ascontiguousarray(disp)
        self.viewer.setImage(self.cover_preview_array)
        preset = self._current_phone_preset()
        self.lbl_status.setText(
            f"👁 Plate {self.combo_phone_model.currentText()}: drag it with the sliders, "
            f"the dark areas are the camera holes.")

    # ------------------------------------------------------------------
    # SPOT COLOR — picking, auto-detect, mockup
    # ------------------------------------------------------------------

    # Indice del selettore di modalità -> modalità del motore
    _MODE_BY_INDEX = {
        0: GenerationMode.STANDARD,
        1: GenerationMode.TOPOGRAPHIC,
        2: GenerationMode.DECKBOX,
        3: GenerationMode.SPOT_COLOR,
        4: GenerationMode.PHONE_COVER,
        5: GenerationMode.KEYCHAIN,
    }

    def _current_generation_mode(self) -> str:
        return self._MODE_BY_INDEX.get(self.mode_selector.currentIndex(),
                                       GenerationMode.STANDARD)

    def _current_max_res_cap(self) -> int:
        """Cap di risoluzione dal selettore Mesh Quality. Usato sia dalla
        generazione sia dalle anteprime, così il mockup mostra esattamente
        la stessa classificazione del file esportato."""
        quality_str = self.cmb_quality.currentText()
        if "800" in quality_str:
            return 800
        if "1600" in quality_str:
            return 1600
        return 1200

    def _get_spot_accents(self):
        """Accenti attivi (1 o 2 in base al selettore), senza i None."""
        n = self.combo_spot_naccents.currentIndex() + 1
        return [a for a in self.spot_accents[:n] if a is not None]

    def set_active_spot_swatch(self, idx):
        if getattr(self, 'img_rgb_original', None) is None:
            QMessageBox.information(self, "Warning", "Please load an image before sampling accents.")
            return
        self.active_spot_swatch = idx
        self.lbl_status.setText("🎯 Left Click on the image to sample the accent color...")

    def _sample_accent_at(self, x, y):
        """Mediana 5x5 attorno al click: robusta contro il rumore JPEG."""
        h, w = self.img_rgb_original.shape[:2]
        x0, x1 = max(0, x - 2), min(w, x + 3)
        y0, y1 = max(0, y - 2), min(h, y + 3)
        patch = self.img_rgb_original[y0:y1, x0:x1].reshape(-1, 3)
        return tuple(int(v) for v in np.median(patch, axis=0))

    def _apply_spot_auto(self):
        if getattr(self, 'img_rgb_original', None) is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        n = self.combo_spot_naccents.currentIndex() + 1
        self.lbl_status.setText("🤖 Detecting accent colors...")
        QApplication.processEvents()
        found = suggest_spot_accents(self.img_rgb_original, n_accents=n)
        if not found:
            self.lbl_status.setText("⚪ No vivid accent found: image is nearly B/W. Pick manually if needed.")
            return
        for i in range(n):
            self.spot_accents[i] = found[i] if i < len(found) else None
        self._update_spot_swatch_colors()
        self.lbl_status.setText(f"✅ {len(found)} accent(s) detected. Fine-tune by clicking the image.")
        self._refresh_spot_mockup()

    def _update_spot_swatch_colors(self):
        for i, btn in enumerate(self.spot_swatches):
            acc = self.spot_accents[i]
            if acc is None:
                btn.setText(f"Accent {i+1}: [ -- ]")
                btn.setStyleSheet("")
            else:
                r, g, b = acc
                lum = 0.299*r + 0.587*g + 0.114*b
                text_color = "white" if lum < 128 else "black"
                btn.setText(f"Accent {i+1}: RGB ({r}, {g}, {b})")
                btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color};")

    def _on_spot_coverage_changed(self, value):
        self.lbl_spot_coverage.setText(f"Accent Coverage: {value}%")
        self._refresh_spot_mockup()

    def _on_spot_mockup_toggled(self, checked):
        if checked and getattr(self, 'img_rgb_original', None) is None:
            self.btn_spot_mockup.setChecked(False)
            return
        if checked:
            self._do_refresh_spot_mockup()   # immediato all'accensione
            self.btn_spot_mockup.setText("👁 Back to Original")
        else:
            self.btn_spot_mockup.setText("👁 Mockup Preview")
            self._update_viewport_mode(self.mode_selector.currentIndex())

    def _refresh_spot_mockup(self):
        """Richiede un ricalcolo dell'anteprima, accorpando le richieste
        ravvicinate: alla risoluzione di generazione un refresh costa
        centinaia di ms, e trascinare uno slider ne genererebbe decine."""
        if not self.btn_spot_mockup.isChecked():
            return
        if getattr(self, '_spot_mockup_timer', None) is None:
            self._spot_mockup_timer = QTimer(self)
            self._spot_mockup_timer.setSingleShot(True)
            self._spot_mockup_timer.timeout.connect(self._do_refresh_spot_mockup)
        self._spot_mockup_timer.start(180)

    def _do_refresh_spot_mockup(self):
        """Ricalcola davvero l'anteprima posterizzata."""
        if not self.btn_spot_mockup.isChecked():
            return
        if getattr(self, 'img_rgb_original', None) is None:
            return
        # Stessa risoluzione della generazione: a 800px fissi l'anteprima
        # mostrava più sgranatura di quella che sarebbe finita nel file
        src = self._thickened(
            self.img_rgb_original,
            self.spin_dim.value() / max(self.img_rgb_original.shape[:2]))
        small = downsample_for_analysis(src, self._current_max_res_cap())
        palette, idx = classify_spot_pixels(
            small, self._get_spot_accents(),
            coverage=self.slider_spot_coverage.value(),
            white_clip=self.spin_white_clip.value(),
            black_clip=self.spin_black_clip.value())
        self.spot_preview_array = np.array(palette, dtype=np.uint8)[idx]
        self.viewer.setImage(self.spot_preview_array)
        names = ", ".join(f"RGB{p}" for p in palette)
        self.lbl_status.setText(f"👁 Mockup: {len(palette)} colors → {names}")

    # ------------------------------------------------------------------
    # STANDARD — copertura a 2 colori e anteprima della classificazione

    def _refresh_bw_labels(self):
        """Il valore del cursore e cosa conta come inchiostro su *questa*
        immagine: la soglia la legge Otsu dall'istogramma, quindi cambia da
        scansione a scansione e dirla a voce e' l'unico modo perche' il
        cursore non sia un numero senza unita'."""
        pct = self.slider_bw_coverage.value()
        self.lbl_bw_coverage.setText(f"Shading darker than {pct}% prints as ink")
        nota = ("A zone prints as ink when at least "
                f"{pct}% of its area is inked, judged over about 0.7 mm — "
                "what the nozzle can resolve. Lower keeps more hatching black, "
                "higher keeps more of it paper.")
        if getattr(self, 'img_filtered_array', None) is not None:
            if getattr(self, '_ink_level', None) is None:
                self._ink_level = ink_level(self.img_filtered_array)
            nota += f" Ink here is anything darker than {self._ink_level}."
        self.lbl_bw_note.setText(nota)

    def _on_bw_coverage_changed(self, _value):
        self._refresh_bw_labels()
        self._refresh_std_mockup()

    def _preview_params(self):
        """I parametri correnti, senza percorsi di uscita: l'anteprima deve
        classificare esattamente come classifichera' la generazione."""
        if self.chk_auto_z.isChecked():
            changes = self._compute_auto_z()
        else:
            changes = [round(self.spin_z1.value(), 3),
                       round(self.spin_z2.value(), 3),
                       round(self.spin_z3.value(), 3)]
        mode = getattr(self, 'color_mode_state', 4)
        return GenerationParams(
            mode=GenerationMode.STANDARD,
            max_dim=self.spin_dim.value(),
            base_h=self.spin_base.value(),
            max_h=self.spin_maxh.value(),
            layer_height=self.spin_layer_height.value(),
            max_res_cap=self._current_max_res_cap(),
            white_clip=self.spin_white_clip.value(),
            black_clip=self.spin_black_clip.value(),
            sampled_values=self.sampled_colors,
            color_mode=mode,
            color_changes_z=changes,
            bw_coverage=self._current_bw_coverage(),
            line_thicken_mm=self._line_thicken_mm(),
        )

    def _current_bw_coverage(self):
        """La copertura vale solo a 2 colori; altrove None lascia al motore la
        strada di sempre."""
        if getattr(self, 'color_mode_state', 4) != 2:
            return None
        return self.slider_bw_coverage.value() / 100.0

    def _on_std_mockup_toggled(self, checked):
        if checked and getattr(self, 'img_filtered_array', None) is None:
            self.btn_std_mockup.setChecked(False)
            return
        if checked:
            self._do_refresh_std_mockup()
            self.btn_std_mockup.setText("👁 Back to Original")
        else:
            self.btn_std_mockup.setText("👁 Mockup Preview")
            self._update_viewport_mode(self.mode_selector.currentIndex())

    def _refresh_std_mockup(self):
        """Come per Spot: le richieste ravvicinate si accorpano, perche' alla
        risoluzione di generazione un ricalcolo costa centinaia di ms e
        trascinare un cursore ne genererebbe decine."""
        if not self.btn_std_mockup.isChecked():
            return
        if getattr(self, '_std_mockup_timer', None) is None:
            self._std_mockup_timer = QTimer(self)
            self._std_mockup_timer.setSingleShot(True)
            self._std_mockup_timer.timeout.connect(self._do_refresh_std_mockup)
        self._std_mockup_timer.start(180)

    def _do_refresh_std_mockup(self):
        """Dipinge ogni pixel col tono in cui stampera' davvero.

        Esegue gli stessi due passi del motore, sullo stesso ingresso, alla
        stessa risoluzione: la posterizzazione e' cio' che assegna un pixel a
        una bobina, e un'anteprima che la salti mostra un'altra immagine — a
        due colori, una insensibile all'unico controllo che esiste.

        La banda di un pixel e' quanti cambi di colore stanno alla sua altezza
        o sotto: un pixel stampa nel colore caricato quando si raggiunge la sua
        superficie. Contano tutti, l'ultimo compreso.
        """
        if not self.btn_std_mockup.isChecked():
            return
        if getattr(self, 'img_filtered_array', None) is None:
            return

        p = self._preview_params()
        src = self._thickened(
            self.img_filtered_array,
            p.max_dim / max(self.img_filtered_array.shape[:2]))
        z = standard_heightmap(prepare_source_image(src, p), p)

        if p.color_mode == 2:
            toni = [p.sampled_values[0], p.sampled_values[3]]
        elif p.color_mode == 3:
            toni = [p.sampled_values[0], p.sampled_values[2], p.sampled_values[3]]
        else:
            toni = list(p.sampled_values)

        banda = np.zeros(z.shape, dtype=np.int32)
        for c in (c for c in p.color_changes_z if c > 0):
            banda += (z >= c - 1e-9).astype(np.int32)
        dipinta = np.array(toni, dtype=np.uint8)[np.clip(banda, 0, len(toni) - 1)]

        self.std_preview_array = cv2.cvtColor(dipinta, cv2.COLOR_GRAY2RGB)
        self.viewer.setImage(self.std_preview_array)
        self.lbl_status.setText(
            f"👁 Mockup: {p.color_mode}-color classification at "
            f"{max(dipinta.shape)}px")

    # ------------------------------------------------------------------
    # RITAGLIO SAGOMA — regioni, click, anteprima

    def _cutout_source(self):
        """L'immagine su cui si segmenta.

        Deve essere *la stessa* che ricevera' il motore, non semplicemente
        l'originale: in Standard il motore lavora sul grigio filtrato, e
        segmentare l'RGB qui vorrebbe dire mostrare confini di regione che poi
        in generazione cadono altrove (una campitura satura ma chiara e' tratto
        a colori e carta in grigio). Rispecchia la scelta di generate_stl.
        """
        idx = self.mode_selector.currentIndex()
        if idx in (1, 3):     # Topographic, Spot
            return getattr(self, 'img_rgb_original', None)
        if idx == 5:          # Keychain: dipende dalla finitura scelta
            return getattr(self, 'img_rgb_original' if self._keychain_spot()
                           else 'img_filtered_array', None)
        return getattr(self, 'img_filtered_array', None)

    def _keychain_spot(self) -> bool:
        return self.combo_keychain_finish.currentIndex() == 0

    def _is_keychain(self) -> bool:
        return self.mode_selector.currentIndex() == 5

    def _on_keychain_finish_changed(self, _idx):
        """La finitura cambia l'immagine su cui si segmenta (RGB in Spot,
        grigio filtrato in B/N), quindi le regioni vanno ricalcolate: la
        cache è indicizzata anche sulla modalità, ma non sulla finitura."""
        self._invalidate_cutout_regions()

    def _invalidate_cutout_regions(self, *_):
        self._cutout_regions = None
        self._cutout_regions_key = None
        self._refresh_cutout_preview()

    def _cutout_regions_now(self):
        """La segmentazione corrente, ricalcolata solo quando serve davvero.

        Chiave: modalita' e White Clip, cioe' le sole cose che spostano i
        confini. La forma del raster invece non dipende da loro, quindi i semi
        gia' raccolti restano validi attraverso un ricalcolo — che e' il motivo
        per cui muovere White Clip non cancella il lavoro fatto a click.
        """
        src = self._cutout_source()
        if src is None:
            return None
        key = (self.mode_selector.currentIndex(), self._keychain_spot(),
               self.spin_white_clip.value(), src.shape[:2])
        if self._cutout_regions is None or self._cutout_regions_key != key:
            self._cutout_regions = segment_regions(
                src, white_clip=self.spin_white_clip.value(), seg_res=SEG_MAX_RES)
            self._cutout_regions_key = key
        return self._cutout_regions

    def _cutout_kwargs(self):
        """I parametri del ritaglio letti dall'interfaccia, in un posto solo:
        anteprima e generazione devono chiedere le stesse cose."""
        return dict(
            max_dim=self.spin_dim.value(),
            white_clip=self.spin_white_clip.value(),
            seg_res=SEG_MAX_RES,
            paint_mask=(self.cutout_paint_mask
                        if self.combo_cutout_src.currentIndex() == 1 else None),
            cut_seeds=list(self.cutout_cut_seeds),
            keep_seeds=list(self.cutout_keep_seeds),
            border_mm=self.slider_cutout_border.value() / 10.0,
            ring_xy=(self.cutout_ring_xy if self.chk_cutout_ring.isChecked() else None),
            ring_d_mm=self.spin_ring_d.value(),
        )

    def _compute_cutout_now(self):
        src = self._cutout_source()
        if src is None:
            return None
        kw = self._cutout_kwargs()
        regions = None if kw['paint_mask'] is not None else self._cutout_regions_now()
        return compute_cutout(src, regions=regions, **kw)

    def _on_cutout_source_changed(self, idx):
        """Con la maschera dipinta le regioni non esistono: i click non hanno
        niente da commutare, e lasciarli attivi prometterebbe un controllo che
        non c'e'."""
        auto = (idx == 0)
        for wdg in (self.btn_cutout_edit, self.btn_cutout_reset):
            wdg.setEnabled(auto)
        if not auto:
            self.btn_cutout_edit.setChecked(False)
            self.lbl_cutout_info.setText(
                "Painted mask: anything that is not white is material, and every "
                "white area is a void — outer and enclosed alike. No clicks needed.")
        else:
            self.lbl_cutout_info.setText(
                "Auto removes the outer background only. Voids enclosed by the "
                "drawing (between a cloud and the hat, inside a curl) stay solid: "
                "turn on Edit regions and click inside the ones to punch out.")
        self._refresh_cutout_preview()

    def _load_paint_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open painted mask", self.last_opened_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff);;All Files (*.*)")
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.critical(self, "Decoder Error", "Unreadable mask file.")
            return
        self.cutout_paint_mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # L'aspetto va detto, non corretto in silenzio: la maschera viene
        # riscalata sulla griglia dell'arte, quindi se le proporzioni non
        # coincidono il ritaglio esce stirato rispetto a quello che si e'
        # dipinto — e la causa sarebbe invisibile.
        src = self._cutout_source()
        note = ""
        if src is not None:
            ar_m = self.cutout_paint_mask.shape[1] / self.cutout_paint_mask.shape[0]
            ar_s = src.shape[1] / src.shape[0]
            if abs(ar_m - ar_s) / ar_s > 0.02:
                note = (f"  ⚠ Aspect ratio differs from the image "
                        f"({ar_m:.3f} vs {ar_s:.3f}): the mask will be stretched.")
        self.lbl_status.setText(f"✅ Mask loaded: {os.path.basename(path)}.{note}")
        self._refresh_cutout_preview()

    def _reset_cutout(self):
        self.cutout_cut_seeds = []
        self.cutout_keep_seeds = []
        self.cutout_ring_xy = None
        self._refresh_cutout_preview()
        self.lbl_status.setText("↺ Regions back to the automatic rule.")

    def _on_cutout_border_changed(self, v):
        self.lbl_cutout_border.setText(f"Sticker border: {v/10.0:.1f} mm")
        self._refresh_cutout_preview()

    def _on_cutout_edit_toggled(self, checked):
        """Entrando in modifica l'anteprima si accende per forza.

        Non e' un vezzo: i click vanno letti sul raster di segmentazione, e
        l'unica immagine mostrata a quella risoluzione e' l'anteprima del
        ritaglio. Sull'originale a piena risoluzione le stesse coordinate
        indicherebbero un'altra regione.
        """
        if checked:
            self.btn_cutout_ring.setChecked(False)
            self.btn_cutout_preview.setChecked(True)
            self.btn_cutout_edit.setText("✂️ Editing… (click)")
            self.lbl_status.setText(
                "✂️ Click INSIDE a white region to punch it out; click again to "
                "fill it back. Clicking on the linework does nothing.")
        else:
            self.btn_cutout_edit.setText("✂️ Edit regions")

    def _on_cutout_preview_toggled(self, checked):
        if checked and self._cutout_source() is None:
            self.btn_cutout_preview.setChecked(False)
            return
        if checked:
            self.btn_spot_mockup.setChecked(False)
            self.btn_std_mockup.setChecked(False)
            self.btn_cover_preview.setChecked(False)
            self.btn_cutout_preview.setText("👁 Back to Original")
            self._do_refresh_cutout_preview()
        else:
            self.btn_cutout_edit.setChecked(False)
            self.btn_cutout_ring.setChecked(False)
            self.btn_cutout_preview.setText("👁 Cutout Preview")
            self._update_viewport_mode(self.mode_selector.currentIndex())

    def _refresh_cutout_preview(self):
        """Come per Spot e Standard: le richieste ravvicinate si accorpano."""
        if not self.btn_cutout_preview.isChecked():
            return
        if getattr(self, '_cutout_timer', None) is None:
            self._cutout_timer = QTimer(self)
            self._cutout_timer.setSingleShot(True)
            self._cutout_timer.timeout.connect(self._do_refresh_cutout_preview)
        self._cutout_timer.start(180)

    def _do_refresh_cutout_preview(self):
        """Giallo cio' che stampa, bianco il vuoto — la stessa lettura con cui
        si guarda il disegno per decidere dove bucare."""
        if not self.btn_cutout_preview.isChecked():
            return
        src = self._cutout_source()
        if src is None:
            return
        res = self._compute_cutout_now()
        if res is None or res.empty:
            self.lbl_status.setText("⚠ The cutout leaves no material.")
            return

        view = to_seg_raster(src, SEG_MAX_RES)
        self.cutout_preview_array = overlay_preview(view, res.mask)
        self.viewer.setImage(self.cutout_preview_array)

        y0, y1, x0, x1 = res.bbox
        long_mm = max(y1 - y0, x1 - x0) * res.pitch_mm
        short_mm = min(y1 - y0, x1 - x0) * res.pitch_mm
        msg = f"✂️ Piece {long_mm:.0f} × {short_mm:.0f} mm"
        if res.n_pieces > 1:
            msg += f"  ⚠ {res.n_pieces} separate pieces: only the largest kept"
        if self.chk_cutout_ring.isChecked():
            msg += ("  ⚠ keyring hole not attached to the piece"
                    if not res.ring_attached else "  · keyring hole ok")
        self.lbl_status.setText(msg)

    def _cutout_click(self, x, y):
        """Commuta la regione sotto il click, o piazza l'occhiello.

        Ritorna True se il click e' stato consumato dal ritaglio.
        """
        if self.btn_cutout_ring.isChecked():
            self.cutout_ring_xy = (int(x), int(y))
            self.btn_cutout_ring.setChecked(False)
            if not self.chk_cutout_ring.isChecked():
                self.chk_cutout_ring.setChecked(True)   # riaccende l'anteprima
            else:
                self._refresh_cutout_preview()
            self.lbl_status.setText(f"📍 Keyring hole placed at ({x}, {y}).")
            return True

        if not self.btn_cutout_edit.isChecked():
            return False

        regions = self._cutout_regions_now()
        if regions is None:
            return True
        lb = regions.region_at(x, y)
        if lb <= 0:
            self.lbl_status.setText(
                "That point is linework, not a region: click inside a white area.")
            return True

        pt = (int(x), int(y))
        # Lo stato di una regione e' l'automatismo piu' le correzioni, quindi
        # per commutare si guarda dove sta ADESSO e si scrive il seme opposto,
        # ripulendo l'altra lista: due semi contrastanti sulla stessa regione
        # renderebbero il prossimo click imprevedibile.
        cur_cut = bool(self._region_is_cut(regions, lb))
        self.cutout_cut_seeds = [s for s in self.cutout_cut_seeds
                                 if regions.region_at(*s) != lb]
        self.cutout_keep_seeds = [s for s in self.cutout_keep_seeds
                                  if regions.region_at(*s) != lb]
        if cur_cut:
            self.cutout_keep_seeds.append(pt)
        else:
            self.cutout_cut_seeds.append(pt)

        self._do_refresh_cutout_preview()
        return True

    def _region_is_cut(self, regions, lb):
        from engine.cutout_utils import resolve_cut_flags
        flags = resolve_cut_flags(regions, self.cutout_cut_seeds, self.cutout_keep_seeds)
        return flags[lb]

    # ------------------------------------------------------------------
    # QUANTO E' FINE IL TRATTO, A QUESTA DIMENSIONE

    def _line_thicken_mm(self) -> float:
        """Il cursore in millimetri di larghezza aggiunta (0 - 1,00 mm)."""
        return self.slider_line_thicken.value() / 20.0

    def _on_line_thicken_changed(self, _v):
        # Il testo definitivo lo scrive _do_refresh_feature_scale, che e'
        # l'unico posto dove si conosce il passo mm/pixel — e senza quello non
        # si sa di quanto si ingrossera' davvero.
        self._refresh_feature_scale()
        self._refresh_std_mockup()
        self._refresh_spot_mockup()

    def _update_thicken_label(self, mm_per_px):
        """Scrive l'ingrossamento REALE, non quello chiesto.

        Fra i due c'e' l'arrotondamento a pixel interi della dilatazione, che
        su una sorgente a bassa risoluzione vale piu' di mezzo millimetro:
        mostrare il valore chiesto farebbe sembrare rotta la misura qui sotto,
        che invece sta dicendo la verita' su un ingrossamento diverso.
        """
        mm = self._line_thicken_mm()
        if mm <= 0:
            self.lbl_line_thicken.setText("Line thickening: off")
            return
        reale = thicken_applied_mm(mm, mm_per_px) if mm_per_px else 0.0
        self.lbl_line_thicken.setText(
            f"Line thickening: +{mm:.2f} mm → none (under 1 px)" if reale <= 0
            else f"Line thickening: +{reale:.2f} mm")

    def _thickened(self, img, mm_per_px):
        """L'immagine come la vedra' il motore dopo l'ingrossamento.

        Le anteprime devono passare di qui, o mostrano una geometria che non
        e' quella che verra' generata — ed e' proprio la geometria la cosa che
        questo cursore cambia.
        """
        mm = self._line_thicken_mm()
        if mm <= 0 or img is None or not mm_per_px:
            return img
        return thicken_ink(img, mm, mm_per_px)

    def _feature_scale_source(self):
        """L'immagine e il passo mm/pixel su cui misurare.

        Il passo e' quello del PEZZO, non del foglio: in modalita' portachiavi
        la sorgente viene ritagliata alla sagoma, quindi un disegno che occupa
        un quarto dell'immagine ha tratti quattro volte piu' grossi di quanto
        direbbe il conto fatto sul foglio intero. Il ritaglio quel passo lo ha
        gia' calcolato (CutoutResult.pitch_mm), e va usato quello.
        """
        if self._is_keychain():
            res = self._compute_cutout_now()
            if res is None or res.empty or not res.pitch_mm:
                return None, None
            src = self._cutout_source()
            return to_seg_raster(src, SEG_MAX_RES), res.pitch_mm

        img = getattr(self, 'img_filtered_array', None)
        if img is None:
            return None, None
        return img, self.spin_dim.value() / max(img.shape[0], img.shape[1])

    def _refresh_feature_scale(self):
        """Accorpa le richieste ravvicinate: la misura costa una distance
        transform, e in portachiavi anche una segmentazione."""
        if getattr(self, '_fscale_timer', None) is None:
            self._fscale_timer = QTimer(self)
            self._fscale_timer.setSingleShot(True)
            self._fscale_timer.timeout.connect(self._do_refresh_feature_scale)
        self._fscale_timer.start(200)

    def _do_refresh_feature_scale(self):
        img, mm_per_px = self._feature_scale_source()
        if img is None:
            self.lbl_feature_scale.setVisible(False)
            self._update_thicken_label(0.0)
            return
        self._update_thicken_label(mm_per_px)
        img = self._thickened(img, mm_per_px)
        try:
            info = feature_scale(img, mm_per_px,
                                 white_clip=self.spin_white_clip.value(),
                                 max_dim_mm=self.spin_dim.value())
        except Exception as e:          # una misura non deve mai fermare la UI
            print(f"Warning: feature scale not computed ({e})")
            self.lbl_feature_scale.setVisible(False)
            return

        self.lbl_feature_scale.setText(info['message'])
        self.lbl_feature_scale.setVisible(True)
        self.lbl_feature_scale.setProperty("state", "" if info['ok'] else "warn")
        self.lbl_feature_scale.style().unpolish(self.lbl_feature_scale)
        self.lbl_feature_scale.style().polish(self.lbl_feature_scale)

    def _get_rgb_filtered(self):
        """Filtro bilaterale RGB calcolato lazy alla prima richiesta (serve solo all'anteprima Topo)."""
        if getattr(self, 'img_rgb_filtered', None) is None:
            if getattr(self, 'img_rgb_original', None) is None:
                return None
            self.lbl_status.setText("🛠 Applying Bilateral Filter (RGB)...")
            QApplication.processEvents()
            self.img_rgb_filtered = cv2.bilateralFilter(self.img_rgb_original, d=5, sigmaColor=50, sigmaSpace=50)
        return self.img_rgb_filtered

    def _on_threshold_changed(self, value):
        self.lbl_threshold.setText(f"Halftone Threshold: {value}%")
        self._refresh_color_mode()

    def _apply_auto_white(self):
        """Apply the suggested white clip value calculated during image analysis."""
        if hasattr(self, 'auto_white_suggestion'):
            self.spin_white_clip.setValue(self.auto_white_suggestion)

    def _refresh_color_mode(self):
        """Update color mode visibility and logic based on slider and midtone analysis."""
        if not hasattr(self, 'last_midtone_pct'):
            return

        threshold = self.slider_threshold.value()
        real_midtones = self.last_midtone_pct
        
        if real_midtones < 100.0:  # If image is loaded
            self.lbl_real_midtones.setText(f"Image Halftones: {real_midtones:.1f}%")

        if real_midtones >= threshold:
            self.color_mode_state = 4
            self.lbl_color_mode.setText("🎨 4-Color Mode (Full Halftone)")
        elif real_midtones >= (threshold / 2.0):
            self.color_mode_state = 3
            self.lbl_color_mode.setText("🌗 3-Color Mode (Partial Halftone)\nL1 hidden, L2/L3 active.")
        else:
            self.color_mode_state = 2
            self.lbl_color_mode.setText("⚫ 2-Color Mode (B&W)\nL1/L2 hidden. L3 low and thick.")

        # A 2 colori il pannello di campionamento diventa un'altra cosa: non
        # c'e' un tono da scegliere, c'e' una soglia. Gli swatch resterebbero
        # quattro pulsanti di cui meta' non fa niente, ed e' meglio non
        # mostrarli che mostrarli inerti.
        due = (self.color_mode_state == 2)
        self.group_swatch.setTitle("Ink Coverage (2-Color Mode)" if due
                                   else "Color Picking (Click to calibrate)")
        self.chk_auto_midtones.setVisible(not due)
        self.lbl_swatch_info.setVisible(not due)
        for _sw in self.swatches:
            _sw.setVisible(not due)
        self.lbl_bw_coverage.setVisible(due)
        self.slider_bw_coverage.setVisible(due)
        self.lbl_bw_note.setVisible(due)
        if due:
            self._refresh_bw_labels()

        # Update visibility for Z Heights
        self.lbl_z1.setVisible(self.color_mode_state == 4)
        self.spin_z1.setVisible(self.color_mode_state == 4)
        
        self.lbl_z2.setVisible(self.color_mode_state >= 3)
        self.spin_z2.setVisible(self.color_mode_state >= 3)

        # Recalculate Z based on new mode
        if self.chk_auto_z.isChecked():
            self._refresh_auto_z_display()

    def _compute_auto_z(self):
        """Return the 3 auto-computed color-change Z heights based on current spinbox values, snapped to layer height."""
        base_z = self.spin_base.value()
        max_z  = self.spin_maxh.value()
        layer_height = self.spin_layer_height.value()
        available_z = max_z - base_z
        
        mode = getattr(self, 'color_mode_state', 4)

        if mode == 2:
            z1 = 0.0
            z2 = 0.0
            z3 = max_z
        elif mode == 3:
            z1 = 0.0
            z3 = max_z
            
            target = base_z + (available_z / 2.0)
            z2 = round(target / layer_height) * layer_height
            
            # Safety Check: assicurati che sia sempre almeno 1 layer sotto il nero
            if z2 >= max_z:
                z2 = max_z - layer_height
        else: # mode == 4
            z3 = max_z
            
            target1 = base_z + (available_z / 3.0)
            z1 = round(target1 / layer_height) * layer_height
            
            target2 = base_z + 2.0 * (available_z / 3.0)
            z2 = round(target2 / layer_height) * layer_height
            
            # Safety Check: assicurati che L1_Z < L2_Z < L3_Z
            if z2 >= z3:
                z2 = z3 - layer_height
            if z1 >= z2:
                z1 = z2 - layer_height
        
        return [round(z1, 3), round(z2, 3), round(z3, 3)]

    def _refresh_auto_z_display(self):
        """Update the Z spinboxes with the currently computed auto values (read-only display)."""
        z1, z2, z3 = self._compute_auto_z()
        self.spin_z1.setValue(z1)
        self.spin_z2.setValue(z2)
        self.spin_z3.setValue(z3)

    def _on_physical_param_changed(self):
        """Called whenever Base or MaxZ spinboxes change — refresh Z display if auto mode is on."""
        if self.chk_auto_z.isChecked():
            self._refresh_auto_z_display()

    def _on_auto_z_toggled(self, checked):
        """Enable/disable the manual Z spinboxes depending on the checkbox state."""
        for sp in (self.spin_z1, self.spin_z2, self.spin_z3):
            sp.setEnabled(not checked)
            sp.setReadOnly(checked)
        if checked:
            self._refresh_auto_z_display()

    def _on_auto_midtones_toggled(self, checked):
        """Enable/disable manual color picking depending on the Auto-Detect checkbox state."""
        for btn in self.swatches:
            btn.setEnabled(not checked)
        if checked and self.img_filtered_array is not None:
            # If re-enabled, calculate right away
            l1, l2 = suggest_midtones(self.img_filtered_array)
            self.sampled_colors[1] = l1
            self.sampled_colors[2] = l2
            self.update_swatch_colors()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File",
            self.last_opened_dir,
            "Images (*.png *.jpg *.jpeg *.jfif *.avif *.webp *.bmp *.tiff *.heic);;All Files (*.*)"
        )
        if not file_path:
            return

        self.last_opened_dir = os.path.dirname(file_path)
        self.loaded_image_path = file_path
        self.lbl_status.setText("Loading image...")
        QApplication.processEvents()
        
        # Load RGB first to avoid redundant conversions
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            try:
                pil_img = Image.open(file_path).convert('RGB')
                self.img_rgb_original = np.array(pil_img)
                img = np.array(pil_img.convert('L'))
            except Exception as e:
                QMessageBox.critical(self, "Decoder Error", f"Unable to read source file:\n{e}")
                self.lbl_status.setText("Loading Failed.")
                return
        else:
            self.img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
        if img is None:
            QMessageBox.critical(self, "Decoder Error", "Unreadable format.")
            self.lbl_status.setText("Loading Failed.")
            return

        # Apply bilateral filter to both for consistency in preview
        self.lbl_status.setText("🛠 Applying Bilateral Filter...")
        QApplication.processEvents()
        self.img_filtered_array = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        # La versione RGB filtrata viene calcolata lazy da _get_rgb_filtered()
        self.img_rgb_filtered = None

        # Display correct version based on mode (spegne anche eventuali anteprime attive)
        self.btn_spot_mockup.setChecked(False)
        self.btn_spot_mockup.setEnabled(True)
        self.btn_std_mockup.setChecked(False)
        self.btn_std_mockup.setEnabled(True)
        # La soglia dell'inchiostro si rilegge dall'istogramma della nuova
        # immagine: e' un valore di quella scansione, non del programma.
        self._ink_level = None
        self.btn_cover_preview.setChecked(False)
        self.btn_cover_preview.setEnabled(True)
        # Il ritaglio e' fatto di scelte su QUESTA immagine: i semi di un'altra
        # nominerebbero regioni che non esistono piu'.
        self.btn_cutout_preview.setChecked(False)
        self.btn_cutout_edit.setChecked(False)
        self.btn_cutout_ring.setChecked(False)
        self.cutout_cut_seeds = []
        self.cutout_keep_seeds = []
        self.cutout_paint_mask = None
        self.cutout_ring_xy = None
        self._invalidate_cutout_regions()
        self.btn_cutout_preview.setEnabled(True)
        self._update_viewport_mode(self.mode_selector.currentIndex())

        h, w = img.shape
        self.lbl_info.setText(f"Preview HD: {w} × {h} px\n{os.path.basename(file_path)}")
        self.btn_generate.setEnabled(True)

        if self.img_filtered_array is None:
            return

        # --- Auto-Color Depth Analysis ---
        total_pixels = self.img_filtered_array.size
        midtone_pixels = np.count_nonzero(
            (self.img_filtered_array > 30) & (self.img_filtered_array < 225)
        )
        self.last_midtone_pct = (midtone_pixels / total_pixels) * 100.0

        # --- Auto-White Clip Suggestion Logic ---
        # Calculate histogram to find the white background peak
        hist = cv2.calcHist([self.img_filtered_array], [0], None, [256], [0, 256])
        # Find the most frequent value in the highlights (200-255)
        white_peak_bin = 200 + np.argmax(hist[200:])
        
        # Suggest a value just below the peak to swallow JPEG noise
        # We look for where the distribution starts rising towards the peak
        suggested_white = white_peak_bin - 15
        
        # Safety bounds
        self.auto_white_suggestion = int(np.clip(suggested_white, 180, 250))
        self.btn_auto_white.setText(f"\ud83e\ude84 {self.auto_white_suggestion}")
        self.btn_auto_white.setEnabled(True)

        # K-Means Auto-Detect Midtones
        if self.chk_auto_midtones.isChecked():
            self.lbl_status.setText("🤖 Analyzing midtones (K-Means)...")
            QApplication.processEvents()
            l1, l2 = suggest_midtones(self.img_filtered_array)
            self.sampled_colors[1] = l1
            self.sampled_colors[2] = l2
            self.update_swatch_colors()
            
            # Disable swatches because Auto is active
            for btn in self.swatches:
                btn.setEnabled(False)

        # Trigger real-time UI update based on new midtone percentage
        self._refresh_color_mode()
        self._refresh_feature_scale()
        self.lbl_status.setText("✅ Ready. Use the layer swatches to pick grey tones.")

    def set_active_swatch(self, idx):
        if self.img_filtered_array is None:
            QMessageBox.information(self, "Warning", "Please load an image before acquiring samples.")
            return

        self.active_swatch_index = idx
        self.lbl_status.setText("🎯 Now Left Click on the image to capture the grey value...")
        for i, btn in enumerate(self.swatches):
            if i == idx:
                btn.setProperty("class", "swatch swatch_active")
            else:
                btn.setProperty("class", "swatch")
            
            # Flush styles
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def on_pixel_clicked(self, x, y):
        # Ramo ritaglio: ha la precedenza su tutto perche' e' l'unico che si
        # arma esplicitamente (Edit regions / Place) e che mostra un'immagine
        # sua, sulla quale un campionamento di colore leggerebbe il giallo
        # dell'anteprima invece del disegno.
        if self._is_keychain() and self._cutout_click(x, y):
            return

        # Ramo Spot Color: campionamento accento (ha priorità quando armato)
        if (self.active_spot_swatch is not None
                and getattr(self, 'img_rgb_original', None) is not None
                and self.mode_selector.currentIndex() == 3):
            idx = self.active_spot_swatch
            self.active_spot_swatch = None
            # In mockup il click deve campionare dall'originale, non dall'anteprima
            if self.btn_spot_mockup.isChecked():
                self.btn_spot_mockup.setChecked(False)
                self.lbl_status.setText("🎯 Mockup off: click again on the ORIGINAL image to sample.")
                self.active_spot_swatch = idx
                return
            self.spot_accents[idx] = self._sample_accent_at(x, y)
            self._update_spot_swatch_colors()
            self.lbl_status.setText(f"✅ Accent {idx+1} set to RGB {self.spot_accents[idx]}.")
            self._refresh_spot_mockup()
            return

        if self.active_swatch_index is not None and self.img_filtered_array is not None:
            val = int(self.img_filtered_array[y, x])
            idx = self.active_swatch_index
            self.sampled_colors[idx] = val
            self.update_swatch_colors()
            
            # Sblocca lo stato di attesa
            self.active_swatch_index = None
            for btn in self.swatches:
                btn.setProperty("class", "swatch")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                
            match_name = self.swatch_labels[idx]
            self.lbl_status.setText(f"✅ Color '{match_name}' correctly assigned to value [{val}].")

    def _extract_topo_colors(self):
        """Extract dominant colors from the current image and populate the list for topo mode."""
        if getattr(self, 'img_rgb_original', None) is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
            
        self.lbl_status.setText("🤖 Extracting dominant colors (K-Means)...")
        QApplication.processEvents()
        
        try:
            # Use the already loaded RGB image
            colors = extract_dominant_colors(self.img_rgb_original, n_colors=5)
            
            self.topo_color_list.clear()
            for i, rgb in enumerate(colors):
                item = QListWidgetItem(f"Layer {i+1}: RGB {rgb}")
                item.setData(Qt.ItemDataRole.UserRole, rgb)
                
                # Visual feedback: set background color
                color = QColor(*rgb)
                item.setBackground(color)
                # Contrast text: calculate luminance for text readability
                lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                item.setForeground(QColor(255, 255, 255) if lum < 128 else QColor(0, 0, 0))
                
                self.topo_color_list.addItem(item)
                
            self.lbl_status.setText(f"✅ Extracted {len(colors)} colors. Drag to reorder (Top=Base).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Color extraction failed:\n{e}")

    def update_swatch_colors(self):
        for i, btn in enumerate(self.swatches):
            val = self.sampled_colors[i]
            btn.setText(f"{self.swatch_labels[i]} : [ {val} ]")
            
            r = g = b = val
            text_color = "white" if val < 128 else "black"
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color};")
            
            # Refresh styles
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def generate_stl(self):
        if self.img_filtered_array is None or getattr(self, 'loaded_image_path', None) is None:
            return

        is_cover = (self.mode_selector.currentIndex() == 4)
        if is_cover and self._current_phone_preset() is None:
            QMessageBox.warning(self, "Phone Cover", "Please select a phone model.")
            return

        export_3mf = self.chk_export_3mf.isChecked()
        export_stl = self.chk_export_stl.isChecked()
        
        if not export_3mf and not export_stl:
            QMessageBox.warning(self, "Export Error", "Please select at least one export format.")
            return

        # --- Auto-compute output paths (no user prompt) ---
        base_dir = os.path.dirname(self.loaded_image_path)
        base_name = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        # Il nome dice da quale modalità viene il file, perché nella cartella
        # output/ finiscono fianco a fianco e "<nome>_3D" da solo non basta a
        # distinguere un pannello da un portachiavi dello stesso disegno.
        # Prefisso solo dove la modalità produce PIÙ file che vanno tenuti
        # insieme (cover_plate_ / cover_bumper_, full_deckbox_): lì raggruppa
        # per ruolo. Il portachiavi ne produce uno, quindi segue i pannelli e
        # va in coda.
        if is_cover:
            file_stem = f"cover_plate_{base_name}"
        elif self._is_keychain():
            file_stem = f"{base_name}_keychain"
        else:
            file_stem = f"{base_name}_3D"

        save_path_stl = None
        if export_stl:
            output_dir_stl = os.path.join(base_dir, "output", "stl")
            os.makedirs(output_dir_stl, exist_ok=True)
            save_path_stl = os.path.join(output_dir_stl, f"{file_stem}.stl")
            counter = 1
            while os.path.exists(save_path_stl):
                save_path_stl = os.path.join(output_dir_stl, f"{file_stem}_{counter}.stl")
                counter += 1

        save_path_3mf = None
        if export_3mf:
            output_dir_3mf = os.path.join(base_dir, "output", "3mf")
            os.makedirs(output_dir_3mf, exist_ok=True)
            save_path_3mf = os.path.join(output_dir_3mf, f"{file_stem}.3mf")
            counter = 1
            while os.path.exists(save_path_3mf):
                save_path_3mf = os.path.join(output_dir_3mf, f"{file_stem}_{counter}.3mf")
                counter += 1

        # --- Compute or read color-change Z heights ---
        base_h = self.spin_base.value()
        max_h  = self.spin_maxh.value()
        relief = max_h - base_h

        if self.chk_auto_z.isChecked():
            color_changes_z = self._compute_auto_z()
            # Show computed values in the (read-only) spinboxes
            self._refresh_auto_z_display()
        else:
            color_changes_z = [
                round(self.spin_z1.value(), 3),
                round(self.spin_z2.value(), 3),
                round(self.spin_z3.value(), 3),
            ]

        # --- Validate topo colors BEFORE locking the UI (otherwise the app stays frozen) ---
        is_topo = (self.mode_selector.currentIndex() == 1)
        topo_colors = []
        if is_topo:
            for i in range(self.topo_color_list.count()):
                topo_colors.append(self.topo_color_list.item(i).data(Qt.ItemDataRole.UserRole))
            if not topo_colors:
                QMessageBox.warning(self, "Error", "Please extract colors before generating.")
                return

        # --- Lock UI ---
        self.toggle_ui_state(disabled=True)
        self.progress_bar.setValue(0)

        max_res_cap = self._current_max_res_cap()

        # --- Launch background QThread ---

        self.generation_start_time = time.time()
        
        # In Topo/Spot/Cover mode, pass the RGB image instead of the filtered grayscale one
        is_spot = (self.mode_selector.currentIndex() == 3)
        # Il portachiavi segue la sua finitura, come ovunque altro: e' la
        # stessa scelta che fa _cutout_source(), e le due devono coincidere o
        # i confini delle regioni mostrati non sono quelli generati.
        want_rgb = is_topo or is_spot or is_cover or (self._is_keychain() and self._keychain_spot())
        input_img = self.img_rgb_original if want_rgb else self.img_filtered_array

        params = GenerationParams(
            mode=self._current_generation_mode(),
            max_dim=self.spin_dim.value(),
            base_h=base_h,
            max_h=max_h,
            layer_height=self.spin_layer_height.value(),
            max_res_cap=max_res_cap,
            smart_decimate=self.chk_smart_decimate.isChecked(),
            white_clip=self.spin_white_clip.value(),
            black_clip=self.spin_black_clip.value(),
            sampled_values=self.sampled_colors,
            color_mode=getattr(self, 'color_mode_state', 4),
            color_changes_z=color_changes_z,
            bw_coverage=self._current_bw_coverage(),
            line_thicken_mm=self._line_thicken_mm(),
            topo_colors=topo_colors,
            spot_accents=self._get_spot_accents(),
            spot_coverage=self.slider_spot_coverage.value(),
            tcg_name=self.combo_tcg_select.currentText(),
            cover_preset=self._current_phone_preset(),
            cover_scale=self.slider_cover_scale.value() / 100.0,
            cover_off_x=float(self.slider_cover_offx.value()),
            cover_off_y=float(self.slider_cover_offy.value()),
            cover_finish_spot=(self.combo_cover_finish.currentIndex() == 1),
            cover_avoid_camera=self.chk_cover_avoid_camera.isChecked(),
            cover_engraved=(self.combo_cover_surface.currentIndex() == 0),
            cover_gray_levels=self.combo_cover_levels.currentIndex() + 2,
            include_bumper=self.chk_cover_bumper.isChecked(),
            keychain_finish_spot=self._keychain_spot(),
            cutout_cut_seeds=list(self.cutout_cut_seeds),
            cutout_keep_seeds=list(self.cutout_keep_seeds),
            cutout_paint_mask=(self.cutout_paint_mask
                               if self.combo_cutout_src.currentIndex() == 1 else None),
            cutout_seg_res=SEG_MAX_RES,
            cutout_border_mm=self.slider_cutout_border.value() / 10.0,
            cutout_ring=self.chk_cutout_ring.isChecked(),
            cutout_ring_xy=self.cutout_ring_xy,
            cutout_ring_d_mm=self.spin_ring_d.value(),
            output_path=save_path_stl,
            output_path_3mf=save_path_3mf,
            source_image_name=base_name,
        )
        self.worker = MeshWorker(params, input_img)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_generate_done)
        self.worker.finished_err.connect(self.on_generate_error)
        self.worker.start()

    def on_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def toggle_ui_state(self, disabled=True):
        for wdg in self.lockable_widgets:
            wdg.setEnabled(not disabled)

        if not disabled:
            # Ripristina gli stati condizionali sovrascritti dal lock generale
            auto_z = self.chk_auto_z.isChecked()
            for sp in (self.spin_z1, self.spin_z2, self.spin_z3):
                sp.setEnabled(not auto_z)
                sp.setReadOnly(auto_z)
            for btn in self.swatches:
                btn.setEnabled(not self.chk_auto_midtones.isChecked())
            self.btn_spot_mockup.setEnabled(self.img_filtered_array is not None)
            self.btn_std_mockup.setEnabled(self.img_filtered_array is not None)
            self.btn_cover_preview.setEnabled(self.img_filtered_array is not None)
            # I controlli del ritaglio dipendono dalla spunta, e quelli a click
            # anche dalla sorgente scelta: l'unlock generale li riaccenderebbe
            # tutti, promettendo comandi che non hanno su cosa agire.
            self._on_cutout_enabled_restore()
            # In Deckbox i parametri fisici restano bloccati anche dopo l'unlock
            self._on_mode_changed(self.mode_selector.currentIndex())

        if disabled:
            self.btn_generate.setText("🛑 Cancel")
            self.btn_generate.setProperty("state", "cancel")
            self.btn_generate.style().unpolish(self.btn_generate)
            self.btn_generate.style().polish(self.btn_generate)
            try:
                self.btn_generate.clicked.disconnect()
            except TypeError:
                pass
            self.btn_generate.clicked.connect(self.cancel_generation)
            self.btn_generate.setEnabled(True)  # Keep the cancel button interactive
        else:
            self.btn_generate.setText("🚀 Generate 3D Models")
            self.btn_generate.setProperty("state", "")
            self.btn_generate.style().unpolish(self.btn_generate)
            self.btn_generate.style().polish(self.btn_generate)
            try:
                self.btn_generate.clicked.disconnect()
            except TypeError:
                pass
            self.btn_generate.clicked.connect(self.generate_stl)
            self.btn_generate.setEnabled(True)

    def _on_cutout_enabled_restore(self):
        """Con la maschera dipinta i comandi a click non hanno regioni su cui
        agire: l'unlock generale li riaccenderebbe, promettendo un controllo
        che in quella strada non esiste."""
        if self.combo_cutout_src.currentIndex() == 1:
            self.btn_cutout_edit.setEnabled(False)
            self.btn_cutout_reset.setEnabled(False)

    def cancel_generation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel_requested = True
            self.lbl_status.setText("🛑 Cancelling process, please wait...")
            self.btn_generate.setProperty("state", "cancel")
            self.btn_generate.style().unpolish(self.btn_generate)
            self.btn_generate.style().polish(self.btn_generate)
            self.btn_generate.setEnabled(False)
            self.btn_generate.setText("Stopping...")

    def unlock_ui(self):
        self.toggle_ui_state(disabled=False)

    def _build_color_change_instructions(self) -> str:
        """
        Build a human-readable string describing the colour-change Z heights
        for the export success dialog. Handles Topo, Standard (4/3/2-colour) modes.
        """
        mode_idx = self.mode_selector.currentIndex()
        if mode_idx in (3, 4):  # Spot Color / Phone Cover
            cover_bn = mode_idx == 4 and self.combo_cover_finish.currentIndex() == 0
            if cover_bn:
                # finitura B/N: livelli di grigio quantizzati (2-4), non gli accenti Spot
                palette = grayscale_palette(self.combo_cover_levels.currentIndex() + 2)
            else:
                accents = self._get_spot_accents()
                palette = build_spot_palette(accents)
            if mode_idx == 4 and self.combo_cover_surface.currentIndex() == 0:
                palette = palette[::-1]  # inciso: scuro per primo, chiaro in superficie
            layer_h  = self.spin_layer_height.value()
            z_heights = compute_topo_z_heights(self.spin_base.value(),
                                               self.spin_maxh.value(),
                                               layer_h, len(palette))
            switch_z = compute_topo_switch_z(z_heights, layer_h)

            lines = "🎯 SPOT COLOR FILAMENT STEPS (Quantized):\n"
            for i, rgb in enumerate(palette):
                if i == 0:
                    lines += f"  • Start with: RGB{rgb} (Base up to {z_heights[0]}mm)\n"
                else:
                    lines += f"  • at Z = {switch_z[i-1]} mm  →  Switch to RGB{rgb}\n"
            return lines

        is_topo = (self.mode_selector.currentIndex() == 1)
        if is_topo:
            base_z   = self.spin_base.value()
            total_z  = self.spin_maxh.value()
            layer_h  = self.spin_layer_height.value()
            n_colors = self.topo_color_list.count()
            z_heights = compute_topo_z_heights(base_z, total_z, layer_h, n_colors)
            # Il cambio va al primo layer sopra la terrazza del colore precedente,
            # non al top della terrazza del colore stesso (off-by-one di banda)
            switch_z = compute_topo_switch_z(z_heights, layer_h)

            lines = "\U0001f3a8 TOPOGRAPHIC FILAMENT STEPS (Quantized):\n"
            for i in range(n_colors):
                rgb = self.topo_color_list.item(i).data(Qt.ItemDataRole.UserRole)
                if i == 0:
                    lines += f"  \u2022 Start with: RGB{rgb} (Base up to {z_heights[0]}mm)\n"
                else:
                    lines += f"  \u2022 at Z = {switch_z[i-1]} mm  \u2192  Switch to RGB{rgb}\n"
            return lines

        mode = getattr(self, 'color_mode_state', 4)
        # Le quote da mostrare sono quelle scritte nel 3MF, calcolate
        # dall'engine sulle terrazze reali della mesh: non gli spinbox, che
        # sono le CIME delle terrazze (un cambio li' colora un layer solo).
        real = getattr(getattr(self, 'worker', None), 'result', None)
        real_z = list(getattr(real, 'color_changes_z', []) or [])
        n_switch = {2: 1, 3: 2}.get(mode, 3)
        if len(real_z) >= n_switch:
            if mode == 4:
                z1, z2, z3 = real_z[0], real_z[1], real_z[2]
            elif mode == 3:
                z1, z2, z3 = 0.0, real_z[0], real_z[1]
            else:
                z1, z2, z3 = 0.0, 0.0, real_z[0]
        else:
            z1   = self.spin_z1.value()
            z2   = self.spin_z2.value()
            z3   = self.spin_z3.value()
        if mode == 4:
            return (
                f"  \u2022 L1 Light Gray  \u2192  Z = {z1} mm  (Filament 2)\n"
                f"  \u2022 L2 Dark Gray   \u2192  Z = {z2} mm  (Filament 3)\n"
                f"  \u2022 L3 Black/Inks  \u2192  Z = {z3} mm  (Filament 4)\n"
            )
        if mode == 3:
            return (
                f"  \u2022 L2 Dark Gray   \u2192  Z = {z2} mm  (Filament 2)\n"
                f"  \u2022 L3 Black/Inks  \u2192  Z = {z3} mm  (Filament 3)\n"
            )
        return f"  \u2022 L3 Black/Inks  \u2192  Z = {z3} mm  (Filament 2)\n"

    def on_generate_done(self, stl_path, path_3mf):
        self.unlock_ui()
        self.progress_bar.setValue(100)
        
        elapsed = time.time() - getattr(self, 'generation_start_time', time.time())
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        self.lbl_status.setText(f"\U0001f3c1 STL + 3MF Export Completed in {time_str}!")

        color_lines = self._build_color_change_instructions()
        is_deckbox = (self.mode_selector.currentIndex() == 2)

        msg = "Files saved:\n"
        if is_deckbox:
            # In deckbox mode the worker emits (full_3mf, full_stl) for the combined plate
            if stl_path:
                msg += f"🎨 3MF (Full Plate) → {stl_path}\n"
            if path_3mf:
                msg += f"📄 STL (Full Plate) → {path_3mf}\n"
        else:
            if stl_path:
                msg += f"📄 STL → {stl_path}\n"
            if path_3mf:
                msg += f"🎨 3MF → {path_3mf}\n"
            if (self.mode_selector.currentIndex() == 4 and self.chk_cover_bumper.isChecked()
                    and (stl_path or path_3mf)):
                companion = MeshWorker.companion_path_for(stl_path or path_3mf)
                msg += f"🧷 Cover/Bumper (print in TPU) → {companion}\n"
            
        msg += f"\n⏱️ Time elapsed: {time_str}\n\n"

        # Quello che il ritaglio ha dovuto decidere da solo. Sono scelte
        # ragionevoli ma non ovvie, e restano invisibili nel file: se il
        # disegno era in piu' tronconi ne e' uscito uno, e un occhiello
        # staccato e' un anellino che si stacca alla prima tirata.
        res = getattr(getattr(self, 'worker', None), 'result', None)
        if self._is_keychain() and res is not None:
            if getattr(res, 'cutout_n_pieces', 0) > 1:
                msg += (f"⚠️  The drawing was in {res.cutout_n_pieces} separate pieces:\n"
                        f"    only the largest one was kept.\n"
                        f"    Use the Sticker border to join them, or keep\n"
                        f"    a void solid so that it bridges them.\n\n")
            if self.chk_cutout_ring.isChecked() and not getattr(res, 'cutout_ring_attached', True):
                msg += ("⚠️  The keyring hole is not touching the piece:\n"
                        "    reposition it inside the material with 📍 Place.\n\n")

        
        if path_3mf:
            msg += (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎨  BAMBU STUDIO — Color Changes\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"After slicing, add these layer pauses\n"
                f"via the colored bar on the right side:\n\n"
                f"{color_lines}\n"
            )
            
        msg += (
            f"💡 PRO TIP: For high-detail manga panels,\n"
            f"set Wall Generator to Arachne in Bambu Studio.\n"
            f"This prevents fine lines and small details\n"
            f"from disappearing during slicing!"
        )

        QMessageBox.information(self, "Export Successful", msg)

    def on_generate_error(self, err_msg):
        self.unlock_ui()
        if "cancelled" in err_msg.lower():
            self.lbl_status.setText("🛑 Process cancelled by user.")
            self.progress_bar.setValue(0)
            QMessageBox.information(self, "Cancelled", "Generazione annullata correttamente.")
        else:
            self.lbl_status.setText("❌ Critical QThread error.")
            QMessageBox.critical(self, "Worker Error", f"The generation thread crashed:\n{err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Caricamento dello stile dal file esterno QSS
    try:
        with open(resource_path("style.qss"), "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Warning: Could not load style.qss: {e}")
        
    window = Manga3DAppController()
    window.show()
    sys.exit(app.exec())

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
from PyQt6.QtCore import Qt

from utils import resource_path
from ui_main_window import MainWindowUI
from color_utils import (extract_dominant_colors, suggest_midtones,
                         suggest_spot_accents, classify_spot_pixels,
                         downsample_for_analysis, build_spot_palette,
                         grayscale_palette)
from mesh_utils import compute_topo_z_heights, compute_topo_switch_z
from case_utils import (load_phone_presets, build_plate_raster,
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
        self.active_swatch_index = None
        self.loaded_image_path = None
        self.last_opened_dir = ""
        self.sampled_colors = [250, 210, 150, 15]
        self.color_mode_state = 4
        self.last_midtone_pct = 100.0

        # Spot Color state
        self.spot_accents = [None, None]
        self.active_spot_swatch = None

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
        self.btn_spot_mockup.toggled.connect(self._on_spot_mockup_toggled)

    def _update_viewport_mode(self, index):
        """Switch viewport display between Color and Grayscale based on selected mode."""
        # Cambiando modalità le anteprime toggle si spengono sempre
        if self.btn_spot_mockup.isChecked():
            self.btn_spot_mockup.setChecked(False)
            return  # il toggle handler richiama questo metodo col viewport giusto
        if self.btn_cover_preview.isChecked():
            self.btn_cover_preview.setChecked(False)
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

    def _on_mode_defaults(self, index):
        """Entrando in modalità Cover imposta i default 'slim' per la sede
        della plate (spessore max ~1mm, layer fini per le bande colore)."""
        if index == 4:
            self.spin_base.setValue(0.3)
            self.spin_maxh.setValue(1.0)
            self.spin_layer_height.setValue(0.10)

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
            f"👁 Plate {self.combo_phone_model.currentText()}: trascina con gli slider, "
            f"i fori camera sono le zone scure.")

    # ------------------------------------------------------------------
    # SPOT COLOR — picking, auto-detect, mockup
    # ------------------------------------------------------------------

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
            self._refresh_spot_mockup()
            self.btn_spot_mockup.setText("👁 Back to Original")
        else:
            self.btn_spot_mockup.setText("👁 Mockup Preview")
            self._update_viewport_mode(self.mode_selector.currentIndex())

    def _refresh_spot_mockup(self):
        """Ricalcola l'anteprima posterizzata (solo se il mockup è attivo)."""
        if not self.btn_spot_mockup.isChecked():
            return
        if getattr(self, 'img_rgb_original', None) is None:
            return
        # A ~800px la classificazione è istantanea e l'anteprima resta fedele
        small = downsample_for_analysis(self.img_rgb_original)
        palette, idx = classify_spot_pixels(small, self._get_spot_accents(),
                                            coverage=self.slider_spot_coverage.value())
        self.spot_preview_array = np.array(palette, dtype=np.uint8)[idx]
        self.viewer.setImage(self.spot_preview_array)
        names = ", ".join(f"RGB{p}" for p in palette)
        self.lbl_status.setText(f"👁 Mockup: {len(palette)} colors → {names}")

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
        self.lbl_status.setText("Caricamento immagine...")
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
        self.btn_cover_preview.setChecked(False)
        self.btn_cover_preview.setEnabled(True)
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
            QMessageBox.warning(self, "Phone Cover", "Seleziona un modello di telefono.")
            return

        export_3mf = self.chk_export_3mf.isChecked()
        export_stl = self.chk_export_stl.isChecked()
        
        if not export_3mf and not export_stl:
            QMessageBox.warning(self, "Export Error", "Please select at least one export format.")
            return

        # --- Auto-compute output paths (no user prompt) ---
        base_dir = os.path.dirname(self.loaded_image_path)
        base_name = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        # In modalità cover i file si chiamano cover_plate_<nome> (+ cover_bumper_<nome>)
        file_stem = f"cover_plate_{base_name}" if is_cover else f"{base_name}_3D"

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

        # --- Parse selected quality ---
        quality_str = self.cmb_quality.currentText()
        if "800" in quality_str:
            max_res_cap = 800
        elif "1600" in quality_str:
            max_res_cap = 1600
        else:
            max_res_cap = 1200

        # --- Launch background QThread ---

        self.generation_start_time = time.time()
        
        # In Topo/Spot/Cover mode, pass the RGB image instead of the filtered grayscale one
        is_spot = (self.mode_selector.currentIndex() == 3)
        input_img = self.img_rgb_original if (is_topo or is_spot or is_cover) else self.img_filtered_array

        self.worker = MeshWorker(
            img_filtered=input_img,
            sampled_values=self.sampled_colors,
            max_dim=self.spin_dim.value(),
            max_h=max_h,
            base_h=base_h,
            output_path=save_path_stl,
            output_path_3mf=save_path_3mf,
            color_changes_z=color_changes_z,
            layer_height=self.spin_layer_height.value(),
            max_res_cap=max_res_cap,
            smart_decimate=self.chk_smart_decimate.isChecked(),
            white_clip=self.spin_white_clip.value(),
            black_clip=self.spin_black_clip.value(),
            color_mode=getattr(self, 'color_mode_state', 4),
            is_deckbox_mode=(self.mode_selector.currentIndex() == 2),
            tcg_name=self.combo_tcg_select.currentText(),
            is_topo_mode=is_topo,
            topo_colors=topo_colors,
            source_image_name=base_name,
            is_spot_mode=is_spot,
            spot_accents=self._get_spot_accents(),
            spot_coverage=self.slider_spot_coverage.value(),
            is_cover_mode=is_cover,
            cover_preset=self._current_phone_preset(),
            cover_scale=self.slider_cover_scale.value() / 100.0,
            cover_off_x=float(self.slider_cover_offx.value()),
            cover_off_y=float(self.slider_cover_offy.value()),
            cover_finish_spot=(self.combo_cover_finish.currentIndex() == 1),
            include_bumper=self.chk_cover_bumper.isChecked(),
            cover_avoid_camera=self.chk_cover_avoid_camera.isChecked(),
            cover_engraved=(self.combo_cover_surface.currentIndex() == 0),
            cover_gray_levels=self.combo_cover_levels.currentIndex() + 2
        )
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
            self.btn_cover_preview.setEnabled(self.img_filtered_array is not None)
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
                msg += f"🧷 Cover/Bumper (stampa in TPU) → {companion}\n"
            
        msg += f"\n⏱️ Time elapsed: {time_str}\n\n"
        
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

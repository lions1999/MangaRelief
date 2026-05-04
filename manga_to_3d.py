import sys
import os
import time
import cv2
import numpy as np
from PIL import Image
import pillow_heif
import ctypes

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from utils import resource_path
from ui_main_window import MainWindowUI
from core_engine import MeshWorker

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
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self.chk_auto_z.toggled.connect(self._on_auto_z_toggled)
        
        self.spin_base.valueChanged.connect(self._on_physical_param_changed)
        self.spin_maxh.valueChanged.connect(self._on_physical_param_changed)
        self.spin_layer_height.valueChanged.connect(self._on_physical_param_changed)

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
        
        # Update visibility for Clipping controls
        self.lbl_l1_clip.setVisible(self.color_mode_state == 4)
        self.spin_l1_clip.setVisible(self.color_mode_state == 4)
        
        self.lbl_l2_clip.setVisible(self.color_mode_state >= 3)
        self.spin_l2_clip.setVisible(self.color_mode_state >= 3)

        # Recalculate Z based on new mode
        if self.chk_auto_z.isChecked():
            self._refresh_auto_z_display()

    def _compute_auto_z(self):
        """Return the 3 auto-computed color-change Z heights based on current spinbox values, snapped to layer height."""
        base_h = self.spin_base.value()
        max_h  = self.spin_maxh.value()
        layer_h = self.spin_layer_height.value()
        relief = max_h - base_h
        
        mode = getattr(self, 'color_mode_state', 4)

        # Calculate theoretical heights
        if mode == 2:
            z1_theo = 0.0
            z2_theo = 0.0
            z3_theo = base_h + (layer_h * 2.0)
        elif mode == 3:
            z1_theo = 0.0
            z2_theo = base_h + (layer_h * 2.0)
            z3_theo = base_h + max(layer_h * 4.0, 0.66 * relief)
        else: # mode == 4
            z1_theo = base_h + 0.33 * relief
            z2_theo = base_h + 0.66 * relief
            z3_theo = base_h + 1.00 * relief
            
        # Ensure they don't exceed max_h
        z2_theo = min(z2_theo, max_h)
        z3_theo = min(z3_theo, max_h)
        
        # Snap to nearest multiple of layer height
        z1 = round(z1_theo / layer_h) * layer_h
        z2 = round(z2_theo / layer_h) * layer_h
        z3 = round(z3_theo / layer_h) * layer_h
        
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

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File",
            self.last_opened_dir,   # start in last-used folder
            "Images (*.png *.jpg *.jpeg *.jfif *.avif *.webp *.bmp *.tiff *.heic);;All Files (*.*)"
        )
        if not file_path:
            return

        # Remember this folder for next time
        self.last_opened_dir = os.path.dirname(file_path)
        self.loaded_image_path = file_path
        self.lbl_status.setText("Caricamento immagine...")
        QApplication.processEvents()
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                pil_img = Image.open(file_path).convert('L')
                img = np.array(pil_img)
            except Exception as e:
                QMessageBox.critical(self, "Decoder Error", f"Unable to read source file:\n{e}")
                self.lbl_status.setText("Loading Failed.")
                return
                
        if img is None:
            QMessageBox.critical(self, "Decoder Error", "Unreadable format.")
            self.lbl_status.setText("Loading Failed.")
            return

        h, w = img.shape
        self.lbl_info.setText(f"Preview HD: {w} × {h} px\n{os.path.basename(file_path)}")
        
        # Filtro CV2
        self.lbl_status.setText("🛠 Applying Bilateral Filter...")
        QApplication.processEvents()
        self.img_filtered_array = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        
        self.viewer.setImage(self.img_filtered_array)
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

    def update_swatch_colors(self):
        for i, btn in enumerate(self.swatches):
            val = self.sampled_colors[i]
            # Aggiunge visivamente il numero catturato testualmente sul pulsante!
            btn.setText(f"{self.swatch_labels[i]} : [ {val} ]")
            
            # Dark theme adaptation
            style = f"background-color: rgb({val},{val},{val}); "
            if val < 130:
                style += "color: #ffffff; border: 1px solid #7f849c;"
            else:
                style += "color: #11111b; border: 1px solid #45475a;"
            
            # Se ha una classe diversa da "swatch", aggiungiamo il bordo pink per indicare active overridando
            current_class = btn.property("class")
            if current_class == "swatch swatch_active":
                style += "border: 2px solid #f38ba8;"
                
            btn.setStyleSheet(style)

    def generate_stl(self):
        if self.img_filtered_array is None or getattr(self, 'loaded_image_path', None) is None:
            return

        # --- Auto-compute output paths (no user prompt) ---
        base_dir = os.path.dirname(self.loaded_image_path)
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        save_path_stl = os.path.join(output_dir, f"{base_name}_3D.stl")
        save_path_3mf = os.path.join(output_dir, f"{base_name}_3D.3mf")

        # Anti-overwrite: append progressive counter to BOTH files simultaneously
        counter = 1
        while os.path.exists(save_path_stl) or os.path.exists(save_path_3mf):
            save_path_stl = os.path.join(output_dir, f"{base_name}_3D_{counter}.stl")
            save_path_3mf = os.path.join(output_dir, f"{base_name}_3D_{counter}.3mf")
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

        # --- Lock UI ---
        self.btn_load.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)
        for btn in self.swatches:
            btn.setEnabled(False)

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
        self.worker = MeshWorker(
            img_filtered=self.img_filtered_array,
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
            l1_clip=self.spin_l1_clip.value(),
            l2_clip=self.spin_l2_clip.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_generate_done)
        self.worker.finished_err.connect(self.on_generate_error)
        self.worker.start()

    def on_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def unlock_ui(self):
        self.btn_load.setEnabled(True)
        self.btn_generate.setEnabled(True)
        for btn in self.swatches:
            btn.setEnabled(True)

    def on_generate_done(self, stl_path, path_3mf):
        self.unlock_ui()
        self.progress_bar.setValue(100)
        
        elapsed = time.time() - getattr(self, 'generation_start_time', time.time())
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        self.lbl_status.setText(f"🏁 STL + 3MF Export Completed in {time_str}!")

        # Retrieve the Z values that were actually used
        z1 = self.spin_z1.value()
        z2 = self.spin_z2.value()
        z3 = self.spin_z3.value()

        # Build color-change instructions based on detected mode
        mode = getattr(self, 'color_mode_state', 4)
        if mode == 4:
            color_lines = (
                f"  • L1 Light Gray  →  Z = {z1} mm  (Filament 2)\n"
                f"  • L2 Dark Gray   →  Z = {z2} mm  (Filament 3)\n"
                f"  • L3 Black/Inks  →  Z = {z3} mm  (Filament 4)\n"
            )
        elif mode == 3:
            color_lines = (
                f"  • L2 Dark Gray   →  Z = {z2} mm  (Filament 2)\n"
                f"  • L3 Black/Inks  →  Z = {z3} mm  (Filament 3)\n"
            )
        else: # mode == 2
            # 2-color mode: only show L3
            color_lines = (
                f"  • L3 Black/Inks  →  Z = {z3} mm  (Filament 2)\n"
            )

        QMessageBox.information(
            self, "Export Successful",
            f"Files saved:\n"
            f"📄 STL → {stl_path}\n"
            f"🎨 3MF → {path_3mf}\n\n"
            f"⏱️ Time elapsed: {time_str}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎨  BAMBU STUDIO — Color Changes\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"After slicing, add these layer pauses\n"
            f"via the colored bar on the right side:\n\n"
            f"{color_lines}\n"
            f"💡 PRO TIP: For high-detail manga panels,\n"
            f"set Wall Generator to Arachne in Bambu Studio.\n"
            f"This prevents fine lines and small details\n"
            f"from disappearing during slicing!"
        )

    def on_generate_error(self, err_msg):
        self.unlock_ui()
        self.lbl_status.setText("❌ Critical QThread error.")
        QMessageBox.critical(self, "Trimesh Error", f"An error occurred during the 3D generation:\n{err_msg}")


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

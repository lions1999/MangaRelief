import sys
import os
import io
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import trimesh
from PIL import Image
import pillow_heif

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSplitter, QProgressBar, QDoubleSpinBox, QSpinBox, QMessageBox, 
                             QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QIcon
import ctypes

# Imposta l'AppUserModelID di Windows per mostrare l'icona nativa sulla taskbar
try:
    myappid = 'antigravity.mangareliefpro.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

# Helper per i percorsi dei file congelati da PyInstaller (per trovare l'icona estratta)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Abilitiamo i plugin HEIF e AVIF in caso di fallimento OpenCV
try:
    pillow_heif.register_heif_opener()
except AttributeError:
    pass
try:
    pillow_heif.register_avif_opener()
except AttributeError:
    pass

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QWidget {
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    font-weight: bold;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
QPushButton {
    background-color: #89b4fa;
    color: #11111b;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #a6adc8;
}
QPushButton.swatch {
    background-color: #313244;
    color: #cdd6f4;
    border: 2px solid #585b70;
}
QPushButton.swatch_active {
    border: 2px solid #f38ba8;
    background-color: #45475a;
}
QDoubleSpinBox {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 3px;
    padding: 4px;
}
QProgressBar {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 3px;
}
QLabel {
    font-size: 13px;
}
"""

# ---------------------------------------------------------------------------
# .3MF EXPORT — Bambu Studio compatible (geometry via Trimesh, metadata injected)
# ---------------------------------------------------------------------------

def export_3mf(mesh, output_path_3mf, color_changes_z):
    """
    Exports a Trimesh mesh as a valid .3mf file compatible with Bambu Studio.

    Strategy:
      1. Let Trimesh generate a spec-compliant .3mf (with proper XML geometry)
         into an in-memory buffer.
      2. Open that buffer as a ZIP archive in APPEND mode.
      3. Inject the Bambu color-change metadata XML as an extra entry.
      4. Write the final bytes to disk.

    Args:
        mesh (trimesh.Trimesh): The fully-built, watertight mesh object.
        output_path_3mf (str): Destination path, e.g. 'output/panel_3D.3mf'.
        color_changes_z (list[float]): Ascending list of Z heights (mm) at which
            a color-change pause (M600) should be inserted, e.g. [1.0, 1.6, 2.5].
    """
    # --- 1. Let Trimesh generate a geometrically valid .3mf in memory ---
    # This produces the correct 3D/3dmodel.model XML that Bambu Studio requires.
    tmf_buffer = io.BytesIO()
    mesh.export(tmf_buffer, file_type='3mf')
    tmf_buffer.seek(0)  # Rewind so zipfile can read from the beginning

    # --- 2. Build the Bambu color-change metadata XML ---
    # One <color_change> node per halftone terrain level (e.g. 3 nodes for 3 levels)
    config = ET.Element("config")
    plate = ET.SubElement(config, "plate")
    ET.SubElement(plate, "metadata", name="schema_version", value="2")

    for z_height in sorted(color_changes_z):
        cc = ET.SubElement(plate, "color_change")
        cc.set("z", str(round(z_height, 3)))
        cc.set("extruder", "1")  # Single-extruder workflow: always extruder 1

    settings_xml_bytes = ET.tostring(
        config, encoding='unicode', xml_declaration=False
    ).encode('utf-8')

    # --- 3. Open the Trimesh-generated archive in APPEND mode and inject metadata ---
    # 'a' mode adds files without touching existing entries.
    with zipfile.ZipFile(tmf_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Metadata/Bambu_model_settings.xml", settings_xml_bytes)

    # --- 4. Write the enriched archive buffer to disk ---
    tmf_buffer.seek(0)
    with open(output_path_3mf, 'wb') as f:
        f.write(tmf_buffer.read())


# ---------------------------------------------------------------------------
# BACKGROUND MESH WORKER
# ---------------------------------------------------------------------------
class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    def __init__(self, img_filtered, sampled_values, max_dim, max_h, base_h,
                 output_path, output_path_3mf, color_changes_z, max_res):
        super().__init__()
        self.img_filtered = img_filtered
        self.sampled_values = sampled_values
        self.max_dim = max_dim
        self.max_h = max_h
        self.base_h = base_h
        self.output_path = output_path
        self.output_path_3mf = output_path_3mf
        self.color_changes_z = color_changes_z
        self.max_res = max_res

    def run(self):
        try:
            self.progress.emit(5, "Optimizing resolution for 3D mesh...")
            img = self.img_filtered
            
            # Ridimensioniamo la foto solo per la costruzione del 3D, lasciando intatta l'alta risoluzione dell'interfaccia UI
            h, w = img.shape
            if w > self.max_res or h > self.max_res:
                scale_res = self.max_res / max(w, h)
                new_w, new_h = int(w * scale_res), int(h * scale_res)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            img_filtered = img
            self.progress.emit(10, "Preparing logical matrices...")
            
            # Ordinamento soglie
            s0, s1, s2, s3 = sorted(self.sampled_values, reverse=True)
            th_1 = (s0 + s1) / 2.0
            th_2 = (s1 + s2) / 2.0
            th_3 = (s2 + s3) / 2.0
            
            self.progress.emit(25, "Calculating asymmetric terraced extrusion...")
            norm_img = np.zeros_like(img_filtered, dtype=float)
            norm_img[img_filtered < th_1] = 0.33
            norm_img[img_filtered < th_2] = 0.66
            norm_img[img_filtered < th_3] = 1.0
            
            relief_height = self.max_h - self.base_h
            Z = self.base_h + (norm_img * relief_height)
            
            self.progress.emit(40, "Generating mathematical X and Y points (MeshGrid)...")
            h, w = img_filtered.shape
            width_pixel = w
            height_pixel = h

            if width_pixel >= height_pixel:
                dim_x_mm = float(self.max_dim)
                dim_y_mm = float(self.max_dim) * (height_pixel / float(width_pixel))
            else:
                dim_y_mm = float(self.max_dim)
                dim_x_mm = float(self.max_dim) * (width_pixel / float(height_pixel))

            x = np.linspace(0, dim_x_mm, width_pixel)
            y = np.linspace(0, dim_y_mm, height_pixel)[::-1]
            X, Y = np.meshgrid(x, y)
            
            self.progress.emit(55, "Reconstructing TOP FACES (Triangulation)...")
            vertices_top = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            idx = np.arange(w * h).reshape((h, w))
            tl = idx[:-1, :-1].flatten()
            tr = idx[:-1, 1:].flatten()
            bl = idx[1:, :-1].flatten()
            br = idx[1:, 1:].flatten()
            faces_top = np.vstack((np.column_stack((bl, tr, tl)), np.column_stack((br, tr, bl))))
            
            self.progress.emit(70, "Reconstructing BOTTOM MANIFOLD (Inverse triangulation)...")
            vertices_bottom = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(Z.flatten())))
            offset = w * h
            tl_b = tl + offset
            tr_b = tr + offset
            bl_b = bl + offset
            br_b = br + offset
            faces_bottom = np.vstack((np.column_stack((tl_b, tr_b, bl_b)), np.column_stack((bl_b, tr_b, br_b))))
            
            self.progress.emit(85, "Sealing side edges to ensure solidity...")
            v1, v2 = idx[0, :-1], idx[0, 1:]
            top_sides = np.vstack((np.column_stack((v1, v2, v1 + offset)), np.column_stack((v2, v2 + offset, v1 + offset))))
            
            v1, v2 = idx[-1, :-1], idx[-1, 1:]
            bot_sides = np.vstack((np.column_stack((v2, v1, v1 + offset)), np.column_stack((v2 + offset, v2, v1 + offset))))
            
            v1, v2 = idx[:-1, 0], idx[1:, 0]
            left_sides = np.vstack((np.column_stack((v1, v2, v1 + offset)), np.column_stack((v2, v2 + offset, v1 + offset))))
            
            v1, v2 = idx[:-1, -1], idx[1:, -1]
            right_sides = np.vstack((np.column_stack((v2, v1, v1 + offset)), np.column_stack((v2 + offset, v2, v1 + offset))))
            
            self.progress.emit(90, "Converting to 3D OBJECT (Trimesh Repair)...")
            all_vertices = np.vstack((vertices_top, vertices_bottom))
            all_faces = np.vstack((faces_top, faces_bottom, top_sides, bot_sides, left_sides, right_sides))
            
            mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
            trimesh.repair.fix_normals(mesh) # Necessario per renderlo watertight
            
            self.progress.emit(96, "Exporting STL file...")
            mesh.export(self.output_path)

            self.progress.emit(98, "Packaging .3MF for Bambu Studio...")
            export_3mf(mesh, self.output_path_3mf, self.color_changes_z)

            self.progress.emit(100, "STL + 3MF exported successfully!")
            self.finished_ok.emit(self.output_path, self.output_path_3mf)
            
        except Exception as e:
            self.finished_err.emit(str(e))

class ImageGraphicsView(QGraphicsView):
    pixelClicked = pyqtSignal(int, int) # Ritorna le coordinate X, Y originali

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.is_panning = False
        self.pan_start_pos = None

    def setImage(self, img_filtered_array):
        h, w = img_filtered_array.shape
        if not img_filtered_array.flags['C_CONTIGUOUS']:
            img_filtered_array = np.ascontiguousarray(img_filtered_array)
            
        qimage = QImage(img_filtered_array.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)
        
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.is_panning = True
            self.pan_start_pos = event.pos()
        elif event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            # Controlla se il clic è all'interno della foto
            if 0 <= x < self.pixmap_item.pixmap().width() and 0 <= y < self.pixmap_item.pixmap().height():
                self.pixelClicked.emit(x, y)
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning and self.pan_start_pos is not None:
            delta = event.pos() - self.pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_start_pos = event.pos()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.is_panning = False
            self.pan_start_pos = None
        super().mouseReleaseEvent(event)

class Manga3DApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MangaRelief Pro")
        self.setWindowIcon(QIcon(resource_path('icon.ico')))
        self.resize(1200, 800)
        
        self.img_filtered_array = None
        self.active_swatch_index = None
        self.loaded_image_path = None
        # Valori di default di ripiego
        self.sampled_colors = [250, 210, 150, 15] 
        
        self.initUI()
        self.update_swatch_colors()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- LEFT PANEL: VIEWPORT ---
        self.viewer = ImageGraphicsView()
        self.viewer.pixelClicked.connect(self.on_pixel_clicked)
        splitter.addWidget(self.viewer)
        
        # --- RIGHT PANEL: CONTROLLI ---
        right_panel = QWidget()
        right_panel.setFixedWidth(320) # Impedisce alla colonna laterale di allargarsi e restringersi
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 0, 10, 0)
        
        self.btn_load = QPushButton("📂 Load Manga")
        self.btn_load.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_load)
        
        self.lbl_info = QLabel("No project opened.")
        self.lbl_info.setStyleSheet("color: #a6adc8; margin-bottom: 20px;")
        right_layout.addWidget(self.lbl_info)
        
        # SWATCH PANEL
        group_swatch = QGroupBox("Color Picking (Click to calibrate)")
        swatch_layout = QVBoxLayout()
        
        self.lbl_swatch_info = QLabel("Choose a layer below and click on the image.")
        self.lbl_swatch_info.setWordWrap(True)
        swatch_layout.addWidget(self.lbl_swatch_info)
        
        self.swatches = []
        self.swatch_labels = [
            "L0 (White/BG)", 
            "L1 (Light Gray)", 
            "L2 (Dark Gray)", 
            "L3 (Black/Inks)"
        ]
        
        for i in range(4):
            btn = QPushButton(self.swatch_labels[i])
            btn.setProperty("class", "swatch")
            btn.clicked.connect(lambda checked, idx=i: self.set_active_swatch(idx))
            swatch_layout.addWidget(btn)
            self.swatches.append(btn)
            
        group_swatch.setLayout(swatch_layout)
        right_layout.addWidget(group_swatch)
        
        # PARAMS PANEL
        group_params = QGroupBox("Physical Parameters")
        form_layout = QFormLayout()
        
        self.spin_dim = QDoubleSpinBox()
        self.spin_dim.setRange(50.0, 600.0)
        self.spin_dim.setValue(200.0)
        form_layout.addRow("Max Dim (mm):", self.spin_dim)
        
        self.spin_base = QDoubleSpinBox()
        self.spin_base.setRange(0.5, 10.0)
        self.spin_base.setValue(1.0)
        self.spin_base.setSingleStep(0.1)
        form_layout.addRow("Base (mm):", self.spin_base)
        
        self.spin_maxh = QDoubleSpinBox()
        self.spin_maxh.setRange(1.0, 20.0)
        self.spin_maxh.setValue(2.5)
        self.spin_maxh.setSingleStep(0.1)
        form_layout.addRow("Max Z (mm):", self.spin_maxh)
        
        self.spin_res = QSpinBox()
        self.spin_res.setRange(200, 4000)
        self.spin_res.setValue(800)
        self.spin_res.setSingleStep(100)
        form_layout.addRow("Mesh Res. (px):", self.spin_res)
        
        group_params.setLayout(form_layout)
        right_layout.addWidget(group_params)
        
        right_layout.addStretch()
        
        # BOTTOM CONTROLS
        self.btn_generate = QPushButton("🚀 Generate STL")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self.generate_stl)
        right_layout.addWidget(self.btn_generate)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Standby.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setFixedHeight(40) # Altezza fissa per prevenire sbalzi
        right_layout.addWidget(self.lbl_status)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([850, 350])

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Images (*.png *.jpg *.jpeg *.jfif *.avif *.webp *.bmp *.tiff *.heic);;All Files (*.*)"
        )
        if not file_path:
            return
            
        self.loaded_image_path = file_path
        self.lbl_status.setText("🛠 Decoding intermediate file...")
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

        # --- Compute color-change Z heights from the sampled threshold values ---
        # Sort the 4 sampled grey values descending (brightest first)
        s0, s1, s2, s3 = sorted(self.sampled_colors, reverse=True)
        base_h  = self.spin_base.value()
        max_h   = self.spin_maxh.value()
        relief  = max_h - base_h
        # The 3 terrain steps in normalised space are 0.33 / 0.66 / 1.0
        # → map back to absolute Z heights
        color_changes_z = [
            round(base_h + 0.33 * relief, 3),   # L1 Light Gray level
            round(base_h + 0.66 * relief, 3),   # L2 Dark Gray level
            round(base_h + 1.00 * relief, 3),   # L3 Black/Inks peak
        ]

        # --- Lock UI ---
        self.btn_load.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.progress_bar.setValue(0)
        for btn in self.swatches:
            btn.setEnabled(False)

        # --- Launch background QThread ---
        self.worker = MeshWorker(
            img_filtered=self.img_filtered_array,
            sampled_values=self.sampled_colors,
            max_dim=self.spin_dim.value(),
            max_h=max_h,
            base_h=base_h,
            output_path=save_path_stl,
            output_path_3mf=save_path_3mf,
            color_changes_z=color_changes_z,
            max_res=self.spin_res.value()
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
        self.lbl_status.setText("🏁 STL + 3MF Export Completed!")
        QMessageBox.information(
            self, "Export Successful",
            f"Both files have been saved to:\n"
            f"📄 STL → {stl_path}\n"
            f"🎨 3MF → {path_3mf}\n\n"
            f"The .3mf file includes automatic color-change pauses\n"
            f"for halftone levels — ready to slice in Bambu Studio!"
        )

    def on_generate_error(self, err_msg):
        self.unlock_ui()
        self.lbl_status.setText("❌ Critical QThread error.")
        QMessageBox.critical(self, "Trimesh Error", f"An error occurred during the 3D generation:\n{err_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = Manga3DApp()
    window.show()
    sys.exit(app.exec())

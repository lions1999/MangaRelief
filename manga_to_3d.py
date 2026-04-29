import sys
import os
import io
import time
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import trimesh
from PIL import Image
import gc
import fast_simplification
import pillow_heif

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QSplitter, QProgressBar, QDoubleSpinBox, QSpinBox,
                             QMessageBox, QGroupBox, QFormLayout, QCheckBox, QSlider, QComboBox, QSizePolicy)
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
    padding: 4px 10px;
    font-weight: bold;
    font-size: 13px;
    min-height: 24px;
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
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
    font-size: 13px;
    min-height: 24px;
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
# .3MF EXPORT  —  Hybrid: Trimesh geometry + Bambu Studio metadata injection
# ---------------------------------------------------------------------------
# Lessons learned from reference project reverse-engineering:
#
#  ✅ Geometry: Trimesh's own 3MF writer produces correct, loadable geometry.
#     We must NOT replace it with a hand-rolled serialiser.
#
#  ✅ Color changes: they live in Metadata/custom_gcode_per_layer.xml as:
#       <layer top_z="Z" type="2" extruder="1" color="#000000"
#              extra="" gcode="tool_change"/>
#     NOT in model_settings or Bambu_model_settings.
#
#  ✅ [Content_Types].xml: Bambu uses only <Default> entries (no <Override>).
#     We just need to add the .config and .xml MIME types.
#
#  Strategy: let Trimesh write the 3MF, rebuild the ZIP copying every entry
#  verbatim while patching [Content_Types].xml, then inject our two extra
#  Bambu metadata files.
# ---------------------------------------------------------------------------

_SLICE_INFO = """\
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="02.06.00.51"/>
  </header>
</config>"""

_CUSTOM_GCODE_TPL = """\
<?xml version="1.0" encoding="utf-8"?>
<custom_gcodes_per_layer>
<plate>
<plate_info id="1"/>
{layer_nodes}<mode value="MultiAsSingle"/>
</plate>
</custom_gcodes_per_layer>"""

# Extra MIME type declarations to append to [Content_Types].xml
_CT_EXTRA = (
    ' <Default Extension="config" ContentType="application/xml"/>\n'
    ' <Default Extension="xml"    ContentType="application/xml"/>\n'
)


def export_3mf(mesh, output_path_3mf, color_changes_z):
    """
    Exports a Trimesh mesh as a Bambu-Studio-compatible .3mf project file.

    Strategy:
      1. Trimesh generates a geometrically valid .3mf in memory.
      2. We iterate over every ZIP entry, copying each verbatim — EXCEPT
         [Content_Types].xml, which we patch to declare the Bambu config MIME
         types before </Types>.
      3. We inject two Bambu-specific metadata files:
           • Metadata/custom_gcode_per_layer.xml  — color-change tool_change commands
           • Metadata/slice_info.config           — Bambu client version signature
      4. The result is written to disk.

    Args:
        mesh (trimesh.Trimesh): Fully-built watertight mesh.
        output_path_3mf (str): Destination path, e.g. 'output/panel_3D.3mf'.
        color_changes_z (list[float]): Z heights (mm) for color-change pauses.
    """
    # 1. Trimesh → valid 3MF geometry (proven to load correctly in Bambu)
    src_buf = io.BytesIO()
    mesh.export(src_buf, file_type='3mf')
    src_buf.seek(0)

    # 2. Build custom_gcode_per_layer.xml layer nodes
    # Each change must point to a DIFFERENT extruder slot (2, 3, 4...).
    # Slot 1 is the starting filament (white base); slots 2+ are the relief colors.
    # Color hex values are descriptive labels Bambu shows in the UI per slot.
    slot_colors = ["#C8C8C8", "#646464", "#000000", "#1a1a1a"]  # Light→Dark per slot
    layer_nodes = ""
    for i, z in enumerate(sorted(color_changes_z)):
        extruder = i + 2          # slot 2, 3, 4 — never 1 (that is the starting slot)
        color    = slot_colors[i] if i < len(slot_colors) else "#000000"
        layer_nodes += (
            f'<layer top_z="{round(z, 4)}" type="2" extruder="{extruder}" '
            f'color="{color}" extra="" gcode="tool_change"/>\n'
        )
    custom_gcode = _CUSTOM_GCODE_TPL.format(layer_nodes=layer_nodes)

    # 3. Rebuild ZIP: copy Trimesh entries, patch [Content_Types].xml, inject metadata
    dst_buf = io.BytesIO()
    with zipfile.ZipFile(src_buf, 'r') as src_zip, \
         zipfile.ZipFile(dst_buf, 'w', zipfile.ZIP_DEFLATED) as dst_zip:

        for item in src_zip.infolist():
            data = src_zip.read(item.filename)

            if item.filename == '[Content_Types].xml':
                ct_text = data.decode('utf-8')
                # Inject Bambu MIME types if not already present
                if 'Extension="config"' not in ct_text:
                    ct_text = ct_text.replace('</Types>', _CT_EXTRA + '</Types>')
                data = ct_text.encode('utf-8')

            dst_zip.writestr(item, data)

        # Inject Bambu metadata
        dst_zip.writestr('Metadata/custom_gcode_per_layer.xml',
                         custom_gcode.encode('utf-8'))
        dst_zip.writestr('Metadata/slice_info.config',
                         _SLICE_INFO.encode('utf-8'))

    # 4. Write to disk
    dst_buf.seek(0)
    with open(output_path_3mf, 'wb') as f:
        f.write(dst_buf.read())


# ---------------------------------------------------------------------------
# BACKGROUND MESH WORKER
# ---------------------------------------------------------------------------
class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    def __init__(self, img_filtered, sampled_values, max_dim, max_h, base_h,
                 output_path, output_path_3mf, color_changes_z, layer_height, max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15):
        super().__init__()
        self.img_filtered = img_filtered
        self.sampled_values = sampled_values
        self.max_dim = max_dim
        self.max_h = max_h
        self.base_h = base_h
        self.output_path = output_path
        self.output_path_3mf = output_path_3mf
        self.color_changes_z = color_changes_z
        self.layer_height = layer_height
        self.max_res_cap = max_res_cap
        self.smart_decimate = smart_decimate
        self.white_clip = white_clip
        self.black_clip = black_clip

    def run(self):
        try:
            self.progress.emit(5, "Optimizing resolution for 3D mesh...")
            img = self.img_filtered
            
            # Ridimensioniamo la foto basandoci sulla Max Dim per garantire una fidelity di ~0.05mm/pixel
            h, w = img.shape
            target_res = int(self.max_dim / 0.05)
            
            # Applichiamo il cap selezionato dall'utente per limitare i poligoni ed evitare tempi di slicing infiniti
            target_res = min(target_res, self.max_res_cap)
            
            if max(w, h) != target_res:
                scale_res = target_res / max(w, h)
                new_w, new_h = int(w * scale_res), int(h * scale_res)
                # Uso un'interpolazione di alta qualità per non distruggere i dettagli negli halftone
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            img_filtered = img
            
            self.progress.emit(20, "Applying Smart Pixel Snapping (Clipping)...")
            # We need to ensure array is writable; slice assignment does it in-place
            img_filtered = img_filtered.copy()
            img_filtered[img_filtered >= self.white_clip] = 255
            img_filtered[img_filtered <= self.black_clip] = 0
            img_filtered[(img_filtered >= 165) & (img_filtered <= 175)] = 170
            img_filtered[(img_filtered >= 80) & (img_filtered <= 90)] = 85
                
            self.progress.emit(25, "Generazione Vertici...")
            
            relief_height = self.max_h - self.base_h
            # Map 0-255 linearly to Z: 0 (black) is max_h, 255 (white) is base_h
            Z = self.base_h + ((255.0 - img_filtered) / 255.0) * relief_height
            
            self.progress.emit(40, "Generazione Vertici (MeshGrid)...")
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
            
            self.progress.emit(55, "Creazione Facce...")
            vertices_top = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            idx = np.arange(w * h).reshape((h, w))
            tl = idx[:-1, :-1].flatten()
            tr = idx[:-1, 1:].flatten()
            bl = idx[1:, :-1].flatten()
            br = idx[1:, 1:].flatten()
            faces_top = np.vstack((np.column_stack((bl, tr, tl)), np.column_stack((br, tr, bl))))
            
            self.progress.emit(70, "Creazione Facce (Fondo)...")
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
            
            self.progress.emit(90, "Trimesh Repair...")
            all_vertices = np.vstack((vertices_top, vertices_bottom))
            all_faces = np.vstack((faces_top, faces_bottom, top_sides, bot_sides, left_sides, right_sides))
            
            mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
            trimesh.repair.fix_normals(mesh) # Necessario per renderlo watertight
            
            # Smart Optimization: decimate if face count exceeds threshold
            if self.smart_decimate and len(mesh.faces) > 1_500_000:
                target_faces = 900_000
                self.progress.emit(92, f"Optimizing mesh geometry (Decimating {len(mesh.faces):,} → {target_faces:,} faces)...")
                verts_out, faces_out = fast_simplification.simplify(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                    target_count=target_faces,
                    agg=5.0
                )
                mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
                trimesh.repair.fix_normals(mesh)
            elif self.smart_decimate and len(mesh.faces) > 500_000:
                target_faces = int(len(mesh.faces) * 0.6)
                self.progress.emit(92, f"Optimizing mesh geometry (Decimating {len(mesh.faces):,} → {target_faces:,} faces)...")
                verts_out, faces_out = fast_simplification.simplify(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                    target_count=target_faces,
                    agg=5.0
                )
                mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
                trimesh.repair.fix_normals(mesh)
            
            self.progress.emit(96, "Esportazione STL...")
            mesh.export(self.output_path)

            self.progress.emit(98, "Esportazione 3MF...")
            export_3mf(mesh, self.output_path_3mf, self.color_changes_z)

            self.progress.emit(100, "Esportazione completata!")
            self.finished_ok.emit(self.output_path, self.output_path_3mf)
            
            # Pulizia aggressiva della memoria
            del img
            del img_filtered
            del Z
            del X
            del Y
            del vertices_top
            del vertices_bottom
            del all_vertices
            del faces_top
            del faces_bottom
            del all_faces
            del mesh
            gc.collect()
            
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
        self.last_opened_dir = ""          # UX: remember last browsed folder
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
        right_panel.setFixedWidth(400) # Ingrandita verso sinistra per far spazio ai controlli
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
        self.spin_dim.setFixedHeight(30)
        form_layout.addRow("Max Dim (mm):", self.spin_dim)

        self.spin_base = QDoubleSpinBox()
        self.spin_base.setRange(0.5, 10.0)
        self.spin_base.setValue(1.0)
        self.spin_base.setSingleStep(0.1)
        self.spin_base.setFixedHeight(30)
        form_layout.addRow("Base (mm):", self.spin_base)

        self.spin_maxh = QDoubleSpinBox()
        self.spin_maxh.setRange(1.0, 20.0)
        self.spin_maxh.setValue(2.5)
        self.spin_maxh.setSingleStep(0.1)
        self.spin_maxh.setFixedHeight(30)
        form_layout.addRow("Max Z (mm):", self.spin_maxh)

        self.spin_layer_height = QDoubleSpinBox()
        self.spin_layer_height.setRange(0.01, 1.0)
        self.spin_layer_height.setValue(0.20)
        self.spin_layer_height.setSingleStep(0.01)
        self.spin_layer_height.setFixedHeight(30)
        form_layout.addRow("Printing Layer Height (mm):", self.spin_layer_height)

        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["Draft (800px)", "Standard (1200px)", "Ultra (1600px)"])
        self.cmb_quality.setCurrentIndex(1) # Default to Standard
        self.cmb_quality.setFixedHeight(30)
        form_layout.addRow("Mesh Quality:", self.cmb_quality)

        self.chk_smart_decimate = QCheckBox("Smart Optimization (Decimate)")
        self.chk_smart_decimate.setChecked(True)
        form_layout.addRow(self.chk_smart_decimate)

        self.spin_white_clip = QSpinBox()
        self.spin_white_clip.setRange(128, 255)
        self.spin_white_clip.setValue(235)
        self.spin_white_clip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spin_white_clip.setToolTip("Pixels lighter than this value become perfectly flat white background.")
        
        self.btn_auto_white = QPushButton("\ud83e\ude84 Auto: --")
        self.btn_auto_white.setFixedWidth(85)
        self.btn_auto_white.setFixedHeight(30) # Unify height with spinbox
        self.btn_auto_white.clicked.connect(self._apply_auto_white)
        self.btn_auto_white.setEnabled(False)

        self.spin_white_clip.setFixedHeight(30) # Unify height with button

        self.white_clip_container = QWidget()
        white_clip_layout = QHBoxLayout(self.white_clip_container)
        white_clip_layout.setSpacing(5)
        white_clip_layout.setContentsMargins(0, 0, 0, 0)
        white_clip_layout.addWidget(self.spin_white_clip)
        white_clip_layout.addWidget(self.btn_auto_white)
        form_layout.addRow("White Clip:", self.white_clip_container)

        self.spin_black_clip = QSpinBox()
        self.spin_black_clip.setRange(0, 127)
        self.spin_black_clip.setValue(15)
        self.spin_black_clip.setFixedHeight(30)
        self.spin_black_clip.setToolTip("Pixels darker than this value become perfectly flat max height.")
        form_layout.addRow("Black Clip:", self.spin_black_clip)

        group_params.setLayout(form_layout)
        right_layout.addWidget(group_params)

        # HALFTONE Z PANEL
        group_z = QGroupBox("Halftone Color-Change Z (mm)")
        z_layout = QFormLayout()

        self.chk_auto_z = QCheckBox("Auto-Calculate Halftone Z")
        self.chk_auto_z.setChecked(True)
        self.chk_auto_z.toggled.connect(self._on_auto_z_toggled)
        z_layout.addRow(self.chk_auto_z)

        # Halftone Threshold Slider
        self.lbl_threshold = QLabel("Halftone Threshold: 10%")
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(1, 50)
        self.slider_threshold.setValue(10)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        z_layout.addRow(self.lbl_threshold, self.slider_threshold)

        self.lbl_real_midtones = QLabel("Image Halftones: N/A")
        self.lbl_real_midtones.setStyleSheet("color: #aaaaaa; font-style: italic;")
        z_layout.addRow("", self.lbl_real_midtones)

        self.lbl_z1 = QLabel("L1 Z (Light Gray):")
        self.lbl_z2 = QLabel("L2 Z (Dark Gray):")
        self.lbl_z3 = QLabel("L3 Z (Black/Inks):")
        self.spin_z1 = QDoubleSpinBox(); self.spin_z1.setRange(0.1, 50.0); self.spin_z1.setSingleStep(0.1); self.spin_z1.setFixedHeight(30)
        self.spin_z2 = QDoubleSpinBox(); self.spin_z2.setRange(0.1, 50.0); self.spin_z2.setSingleStep(0.1); self.spin_z2.setFixedHeight(30)
        self.spin_z3 = QDoubleSpinBox(); self.spin_z3.setRange(0.1, 50.0); self.spin_z3.setSingleStep(0.1); self.spin_z3.setFixedHeight(30)
        z_layout.addRow(self.lbl_z1, self.spin_z1)
        z_layout.addRow(self.lbl_z2, self.spin_z2)
        z_layout.addRow(self.lbl_z3, self.spin_z3)

        self.lbl_color_mode = QLabel("")
        self.lbl_color_mode.setWordWrap(True)
        z_layout.addRow(self.lbl_color_mode)

        group_z.setLayout(z_layout)
        right_layout.addWidget(group_z)

        # Track halftone mode (2, 3, or 4 colors)
        self.color_mode_state = 4
        self.last_midtone_pct = 100.0 # Default value until image loads

        # Initialise the spinboxes to default computed values and set read-only
        self._refresh_auto_z_display()
        self._on_auto_z_toggled(True)

        # Live-refresh Z display whenever physical parameters change
        self.spin_base.valueChanged.connect(self._on_physical_param_changed)
        self.spin_maxh.valueChanged.connect(self._on_physical_param_changed)
        self.spin_layer_height.valueChanged.connect(self._on_physical_param_changed)
        
        right_layout.addStretch()

        # BOTTOM CONTROLS
        self.btn_generate = QPushButton("🚀 Generate STL + 3MF")
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
        splitter.setSizes([800, 400])
        
        # Configura il divisore dopo aver aggiunto i widget (altrimenti l'handle non esiste)
        splitter.setHandleWidth(1)
        splitter.handle(1).setCursor(Qt.CursorShape.ArrowCursor)
        splitter.handle(1).setEnabled(False) # Disabilita completamente il trascinamento

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

        # Update visibility
        self.lbl_z1.setVisible(self.color_mode_state == 4)
        self.spin_z1.setVisible(self.color_mode_state == 4)
        
        self.lbl_z2.setVisible(self.color_mode_state >= 3)
        self.spin_z2.setVisible(self.color_mode_state >= 3)

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
            black_clip=self.spin_black_clip.value()
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
    app.setStyleSheet(DARK_STYLESHEET)
    window = Manga3DApp()
    window.show()
    sys.exit(app.exec())

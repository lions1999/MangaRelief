import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QSplitter, QProgressBar, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QFormLayout, QCheckBox, QSlider, QComboBox, 
                             QSizePolicy, QScrollArea, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor

from utils import resource_path

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
        if img_filtered_array is None:
            return
            
        if len(img_filtered_array.shape) == 3:
            h, w, c = img_filtered_array.shape
            fmt = QImage.Format.Format_RGB888
            # OpenCV usa BGR, PyQt vuole RGB. Se l'array viene da OpenCV, convertiamolo.
            # Ma nel nostro caso lo carichiamo già come RGB o lo convertiamo prima di passarlo.
            bytes_per_line = c * w
        else:
            h, w = img_filtered_array.shape
            fmt = QImage.Format.Format_Grayscale8
            bytes_per_line = w

        if not img_filtered_array.flags['C_CONTIGUOUS']:
            img_filtered_array = np.ascontiguousarray(img_filtered_array)
            
        qimage = QImage(img_filtered_array.data, w, h, bytes_per_line, fmt)
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

class MainWindowUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MangaRelief Pro")
        self.setWindowIcon(QIcon(resource_path('icon.ico')))
        self.resize(1240, 720)
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- LEFT PANEL: VIEWPORT ---
        self.viewer = ImageGraphicsView()
        splitter.addWidget(self.viewer)
        
        # --- RIGHT PANEL: CONTROLLI (Wrapped in ScrollArea) ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(450)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        splitter.addWidget(self.scroll_area)

        right_panel = QWidget()
        self.scroll_area.setWidget(right_panel)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 5, 25, 5)
        
        self.btn_load = QPushButton("📂 Load Manga")
        right_layout.addWidget(self.btn_load)
        
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Standard Manga Relief", "Topographic Color (Single Extruder)"])
        self.mode_selector.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        right_layout.addWidget(self.mode_selector)
        
        self.lbl_info = QLabel("No project opened.")
        self.lbl_info.setStyleSheet("color: #a6adc8; margin-bottom: 10px;")
        right_layout.addWidget(self.lbl_info)

        # TOPO COLOR PANEL (Hidden by default)
        self.group_topo = QGroupBox("Topographic Color Settings")
        topo_layout = QVBoxLayout()
        self.btn_extract_topo = QPushButton("🎨 1. Extract Colors (K-Means)")
        topo_layout.addWidget(self.btn_extract_topo)
        self.topo_color_list = QListWidget()
        self.topo_color_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.topo_color_list.setFixedHeight(130)
        self.topo_color_list.setToolTip("Drag to reorder: Top = Bottom Layer, Bottom = Top Layer")
        topo_layout.addWidget(self.topo_color_list)
        self.group_topo.setLayout(topo_layout)
        self.group_topo.setVisible(False)
        right_layout.addWidget(self.group_topo)
        
        # SWATCH PANEL
        group_swatch = QGroupBox("Color Picking (Click to calibrate)")
        swatch_layout = QVBoxLayout()
        
        self.chk_auto_midtones = QCheckBox("Auto-Detect Midtones (K-Means)")
        self.chk_auto_midtones.setChecked(True)
        swatch_layout.addWidget(self.chk_auto_midtones)
        
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
        self.spin_maxh.setValue(2.40)
        self.spin_maxh.setSingleStep(0.1)
        form_layout.addRow("Max Z (mm):", self.spin_maxh)

        self.spin_layer_height = QDoubleSpinBox()
        self.spin_layer_height.setRange(0.01, 1.0)
        self.spin_layer_height.setValue(0.20)
        self.spin_layer_height.setSingleStep(0.01)
        form_layout.addRow("Printing Layer Height (mm):", self.spin_layer_height)

        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["Draft (800px)", "Standard (1200px)", "Ultra (1600px)"])
        self.cmb_quality.setCurrentIndex(1)
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
        self.btn_auto_white.setEnabled(False)

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
        self.spin_black_clip.setToolTip("Pixels darker than this value become perfectly flat max height.")
        form_layout.addRow("Black Clip:", self.spin_black_clip)

        group_params.setLayout(form_layout)
        right_layout.addWidget(group_params)

        # HALFTONE Z PANEL
        group_z = QGroupBox("Halftone Color-Change Z (mm)")
        z_layout = QFormLayout()

        self.chk_auto_z = QCheckBox("Auto-Calculate Halftone Z")
        self.chk_auto_z.setChecked(True)
        z_layout.addRow(self.chk_auto_z)

        self.lbl_threshold = QLabel("Halftone Threshold: 10%")
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(1, 50)
        self.slider_threshold.setValue(10)
        z_layout.addRow(self.lbl_threshold, self.slider_threshold)

        self.lbl_real_midtones = QLabel("Image Halftones: N/A")
        self.lbl_real_midtones.setStyleSheet("color: #aaaaaa; font-style: italic;")
        z_layout.addRow("", self.lbl_real_midtones)

        self.lbl_z1 = QLabel("L1 Z (Light Gray):")
        self.lbl_z2 = QLabel("L2 Z (Dark Gray):")
        self.lbl_z3 = QLabel("L3 Z (Black/Inks):")
        self.spin_z1 = QDoubleSpinBox(); self.spin_z1.setRange(0.1, 50.0); self.spin_z1.setSingleStep(0.1)
        self.spin_z2 = QDoubleSpinBox(); self.spin_z2.setRange(0.1, 50.0); self.spin_z2.setSingleStep(0.1)
        self.spin_z3 = QDoubleSpinBox(); self.spin_z3.setRange(0.1, 50.0); self.spin_z3.setSingleStep(0.1)
        z_layout.addRow(self.lbl_z1, self.spin_z1)
        z_layout.addRow(self.lbl_z2, self.spin_z2)
        z_layout.addRow(self.lbl_z3, self.spin_z3)

        self.lbl_color_mode = QLabel("")
        self.lbl_color_mode.setWordWrap(True)
        z_layout.addRow(self.lbl_color_mode)

        group_z.setLayout(z_layout)
        right_layout.addWidget(group_z)

        right_layout.addStretch()

        # BOTTOM CONTROLS
        right_layout.addSpacing(10)
        deckbox_layout = QHBoxLayout()
        deckbox_layout.setContentsMargins(0, 5, 0, 5)
        self.chk_deckbox_mode = QCheckBox("Generate Deckbox")
        self.chk_deckbox_mode.setChecked(False)
        self.chk_deckbox_mode.setStyleSheet("font-weight: bold; color: #a6e3a1;")
        deckbox_layout.addWidget(self.chk_deckbox_mode)
        
        self.combo_tcg_select = QComboBox()
        self.combo_tcg_select.addItems(["Yu-Gi-Oh!", "Pokémon", "Magic", "One Piece"])
        self.combo_tcg_select.setMinimumWidth(160)
        self.combo_tcg_select.setStyleSheet("font-weight: bold; padding: 4px 8px; min-height: 22px;")
        deckbox_layout.addWidget(self.combo_tcg_select)
        right_layout.addLayout(deckbox_layout)

        export_layout = QHBoxLayout()
        self.chk_export_3mf = QCheckBox("Export .3MF")
        self.chk_export_3mf.setChecked(True)
        self.chk_export_stl = QCheckBox("Export .STL")
        self.chk_export_stl.setChecked(False)
        export_layout.addWidget(self.chk_export_3mf)
        export_layout.addWidget(self.chk_export_stl)
        right_layout.addLayout(export_layout)

        self.btn_generate = QPushButton("🚀 Generate 3D Models")
        self.btn_generate.setFixedHeight(50)
        self.btn_generate.setEnabled(False)
        right_layout.addWidget(self.btn_generate)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Standby.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setFixedHeight(40)
        right_layout.addWidget(self.lbl_status)
        
        # Connect mode selector to visibility toggle
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        
        splitter.addWidget(self.scroll_area)
        splitter.setSizes([800, 420])
        
        splitter.setHandleWidth(1)
        splitter.handle(1).setCursor(Qt.CursorShape.ArrowCursor)
        splitter.handle(1).setEnabled(False)
        
    def _on_mode_changed(self, index):
        """Toggle visibility of specific panels based on the selected mode."""
        is_topo = (index == 1)
        self.group_topo.setVisible(is_topo)
        
        # Hide standard relief controls if topo is active
        for child in self.findChildren(QGroupBox):
            if child.title() in ["Color Picking (Click to calibrate)", "Halftone Color-Change Z (mm)"]:
                child.setVisible(not is_topo)

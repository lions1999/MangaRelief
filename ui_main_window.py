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
        self.mode_selector.setObjectName("mode_selector")
        self.mode_selector.addItems(["Standard Manga Relief", "Topographic Color (Single Extruder)", "Deckbox Engraving", "Spot Color (Silkscreen)", "Phone Cover Plate"])
        right_layout.addWidget(self.mode_selector)
        
        self.group_deckbox = QGroupBox("Deckbox Settings")
        deckbox_layout = QVBoxLayout()
        self.combo_tcg_select = QComboBox()
        self.combo_tcg_select.setObjectName("combo_tcg_select")
        self.combo_tcg_select.addItems(["Yu-Gi-Oh!", "Pokémon", "Magic", "One Piece", "Hunter x Hunter"])
        deckbox_layout.addWidget(QLabel("Select TCG Game:"))
        deckbox_layout.addWidget(self.combo_tcg_select)
        self.group_deckbox.setLayout(deckbox_layout)
        self.group_deckbox.setVisible(False)
        right_layout.addWidget(self.group_deckbox)
        
        self.lbl_info = QLabel("No project opened.")
        self.lbl_info.setObjectName("lbl_info")
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
        
        # PHONE COVER PANEL (Hidden by default)
        self.group_cover = QGroupBox("Phone Cover Settings")
        cover_layout = QFormLayout()

        self.combo_phone_model = QComboBox()
        cover_layout.addRow("Phone Model:", self.combo_phone_model)

        self.combo_cover_finish = QComboBox()
        self.combo_cover_finish.addItems(["B/N (Standard)", "Spot Color"])
        cover_layout.addRow("Finish:", self.combo_cover_finish)

        self.lbl_cover_scale = QLabel("Zoom: 100%")
        self.slider_cover_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_cover_scale.setRange(100, 300)
        self.slider_cover_scale.setValue(100)
        cover_layout.addRow(self.lbl_cover_scale, self.slider_cover_scale)

        self.lbl_cover_offx = QLabel("Offset X: 0 mm")
        self.slider_cover_offx = QSlider(Qt.Orientation.Horizontal)
        self.slider_cover_offx.setRange(-60, 60)
        self.slider_cover_offx.setValue(0)
        cover_layout.addRow(self.lbl_cover_offx, self.slider_cover_offx)

        self.lbl_cover_offy = QLabel("Offset Y: 0 mm")
        self.slider_cover_offy = QSlider(Qt.Orientation.Horizontal)
        self.slider_cover_offy.setRange(-80, 80)
        self.slider_cover_offy.setValue(0)
        cover_layout.addRow(self.lbl_cover_offy, self.slider_cover_offy)

        self.chk_cover_avoid_camera = QCheckBox("Art below camera block (keep lens area clean)")
        self.chk_cover_avoid_camera.setChecked(True)
        cover_layout.addRow(self.chk_cover_avoid_camera)

        self.chk_cover_bumper = QCheckBox("Generate TPU bumper too (separate STL)")
        self.chk_cover_bumper.setChecked(True)
        self.chk_cover_bumper.setToolTip("Deseleziona se hai già stampato il bumper: verrà generata solo la plate.")
        cover_layout.addRow(self.chk_cover_bumper)

        self.btn_cover_preview = QPushButton("👁 Plate Preview")
        self.btn_cover_preview.setCheckable(True)
        self.btn_cover_preview.setEnabled(False)
        cover_layout.addRow(self.btn_cover_preview)

        self.group_cover.setLayout(cover_layout)
        self.group_cover.setVisible(False)
        right_layout.addWidget(self.group_cover)

        # SPOT COLOR PANEL (Hidden by default)
        self.group_spot = QGroupBox("Spot Color Settings")
        spot_layout = QVBoxLayout()

        self.combo_spot_naccents = QComboBox()
        self.combo_spot_naccents.addItems(["1 Accent Color", "2 Accent Colors"])
        spot_layout.addWidget(self.combo_spot_naccents)

        self.btn_spot_auto = QPushButton("🤖 Auto-Detect Accents")
        spot_layout.addWidget(self.btn_spot_auto)

        self.lbl_spot_info = QLabel("Click an accent below, then click on the image to sample its color.")
        self.lbl_spot_info.setWordWrap(True)
        spot_layout.addWidget(self.lbl_spot_info)

        self.spot_swatches = []
        for i in range(2):
            btn = QPushButton(f"Accent {i+1}: [ -- ]")
            btn.setProperty("class", "swatch")
            spot_layout.addWidget(btn)
            self.spot_swatches.append(btn)

        self.lbl_spot_coverage = QLabel("Accent Coverage: 40%")
        self.slider_spot_coverage = QSlider(Qt.Orientation.Horizontal)
        self.slider_spot_coverage.setRange(0, 100)
        self.slider_spot_coverage.setValue(40)
        self.slider_spot_coverage.setToolTip("Low = only vivid pixels become accent. High = muted shades too.")
        spot_layout.addWidget(self.lbl_spot_coverage)
        spot_layout.addWidget(self.slider_spot_coverage)

        self.btn_spot_mockup = QPushButton("👁 Mockup Preview")
        self.btn_spot_mockup.setCheckable(True)
        self.btn_spot_mockup.setEnabled(False)
        spot_layout.addWidget(self.btn_spot_mockup)

        self.group_spot.setLayout(spot_layout)
        self.group_spot.setVisible(False)
        right_layout.addWidget(self.group_spot)

        # Il secondo swatch accento compare solo scegliendo "2 Accent Colors"
        self.spot_swatches[1].setVisible(False)
        self.combo_spot_naccents.currentIndexChanged.connect(
            lambda i: self.spot_swatches[1].setVisible(i == 1))

        # SWATCH PANEL
        self.group_swatch = QGroupBox("Color Picking (Click to calibrate)")
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
            
        self.group_swatch.setLayout(swatch_layout)
        right_layout.addWidget(self.group_swatch)
        
        # PARAMS PANEL
        group_params = QGroupBox("Physical Parameters")
        form_layout = QFormLayout()

        self.spin_dim = QDoubleSpinBox()
        self.spin_dim.setRange(50.0, 600.0)
        self.spin_dim.setValue(200.0)
        form_layout.addRow("Max Dim (mm):", self.spin_dim)

        self.spin_base = QDoubleSpinBox()
        self.spin_base.setRange(0.2, 10.0)  # min 0.2: le plate cover sono slim
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
        self.group_z = QGroupBox("Halftone Color-Change Z (mm)")
        z_layout = QFormLayout()

        self.chk_auto_z = QCheckBox("Auto-Calculate Halftone Z")
        self.chk_auto_z.setChecked(True)
        z_layout.addRow(self.chk_auto_z)

        self.lbl_threshold = QLabel("Halftone Threshold: 10%")
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(1, 100)
        self.slider_threshold.setValue(10)
        z_layout.addRow(self.lbl_threshold, self.slider_threshold)

        self.lbl_real_midtones = QLabel("Image Halftones: N/A")
        self.lbl_real_midtones.setObjectName("lbl_real_midtones")
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

        self.group_z.setLayout(z_layout)
        right_layout.addWidget(self.group_z)

        right_layout.addStretch()

        # BOTTOM CONTROLS
        right_layout.addSpacing(10)

        export_layout = QHBoxLayout()
        self.chk_export_3mf = QCheckBox("Export .3MF")
        self.chk_export_3mf.setChecked(True)
        self.chk_export_stl = QCheckBox("Export .STL")
        self.chk_export_stl.setChecked(True)
        export_layout.addWidget(self.chk_export_3mf)
        export_layout.addWidget(self.chk_export_stl)
        right_layout.addLayout(export_layout)

        self.btn_generate = QPushButton("🚀 Generate 3D Models")
        self.btn_generate.setObjectName("btn_generate")
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
        self.combo_cover_finish.currentIndexChanged.connect(
            lambda _: self._on_mode_changed(self.mode_selector.currentIndex()))

        # Registro dei widget da bloccare durante la generazione: ogni nuovo
        # controllo va aggiunto QUI, non dentro toggle_ui_state
        self.lockable_widgets = [
            self.btn_load, self.mode_selector, self.combo_tcg_select,
            self.btn_extract_topo, self.topo_color_list,
            self.combo_phone_model, self.combo_cover_finish,
            self.slider_cover_scale, self.slider_cover_offx,
            self.slider_cover_offy, self.btn_cover_preview, self.chk_cover_bumper,
            self.chk_cover_avoid_camera,
            self.combo_spot_naccents, self.btn_spot_auto, *self.spot_swatches,
            self.slider_spot_coverage, self.btn_spot_mockup,
            self.chk_auto_midtones, *self.swatches,
            self.spin_dim, self.spin_base, self.spin_maxh, self.spin_layer_height,
            self.cmb_quality, self.chk_smart_decimate,
            self.spin_white_clip, self.btn_auto_white, self.spin_black_clip,
            self.chk_auto_z, self.slider_threshold,
            self.spin_z1, self.spin_z2, self.spin_z3,
            self.chk_export_3mf, self.chk_export_stl,
        ]
        
        splitter.addWidget(self.scroll_area)
        splitter.setSizes([800, 420])
        
        splitter.setHandleWidth(1)
        splitter.handle(1).setCursor(Qt.CursorShape.ArrowCursor)
        splitter.handle(1).setEnabled(False)
        
    def _on_mode_changed(self, index):
        """Toggle visibility of specific panels based on the selected mode."""
        is_topo    = (index == 1)
        is_deckbox = (index == 2)
        is_spot    = (index == 3)
        is_cover   = (index == 4)
        cover_spot = is_cover and self.combo_cover_finish.currentIndex() == 1

        self.group_topo.setVisible(is_topo)
        self.group_deckbox.setVisible(is_deckbox)
        self.group_cover.setVisible(is_cover)
        # il gruppo Spot serve anche alla finitura Spot della cover
        self.group_spot.setVisible(is_spot or cover_spot)

        # Hide standard relief controls when topo/spot/cover are active
        self.group_swatch.setVisible(not (is_topo or is_spot or is_cover))
        self.group_z.setVisible(not (is_topo or is_spot or is_cover))

        # Dynamically lock physical parameters for Deckbox mode
        if is_deckbox:
            self.spin_dim.setEnabled(False)
            self.spin_base.setValue(4.0)
            self.spin_base.setEnabled(False)
            self.spin_maxh.setValue(2.0)
            self.spin_maxh.setEnabled(False)
        else:
            self.spin_dim.setEnabled(True)
            self.spin_base.setEnabled(True)
            self.spin_maxh.setEnabled(True)

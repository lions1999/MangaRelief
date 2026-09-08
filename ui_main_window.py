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
        self.mode_selector.addItems(["Standard Manga Relief", "Topographic Color (Single Extruder)", "Deckbox Engraving", "Spot Color (Silkscreen)", "Phone Cover Plate", "Keychain / Cutout"])
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
        self.combo_cover_finish.addItems(["B&W (Standard)", "Spot Color"])
        cover_layout.addRow("Finish:", self.combo_cover_finish)

        self.combo_cover_surface = QComboBox()
        self.combo_cover_surface.addItems(["Engraved (smooth in the pocket)", "Raised (relief)"])
        self.combo_cover_surface.setToolTip("Engraved: flat outer surface, artwork carved underneath. Raised: artwork in relief, like the panels.")
        cover_layout.addRow("Surface:", self.combo_cover_surface)

        self.combo_cover_levels = QComboBox()
        self.combo_cover_levels.addItems(["2 (B/N puro)", "3 (consigliato)", "4 (max dettaglio)"])
        self.combo_cover_levels.setCurrentIndex(1)
        self.combo_cover_levels.setToolTip("Quantised grey levels: more levels = more halftones and screentones preserved.")
        cover_layout.addRow("Gray Levels:", self.combo_cover_levels)

        self.lbl_cover_scale = QLabel("Zoom: 100%")
        self.slider_cover_scale = QSlider(Qt.Orientation.Horizontal)
        # 100% = riempimento esatto della plate; sotto il 100% l'immagine
        # rimpicciolisce (bordi bianchi base) per posizionarla più facilmente
        self.slider_cover_scale.setRange(50, 300)
        self.slider_cover_scale.setValue(100)
        self.slider_cover_scale.setToolTip("100% = exact fill. <100% shrinks the image (white margins), >100% zooms in.")
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
        self.chk_cover_bumper.setToolTip("Uncheck if you have already printed the bumper: only the plate will be generated.")
        cover_layout.addRow(self.chk_cover_bumper)

        self.btn_cover_preview = QPushButton("👁 Plate Preview")
        self.btn_cover_preview.setCheckable(True)
        self.btn_cover_preview.setEnabled(False)
        cover_layout.addRow(self.btn_cover_preview)

        self.group_cover.setLayout(cover_layout)
        self.group_cover.setVisible(False)
        right_layout.addWidget(self.group_cover)

        # --- ordine dei riquadri ---
        # Il pannello di una modalita' sta SOPRA i pannelli che accende. La
        # cover e il portachiavi scelgono la finitura, e la finitura decide se
        # compaiono gli accenti Spot (sotto) o gli swatch e le quote Standard
        # (piu' sotto ancora): con il selettore sotto al riquadro che governa,
        # si sceglie una cosa guardandone un'altra che e' gia' passata.
        # Vale anche per il selettore di modalita', che sta in cima a tutto.

        # KEYCHAIN / CUTOUT PANEL (Hidden by default)
        # E' una modalita' come le altre, quindi il pannello si vede solo
        # quando la modalita' e' scelta e i suoi controlli sono sempre vivi:
        # non c'e' una spunta da accendere prima, perche' la voce del menu e'
        # gia' quella spunta.
        self.group_keychain = QGroupBox("Keychain / Cutout Settings")
        cut_layout = QVBoxLayout()

        # Stessa domanda della cover, stessa forma di risposta: il ritaglio
        # decide la SAGOMA, non i colori, e come colorare l'arte resta da
        # scegliere.
        self.combo_keychain_finish = QComboBox()
        self.combo_keychain_finish.addItems(["Spot Color", "B&W (Standard)"])
        self.combo_keychain_finish.setToolTip(
            "Spot Color: white base + accents + black.\n"
            "B&W: the Standard mode posterisation.")
        cut_form_top = QFormLayout()
        cut_form_top.addRow("Finish:", self.combo_keychain_finish)
        cut_layout.addLayout(cut_form_top)

        self.combo_cutout_src = QComboBox()
        self.combo_cutout_src.addItems(["Auto + click (regions)", "Painted mask (file)"])
        self.combo_cutout_src.setToolTip(
            "Auto: only what touches the image border is void; you correct the\n"
            "rest by clicking. Painted mask: a second image where you coloured\n"
            "the parts to print and left the voids white.")
        cut_layout.addWidget(self.combo_cutout_src)

        self.btn_cutout_paint = QPushButton("📂 Load painted mask…")
        self.btn_cutout_paint.setVisible(False)
        cut_layout.addWidget(self.btn_cutout_paint)

        # Il testo che spiega *perche'* servono i click: senza, la modalita'
        # sembra rotta ogni volta che l'automatismo tiene un vuoto racchiuso.
        self.lbl_cutout_info = QLabel(
            "Auto removes the outer background only. Voids enclosed by the "
            "drawing (between a cloud and the hat, inside a curl) stay solid: "
            "turn on Edit regions and click inside the ones to punch out.")
        self.lbl_cutout_info.setWordWrap(True)
        cut_layout.addWidget(self.lbl_cutout_info)

        edit_row = QHBoxLayout()
        self.btn_cutout_edit = QPushButton("✂️ Edit regions")
        self.btn_cutout_edit.setCheckable(True)
        self.btn_cutout_edit.setToolTip("Click inside a region to punch it out; click again to fill it back.")
        self.btn_cutout_reset = QPushButton("↺ Reset")
        self.btn_cutout_reset.setFixedWidth(70)
        edit_row.addWidget(self.btn_cutout_edit)
        edit_row.addWidget(self.btn_cutout_reset)
        cut_layout.addLayout(edit_row)

        cut_form = QFormLayout()

        self.lbl_cutout_border = QLabel("Sticker border: 0.0 mm")
        self.slider_cutout_border = QSlider(Qt.Orientation.Horizontal)
        self.slider_cutout_border.setRange(0, 30)   # decimi di mm
        self.slider_cutout_border.setValue(0)
        self.slider_cutout_border.setToolTip(
            "Grows the silhouette outwards: the white rim of a sticker.\n"
            "Also thickens thin strokes that would not print on their own.")
        cut_form.addRow(self.lbl_cutout_border, self.slider_cutout_border)

        # Il diametro va etichettato e gli va lasciato spazio: a 70 px il
        # valore finiva sotto le frecce, cioe' il campo non diceva ne' quanto
        # vale ne' cosa misura.
        ring_row = QHBoxLayout()
        self.chk_cutout_ring = QCheckBox("Keyring hole")
        self.btn_cutout_ring = QPushButton("📍 Place")
        self.btn_cutout_ring.setCheckable(True)
        self.btn_cutout_ring.setMinimumWidth(80)
        self.lbl_ring_d = QLabel("Ø mm:")
        self.spin_ring_d = QDoubleSpinBox()
        self.spin_ring_d.setRange(1.5, 12.0)
        self.spin_ring_d.setValue(4.0)
        self.spin_ring_d.setSingleStep(0.5)
        self.spin_ring_d.setMinimumWidth(80)
        self.spin_ring_d.setToolTip("Diameter of the keyring hole, in mm.")
        ring_row.addWidget(self.chk_cutout_ring)
        ring_row.addStretch()
        ring_row.addWidget(self.btn_cutout_ring)
        ring_row.addWidget(self.lbl_ring_d)
        ring_row.addWidget(self.spin_ring_d)
        cut_form.addRow(ring_row)

        cut_layout.addLayout(cut_form)

        self.btn_cutout_preview = QPushButton("👁 Cutout Preview")
        self.btn_cutout_preview.setCheckable(True)
        self.btn_cutout_preview.setEnabled(False)
        cut_layout.addWidget(self.btn_cutout_preview)

        self.group_keychain.setLayout(cut_layout)
        self.group_keychain.setVisible(False)
        right_layout.addWidget(self.group_keychain)

        self.cutout_widgets = [
            self.combo_keychain_finish, self.combo_cutout_src,
            self.btn_cutout_paint, self.btn_cutout_edit, self.btn_cutout_reset,
            self.slider_cutout_border, self.chk_cutout_ring,
            self.btn_cutout_ring, self.spin_ring_d, self.btn_cutout_preview,
        ]
        self.combo_cutout_src.currentIndexChanged.connect(
            lambda i: self.btn_cutout_paint.setVisible(i == 1))
        self.combo_keychain_finish.currentIndexChanged.connect(
            lambda _: self._on_mode_changed(self.mode_selector.currentIndex()))

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

        # --- 2 colori: non c'e' un tono da campionare, c'e' una soglia ---
        # A due colori L1 e L2 non esistono e l'immagine decide da sola cosa e'
        # inchiostro: gli swatch restano quattro pulsanti di cui meta' non fa
        # niente. Al loro posto compare l'unica domanda che resta, cioe' quanto
        # carica debba essere una zona sfumata perche' stampi nera.
        # Nascosti finche' la modalita' non diventa 2: li accende
        # _refresh_color_mode, che e' gia' il punto in cui il pannello cambia
        # forma con la modalita'.
        self.lbl_bw_coverage = QLabel("Shading darker than 35% prints as ink")
        self.lbl_bw_coverage.setWordWrap(True)
        self.lbl_bw_coverage.setVisible(False)
        swatch_layout.addWidget(self.lbl_bw_coverage)

        self.slider_bw_coverage = QSlider(Qt.Orientation.Horizontal)
        self.slider_bw_coverage.setRange(10, 90)
        self.slider_bw_coverage.setSingleStep(5)
        self.slider_bw_coverage.setPageStep(5)
        self.slider_bw_coverage.setValue(35)
        self.slider_bw_coverage.setVisible(False)
        swatch_layout.addWidget(self.slider_bw_coverage)

        self.lbl_bw_note = QLabel("")
        self.lbl_bw_note.setObjectName("lbl_bw_note")
        self.lbl_bw_note.setWordWrap(True)
        self.lbl_bw_note.setVisible(False)
        swatch_layout.addWidget(self.lbl_bw_note)

        # L'anteprima vale per tutte le sotto-modalita' Standard, non solo a
        # due colori: dice su quale bobina finisce ogni pixel, che e' la
        # domanda a cui a 3 e 4 colori servono gli swatch. Senza, la copertura
        # sarebbe un cursore cieco — muovi, generi, guardi, ripeti.
        self.btn_std_mockup = QPushButton("👁 Mockup Preview")
        self.btn_std_mockup.setCheckable(True)
        self.btn_std_mockup.setEnabled(False)
        swatch_layout.addWidget(self.btn_std_mockup)

        self.group_swatch.setLayout(swatch_layout)
        right_layout.addWidget(self.group_swatch)
        
        # PARAMS PANEL
        group_params = QGroupBox("Physical Parameters")
        form_layout = QFormLayout()

        self.spin_dim = QDoubleSpinBox()
        self.spin_dim.setRange(50.0, 600.0)
        self.spin_dim.setValue(200.0)
        form_layout.addRow("Max Dim (mm):", self.spin_dim)

        # Quanto viene il tratto piu' fine a QUESTA dimensione. Va qui e non
        # nella barra di stato perche' e' un numero che si consulta mentre si
        # gira la manopola sopra, e un messaggio di stato lo cancella il primo
        # altro evento.
        self.lbl_feature_scale = QLabel("")
        self.lbl_feature_scale.setObjectName("lbl_feature_scale")
        self.lbl_feature_scale.setWordWrap(True)
        self.lbl_feature_scale.setVisible(False)
        form_layout.addRow("", self.lbl_feature_scale)

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
            self.combo_phone_model, self.combo_cover_finish, self.combo_cover_surface,
            self.combo_cover_levels,
            self.slider_cover_scale, self.slider_cover_offx,
            self.slider_cover_offy, self.btn_cover_preview, self.chk_cover_bumper,
            self.chk_cover_avoid_camera,
            self.combo_spot_naccents, self.btn_spot_auto, *self.spot_swatches,
            self.slider_spot_coverage, self.btn_spot_mockup,
            *self.cutout_widgets,
            self.chk_auto_midtones, *self.swatches,
            self.slider_bw_coverage, self.btn_std_mockup,
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
        is_keychain = (index == 5)
        cover_spot = is_cover and self.combo_cover_finish.currentIndex() == 1
        keychain_spot = is_keychain and self.combo_keychain_finish.currentIndex() == 0

        self.group_topo.setVisible(is_topo)
        self.group_deckbox.setVisible(is_deckbox)
        self.group_cover.setVisible(is_cover)
        self.group_keychain.setVisible(is_keychain)
        # il gruppo Spot serve anche alle finiture Spot di cover e portachiavi
        self.group_spot.setVisible(is_spot or cover_spot or keychain_spot)
        # il selettore livelli grigio serve solo alla finitura B/N della cover
        self.combo_cover_levels.setVisible(is_cover and not cover_spot)

        # Gli swatch e le quote Standard servono a chi posterizza in grigio:
        # il portachiavi B/N passa esattamente da li', quindi in quel caso
        # restano — spariscono solo con la finitura Spot.
        std_controls = not (is_topo or is_spot or is_cover or keychain_spot)
        self.group_swatch.setVisible(std_controls)
        self.group_z.setVisible(std_controls)

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

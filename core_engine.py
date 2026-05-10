import os
import io
import zipfile
import cv2
import numpy as np
import trimesh
import fast_simplification

from PyQt6.QtCore import QThread, pyqtSignal

# ---------------------------------------------------------------------------
# .3MF EXPORT  —  Hybrid: Trimesh geometry + Bambu Studio metadata injection
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

_CT_EXTRA = """\
  <Default Extension="config" ContentType="text/xml"/>
  <Default Extension="xml" ContentType="text/xml"/>
"""

def export_3mf(mesh, output_path_3mf, color_changes_z):
    """
    Exports a 3MF using trimesh, then injects Bambu Studio specific XMLs 
    for color changing at specific Z heights.
    """
    # 1. Generate base 3MF with trimesh in memory
    src_buf = io.BytesIO()
    mesh.export(src_buf, file_type='3mf')
    src_buf.seek(0)

    # 2. Build custom_gcode_per_layer.xml layer nodes
    slot_colors = ["#C8C8C8", "#646464", "#000000", "#1a1a1a"]
    layer_nodes = ""
    for i, z in enumerate(sorted(color_changes_z)):
        extruder = i + 2
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

def suggest_midtones(image):
    """
    Uses K-Means clustering (K=4) to find the 4 dominant grayscale values in the image.
    Sorts them from darkest to lightest. Discards the darkest (Black) and lightest (White/BG).
    Returns the two intermediate values (L1, L2).
    """
    # Downsample image for faster k-means
    small_img = cv2.resize(image, (256, 256))
    data = np.float32(small_img.flatten())
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, _, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort centers from darkest to lightest
    centers = np.sort(centers.flatten())
    
    # centers[0] is Black (L3)
    # centers[1] is Dark Gray (L2)
    # centers[2] is Light Gray (L1)
    # centers[3] is White (L0)
    l2_val = int(centers[1])
    l1_val = int(centers[2])
    
    return l1_val, l2_val

# ---------------------------------------------------------------------------
# BACKGROUND MESH WORKER
# ---------------------------------------------------------------------------
class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    def __init__(self, img_filtered, sampled_values, max_dim, max_h, base_h,
                 output_path, output_path_3mf, color_changes_z, layer_height, max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15, color_mode=4, is_deckbox_mode=False):
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
        self.color_mode = color_mode
        self.is_deckbox_mode = is_deckbox_mode
        self.cancel_requested = False

    def run(self):
        import gc
        import time
        try:
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            t_start_total = time.time()
            
            t_img_start = time.time()
            self.progress.emit(5, "Optimizing resolution for 3D mesh...")
            img = self.img_filtered
            
            h, w = img.shape
            target_res = int(self.max_dim / 0.05)
            target_res = min(target_res, self.max_res_cap)
            
            if max(w, h) != target_res:
                scale_res = target_res / max(w, h)
                new_w, new_h = int(w * scale_res), int(h * scale_res)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            self.progress.emit(15, "Applying Smooth & Snap pre-processing...")
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            if self.color_mode == 2:
                # Binary thresholding aggressivo per disegni al tratto (B/W puro)
                _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
            else:
                # Multi-level Snap per preservare i grigi ma distruggere l'effetto pixel quadrato
                if self.color_mode == 3:
                    targets = np.array([0, self.sampled_values[2], 255])
                else:
                    targets = np.array([0, self.sampled_values[2], self.sampled_values[1], 255])
                
                # Trova il target più vicino per ogni pixel sfocato
                idx = np.abs(img[..., np.newaxis] - targets).argmin(axis=-1)
                img = targets[idx].astype(np.uint8)
            
            img_filtered = img
            
            self.progress.emit(20, "Applying Piecewise Interpolation (Z Mapping)...")
            
            L1_Z = self.color_changes_z[0]
            L2_Z = self.color_changes_z[1]
            
            l1_target = self.sampled_values[1]
            l2_target = self.sampled_values[2]
            
            if self.is_deckbox_mode:
                # DEBOSS: superficie piatta in alto (bianco=4mm), solchi scavati in basso (nero=1mm)
                deboss_depth = 3.0      # Profondità dell'incisione (mm)
                base_thickness = 1.0    # Spessore solido sul retro (mm)
                deboss_floor = base_thickness                  # Z per Nero (fondo solchi) = 1.0mm
                deboss_surface = base_thickness + deboss_depth  # Z per Bianco (superficie) = 4.0mm
                
                # Calcola le altezze intermedie proporzionalmente
                relief_range = self.max_h - self.base_h
                if relief_range > 0:
                    L1_ratio = (L1_Z - self.base_h) / relief_range
                    L2_ratio = (L2_Z - self.base_h) / relief_range
                else:
                    L1_ratio = 0.33
                    L2_ratio = 0.66
                # L1 (grigio chiaro) → poco sotto la superficie, L2 (grigio scuro) → più vicino al fondo
                L1_deboss = deboss_surface - L1_ratio * deboss_depth
                L2_deboss = deboss_surface - L2_ratio * deboss_depth
                
                # Mappatura esplicita: pixel scuro → Z basso (scavo), pixel chiaro → Z alto (superficie)
                if self.color_mode == 4:
                    midpoint = (l2_target + l1_target) / 2.0
                    x_points = [0, self.black_clip, midpoint, self.white_clip - 1, self.white_clip, 255]
                    y_points = [deboss_floor, deboss_floor, L2_deboss, L1_deboss, deboss_surface, deboss_surface]
                elif self.color_mode == 3:
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [deboss_floor, deboss_floor, L1_deboss, deboss_surface, deboss_surface]
                else: # 2 Colori
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [deboss_floor, deboss_floor, deboss_floor, deboss_surface, deboss_surface]
                
                print(f"[Deckbox Z-Map] Black(0)→{deboss_floor}mm, White(255)→{deboss_surface}mm")
            else:
                # Relief
                if self.color_mode == 4:
                    midpoint = (l2_target + l1_target) / 2.0
                    x_points = [0, self.black_clip, midpoint, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.max_h, self.max_h, L2_Z, L1_Z, self.base_h, self.base_h]
                elif self.color_mode == 3:
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.max_h, self.max_h, L1_Z, self.base_h, self.base_h]
                else: # 2 Colori
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.max_h, self.max_h, self.max_h, self.base_h, self.base_h]
            
            # Evitiamo valori non ordinati (es. se white_clip = 0 o simili configurazioni estreme)
            x_points = np.array(x_points, dtype=np.float64)
            y_points = np.array(y_points, dtype=np.float64)
            sort_idx = np.argsort(x_points)
            x_points = x_points[sort_idx]
            y_points = y_points[sort_idx]
            
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            
            t_img_end = time.time()
            print(f"[Profiling] Image processing & Z-Mapping setup: {t_img_end - t_img_start:.2f}s")
            
            t_mesh_start = time.time()
            self.progress.emit(25, "Generazione Vertici (Z-Mapping)...")
            
            Z_flat = np.interp(img_filtered.flatten(), x_points, y_points)
            Z_flat = np.round(Z_flat, 3)
            
            # Piallatura estrema della Base Z per i pixel bianchi puri
            if self.is_deckbox_mode:
                Z_flat[img_filtered.flatten() == 255] = deboss_surface
            else:
                Z_flat[img_filtered.flatten() == 255] = self.base_h
            
            Z = Z_flat.reshape(img_filtered.shape)
            
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
            vertices_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            idx = np.arange(w * h).reshape((h, w))
            tl = idx[:-1, :-1].ravel()
            tr = idx[:-1, 1:].ravel()
            bl = idx[1:, :-1].ravel()
            br = idx[1:, 1:].ravel()
            faces_top = np.vstack((np.column_stack((bl, tr, tl)), np.column_stack((br, tr, bl))))
            
            self.progress.emit(70, "Creazione Facce (Fondo)...")
            vertices_bottom = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(Z.ravel())))
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
            
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            
            mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
            
            if self.is_deckbox_mode:
                self.progress.emit(91, "Assembling Deckbox (Wall Replacement)...")
                import os
                import trimesh.transformations as tf
                template_path = os.path.join("assets", "template_deckbox_open.stl")
                if os.path.exists(template_path):
                    box_mesh = trimesh.load(template_path)
                    
                    # 1. Scala Fissa: forza le dimensioni esatte per l'incastro nel template
                    WALL_WIDTH  = 78.14   # mm (larghezza X)
                    WALL_HEIGHT = 98.0    # mm (altezza, 98mm per passare sotto la linguetta del coperchio)
                    
                    mesh_extents = mesh.extents
                    scale_x = WALL_WIDTH  / mesh_extents[0]
                    scale_y = WALL_HEIGHT / mesh_extents[1]
                    mesh.vertices[:, 0] *= scale_x
                    mesh.vertices[:, 1] *= scale_y
                    # Z resta intatta (4.0mm totali)
                    print(f"[Deckbox] Fixed scale: X={scale_x:.4f}, Y={scale_y:.4f} -> {WALL_WIDTH:.2f} x {WALL_HEIGHT:.2f} mm")
                    print(f"[Deckbox] Wall: {deboss_surface:.1f}mm total ({base_thickness:.1f}mm base + {deboss_depth:.1f}mm deboss)")
                    
                    # 2. Rotazione: ruota di 90° attorno all'asse X per mettere la mesh in piedi
                    rot_matrix = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
                    mesh.apply_transform(rot_matrix)
                    
                    # 3. Posizionamento: Anchor-point alignment (no centroid)
                    box_min = box_mesh.bounds[0]
                    box_max = box_mesh.bounds[1]
                    mesh_min = mesh.bounds[0]
                    mesh_max = mesh.bounds[1]
                    art_thickness = mesh_max[1] - mesh_min[1]
                    
                    # X: centra la mesh nella larghezza del vano
                    box_center_x = (box_min[0] + box_max[0]) / 2.0
                    mesh_center_x = (mesh_min[0] + mesh_max[0]) / 2.0
                    tx = box_center_x - mesh_center_x
                    
                    # Y: faccia deboss a filo con l'esterno della scatola (box_min[1])
                    #    la superficie (mesh_min[1] dopo rotazione = vecchia Z alta = bianco)
                    #    deve essere a filo con l'esterno, il resto va verso l'interno
                    ty = box_min[1] - mesh_min[1] + 0.1  # +0.1mm per micro-saldatura bordi
                    
                    # Z: base della mesh a terra, allineata con la base della scatola
                    tz = box_min[2] - mesh_min[2]
                    
                    mesh.apply_translation([tx, ty, tz])
                    print(f"[Deckbox] Art thickness: {art_thickness:.2f}mm")
                    print(f"[Deckbox] Mesh bounds PRE-translate: min={mesh_min}, max={mesh_max}")
                    print(f"[Deckbox] Box bounds: min={box_min}, max={box_max}")
                    print(f"[Deckbox] Translation: X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}")
                    
                    # 4. Concatenazione: incolla la parete art sul template aperto
                    mesh = trimesh.util.concatenate([box_mesh, mesh])
                else:
                    print(f"Warning: Deckbox template not found at {template_path}")
            
            t_mesh_end = time.time()
            print(f"[Profiling] Raw mesh generation: {t_mesh_end - t_mesh_start:.2f}s")
            
            if self.smart_decimate and len(mesh.faces) > 200_000:
                t_dec_start = time.time()
                # Calcolo del target: partiamo da 100k e aggiungiamo un buffer basato sulla dimensione originale,
                # ma non superiamo mai i 150k triangoli per mantenere il file .3mf leggerissimo.
                base_target = 100_000
                extra = min(50_000, int((len(mesh.faces) - 200_000) * 0.05))
                target_faces = base_target + extra
                
                self.progress.emit(92, f"Brutal Decimation ({len(mesh.faces):,} → {target_faces:,} faces)...")
                verts_out, faces_out = fast_simplification.simplify(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                    target_count=target_faces,
                    agg=6.0  # Maggiore aggressività per collassare più facce piatte
                )
                mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
                trimesh.repair.fix_normals(mesh)
                
                # Safety check: Assicuriamoci che la mesh sia ancora chiusa (watertight)
                if not mesh.is_watertight:
                    self.progress.emit(94, "Repairing decimated mesh (filling holes)...")
                    try:
                        trimesh.repair.fill_holes(mesh)
                        trimesh.repair.fix_normals(mesh)
                    except Exception as e:
                        print(f"Warning: Hole filling failed - {e}")
                
                t_dec_end = time.time()
                print(f"[Profiling] Smart Optimization (Decimation): {t_dec_end - t_dec_start:.2f}s")
            
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            
            t_export_start = time.time()
            self.progress.emit(95, "Mathematical flattening and absolute Z clamping...")
            verts = mesh.vertices.copy()
            
            # 1. Forza la base piatta solo se non siamo in deckbox mode (altrimenti pialliamo il box template!)
            if not self.is_deckbox_mode:
                bottom_mask = verts[:, 2] < 0.05
                verts[bottom_mask, 2] = 0.0
            
            # 2. Ghigliottina matematica (Clamping assoluto) per rimuovere eventuali 'overshoot' da decimazione o smoothing
            if not self.is_deckbox_mode:
                verts[:, 2] = np.clip(verts[:, 2], 0.0, self.max_h)
            
            # 3. Quantizzazione finale (Tronca decimali superflui per evitare layer fantasma nello slicer)
            verts[:, 2] = np.round(verts[:, 2], 3)
            
            mesh.vertices = verts
            
            if self.is_deckbox_mode:
                self.progress.emit(96, "Esportazione Deckbox Files...")
                import os
                out_dir = os.path.dirname(self.output_path_3mf) if self.output_path_3mf else os.path.dirname(self.output_path)
                os.makedirs(out_dir, exist_ok=True)
                
                front_path = os.path.join(out_dir, "deckbox_custom_front.3mf")
                lid_path = os.path.join(out_dir, "deckbox_lid_blank.3mf")
                
                # Esporta il corpo del deckbox (.3mf)
                export_3mf(mesh, front_path, self.color_changes_z)
                
                # Esporta anche STL se richiesto
                if self.output_path:
                    stl_dir = os.path.dirname(self.output_path)
                    os.makedirs(stl_dir, exist_ok=True)
                    front_stl = os.path.join(stl_dir, "deckbox_custom_front.stl")
                    mesh.export(front_stl)
                    print(f"[Deckbox] STL exported: {front_stl}")
                
                # Esporta il coperchio liscio come file separato
                template_lid = os.path.join("assets", "template_coperchio_base.stl")
                if os.path.exists(template_lid):
                    lid_mesh = trimesh.load(template_lid)
                    export_3mf(lid_mesh, lid_path, self.color_changes_z)
                    if self.output_path:
                        lid_stl = os.path.join(stl_dir, "deckbox_lid_blank.stl")
                        lid_mesh.export(lid_stl)
                
                t_export_end = time.time()
                print(f"[Profiling] Export files: {t_export_end - t_export_start:.2f}s")
                print(f"[Profiling] TOTAL TIME: {t_export_end - t_start_total:.2f}s")

                self.progress.emit(100, "Esportazione Deckbox completata!")
                self.finished_ok.emit(front_path, lid_path)

            else:
                if self.output_path:
                    self.progress.emit(96, "Esportazione STL...")
                    mesh.export(self.output_path)
                    
                if self.cancel_requested: raise InterruptedError("Process cancelled by user")

                if self.output_path_3mf:
                    self.progress.emit(98, "Esportazione 3MF...")
                    export_3mf(mesh, self.output_path_3mf, self.color_changes_z)

                t_export_end = time.time()
                print(f"[Profiling] Export files: {t_export_end - t_export_start:.2f}s")
                print(f"[Profiling] TOTAL TIME: {t_export_end - t_start_total:.2f}s")

                self.progress.emit(100, "Esportazione completata!")
                self.finished_ok.emit(self.output_path, self.output_path_3mf)
            
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

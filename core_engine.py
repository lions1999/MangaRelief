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
                # Deboss: White is flush surface (max_h), Black is deepest groove (base_h)
                L1_inverted = self.max_h - (L1_Z - self.base_h)
                L2_inverted = self.max_h - (L2_Z - self.base_h)
                
                if self.color_mode == 4:
                    midpoint = (l2_target + l1_target) / 2.0
                    x_points = [0, self.black_clip, midpoint, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.base_h, self.base_h, L2_inverted, L1_inverted, self.max_h, self.max_h]
                elif self.color_mode == 3:
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.base_h, self.base_h, L1_inverted, self.max_h, self.max_h]
                else: # 2 Colori
                    x_points = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                    y_points = [self.base_h, self.base_h, self.base_h, self.max_h, self.max_h]
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
            
            # Piallatura estrema della Base Z per i pixel bianchi puri (evita fluttuazioni millimetriche)
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
                self.progress.emit(91, "Assembling Deckbox Template...")
                import os
                import trimesh.transformations as tf
                template_path = os.path.join("assets", "template_deckbox_base.stl")
                if os.path.exists(template_path):
                    box_mesh = trimesh.load(template_path)
                    
                    # 1. Scala Dinamica: adatta la larghezza della mesh a quella del frontale della scatola
                    box_extents = box_mesh.extents
                    mesh_extents = mesh.extents
                    scale_factor = box_extents[0] / mesh_extents[0]
                    mesh.apply_scale(scale_factor)
                    print(f"[Deckbox] Scale factor: {scale_factor:.4f} (box X={box_extents[0]:.2f}, mesh X={mesh_extents[0]:.2f})")
                    
                    # 2. Rotazione: ruota di 90° attorno all'asse X per mettere la mesh in piedi
                    rot_matrix = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
                    mesh.apply_transform(rot_matrix)
                    
                    # 3. Allineamento Frontale: centra la mesh sulla faccia frontale (min Y) della scatola
                    box_min = box_mesh.bounds[0]
                    box_max = box_mesh.bounds[1]
                    mesh_min = mesh.bounds[0]
                    mesh_max = mesh.bounds[1]
                    mesh_center = (mesh_min + mesh_max) / 2.0
                    
                    target_x = (box_min[0] + box_max[0]) / 2.0
                    target_y = box_min[1] + 0.5  # Compenetrazione di 0.5mm nella parete frontale
                    target_z = (box_min[2] + box_max[2]) / 2.0
                    
                    translation = [
                        target_x - mesh_center[0],
                        target_y - mesh_min[1],    # Allinea il fronte della mesh al min Y della scatola
                        target_z - mesh_center[2],
                    ]
                    mesh.apply_translation(translation)
                    print(f"[Deckbox] Translation: X={translation[0]:.2f}, Y={translation[1]:.2f}, Z={translation[2]:.2f}")
                    
                    # 4. Merge
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
                
                # Esporta 3MF se richiesto
                if self.output_path_3mf:
                    out_dir_3mf = os.path.dirname(self.output_path_3mf)
                    os.makedirs(out_dir_3mf, exist_ok=True)
                    
                    front_path_3mf = os.path.join(out_dir_3mf, "deckbox_custom_front.3mf")
                    lid_path_3mf = os.path.join(out_dir_3mf, "deckbox_lid_blank.3mf")
                    
                    export_3mf(mesh, front_path_3mf, self.color_changes_z)
                    
                    template_lid = os.path.join("assets", "template_coperchio_base.stl")
                    if os.path.exists(template_lid):
                        lid_mesh = trimesh.load(template_lid)
                        export_3mf(lid_mesh, lid_path_3mf, self.color_changes_z)
                
                # Esporta STL se richiesto
                front_path_stl = None
                if self.output_path:
                    out_dir_stl = os.path.dirname(self.output_path)
                    os.makedirs(out_dir_stl, exist_ok=True)
                    
                    front_path_stl = os.path.join(out_dir_stl, "deckbox_custom_front.stl")
                    lid_path_stl = os.path.join(out_dir_stl, "deckbox_lid_blank.stl")
                    
                    mesh.export(front_path_stl)
                    
                    template_lid = os.path.join("assets", "template_coperchio_base.stl")
                    if os.path.exists(template_lid):
                        lid_mesh = trimesh.load(template_lid)
                        lid_mesh.export(lid_path_stl)
                
                t_export_end = time.time()
                print(f"[Profiling] Export files: {t_export_end - t_export_start:.2f}s")
                print(f"[Profiling] TOTAL TIME: {t_export_end - t_start_total:.2f}s")

                self.progress.emit(100, "Esportazione Deckbox completata!")
                stl_result = front_path_stl if front_path_stl else ""
                mf_result = front_path_3mf if self.output_path_3mf else ""
                self.finished_ok.emit(stl_result, mf_result)

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

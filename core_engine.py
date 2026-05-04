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
                 output_path, output_path_3mf, color_changes_z, layer_height, max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15, color_mode=4):
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

    def run(self):
        import gc
        try:
            self.progress.emit(5, "Optimizing resolution for 3D mesh...")
            img = self.img_filtered
            
            h, w = img.shape
            target_res = int(self.max_dim / 0.05)
            target_res = min(target_res, self.max_res_cap)
            
            if max(w, h) != target_res:
                scale_res = target_res / max(w, h)
                new_w, new_h = int(w * scale_res), int(h * scale_res)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            img_filtered = img
            
            self.progress.emit(20, "Applying Piecewise Interpolation (Z Mapping)...")
            
            L1_Z = self.color_changes_z[0]
            L2_Z = self.color_changes_z[1]
            
            l1_target = self.sampled_values[1]
            l2_target = self.sampled_values[2]
            
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
            
            self.progress.emit(25, "Generazione Vertici (Z-Mapping)...")
            
            Z_flat = np.interp(img_filtered.flatten(), x_points, y_points)
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
            trimesh.repair.fix_normals(mesh)
            
            if self.smart_decimate and len(mesh.faces) > 200_000:
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
            
            self.progress.emit(96, "Esportazione STL...")
            mesh.export(self.output_path)

            self.progress.emit(98, "Esportazione 3MF...")
            export_3mf(mesh, self.output_path_3mf, self.color_changes_z)

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

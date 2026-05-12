import os
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from PIL import Image

import io
import zipfile
import cv2
import numpy as np
import trimesh
import fast_simplification

from PyQt6.QtCore import QThread, pyqtSignal

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
class DeckboxConfig:
    """Standard dimensions for the deckbox templates."""
    WALL_WIDTH = 78.14
    WALL_HEIGHT = 98.0
    DEBOSS_DEPTH = 3.0
    BASE_THICKNESS = 1.0
    
    # Lid Logo (Plug & Play)
    PLUG_W = 60.0
    PLUG_H = 30.0
    PLUG_Z = 2.0
    ENGRAVE_FLOOR = 0.4
    NOTCH_Y_OFFSET = 16.818

def create_solid_mesh(X, Y, Z, bottom_z=0.0):
    """
    Generates a solid watertight mesh from X, Y, Z meshgrids.
    Seals the bottom and the four sides.
    """
    h, w = Z.shape
    offset = w * h
    
    # Top vertices and faces
    vertices_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    idx = np.arange(w * h).reshape((h, w))
    tl = idx[:-1, :-1].ravel()
    tr = idx[:-1, 1:].ravel()
    bl = idx[1:, :-1].ravel()
    br = idx[1:, 1:].ravel()
    faces_top = np.vstack((np.column_stack((bl, tr, tl)), np.column_stack((br, tr, bl))))
    
    # Bottom vertices and faces
    vertices_bottom = np.column_stack((X.ravel(), Y.ravel(), np.full_like(Z.ravel(), bottom_z)))
    tl_b = tl + offset
    tr_b = tr + offset
    bl_b = bl + offset
    br_b = br + offset
    faces_bottom = np.vstack((np.column_stack((tl_b, tr_b, bl_b)), np.column_stack((bl_b, tr_b, br_b))))
    
    # Side faces (Sealing edges)
    # Top edge
    v1, v2 = idx[0, :-1], idx[0, 1:]
    top_sides = np.vstack((np.column_stack((v1, v2, v1 + offset)), np.column_stack((v2, v2 + offset, v1 + offset))))
    
    # Bottom edge
    v1, v2 = idx[-1, :-1], idx[-1, 1:]
    bot_sides = np.vstack((np.column_stack((v2, v1, v1 + offset)), np.column_stack((v2 + offset, v2, v1 + offset))))
    
    # Left edge
    v1, v2 = idx[:-1, 0], idx[1:, 0]
    left_sides = np.vstack((np.column_stack((v1, v2, v1 + offset)), np.column_stack((v2, v2 + offset, v1 + offset))))
    
    # Right edge
    v1, v2 = idx[:-1, -1], idx[1:, -1]
    right_sides = np.vstack((np.column_stack((v2, v1, v1 + offset)), np.column_stack((v2 + offset, v2, v1 + offset))))
    
    all_vertices = np.vstack((vertices_top, vertices_bottom))
    all_faces = np.vstack((faces_top, faces_bottom, top_sides, bot_sides, left_sides, right_sides))
    
    return trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)

# ---------------------------------------------------------------------------
# TOPOGRAPHIC COLOR MODE (K-Means & Terrace Generation)
# ---------------------------------------------------------------------------
def extract_dominant_colors(image_rgb: np.ndarray, n_colors: int = 5) -> list:
    """Estrae i colori dominanti e li ordina per luminosità (dal più scuro al più chiaro)."""
    pixels = image_rgb.reshape(-1, 3)
    # n_init='auto' per sopprimere warning e velocizzare
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Ordina i colori per luminosità percepita (Luminance)
    luminances = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors]
    sorted_indices = np.argsort(luminances)
    sorted_colors = colors[sorted_indices]
    
    return [tuple(c) for c in sorted_colors]

def process_mesh_topo(image_rgb: np.ndarray, sorted_colors_rgb: list, 
                      base_z: float = 1.0, total_z: float = 2.4, max_dim: float = 100.0):
    """Genera una mesh a terrazze basata sui colori forniti."""
    # Pre-scaling a 800px per performance e pulizia stampa
    h, w = image_rgb.shape[:2]
    max_size = 800
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        image_rgb = np.array(img_pil)
        h, w = image_rgb.shape[:2]

    n_colors = len(sorted_colors_rgb)
    layer_step = (total_z - base_z) / (n_colors - 1) if n_colors > 1 else 0

    # Mappa pixel ai colori tramite cKDTree (velocissimo)
    tree = cKDTree(sorted_colors_rgb)
    pixels_flat = image_rgb.reshape(-1, 3)
    _, indices = tree.query(pixels_flat)
    indices = indices.reshape(h, w)

    # Costruisci heightmap discreta
    Z = np.zeros((h, w), dtype=np.float32)
    for i in range(n_colors):
        mask = (indices == i)
        # Arrotondamento per evitare problemi di precisione nello slicer
        Z[mask] = round(base_z + (i * layer_step), 3)

    # Calcolo dimensioni meshgrid
    if w >= h:
        dim_x = float(max_dim)
        dim_y = float(max_dim) * (h / w)
    else:
        dim_y = float(max_dim)
        dim_x = float(max_dim) * (w / h)

    x = np.linspace(0, dim_x, w)
    y = np.linspace(0, dim_y, h)[::-1]
    X, Y = np.meshgrid(x, y)

    # Generazione Mesh tramite la utility interna
    mesh = create_solid_mesh(X, Y, Z, bottom_z=0.0)
    return mesh


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
                 output_path, output_path_3mf, color_changes_z, layer_height, max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15, color_mode=4, is_deckbox_mode=False, tcg_name='Yu-Gi-Oh!'):
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
        self.tcg_name = tcg_name
        self.cancel_requested = False
        
        # Mapping TCG names to logo asset files
        self.TCG_LOGO_MAP = {
            'Yu-Gi-Oh!': 'yugioh_logo.jpg',
            'Pokémon': 'pokemon_logo.jpg',
            'Magic': 'magic_logo.jpg',
            'One Piece': 'onepiece_logo.jpg',
        }

    def _prepare_image_data(self, img):
        """Optimizes resolution and applies Gaussian blur/Snap filters."""
        h, w = img.shape
        target_res = int(self.max_dim / 0.05)
        target_res = min(target_res, self.max_res_cap)
        
        if max(w, h) != target_res:
            scale_res = target_res / max(w, h)
            new_w, new_h = int(w * scale_res), int(h * scale_res)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        if self.color_mode == 2:
            _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        else:
            if self.color_mode == 3:
                targets = np.array([0, self.sampled_values[2], 255])
            else:
                targets = np.array([0, self.sampled_values[2], self.sampled_values[1], 255])
            
            idx = np.abs(img[..., np.newaxis] - targets).argmin(axis=-1)
            img = targets[idx].astype(np.uint8)
        return img

    def _get_z_mapping(self):
        """Calculates X and Y points for piecewise linear interpolation."""
        L1_Z = self.color_changes_z[0]
        L2_Z = self.color_changes_z[1]
        l1_target = self.sampled_values[1]
        l2_target = self.sampled_values[2]
        
        if self.is_deckbox_mode:
            deboss_depth = DeckboxConfig.DEBOSS_DEPTH
            base_thickness = DeckboxConfig.BASE_THICKNESS
            deboss_floor = base_thickness
            deboss_surface = base_thickness + deboss_depth
            
            relief_range = self.max_h - self.base_h
            L1_ratio = (L1_Z - self.base_h) / relief_range if relief_range > 0 else 0.33
            L2_ratio = (L2_Z - self.base_h) / relief_range if relief_range > 0 else 0.66
            L1_deboss = deboss_floor + L1_ratio * deboss_depth
            L2_deboss = deboss_floor + L2_ratio * deboss_depth
            
            if self.color_mode == 4:
                midpoint = (l2_target + l1_target) / 2.0
                x_pts = [0, self.black_clip, midpoint, self.white_clip - 1, self.white_clip, 255]
                y_pts = [deboss_surface, deboss_surface, L2_deboss, L1_deboss, deboss_floor, deboss_floor]
            elif self.color_mode == 3:
                x_pts = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                y_pts = [deboss_surface, deboss_surface, L1_deboss, deboss_floor, deboss_floor]
            else: # 2 Colors
                x_pts = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                y_pts = [deboss_surface, deboss_surface, deboss_surface, deboss_floor, deboss_floor]
            
            # Sort points to avoid interpolation errors
            x_pts, y_pts = np.array(x_pts), np.array(y_pts)
            s_idx = np.argsort(x_pts)
            return x_pts[s_idx], y_pts[s_idx], deboss_floor, deboss_surface
        else:
            if self.color_mode == 4:
                midpoint = (l2_target + l1_target) / 2.0
                x_pts = [0, self.black_clip, midpoint, self.white_clip - 1, self.white_clip, 255]
                y_pts = [self.max_h, self.max_h, L2_Z, L1_Z, self.base_h, self.base_h]
            elif self.color_mode == 3:
                x_pts = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                y_pts = [self.max_h, self.max_h, L1_Z, self.base_h, self.base_h]
            else: # 2 Colors
                x_pts = [0, self.black_clip, self.white_clip - 1, self.white_clip, 255]
                y_pts = [self.max_h, self.max_h, self.max_h, self.base_h, self.base_h]
            
            x_pts, y_pts = np.array(x_pts), np.array(y_pts)
            s_idx = np.argsort(x_pts)
            return x_pts[s_idx], y_pts[s_idx], self.base_h, self.max_h

    def _assemble_deckbox_mesh(self, mesh):
        """Scales, rotates and concatenates the art mesh with the deckbox template."""
        import trimesh.transformations as tf
        template_path = os.path.join("assets", "template_deckbox_open.stl")
        if not os.path.exists(template_path):
            print(f"Warning: Deckbox template not found at {template_path}")
            return mesh

        box_mesh = trimesh.load(template_path)
        
        # Fixed scale to fit the template notch
        mesh_extents = mesh.extents
        scale_x = DeckboxConfig.WALL_WIDTH  / mesh_extents[0]
        scale_y = DeckboxConfig.WALL_HEIGHT / mesh_extents[1]
        mesh.vertices[:, 0] *= scale_x
        mesh.vertices[:, 1] *= scale_y
        
        # Rotation: 90° around X axis to stand up
        mesh.apply_transform(tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
        
        # Alignment
        box_min, box_max = box_mesh.bounds
        mesh_min, mesh_max = mesh.bounds
        
        tx = ((box_min[0] + box_max[0]) / 2.0) - ((mesh_min[0] + mesh_max[0]) / 2.0)
        ty = box_min[1] - mesh_min[1] + 0.1  # Micro-weld overlap
        tz = box_min[2] - mesh_min[2]
        
        mesh.apply_translation([tx, ty, tz])
        return trimesh.util.concatenate([box_mesh, mesh])

    def _process_lid_logo(self, out_dir, stl_dir):
        """Generates the TCG logo plug and engravings for the deckbox lid."""
        template_lid = os.path.join("assets", "template_coperchio_bucato.stl")
        if not os.path.exists(template_lid):
            print(f"WARNING: Lid template not found at {template_lid}")
            return None
            
        lid_mesh = trimesh.load(template_lid)
        logo_filename = self.TCG_LOGO_MAP.get(self.tcg_name)
        logo_path = os.path.join("assets", logo_filename) if logo_filename else None
        
        if logo_path and os.path.exists(logo_path):
            self.progress.emit(97, f"Engraving {self.tcg_name} logo on lid...")
            try:
                logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
                if logo_img is not None:
                    logo_img = cv2.GaussianBlur(logo_img, (3, 3), 0)
                    _, logo_img = cv2.threshold(logo_img, 150, 255, cv2.THRESH_BINARY)
                    
                    engrave_depth = DeckboxConfig.PLUG_Z - DeckboxConfig.ENGRAVE_FLOOR
                    logo_norm = logo_img.astype(np.float64) / 255.0
                    Z_logo = DeckboxConfig.PLUG_Z - logo_norm * engrave_depth
                    Z_logo = np.round(Z_logo, 3)
                    
                    lh, lw = logo_img.shape
                    lx = np.linspace(0, DeckboxConfig.PLUG_W, lw)
                    ly = np.linspace(0, DeckboxConfig.PLUG_H, lh)[::-1]
                    LX, LY = np.meshgrid(lx, ly)
                    
                    logo_mesh = create_solid_mesh(LX, LY, Z_logo, bottom_z=DeckboxConfig.ENGRAVE_FLOOR)
                    
                    # Alignment
                    lid_min, lid_max = lid_mesh.bounds
                    logo_min, logo_max = logo_mesh.bounds
                    tx = ((lid_min[0] + lid_max[0]) / 2.0) - ((logo_min[0] + logo_max[0]) / 2.0)
                    ty = (lid_min[1] + DeckboxConfig.NOTCH_Y_OFFSET) - logo_min[1]
                    tz = lid_max[2] - logo_max[2]
                    
                    logo_mesh.apply_translation([tx, ty, tz])
                    lid_mesh = trimesh.util.concatenate([lid_mesh, logo_mesh])
            except Exception as e:
                print(f"Warning: Logo engraving failed ({e})")
        
        lid_custom_path = os.path.join(out_dir, "deckbox_lid_custom.3mf")
        export_3mf(lid_mesh, lid_custom_path, self.color_changes_z)
        if self.output_path:
            lid_mesh.export(os.path.join(stl_dir, "deckbox_lid_custom.stl"))
        return lid_custom_path

    def run(self):
        import gc
        import time
        import os
        try:
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            t_start_total = time.time()
            
            # --- 1. Image Preparation ---
            self.progress.emit(5, "Optimizing resolution for 3D mesh...")
            img = self._prepare_image_data(self.img_filtered)
            
            # --- 2. Z-Mapping & Heightmap ---
            self.progress.emit(20, "Applying Piecewise Interpolation (Z Mapping)...")
            x_pts, y_pts, floor_z, surface_z = self._get_z_mapping()
            
            Z_flat = np.interp(img.flatten(), x_pts, y_pts)
            Z_flat = np.round(Z_flat, 3)
            
            # Clamp extremes for pure black/white
            Z_flat[img.flatten() == 255] = floor_z
            Z_flat[img.flatten() == 0] = surface_z
            Z = Z_flat.reshape(img.shape)
            
            # --- 3. Grid Generation ---
            self.progress.emit(40, "Generazione Vertici (MeshGrid)...")
            h, w = img.shape
            if w >= h:
                dim_x = float(self.max_dim)
                dim_y = float(self.max_dim) * (h / w)
            else:
                dim_y = float(self.max_dim)
                dim_x = float(self.max_dim) * (w / h)

            x = np.linspace(0, dim_x, w)
            y = np.linspace(0, dim_y, h)[::-1]
            X, Y = np.meshgrid(x, y)
            
            if self.is_deckbox_mode:
                # Add 2mm border logic using constants
                bx = 2.0 * (dim_x / DeckboxConfig.WALL_WIDTH)
                by = 2.0 * (dim_y / DeckboxConfig.WALL_HEIGHT)
                b_mask = (X < bx) | (X > dim_x - bx) | (Y < by) | (Y > dim_y - by)
                Z[b_mask] = surface_z
            
            # --- 4. Mesh Assembly ---
            self.progress.emit(55, "Creating 3D Geometry...")
            mesh = create_solid_mesh(X, Y, Z, bottom_z=0.0)
            
            if self.is_deckbox_mode:
                self.progress.emit(91, "Assembling Deckbox (Wall Replacement)...")
                mesh = self._assemble_deckbox_mesh(mesh)
            
            # --- 5. Optimization (Decimation) ---
            if self.smart_decimate and len(mesh.faces) > 200_000:
                self.progress.emit(92, "Optimizing Mesh (Decimation)...")
                target = 100_000 + min(50_000, int((len(mesh.faces) - 200_000) * 0.05))
                v_out, f_out = fast_simplification.simplify(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                    target_count=target, agg=6.0
                )
                mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
                trimesh.repair.fix_normals(mesh)
                if not mesh.is_watertight:
                    trimesh.repair.fill_holes(mesh)

            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            
            # --- 6. Final Clamping & Export ---
            self.progress.emit(95, "Finalizing and Exporting...")
            if not self.is_deckbox_mode:
                # Absolute Z clamping for relief panels
                mesh.vertices[:, 2] = np.clip(mesh.vertices[:, 2], 0.0, self.max_h)
            mesh.vertices[:, 2] = np.round(mesh.vertices[:, 2], 3)
            
            if self.is_deckbox_mode:
                out_dir = os.path.dirname(self.output_path_3mf or self.output_path)
                os.makedirs(out_dir, exist_ok=True)
                stl_dir = os.path.dirname(self.output_path) if self.output_path else out_dir
                
                # Export Main Body
                front_path = os.path.join(out_dir, "deckbox_custom_front.3mf")
                export_3mf(mesh, front_path, self.color_changes_z)
                if self.output_path:
                    mesh.export(os.path.join(stl_dir, "deckbox_custom_front.stl"))
                
                # Export Lid
                lid_path = self._process_lid_logo(out_dir, stl_dir)
                
                self.progress.emit(100, "Export completed!")
                self.finished_ok.emit(front_path, lid_path or "")
            else:
                if self.output_path:
                    mesh.export(self.output_path)
                if self.output_path_3mf:
                    export_3mf(mesh, self.output_path_3mf, self.color_changes_z)
                
                self.progress.emit(100, "Export completed!")
                self.finished_ok.emit(self.output_path or "", self.output_path_3mf or "")

            gc.collect()
            t_total = time.time() - t_start_total
            print(f"[Profiling] TOTAL REFACTORED TIME: {t_total:.2f}s")
            
        except Exception as e:
            self.finished_err.emit(str(e))

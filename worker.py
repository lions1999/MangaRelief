import os
import time
import gc
import numpy as np
import cv2
import trimesh
import fast_simplification
from PyQt6.QtCore import QThread, pyqtSignal

from config import DeckboxConfig
from mesh_utils import create_solid_mesh, process_mesh_topo, export_3mf
from utils import resource_path

class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    def __init__(self, img_filtered, sampled_values, max_dim, max_h, base_h,
                 output_path, output_path_3mf, color_changes_z, layer_height, 
                 max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15, 
                 color_mode=4, is_deckbox_mode=False, tcg_name='Yu-Gi-Oh!',
                 is_topo_mode=False, topo_colors=None):
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
        self.is_topo_mode = is_topo_mode
        self.topo_colors = topo_colors
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
            
            deboss_surface = base_thickness
            # FIX: Inseriamo il limite di sicurezza basato sullo spessore residuo solido
            deboss_floor = max(DeckboxConfig.MIN_SOLID_WALL_THICKNESS, base_thickness - deboss_depth) 
            
            relief_range = self.max_h - self.base_h
            L1_ratio = (L1_Z - self.base_h) / relief_range if relief_range > 0 else 0.33
            L2_ratio = (L2_Z - self.base_h) / relief_range if relief_range > 0 else 0.66
            
            # I calcoli dei layer Z scalano da deboss_floor (fondo scavo) a deboss_surface (superficie muro)
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
        template_path = resource_path(os.path.join("assets", "template_deckbox_open.stl"))
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
        template_lid = resource_path(os.path.join("assets", "template_coperchio_bucato.stl"))
        if not os.path.exists(template_lid):
            print(f"WARNING: Lid template not found at {template_lid}")
            return None
            
        lid_mesh = trimesh.load(template_lid)
        logo_filename = self.TCG_LOGO_MAP.get(self.tcg_name)
        logo_path = resource_path(os.path.join("assets", logo_filename)) if logo_filename else None
        
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
        try:
            if self.cancel_requested: raise InterruptedError("Process cancelled by user")
            t_start_total = time.time()
            
            if self.is_topo_mode and self.topo_colors:
                self.progress.emit(10, "🏔 Starting Topographic Color Processing...")
                # Ensure we have RGB image for K-Means consistency
                if isinstance(self.img_filtered, np.ndarray) and len(self.img_filtered.shape) == 2:
                    # If passed grayscale, we convert back to RGB for the color pipeline
                    img_rgb = cv2.cvtColor(self.img_filtered, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = self.img_filtered

                mesh = process_mesh_topo(
                    img_rgb, 
                    self.topo_colors, 
                    base_z=self.base_h, 
                    total_z=self.max_h,
                    max_dim=self.max_dim,
                    layer_height=self.layer_height
                )
                self.progress.emit(80, "Optimizing Topo Mesh...")
                
                # In modalità Topo la mesh è già completa, saltiamo la pipeline standard
                goto_export = True
            else:
                goto_export = False
                # --- 1. Image Preparation ---
                self.progress.emit(5, "Optimizing resolution for 3D mesh...")
                img = self._prepare_image_data(self.img_filtered)
            
            if not goto_export:
                # --- 2. Z-Mapping & Heightmap ---
                self.progress.emit(20, "Applying Piecewise Interpolation (Z Mapping)...")
                x_pts, y_pts, floor_z, surface_z = self._get_z_mapping()
                
                Z_flat = np.interp(img.flatten(), x_pts, y_pts)
                Z_flat = np.round(Z_flat, 3)
                
                # Clamp extremes for pure black/white
                Z_flat[img.flatten() <= self.black_clip] = surface_z
                Z_flat[img.flatten() >= self.white_clip] = floor_z
                
                h, w = img.shape
                Z = Z_flat.reshape((h, w))
                
                # --- 3. Grid & Geometry ---
                if w >= h:
                    dim_x = float(self.max_dim)
                    dim_y = float(self.max_dim) * (h / w)
                else:
                    dim_y = float(self.max_dim)
                    dim_x = float(self.max_dim) * (w / h)
                
                x = np.linspace(0, dim_x, w)
                y = np.linspace(0, dim_y, h)[::-1]
                X, Y = np.meshgrid(x, y)
                
                self.progress.emit(40, "Generating solid vertices (Watertight)...")
                mesh = create_solid_mesh(X, Y, Z, bottom_z=0.0)
                
                if self.is_deckbox_mode:
                    # Add 2mm border logic using constants
                    bx = 2.0 * (dim_x / DeckboxConfig.WALL_WIDTH)
                    by = 2.0 * (dim_y / DeckboxConfig.WALL_HEIGHT)
                    b_mask = (X < bx) | (X > dim_x - bx) | (Y < by) | (Y > dim_y - by)
                    Z[b_mask] = surface_z
            
            # --- 4. Mesh Assembly ---
            self.progress.emit(55, "Finalizing Geometry...")
            
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

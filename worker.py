import os
import time
import gc
import numpy as np
import cv2
import trimesh
import fast_simplification
from PyQt6.QtCore import QThread, pyqtSignal

from config import DeckboxConfig
from mesh_utils import (create_solid_mesh, process_mesh_topo, export_3mf,
                        compute_topo_z_heights, compute_topo_switch_z)
from color_utils import classify_spot_pixels, downsample_for_analysis
from case_utils import build_plate_raster, compose_cover_art, build_bumper
from utils import resource_path

class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    # Mapping TCG game names → logo asset filenames (class-level: shared across all instances)
    TCG_LOGO_MAP = {
        'Yu-Gi-Oh!':       'yugioh_logo.jpg',
        'Pokémon':         'pokemon_logo.jpg',
        'Magic':           'magic_logo.jpg',
        'One Piece':       'onepiece_logo.jpg',
        'Hunter x Hunter': 'hxh_logo.jpg',
    }

    def __init__(self, img_filtered, sampled_values, max_dim, max_h, base_h,
                 output_path, output_path_3mf, color_changes_z, layer_height,
                 max_res_cap=1200, smart_decimate=True, white_clip=235, black_clip=15,
                 color_mode=4, is_deckbox_mode=False, tcg_name='Yu-Gi-Oh!',
                 is_topo_mode=False, topo_colors=None, source_image_name='panel',
                 is_spot_mode=False, spot_accents=None, spot_coverage=40,
                 is_cover_mode=False, cover_preset=None, cover_scale=1.0,
                 cover_off_x=0.0, cover_off_y=0.0, cover_finish_spot=False,
                 include_bumper=False):
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
        self.source_image_name = source_image_name
        self.is_spot_mode = is_spot_mode
        self.spot_accents = spot_accents or []
        self.spot_coverage = spot_coverage
        self.is_cover_mode = is_cover_mode
        self.cover_preset = cover_preset
        self.cover_scale = cover_scale
        self.cover_off_x = cover_off_x
        self.cover_off_y = cover_off_y
        self.cover_finish_spot = cover_finish_spot
        self.include_bumper = include_bumper
        self.cancel_requested = False

    def _check_cancel(self):
        """Interrompe subito la pipeline se l'utente ha premuto Cancel."""
        if self.cancel_requested:
            raise InterruptedError("Process cancelled by user")

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
            # Soglia dinamica: punto medio tra il bianco (L0) e il nero (L3) campionati,
            # così la calibrazione degli swatch conta anche in modalità B&N
            bw_threshold = int((self.sampled_values[0] + self.sampled_values[3]) / 2.0)
            bw_threshold = int(np.clip(bw_threshold, 1, 254))
            _, img = cv2.threshold(img, bw_threshold, 255, cv2.THRESH_BINARY)
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
        
        # Fixed scale to fit the template notch + 0.2mm overlap to prevent gaps
        mesh_extents = mesh.extents
        scale_x = (DeckboxConfig.WALL_WIDTH + 0.2)  / mesh_extents[0]
        scale_y = (DeckboxConfig.WALL_HEIGHT + 0.2) / mesh_extents[1]
        mesh.vertices[:, 0] *= scale_x
        mesh.vertices[:, 1] *= scale_y
        
        # Rotation: 90° around X axis to stand up
        mesh.apply_transform(tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
        
        # Alignment
        box_min, box_max = box_mesh.bounds
        mesh_min, mesh_max = mesh.bounds
        
        tx = ((box_min[0] + box_max[0]) / 2.0) - ((mesh_min[0] + mesh_max[0]) / 2.0)
        ty = box_min[1] - mesh_min[1] + 0.05  # Micro-weld overlap (flush)
        tz = box_min[2] - mesh_min[2] + 4.0   # 4mm bottom frame offset
        
        mesh.apply_translation([tx, ty, tz])
        return trimesh.util.concatenate([box_mesh, mesh])

    def _process_lid_logo(self):
        """Generates the TCG logo plug and engravings for the deckbox lid mesh (in memory only)."""
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
                    logo_norm = 1.0 - (logo_img.astype(np.float64) / 255.0)  # Invert so text is engraved
                    # +0.05mm on surface to prevent Z-fighting with lid (coplanar faces)
                    Z_logo = (DeckboxConfig.PLUG_Z + 0.05) - logo_norm * engrave_depth
                    Z_logo = np.round(Z_logo, 3)
                    
                    lh, lw = logo_img.shape
                    lx = np.linspace(0, DeckboxConfig.PLUG_W + 0.2, lw)
                    ly = np.linspace(0, DeckboxConfig.PLUG_H + 0.2, lh)[::-1]
                    LX, LY = np.meshgrid(lx, ly)
                    
                    logo_mesh = create_solid_mesh(LX, LY, Z_logo, bottom_z=-0.05)
                    
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
        
        # Decimate lid mesh if needed (same logic as main mesh)
        if self.smart_decimate and len(lid_mesh.faces) > 200_000:
            self.progress.emit(98, "Optimizing Lid Mesh (Decimation)...")
            target = 100_000 + min(50_000, int((len(lid_mesh.faces) - 200_000) * 0.05))
            v_out, f_out = fast_simplification.simplify(
                lid_mesh.vertices.astype(np.float64),
                lid_mesh.faces.astype(np.int64),
                target_count=target, agg=6.0
            )
            lid_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
            trimesh.repair.fix_normals(lid_mesh)
            if not lid_mesh.is_watertight:
                trimesh.repair.fill_holes(lid_mesh)

        return lid_mesh

    def run(self):
        try:
            self._check_cancel()
            t_start_total = time.time()
            
            # Quote dei cambi colore da iniettare nel 3MF: in topo vengono
            # ricalcolate sulle terrazze, altrimenti valgono quelle Standard
            export_changes_z = self.color_changes_z
            export_slot_colors = None

            plate_mask = None
            if self.is_cover_mode and self.cover_preset:
                self.progress.emit(6, "📱 Composing artwork on plate...")
                if isinstance(self.img_filtered, np.ndarray) and len(self.img_filtered.shape) == 2:
                    img_rgb_src = cv2.cvtColor(self.img_filtered, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb_src = self.img_filtered
                plate_mask, res, pd = build_plate_raster(self.cover_preset, self.max_res_cap)
                h_p, w_p = plate_mask.shape
                art = compose_cover_art(img_rgb_src, w_p, h_p, res,
                                        user_scale=self.cover_scale,
                                        offset_x_mm=self.cover_off_x,
                                        offset_y_mm=self.cover_off_y)
                accents = self.spot_accents if self.cover_finish_spot else []
                palette, idx_map = classify_spot_pixels(art, accents,
                                                        coverage=self.spot_coverage)
                self.img_filtered = np.array(palette, dtype=np.uint8)[idx_map]
                export_slot_colors = ['#%02x%02x%02x' % tuple(c) for c in palette[1:]]
                # la plate segue la pipeline Topographic (terrazze + snap)
                self.is_topo_mode = True
                self.topo_colors = palette
                self.max_dim = max(pd['width'], pd['height'])

            if self.is_spot_mode:
                self.progress.emit(8, "🎯 Spot Color classification...")
                if isinstance(self.img_filtered, np.ndarray) and len(self.img_filtered.shape) == 2:
                    img_rgb_src = cv2.cvtColor(self.img_filtered, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb_src = self.img_filtered
                small = downsample_for_analysis(img_rgb_src, self.max_res_cap)
                palette, idx_map = classify_spot_pixels(small, self.spot_accents,
                                                        coverage=self.spot_coverage)
                self.img_filtered = np.array(palette, dtype=np.uint8)[idx_map]
                # Nei metadata 3MF finiscono i colori reali della palette (non i grigi)
                export_slot_colors = ['#%02x%02x%02x' % tuple(c) for c in palette[1:]]
                # Da qui in poi la pipeline coincide con la Topographic:
                # terrazze della palette, snap post-decimazione, cambi filamento
                self.is_topo_mode = True
                self.topo_colors = palette

            if self.is_topo_mode and self.topo_colors:
                self.progress.emit(10, "🏔 Starting Topographic Color Processing...")
                topo_z_heights = compute_topo_z_heights(
                    self.base_h, self.max_h, self.layer_height, len(self.topo_colors))
                export_changes_z = compute_topo_switch_z(topo_z_heights, self.layer_height)
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
                    layer_height=self.layer_height,
                    max_res_cap=self.max_res_cap,
                    mask=plate_mask
                )
                self._check_cancel()
                self.progress.emit(80, "Optimizing Topo Mesh...")
                
                # In modalità Topo la mesh è già completa, saltiamo la pipeline standard
                goto_export = True
            else:
                goto_export = False
                # --- 1. Image Preparation ---
                self.progress.emit(5, "Optimizing resolution for 3D mesh...")
                img = self._prepare_image_data(self.img_filtered)
                self._check_cancel()

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
                self._check_cancel()

                # 2mm border logic removed as requested (handled by CAD template)
            # --- 4. Mesh Assembly ---
            self.progress.emit(55, "Finalizing Geometry...")
            
            if self.is_deckbox_mode:
                self.progress.emit(91, "Assembling Deckbox (Wall Replacement)...")
                mesh = self._assemble_deckbox_mesh(mesh)
                self._check_cancel()

            # --- 5. Optimization (Decimation) ---
            self._check_cancel()
            if self.smart_decimate and len(mesh.faces) > 200_000:
                self.progress.emit(92, "Optimizing Mesh (Decimation)...")
                target = 100_000 + min(50_000, int((len(mesh.faces) - 200_000) * 0.05))
                v_out, f_out = fast_simplification.simplify(
                    mesh.vertices.astype(np.float64),
                    mesh.faces.astype(np.int64),
                    target_count=target, agg=6.0
                )
                mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)

                if self.is_topo_mode and self.topo_colors:
                    # Ri-snap delle quote alle terrazze: la decimazione inclina le
                    # pareti verticali e lo slicer mostrerebbe anelli di colori
                    # intermedi attorno ad ogni bordo (es. frangia rossa sul nero)
                    allowed_z = np.array([0.0] + topo_z_heights)
                    nearest = np.abs(mesh.vertices[:, [2]] - allowed_z[None, :]).argmin(axis=1)
                    mesh.vertices[:, 2] = allowed_z[nearest]

                trimesh.repair.fix_normals(mesh)
                if not mesh.is_watertight:
                    trimesh.repair.fill_holes(mesh)

            self._check_cancel()

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

                # Build lid mesh (in memory only)
                self.progress.emit(97, "Generating Lid...")
                lid_mesh = self._process_lid_logo()

                # --- Combine front + lid side by side (PLATE_GAP_MM gap) ---
                self.progress.emit(99, "Assembling full deckbox plate...")
                GAP_MM = DeckboxConfig.PLATE_GAP_MM
                full_path_3mf = None
                full_path_stl = None
                if lid_mesh is not None:
                    front_mesh_copy = mesh.copy()
                    lid_mesh_copy = lid_mesh.copy()

                    front_min_x = front_mesh_copy.bounds[0][0]
                    front_mesh_copy.apply_translation([-front_min_x, 0, 0])

                    front_max_x = front_mesh_copy.bounds[1][0]
                    lid_min_x = lid_mesh_copy.bounds[0][0]
                    lid_mesh_copy.apply_translation([front_max_x + GAP_MM - lid_min_x, 0, 0])

                    full_mesh = trimesh.util.concatenate([front_mesh_copy, lid_mesh_copy])

                    if self.output_path_3mf:
                        full_path_3mf = os.path.join(out_dir, f"full_deckbox_{self.source_image_name}.3mf")
                        export_3mf(full_mesh, full_path_3mf, self.color_changes_z)
                    if self.output_path:
                        full_path_stl = os.path.join(stl_dir, f"full_deckbox_{self.source_image_name}.stl")
                        full_mesh.export(full_path_stl)

                self.progress.emit(100, "Export completed!")
                self.finished_ok.emit(full_path_3mf or "", full_path_stl or "")
            else:
                if self.output_path:
                    mesh.export(self.output_path)
                if self.output_path_3mf:
                    export_3mf(mesh, self.output_path_3mf, export_changes_z,
                               slot_colors=export_slot_colors)

                if self.is_cover_mode and self.include_bumper and self.cover_preset:
                    self.progress.emit(99, "Generating TPU bumper (separate STL)...")
                    p = self.cover_preset
                    bumper = build_bumper(
                        p['width'], p['height'], p['thickness'], p['corner_radius'],
                        bottom_opening_w=p.get('bottom_opening', 45.0),
                        side_cutouts=[tuple(c) for c in p.get('side_cutouts', [])],
                        top_cutouts=[tuple(c) for c in p.get('top_cutouts', [])])
                    ref = self.output_path or self.output_path_3mf
                    bumper_path = os.path.splitext(ref)[0] + "_bumper_TPU.stl"
                    bumper.export(bumper_path)
                
                self.progress.emit(100, "Export completed!")
                self.finished_ok.emit(self.output_path or "", self.output_path_3mf or "")

            gc.collect()
            t_total = time.time() - t_start_total
            print(f"[Profiling] TOTAL REFACTORED TIME: {t_total:.2f}s")
            
        except Exception as e:
            self.finished_err.emit(str(e))

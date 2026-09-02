"""
Generazione parametrica della cover telefono a due pezzi:
- bumper perimetrale (da stampare in TPU, monocolore)
- back plate artistica (PLA multicolore, generata dalla pipeline heightmap)

Il bumper è un anello a bande sovrapposte (dal retro al fronte):
  A. cornice posteriore  - trattiene la plate come una cornice fotografica
  B. scanalatura         - il bordo della plate si incastra qui (il TPU flette)
  C. corpo telefono      - cavità a misura del telefono + gioco
  D. labbro frontale     - bordo rialzato che avvolge il fronte e protegge lo schermo

Tutte le misure sono in mm. Il fit reale va tarato stampando: i default dei
giochi (clearance) sono un punto di partenza ragionevole per TPU 95A.
"""

import json
import os

import numpy as np
import trimesh
from shapely.geometry import box as shapely_box

from .resources import asset_path


def _rounded_rect(w: float, h: float, r: float):
    """Poligono shapely a rettangolo arrotondato centrato nell'origine."""
    r = max(0.05, min(r, w / 2.0 - 0.01, h / 2.0 - 0.01))
    return shapely_box(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r).buffer(r, quad_segs=24)


def _ring(outer_poly, inner_poly, height: float, z0: float) -> trimesh.Trimesh:
    """Estrusione di una corona (outer - inner) traslata a quota z0."""
    m = trimesh.creation.extrude_polygon(outer_poly.difference(inner_poly), height)
    m.apply_translation([0.0, 0.0, z0])
    return m


def _cut_box(x0, x1, y0, y1, z0, z1) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])
    return b


def build_bumper(phone_w: float, phone_h: float, phone_t: float, corner_r: float,
                 wall_t: float = 2.0, clearance: float = 0.2,
                 back_lip_w: float = 1.8, back_lip_t: float = 1.0,
                 groove_depth: float = 1.0, groove_h: float = 1.4,
                 front_lip_w: float = 1.2, screen_guard_t: float = 1.0,
                 bottom_opening_w: float = 45.0,
                 side_cutouts=(), top_cutouts=()) -> trimesh.Trimesh:
    """Genera il bumper TPU. Origine al centro, Z=0 sul retro (lato plate).

    CONVENZIONE: lati e distanze sono espressi GUARDANDO IL RETRO del telefono
    (la stessa vista dell'artwork sulla plate). Nello spazio del modello la X
    risulta quindi specchiata: 'left' visto dal retro = +X del modello.

    side_cutouts: sequenza di (lato, distanza_dal_top, lunghezza) con lato in
    {'left','right'}, per lasciare scoperti bottoni/slider.
    top_cutouts: sequenza di (distanza_da_sinistra, lunghezza) di aperture sul
    bordo superiore (microfoni/IR).
    """
    z_a = back_lip_t                    # fine cornice posteriore
    z_b = z_a + groove_h                # fine scanalatura plate
    z_c = z_b + phone_t                 # fronte del telefono
    z_top = z_c + screen_guard_t        # cima del labbro

    cav_w = phone_w + 2 * clearance     # cavità telefono
    cav_h = phone_h + 2 * clearance
    cav_r = corner_r + clearance

    outer = _rounded_rect(cav_w + 2 * wall_t, cav_h + 2 * wall_t, cav_r + wall_t)
    inner_frame = _rounded_rect(cav_w - 2 * back_lip_w, cav_h - 2 * back_lip_w,
                                max(0.5, cav_r - back_lip_w))
    inner_groove = _rounded_rect(cav_w + 2 * groove_depth, cav_h + 2 * groove_depth,
                                 cav_r + groove_depth)
    inner_body = _rounded_rect(cav_w, cav_h, cav_r)
    inner_guard = _rounded_rect(cav_w - 2 * front_lip_w, cav_h - 2 * front_lip_w,
                                max(0.5, cav_r - front_lip_w))

    bands = [
        _ring(outer, inner_frame, back_lip_t, 0.0),        # A. cornice posteriore
        _ring(outer, inner_groove, groove_h, z_a),         # B. scanalatura
        _ring(outer, inner_body, phone_t, z_b),            # C. corpo
        _ring(outer, inner_guard, screen_guard_t, z_c),    # D. labbro frontale
    ]
    bumper = trimesh.boolean.union(bands, engine='manifold')

    # Ritagli: aperture passanti dal piano del telefono in su, la cornice
    # e la scanalatura restano integre così la plate è trattenuta a 360°
    cuts = []
    half_w = cav_w / 2 + wall_t
    half_h = cav_h / 2 + wall_t
    if bottom_opening_w > 0:
        cuts.append(_cut_box(-bottom_opening_w / 2, bottom_opening_w / 2,
                             -half_h - 1, -(cav_h / 2 - 2), z_b, z_top + 1))
    for side, from_top, length in side_cutouts:
        y1 = cav_h / 2 - from_top
        y0 = y1 - length
        # vista dal retro: 'left' cade sul lato +X del modello
        if side == 'left':
            cuts.append(_cut_box(cav_w / 2 - 2, half_w + 1, y0, y1, z_b, z_top + 1))
        else:
            cuts.append(_cut_box(-half_w - 1, -(cav_w / 2 - 2), y0, y1, z_b, z_top + 1))
    for from_left, length in top_cutouts:
        x1 = cav_w / 2 - from_left          # specchiatura back-view
        x0 = x1 - length
        cuts.append(_cut_box(x0, x1, cav_h / 2 - 2, half_h + 1, z_b, z_top + 1))
    if cuts:
        bumper = trimesh.boolean.difference(
            [bumper, trimesh.boolean.union(cuts, engine='manifold')], engine='manifold')

    return bumper


def compute_plate_dims(phone_w: float, phone_h: float, corner_r: float,
                       clearance: float = 0.2, groove_depth: float = 1.0,
                       groove_h: float = 1.4, plate_fit_clearance: float = 0.3) -> dict:
    """Dimensioni della back plate che si incastra nella scanalatura del bumper.
    La plate è più larga della cavità telefono (entra nel groove) e più sottile
    dell'altezza del groove, così scatta in sede senza forzare."""
    extra = clearance + groove_depth - plate_fit_clearance
    return {
        'width': round(phone_w + 2 * extra, 2),
        'height': round(phone_h + 2 * extra, 2),
        'corner_radius': round(corner_r + extra, 2),
        'max_thickness': round(groove_h - 0.2, 2),
    }


def load_phone_presets() -> dict:
    """Carica i preset telefono dal JSON in assets (misure indicative:
    la finestra fotocamera va verificata col righello sul telefono reale)."""
    path = asset_path("phone_presets.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RETROFIT: scavo sede plate in una cover esistente (STL di terze parti)
# ---------------------------------------------------------------------------

def _extrude_xz(poly, y_from: float, y_to: float) -> trimesh.Trimesh:
    """Estrude un poligono shapely (coordinate = piano X-Z del case) lungo Y."""
    m = trimesh.creation.extrude_polygon(poly, y_to - y_from)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, y_to, 0])
    return m


def carve_plate_recess(case_mesh: trimesh.Trimesh,
                       cavity_xz: tuple, corner_r: float,
                       wall_y: tuple,
                       keep_zones=(),
                       frame_w: float = 3.0, lip_t: float = 0.8,
                       groove_w: float = 1.2, plate_clearance: float = 0.3):
    """Scava in una cover esistente la sede per la back plate artistica.

    - finestra passante nella parete posteriore, arretrata di frame_w dal
      bordo cavità (resta una cornice estetica)
    - tasca sottosquadro dall'interno, più larga di groove_w: la plate si
      infila dal lato telefono e resta trattenuta dal labbro esterno (lip_t)
    - keep_zones: poligoni shapely (piano X-Z) da NON scavare, es. il blocco
      fotocamera originale con i suoi dettagli

    cavity_xz = (x0, z0, x1, z1) della cavità telefono; wall_y = (y_interno,
    y_esterno) della parete posteriore.
    Ritorna (case_scavato, sagoma_plate_shapely, spessore_plate_max).
    """
    x0, z0, x1, z1 = cavity_xz
    y_in, y_out = wall_y

    cavity = shapely_box(x0 + corner_r, z0 + corner_r,
                         x1 - corner_r, z1 - corner_r).buffer(corner_r, quad_segs=24)
    opening = cavity.buffer(-frame_w, quad_segs=24)
    for zone in keep_zones:
        opening = opening.difference(zone)

    pocket = opening.buffer(groove_w, quad_segs=8)
    window_cut = _extrude_xz(opening, y_in - 1.0, y_out + 1.0)
    pocket_cut = _extrude_xz(pocket, y_in - 1.0, y_out - lip_t)

    carved = trimesh.boolean.difference(
        [case_mesh, trimesh.boolean.union([window_cut, pocket_cut], engine='manifold')],
        engine='manifold')

    plate_outline = opening.buffer(groove_w - plate_clearance, quad_segs=8)
    plate_thickness = round((y_out - lip_t) - y_in - 0.1, 2)
    return carved, plate_outline, plate_thickness


# ---------------------------------------------------------------------------
# COMPOSIZIONE ARTWORK SU PLATE (modalità Phone Cover)
# ---------------------------------------------------------------------------

def build_plate_raster(preset: dict, max_res_cap: int = 1200):
    """Costruisce la maschera raster della back plate dal preset telefono:
    sagoma arrotondata (dimensioni da compute_plate_dims) meno i fori camera.
    Ritorna (mask HxW, res_mm_per_px, plate_dims)."""
    from .mesh_utils import rounded_rect_mask

    pd = compute_plate_dims(preset['width'], preset['height'], preset['corner_radius'])
    W, H = pd['width'], pd['height']
    res = max(W, H) / float(max_res_cap)
    w_px, h_px = int(round(W / res)), int(round(H / res))
    # i fori camera del preset sono riferiti al corpo telefono: la plate
    # sborda di 'extra' mm su ogni lato, quindi vanno traslati
    extra = (W - preset['width']) / 2.0
    holes = [(int((extra + h_['x']) / res), int((extra + h_['y']) / res),
              int(h_['w'] / res), int(h_['h'] / res), h_['r'] / res)
             for h_ in preset.get('camera_holes', [])]
    mask = rounded_rect_mask(h_px, w_px, pd['corner_radius'] / res, holes=holes)
    return mask, res, pd


def compose_cover_art(image_rgb: np.ndarray, plate_w_px: int, plate_h_px: int,
                      res: float, user_scale: float = 1.0,
                      offset_x_mm: float = 0.0, offset_y_mm: float = 0.0,
                      fill_rgb=(245, 245, 245)) -> np.ndarray:
    """Campiona l'immagine sul raster della plate ('cover fill' + zoom/offset).

    - user_scale = 1.0: l'immagine copre esattamente la plate (riempimento);
      valori maggiori zoomano dentro.
    - offset_x/y_mm: spostano l'IMMAGINE rispetto alla plate (x>0 destra,
      y>0 giù, nella vista dal retro). Le zone scoperte diventano fill_rgb.
    """
    import cv2

    h_img, w_img = image_rgb.shape[:2]
    s_fill = max(plate_w_px / w_img, plate_h_px / h_img)
    s = s_fill * max(0.05, user_scale)

    src_w, src_h = plate_w_px / s, plate_h_px / s
    src_cx = w_img / 2.0 - (offset_x_mm / res) / s
    src_cy = h_img / 2.0 - (offset_y_mm / res) / s
    x0, y0 = src_cx - src_w / 2.0, src_cy - src_h / 2.0
    x1, y1 = x0 + src_w, y0 + src_h

    # padding con colore di riempimento dove il ritaglio esce dall'immagine
    pad_l = int(np.ceil(max(0, -x0)))
    pad_t = int(np.ceil(max(0, -y0)))
    pad_r = int(np.ceil(max(0, x1 - w_img)))
    pad_b = int(np.ceil(max(0, y1 - h_img)))
    if pad_l or pad_t or pad_r or pad_b:
        image_rgb = cv2.copyMakeBorder(image_rgb, pad_t, pad_b, pad_l, pad_r,
                                       cv2.BORDER_CONSTANT, value=fill_rgb)
        x0 += pad_l; x1 += pad_l
        y0 += pad_t; y1 += pad_t

    crop = image_rgb[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(crop, (plate_w_px, plate_h_px), interpolation=interp)


def compose_plate_art(image_rgb, preset: dict, mask, res: float,
                      user_scale: float = 1.0, off_x: float = 0.0, off_y: float = 0.0,
                      avoid_camera: bool = True, camera_margin_mm: float = 3.0,
                      fill_rgb=(245, 245, 245)):
    """Compone l'artwork sul raster della plate. Con avoid_camera=True l'immagine
    viene adattata SOLO alla zona sotto il blocco fotocamere (+margine): la
    fascia superiore resta del colore base, con i soli fori camera."""
    h_p, w_p = mask.shape
    y_start = 0
    holes = preset.get('camera_holes', [])
    if avoid_camera and holes:
        pd = compute_plate_dims(preset['width'], preset['height'], preset['corner_radius'])
        extra = (pd['width'] - preset['width']) / 2.0
        zone_bottom = max(h_['y'] + h_['h'] for h_ in holes) + camera_margin_mm
        y_start = min(int(round((extra + zone_bottom) / res)), h_p - 10)

    art = np.full((h_p, w_p, 3), fill_rgb, dtype=np.uint8)
    art[y_start:] = compose_cover_art(image_rgb, w_p, h_p - y_start, res,
                                      user_scale=user_scale,
                                      offset_x_mm=off_x, offset_y_mm=off_y,
                                      fill_rgb=fill_rgb)
    return art


def build_case_plate_raster(preset: dict, max_res_cap: int = 1200):
    """Raster della plate per i preset con 'case_plate' (sede ricavata da una
    cover reale): la sagoma è il poligono misurato, fotocamera già esclusa.
    Ritorna (mask, res_mm_per_px, dims) come build_plate_raster."""
    import cv2

    cp = preset['case_plate']
    W, H = cp['width'], cp['height']
    res = max(W, H) / float(max_res_cap)
    w_px, h_px = int(round(W / res)), int(round(H / res))
    pts = np.round(np.array(cp['outline'], dtype=np.float64) / res).astype(np.int32)
    mask_u8 = np.zeros((h_px, w_px), dtype=np.uint8)
    cv2.fillPoly(mask_u8, [pts], 255)
    dims = {'width': W, 'height': H, 'max_thickness': cp['thickness']}
    return mask_u8 > 0, res, dims

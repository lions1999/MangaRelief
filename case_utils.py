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

from utils import resource_path


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
    path = resource_path(os.path.join("assets", "phone_presets.json"))
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

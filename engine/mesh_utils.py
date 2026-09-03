import os
import io
import zipfile
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import (median_filter, label, binary_erosion,
                           distance_transform_edt)

from .config import SLOT_COLORS_3MF
from .color_utils import rgb_to_lab, CHROMA_MATCH_WEIGHT

def create_solid_mesh(X, Y, Z, bottom_z=0.0, mask=None):
    """
    Generates a solid watertight mesh from X, Y, Z meshgrids.
    Seals the bottom and the four sides.
    mask (opzionale): array booleano HxW, True = incluso. Genera superficie solo
    per le celle interamente incluse e sigilla con pareti verticali tutti i
    bordi, esterni e fori interni (es. sagoma cover + foro fotocamera).
    """
    if mask is not None:
        return _create_masked_solid_mesh(X, Y, Z, bottom_z, mask)

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
    
    # Left edge (normale esterna -X: winding corretto, prima era invertito)
    v1, v2 = idx[:-1, 0], idx[1:, 0]
    left_sides = np.vstack((np.column_stack((v2, v1, v1 + offset)), np.column_stack((v2 + offset, v2, v1 + offset))))

    # Right edge (normale esterna +X: winding corretto, prima era invertito)
    v1, v2 = idx[:-1, -1], idx[1:, -1]
    right_sides = np.vstack((np.column_stack((v1, v2, v1 + offset)), np.column_stack((v2, v2 + offset, v1 + offset))))
    
    all_vertices = np.vstack((vertices_top, vertices_bottom))
    all_faces = np.vstack((faces_top, faces_bottom, top_sides, bot_sides, left_sides, right_sides))
    
    return trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)


def _create_masked_solid_mesh(X, Y, Z, bottom_z, mask):
    """Variante di create_solid_mesh limitata a una sagoma arbitraria.
    Una cella entra nella mesh solo se tutti e 4 i suoi vertici sono nel mask;
    ogni lato di cella confinante con l'esterno (o con un foro) genera una
    parete verticale con lo stesso winding delle 4 pareti del caso pieno."""
    h, w = Z.shape
    offset = w * h
    idx = np.arange(w * h).reshape((h, w))

    vertices_top = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    vertices_bottom = np.column_stack((X.ravel(), Y.ravel(),
                                       np.full(w * h, bottom_z, dtype=Z.dtype)))

    inc = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]

    tl = idx[:-1, :-1][inc]
    tr = idx[:-1, 1:][inc]
    bl = idx[1:, :-1][inc]
    br = idx[1:, 1:][inc]
    faces_top = np.vstack((np.column_stack((bl, tr, tl)), np.column_stack((br, tr, bl))))
    faces_bottom = np.vstack((np.column_stack((tl + offset, tr + offset, bl + offset)),
                              np.column_stack((bl + offset, tr + offset, br + offset))))

    false_row = np.zeros((1, inc.shape[1]), dtype=bool)
    false_col = np.zeros((inc.shape[0], 1), dtype=bool)
    walls = []

    # Nord: cella inclusa senza vicina sopra
    r, c = np.nonzero(inc & ~np.vstack((false_row, inc[:-1])))
    v1, v2 = idx[r, c], idx[r, c + 1]
    walls += [np.column_stack((v1, v2, v1 + offset)),
              np.column_stack((v2, v2 + offset, v1 + offset))]
    # Sud: cella inclusa senza vicina sotto
    r, c = np.nonzero(inc & ~np.vstack((inc[1:], false_row)))
    v1, v2 = idx[r + 1, c], idx[r + 1, c + 1]
    walls += [np.column_stack((v2, v1, v1 + offset)),
              np.column_stack((v2 + offset, v2, v1 + offset))]
    # Ovest: cella inclusa senza vicina a sinistra (normale esterna -X)
    r, c = np.nonzero(inc & ~np.hstack((false_col, inc[:, :-1])))
    v1, v2 = idx[r, c], idx[r + 1, c]
    walls += [np.column_stack((v2, v1, v1 + offset)),
              np.column_stack((v2 + offset, v2, v1 + offset))]
    # Est: cella inclusa senza vicina a destra (normale esterna +X)
    r, c = np.nonzero(inc & ~np.hstack((inc[:, 1:], false_col)))
    v1, v2 = idx[r, c + 1], idx[r + 1, c + 1]
    walls += [np.column_stack((v1, v2, v1 + offset)),
              np.column_stack((v2, v2 + offset, v1 + offset))]

    all_vertices = np.vstack((vertices_top, vertices_bottom))
    all_faces = np.vstack([faces_top, faces_bottom] + walls)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def rounded_rect_mask(h: int, w: int, radius_px: float,
                      holes=None) -> np.ndarray:
    """Maschera booleana HxW a rettangolo con angoli arrotondati, meno una
    lista di fori rounded-rect. holes = [(x0, y0, w_px, h_px, r_px), ...] in
    coordinate pixel (un cerchio è il caso w=h=2r). È la sagoma della back
    plate per cover telefono: fori fotocamera/flash inclusi."""
    yy, xx = np.mgrid[0:h, 0:w]

    def _rrect(x0, y0, x1, y1, r):
        r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
        inside_x = (xx >= x0 + r) & (xx <= x1 - r) & (yy >= y0) & (yy <= y1)
        inside_y = (xx >= x0) & (xx <= x1) & (yy >= y0 + r) & (yy <= y1 - r)
        m = inside_x | inside_y
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            m |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        return m

    m = _rrect(0, 0, w - 1, h - 1, radius_px)
    for hx, hy, hw_, hh_, hr in (holes or []):
        m &= ~_rrect(hx, hy, hx + hw_, hy + hh_, hr)
    return m


def compute_topo_z_heights(base_z: float, total_z: float, layer_height: float, n_colors: int) -> list:
    """
    Compute discrete, layer-snapped Z heights for each colour in Topographic mode.
    The first height corresponds to the base layer; subsequent heights are evenly
    distributed across the remaining print height.
    """
    base_layers  = int(round(base_z    / layer_height))
    total_layers = int(round(total_z   / layer_height))
    remaining    = total_layers - base_layers

    z_heights = [round(base_layers * layer_height, 3)]
    if n_colors > 1 and remaining > 0:
        base_dist          = remaining // (n_colors - 1)
        extra              = remaining  % (n_colors - 1)
        layers_per_color   = [base_dist] * (n_colors - 1)
        for i in range(extra):
            layers_per_color[i] += 1
        current_l = base_layers
        for lc in layers_per_color:
            current_l += lc
            z_heights.append(round(current_l * layer_height, 3))
    else:
        z_heights = [round(base_z, 3)] * n_colors
    return z_heights


def compute_topo_switch_z(z_heights: list, layer_height: float) -> list:
    """
    Quote dei cambi filamento per la modalità Topographic.
    Il cambio verso il colore i va inserito al primo layer SOPRA la terrazza
    del colore precedente (z_heights[i-1] + layer_height), NON al top della
    terrazza del colore stesso: altrimenti ogni colore riceve un solo layer
    utile e i layer sbagliati restano nascosti sotto le superfici.
    """
    return [round(z_heights[i - 1] + layer_height, 3) for i in range(1, len(z_heights))]


def standard_switch_z(levels: list, layer_height: float) -> list:
    """Quote dei cambi filamento per la modalita' Standard, dalle terrazze REALI.

    `levels` sono le quote distinte della heightmap prodotta (carta in fondo,
    inchiostro in cima). Stessa regola del topo: il colore i entra al primo
    layer sopra la terrazza i-1. In piu' la terrazza viene prima portata al
    layer che lo slicer stampera' davvero (campiona a meta' layer, quindi
    round), cosi' una quota come 1,512 conta come 1,6 e non come 1,4.

    Prima di questa funzione la Standard esportava le quote *in cima* a ogni
    terrazza: a 2 colori il nero entrava a 2,40 e copriva solo l'ultimo layer
    di una colonna alta 1,40 — il 14% della parete — e ogni forma nera usciva
    con un bordo bianco tutt'intorno.
    """
    lh = float(layer_height)
    snapped = [round(round(float(z) / lh) * lh, 3) for z in levels]
    return [round(snapped[i - 1] + lh, 3) for i in range(1, len(snapped))]


def process_mesh_topo(image_rgb: np.ndarray, sorted_colors_rgb: list,
                      base_z: float = 1.0, total_z: float = 2.4,
                      max_dim: float = 100.0, layer_height: float = 0.2,
                      max_res_cap: int = 800, mask=None, min_feature_mm: float = 0.5):
    """Genera una mesh a terrazze basata sui colori forniti, quantizzata sui layer di stampa.
    mask (opzionale): sagoma booleana della stessa shape dell'immagine (es. plate
    cover con fori camera); implica che l'immagine sia già alla risoluzione finale.
    min_feature_mm: soglia di pulizia per dettagli/frange (default 0.5mm, il
    diametro minimo stampabile in rilievo). Nelle incisioni un solco è assenza
    di materiale, non una parete: la soglia può scendere fino a ~0.2mm."""
    # Pre-scaling al cap del selettore Mesh Quality (Draft 800 / Standard 1200 / Ultra 1600)
    h, w = image_rgb.shape[:2]
    max_size = int(max_res_cap)
    if mask is not None:
        assert mask.shape == (h, w), "mask e immagine devono avere la stessa shape"
    elif max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.Resampling.LANCZOS)
        image_rgb = np.array(img_pil)
        h, w = image_rgb.shape[:2]

    n_colors = len(sorted_colors_rgb)

    # --- LAYER QUANTISATION ---
    exact_z_heights = compute_topo_z_heights(base_z, total_z, layer_height, n_colors)

    # Mappa pixel ai colori tramite cKDTree in spazio Lab percettivo: con la
    # distanza RGB i grigi di anti-aliasing venivano assegnati ai rossi scuri,
    # facendo "sbucare" colori dai layer sbagliati
    tree = cKDTree(rgb_to_lab(np.array(sorted_colors_rgb, dtype=np.uint8),
                              chroma_weight=CHROMA_MATCH_WEIGHT))
    pixels_flat = rgb_to_lab(image_rgb, chroma_weight=CHROMA_MATCH_WEIGHT).reshape(-1, 3)
    _, indices = tree.query(pixels_flat)
    indices = indices.reshape(h, w)

    # Applica un filtro mediana per "compattare" le zone di colore e rimuovere il rumore
    # (pixel isolati). Alle risoluzioni alte il kernel scende a 3 per non mangiare
    # le linee fini (a 800px un kernel 5 cancella dettagli sotto ~1.5mm di stampa)
    indices = median_filter(indices, size=5 if max(h, w) <= 800 else 3)

    # --- Pulizia per dimensione minima stampabile (~ugello 0.4mm) ---
    # 1) isole più piccole dell'area di un punto da 0.5mm -> riassegnate al colore
    #    circostante (pulviscolo, per tutti i colori)
    # 2) solo per i colori saturi: componenti senza "nucleo", cioè più strette di
    #    0.5mm ovunque (le frange di ringing JPEG lungo i bordi neri) -> riassegnate.
    #    I colori neutri sono esclusi per preservare le linee fini bianche/nere.
    pitch = max_dim / max(h, w)  # mm per pixel
    radius_px = max(1, int((min_feature_mm / 2.0) / pitch))
    min_area_px = max(2, int(np.pi * ((min_feature_mm / 2.0) / pitch) ** 2))
    colors_lab = rgb_to_lab(np.array(sorted_colors_rgb, dtype=np.uint8))
    is_chromatic = np.max(np.abs(colors_lab[:, 1:] - 128.0), axis=1) > 12.0

    structure = np.ones((3, 3), dtype=bool)
    remove_mask = np.zeros((h, w), dtype=bool)
    for i in range(n_colors):
        mask_i = (indices == i)
        labels, n_labels = label(mask_i, structure=structure)
        if n_labels == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
        drop = sizes < min_area_px
        if is_chromatic[i]:
            core = binary_erosion(mask_i, structure=structure, iterations=radius_px)
            has_core = np.zeros(n_labels + 1, dtype=bool)
            has_core[np.unique(labels[core])] = True
            drop |= ~has_core
        drop[0] = False
        if drop.any():
            remove_mask |= drop[labels]
    if remove_mask.any() and not remove_mask.all():
        nearest = distance_transform_edt(remove_mask, return_distances=False,
                                         return_indices=True)
        indices = indices[nearest[0], nearest[1]]

    # Costruisci heightmap discreta usando le altezze quantizzate
    Z = np.zeros((h, w), dtype=np.float32)
    for i in range(n_colors):
        color_mask = (indices == i)   # NB: non chiamarla 'mask', ombreggerebbe il parametro sagoma
        Z[color_mask] = exact_z_heights[i]

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
    mesh = create_solid_mesh(X, Y, Z, bottom_z=0.0, mask=mask)
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

def export_3mf(mesh, output_path_3mf, color_changes_z, slot_colors=None):
    """
    Exports a 3MF using trimesh, then injects Bambu Studio specific XMLs
    for color changing at specific Z heights.
    slot_colors: lista opzionale di hex (dal 2° filamento in poi) per
    sovrascrivere i grigi di default, es. i colori reali della palette Spot.
    """
    # 1. Generate base 3MF with trimesh in memory
    src_buf = io.BytesIO()
    mesh.export(src_buf, file_type='3mf')
    src_buf.seek(0)

    # 2. Build custom_gcode_per_layer.xml layer nodes
    # Scarta i livelli non usati (z=0 nelle modalità 2/3 colori) e i duplicati,
    # altrimenti Bambu Studio riceve cambi colore fantasma al layer 0
    slot_colors = slot_colors or SLOT_COLORS_3MF
    layer_nodes = ""
    valid_z = sorted({round(z, 4) for z in color_changes_z if z > 0})
    for i, z in enumerate(valid_z):
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

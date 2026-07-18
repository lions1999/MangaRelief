import os
import io
import zipfile
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import (median_filter, label, binary_erosion,
                           distance_transform_edt)

from config import SLOT_COLORS_3MF
from color_utils import rgb_to_lab, CHROMA_MATCH_WEIGHT

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


def process_mesh_topo(image_rgb: np.ndarray, sorted_colors_rgb: list,
                      base_z: float = 1.0, total_z: float = 2.4,
                      max_dim: float = 100.0, layer_height: float = 0.2,
                      max_res_cap: int = 800):
    """Genera una mesh a terrazze basata sui colori forniti, quantizzata sui layer di stampa."""
    # Pre-scaling al cap del selettore Mesh Quality (Draft 800 / Standard 1200 / Ultra 1600)
    h, w = image_rgb.shape[:2]
    max_size = int(max_res_cap)
    if max(h, w) > max_size:
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
    MIN_FEATURE_MM = 0.5
    pitch = max_dim / max(h, w)  # mm per pixel
    radius_px = max(1, int((MIN_FEATURE_MM / 2.0) / pitch))
    min_area_px = max(2, int(np.pi * ((MIN_FEATURE_MM / 2.0) / pitch) ** 2))
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
        mask = (indices == i)
        Z[mask] = exact_z_heights[i]

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

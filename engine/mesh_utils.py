import os
import io
import json
import uuid as _uuid
import zipfile
from datetime import date
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
# .3MF EXPORT — un progetto Bambu Studio, non un 3MF qualunque
#
# Bambu Studio decide se un 3MF e' "suo" da un solo metadato dentro
# 3dmodel.model: `BambuStudio:3mfVersion`. Il suo importatore accende
# `m_is_bbl_3mf` quando lo trova, e se non lo trova carica la geometria e butta
# via tutto il resto — compreso il custom_gcode_per_layer.xml con i cambi
# colore, che quindi non veniva nemmeno aperto. E' il motivo per cui ogni file
# prodotto qui e' sempre arrivato allo slicer come "solo geometria".
#
# Quello che serve non sono le impostazioni di stampa (project_settings.config
# puo' restare vuoto), e' lo scheletro:
#
#   3D/3dmodel.model            guscio: metadati, un <object> con <components>
#   3D/Objects/object_1.model   la geometria vera, referenziata da p:path
#   3D/_rels/3dmodel.model.rels il collegamento fra i due
#   Metadata/model_settings.config    oggetto, plate, istanza
#   Metadata/project_settings.config  vuoto: non fingiamo di sapere la stampante
#
# Ricostruito confrontando entry per entry un file generato da qui e poi
# salvato da Bambu Studio 02.08.02.61 — non da documentazione, che per questa
# parte del formato non esiste.
# ---------------------------------------------------------------------------

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_PROD_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
_BBS_NS = "http://schemas.bambulab.com/package/2021"

# Il generatore dichiarato, e qui non c'e' liberta' di scelta: l'importatore
# ricava da questa stringa la versione del programma che ha scritto il file, e
# se non ci riesce non legge la configurazione — cioe' butta via i cambi
# colore. Misurato: lo stesso identico file con "MangaRelief-1.0.0" viene
# rifiutato, con "BambuStudio-<versione>" viene accettato.
#
# Non e' una firma sul modello, e' una dichiarazione di compatibilita' di
# formato, come lo user agent di un browser: il nostro nome resta ovunque
# altro nel file.
#
# La versione non va aggiornata a ogni uscita di Bambu Studio, e inseguirla
# sarebbe anzi la scelta sbagliata. Il confronto fra la versione del file e
# quella del programma nel loro codice e' commentato:
#
#     /* if (file_version.maj() > app_version.maj())
#            dont_load_config = true;*/
#
# quindi oggi il numero non viene paragonato a niente — conta solo che esista e
# si lasci interpretare. E se un domani riattivassero quella riga, il confronto
# e' sul numero maggiore: un file che dichiara l'ultima versione verrebbe
# rifiutato da chi non ha ancora aggiornato, mentre uno basso passa sempre.
# L'unica direzione sicura e' verso il basso, e una volta sola.
_APPLICATION = "BambuStudio-02.00.00.00"

# Le cinque chiavi di project_settings.config che servono, non una di piu'.
# Trovate per bisezione contro Bambu Studio 02.08.02.61: con le sole tre di
# identita' il file non si apre, e fra le impostazioni quella indispensabile e'
# `nozzle_diameter` — lo stesso file con `printer_technology` al suo posto
# viene rifiutato. Un file vuoto o `{}` non basta: rompe perfino un progetto
# scritto da Bambu Studio stesso.
#
# Che si debba dichiarare un ugello e' spiacevole — 0.4 e' il valore di
# fabbrica di tutte le Bambu, ma non di tutti gli utenti — ed e' anche il
# motivo dell'avviso sui "preset personalizzati" che compare all'apertura:
# Bambu trova un'impostazione di stampante senza un profilo a cui appartenga.
# Non e' un avviso falso, e non si toglie senza rinunciare ai cambi colore.
# Stessa ragione: un numero fermo, dentro il 2.x.
_PROJECT_VERSION = "02.00.00.00"
_DEFAULT_NOZZLE = "0.4"

# Quello che dichiariamo qui non si aggiunge al profilo di chi apre il file:
# lo sostituisce. Bambu costruisce un processo nuovo, intitolato al nostro
# file, e ogni chiave che non nominiamo prende il valore di fabbrica — non
# quello del profilo scelto. Misurato confrontando due file affettati dalla
# stessa persona: 268 impostazioni su 557 diverse, fra cui l'ugello a 200 °C
# invece di 220 e il muro esterno a 60 mm/s invece di 200.
#
# Non c'e' modo di dire "tieni il tuo processo": un progetto ne definisce uno
# per forza. Quindi si dichiara il minimo che riguarda *questo* oggetto e si
# dice a chi scarica di scegliere il proprio profilo — un menu a tendina, che
# rimette a posto tutte e 268 in un colpo.
#
# skirt_loops a zero perche' e' l'unico valore di fabbrica che si vede a occhio
# nudo: un giro di perimetro sul primo layer, in filamento 1, che su una lastra
# larga non serve a niente e sembra un bordo del disegno.
_SKIRT_LOOPS = "0"

_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
</Types>"""

_ROOT_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

_MODEL_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""

_OBJECT_TPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="{core}" xmlns:BambuStudio="{bbs}" xmlns:p="{prod}" requiredextensions="p">
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <resources>
  <object id="1" p:UUID="{uuid_obj}" type="model">
   {mesh_xml}
  </object>
 </resources>
 <build/>
</model>"""

_ROOT_TPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="{core}" xmlns:BambuStudio="{bbs}" xmlns:p="{prod}" requiredextensions="p">
 <metadata name="Application">{application}</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="CreationDate">{today}</metadata>
 <metadata name="ModificationDate">{today}</metadata>
 <metadata name="Title">{title}</metadata>
 <resources>
  <object id="2" p:UUID="{uuid_comp}" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{uuid_ref}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>
 </resources>
 <build p:UUID="{uuid_build}">
  <item objectid="2" p:UUID="{uuid_item}" transform="1 0 0 0 1 0 0 0 1 {dx} {dy} {dz}" printable="1"/>
 </build>
</model>"""

# La mesh sta nelle proprie coordinate (angolo all'origine) e l'item non la
# sposta. Bambu al salvataggio la ricentra e mette la posizione qui, ma quella
# posizione dipende dal piatto della stampante scelta — che con
# project_settings.config vuoto non sappiamo. Meglio un oggetto in un angolo,
# che si centra con un tasto, di una posizione inventata.

_MODEL_SETTINGS_TPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="{name}"/>
    <metadata key="extruder" value="1"/>
    <metadata face_count="{faces}"/>
    <part id="1" subtype="normal_part" uuid="{uuid_part}">
      <metadata key="name" value="{name}"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <metadata key="source_file" value="{source}"/>
      <metadata key="source_object_id" value="0"/>
      <metadata key="source_volume_id" value="0"/>
      <metadata key="source_offset_x" value="{off_x}"/>
      <metadata key="source_offset_y" value="{off_y}"/>
      <metadata key="source_offset_z" value="{off_z}"/>
      <mesh_stat face_count="{faces}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <model_instance>
      <metadata key="object_id" value="2"/>
      <metadata key="instance_id" value="0"/>
      <metadata key="identify_id" value="1"/>
    </model_instance>
  </plate>
  <assemble>
   <assemble_item object_id="2" instance_id="0" transform="1 0 0 0 1 0 0 0 1 {dx} {dy} {dz}" offset="0 0 0" />
   <assemble_item object_id="2" volume_id="0" transform="1 0 0 0 1 0 0 0 1 0 0 0" />
  </assemble>
</config>"""

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

def export_3mf(mesh, output_path_3mf, color_changes_z, slot_colors=None,
               object_name="geometry_0", palette_hex=None, layer_height=0.2):
    """Scrive un 3MF che Bambu Studio apre come progetto, non come geometria.

    La geometria la serializza trimesh — non c'e' ragione di riscrivere un
    serializzatore di mesh — ma il pacchetto attorno lo costruiamo noi, perche'
    quello di trimesh e' un 3MF generico e Bambu Studio, non trovandoci il
    proprio metadato di versione, scarta tutto tranne i triangoli.

    `color_changes_z`: le quote a cui cambiare bobina. Quelle a zero e i
    duplicati si scartano — un cambio colore al layer 0 e' un cambio fantasma.
    `slot_colors`: gli esadecimali dal secondo filamento in poi.
    `palette_hex`: la palette intera, dal primo filamento — quello della base —
    all'ultimo. Ne esce l'elenco dei filamenti che lo slicer mostra, e deve
    contenere quelli che si stampano davvero e nessun altro: una lista piu'
    lunga della palette fa comparire bobine che non esistono.
    """
    # La posa, copiata da come la scrive Bambu Studio stesso: la mesh centrata
    # sulla propria origine e la posizione tutta nella traslazione dell'item.
    #
    # Aritmeticamente sarebbe lo stesso lasciare la mesh dov'e' e spostare
    # l'item di quel tanto — il centro cade allo stesso punto, verificato al
    # centesimo — ma all'apertura l'oggetto compariva comunque spostato. Non
    # sappiamo che cosa faccia il loro caricatore con l'origine di un oggetto
    # non centrato, e non serve saperlo: rispecchiare il file che scrive lui
    # toglie la domanda.
    #
    # 128 e' il centro del piatto da 256 mm di A1, P1 e X1. Su un piatto
    # diverso l'oggetto andra' spostato, e non e' una cosa che possiamo sapere
    # da qui: dipende dalla stampante di chi apre il file.
    basso, alto = mesh.bounds
    centro = (basso + alto) / 2.0
    mesh = mesh.copy()
    # In Z si centra come gli altri due assi, e l'item lo rialza di mezza
    # altezza: cosi' la base torna appoggiata al piatto.
    mesh.apply_translation(-centro)
    dx, dy, dz = 128.0, 128.0, round(float(centro[2] - basso[2]), 6)

    # 1. La geometria, da trimesh, in memoria.
    src_buf = io.BytesIO()
    mesh.export(src_buf, file_type='3mf')
    src_buf.seek(0)
    with zipfile.ZipFile(src_buf, 'r') as src_zip:
        modello = src_zip.read('3D/3dmodel.model').decode('utf-8')
    # Solo il blocco <mesh>: vertici e triangoli, senza il pacchetto attorno.
    mesh_xml = modello[modello.index('<mesh>'):modello.index('</mesh>') + len('</mesh>')]

    # 2. I cambi colore, nella forma che Bambu si aspetta.
    slot_colors = slot_colors or SLOT_COLORS_3MF
    layer_nodes = ""
    valid_z = sorted({round(z, 4) for z in color_changes_z if z > 0})
    for i, z in enumerate(valid_z):
        extruder = i + 2
        color = slot_colors[i] if i < len(slot_colors) else "#000000"
        layer_nodes += (
            f'<layer top_z="{round(z, 4)}" type="2" extruder="{extruder}" '
            f'color="{color}" extra="" gcode="tool_change"/>\n'
        )

    # 3. Il pacchetto. Gli UUID sono nuovi a ogni esportazione: identificano
    #    questo file, non il modello.
    def _id():
        return str(_uuid.uuid4())

    oggi = date.today().isoformat()
    comune = {"core": _CORE_NS, "prod": _PROD_NS, "bbs": _BBS_NS}
    titolo = os.path.splitext(os.path.basename(output_path_3mf))[0]

    # I filamenti del progetto: esattamente quelli che si stampano, nell'ordine
    # in cui si incontrano salendo — la base per prima, l'inchiostro per
    # ultimo. Senza `palette_hex` si ricostruisce dai cambi, che e' comunque il
    # numero giusto: un filamento di partenza piu' uno per ogni cambio.
    palette = list(palette_hex) if palette_hex else (
        ["#FFFFFF"] + [slot_colors[i] if i < len(slot_colors) else "#000000"
                       for i in range(len(valid_z))])
    # L'altezza layer la dichiariamo perche' e' *nostra*: le quote dei cambi
    # colore sono calcolate su quella, e con un valore diverso cadrebbero in
    # mezzo a un layer invece che sul suo confine.
    project_settings = json.dumps({
        "version": _PROJECT_VERSION,
        "from": "project",
        "name": "project_settings",
        "nozzle_diameter": [_DEFAULT_NOZZLE],
        "filament_colour": palette,
        "layer_height": str(layer_height),
        "initial_layer_print_height": str(layer_height),
        "skirt_loops": _SKIRT_LOOPS,
    }, indent=4)


    entries = {
        '[Content_Types].xml': _CONTENT_TYPES,
        '_rels/.rels': _ROOT_RELS,
        '3D/3dmodel.model': _ROOT_TPL.format(
            application=_APPLICATION, today=oggi, title=titolo, dx=dx, dy=dy, dz=dz,
            uuid_comp=_id(), uuid_ref=_id(), uuid_build=_id(), uuid_item=_id(),
            **comune),
        '3D/Objects/object_1.model': _OBJECT_TPL.format(
            uuid_obj=_id(), mesh_xml=mesh_xml, **comune),
        '3D/_rels/3dmodel.model.rels': _MODEL_RELS,
        # La posa e' ripetuta qui, in <assemble>, esattamente come nell'item:
        # nel file scritto da Bambu Studio compare in tutti e due i posti, e
        # dandogliela in uno solo l'oggetto atterrava fuori centro. Anche i
        # source_offset sono suoi: dicono di quanto la mesh e' stata spostata
        # per centrarla sulla propria origine.
        'Metadata/model_settings.config': _MODEL_SETTINGS_TPL.format(
            name=object_name, faces=len(mesh.faces), uuid_part=_id(),
            source=os.path.basename(output_path_3mf),
            off_x=round(float(centro[0]), 6), off_y=round(float(centro[1]), 6),
            off_z=round(float(centro[2]), 6), dx=dx, dy=dy, dz=dz),
        'Metadata/project_settings.config': project_settings,
        'Metadata/custom_gcode_per_layer.xml': _CUSTOM_GCODE_TPL.format(layer_nodes=layer_nodes),
        'Metadata/slice_info.config': _SLICE_INFO,
    }

    with zipfile.ZipFile(output_path_3mf, 'w', zipfile.ZIP_DEFLATED) as z:
        for nome, testo in entries.items():
            z.writestr(nome, testo.encode('utf-8'))

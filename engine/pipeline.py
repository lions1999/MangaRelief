"""
Pipeline di generazione: da immagine + parametri a file STL/3MF.

Questo modulo è il motore vero e proprio, e non dipende da PyQt: la stessa
funzione `generate` serve l'app desktop (avvolta in un QThread) e, in
prospettiva, un backend web (avvolta in un job asincrono). L'avanzamento e
l'annullamento arrivano da fuori come semplici callable, non come segnali Qt.
"""

import os
import shutil
import time

import cv2
import numpy as np
import trimesh
import fast_simplification

from .config import DeckboxConfig
from .params import GenerationParams, GenerationResult
from .resources import asset_path
from .mesh_utils import (standard_switch_z, create_solid_mesh, process_mesh_topo, export_3mf,
                         compute_topo_z_heights, compute_topo_switch_z)
from .color_utils import (bw_coverage_map, classify_spot_pixels, downsample_for_analysis,
                          quantize_grayscale_levels)
from .case_utils import (build_plate_raster, build_case_plate_raster,
                         compose_plate_art, build_bumper)

# Mapping TCG game names → logo asset filenames
TCG_LOGO_MAP = {
    'Yu-Gi-Oh!':       'yugioh_logo.jpg',
    'Pokémon':         'pokemon_logo.jpg',
    'Magic':           'magic_logo.jpg',
    'One Piece':       'onepiece_logo.jpg',
    'Hunter x Hunter': 'hxh_logo.jpg',
}

DECIMATE_THRESHOLD = 200_000


def companion_path_for(plate_path: str) -> str:
    """Percorso del bumper/case TPU accanto alla plate:
    cover_plate_<nome>.* -> cover_bumper_<nome>.stl"""
    d = os.path.dirname(plate_path)
    stem = os.path.splitext(os.path.basename(plate_path))[0]
    if stem.startswith("cover_plate_"):
        stem = stem.replace("cover_plate_", "cover_bumper_", 1)
    else:
        stem += "_bumper_TPU"
    return os.path.join(d, stem + ".stl")


def _as_rgb(img):
    """La pipeline colore lavora sempre in RGB."""
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _as_gray(img):
    """La pipeline Standard/Deckbox lavora in scala di grigi. L'app desktop
    passa già un array 2D, ma un chiamante esterno (es. backend web) ha in mano
    solo il file caricato: accettiamo anche RGB e convertiamo qui."""
    if isinstance(img, np.ndarray) and img.ndim == 3:
        return cv2.cvtColor(np.ascontiguousarray(img, np.uint8), cv2.COLOR_RGB2GRAY)
    return img


def _tone_knots(p: GenerationParams):
    """I due toni intermedi, forzati in ordine dentro (black_clip, white_clip).

    Usati sia per posterizzare sia come nodi della rampa: devono essere gli
    stessi numeri, altrimenti un pixel posterizzato a un tono cadrebbe fra due
    nodi e la sua terrazza finirebbe a una quota che nessuno ha dichiarato.
    """
    lo, hi = int(p.black_clip) + 1, int(p.white_clip) - 1
    l2 = int(np.clip(p.sampled_values[2], lo, hi - 1))
    l1 = int(np.clip(p.sampled_values[1], l2 + 1, hi))
    return l2, l1


def tone_targets(p: GenerationParams):
    """I grigi a cui la modalita' posterizza: uno per bobina, dal nero alla carta."""
    l2, l1 = _tone_knots(p)
    if p.color_mode == 2:
        return [0, 255]
    if p.color_mode == 3:
        return [0, l2, 255]
    return [0, l2, l1, 255]


def posterize_tones(img, p: GenerationParams):
    """Riduce il grigio ai soli livelli che la modalità stampa davvero.

    È il passo che decide su quale bobina finisce ogni pixel, e a 2 colori è
    *tutta* la modalità. Pubblica perché serve anche a chi vuole mostrare la
    classificazione senza generare la mesh: un'anteprima che salta questo
    passaggio mostra un'altra cosa — con due colori, una insensibile all'unico
    controllo che esiste.

    Niente ridimensionamento qui: chi chiama sceglie a che risoluzione lavorare.
    """
    img = _as_gray(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    if p.color_mode == 2:
        # Soglia dinamica: punto medio tra il bianco (L0) e il nero (L3) campionati,
        # così la calibrazione degli swatch conta anche in modalità B&N
        bw_threshold = int((p.sampled_values[0] + p.sampled_values[3]) / 2.0)
        bw_threshold = int(np.clip(bw_threshold, 1, 254))
        _, img = cv2.threshold(img, bw_threshold, 255, cv2.THRESH_BINARY)
    else:
        targets = np.array(tone_targets(p))

        idx = np.abs(img[..., np.newaxis] - targets).argmin(axis=-1)
        img = targets[idx].astype(np.uint8)
    return img


# Finestra su cui si misura la copertura d'inchiostro a 2 colori, in mm.
# Misurata, non scelta: a 180 mm su 800 px (0,225 mm/cella) sono 3 celle.
#   1 cella  -> il taglio non ha effetto sul tratteggio (aliasing puro: righe
#               alternate qualunque valore si scelga)
#   3 celle  -> tratti sottili conservati all'86-96% con taglio 0,25-0,35
#               (lo storico ne conservava il 63%), ombre decise dal taglio
#   5 celle  -> il testo dei balloon sparisce: un tratto da 0,3 mm copre meno
#               di meta' di una finestra da 1,1 mm
BW_WINDOW_MM = 0.7


def _target_size(shape, p: GenerationParams):
    h, w = shape[:2]
    target_res = min(int(p.max_dim / 0.05), p.max_res_cap)
    scale = target_res / max(w, h)
    return (int(w * scale), int(h * scale)), target_res


def prepare_source_image(img, p: GenerationParams):
    """Porta l'immagine alla risoluzione della mesh e la posterizza.

    Pubblica insieme a standard_heightmap perché le due, in quest'ordine,
    *sono* la classificazione: chi vuole mostrarla senza costruire la mesh deve
    passare di qui, non ricostruirsela.
    """
    img = _as_gray(img)
    (new_w, new_h), target_res = _target_size(img.shape, p)

    if p.color_mode == 2 and p.bw_coverage is not None:
        # Percorso per copertura: l'inchiostro si decide sull'immagine intera,
        # prima di qualunque ridimensionamento, e la soglia e' una frazione di
        # area — vedi color_utils.bw_coverage_map.
        mm_per_px = p.max_dim / target_res
        window = int(round(BW_WINDOW_MM / mm_per_px))
        frac = bw_coverage_map(img, (new_w, new_h), window)
        return np.where(frac >= float(p.bw_coverage), 0, 255).astype(np.uint8)

    if max(img.shape[1], img.shape[0]) != target_res:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return posterize_tones(img, p)


def _get_z_mapping(p: GenerationParams):
    """Calculates X and Y points for piecewise linear interpolation."""
    L1_Z = p.color_changes_z[0]
    L2_Z = p.color_changes_z[1]
    # I nodi della rampa stanno SUI toni campionati, non sul loro punto medio
    # e su white_clip-1 come in origine: l'immagine arriva qui gia'
    # posterizzata esattamente a quei toni, quindi solo il valore della rampa
    # in quei punti conta — e li' deve valere la quota che la UI dichiara.
    # Con i nodi vecchi la terrazza "L1" di un pannello vero finiva a 1,512
    # invece che a 1,40: fuori layer, e con 0,4 mm di parete del colore sotto.
    # Ordine forzato black_clip < L2 < L1 < white_clip, altrimenti np.interp
    # e' indefinita.
    l2_target, l1_target = _tone_knots(p)

    # A 3 colori il selettore nasconde L1 e lascia in gioco L2 (vedi
    # _compute_auto_z / manga_to_3d._refresh_color_mode): l'unico livello
    # intermedio e' color_changes_z[1], mentre [0] vale 0.0 per convenzione
    # ("layer non usato"). Leggendo [0] il midtone finiva a quota 0 — sotto la
    # base — cosi' la fascia di mezzo restava senza un solo pixel e la seconda
    # bobina non colorava nulla.
    mid_Z = L2_Z if p.color_mode == 3 else L1_Z

    if p.is_deckbox_mode:
        deboss_depth = DeckboxConfig.DEBOSS_DEPTH
        base_thickness = DeckboxConfig.BASE_THICKNESS

        deboss_surface = base_thickness
        # Limite di sicurezza basato sullo spessore residuo solido
        deboss_floor = max(DeckboxConfig.MIN_SOLID_WALL_THICKNESS, base_thickness - deboss_depth)

        relief_range = p.max_h - p.base_h
        L1_ratio = (L1_Z - p.base_h) / relief_range if relief_range > 0 else 0.33
        L2_ratio = (L2_Z - p.base_h) / relief_range if relief_range > 0 else 0.66

        mid_ratio = L2_ratio if p.color_mode == 3 else L1_ratio

        # I layer Z scalano da deboss_floor (fondo scavo) a deboss_surface (superficie muro)
        L1_deboss = deboss_floor + L1_ratio * deboss_depth
        L2_deboss = deboss_floor + L2_ratio * deboss_depth
        mid_deboss = deboss_floor + mid_ratio * deboss_depth

        if p.color_mode == 4:
            x_pts = [0, p.black_clip, l2_target, l1_target, p.white_clip, 255]
            y_pts = [deboss_surface, deboss_surface, L2_deboss, L1_deboss, deboss_floor, deboss_floor]
        elif p.color_mode == 3:
            x_pts = [0, p.black_clip, l2_target, p.white_clip, 255]
            y_pts = [deboss_surface, deboss_surface, mid_deboss, deboss_floor, deboss_floor]
        else:  # 2 Colors
            x_pts = [0, p.black_clip, p.white_clip - 1, p.white_clip, 255]
            y_pts = [deboss_surface, deboss_surface, deboss_surface, deboss_floor, deboss_floor]

        x_pts, y_pts = np.array(x_pts), np.array(y_pts)
        s_idx = np.argsort(x_pts)
        return x_pts[s_idx], y_pts[s_idx], deboss_floor, deboss_surface
    else:
        if p.color_mode == 4:
            x_pts = [0, p.black_clip, l2_target, l1_target, p.white_clip, 255]
            y_pts = [p.max_h, p.max_h, L2_Z, L1_Z, p.base_h, p.base_h]
        elif p.color_mode == 3:
            x_pts = [0, p.black_clip, l2_target, p.white_clip, 255]
            y_pts = [p.max_h, p.max_h, mid_Z, p.base_h, p.base_h]
        else:  # 2 Colors
            x_pts = [0, p.black_clip, p.white_clip - 1, p.white_clip, 255]
            y_pts = [p.max_h, p.max_h, p.max_h, p.base_h, p.base_h]

        x_pts, y_pts = np.array(x_pts), np.array(y_pts)
        s_idx = np.argsort(x_pts)
        return x_pts[s_idx], y_pts[s_idx], p.base_h, p.max_h


def standard_heightmap(gray, params: GenerationParams):
    """Quota Z di ogni pixel in modalità Standard/Deckbox.

    È il cuore del rilievo: l'interpolazione lineare a tratti fra i toni
    campionati e le quote di cambio colore, coi due estremi bloccati sul
    fondo e sulla superficie. Pubblica perché serve anche a chi vuole
    *mostrare* dove finirà ogni tono senza generare la mesh — un'anteprima
    che la ricalcolasse per conto suo divergerebbe alla prima modifica.
    """
    x_pts, y_pts, floor_z, surface_z = _get_z_mapping(params)
    flat = gray.flatten()
    z = np.round(np.interp(flat, x_pts, y_pts), 3)
    z[flat <= params.black_clip] = surface_z
    z[flat >= params.white_clip] = floor_z
    return z.reshape(gray.shape)


def _decimate(mesh, allowed_z=None):
    """Riduce la mesh sopra la soglia, poi (in topo) ri-snappa le quote alle
    terrazze: la decimazione inclina le pareti verticali e lo slicer mostrerebbe
    anelli di colori intermedi attorno ad ogni bordo."""
    target = 100_000 + min(50_000, int((len(mesh.faces) - DECIMATE_THRESHOLD) * 0.05))
    v_out, f_out = fast_simplification.simplify(
        mesh.vertices.astype(np.float64),
        mesh.faces.astype(np.int64),
        target_count=target, agg=6.0
    )
    mesh = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)

    if allowed_z is not None:
        allowed = np.asarray(allowed_z)
        nearest = np.abs(mesh.vertices[:, [2]] - allowed[None, :]).argmin(axis=1)
        mesh.vertices[:, 2] = allowed[nearest]

    # Il decimatore lascia una decina di triangoli ad area zero, e lo snap qui
    # sopra puo' aggiungerne altri schiacciando vertici sulla stessa quota. Non
    # sono buchi — fill_holes non trova nulla da chiudere e restituisce False —
    # ma spigoli condivisi da tre facce, che bastano a rendere la mesh non
    # watertight (su un pannello vero: 150.000 facce, 0 bordi aperti, 4 spigoli
    # tripli, 10 degeneri).
    #
    # L'ordine conta. Un triangolo degenere ha due vertici coincidenti ma
    # distinti: rimuoverlo cosi' com'e' lascia un varco fra i vicini (4 bordi
    # aperti in Standard, 12 in Spot, che fill_holes richiude male). Fondendo
    # prima i vertici coincidenti, i vicini si ricuciono da soli e il triangolo
    # sparisce senza lasciare nulla. Verificato in entrambe le modalita' e con
    # due aggressivita' di decimazione: chiusa, volume invariato allo 0,0000%.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    # Per ultimo: le facce eventualmente aggiunte da fill_holes vanno
    # orientate insieme al resto.
    trimesh.repair.fix_normals(mesh)
    return mesh


def _assemble_deckbox_mesh(mesh):
    """Scales, rotates and concatenates the art mesh with the deckbox template."""
    import trimesh.transformations as tf
    template_path = asset_path("template_deckbox_open.stl")
    if not os.path.exists(template_path):
        print(f"Warning: Deckbox template not found at {template_path}")
        return mesh

    box_mesh = trimesh.load(template_path)

    # Fixed scale to fit the template notch + 0.2mm overlap to prevent gaps
    mesh_extents = mesh.extents
    scale_x = (DeckboxConfig.WALL_WIDTH + 0.2) / mesh_extents[0]
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


def _process_lid_logo(p: GenerationParams, progress):
    """Generates the TCG logo plug and engravings for the deckbox lid mesh (in memory only)."""
    template_lid = asset_path("template_coperchio_bucato.stl")
    if not os.path.exists(template_lid):
        print(f"WARNING: Lid template not found at {template_lid}")
        return None

    lid_mesh = trimesh.load(template_lid)
    logo_filename = TCG_LOGO_MAP.get(p.tcg_name)
    logo_path = asset_path(logo_filename) if logo_filename else None

    if logo_path and os.path.exists(logo_path):
        progress(97, f"Engraving {p.tcg_name} logo on lid...")
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

    if p.smart_decimate and len(lid_mesh.faces) > DECIMATE_THRESHOLD:
        progress(98, "Optimizing Lid Mesh (Decimation)...")
        lid_mesh = _decimate(lid_mesh)

    return lid_mesh


def generate(image, params: GenerationParams, progress=None, should_cancel=None) -> GenerationResult:
    """Genera i modelli 3D dall'immagine secondo i parametri dati.

    image         : array RGB o scala di grigi (numpy)
    progress      : callable(percentuale:int, messaggio:str), opzionale
    should_cancel : callable() -> bool, interrogato tra una fase e l'altra

    Solleva InterruptedError se l'annullamento viene richiesto; gli altri errori
    si propagano al chiamante, che decide come presentarli.
    """
    p = params
    emit = progress or (lambda pct, msg: None)

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise InterruptedError("Process cancelled by user")

    check_cancel()
    t_start_total = time.time()

    img_work = image
    max_dim = p.max_dim
    is_topo = p.is_topo_mode
    topo_colors = p.topo_colors
    topo_z_heights = None

    # Quote dei cambi colore da iniettare nel 3MF: in topo vengono
    # ricalcolate sulle terrazze, altrimenti valgono quelle Standard
    export_changes_z = p.color_changes_z
    export_slot_colors = None

    plate_mask = None
    if p.is_cover_mode and p.cover_preset:
        emit(6, "📱 Composing artwork on plate...")
        img_rgb_src = _as_rgb(img_work)
        if 'case_plate' in p.cover_preset:
            plate_mask, res, pd = build_case_plate_raster(p.cover_preset, p.max_res_cap)
            avoid = False  # la sede reale esclude già la fotocamera
        else:
            plate_mask, res, pd = build_plate_raster(p.cover_preset, p.max_res_cap)
            avoid = p.cover_avoid_camera
        art = compose_plate_art(img_rgb_src, p.cover_preset, plate_mask, res,
                                user_scale=p.cover_scale,
                                off_x=p.cover_off_x, off_y=p.cover_off_y,
                                avoid_camera=avoid)
        if p.cover_finish_spot:
            palette, idx_map = classify_spot_pixels(
                art, p.spot_accents, coverage=p.spot_coverage,
                white_clip=p.white_clip, black_clip=p.black_clip)
        else:
            # B/N: quantizzazione calibrata a livelli (White/Black Clip +
            # K-Means sui mezzitoni), non la soglia secca dello Spot a 0 accenti
            palette, idx_map = quantize_grayscale_levels(
                art, n_levels=p.cover_gray_levels,
                white_clip=p.white_clip, black_clip=p.black_clip)
        if p.cover_engraved:
            # Inciso: ordine di stampa invertito (scuro per primo), la
            # superficie esterna è il piano chiaro e l'arte sta scavata
            palette = palette[::-1]
            idx_map = (len(palette) - 1) - idx_map
        img_work = np.array(palette, dtype=np.uint8)[idx_map]
        export_slot_colors = ['#%02x%02x%02x' % tuple(c) for c in palette[1:]]
        # la plate segue la pipeline Topographic (terrazze + snap)
        is_topo = True
        topo_colors = palette
        max_dim = max(pd['width'], pd['height'])

    if p.is_spot_mode:
        emit(8, "🎯 Spot Color classification...")
        img_rgb_src = _as_rgb(img_work)
        small = downsample_for_analysis(img_rgb_src, p.max_res_cap)
        palette, idx_map = classify_spot_pixels(
            small, p.spot_accents, coverage=p.spot_coverage,
            white_clip=p.white_clip, black_clip=p.black_clip)
        img_work = np.array(palette, dtype=np.uint8)[idx_map]
        # Nei metadata 3MF finiscono i colori reali della palette (non i grigi)
        export_slot_colors = ['#%02x%02x%02x' % tuple(c) for c in palette[1:]]
        # Da qui in poi la pipeline coincide con la Topographic
        is_topo = True
        topo_colors = palette

    if is_topo and topo_colors:
        emit(10, "🏔 Starting Topographic Color Processing...")
        topo_z_heights = compute_topo_z_heights(
            p.base_h, p.max_h, p.layer_height, len(topo_colors))
        export_changes_z = compute_topo_switch_z(topo_z_heights, p.layer_height)
        img_rgb = _as_rgb(img_work)

        # Le plate cover sono piccole (~70mm): un dettaglio manga da 0.5mm
        # coprirebbe una frazione enorme del disegno. In incisione il solco è
        # assenza di materiale (non una parete stampata), quindi si può scendere.
        min_feature = 0.5
        if p.is_cover_mode:
            min_feature = 0.25 if p.cover_engraved else 0.35

        mesh = process_mesh_topo(
            img_rgb,
            topo_colors,
            base_z=p.base_h,
            total_z=p.max_h,
            max_dim=max_dim,
            layer_height=p.layer_height,
            max_res_cap=p.max_res_cap,
            mask=plate_mask,
            min_feature_mm=min_feature
        )
        check_cancel()
        emit(80, "Optimizing Topo Mesh...")
    else:
        # --- 1. Image Preparation ---
        emit(5, "Optimizing resolution for 3D mesh...")
        img = prepare_source_image(img_work, p)
        check_cancel()

        # --- 2. Z-Mapping & Heightmap ---
        emit(20, "Applying Piecewise Interpolation (Z Mapping)...")
        Z = standard_heightmap(img, p)
        h, w = img.shape

        # Le quote di cambio filamento vengono dalle terrazze che ogni tono
        # OCCUPA nella mappa Z — non da color_changes_z, che sono le cime delle
        # terrazze (un cambio in cima colora un layer solo). Si calcolano dai
        # toni attesi e non dalle quote presenti nella heightmap: se un tono
        # non ha pixel la sua terrazza manca, e derivando dalle quote presenti
        # i cambi scalerebbero di posto — la bobina 3 finirebbe sull'inchiostro.
        # Cosi' ogni bobina resta la sua; una terrazza vuota colora nulla.
        tone_levels = sorted(set(
            float(v) for v in np.round(
                standard_heightmap(np.array(tone_targets(p), dtype=np.uint8), p), 3)))
        export_changes_z = standard_switch_z(tone_levels, p.layer_height)
        # Il ri-snap dopo la decimazione invece usa le quote davvero presenti
        # (altrimenti le pareti si inclinano: vertici a 0,983 e 1,004 attorno
        # a una base a 1,0 in un job a 4 colori).
        standard_levels = sorted(float(v) for v in np.unique(np.round(Z, 3)))

        # --- 3. Grid & Geometry ---
        if w >= h:
            dim_x = float(max_dim)
            dim_y = float(max_dim) * (h / w)
        else:
            dim_y = float(max_dim)
            dim_x = float(max_dim) * (w / h)

        x = np.linspace(0, dim_x, w)
        y = np.linspace(0, dim_y, h)[::-1]
        X, Y = np.meshgrid(x, y)

        emit(40, "Generating solid vertices (Watertight)...")
        mesh = create_solid_mesh(X, Y, Z, bottom_z=0.0)
        check_cancel()

    # --- 4. Mesh Assembly ---
    emit(55, "Finalizing Geometry...")

    if p.is_deckbox_mode:
        emit(91, "Assembling Deckbox (Wall Replacement)...")
        mesh = _assemble_deckbox_mesh(mesh)
        check_cancel()

    # --- 5. Optimization (Decimation) ---
    check_cancel()
    if p.smart_decimate and len(mesh.faces) > DECIMATE_THRESHOLD:
        emit(92, "Optimizing Mesh (Decimation)...")
        if is_topo and topo_colors:
            allowed_z = [0.0] + topo_z_heights
        elif not p.is_deckbox_mode:
            allowed_z = [0.0] + standard_levels
        else:
            allowed_z = None      # la scatola ha le sue quote: non si tocca
        mesh = _decimate(mesh, allowed_z=allowed_z)

    check_cancel()

    # --- 6. Final Clamping & Export ---
    emit(95, "Finalizing and Exporting...")
    if not p.is_deckbox_mode:
        # Absolute Z clamping for relief panels
        mesh.vertices[:, 2] = np.clip(mesh.vertices[:, 2], 0.0, p.max_h)
    mesh.vertices[:, 2] = np.round(mesh.vertices[:, 2], 3)

    result = GenerationResult()

    if p.is_deckbox_mode:
        out_dir = os.path.dirname(p.output_path_3mf or p.output_path)
        os.makedirs(out_dir, exist_ok=True)
        stl_dir = os.path.dirname(p.output_path) if p.output_path else out_dir

        # Build lid mesh (in memory only)
        emit(97, "Generating Lid...")
        lid_mesh = _process_lid_logo(p, emit)

        # --- Combine front + lid side by side (PLATE_GAP_MM gap) ---
        emit(99, "Assembling full deckbox plate...")
        GAP_MM = DeckboxConfig.PLATE_GAP_MM
        if lid_mesh is not None:
            front_mesh_copy = mesh.copy()
            lid_mesh_copy = lid_mesh.copy()

            front_min_x = front_mesh_copy.bounds[0][0]
            front_mesh_copy.apply_translation([-front_min_x, 0, 0])

            front_max_x = front_mesh_copy.bounds[1][0]
            lid_min_x = lid_mesh_copy.bounds[0][0]
            lid_mesh_copy.apply_translation([front_max_x + GAP_MM - lid_min_x, 0, 0])

            full_mesh = trimesh.util.concatenate([front_mesh_copy, lid_mesh_copy])

            if p.output_path_3mf:
                result.mf3_path = os.path.join(out_dir, f"full_deckbox_{p.source_image_name}.3mf")
                export_3mf(full_mesh, result.mf3_path, export_changes_z)
            if p.output_path:
                result.stl_path = os.path.join(stl_dir, f"full_deckbox_{p.source_image_name}.stl")
                full_mesh.export(result.stl_path)
    else:
        if p.output_path:
            mesh.export(p.output_path)
            result.stl_path = p.output_path
        if p.output_path_3mf:
            export_3mf(mesh, p.output_path_3mf, export_changes_z,
                       slot_colors=export_slot_colors)
            result.mf3_path = p.output_path_3mf

        if p.is_cover_mode and p.include_bumper and p.cover_preset:
            ref = p.output_path or p.output_path_3mf
            companion = companion_path_for(ref)
            case_plate = p.cover_preset.get('case_plate') or {}
            if case_plate.get('template'):
                # sede reale: il companion è la cover scavata (template)
                emit(99, "Copying carved case (TPU) next to the plate...")
                shutil.copyfile(asset_path(case_plate['template']), companion)
            else:
                emit(99, "Generating TPU bumper (separate STL)...")
                cp = p.cover_preset
                bumper = build_bumper(
                    cp['width'], cp['height'], cp['thickness'], cp['corner_radius'],
                    bottom_opening_w=cp.get('bottom_opening', 45.0),
                    side_cutouts=[tuple(c) for c in cp.get('side_cutouts', [])],
                    top_cutouts=[tuple(c) for c in cp.get('top_cutouts', [])])
                bumper.export(companion)
            result.companion_path = companion

    emit(100, "Export completed!")
    result.color_changes_z = [round(float(z), 3)
                              for z in (export_changes_z or []) if z > 0]
    result.slot_colors = list(export_slot_colors or [])
    result.elapsed_s = time.time() - t_start_total
    return result

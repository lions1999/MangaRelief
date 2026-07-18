import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from config import SPOT_BASE_RGB, SPOT_TOP_RGB

# Peso della cromaticità (canali a/b) nel matching pixel->colore: con la distanza
# Lab pura un grigio medio risulta più "vicino" a un rosso saturo che al nero,
# perché la differenza di luminosità pesa quanto quella di tinta. Amplificando
# a/b, un pixel senza tinta non può mai finire su un cluster saturo.
CHROMA_MATCH_WEIGHT = 2.5

def rgb_to_lab(rgb_array: np.ndarray, chroma_weight: float = 1.0) -> np.ndarray:
    """Converte un array RGB uint8 (HxWx3 oppure Nx3) nello spazio Lab di OpenCV.
    Le distanze in Lab rispecchiano la percezione umana, a differenza dell'RGB.
    chroma_weight > 1 amplifica i canali a/b per il matching percettivo."""
    arr = np.ascontiguousarray(rgb_array, dtype=np.uint8)
    lab = cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    lab = lab.reshape(arr.shape).astype(np.float32)
    if chroma_weight != 1.0:
        lab[..., 1:] *= chroma_weight
    return lab

def downsample_for_analysis(image_rgb: np.ndarray, max_size: int = 800) -> np.ndarray:
    """Riduce l'immagine per le analisi colore: i cluster non cambiano,
    ma i tempi di calcolo crollano e la UI non si congela."""
    h, w = image_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    return image_rgb

def merge_lab_clusters(centers_lab: np.ndarray, counts: np.ndarray,
                       merge_threshold: float = 25.0):
    """Fusione pesata dei cluster Lab più vicini di merge_threshold.
    Ritorna (centri fusi, pesi totali), dal cluster più popoloso."""
    merged = []  # coppie [somma pesata dei centri, peso totale]
    for idx in np.argsort(-counts):
        center, weight = centers_lab[idx], float(counts[idx])
        for m in merged:
            if np.linalg.norm(m[0] / m[1] - center) < merge_threshold:
                m[0] += center * weight
                m[1] += weight
                break
        else:
            merged.append([center * weight, weight])
    centers = np.array([m[0] / m[1] for m in merged])
    weights = np.array([m[1] for m in merged])
    return centers, weights

def extract_dominant_colors(image_rgb: np.ndarray, n_colors: int = 5,
                            merge_threshold: float = 25.0) -> list:
    """Estrae i colori dominanti con K-Means in spazio Lab (percettivo) e li ordina
    per luminosità dal più chiaro al più scuro (il primo in lista = base di stampa).
    I cluster quasi identici vengono fusi: su immagini con pochi colori reali
    (es. bianco/nero/rosso) K=5 creerebbe cluster spuri dai bordi anti-aliasing,
    che in stampa finiscono su layer sbagliati."""
    image_rgb = downsample_for_analysis(image_rgb)

    # Escludi i pixel di transizione (anti-aliasing e sfumature dei bordi):
    # altrimenti il K-Means dedica interi cluster ai colori "misti" dei contorni,
    # che poi in stampa emergono come terrazze fantasma dai layer sbagliati
    grad = cv2.morphologyEx(image_rgb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    flat_mask = grad.max(axis=2) < 12
    pixels_rgb = image_rgb.reshape(-1, 3)
    if np.count_nonzero(flat_mask) > max(1000, flat_mask.size // 20):
        pixels_rgb = pixels_rgb[flat_mask.ravel()]

    pixels_lab = rgb_to_lab(pixels_rgb).reshape(-1, 3)
    # n_init='auto' per sopprimere warning e velocizzare
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels_lab)
    centers_lab, _ = merge_lab_clusters(kmeans.cluster_centers_,
                                        np.bincount(kmeans.labels_, minlength=n_colors),
                                        merge_threshold)

    colors = cv2.cvtColor(
        np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
        cv2.COLOR_LAB2RGB
    ).reshape(-1, 3).astype(int)

    # Ordina per luminosità percepita, dal più chiaro (base) al più scuro
    luminances = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors]
    sorted_indices = np.argsort(luminances)[::-1]
    sorted_colors = colors[sorted_indices]

    return [tuple(c) for c in sorted_colors]

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
# SPOT COLOR MODE  —  serigrafia: base bianca + 1-2 accenti + nero in cima
# ---------------------------------------------------------------------------

def suggest_spot_accents(image_rgb: np.ndarray, n_accents: int = 2) -> list:
    """Suggerisce gli accenti più salienti per la modalità Spot Color.
    K-Means in Lab sui soli pixel vividi (saturi e non troppo scuri),
    classifica per punteggio = numerosità × croma media del cluster.
    Ritorna una lista di 0..n_accents tuple RGB (vuota se l'immagine è
    praticamente in bianco e nero)."""
    img = downsample_for_analysis(image_rgb)
    hsv = cv2.cvtColor(np.ascontiguousarray(img, np.uint8), cv2.COLOR_RGB2HSV)
    vivid = (hsv[..., 1] >= 90) & (hsv[..., 2] >= 60)
    pixels = img.reshape(-1, 3)[vivid.ravel()]

    # Sotto lo 0.5% di pixel vividi non c'è nessun accento sensato da proporre
    if len(pixels) < max(500, vivid.size // 200):
        return []

    k = min(5, len(pixels))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(rgb_to_lab(pixels))
    centers, weights = merge_lab_clusters(kmeans.cluster_centers_,
                                          np.bincount(kmeans.labels_, minlength=k))

    # Punteggio di salienza: croma alta e area consistente, ma con penalità per i
    # colori "da sfondo" (quota di pixel nella cornice esterna dell'immagine) e
    # radice sull'area, così un soggetto vivido batte uno sfondo enorme
    hh, ww = img.shape[:2]
    m = max(2, int(0.08 * min(hh, ww)))
    frame = np.zeros((hh, ww), dtype=bool)
    frame[:m, :] = frame[-m:, :] = frame[:, :m] = frame[:, -m:] = True
    _, member = cKDTree(centers).query(rgb_to_lab(pixels))
    frame_flags = frame.ravel()[np.flatnonzero(vivid.ravel())]
    border_share = np.array([
        frame_flags[member == i].mean() if np.any(member == i) else 0.0
        for i in range(len(centers))
    ])
    chroma = np.linalg.norm(centers[:, 1:] - 128.0, axis=1)
    score = np.sqrt(weights) * chroma * (1.0 - border_share) ** 2

    order = np.argsort(-score)
    top = np.clip(centers[order[:n_accents]], 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(top.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return [tuple(int(v) for v in c) for c in rgb]


# Un pixel appartiene a un accento solo se la sua tinta circolare dista meno di
# ±36° (18 unità OpenCV su 180) da quella dell'accento, e se non è quasi nero
SPOT_HUE_TOL = 18
SPOT_V_MIN = 60

def build_spot_palette(accents_rgb: list) -> list:
    """Palette Spot Color ordinata per la stampa:
    [base bianca, accenti dal più chiaro al più scuro, nero top]."""
    accents = sorted((tuple(int(v) for v in a) for a in accents_rgb),
                     key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2],
                     reverse=True)
    return [SPOT_BASE_RGB] + accents + [SPOT_TOP_RGB]

def classify_spot_pixels(image_rgb: np.ndarray, accents_rgb: list,
                         coverage: int = 40):
    """Classifica ogni pixel sulla palette Spot Color:
    [base bianca, accenti ordinati dal più chiaro al più scuro, nero top].
    Un pixel va su un accento se è abbastanza saturo (soglia guidata da
    coverage 0-100: basso = solo pixel vividi, alto = anche sfumature spente)
    e la sua tinta è entro ±36° da quella dell'accento; tutto il resto viene
    binarizzato bianco/nero sulla luminosità.
    Ritorna (palette_rgb, indices HxW di indici nella palette)."""
    palette = build_spot_palette(accents_rgb)
    accents = palette[1:-1]
    n = len(palette)
    h, w = image_rgb.shape[:2]

    img_u8 = np.ascontiguousarray(image_rgb, np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    lum = rgb_to_lab(img_u8)[..., 0]

    # Neutri: binarizzazione bianco/nero sul punto medio di luminosità
    pal_lab = rgb_to_lab(np.array(palette, dtype=np.uint8))
    lum_split = (pal_lab[0, 0] + pal_lab[-1, 0]) / 2.0
    idx = np.where(lum >= lum_split, 0, n - 1).astype(np.intp)

    if accents:
        accent_hues = cv2.cvtColor(
            np.array(accents, dtype=np.uint8).reshape(-1, 1, 3),
            cv2.COLOR_RGB2HSV)[:, 0, 0].astype(int)

        # Distanza di tinta circolare da ogni accento (unità OpenCV, 0-180)
        hue = hsv[..., 0].astype(int)
        dists = np.stack([np.minimum(np.abs(hue - ah), 180 - np.abs(hue - ah))
                          for ah in accent_hues])
        best_accent = dists.argmin(axis=0)
        best_dist = dists.min(axis=0)

        # coverage guida la saturazione minima: 0 -> solo vividi, 100 -> quasi tutto
        sat_min = int(np.clip(170 - 1.5 * coverage, 15, 170))
        accent_mask = ((hsv[..., 1] >= sat_min) & (hsv[..., 2] >= SPOT_V_MIN)
                       & (best_dist <= SPOT_HUE_TOL))
        idx[accent_mask] = 1 + best_accent[accent_mask]

    return palette, idx

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

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

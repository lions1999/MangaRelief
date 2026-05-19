import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_colors(image_rgb: np.ndarray, n_colors: int = 5) -> list:
    """Estrae i colori dominanti e li ordina per luminosità (dal più scuro al più chiaro)."""
    pixels = image_rgb.reshape(-1, 3)
    # n_init='auto' per sopprimere warning e velocizzare
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Ordina i colori per luminosità percepita (Luminance)
    luminances = [0.299*c[0] + 0.587*c[1] + 0.114*c[2] for c in colors]
    sorted_indices = np.argsort(luminances)
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

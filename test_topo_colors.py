import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import sys
import os

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Errore: Immagine '{image_path}' non trovata.")
        return
        
    print(f"Caricamento immagine: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    # Pre-scaling per velocizzare il test ed eliminare micro-dettagli impossibili da stampare
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    h, w, c = pixels.shape
    
    print("Esecuzione K-Means (5 clusters)...")
    flat_pixels = pixels.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto').fit(flat_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    print("\nColori Estratti (RGB):")
    for i, color in enumerate(colors):
        print(f"Cluster {i}: RGB{tuple(color)}")
        
    print("\nMappatura dei pixel al cluster più vicino...")
    tree = cKDTree(colors)
    _, indices = tree.query(flat_pixels)
    
    # Ricostruisce l'immagine usando solo i 5 colori esatti
    posterized_pixels = colors[indices].reshape(h, w, 3).astype(np.uint8)
    posterized_img = Image.fromarray(posterized_pixels)
    
    output_path = "debug_posterized.png"
    posterized_img.save(output_path)
    print(f"\n[SUCCESSO] Immagine di debug salvata in: {output_path}")
    
    print("\nGenerazione Mesh 3D a terrazze...")
    from engine.color_utils import extract_dominant_colors
    from engine.mesh_utils import process_mesh_topo
    
    # Estrae e ordina
    ordered_colors = extract_dominant_colors(pixels, 5)
    
    # Genera mesh
    mesh = process_mesh_topo(pixels, ordered_colors, base_z=1.0, total_z=2.4)
    
    stl_path = "debug_topo_mesh.stl"
    mesh.export(stl_path)
    print(f"[SUCCESSO] STL di test salvato in: {stl_path}")
    print("Controlla l'immagine e la mesh per verificare la presenza di rumore, artefatti o bordi spuri!")

if __name__ == "__main__":
    # Sostituisci "metroid.jpg" con il nome reale di una tua immagine di test se non passi argomenti
    test_img = sys.argv[1] if len(sys.argv) > 1 else "metroid.jpg" 
    main(test_img)

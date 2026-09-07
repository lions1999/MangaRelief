"""
Schema dei parametri di generazione.

Sostituisce i ~30 argomenti che venivano passati singolarmente al worker: averli
in un oggetto solo rende esplicito il contratto del motore, permette di
serializzarli (il log delle generazioni, e in prospettiva una richiesta HTTP) e
dà un punto unico dove validare valori che arrivano dall'esterno.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


class GenerationMode:
    """Modalità di generazione, allineate all'ordine del selettore nella UI."""
    STANDARD = "standard"
    TOPOGRAPHIC = "topographic"
    DECKBOX = "deckbox"
    SPOT_COLOR = "spot_color"
    PHONE_COVER = "phone_cover"


@dataclass
class GenerationParams:
    """Tutti i parametri di una generazione, immagine esclusa."""

    # --- Modalità ---
    mode: str = GenerationMode.STANDARD

    # --- Parametri fisici (comuni a tutte le modalità) ---
    max_dim: float = 200.0          # dimensione del lato lungo, in mm
    base_h: float = 1.0             # spessore della base, in mm
    max_h: float = 2.4              # altezza totale, in mm
    layer_height: float = 0.2       # altezza layer di stampa, in mm
    max_res_cap: int = 1200         # cap risoluzione (Mesh Quality)
    smart_decimate: bool = True

    # --- Calibrazione toni ---
    white_clip: int = 235
    black_clip: int = 15
    sampled_values: List[int] = field(default_factory=lambda: [250, 210, 150, 15])
    color_mode: int = 4             # 2/3/4 colori della modalità Standard
    color_changes_z: List[float] = field(default_factory=lambda: [1.4, 2.0, 2.4])
    # Solo a 2 colori. None = percorso storico: sfoca la ridotta e soglia a
    # (L0+L3)/2, cioe' quello che fanno gli swatch Paper/Ink del desktop.
    # Un valore 0..1 attiva la binarizzazione per copertura: l'inchiostro si
    # decide a piena risoluzione e una zona diventa nera quando almeno questa
    # frazione della sua area e' inchiostrata. Vedi bw_coverage_map.
    bw_coverage: Optional[float] = None

    # --- Topographic ---
    topo_colors: Optional[List[Tuple[int, int, int]]] = None

    # --- Spot Color ---
    spot_accents: List[Tuple[int, int, int]] = field(default_factory=list)
    spot_coverage: int = 40

    # --- Deckbox ---
    tcg_name: str = "Yu-Gi-Oh!"

    # --- Ritaglio sagoma (portachiavi) ---
    # Le correzioni all'automatismo viaggiano come SEMI, non come raster: una
    # lista di (x, y) che nominano una regione. Le coordinate sono quelle del
    # raster di segmentazione (engine.cutout_utils.seg_shape_for con
    # cutout_seg_res), che dipende solo dalla forma della sorgente e dal cap —
    # quindi restano valide mentre si muove White Clip, e questi parametri
    # restano serializzabili come tutti gli altri.
    cutout_enabled: bool = False
    cutout_cut_seeds: List[Tuple[int, int]] = field(default_factory=list)
    cutout_keep_seeds: List[Tuple[int, int]] = field(default_factory=list)
    # Strada alternativa: la maschera dipinta a mano (giallo = stampa). Se c'e',
    # sostituisce del tutto segmentazione e semi.
    cutout_paint_mask: Optional[Any] = None      # ndarray RGB
    cutout_seg_res: int = 1600
    cutout_border_mm: float = 0.0                # bordino da sticker
    cutout_min_feature_mm: float = 0.8           # pulviscolo e fori sotto soglia
    cutout_keep_largest: bool = True             # un portachiavi e' un pezzo solo
    cutout_ring: bool = False
    cutout_ring_xy: Optional[Tuple[int, int]] = None
    cutout_ring_d_mm: float = 4.0                # foro per l'anellino
    cutout_ring_rim_mm: float = 2.0              # materiale attorno al foro

    # --- Phone Cover ---
    cover_preset: Optional[Dict[str, Any]] = None
    cover_scale: float = 1.0
    cover_off_x: float = 0.0
    cover_off_y: float = 0.0
    cover_finish_spot: bool = False
    cover_avoid_camera: bool = True
    cover_engraved: bool = True
    cover_gray_levels: int = 3
    include_bumper: bool = False

    # --- Destinazioni ---
    output_path: Optional[str] = None       # STL
    output_path_3mf: Optional[str] = None   # 3MF
    source_image_name: str = "panel"

    # ------------------------------------------------------------------
    # Comodità: i rami interni della pipeline ragionano ancora per flag
    # ------------------------------------------------------------------
    @property
    def is_topo_mode(self) -> bool:
        return self.mode == GenerationMode.TOPOGRAPHIC

    @property
    def is_deckbox_mode(self) -> bool:
        return self.mode == GenerationMode.DECKBOX

    @property
    def is_spot_mode(self) -> bool:
        return self.mode == GenerationMode.SPOT_COLOR

    @property
    def is_cover_mode(self) -> bool:
        return self.mode == GenerationMode.PHONE_COVER

    def to_dict(self) -> Dict[str, Any]:
        """Forma serializzabile, per loggare cosa è stato generato.

        La maschera dipinta è l'unico campo che non è un numero: asdict ne
        farebbe una copia profonda di qualche megabyte dentro quello che
        dovrebbe essere una riga di log. Nel dizionario ne resta la forma, che
        è tutto ciò che serve per ricostruire cosa è stato passato.
        """
        d = asdict(self)
        pm = d.get('cutout_paint_mask')
        if pm is not None:
            d['cutout_paint_mask'] = f"<mask {getattr(self.cutout_paint_mask, 'shape', '?')}>"
        return d


@dataclass
class GenerationResult:
    """Esito di una generazione: cosa è stato scritto e dove."""
    stl_path: Optional[str] = None
    mf3_path: Optional[str] = None
    companion_path: Optional[str] = None   # bumper/case TPU della modalità cover
    elapsed_s: float = 0.0

    # Il piano di stampa: a quali Z cambiare filamento e con quale colore.
    # Il 3MF se lo porta dentro, ma chi stampa dallo STL ha solo questi numeri
    # per sapere cosa fare, quindi il motore li restituisce invece di
    # lasciarli sepolti nel file.
    color_changes_z: List[float] = field(default_factory=list)
    slot_colors: List[str] = field(default_factory=list)   # '#rrggbb', dal 2° slot

    # Ritaglio: quello che il motore ha scoperto e chi stampa deve sapere.
    # n_pieces > 1 significa che il disegno era in più tronconi e ne è stato
    # tenuto uno solo; ring_attached False che l'occhiello è staccato dal
    # pezzo. Nessuna delle due è un errore di generazione — sono scelte da
    # rivedere, e restano invisibili se il motore non le dichiara.
    cutout_n_pieces: int = 0
    cutout_ring_attached: bool = True

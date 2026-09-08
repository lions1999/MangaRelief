"""Ritaglio della sagoma: decidere quali regioni bianche sono buco e quali no.

Il problema, per un portachiavi, non e' "togliere il bianco". E' che due
regioni bianche indistinguibili — il vuoto fra la corona di nuvole e il
cappello, e l'interno della cupola del cappello — vanno una tolta e una
tenuta. Nessuna proprieta' geometrica le separa: entrambe sono chiuse da un
contorno nero, entrambe sono vuote, entrambe hanno un'area confrontabile.
L'informazione che le distingue *non e' nell'immagine*, sta in chi guarda e
sa cos'e' un cappello.

Quindi qui non si indovina. Si segmenta, si applica la regola che copre la
maggioranza dei casi, e si lascia correggere il resto con un click:

    e' buco solo cio' che tocca il bordo dell'immagine.

Su un disegno al tratto centrato questo taglia lo sfondo esterno e tiene tutto
il resto, cioe' sbaglia solo sulle regioni chiuse che vanno bucate — che si
contano sulle dita di una mano. Le correzioni arrivano come *semi*: coppie
(x, y) che nominano una regione. Sono coordinate, non raster, quindi i
parametri di generazione restano serializzabili e la stessa scelta si rifa'
identica sul motore e nell'anteprima.

Due cose non ovvie, che sono il motivo per cui questo file e' corto:

**L'inchiostro non appartiene a nessuna regione.** La maschera finale e' il
complemento delle sole regioni marcate "buco" (`mask_from_regions`), non
l'unione di quelle marcate "pieno". Cosi' ogni pixel di tratto resta materiale
qualunque cosa succeda, e diventa la parete verticale del pezzo: il bordo del
portachiavi *e'* il contorno del disegno.

**La sigillatura non mangia il disegno.** `seal_px` dilata il tratto prima di
segmentare, per impedire allo sfondo di colare dentro da un varco di uno o due
pixel in un contorno non perfettamente chiuso. Ma quella dilatazione esce
dalle regioni, non dalla maschera: siccome la maschera e' il complemento dei
buchi, l'effetto e' semmai un filo di materiale in piu' sul bordo dei buchi,
mai di meno. Nella direzione giusta, per una parete che deve stampare.

L'alternativa alla segmentazione e' `mask_from_paint`: la maschera dipinta a
mano. Li' non serve nessuna connettivita', perche' chi ha dipinto ha gia'
risposto alla domanda.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Risoluzione a cui si segmenta e si mostra l'anteprima. I semi sono espressi
# in questo raster, quindi il valore fa parte del contratto fra interfaccia e
# motore: se cambia, cambiano le coordinate dei click salvati. Viaggia dentro
# GenerationParams (cutout_seg_res) proprio per non essere una costante che i
# due lati assumono in silenzio.
SEG_MAX_RES = 1600

# Un pixel e' inchiostro se e' piu' scuro di white_clip OPPURE abbastanza
# saturo: senza il secondo termine un accento pieno e luminoso (il rosso della
# fascia del cappello, il giallo di una maschera dipinta) sarebbe carta.
SAT_INK = 40

# Soglia della maschera dipinta: e' carta solo cio' che e' bianco su tutti e
# tre i canali. Il giallo (255, 255, 0) ha il canale blu a zero, quindi passa.
PAINT_WHITE = 235

# Chiusura del tratto prima di segmentare, in pixel del raster di
# segmentazione. A 1600px su un pezzo da 60mm valgono 0,075mm: irrilevanti per
# la geometria, sufficienti a chiudere i varchi di antialiasing.
SEAL_PX = 2


# ---------------------------------------------------------------------------
# Segmentazione in regioni
# ---------------------------------------------------------------------------

@dataclass
class RegionMap:
    """Le regioni bianche di un'immagine, piu' l'inchiostro che le separa.

    labels: 0 = inchiostro (mai un buco), 1..count = una regione ciascuno.
    """
    labels: np.ndarray            # int32 (h, w) sul raster di segmentazione
    count: int
    touches_border: np.ndarray    # bool (count+1,)
    areas: np.ndarray             # int64 (count+1,)
    shape: Tuple[int, int]        # (h, w) del raster di segmentazione
    src_shape: Tuple[int, int]    # (h, w) dell'immagine sorgente

    def region_at(self, x: int, y: int) -> int:
        """La regione sotto un click, in coordinate del raster di
        segmentazione. 0 se il click e' finito sul tratto."""
        h, w = self.shape
        if not (0 <= int(y) < h and 0 <= int(x) < w):
            return 0
        return int(self.labels[int(y), int(x)])


def seg_shape_for(src_shape, seg_res: int = SEG_MAX_RES) -> Tuple[int, int]:
    """La forma del raster di segmentazione per una sorgente data.

    Dipende solo dalla forma sorgente e dal cap, mai dai parametri di
    classificazione: e' cio' che rende stabili le coordinate dei semi mentre
    si trascina White Clip.
    """
    h, w = int(src_shape[0]), int(src_shape[1])
    s = min(1.0, float(seg_res) / max(h, w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def _as_rgb(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def to_seg_raster(img: np.ndarray, seg_res: int = SEG_MAX_RES) -> np.ndarray:
    """Porta un'immagine sul raster di segmentazione (RGB)."""
    rgb = _as_rgb(img)
    h, w = seg_shape_for(rgb.shape[:2], seg_res)
    if (h, w) != rgb.shape[:2]:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb


def ink_mask(image_rgb: np.ndarray, white_clip: int = 235,
             sat_ink: int = SAT_INK) -> np.ndarray:
    """True dove c'e' segno: tratto, retino, campitura colorata."""
    img = np.ascontiguousarray(_as_rgb(image_rgb), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    return (gray < int(white_clip)) | (sat >= int(sat_ink))


def segment_regions(image: np.ndarray, white_clip: int = 235,
                    seal_px: int = SEAL_PX,
                    seg_res: int = SEG_MAX_RES) -> RegionMap:
    """Divide la carta nelle regioni delimitate dal tratto.

    Connettivita' 4 e non 8: con la 8 una diagonale di pixel di tratto non
    separa le due regioni che sta dividendo a vista, e lo sfondo cola dentro
    il disegno attraverso ogni contorno inclinato.
    """
    src_shape = tuple(image.shape[:2])
    rgb = to_seg_raster(image, seg_res)
    ink = ink_mask(rgb, white_clip)

    if seal_px > 0:
        k = 2 * int(seal_px) + 1
        ink = cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)

    paper = (~ink).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    # cv2 assegna 0 ai pixel a zero dell'ingresso, cioe' esattamente
    # l'inchiostro: le regioni sono 1..count-1.
    n_regions = count - 1

    touches = np.zeros(count, dtype=bool)
    for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        touches[np.unique(edge)] = True
    touches[0] = False

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)

    return RegionMap(labels=labels.astype(np.int32), count=n_regions,
                     touches_border=touches, areas=areas,
                     shape=tuple(rgb.shape[:2]), src_shape=src_shape)


def default_cut_flags(regions: RegionMap) -> np.ndarray:
    """La regola automatica: buco solo cio' che tocca il bordo."""
    cut = regions.touches_border.copy()
    cut[0] = False
    return cut


def resolve_cut_flags(regions: RegionMap,
                      cut_seeds: Sequence[Tuple[int, int]] = (),
                      keep_seeds: Sequence[Tuple[int, int]] = ()) -> np.ndarray:
    """La regola automatica, piu' le correzioni dell'utente.

    I "tieni" si applicano dopo i "taglia": se lo stesso punto compare in
    entrambe le liste vince il pieno, che e' l'esito recuperabile — un pezzo
    di troppo si toglie con un altro click, un buco stampato no.
    """
    cut = default_cut_flags(regions)
    for x, y in cut_seeds:
        lb = regions.region_at(x, y)
        if lb > 0:
            cut[lb] = True
    for x, y in keep_seeds:
        lb = regions.region_at(x, y)
        if lb > 0:
            cut[lb] = False
    return cut


def mask_from_regions(regions: RegionMap, cut_flags: np.ndarray) -> np.ndarray:
    """La maschera del materiale: tutto tranne le regioni marcate buco.

    Il complemento, non l'unione dei pieni — vedi il commento in testa al
    modulo: e' cio' che tiene l'inchiostro sempre dentro il pezzo.
    """
    return ~cut_flags[regions.labels]


def mask_from_paint(paint_image: np.ndarray, shape: Tuple[int, int],
                    white: int = PAINT_WHITE) -> np.ndarray:
    """Opzione B: la maschera dipinta a mano.

    Qui non serve nessuna connettivita' e nessuna regola: chi ha dipinto ha
    gia' risposto. E' materiale tutto cio' che non e' carta bianca — il colore
    steso sopra e il tratto nero che lo delimita — ed e' buco tutto il bianco,
    quello esterno e quello racchiuso allo stesso modo. E' esattamente la
    proprieta' che la segmentazione non puo' avere, ed e' il motivo per cui
    questa strada resta come scorciatoia per i casi che il click non risolve.
    """
    rgb = np.ascontiguousarray(_as_rgb(paint_image), np.uint8)
    h, w = int(shape[0]), int(shape[1])
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return rgb.min(axis=2) < int(white)


# ---------------------------------------------------------------------------
# Pulizia, bordo, anello
# ---------------------------------------------------------------------------

def _components(binary: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    return n, labels, stats[:, cv2.CC_STAT_AREA]


def cleanup_mask(mask: np.ndarray, min_area_px: int,
                 keep_largest: bool = True) -> Tuple[np.ndarray, int]:
    """Toglie il pulviscolo e, se richiesto, tiene un pezzo solo.

    Ritorna (maschera, numero di pezzi prima dell'eventuale scarto): il
    conteggio serve a dire all'utente che il disegno *era* in piu' tronconi,
    informazione che sparirebbe tenendo il maggiore in silenzio.
    """
    mask = mask.astype(bool)
    min_area_px = max(1, int(min_area_px))

    # Isole di materiale troppo piccole per stampare: coriandoli sul piatto.
    n, labels, areas = _components(mask)
    n_pieces = max(0, n - 1)
    if n > 1:
        drop = np.zeros(n, dtype=bool)
        drop[1:] = areas[1:] < min_area_px
        drop[0] = False
        if drop.any():
            mask = mask & ~drop[labels]

    # Buchi piu' piccoli della soglia: sbavature del pennello o antialiasing,
    # non fori voluti. Si richiudono.
    n_h, labels_h, areas_h = _components(~mask)
    if n_h > 1:
        fill = np.zeros(n_h, dtype=bool)
        fill[1:] = areas_h[1:] < min_area_px
        # Un buco che tocca il bordo e' lo sfondo, non un buco: mai riempirlo.
        for edge in (labels_h[0, :], labels_h[-1, :], labels_h[:, 0], labels_h[:, -1]):
            fill[np.unique(edge)] = False
        fill[0] = False
        if fill.any():
            mask = mask | fill[labels_h]

    if keep_largest:
        n2, labels2, areas2 = _components(mask)
        n_pieces = max(0, n2 - 1)
        if n2 > 2:
            biggest = 1 + int(np.argmax(areas2[1:]))
            mask = (labels2 == biggest)

    return mask, n_pieces


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Il bordino da sticker: allarga la sagoma verso l'esterno."""
    px = int(px)
    if px <= 0:
        return mask
    k = 2 * px + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _disc(shape, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def add_ring(mask: np.ndarray, cx: float, cy: float, hole_r_px: float,
             rim_px: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Aggiunge l'occhiello: una corona di materiale meno il foro.

    Ritorna anche la corona stessa, e non e' un di piu': la sagoma dice solo
    DOVE c'e' materiale, l'altezza gliela da' la classificazione del disegno
    sotto — e sotto la corona, quando sporge dal soggetto, c'e' carta bianca.
    Ne esce un anello alto un layer, cioe' la parte piu' fragile del pezzo
    messa esattamente dove lo si tira. Chi chiama usa questa maschera per
    dipingerla come inchiostro, e allora sale a tutta altezza col tratto.

    Ritorna inoltre se l'occhiello e' rimasto attaccato al pezzo. Non lo
    aggiusta da solo: spostarlo di autorita' sarebbe peggio che dirlo, perche'
    dove va l'anello lo sa solo chi guarda il disegno.
    """
    outer_r = float(hole_r_px) + float(rim_px)
    boss = _disc(mask.shape, cx, cy, outer_r)
    hole = _disc(mask.shape, cx, cy, float(hole_r_px))
    corona = boss & ~hole

    # Attaccato = la corona tocca il materiale che c'era gia'. Si guarda prima
    # di unire, altrimenti la risposta e' sempre si'.
    attached = bool((corona & mask).any())

    return (mask | boss) & ~hole, corona, attached


def mask_bbox(mask: np.ndarray, pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) semiaperto sul materiale, con margine. None se vuota."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    h, w = mask.shape
    y0 = max(0, int(rows[0]) - pad)
    y1 = min(h, int(rows[-1]) + 1 + pad)
    x0 = max(0, int(cols[0]) - pad)
    x1 = min(w, int(cols[-1]) + 1 + pad)
    return y0, y1, x0, x1


def crop(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return arr[y0:y1, x0:x1]


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Riporta una maschera booleana su un'altra risoluzione.

    INTER_NEAREST e non area: una maschera interpolata ha valori intermedi, e
    qualunque soglia si scelga per riportarla a booleana sposta il bordo di
    mezzo pixel in modo non uniforme lungo il contorno.
    """
    h, w = int(shape[0]), int(shape[1])
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


# ---------------------------------------------------------------------------
# Il ritaglio completo
# ---------------------------------------------------------------------------

@dataclass
class CutoutResult:
    """La sagoma e cosa c'e' da sapere per raccontarla a chi stampa."""
    mask: np.ndarray                   # sul raster di segmentazione
    bbox: Optional[Tuple[int, int, int, int]]
    regions: Optional[RegionMap]
    pitch_mm: float = 0.0              # mm per pixel del raster segmentazione
    n_pieces: int = 0                  # tronconi trovati prima dello scarto
    ring_attached: bool = True
    # La corona dell'occhiello, gia' intersecata con la sagoma finale: serve a
    # chi genera per dipingerla come inchiostro e portarla a tutta altezza.
    ring_mask: Optional[np.ndarray] = None
    empty: bool = False


def compute_cutout(image: np.ndarray, *,
                   max_dim: float = 200.0,
                   white_clip: int = 235,
                   seg_res: int = SEG_MAX_RES,
                   paint_mask: Optional[np.ndarray] = None,
                   cut_seeds: Sequence[Tuple[int, int]] = (),
                   keep_seeds: Sequence[Tuple[int, int]] = (),
                   border_mm: float = 0.0,
                   min_feature_mm: float = 0.8,
                   keep_largest: bool = True,
                   ring_xy: Optional[Tuple[int, int]] = None,
                   ring_d_mm: float = 4.0,
                   ring_rim_mm: float = 2.0,
                   regions: Optional[RegionMap] = None) -> CutoutResult:
    """Dalla sorgente alla sagoma, sul raster di segmentazione.

    Il passaggio in millimetri non e' immediato: `max_dim` misura il *pezzo
    ritagliato*, non l'immagine di partenza, quindi il passo in mm/pixel si
    conosce solo dopo aver saputo quanto e' grande la sagoma — e la sagoma
    dipende dal bordino e dall'occhiello, che sono espressi in millimetri.
    Si stima il passo sulla sagoma nuda e si rifa' il conto con quello vero:
    due giri bastano, il terzo cambierebbe le cifre dopo la virgola.

    `regions`: una segmentazione gia' calcolata, per non rifarla a ogni click.
    """
    seg_h, seg_w = seg_shape_for(image.shape[:2], seg_res)

    if paint_mask is not None:
        regions = None
        base = mask_from_paint(paint_mask, (seg_h, seg_w))
    else:
        # `regions` puo' arrivare gia' pronto: l'anteprima ne rifa' una a ogni
        # click, e segmentare da capo ogni volta renderebbe il click lento
        # abbastanza da sembrare rotto.
        if regions is None:
            regions = segment_regions(image, white_clip=white_clip, seg_res=seg_res)
        base = mask_from_regions(regions, resolve_cut_flags(regions, cut_seeds, keep_seeds))

    if not base.any():
        return CutoutResult(mask=base, bbox=None, regions=regions, empty=True)

    result = None
    pitch = float(max_dim) / max(seg_h, seg_w)   # prima stima: l'immagine intera
    for _ in range(2):
        mask = base
        if border_mm > 0:
            mask = dilate_mask(mask, int(round((border_mm / 2.0) / pitch)))

        attached, corona = True, None
        if ring_xy is not None:
            mask, corona, attached = add_ring(
                mask, float(ring_xy[0]), float(ring_xy[1]),
                hole_r_px=(ring_d_mm / 2.0) / pitch,
                rim_px=ring_rim_mm / pitch)

        min_area = max(2, int(np.pi * ((min_feature_mm / 2.0) / pitch) ** 2))
        mask, n_pieces = cleanup_mask(mask, min_area, keep_largest=keep_largest)

        bbox = mask_bbox(mask)
        if bbox is None:
            return CutoutResult(mask=mask, bbox=None, regions=regions, empty=True)

        y0, y1, x0, x1 = bbox
        pitch = float(max_dim) / max(y1 - y0, x1 - x0)
        # La corona va intersecata con la sagoma DOPO la pulizia: se
        # keep_largest ha scartato un occhiello staccato, li' non c'e' piu'
        # niente da dipingere.
        result = CutoutResult(mask=mask, bbox=bbox, regions=regions,
                              pitch_mm=pitch, n_pieces=n_pieces,
                              ring_attached=attached,
                              ring_mask=(corona & mask) if corona is not None else None)

    return result


# ---------------------------------------------------------------------------
# Anteprima
# ---------------------------------------------------------------------------

def overlay_preview(image: np.ndarray, mask: np.ndarray,
                    tint=(255, 216, 0), alpha: float = 0.45,
                    void=(255, 255, 255)) -> np.ndarray:
    """L'immagine con addosso la risposta: tinto cio' che stampa, bianco il vuoto.

    Sta nel motore e non nell'interfaccia perche' e' solo numpy, e perche' un
    frontend web deve poter mostrare la stessa cosa senza riscriverla.
    """
    rgb = _as_rgb(image).astype(np.float32)
    m = resize_mask(mask, rgb.shape[:2])

    tinted = rgb * (1.0 - alpha) + np.array(tint, np.float32) * alpha
    out = np.where(m[..., None], tinted, np.array(void, np.float32))
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))

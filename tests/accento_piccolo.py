"""L'auto-detect degli accenti su una vignetta quasi tutta in bianco e nero.

Il caso tipico dello Spot Color non e' un'illustrazione a colori: e' una
tavola in bianco e nero con UNA cosa colorata — due occhi, una fascia, un
logo. Quella cosa occupa lo 0,2% dell'immagine, e la soglia che decideva se
proporre un accento ne chiedeva lo 0,5%: scartava esattamente il caso per cui
la modalita' esiste.

La soglia serviva a non proporre accenti sul rumore di croma del JPEG. Ma
quel lavoro lo fa gia' la maschera dei pixel vividi (S>=90 e V>=60): un grigio
non arriva a quella saturazione per quanto lo si comprima. Questa prova fissa
entrambe le meta' della calibrazione, perche' abbassare la soglia e' utile
solo finche' i falsi positivi restano zero:

  - un accento piccolo ma voluto viene trovato
  - una tavola davvero in bianco e nero non ne produce, a nessuna qualita'

    python tests/accento_piccolo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import cv2
import numpy as np

fails = []


def check(n, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + n + ((" | " + str(extra)) if extra else ""))
    if not ok:
        fails.append(n)


from engine.color_utils import suggest_spot_accents

W, H = 736, 260
GIALLO = (255, 190, 0)


def _jpeg(img, q):
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _sfondo():
    """Vignetta: carta bianca, cornice nera, una massa nera di capelli."""
    img = np.full((H, W, 3), 250, np.uint8)
    cv2.rectangle(img, (95, 60), (645, 240), (255, 255, 255), -1)
    cv2.rectangle(img, (95, 60), (645, 240), (0, 0, 0), 7)
    cv2.fillPoly(img, [np.array([[250, 240], [270, 90], [300, 140], [330, 70],
                                 [360, 150], [400, 80], [440, 240]])], (10, 10, 10))
    return img


def con_occhi(rx, ry, q=None):
    img = _sfondo()
    for cx in (320, 378):
        cv2.ellipse(img, (cx, 175), (rx, ry), 0, 0, 360, GIALLO, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, 175), (max(3, rx // 3), max(4, ry - 1)), 0, 0, 360,
                    (0, 0, 0), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, 175), (rx, ry), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
    return _jpeg(img, q) if q else img


def senza_colore(q=None, rumore=0):
    img = _sfondo()
    for x in range(120, 620, 7):
        cv2.line(img, (x, 80), (x + 18, 230), (0, 0, 0), 2, cv2.LINE_AA)
    if rumore:
        rng = np.random.RandomState(0)
        img = np.clip(img.astype(np.int16)
                      + rng.randint(-rumore, rumore + 1, img.shape), 0, 255).astype(np.uint8)
    return _jpeg(img, q) if q else img


def quota(img):
    """Che frazione dell'immagine occupa il colore: il numero che conta."""
    hsv = cv2.cvtColor(np.ascontiguousarray(img, np.uint8), cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] >= 90) & (hsv[..., 2] >= 60)).mean()


def giallo(accenti):
    """Almeno un accento e' il giallo degli occhi (tinta entro ~20 gradi)."""
    if not accenti:
        return False
    h_att = cv2.cvtColor(np.uint8([[GIALLO]]), cv2.COLOR_RGB2HSV)[0, 0, 0]
    for a in accenti:
        h = cv2.cvtColor(np.uint8([[a]]), cv2.COLOR_RGB2HSV)[0, 0, 0]
        if min(abs(int(h) - int(h_att)), 180 - abs(int(h) - int(h_att))) <= 10:
            return True
    return False


print("\n=== l'accento piccolo va trovato ===")
for rx, ry, q in ((22, 13, None), (14, 9, None), (14, 9, 85), (10, 7, 75)):
    img = con_occhi(rx, ry, q)
    acc = suggest_spot_accents(img)
    check(f"occhi {rx}x{ry}{' JPEG ' + str(q) if q else ''} "
          f"({100*quota(img):.2f}% dell'immagine): trovato il giallo",
          giallo(acc), acc)

# Il limite dichiarato: sotto il pavimento non si propone, ed e' giusto cosi'
# — una macchia di quelle dimensioni potrebbe essere un timbro o una velatura.
minuscoli = con_occhi(6, 4)
check("sotto il pavimento non propone nulla (limite voluto)",
      not suggest_spot_accents(minuscoli), f"{100*quota(minuscoli):.3f}%")


print("\n=== e il bianco e nero non deve produrne ===")
for q, r in ((None, 0), (95, 0), (85, 6), (75, 6), (60, 14), (45, 14), (30, 14)):
    img = senza_colore(q, r)
    acc = suggest_spot_accents(img)
    check(f"B/N {'originale' if not q else f'JPEG {q}'} rumore ±{r}: nessun accento",
          acc == [], f"{acc} (vividi {100*quota(img):.4f}%)")


print("\n=== niente rumore di libreria in console ===")
# KMeans avvisa se i cluster distinti sono meno di quelli chiesti: succede con
# un accento a tinta piatta, cioe' nel caso piu' comune. Il warning non e' un
# errore, ma stampato a ogni click fa sembrare rotta una cosa che funziona.
with warnings.catch_warnings(record=True) as catturati:
    warnings.simplefilter("always")
    suggest_spot_accents(con_occhi(14, 9))
    rumorosi = [w for w in catturati if "cluster" in str(w.message).lower()]
check("nessun avviso di convergenza da KMeans", not rumorosi,
      [str(w.message)[:60] for w in rumorosi])


print()
if fails:
    print(f"❌ {len(fails)} FALLITE: " + ", ".join(fails))
    sys.exit(1)
print("✅ tutte passate")

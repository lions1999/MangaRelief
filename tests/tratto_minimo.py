"""Il tratto più fine in millimetri: che il numero sia quello vero.

La misura serve a rispondere prima della stampa a una domanda che altrimenti
si scopre solo nello slicer — a questa dimensione il tratto di *questo*
disegno esiste? — quindi l'unica cosa che conta e' che sia calibrata. Un
avvertimento che sbaglia numero e' peggio di nessun avvertimento: lo si impara
a ignorare.

Le prove sono tutte su rapporti noti, non su valori assoluti: un tratto largo
il doppio deve misurare il doppio, e la dimensione consigliata deve davvero
portare quel tratto sopra la soglia. Cosi' la prova non dipende da come
OpenCV disegna una linea spessa 2.

Il caso che si rompe in silenzio e' il passo mm/pixel per modalita': nel
portachiavi la sorgente viene ritagliata alla sagoma, quindi il pezzo e' piu'
piccolo del foglio e i tratti sono piu' grossi di quanto direbbe il conto
fatto sul foglio. Se qualcuno riporta quel conto sull'immagine intera, il
numero resta plausibile — e' solo sbagliato.

    python tests/tratto_minimo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile

import cv2
import numpy as np

fails = []


def check(n, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + n + ((" | " + str(extra)) if extra else ""))
    if not ok:
        fails.append(n)


from engine.color_utils import NOZZLE_MM, SOLID_MM, feature_scale

H, W = 488, 610
TMP = tempfile.mkdtemp(prefix="tratto_")


def tavola(spessore, passo=None):
    """Barre verticali larghe ESATTAMENTE `spessore` pixel.

    Disegnate a fette di array e non con cv2.line, che con thickness=2 ne
    stende 3: con quelle, i rapporti che questa prova verifica non sarebbero
    quelli che ha disegnato, e il primo giro e' fallito proprio cosi'.

    Spessori dispari, perche' la misura ricava w da 2·d−1 e sui pari
    sottostima di un pixel (per prudenza, vedi feature_scale): con 3 e 9 il
    rapporto atteso e' esattamente 3.
    """
    img = np.full((H, W), 255, np.uint8)
    for x in range(80, 540, passo or 40):
        img[90:400, x:x + spessore] = 0
    return img


# --- calibrazione: rapporti noti -------------------------------------------
print("\n=== la misura segue la scala fisica ===")

fine = tavola(3)
a = feature_scale(fine, 60.0 / W)['ink_mm']
b = feature_scale(fine, 120.0 / W)['ink_mm']
check("raddoppiando Max Dim il tratto misura il doppio",
      abs(b / a - 2.0) < 0.02, f"{a:.3f} -> {b:.3f} mm")

grosso = tavola(9)   # il tratto di un disegno pensato per stampare piccolo
c = feature_scale(grosso, 60.0 / W)['ink_mm']
check("un tratto tre volte piu' spesso misura esattamente il triplo",
      abs(c / a - 3.0) < 0.05, f"{a:.3f} vs {c:.3f} mm (rapporto {c/a:.3f})")

# --- il consiglio deve funzionare davvero ----------------------------------
print("\n=== la dimensione consigliata mantiene la promessa ===")

r60 = feature_scale(fine, 60.0 / W, max_dim_mm=60.0)
check("a taglia portachiavi il tratto fine e' sotto l'ugello",
      r60['ink_mm'] < NOZZLE_MM and not r60['ok'],
      f"{r60['ink_mm']:.2f} mm, consiglia {r60['min_dim_mm']:.0f} mm")

consigliata = r60['min_dim_mm']
rc = feature_scale(fine, consigliata / W, max_dim_mm=consigliata)
check("...e a quella dimensione il tratto e' davvero una parete solida",
      rc['ink_mm'] >= SOLID_MM - 1e-6 and rc['ok'],
      f"{rc['ink_mm']:.2f} mm >= {SOLID_MM}")

r120 = feature_scale(fine, 120.0 / W, max_dim_mm=120.0)
check("il consiglio non dipende da dove si parte",
      abs(r120['min_dim_mm'] - consigliata) < 1.0,
      f"{consigliata:.0f} vs {r120['min_dim_mm']:.0f} mm")

# Un disegno a tratto grosso — quello che a un portachiavi serve — deve
# passare a 60 mm senza avvertimenti, altrimenti l'avviso urla sempre e si
# smette di leggerlo.
rg = feature_scale(grosso, 60.0 / W, max_dim_mm=60.0)
check("il tratto grosso passa a 60 mm senza avvertimenti",
      rg['ok'] and rg['ink_mm'] >= SOLID_MM, rg['message'])

# --- il secondo modo di fallire: i vuoti ------------------------------------
print("\n=== i tratti che si fondono ===")

fitto = tavola(3, passo=7)
rado = tavola(3, passo=40)
rf = feature_scale(fitto, 60.0 / W, max_dim_mm=60.0)
rr = feature_scale(rado, 60.0 / W, max_dim_mm=60.0)
check("il tratteggio fitto avvisa che i vuoti si chiudono",
      rf['gap_mm'] < NOZZLE_MM and "merge into a solid area" in rf['message'],
      f"vuoti {rf['gap_mm']:.2f} mm")
check("quello rado no, a parita' di spessore del tratto",
      rr['gap_mm'] > NOZZLE_MM, f"vuoti {rr['gap_mm']:.2f} mm")

# --- niente da misurare -----------------------------------------------------
print("\n=== casi degeneri ===")
for nome, im in (("tutto bianco", np.full((100, 100), 255, np.uint8)),
                 ("tutto nero", np.zeros((100, 100), np.uint8))):
    r = feature_scale(im, 0.1)
    check(f"{nome}: non esplode e lo dice",
          r['ink_mm'] is None or r['gap_mm'] is None, r['message'])


# --- il passo mm/pixel per modalita', che e' la parte fragile ---------------
print("\n=== il passo e' quello del PEZZO, non del foglio ===")

from PyQt6.QtWidgets import QApplication, QFileDialog

from manga_to_3d import Manga3DAppController

# Disegno a tratto grosso che occupa un terzo del foglio: sul foglio intero
# i suoi tratti sembrano fini, sul pezzo ritagliato sono quello che sono.
piccolo = np.full((H, W), 255, np.uint8)
cv2.circle(piccolo, (300, 240), 90, 0, 8)    # il corpo del pezzo
cv2.circle(piccolo, (300, 240), 40, 0, 1)    # un tratto interno finissimo
path = os.path.join(TMP, "piccolo.png")
cv2.imwrite(path, piccolo)

app = QApplication.instance() or QApplication(sys.argv)
win = Manga3DAppController()
win.show()
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))
win.load_image()

win.mode_selector.setCurrentIndex(0)
win.spin_dim.setValue(60.0)
win._do_refresh_feature_scale()
testo_std = win.lbl_feature_scale.text()
img_std, passo_std = win._feature_scale_source()

win.mode_selector.setCurrentIndex(5)
win.spin_dim.setValue(60.0)
win._do_refresh_feature_scale()
testo_key = win.lbl_feature_scale.text()
img_key, passo_key = win._feature_scale_source()

check("in portachiavi il passo mm/px e' piu' grosso (il pezzo e' meno del foglio)",
      passo_key > passo_std * 1.5, f"{passo_std:.4f} vs {passo_key:.4f} mm/px")
check("quindi lo stesso disegno ha tratti piu' larghi a parita' di Max Dim",
      feature_scale(img_key, passo_key)['ink_mm']
      > feature_scale(img_std, passo_std)['ink_mm'] * 1.5,
      f"{testo_std}  /  {testo_key}")

# Il consiglio in mm deve restare confrontabile con la manopola che l'utente
# gira: se si derivasse dal lato lungo della sorgente, in portachiavi
# uscirebbe gonfiato del rapporto fra foglio e pezzo.
r = feature_scale(img_key, passo_key, max_dim_mm=60.0)
if r['min_dim_mm'] is not None and r['ink_mm'] < SOLID_MM:
    atteso = 60.0 * SOLID_MM / r['ink_mm']
    check("il consiglio e' calcolato su Max Dim, non sul lato lungo della sorgente",
          abs(r['min_dim_mm'] - atteso) < 0.5,
          f"{r['min_dim_mm']:.1f} vs {atteso:.1f} mm")
else:
    check("il consiglio e' calcolato su Max Dim, non sul lato lungo della sorgente",
          r['ok'], "tratto gia' solido, niente consiglio da dare")

# L'etichetta deve accendersi in arancio solo quando c'e' un problema.
win.mode_selector.setCurrentIndex(0)
# Gli estremi della manopola, non due numeri a caso: il tratto interno della
# fixture e' largo un pixel, quindi diventa una parete solida solo vicino al
# fondo scala — ed e' li' che l'avviso deve tacere.
win.spin_dim.setValue(win.spin_dim.minimum())
win._do_refresh_feature_scale()
stato_piccolo = win.lbl_feature_scale.property("state")
win.spin_dim.setValue(win.spin_dim.maximum())
win._do_refresh_feature_scale()
stato_grande = win.lbl_feature_scale.property("state")
check("l'etichetta avvisa in piccolo e tace in grande",
      stato_piccolo == "warn" and stato_grande != "warn",
      f"{stato_piccolo!r} -> {stato_grande!r}")
check("ed e' visibile appena c'e' un'immagine", win.lbl_feature_scale.isVisible())


# ---------------------------------------------------------------------------
print("\n=== ingrossare il tratto ===")

from engine.color_utils import thicken_applied_mm, thicken_ink

# Il rimedio a un tratto troppo fine. Il contrasto non c'entra: sposta i
# grigi, la geometria resta dov'e' — questa e' la ragione per cui il cursore
# esiste, e vale la pena che una prova lo fissi.
sottile = tavola(1, passo=40)
passo_mm = 60.0 / W

contrastata = np.where(sottile < 128, 0, 255).astype(np.uint8)
check("il contrasto non cambia la larghezza del tratto",
      abs(feature_scale(contrastata, passo_mm)['ink_mm']
          - feature_scale(sottile, passo_mm)['ink_mm']) < 1e-9)

prima = feature_scale(sottile, passo_mm, max_dim_mm=60.0)
check("un tratto da 1 px a 60 mm e' sotto l'ugello",
      prima['ink_mm'] < NOZZLE_MM, f"{prima['ink_mm']:.2f} mm")

# L'ingrossamento chiesto e quello ottenuto differiscono per l'arrotondamento
# a pixel interi, e il secondo e' quello che va mostrato: un cursore che
# dichiara un numero e ne applica un altro fa sembrare rotta la misura.
for chiesto in (0.10, 0.25, 0.50, 1.00):
    reale = thicken_applied_mm(chiesto, passo_mm)
    dopo = feature_scale(thicken_ink(sottile, chiesto, passo_mm), passo_mm)
    cresciuto = (dopo['ink_mm'] - prima['ink_mm']) if dopo['ink_mm'] else 0.0
    check(f"chiedendo +{chiesto:.2f} mm ne applica {reale:.2f} e li aggiunge davvero",
          abs(cresciuto - reale) < 0.02,
          f"misurato +{cresciuto:.2f} mm")

dopo = feature_scale(thicken_ink(sottile, 1.00, passo_mm), passo_mm, max_dim_mm=60.0)
check("ingrossando, lo stesso disegno diventa stampabile a 60 mm",
      dopo['ok'] and dopo['ink_mm'] >= SOLID_MM,
      f"{prima['ink_mm']:.2f} -> {dopo['ink_mm']:.2f} mm")
check("e la dimensione consigliata crolla",
      dopo['min_dim_mm'] < prima['min_dim_mm'] / 2,
      f"{prima['min_dim_mm']:.0f} -> {dopo['min_dim_mm']:.0f} mm")

# Il prezzo, che va detto e non nascosto: dove i tratti sono piu' vicini del
# doppio dell'ingrossamento si fondono. Su un retino e' spesso l'esito giusto,
# ma deve essere una scelta, non una sorpresa.
fitto2 = tavola(1, passo=5)
vuoti_prima = feature_scale(fitto2, passo_mm)['gap_mm']
fuso = thicken_ink(fitto2, 1.00, passo_mm)
check("un tratteggio fitto ingrossato diventa una campitura piena",
      (fuso < 128).mean() > (fitto2 < 128).mean() * 2.5,
      f"inchiostro {100*(fitto2<128).mean():.0f}% -> {100*(fuso<128).mean():.0f}%")
check("(e prima i vuoti c'erano)", vuoti_prima > 0)

# --- il cursore nella UI ----------------------------------------------------
win.mode_selector.setCurrentIndex(0)
win.spin_dim.setValue(60.0)
QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (os.path.join(TMP, "sottile.png"), ""))
cv2.imwrite(os.path.join(TMP, "sottile.png"), sottile)
win.load_image()

win.slider_line_thicken.setValue(0)
win._do_refresh_feature_scale()
check("a zero il cursore dice di essere spento",
      win.lbl_line_thicken.text().endswith("off"), win.lbl_line_thicken.text())
stato_zero = win.lbl_feature_scale.property("state")

win.slider_line_thicken.setValue(20)
win._do_refresh_feature_scale()
_img, _passo = win._feature_scale_source()
atteso = thicken_applied_mm(1.00, _passo)
check("l'etichetta mostra l'ingrossamento APPLICATO, non quello chiesto",
      f"+{atteso:.2f} mm" in win.lbl_line_thicken.text(),
      f"{win.lbl_line_thicken.text()} (atteso +{atteso:.2f} mm)")
check("e l'avviso sul tratto si spegne di conseguenza",
      stato_zero == "warn" and win.lbl_feature_scale.property("state") != "warn",
      f"{stato_zero!r} -> {win.lbl_feature_scale.property('state')!r}")

# E la generazione deve usare lo stesso valore dell'anteprima, o il cursore
# regola solo l'immagine sullo schermo.
import trimesh

from engine import GenerationMode, GenerationParams, generate


def area_inchiostro(mm):
    p = GenerationParams(mode=GenerationMode.STANDARD, max_dim=60.0, base_h=1.0,
                         max_h=2.4, layer_height=0.2, max_res_cap=800,
                         smart_decimate=False, color_mode=2,
                         color_changes_z=[1.4, 2.0, 2.4], line_thicken_mm=mm,
                         output_path=os.path.join(TMP, f"th{mm}.stl"))
    m = trimesh.load(generate(sottile, p).stl_path)
    z = m.vertices[m.faces][:, :, 2]
    return float(m.area_faces[(np.abs(z - z.max()) < 1e-6).all(axis=1)].sum())


a0, a1 = area_inchiostro(0.0), area_inchiostro(1.0)
check("nella mesh generata l'inchiostro cresce davvero", a1 > a0 * 1.8,
      f"{a0:.0f} -> {a1:.0f} mm2")


print()
if fails:
    print(f"❌ {len(fails)} FALLITE: " + ", ".join(fails))
    sys.exit(1)
print("✅ tutte passate")

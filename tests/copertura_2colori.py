"""La copertura a 2 colori: sul motore e sull'interfaccia.

Due domande separate, e nessuna delle due si risponde guardando lo schermo.

Che il parametro cambi davvero la geometria — un cursore che non sposta
niente e' un placebo, e la prima versione di questa prova non se ne sarebbe
accorta perche' misurava la cosa sbagliata. E che il pannello si trasformi
quando la modalita' diventa 2, con l'anteprima che mostra la classificazione
vera invece di somigliarle.

    python tests/copertura_2colori.py

Un paio di minuti: genera quattro mesh vere. Non serve uno schermo, la parte
Qt gira offscreen — ma la finestra va comunque mostrata, altrimenti Qt
considera ogni figlio non visibile e ogni asserzione sulla visibilita'
risponde di no qualunque cosa sia successa.
"""
import os
import sys

# La radice del repo, non un percorso di questa macchina: la prova deve girare
# da dove la si lancia.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Qt senza schermo. Va impostato prima di importare PyQt, non dopo.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2, numpy as np

fails = []
def check(n, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + n + ((" | " + str(extra)) if extra else ""))
    if not ok: fails.append(n)

def tavola():
    """Carta bianca e due retini di densita' diversa: 20% e 60% di inchiostro.

    E' il caso per cui la copertura esiste. Con un retino solo qualunque coppia
    di soglie lo classifica allo stesso modo, e la prova passerebbe senza aver
    provato niente — che e' come era scritta la prima volta.
    """
    h, w = 480, 640
    img = np.full((h, w), 250, np.uint8)
    for x0, spessore in ((60, 4), (360, 12)):      # 4/20 = 20%, 12/20 = 60%
        for x in range(x0, x0 + 220, 20):
            cv2.rectangle(img, (x, 60), (x + spessore, 420), 15, -1)
    return img


def densita(img, x0):
    """Quanta parte di quel blocco e' inchiostro davvero."""
    blocco = img[60:420, x0:x0 + 220]
    return float((blocco < 128).mean())


# ---------------------------------------------------------------- motore
from engine import GenerationParams, GenerationMode, generate
import trimesh, tempfile

img = tavola()
d = tempfile.mkdtemp()
aree = {}
for etichetta, cov in (("senza", None), ("15%", 0.15), ("40%", 0.40), ("70%", 0.70)):
    p = GenerationParams(mode=GenerationMode.STANDARD, color_mode=2,
                         max_dim=60, max_res_cap=300, bw_coverage=cov,
                         output_path=os.path.join(d, f"{etichetta}.stl"),
                         output_path_3mf=os.path.join(d, f"{etichetta}.3mf"))
    r = generate(img, p)
    m = trimesh.load(r.stl_path)
    check(f"motore, copertura {etichetta}: la mesh e' chiusa", m.is_watertight, m.is_watertight)
    z = np.round(m.vertices[:, 2], 3)
    quote = sorted(set(z.tolist()))
    # Tre quote, non due: il fondo piatto a 0, la base, e l'inchiostro in cima.
    # A due colori non esistono terrazze intermedie, ed e' quello il punto.
    check(f"motore, copertura {etichetta}: fondo, base e inchiostro e basta",
          quote == [0.0, p.base_h, p.max_h], quote)

    # L'area vera delle facce in cima, non il numero di vertici: dopo la
    # decimazione i vertici si addensano dove la geometria e' complicata, che
    # e' l'opposto di una misura di quanta superficie stampa nera.
    zf = m.vertices[m.faces][:, :, 2]
    in_cima = np.all(np.abs(zf - z.max()) < 1e-6, axis=1)
    aree[etichetta] = float(m.area_faces[in_cima].sum())

stampa = {k: round(v, 1) for k, v in aree.items()}
print(f"    densita' dei due retini: {densita(img, 60):.0%} e {densita(img, 360):.0%}")
print(f"    area stampata nera: {stampa}")

# Alzando la soglia si stampa meno nero, e di parecchio a ogni scatto.
check("alzare la soglia toglie inchiostro, e si vede",
      aree["15%"] > aree["40%"] * 1.25 > aree["70%"] * 1.25 * 1.25, stampa)
check("e la strada senza copertura e' un'altra ancora",
      abs(aree["senza"] - aree["15%"]) > 1.0, stampa)

# Nero ne resta anche al 70%, e non e' un difetto: la copertura si misura in
# una finestra piccola, quindi l'interno di un tratto spesso e' inchiostro al
# 100% qualunque soglia si scelga. Sparisce il retino, non il segno — che e'
# esattamente cio' che la modalita' a due colori deve preservare.
check("a soglia alta il tratto spesso sopravvive: e' locale, non globale",
      aree["70%"] > 0.0, stampa)

# ------------------------------------------------------------ interfaccia
from PyQt6.QtWidgets import QApplication
from manga_to_3d import Manga3DAppController

app = QApplication.instance() or QApplication(sys.argv)
win = Manga3DAppController()
win.show()          # senza, Qt considera ogni figlio non visibile

win.img_filtered_array = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
win.img_rgb_original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
win.sampled_colors = [245, 200, 120, 15]
win.btn_std_mockup.setEnabled(True)

# --- 4 colori: il pannello e' quello di sempre
win.last_midtone_pct = 40.0
win.slider_threshold.setValue(10)
win._refresh_color_mode()
check("a 4 colori gli swatch ci sono", all(b.isVisible() for b in win.swatches))
check("a 4 colori la copertura non c'e'", not win.slider_bw_coverage.isVisible())
check("a 4 colori non si passa una copertura al motore",
      win._current_bw_coverage() is None, win._current_bw_coverage())

# --- 2 colori: diventa un'altra cosa
win.last_midtone_pct = 1.0
win._refresh_color_mode()
check("a 2 colori gli swatch spariscono", not any(b.isVisible() for b in win.swatches),
      [b.text() for b in win.swatches if b.isVisible()])
check("a 2 colori compare la copertura", win.slider_bw_coverage.isVisible())
check("e il gruppo dice cosa e' diventato",
      "Ink Coverage" in win.group_swatch.title(), win.group_swatch.title())
check("il cursore arriva al motore come frazione",
      win._current_bw_coverage() == win.slider_bw_coverage.value() / 100.0,
      win._current_bw_coverage())
check("l'etichetta dice il valore e cosa conta come inchiostro",
      f"{win.slider_bw_coverage.value()}%" in win.lbl_bw_coverage.text()
      and "darker than" in win.lbl_bw_note.text(), win.lbl_bw_note.text()[:90])

# --- l'anteprima mostra la classificazione, non un'altra immagine
win.btn_std_mockup.setChecked(True)
app.processEvents()
ant = getattr(win, "std_preview_array", None)
check("l'anteprima si accende", ant is not None)
if ant is not None:
    toni = sorted(set(ant[:, :, 0].ravel().tolist()))
    check("a 2 colori mostra due soli toni, quelli campionati",
          toni == [win.sampled_colors[3], win.sampled_colors[0]], toni)

win.last_midtone_pct = 40.0
win._refresh_color_mode()
win._do_refresh_std_mockup()
toni4 = sorted(set(win.std_preview_array[:, :, 0].ravel().tolist()))
check("a 4 colori ne mostra fino a quattro, tutti campionati",
      2 <= len(toni4) <= 4 and set(toni4) <= set(win.sampled_colors), toni4)

check("spegnendola il pulsante torna a proporsi",
      win.btn_std_mockup.text().startswith("👁 Back") , win.btn_std_mockup.text())
win.btn_std_mockup.setChecked(False)
check("e il testo torna quello di partenza",
      win.btn_std_mockup.text() == "👁 Mockup Preview", win.btn_std_mockup.text())

print("\n" + ("TUTTO OK" if not fails else "FALLITI: " + ", ".join(fails)))
sys.exit(1 if fails else 0)

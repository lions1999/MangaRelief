"""Il ritaglio della sagoma: che i buchi siano buchi, e che il pannello lo dica.

La domanda a cui questa prova risponde non e' "il ritaglio toglie il bianco".
E' quella che il ritaglio non puo' risolvere da solo, e che quindi e' l'unica
che vale la pena verificare a macchina:

    due regioni bianche indistinguibili — il vuoto racchiuso fra la corona e
    il cappello, e l'interno della cupola del cappello — devono finire una
    bucata e una piena, e la differenza la fa un click.

La fixture e' costruita apposta perche' le due esistano entrambe. Un disegno
con il solo sfondo esterno passerebbe questa prova senza aver provato niente:
l'automatismo da solo lo risolve, e il click che stiamo verificando non
avrebbe nulla da cambiare.

Le asserzioni sono sulla MESH, non sulla maschera. Che una maschera abbia un
buco e' l'ipotesi; che il buco arrivi fino al file stampabile e' la tesi, e
in mezzo ci sono il ricampionamento sulla griglia della mesh, le pareti
verticali dei fori interni e la decimazione. Si misura con il genere della
superficie (chi = 2 - 2g): una lastra chiusa senza fori ha chi = 2, ogni foro
passante ne toglie 2. E' un numero intero che non si presta a interpretazioni,
al contrario di "quanti pixel bianchi sono rimasti".

    python tests/ritaglio_sagoma.py

Un minuto scarso. La parte Qt gira offscreen ma la finestra va comunque
mostrata: senza un top-level visibile Qt considera ogni figlio invisibile, e
ogni asserzione sulla visibilita' risponde di no qualunque cosa sia successa.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile

import cv2
import numpy as np
import trimesh

fails = []


def check(n, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + n + ((" | " + str(extra)) if extra else ""))
    if not ok:
        fails.append(n)


TMP = tempfile.mkdtemp(prefix="ritaglio_")


def fixture(n=900):
    """Un cappello sotto una corona, ridotto all'osso.

    Tre tipi di regione bianca, che e' tutto il punto:
      A) fuori dalla corona          -> tocca il bordo, va via da sola
      B) fra corona e cupola         -> chiusa, l'automatismo la TIENE, va bucata
      C) dentro la cupola            -> chiusa, va tenuta
    B e C sono la stessa cosa a guardarle. E' per questo che serve il click.
    """
    img = np.full((n, n, 3), 255, np.uint8)
    cv2.circle(img, (450, 400), 370, (0, 0, 0), 14)     # corona
    cv2.circle(img, (450, 560), 200, (0, 0, 0), 14)     # cupola
    for y in range(400, 760, 18):                        # tratteggio dentro la cupola
        cv2.line(img, (300, y), (600, y), (0, 0, 0), 3)
    return img


P_FUORI = (20, 20)      # regione A
P_VUOTO = (450, 120)    # regione B
P_CUPOLA = (450, 590)   # regione C (fra due righe di tratteggio)

IMG = fixture()
IMG_PATH = os.path.join(TMP, "cappello.png")
cv2.imwrite(IMG_PATH, IMG)


# ---------------------------------------------------------------------------
print("\n=== MOTORE: le regioni ===")

from engine.cutout_utils import (compute_cutout, mask_from_regions,
                                 resolve_cut_flags, segment_regions)

R = segment_regions(IMG, white_clip=235)
lb_fuori, lb_vuoto, lb_cupola = (R.region_at(*P_FUORI), R.region_at(*P_VUOTO),
                                 R.region_at(*P_CUPOLA))
check("le tre regioni sono distinte",
      0 not in (lb_fuori, lb_vuoto, lb_cupola)
      and len({lb_fuori, lb_vuoto, lb_cupola}) == 3,
      f"fuori={lb_fuori} vuoto={lb_vuoto} cupola={lb_cupola}")

check("solo la regione esterna tocca il bordo",
      R.touches_border[lb_fuori] and not R.touches_border[lb_vuoto]
      and not R.touches_border[lb_cupola])

m_auto = mask_from_regions(R, resolve_cut_flags(R))
check("auto: fuori tagliato", not m_auto[P_FUORI[1], P_FUORI[0]])
check("auto: cupola tenuta", m_auto[P_CUPOLA[1], P_CUPOLA[0]])
# Questo e' il limite dichiarato dell'automatismo, non un difetto da nascondere:
# se un giorno passasse "tagliato", vorrebbe dire che qualcuno ha aggiunto
# un'euristica che indovina — e allora questa prova va riscritta, non cancellata.
check("auto: il vuoto racchiuso resta PIENO (serve il click)",
      m_auto[P_VUOTO[1], P_VUOTO[0]])

m_click = mask_from_regions(R, resolve_cut_flags(R, cut_seeds=[P_VUOTO]))
check("click: il vuoto e' bucato, la cupola no",
      (not m_click[P_VUOTO[1], P_VUOTO[0]]) and m_click[P_CUPOLA[1], P_CUPOLA[0]])

m_back = mask_from_regions(
    R, resolve_cut_flags(R, cut_seeds=[P_VUOTO], keep_seeds=[P_VUOTO]))
check("il click e' reversibile (keep vince su cut)", m_back[P_VUOTO[1], P_VUOTO[0]])

# White Clip ridisegna i confini ma non la forma del raster: i semi gia'
# raccolti devono restare validi, altrimenti muovere un cursore cancella il
# lavoro fatto a click.
R2 = segment_regions(IMG, white_clip=200)
check("i semi sopravvivono a un cambio di White Clip",
      R2.shape == R.shape and R2.region_at(*P_VUOTO) > 0
      and not mask_from_regions(R2, resolve_cut_flags(R2, cut_seeds=[P_VUOTO]))
      [P_VUOTO[1], P_VUOTO[0]])


# ---------------------------------------------------------------------------
print("\n=== MOTORE: dalla maschera alla mesh ===")

from engine import GenerationMode, GenerationParams, generate


def genera(nome, **kw):
    p = GenerationParams(
        mode=kw.pop("mode", GenerationMode.KEYCHAIN),
        max_dim=60.0, base_h=1.0, max_h=2.4, layer_height=0.2,
        max_res_cap=800, smart_decimate=kw.pop("smart_decimate", False),
        spot_accents=[], output_path=os.path.join(TMP, nome + ".stl"), **kw)
    res = generate(IMG, p)
    return trimesh.load(res.stl_path), res


def genere(mesh):
    """Il numero di manici della superficie: 0 lastra piena, 1 un foro, 2 due."""
    return (2 - mesh.euler_number) // 2


m0, _ = genera("piena", mode=GenerationMode.SPOT_COLOR)
check("Spot Color (senza ritaglio): lastra chiusa senza fori",
      m0.is_watertight and genere(m0) == 0, f"chi={m0.euler_number}")

m1, r1 = genera("auto")
check("ritaglio auto: chiusa, un pezzo, ancora senza fori",
      m1.is_watertight and m1.body_count == 1 and genere(m1) == 0,
      f"chi={m1.euler_number} corpi={m1.body_count}")

m2, r2 = genera("click", cutout_cut_seeds=[P_VUOTO])
check("un click -> un foro passante VERO nella mesh",
      m2.is_watertight and m2.body_count == 1 and genere(m2) == 1,
      f"chi={m2.euler_number}")

m3, r3 = genera("occhiello", cutout_cut_seeds=[P_VUOTO],
                cutout_ring=True, cutout_ring_xy=(450, 45),
                cutout_ring_d_mm=4.0, cutout_ring_rim_mm=2.5)
check("occhiello -> secondo foro, e il motore lo dichiara attaccato",
      m3.is_watertight and genere(m3) == 2 and r3.cutout_ring_attached,
      f"chi={m3.euler_number} attaccato={r3.cutout_ring_attached}")

c_stacc = compute_cutout(IMG, max_dim=60.0, ring_xy=(30, 30),
                         ring_d_mm=4.0, ring_rim_mm=2.0)
check("un occhiello fuori dal materiale viene dichiarato staccato",
      not c_stacc.ring_attached)
check("...e della sua corona non resta niente da dipingere",
      c_stacc.ring_mask is not None and int(c_stacc.ring_mask.sum()) == 0,
      int(c_stacc.ring_mask.sum()) if c_stacc.ring_mask is not None else None)

# --- l'occhiello deve essere alto quanto il pezzo, ma solo dove sporge ----
#
# La sagoma dice solo DOVE c'e' materiale; quanto e' alto lo decide la
# classificazione del disegno sotto. Se l'anello sporge dal soggetto, sotto
# c'e' carta bianca — il livello piu' basso — e ne esce un anello alto un
# layer: la parte piu' fragile del pezzo messa dove lo si tira. Se invece cade
# dentro il disegno, sotto c'e' gia' l'arte, e dipingerla cancella i dettagli
# attorno al foro.
#
# Non serve una spunta "interno/esterno": si dipinge la parte che l'anello ha
# AGGIUNTO al vuoto. Fuori e' quasi tutta la corona, dentro e' niente, a
# cavallo del bordo e' la meta' che sporge — che una spunta non saprebbe dire.
#
# Serve una tavola apposta. Sul cappello il punto "fuori" cadeva nella regione
# bianca racchiusa dalla corona, che e' materiale: la prova passava senza
# provare il caso esterno.

def tavola_occhiello(n=800):
    """Un disco pieno di dettagli, con un gambo che esce in alto."""
    img = np.full((n, n, 3), 255, np.uint8)
    cv2.circle(img, (400, 480), 230, (0, 0, 0), -1)
    for k in range(-150, 151, 26):                       # dettagli dentro
        cv2.line(img, (400 + k, 360), (400 + k, 600), (255, 255, 255), 7)
    cv2.line(img, (400, 250), (400, 180), (0, 0, 0), 20)  # gambo
    return img


OCC = tavola_occhiello()
OCC_FUORI, OCC_DENTRO, OCC_CAVALLO = (400, 150), (400, 480), (400, 265)
RING_D, RING_RIM = 4.0, 2.5


def _cut_occhiello(centro):
    return compute_cutout(OCC, max_dim=60.0, ring_xy=centro,
                          ring_d_mm=RING_D, ring_rim_mm=RING_RIM)


def quota_dipinta(centro):
    """Che frazione della corona viene dipinta come inchiostro."""
    from engine.cutout_utils import _disc, seg_shape_for
    c = _cut_occhiello(centro)
    hh, ww = seg_shape_for(OCC.shape[:2])
    corona = (_disc((hh, ww), centro[0], centro[1],
                    (RING_D / 2 + RING_RIM) / c.pitch_mm)
              & ~_disc((hh, ww), centro[0], centro[1], (RING_D / 2) / c.pitch_mm))
    return int(c.ring_mask.sum()) / max(1, int(corona.sum()))


check("fuori dal disegno la corona e' quasi tutta da dipingere",
      quota_dipinta(OCC_FUORI) > 0.6, f"{100*quota_dipinta(OCC_FUORI):.0f}%")
check("dentro il disegno non c'e' niente da dipingere",
      quota_dipinta(OCC_DENTRO) < 0.05, f"{100*quota_dipinta(OCC_DENTRO):.0f}%")
check("a cavallo del bordo se ne dipinge una parte, non tutta e non niente",
      0.03 < quota_dipinta(OCC_CAVALLO) < 0.6,
      f"{100*quota_dipinta(OCC_CAVALLO):.0f}%")


def mesh_occhiello(nome, centro, **kw):
    p = GenerationParams(
        mode=GenerationMode.KEYCHAIN, max_dim=60.0, base_h=1.6, max_h=2.8,
        layer_height=0.2, max_res_cap=800, smart_decimate=False, spot_accents=[],
        cutout_ring=True, cutout_ring_xy=centro, cutout_ring_d_mm=RING_D,
        cutout_ring_rim_mm=RING_RIM,
        output_path=os.path.join(TMP, nome + ".stl"), **kw)
    src = OCC if kw.get("keychain_finish_spot", True) else cv2.cvtColor(
        OCC, cv2.COLOR_RGB2GRAY)
    return trimesh.load(generate(src, p).stl_path), _cut_occhiello(centro)


def _mm(cut, centro):
    y0, y1, x0, x1 = cut.bbox
    return ((centro[0] - x0) * cut.pitch_mm,
            ((y1 - y0) - (centro[1] - y0)) * cut.pitch_mm)


# Fuori: la corona deve arrivare alla cima, in entrambe le finiture. La prova
# guarda la MESH e non la maschera — fra le due ci sono la classificazione, il
# ricampionamento e le terrazze, ed e' la classificazione che sbagliava.
for spot in (True, False):
    et = "Spot" if spot else "B/N"
    # Il nome del file non puo' portarsi dentro lo slash di "B/N".
    m_r, c_r = mesh_occhiello("occhiello_fuori_" + ("spot" if spot else "bn"),
                              OCC_FUORI,
                              keychain_finish_spot=spot, color_mode=2,
                              color_changes_z=[1.4, 2.2, 2.8])
    cx, cy = _mm(c_r, OCC_FUORI)
    v = m_r.vertices
    r = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
    sulla = (r > RING_D / 2 + 0.2) & (r < RING_D / 2 + RING_RIM - 0.2)
    check(f"finitura {et}: sporgendo, la corona arriva a tutta altezza",
          sulla.any() and abs(float(v[sulla, 2].max()) - 2.8) < 1e-5,
          f"{int(sulla.sum())} vertici, cima {float(v[sulla, 2].max()):.2f} mm")
    check(f"finitura {et}: e resta chiusa e in un pezzo",
          m_r.is_watertight and m_r.body_count == 1)

# Dentro: l'arte attorno al foro conserva le sue quote. Si misura l'AREA a
# quota base, non le quote presenti: una singola faccia superstite basterebbe
# a far passare un conteggio di quote, e non e' quello che si sta chiedendo.
m_d, c_d = mesh_occhiello("occhiello_dentro", OCC_DENTRO)
cx, cy = _mm(c_d, OCC_DENTRO)
tri = m_d.vertices[m_d.faces]
vicino = np.hypot(tri.mean(axis=1)[:, 0] - cx, tri.mean(axis=1)[:, 1] - cy) < 8.0
area_base = float(m_d.area_faces[(np.abs(tri[:, :, 2] - 1.6) < 1e-6).all(axis=1)
                                 & vicino].sum())
area_cima = float(m_d.area_faces[(np.abs(tri[:, :, 2] - 2.8) < 1e-6).all(axis=1)
                                 & vicino].sum())
check("con l'anello dentro, i dettagli attorno al foro sopravvivono",
      area_base > 40.0 and m_d.is_watertight,
      f"{area_base:.0f} mm2 a quota base contro {area_cima:.0f} mm2 di inchiostro")

# La finitura B/N passa da create_solid_mesh invece che da process_mesh_topo:
# e' un secondo percorso dentro la stessa modalita', e va provato come tale.
m4, _ = genera("bn", keychain_finish_spot=False,
               cutout_cut_seeds=[P_VUOTO],
               color_mode=2, color_changes_z=[1.4, 2.0, 2.4])
check("finitura B/N: stessa sagoma, stesso foro",
      m4.is_watertight and genere(m4) == 1, f"chi={m4.euler_number}")


# ---------------------------------------------------------------------------
print("\n=== MOTORE: maschera dipinta, e Max Dim ===")

# Opzione B: il vuoto racchiuso e' bianco quanto quello esterno, e va via
# senza nessun seme. E' la proprieta' che la segmentazione non ha.
dipinta = np.full(IMG.shape, 255, np.uint8)
cv2.circle(dipinta, (450, 400), 370, (255, 216, 0), -1)
cv2.circle(dipinta, (450, 560), 200, (255, 216, 0), -1)
cv2.circle(dipinta, (450, 150), 90, (255, 255, 255), -1)   # un vuoto dipinto
m5, _ = genera("dipinta", cutout_paint_mask=dipinta)
check("maschera dipinta: il vuoto bianco diventa foro senza click",
      m5.is_watertight and genere(m5) == 1, f"chi={m5.euler_number}")


def impronta_mm2(mesh):
    """L'area davvero appoggiata al piatto.

    Non il bounding box e non il conteggio dei vertici: la sagoma e' proprio
    cio' che rende il pezzo piu' piccolo del suo riquadro, quindi misurarla
    col riquadro vorrebbe dire non misurarla.
    """
    z = mesh.vertices[mesh.faces][:, :, 2]
    return float(mesh.area_faces[(np.abs(z) < 1e-6).all(axis=1)].sum())


# Il disegno e' un disco che riempie quasi tutto il foglio: con il ritaglio
# l'impronta e' quella del disco (pi*r^2 con 2r = Max Dim = 60), senza e'
# quella del foglio intero (60x60). Sono due numeri lontani il 27%: nessuna
# tolleranza li confonde.
a_piena, a_cut = impronta_mm2(m0), impronta_mm2(m1)
check("in Spot Color l'impronta e' il foglio (60x60)",
      abs(a_piena - 3600.0) / 3600.0 < 0.02, f"{a_piena:.0f} mm2")
check("in Keychain l'impronta e' il disco, e Max Dim misura il pezzo",
      abs(a_cut - np.pi * 30.0 ** 2) / (np.pi * 30.0 ** 2) < 0.05,
      f"{a_cut:.0f} mm2 (atteso {np.pi*900:.0f})")

# Un disegno piccolo su un foglio grande: e' il caso in cui il ritaglio del
# riquadro conta davvero, e senza di quello il pezzo uscirebbe in scala 1:4.
grande = np.full((1200, 1200, 3), 255, np.uint8)
cv2.circle(grande, (600, 600), 150, (0, 0, 0), 10)
p = GenerationParams(mode=GenerationMode.KEYCHAIN, max_dim=50.0, base_h=1.0,
                     max_h=2.4, layer_height=0.2, max_res_cap=800,
                     smart_decimate=False, spot_accents=[],
                     output_path=os.path.join(TMP, "piccolo.stl"))
m6 = trimesh.load(generate(grande, p).stl_path)
lato = float(max(m6.bounds[1][:2] - m6.bounds[0][:2]))
check("disegno piccolo su foglio grande: il PEZZO misura Max Dim",
      abs(lato - 50.0) < 1.0, f"lato {lato:.1f} mm")

# Il bordino allarga verso l'esterno di quanto dichiara (mezzo per lato).
b0 = compute_cutout(grande, max_dim=50.0, border_mm=0.0)
b2 = compute_cutout(grande, max_dim=50.0, border_mm=2.0)
larg0 = (b0.bbox[3] - b0.bbox[2]) * b0.pitch_mm
larg2 = (b2.bbox[3] - b2.bbox[2]) * b2.pitch_mm
check("lo Sticker border allarga la sagoma (in px, a parita' di scala)",
      (b2.bbox[3] - b2.bbox[2]) > (b0.bbox[3] - b0.bbox[2]),
      f"{b0.bbox[3]-b0.bbox[2]}px -> {b2.bbox[3]-b2.bbox[2]}px")

# Un ritaglio che non lascia niente deve fermarsi con un messaggio, non
# produrre una mesh vuota che si scopre nello slicer.
try:
    genera("vuoto",
           cutout_paint_mask=np.full(IMG.shape, 255, np.uint8))
    check("ritaglio vuoto: errore esplicito", False, "nessuna eccezione")
except ValueError as e:
    check("ritaglio vuoto: errore esplicito", "material" in str(e).lower()
          or "materiale" in str(e).lower(), str(e)[:60])


# ---------------------------------------------------------------------------
print("\n=== INTERFACCIA ===")

from PyQt6.QtWidgets import QApplication, QFileDialog

from manga_to_3d import Manga3DAppController

app = QApplication.instance() or QApplication(sys.argv)
win = Manga3DAppController()
win.show()          # senza questo Qt dice invisibile a tutto

QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (IMG_PATH, ""))
win.load_image()
check("immagine caricata", win.img_rgb_original is not None)

KEYCHAIN = 5

win.mode_selector.setCurrentIndex(3)          # Spot
check("in Spot il pannello portachiavi non c'e'", not win.group_keychain.isVisible())
win.mode_selector.setCurrentIndex(KEYCHAIN)
check("scegliendo Keychain / Cutout compare il suo pannello",
      win.group_keychain.isVisible())
check("il selettore ha la voce in fondo, dopo Phone Cover",
      win.mode_selector.itemText(KEYCHAIN).startswith("Keychain"),
      win.mode_selector.itemText(KEYCHAIN))
check("e la modalita' che arriva al motore e' quella giusta",
      win._current_generation_mode() == GenerationMode.KEYCHAIN)

# La finitura decide quali altri pannelli servono: in Spot gli accenti, in
# B/N gli swatch e le quote Standard. Sono gli stessi pannelli della cover,
# e la regola e' la stessa.
check("finitura Spot: compare il pannello accenti", win.group_spot.isVisible())
check("finitura Spot: swatch e quote Standard spariscono",
      not win.group_swatch.isVisible() and not win.group_z.isVisible())
win.combo_keychain_finish.setCurrentIndex(1)   # B/N
check("finitura B/N: tornano swatch e quote Standard",
      win.group_swatch.isVisible() and win.group_z.isVisible())
check("finitura B/N: il pannello accenti sparisce", not win.group_spot.isVisible())
check("...e il motore riceve la finitura scelta", not win._keychain_spot())
win.combo_keychain_finish.setCurrentIndex(0)

# In una modalita' dedicata i controlli sono vivi da subito: non c'e' una
# spunta da accendere prima, perche' la voce del menu e' gia' quella spunta.
check("i controlli del ritaglio sono attivi senza altre spunte",
      win.btn_cutout_edit.isEnabled() and win.slider_cutout_border.isEnabled()
      and win.btn_cutout_preview.isEnabled())

# Un portachiavi da 200 mm sarebbe un sottopentola.
check("entrando in modalita' Max Dim scende a taglia portachiavi",
      win.spin_dim.value() <= 80.0, f"{win.spin_dim.value()} mm")

# L'altezza di default non e' un numero tondo scelto a occhio: il rilievo deve
# ospitare colori-1 bande di colore, e una banda di un layer solo lascia
# trasparire quello sotto. Il caso peggiore e' due accenti (4 colori), ed e'
# quello che fissa il minimo — abbassare ancora il default lo romperebbe in
# silenzio, perche' il file si genera comunque.
from engine.mesh_utils import compute_topo_z_heights

base, tot, lh = win.spin_base.value(), win.spin_maxh.value(), win.spin_layer_height.value()
for n_colori in (2, 3, 4):
    z = compute_topo_z_heights(base, tot, lh, n_colori)
    layer = [round((z[i] - z[i - 1]) / lh) for i in range(1, n_colori)]
    check(f"default portachiavi: a {n_colori} colori ogni banda ha almeno 2 layer",
          layer and min(layer) >= 2,
          f"base {base} + rilievo {tot-base:.1f} -> bande {layer}")

win.btn_cutout_edit.setChecked(True)
check("entrando in modifica l'anteprima si accende (i click si leggono li')",
      win.btn_cutout_preview.isChecked())

# Le quattro anteprime mostrano ognuna un'immagine sua, a una risoluzione sua.
# Due accese insieme non sono uno stato ambiguo, sono uno stato SBAGLIATO: i
# click del ritaglio si leggono sul raster di segmentazione, e sull'immagine
# di un'altra anteprima indicano un'altra regione.
#
# Il caso da cui viene questa prova: con un mockup acceso, premere Edit
# regions accendeva l'anteprima del ritaglio, che spegneva il mockup, il cui
# handler passava da _update_viewport_mode, che rispegneva l'anteprima del
# ritaglio e con essa la modifica. Il pulsante tornava su da solo e da li' in
# poi i click non tagliavano piu' niente — senza un messaggio d'errore, perche'
# dal punto di vista del codice non era successo nulla di illecito.


def accese():
    return [n for n in win._PREVIEW_BUTTONS if getattr(win, n).isChecked()]


for partenza in ('btn_std_mockup', 'btn_spot_mockup'):
    win.btn_cutout_edit.setChecked(False)
    getattr(win, partenza).setChecked(True)
    check(f"({partenza} acceso da solo)", accese() == [partenza], accese())
    win.btn_cutout_edit.setChecked(True)
    check(f"con {partenza} acceso, Edit regions resta armato",
          win.btn_cutout_edit.isChecked() and accese() == ['btn_cutout_preview'],
          f"edit={win.btn_cutout_edit.isChecked()} accese={accese()}")
    prima = win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]]
    win.on_pixel_clicked(*P_VUOTO)
    check(f"...e il click taglia davvero (partendo da {partenza})",
          prima and not win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]])
    win._reset_cutout()

# E la direzione opposta: accendere un mockup mentre si modificano le regioni
# deve disarmare la modifica, non lasciarla armata su un'immagine che non e'
# quella su cui i click sono definiti.
win.btn_cutout_edit.setChecked(True)
win.btn_std_mockup.setChecked(True)
check("accendendo un mockup, la modifica delle regioni si disarma",
      not win.btn_cutout_edit.isChecked() and accese() == ['btn_std_mockup'],
      f"edit={win.btn_cutout_edit.isChecked()} accese={accese()}")

for n in ('btn_spot_mockup', 'btn_cutout_preview', 'btn_std_mockup',
          'btn_cutout_preview'):
    getattr(win, n).setChecked(True)
    check(f"mai due anteprime insieme (acceso {n})", accese() == [n], accese())

win.btn_cutout_edit.setChecked(True)

# Il click vero, sulle coordinate del raster di segmentazione — che per una
# sorgente 900x900 e cap 1600 coincide con la sorgente.
prima = win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]]
win.on_pixel_clicked(*P_VUOTO)
dopo = win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]]
check("un click sul vuoto lo buca", prima and not dopo)

win.on_pixel_clicked(*P_VUOTO)
check("un secondo click lo richiude",
      win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]])

win.on_pixel_clicked(*P_VUOTO)
win.on_pixel_clicked(*P_CUPOLA)
mm = win._compute_cutout_now().mask
check("due regioni si commutano indipendentemente",
      (not mm[P_VUOTO[1], P_VUOTO[0]]) and (not mm[P_CUPOLA[1], P_CUPOLA[0]]))

win._reset_cutout()
check("Reset riporta all'automatismo",
      win._compute_cutout_now().mask[P_VUOTO[1], P_VUOTO[0]]
      and not win.cutout_cut_seeds)

# Un click sul tratto non deve inventarsi una regione.
win.on_pixel_clicked(450, 400 - 370)     # sul contorno della corona
check("il click sul tratto non aggiunge semi fantasma",
      all(win._cutout_regions_now().region_at(*sd) > 0 for sd in win.cutout_cut_seeds))

# Fuori dalla modalita' il click torna a chi lo aspettava: in Spot serve a
# campionare un accento, e il ritaglio non deve rubarglielo.
win.mode_selector.setCurrentIndex(3)
win.set_active_spot_swatch(0)
win.on_pixel_clicked(450, 590)
check("fuori dalla modalita' il click torna agli accenti Spot",
      win.spot_accents[0] is not None)
win.mode_selector.setCurrentIndex(KEYCHAIN)

# La maschera dipinta non ha regioni: i comandi a click devono spegnersi,
# altrimenti promettono un controllo che li' non esiste.
win.combo_cutout_src.setCurrentIndex(1)
check("con la maschera dipinta i click si spengono",
      not win.btn_cutout_edit.isEnabled() and not win.btn_cutout_reset.isEnabled())
win.combo_cutout_src.setCurrentIndex(0)
check("tornando in auto si riaccendono", win.btn_cutout_edit.isEnabled())

# Il lock di generazione riaccende tutto all'unlock: gli stati condizionali
# vanno rimessi, o dopo una generazione i controlli mentono.
win.toggle_ui_state(disabled=True)
check("durante la generazione i controlli si bloccano",
      not win.btn_cutout_edit.isEnabled())
win.toggle_ui_state(disabled=False)
check("dopo lo sblocco i controlli tornano vivi", win.btn_cutout_edit.isEnabled())
win.combo_cutout_src.setCurrentIndex(1)
win.toggle_ui_state(disabled=True)
win.toggle_ui_state(disabled=False)
check("...ma la maschera dipinta resta senza click",
      not win.btn_cutout_edit.isEnabled())
win.combo_cutout_src.setCurrentIndex(0)

# Gli accenti Spot servono a TRE modalita' (Spot, cover con finitura Spot,
# portachiavi con finitura Spot), ma la condizione del click chiedeva
# "modalita' == Spot". Nelle altre due il pannello si vedeva, il pulsante si
# armava, e il click sull'immagine non faceva niente — senza un messaggio,
# perche' il ramo non partiva proprio.
win.mode_selector.setCurrentIndex(KEYCHAIN)
win.combo_keychain_finish.setCurrentIndex(0)          # Spot
check("in Keychain+Spot il pannello accenti e' attivo", win._spot_panel_active())
win.combo_keychain_finish.setCurrentIndex(1)          # B/N
check("...e in Keychain+B/N no", not win._spot_panel_active())
win.combo_keychain_finish.setCurrentIndex(0)

win.spot_accents = [None, None]
win.btn_cutout_edit.setChecked(False)
win.btn_cutout_preview.setChecked(False)
win.set_active_spot_swatch(0)
win.on_pixel_clicked(*P_CUPOLA)
check("in Keychain il click campiona davvero l'accento",
      win.spot_accents[0] is not None, win.spot_accents[0])

# Con un'anteprima accesa il campione verrebbe da un'altra immagine, a
# un'altra risoluzione: si spegne e si chiede di ripetere invece di
# raccogliere un colore inventato.
win.spot_accents = [None, None]
win.btn_cutout_preview.setChecked(True)
win.set_active_spot_swatch(0)
win.on_pixel_clicked(*P_CUPOLA)
check("con un'anteprima accesa non campiona, la spegne e lo dice",
      win.spot_accents[0] is None
      and not any(getattr(win, n).isChecked() for n in win._PREVIEW_BUTTONS)
      and "ORIGINAL" in win.lbl_status.text(), win.lbl_status.text()[:50])
win.on_pixel_clicked(*P_CUPOLA)
check("...e il click successivo campiona", win.spot_accents[0] is not None)
win.spot_accents = [None, None]


# I parametri fisici tornano indietro cambiando modalita'.
#
# Non e' pulizia: la copertura a 2 colori misura una finestra di 0,7 mm REALI,
# quindi Max Dim decide quanti pixel della sorgente ci stanno dentro. Con i
# default che si applicavano solo entrando (e mai uscendo), dopo un giro nel
# portachiavi la modalita' Standard restava a 60 mm e binarizzava lo stesso
# pannello con una finestra tre volte piu' grossa — cursore fermo, immagine
# uguale, risultato diverso e nessuno a dirlo.
win.mode_selector.setCurrentIndex(0)
dim_std, base_std = win.spin_dim.value(), win.spin_base.value()
win.mode_selector.setCurrentIndex(KEYCHAIN)
dim_key = win.spin_dim.value()
check("la modalita' portachiavi ha una sua taglia", dim_key < dim_std,
      f"{dim_key} vs {dim_std}")
win.mode_selector.setCurrentIndex(0)
check("tornando in Standard i parametri tornano quelli di prima",
      win.spin_dim.value() == dim_std and win.spin_base.value() == base_std,
      f"MaxDim={win.spin_dim.value()} Base={win.spin_base.value()}")
win.mode_selector.setCurrentIndex(4)
win.mode_selector.setCurrentIndex(3)
check("...e vale per ogni modalita', non solo per il portachiavi",
      win.spin_dim.value() == dim_std and win.spin_layer_height.value() == 0.20,
      f"MaxDim={win.spin_dim.value()} Layer={win.spin_layer_height.value()}")

# Il mockup Standard restava acceso attraverso il cambio di modalita',
# mostrando una classificazione calcolata con parametri che non esistono piu'.
win.mode_selector.setCurrentIndex(0)
win.btn_std_mockup.setChecked(True)
win.mode_selector.setCurrentIndex(KEYCHAIN)
check("cambiando modalita' il mockup Standard si spegne, come gli altri",
      not win.btn_std_mockup.isChecked())
check("e il pulsante torna a proporsi", win.btn_std_mockup.text() == "👁 Mockup Preview",
      win.btn_std_mockup.text())
win.mode_selector.setCurrentIndex(KEYCHAIN)


# L'ordine dei riquadri, che e' una proprieta' del pannello e non una
# questione di gusto: il selettore della finitura decide se compaiono gli
# accenti Spot o gli swatch Standard, quindi deve stare SOPRA di loro. Con il
# selettore sotto, si sceglie una cosa guardando un riquadro gia' passato.
# E' l'invariante che si rompe da sola la prossima volta che qualcuno aggiunge
# un pannello in fondo a initUI — come e' successo a questo.
colonna = win.group_spot.parentWidget().layout()


def posizione(g):
    return colonna.indexOf(g)


check("il pannello portachiavi sta sopra gli accenti Spot",
      posizione(win.group_keychain) < posizione(win.group_spot),
      f"keychain={posizione(win.group_keychain)} spot={posizione(win.group_spot)}")
check("...e sopra gli swatch e le quote Standard",
      posizione(win.group_keychain) < posizione(win.group_swatch)
      and posizione(win.group_keychain) < posizione(win.group_z))
check("la stessa regola vale per la cover, che ha lo stesso selettore",
      posizione(win.group_cover) < posizione(win.group_spot)
      and posizione(win.group_cover) < posizione(win.group_swatch))
check("e il selettore di modalita' sta in cima a tutti",
      posizione(win.mode_selector) < min(posizione(g) for g in (
          win.group_topo, win.group_deckbox, win.group_cover,
          win.group_keychain, win.group_spot, win.group_swatch)))


# Il nome del file dice da quale modalita' viene: nella cartella output/ i
# file di modalita' diverse finiscono fianco a fianco, e "<nome>_3D" da solo
# non distingue un pannello da un portachiavi dello stesso disegno.
from PyQt6.QtCore import QObject, pyqtSignal
import manga_to_3d as m3d


class WorkerFinto(QObject):
    """Intercetta i parametri senza far partire nulla: qui interessa come si
    chiamano i file, non cosa ci finisce dentro."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)
    finished_err = pyqtSignal(str)
    ultimo = None

    def __init__(self, params, image):
        super().__init__()
        WorkerFinto.ultimo = params

    def start(self):
        pass


def stem_generato():
    vero, m3d.MeshWorker = m3d.MeshWorker, WorkerFinto
    try:
        win.generate_stl()
    finally:
        m3d.MeshWorker = vero
    return os.path.splitext(os.path.basename(WorkerFinto.ultimo.output_path))[0]


win.chk_export_stl.setChecked(True)
win.chk_export_3mf.setChecked(False)

win.mode_selector.setCurrentIndex(KEYCHAIN)
st_key = stem_generato()
check("il file del portachiavi ha 'keychain' nel nome",
      "keychain" in st_key and st_key.startswith("cappello"), st_key)

win.mode_selector.setCurrentIndex(0)          # Standard
st_std = stem_generato()
check("le altre modalita' non sono state toccate", st_std.endswith("_3D"), st_std)
check("e i due nomi sono diversi", st_key != st_std, f"{st_key} / {st_std}")
win.mode_selector.setCurrentIndex(KEYCHAIN)


# Caricare un'altra immagine deve azzerare i semi: nominano regioni di prima.
win.btn_cutout_edit.setChecked(True)
win.on_pixel_clicked(*P_VUOTO)
check("(semi presenti prima del ricarico)", len(win.cutout_cut_seeds) == 1)
win.load_image()
check("caricando un'altra immagine i semi si azzerano",
      not win.cutout_cut_seeds and not win.cutout_keep_seeds
      and win.cutout_ring_xy is None)


# ---------------------------------------------------------------------------
print()
if fails:
    print(f"❌ {len(fails)} FALLITE: " + ", ".join(fails))
    sys.exit(1)
print("✅ tutte passate")

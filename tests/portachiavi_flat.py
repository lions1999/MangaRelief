"""Portachiavi a stampa piatta (AMS): i colori nel piano, non nella quota.

Le domande a cui risponde, e perche' ognuna e' qui:

- **le parti tassellano la sagoma**: le facce colorate, messe insieme, coprono
  esattamente la sagoma del corpo — ne' fessure ne' sovrapposizioni. Si misura
  sui VOLUMI delle mesh nel 3MF, non sulle maschere: che le maschere siano
  complementari e' l'ipotesi, che lo siano ancora dopo poligoni, specchio ed
  estrusione e' la tesi. Una fessura di un pixel sarebbe una linea di corpo
  che traspare nella faccia; una sovrapposizione, due parti che lo slicer
  deve litigarsi.
- **le quote**: la faccia sta nei primi N layer e il corpo sopra (faccia in
  giu'), o il contrario (faccia in su). Una faccia che sconfina nel corpo e'
  un colore che si stampa per tutto lo spessore — lo spreco che la modalita'
  esiste per evitare.
- **lo specchio**: a faccia in giu' l'immagine si guarda dal piatto, quindi il
  rosso che nel disegno sta a sinistra nel modello deve stare a destra.
  Sbagliare verso non si vede nello slicer, si vede sul portachiavi stampato.
- **il 3MF**: ogni <part> si aggancia al <component> con lo stesso id e porta
  il suo `extruder`; la palette dichiarata arriva esattamente all'extruder
  piu' alto (oltre, Bambu riporta la parte al filamento 1 in silenzio); nessun
  cambio a quota.
- **i rifiuti**: B/N + piatto, e piu' layer colorati di quanti ne stanno.
- **la UI**: il selettore c'e' solo nel portachiavi, e' spento in B/N, e la
  scelta arriva nei parametri.

    python tests/portachiavi_flat.py

Una decina di secondi.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import trimesh

from engine import GenerationMode, GenerationParams, generate

fails = []


def check(n, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + n + ((" | " + str(extra)) if extra else ""))
    if not ok:
        fails.append(n)


TMP = tempfile.mkdtemp(prefix="flat_")
ROSSO, BLU = (220, 30, 30), (30, 60, 220)


def fixture():
    """Un medaglione: contorno nero, tratteggio, e due accenti ai due lati.
    Il rosso sta a SINISTRA e il blu a DESTRA: e' quello che rende lo
    specchio verificabile."""
    img = np.full((900, 900, 3), 255, np.uint8)
    cv2.circle(img, (450, 450), 380, (0, 0, 0), 16)
    cv2.circle(img, (270, 420), 90, ROSSO, -1)
    cv2.circle(img, (640, 480), 70, BLU, -1)
    for y in range(250, 700, 30):
        cv2.line(img, (390, y), (520, y), (0, 0, 0), 5)
    return img


IMG = fixture()
LH, BASE, LAYERS = 0.2, 1.6, 3
FACE = LAYERS * LH
# In piatto la Base e' il corpo e lo spessore ne discende. max_h si passa
# volutamente sbagliato: se il motore lo leggesse, le quote lo direbbero.
MAXH = round(BASE + FACE, 3)


def params(**kw):
    base = dict(mode=GenerationMode.KEYCHAIN, keychain_finish_spot=True,
                keychain_flat=True, spot_accents=[ROSSO, BLU],
                max_dim=60, base_h=BASE, max_h=5.0, layer_height=LH,
                flat_face_layers=LAYERS, max_res_cap=900)
    base.update(kw)
    return GenerationParams(**base)


def leggi_3mf(path):
    """Le parti del 3MF come le vede l'importatore di Bambu: mesh per id,
    component per id, e la configurazione di ogni <part>."""
    z = zipfile.ZipFile(path)
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    obj = ET.fromstring(z.read("3D/Objects/object_1.model"))
    meshes = {}
    for o in obj.iter("{%s}object" % ns["m"]):
        v = np.array([[float(e.get(a)) for a in "xyz"]
                      for e in o.iter("{%s}vertex" % ns["m"])])
        f = np.array([[int(e.get(a)) for a in ("v1", "v2", "v3")]
                      for e in o.iter("{%s}triangle" % ns["m"])])
        meshes[int(o.get("id"))] = trimesh.Trimesh(v, f, process=False)
    root = ET.fromstring(z.read("3D/3dmodel.model"))
    comps = [int(c.get("objectid")) for c in root.iter("{%s}component" % ns["m"])]
    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    parti = {}
    for part in cfg.iter("part"):
        md = {m.get("key"): m.get("value") for m in part.iter("metadata") if m.get("key")}
        parti[int(part.get("id"))] = md
    ps = json.loads(z.read("Metadata/project_settings.config"))
    return z, meshes, comps, parti, ps, root


# ---------------------------------------------------------------------------
print("\n=== FACCIA IN GIU' ===")
p = params(output_path=os.path.join(TMP, "k.stl"),
           output_path_3mf=os.path.join(TMP, "k.3mf"))
r = generate(IMG, p)
z, meshes, comps, parti, ps, root = leggi_3mf(r.mf3_path)

check("una parte per colore piu' il corpo (bianco, rosso, blu, nero + corpo)",
      len(meshes) == 5 and len(r.flat_parts) == 5, [n for n, _, _ in r.flat_parts])
check("ogni <part> ha il suo <component> con lo stesso id",
      sorted(comps) == sorted(parti) == sorted(meshes), (comps, list(parti)))
check("ogni parte e' un solido chiuso con volume positivo",
      all(m.is_watertight and m.is_winding_consistent and m.volume > 0
          for m in meshes.values()))

ext = {k: int(md["extruder"]) for k, md in parti.items()}
nomi = {k: md["name"] for k, md in parti.items()}
corpo = [k for k in nomi if nomi[k].startswith("Body")]
facce = [k for k in nomi if nomi[k].startswith("Face")]
check("il corpo e' uno e sta sul filamento 1",
      len(corpo) == 1 and ext[corpo[0]] == 1, nomi)
check("la palette dichiarata arriva esattamente all'extruder piu' alto",
      len(ps["filament_colour"]) == max(ext.values()) == 4, ps["filament_colour"])
check("filamenti distinti per colori distinti",
      len({ext[k] for k in facce}) == len(facce))

zmin = {k: m.bounds[0][2] for k, m in meshes.items()}
zmax = {k: m.bounds[1][2] for k, m in meshes.items()}
# Il 3MF centra tutto sull'origine comune: le quote si leggono rispetto al
# fondo dell'insieme, che e' la base appoggiata al piatto.
fondo = min(zmin.values())
check("le facce stanno nei primi layer",
      all(abs(zmin[k] - fondo) < 1e-6 and abs(zmax[k] - fondo - FACE) < 1e-6 for k in facce),
      {nomi[k]: (round(zmin[k] - fondo, 3), round(zmax[k] - fondo, 3)) for k in facce})
k = corpo[0]
check("il corpo va dalla faccia a Max Z",
      abs(zmin[k] - fondo - FACE) < 1e-6 and abs(zmax[k] - fondo - MAXH) < 1e-6,
      (zmin[k] - fondo, zmax[k] - fondo))

v_facce = sum(meshes[k].volume for k in facce)
v_corpo = meshes[corpo[0]].volume
area_facce, area_corpo = v_facce / FACE, v_corpo / (MAXH - FACE)
check("le facce tassellano la sagoma (area facce = area corpo)",
      abs(area_facce - area_corpo) / area_corpo < 1e-9,
      f"{area_facce:.4f} vs {area_corpo:.4f} mm²")

# Lo specchio: il baricentro della faccia rossa, nel modello, sta a destra.
rossa = [k for k in facce if nomi[k].endswith("#dc1e1e")]
blu = [k for k in facce if nomi[k].endswith("#1e3cdc")]
cx = (meshes[corpo[0]].bounds[0][0] + meshes[corpo[0]].bounds[1][0]) / 2
check("accenti riconosciuti (rosso e blu hanno la loro parte)",
      len(rossa) == 1 and len(blu) == 1, nomi)
if rossa and blu:
    check("faccia in giu': l'immagine e' specchiata (rosso a destra, blu a sinistra)",
          meshes[rossa[0]].centroid[0] > cx > meshes[blu[0]].centroid[0])

check("nessun cambio colore a quota nel 3MF",
      "Metadata/custom_gcode_per_layer.xml" not in z.namelist())
app = [m.text for m in root.iter("{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}metadata")
       if m.get("name") == "Application"]
check("resta un progetto Bambu Studio (Application + 3mfVersion)",
      app and app[0].startswith("BambuStudio-")
      and "BambuStudio:3mfVersion" in z.read("3D/Objects/object_1.model").decode())
check("stima dei cambi: 3 layer x (4 colori - 1) + 1",
      r.flat_filament_changes == 10, r.flat_filament_changes)

check("uno STL per filamento, e si caricano",
      len(r.stl_paths) == 4 and all(os.path.exists(s) for s in r.stl_paths)
      and all(len(trimesh.load(s).faces) > 0 for s in r.stl_paths),
      [os.path.basename(s) for s in r.stl_paths])
check("gli STL portano filamento e colore nel nome",
      all(re.search(r"_f\d_[0-9a-f]{6}\.stl$", s) for s in r.stl_paths))


# ---------------------------------------------------------------------------
print("\n=== FACCIA IN SU ===")
r2 = generate(IMG, params(flat_face_down=False, output_path_3mf=os.path.join(TMP, "up.3mf")))
_, m2, _, parti2, _, _ = leggi_3mf(r2.mf3_path)
nomi2 = {k: md["name"] for k, md in parti2.items()}
fondo2 = min(m.bounds[0][2] for m in m2.values())
facce2 = [k for k in nomi2 if nomi2[k].startswith("Face")]
corpo2 = [k for k in nomi2 if nomi2[k].startswith("Body")][0]
check("faccia in su: le facce stanno negli ultimi layer",
      all(abs(m2[k].bounds[0][2] - fondo2 - (MAXH - FACE)) < 1e-6
          and abs(m2[k].bounds[1][2] - fondo2 - MAXH) < 1e-6 for k in facce2))
check("faccia in su: il corpo parte dal piatto",
      abs(m2[corpo2].bounds[0][2] - fondo2) < 1e-6)
rossa2 = [k for k in facce2 if nomi2[k].endswith("#dc1e1e")][0]
cx2 = (m2[corpo2].bounds[0][0] + m2[corpo2].bounds[1][0]) / 2
check("faccia in su: niente specchio (rosso a sinistra)", m2[rossa2].centroid[0] < cx2)


# ---------------------------------------------------------------------------
print("\n=== UN COLORE CHE MANCA NON E' UNA BOBINA ===")
senza_blu = IMG.copy()
senza_blu[np.all(senza_blu == BLU, axis=2)] = 255
r3 = generate(senza_blu, params(output_path_3mf=os.path.join(TMP, "noblu.3mf")))
_, _, _, parti3, ps3, _ = leggi_3mf(r3.mf3_path)
check("senza pixel blu: tre filamenti, non quattro",
      len(ps3["filament_colour"]) == 3
      and max(int(md["extruder"]) for md in parti3.values()) == 3,
      ps3["filament_colour"])


# ---------------------------------------------------------------------------
print("\n=== COLORE DI BORDINO E OCCHIELLO ===")


def aree_facce(**kw):
    """Area (mm2) di ogni faccia colorata, per colore."""
    r = generate(IMG, params(cutout_border_mm=2.0, cutout_ring=True, cutout_ring_xy=(450, 40),
                             output_path_3mf=os.path.join(TMP, "bordo.3mf"), **kw))
    _, ms, _, pp, _, _ = leggi_3mf(r.mf3_path)
    return {pp[k]["name"].split()[-1]: m.volume / FACE
            for k, m in ms.items() if pp[k]["name"].startswith("Face")}


base = aree_facce()
scelti = aree_facce(flat_border_rgb=ROSSO, flat_ring_rgb=BLU)
check("bordino rosso: la faccia rossa cresce, la bianca cala",
      scelti["#dc1e1e"] > base["#dc1e1e"] + 5 and scelti["#f5f5f5"] < base["#f5f5f5"] - 5,
      (round(base["#dc1e1e"]), round(scelti["#dc1e1e"])))
check("occhiello blu: la faccia blu cresce, la nera cala",
      scelti["#1e3cdc"] > base["#1e3cdc"] + 2 and scelti["#141414"] < base["#141414"] - 2,
      (round(base["#1e3cdc"]), round(scelti["#1e3cdc"])))
check("i colori scelti non cambiano l'area totale della faccia",
      abs(sum(scelti.values()) - sum(base.values())) < 1e-6)

# Il corpo di un altro colore: un disegno su fondo scuro vuole fianchi e retro neri.
r = generate(IMG, params(flat_body_rgb=(20, 20, 20), output_path_3mf=os.path.join(TMP, "corpo.3mf")))
corpo = [e for nome, hx, e in r.flat_parts if nome.startswith("Body")]
nero = [e for nome, hx, e in r.flat_parts if nome.startswith("Face") and hx == "#141414"]
check("corpo nero: stessa bobina della faccia nera",
      corpo == nero and len(corpo) == 1
      and any(n == "Body #141414" for n, _, _ in r.flat_parts), r.flat_parts)

print("\n=== FRONTE E RETRO ===")


def per_nome(r):
    _, ms, _, pp, _, _ = leggi_3mf(r.mf3_path)
    return {pp[k]["name"]: m for k, m in ms.items()}


uguale = per_nome(generate(IMG, params(flat_back="same",
                                       output_path_3mf=os.path.join(TMP, "fr_same.3mf"))))
facce = {n[5:]: m for n, m in uguale.items() if n.startswith("Face")}
retro = {n[5:]: m for n, m in uguale.items() if n.startswith("Back")}
corpo = uguale[[n for n in uguale if n.startswith("Body")][0]]
check("Same as front: un retro per ogni colore della faccia, stessa area",
      facce.keys() == retro.keys()
      and all(abs(facce[c].volume - retro[c].volume) < 1e-6 for c in facce), sorted(retro))
z0 = min(m.bounds[0][2] for m in uguale.values())   # il 3MF centra il pezzo in Z
zf = [round(v - z0, 3) for v in (min(m.bounds[0][2] for m in facce.values()),
                                 max(m.bounds[1][2] for m in facce.values()))]
zr = [round(v - z0, 3) for v in (min(m.bounds[0][2] for m in retro.values()),
                                 max(m.bounds[1][2] for m in retro.values()))]
zc = [round(v - z0, 3) for v in corpo.bounds[:, 2]]
check("fronte nei primi layer, corpo in mezzo, retro negli ultimi",
      zf == [0, round(FACE, 3)] and zr == [round(BASE + FACE, 3), round(BASE + 2 * FACE, 3)]
      and zc == [round(FACE, 3), round(BASE + FACE, 3)], (zf, zr, zc))
cx = corpo.bounds[:, 0].mean()
check("Same as front: il retro e' allineato alla sagoma (rosso dallo stesso lato del fronte)",
      retro["#dc1e1e"].centroid[0] > cx and facce["#dc1e1e"].centroid[0] > cx)

# Image: l'immagine e' quella vista da dietro. Con lo stesso disegno il rosso,
# guardando il retro dall'alto, deve stare a SINISTRA come nel file.
img_r = per_nome(generate(IMG, params(flat_back="image", flat_back_image=IMG,
                                      output_path_3mf=os.path.join(TMP, "fr_img.3mf"))))
check("Image: il retro si legge dritto da dietro (rosso a sinistra)",
      img_r["Back #dc1e1e"].centroid[0] < cx, img_r["Back #dc1e1e"].centroid[0])
try:
    generate(IMG, params(flat_back="image"))
    check("Image senza immagine viene rifiutato", False)
except ValueError as e:
    check("Image senza immagine viene rifiutato", "Back image" in str(e), e)

print("\n=== RIFIUTI ===")
try:
    generate(IMG, params(keychain_finish_spot=False))
    check("B/N + piatto viene rifiutato", False)
except ValueError as e:
    check("B/N + piatto viene rifiutato", "Spot" in str(e), e)
try:
    generate(IMG, params(base_h=0.2))
    check("un corpo da un layer viene rifiutato", False)
except ValueError as e:
    check("un corpo da un layer viene rifiutato", "Base" in str(e), e)

# Il rilievo a strati non deve accorgersi che la stampa piatta esiste.
r4 = generate(IMG, params(keychain_flat=False, max_h=2.8,
                          output_path_3mf=os.path.join(TMP, "strati.3mf")))
z4 = zipfile.ZipFile(r4.mf3_path)
check("a strati: il 3MF ha ancora i cambi a quota e una parte sola",
      "Metadata/custom_gcode_per_layer.xml" in z4.namelist()
      and z4.read("Metadata/model_settings.config").decode().count("<part ") == 1)


# ---------------------------------------------------------------------------
print("\n=== UI ===")
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from manga_to_3d import Manga3DAppController  # noqa: E402

win = Manga3DAppController()
win.show()
app.processEvents()

win.mode_selector.setCurrentIndex(5)
win.combo_keychain_finish.setCurrentIndex(0)
check("portachiavi Spot: il selettore di stampa c'e' ed e' acceso",
      win.combo_keychain_print.isVisible() and win.combo_keychain_print.isEnabled())
check("a strati: le opzioni piatte sono nascoste",
      not win.flat_options.isVisible() and not win._keychain_flat())
maxh_strati = win.spin_maxh.value()
win.combo_keychain_print.setCurrentIndex(1)
app.processEvents()
check("piatto: compaiono le opzioni, la Base resta, Max Z si spegne",
      win.flat_options.isVisible() and win._keychain_flat()
      and win.spin_base.isEnabled() and not win.spin_maxh.isEnabled())
# 1,8 + 4 x 0,2 = 2,6: diverso dal 2,8 del rilievo a strati, altrimenti la
# prova sul ripristino qui sotto passerebbe senza provare niente
win.spin_base.setValue(1.8)
win.spin_flat_layers.setValue(4)
app.processEvents()
check("piatto: Max Z = Base + layer colorati, e li segue",
      abs(win.spin_maxh.value() - 2.6) < 1e-6 and abs(maxh_strati - 2.6) > 0.1,
      (win.spin_maxh.value(), maxh_strati))
win.combo_keychain_finish.setCurrentIndex(1)
app.processEvents()
check("B/N: il selettore si spegne e il piatto non vale piu'",
      not win.combo_keychain_print.isEnabled() and not win._keychain_flat()
      and not win.flat_options.isVisible() and win.spin_maxh.isEnabled())
check("uscendo dal piatto Max Z torna quello del rilievo a strati",
      abs(win.spin_maxh.value() - maxh_strati) < 1e-6,
      (win.spin_maxh.value(), maxh_strati))
win.combo_keychain_finish.setCurrentIndex(0)
win.toggle_ui_state(disabled=True)
check("durante la generazione i controlli piatti sono bloccati",
      not win.combo_keychain_print.isEnabled() and not win.spin_flat_layers.isEnabled())
win.toggle_ui_state(disabled=False)
check("dopo la generazione tornano come prima",
      win.combo_keychain_print.isEnabled() and win.spin_flat_layers.isEnabled()
      and win.spin_base.isEnabled() and not win.spin_maxh.isEnabled())

# Place deve accendere l'anteprima, come Edit regions: il punto si legge sul
# raster di segmentazione. Prima restava spenta, il click cadeva
# sull'originale a un'altra risoluzione, e "non succedeva nulla".
from PyQt6.QtWidgets import QFileDialog  # noqa: E402
img_path = os.path.join(TMP, "medaglione.png")
cv2.imwrite(img_path, cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (img_path, ""))
win.load_image()
app.processEvents()
win.mode_selector.setCurrentIndex(5)
win.combo_keychain_finish.setCurrentIndex(0)
win.combo_keychain_print.setCurrentIndex(1)
app.processEvents()
win.btn_cutout_preview.setChecked(False)
win.btn_cutout_ring.click()
app.processEvents()
check("Place accende l'anteprima del ritaglio",
      win.btn_cutout_ring.isChecked() and win.btn_cutout_preview.isChecked())
win.on_pixel_clicked(450, 40)
app.processEvents()
check("il click piazza l'occhiello e accende la spunta",
      win.cutout_ring_xy == (450, 40) and win.chk_cutout_ring.isChecked()
      and not win.btn_cutout_ring.isChecked())

# Il nome dice come si stampa: senza, i due file dello stesso disegno sono
# indistinguibili in output/.
from PyQt6.QtWidgets import QMessageBox  # noqa: E402
popup = []
QMessageBox.information = staticmethod(lambda *a, **k: popup.append(a[2]))
QMessageBox.critical = staticmethod(lambda *a, **k: popup.append(a[2]))
win.chk_export_stl.setChecked(False)
win.chk_export_3mf.setChecked(True)
win.generate_stl()
win.worker.wait()
app.processEvents()
nome_3mf = win.worker.params.output_path_3mf
check("il 3MF piatto si chiama <nome>_keychain_ams",
      os.path.basename(nome_3mf).startswith("medaglione_keychain_ams"), nome_3mf)
check("il popup parla di AMS, non di cambi a quota",
      popup and "Flat AMS" in popup[-1] and "Color Changes" not in popup[-1],
      popup[-1][:80] if popup else None)

# Gli STL piatti sono un set: una cartella per generazione, numerata come i
# file. Il 3MF e' uno solo e resta in output/3mf.
win.chk_export_stl.setChecked(True)
cartelle = []
for _ in range(2):
    win.generate_stl()
    win.worker.wait()
    app.processEvents()
    cartelle.append(os.path.dirname(win.worker.result.stl_paths[0]))
stl_root = os.path.join(TMP, "output", "stl")
check("gli STL piatti stanno in output/stl/<nome>/",
      os.path.basename(cartelle[0]).startswith("medaglione_keychain_ams")
      and os.path.dirname(cartelle[0]) == stl_root
      and len(os.listdir(cartelle[0])) == len(win.worker.result.stl_paths) >= 2, (cartelle[0], os.listdir(cartelle[0])))
check("la seconda generazione apre una cartella nuova, non sovrascrive",
      cartelle[0] != cartelle[1] and len(os.listdir(cartelle[1])) == len(win.worker.result.stl_paths)
      and all(os.path.basename(f).startswith(os.path.basename(cartelle[1]) + "_f")
              for f in os.listdir(cartelle[1])), cartelle)
check("il 3MF resta in output/3mf, fuori dalle cartelle",
      os.path.dirname(win.worker.params.output_path_3mf) == os.path.join(TMP, "output", "3mf"))

win.combo_flat_back.setCurrentIndex(1)                # Same as front
due = round(win.spin_base.value() + 2 * win.spin_flat_layers.value()
            * win.spin_layer_height.value(), 3)
check("col retro Max Z conta i layer colorati due volte",
      abs(win.spin_maxh.value() - due) < 1e-6, (win.spin_maxh.value(), due))
win.combo_flat_back.setCurrentIndex(0)

win.mode_selector.setCurrentIndex(3)
app.processEvents()
check("fuori dal portachiavi il pannello (e il selettore) non si vedono",
      not win.combo_keychain_print.isVisible() and win.spin_base.isEnabled()
      and win.spin_maxh.isEnabled())

# Un biglietto da visita: sorgente piccola (640 px per 85 mm, 0,13 mm/px) e
# testo bianco a tratti di 2 px (0,27 mm). La mediana a 5 px se lo mangiava
# anche con l'ugello 0,2, che quel tratto lo stampa.
from engine.mesh_utils import classify_palette_indices
pal = [(245, 245, 245), (20, 20, 20)]
card = np.ones((416, 640), np.int64)
for y in range(40, 400, 12):
    card[y:y + 2, 40:600] = 0                       # righe di "testo" da 2 px
img = np.array(pal, np.uint8)[card]
for noz, atteso in ((0.2, True), (0.4, False)):
    c = classify_palette_indices(img, pal, max_dim=85, min_feature_mm=1.25 * noz)
    tenuto = ((c == 0) & (card == 0)).sum() / (card == 0).sum()
    check(f"testo da 0,27 mm con ugello {noz}: {'resta' if atteso else 'sparisce'}",
          (tenuto > 0.9) == atteso, f"{tenuto:.0%}")

print(f"\n{'OK' if not fails else 'FALLITE: ' + str(len(fails))}")
sys.exit(1 if fails else 0)

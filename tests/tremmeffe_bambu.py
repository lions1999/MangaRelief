"""Il 3MF deve restare un progetto Bambu Studio, non tornare geometria nuda.

Ogni asserzione qui e' costata un giro di prove aprendo file veri in Bambu
Studio 02.08.02.61, e nessuna si deduce dalla documentazione — quella parte del
formato non e' documentata. Sono i fatti misurati, messi dove si rompono
rumorosamente se qualcuno tocca l'esportazione senza saperli:

- senza `BambuStudio:3mfVersion` il file torna "solo geometria";
- `Application` deve cominciare con `BambuStudio-`: lo stesso identico file con
  `MangaRelief-1.0.0` viene rifiutato;
- `project_settings.config` vuoto o `{}` non basta — rompe perfino un progetto
  scritto da Bambu Studio stesso — e la chiave indispensabile e'
  `nozzle_diameter`, non `printer_technology`;
- i filamenti dichiarati sono quelli che si stampano e nessun altro.

E una conseguenza che non e' un difetto ma va detta a chi stampa: Bambu Studio
non FONDE la nostra configurazione col profilo di chi apre il file, ne
costruisce uno nuovo intitolato al progetto — per questo Stampante, Filamenti
e Processo si chiamano tutti "(nomefile.3mf)" — e ogni chiave che non nominiamo
prende il valore di fabbrica. Dichiararne di piu' non aiuterebbe: sarebbe
un'altra impostazione imposta al posto della sua. L'unico rimedio e' dirlo, e
qui si verifica che venga detto davvero.

    python tests/tremmeffe_bambu.py
"""

import json
import os
import re
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zipfile  # noqa: E402

from engine import GenerationMode, GenerationParams, generate  # noqa: E402

fails = []


def check(nome, ok, extra=None):
    # `extra` puo' essere un array numpy, e `if extra` su un array solleva:
    # una prova non deve morire mentre racconta perche' e' fallita.
    coda = "" if extra is None or (isinstance(extra, str) and not extra) else f" | {extra}"
    print(("PASS " if ok else "FAIL ") + nome + coda)
    if not ok:
        fails.append(nome)


def tavola():
    img = np.full((360, 300), 250, np.uint8)
    cv2.rectangle(img, (40, 60), (250, 300), 130, -1)
    cv2.circle(img, (150, 180), 55, 15, -1)
    return img


d = tempfile.mkdtemp(prefix="mr-3mf-")
img = tavola()

for colori in (2, 3, 4):
    p = GenerationParams(mode=GenerationMode.STANDARD, color_mode=colori,
                         max_dim=60, max_res_cap=220,
                         output_path=os.path.join(d, f"{colori}.stl"),
                         output_path_3mf=os.path.join(d, f"{colori}.3mf"))
    r = generate(img, p)
    z = zipfile.ZipFile(r.mf3_path)
    dentro = set(z.namelist())
    radice = z.read("3D/3dmodel.model").decode()
    oggetto = z.read("3D/Objects/object_1.model").decode()
    cfg = json.loads(z.read("Metadata/project_settings.config"))
    cg = z.read("Metadata/custom_gcode_per_layer.xml").decode()
    n = f"{colori} colori"

    # --- lo scheletro che lo rende un progetto
    check(f"{n}: la geometria sta in un file suo, referenziata per percorso",
          "3D/Objects/object_1.model" in dentro
          and 'p:path="/3D/Objects/object_1.model"' in radice
          and "<vertex" not in radice)
    check(f"{n}: il collegamento fra guscio e oggetto c'e'",
          "3D/_rels/3dmodel.model.rels" in dentro)
    for dove, testo in (("radice", radice), ("oggetto", oggetto)):
        check(f"{n}: il flag BambuStudio:3mfVersion e' nel {dove}",
              'name="BambuStudio:3mfVersion"' in testo)
    check(f"{n}: namespace ed estensione richiesta dichiarati",
          "xmlns:BambuStudio=" in radice and 'requiredextensions="p"' in radice)

    # --- il generatore: senza il prefisso giusto la configurazione viene scartata
    app = re.search(r'name="Application">([^<]+)<', radice).group(1)
    check(f"{n}: si dichiara BambuStudio-, o i cambi colore non vengono letti",
          app.startswith("BambuStudio-"), app)
    check(f"{n}: con una versione interpretabile", re.fullmatch(r"[\d.]+", app.split("-", 1)[1]), app)

    # --- le impostazioni di progetto: il minimo, e nient'altro
    check(f"{n}: project_settings non e' vuoto", len(cfg) > 0)
    check(f"{n}: c'e' nozzle_diameter, la chiave senza cui non apre",
          cfg.get("nozzle_diameter") == ["0.4"], cfg.get("nozzle_diameter"))
    # Quello che si dichiara qui sostituisce il profilo di chi apre il file, non
    # ci si aggiunge: ogni chiave in piu' e' un'impostazione imposta a lui.
    # Quindi solo quelle che riguardano *questo* oggetto.
    check(f"{n}: nessuna impostazione di stampa altrui",
          set(cfg) == {"version", "from", "name", "nozzle_diameter", "filament_colour",
                       "layer_height", "initial_layer_print_height", "skirt_loops"},
          sorted(cfg))
    check(f"{n}: l'altezza layer e' quella su cui sono calcolate le quote",
          cfg["layer_height"] == "0.2" and cfg["initial_layer_print_height"] == "0.2",
          (cfg["layer_height"], cfg["initial_layer_print_height"]))
    # Un giro di perimetro sul primo layer, in filamento 1: su una lastra larga
    # non serve, e a occhio sembra un bordo del disegno. Era il valore di
    # fabbrica, ed e' cosi' che si e' scoperto tutto il resto.
    check(f"{n}: niente skirt", cfg["skirt_loops"] == "0", cfg["skirt_loops"])

    # --- i filamenti: quelli che si stampano, in ordine di stampa
    palette = cfg["filament_colour"]
    cambi = re.findall(r'color="(#[0-9a-fA-F]{6})"', cg)
    check(f"{n}: un filamento per colore, non uno di piu'", len(palette) == colori, palette)
    check(f"{n}: si parte dalla carta e si finisce sull'inchiostro",
          palette[0].lower() == "#ffffff" and palette[-1].lower() == "#000000", palette)
    check(f"{n}: i cambi sono i filamenti dal secondo in poi", cambi == palette[1:],
          (cambi, palette[1:]))
    check(f"{n}: un cambio in meno dei filamenti", len(cambi) == colori - 1, len(cambi))
    check(f"{n}: modalita' un-ugello-cambi-manuali",
          '<mode value="MultiAsSingle"/>' in cg)
    zs = [float(x) for x in re.findall(r'top_z="([\d.]+)"', cg)]
    check(f"{n}: nessun cambio fantasma a quota zero", all(q > 0 for q in zs) and zs == sorted(zs), zs)

    # --- la posa, copiata da quella che scrive Bambu Studio
    v = np.array(re.findall(r'<vertex x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"', oggetto),
                 dtype=float)
    mn, mx = v.min(axis=0), v.max(axis=0)
    centro = (mn + mx) / 2
    check(f"{n}: la mesh e' centrata sulla propria origine",
          bool(np.allclose(centro, 0, atol=1e-3)), list(centro.round(4)))
    dx, dy, dz = [float(x) for x in re.search(
        r'<item [^>]*transform="1 0 0 0 1 0 0 0 1 ([-\d.]+) ([-\d.]+) ([-\d.]+)"', radice).groups()]
    check(f"{n}: e la posa sta nell'item, al centro di un piatto da 256",
          (dx, dy) == (128.0, 128.0), (dx, dy))
    check(f"{n}: con la base appoggiata al piatto", abs(mn[2] + dz) < 1e-3, mn[2] + dz)
    print()

# ---------------------------------------------------------------------------
# Il file non dichiara nomi di profilo, e proprio per questo chi lo apre deve
# sceglierli a mano: le due cose vanno verificate insieme, o si finisce col
# togliere l'avvertenza perche' "il file e' a posto" — che e' vero e
# irrilevante.
print("--- i profili di chi apre ---")

cfg_ultimo = json.loads(zipfile.ZipFile(r.mf3_path).read("Metadata/project_settings.config"))
# L'ugello dichiarato deve essere quello scelto: era cablato a 0,4, e chi ne ha
# uno diverso si trovava nel progetto un valore che non e' il suo.
for _n in (0.2, 0.6):
    _p = GenerationParams(mode=GenerationMode.STANDARD, max_dim=60.0, base_h=1.0,
                          max_h=2.4, layer_height=0.1, nozzle_mm=_n,
                          max_res_cap=400, smart_decimate=False,
                          output_path_3mf=os.path.join(d, f"ug{_n}.3mf"))
    _c = json.loads(zipfile.ZipFile(generate(img, _p).mf3_path)
                    .read("Metadata/project_settings.config"))
    check(f"il 3MF dichiara l'ugello scelto ({_n} mm)",
          _c["nozzle_diameter"] == [f"{_n:g}"], _c["nozzle_diameter"])

check("non dichiariamo nomi di profilo (quindi Bambu ne inventa uno col nome del file)",
      not any(k.endswith("_settings_id") for k in cfg_ultimo),
      [k for k in cfg_ultimo if k.endswith("_settings_id")] or sorted(cfg_ultimo))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from manga_to_3d import Manga3DAppController  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)
_src = os.path.join(d, "avviso.png")
cv2.imwrite(_src, np.full((300, 300), 255, np.uint8))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (_src, ""))
_visto = {}
QMessageBox.information = staticmethod(lambda p, t, m: _visto.update(msg=m))

_win = Manga3DAppController()
_win.show()
_win.load_image()
_win.on_generate_done(os.path.join(d, "x.stl"), os.path.join(d, "x.3mf"))
_msg = _visto.get("msg", "")
check("esportando un 3MF, il popup dice di scegliere i propri profili",
      "Printer, Filament and Process" in _msg and "dropdowns" in _msg,
      _msg[:60])
check("...e spiega che i valori mancanti sono quelli di fabbrica",
      "factory values" in _msg)
check("...e chiede di non cambiare l'altezza layer, su cui stanno i cambi",
      "layer" in _msg and "colour changes" in _msg)

_visto.clear()
_win.on_generate_done(os.path.join(d, "x.stl"), "")
check("senza 3MF l'avvertenza non compare (non c'e' nessun progetto da aprire)",
      "BEFORE SLICING" not in _visto.get("msg", ""))
print()

print("TUTTO OK" if not fails else "FALLITI: " + ", ".join(fails))
sys.exit(1 if fails else 0)

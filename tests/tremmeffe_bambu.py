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
    check(f"{n}: cinque chiavi, nessuna impostazione altrui",
          set(cfg) == {"version", "from", "name", "nozzle_diameter", "filament_colour"},
          sorted(cfg))

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

print("TUTTO OK" if not fails else "FALLITI: " + ", ".join(fails))
sys.exit(1 if fails else 0)

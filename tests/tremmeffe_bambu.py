"""Il 3MF deve restare un progetto Bambu Studio, non tornare geometria nuda.

Ogni asserzione qui e' costata un giro di prove aprendo file veri in Bambu
Studio 02.08.02.61, e nessuna si deduce dalla documentazione — quella parte del
formato non e' documentata. Sono i fatti misurati, messi dove si rompono
rumorosamente se qualcuno tocca l'esportazione senza saperli:

- senza `BambuStudio:3mfVersion` il file torna "solo geometria";
- `Application` deve cominciare con `BambuStudio-`: lo stesso identico file con
  `MangaRelief-1.0.0` viene rifiutato;
- `project_settings.config` e' il profilo COMPLETO di un progetto salvato da
  Bambu Studio (assets/bambu/, engine/bambu_profile.py), non il minimo per
  aprirsi. Col minimo (cinque chiavi) Bambu inventava stampante, filamenti e
  processo intitolati al file con i valori di fabbrica, MakerWorld rifiutava
  il processo, e `bambu-studio --slice` andava in crash (0xC0000005). Col
  profilo completo i file si affettano dalla riga di comando con i filamenti
  del profilo (`GFA00`, PLA Basic) e i cambi alle quote dichiarate;
- i filamenti dichiarati sono quelli che si stampano e nessun altro, e le
  ~140 liste per filamento del profilo vanno ridimensionate con loro.

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
from engine.bambu_profile import _per_filament_keys  # noqa: E402

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

    # --- le impostazioni di progetto: un profilo vero, ridimensionato
    check(f"{n}: profilo completo di un progetto Bambu", len(cfg) > 500, len(cfg))
    check(f"{n}: stampante, processo e filamenti sono preset di serie",
          cfg["printer_settings_id"] == "Bambu Lab A1 0.4 nozzle"
          and cfg["print_settings_id"] == "0.20mm Standard @BBL A1"
          and cfg["filament_settings_id"] == ["Bambu PLA Basic @BBL A1"] * colori,
          (cfg["printer_settings_id"], cfg["print_settings_id"], cfg["filament_settings_id"]))
    check(f"{n}: l'ugello e' quello scelto", cfg["nozzle_diameter"] == ["0.4"], cfg["nozzle_diameter"])
    versione = app.split("-", 1)[1]
    check(f"{n}: stessa versione nel guscio, nel profilo e in slice_info",
          cfg["version"] == versione
          and f'value="{versione}"' in z.read("Metadata/slice_info.config").decode(),
          (versione, cfg["version"]))
    # Le liste per filamento seguono il numero di filamenti. Se una restasse a
    # 4 su un file a 2, Bambu leggerebbe bobine fantasma o scarterebbe la chiave.
    # Si controlla la lunghezza ATTESA di ogni tipo di chiave, non "nessuna
    # lista lunga 4": a 2 filamenti 2n, n x n e n + 2 fanno 4 anche loro.
    sbagliate = {k: len(cfg[k]) for k in _per_filament_keys() if len(cfg[k]) != colori}
    check(f"{n}: ogni lista per filamento e' lunga {colori}", not sbagliate, sbagliate)
    check(f"{n}: spurghi {colori}x{colori}, vettore 2n, posti n+2, indici 1..n",
          len(cfg["flush_volumes_matrix"]) == colori * colori
          and len(cfg["flush_volumes_vector"]) == 2 * colori
          and len(cfg["different_settings_to_system"]) == colori + 2
          and cfg["filament_self_index"] == [str(i + 1) for i in range(colori)])
    check(f"{n}: il piatto ha ancora quattro angoli", len(cfg["printable_area"]) == 4)
    check(f"{n}: l'altezza layer e' quella su cui sono calcolate le quote",
          cfg["layer_height"] == "0.2" and cfg["initial_layer_print_height"] == "0.2",
          (cfg["layer_height"], cfg["initial_layer_print_height"]))
    check(f"{n}: niente skirt", cfg["skirt_loops"] == "0", cfg["skirt_loops"])
    blob = json.dumps(cfg).lower()
    check(f"{n}: nessun dato personale nel profilo",
          not any(w in blob for w in ("users\\", "c:\\", "d:\\", "http", "apikey", "access_code")))

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
# L'ugello sceglie il profilo: per 0,2 c'e' un modello suo (A1 0.2, 0.10mm);
# per uno senza modello si usa il piu' vicino e si dichiara il diametro vero.
print("--- ugello e altezza layer ---")
for _n, _stampante in ((0.2, "Bambu Lab A1 0.2 nozzle"), (0.6, "Bambu Lab A1 0.4 nozzle")):
    _p = GenerationParams(mode=GenerationMode.STANDARD, max_dim=60.0, base_h=1.0,
                          max_h=2.4, layer_height=0.1, nozzle_mm=_n,
                          max_res_cap=400, smart_decimate=False,
                          output_path_3mf=os.path.join(d, f"ug{_n}.3mf"))
    _c = json.loads(zipfile.ZipFile(generate(img, _p).mf3_path)
                    .read("Metadata/project_settings.config"))
    check(f"il 3MF dichiara l'ugello scelto ({_n} mm) sul profilo giusto",
          _c["nozzle_diameter"] == [f"{_n:g}"] and _c["printer_settings_id"] == _stampante,
          (_c["nozzle_diameter"], _c["printer_settings_id"]))
    check(f"...a layer 0,1 ({_n} mm)", _c["layer_height"] == "0.1", _c["layer_height"])
    # 0,1 e' il layer del processo 0.2 (nessuna modifica); su 0,4 e' una
    # modifica, e va dichiarata come tale o Bambu non la mostra.
    _mod = _c["different_settings_to_system"][0].split(";")
    check(f"...e la modifica al processo e' dichiarata solo se c'e' ({_n} mm)",
          ("layer_height" in _mod) == (_n != 0.2), _c["different_settings_to_system"][0])

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
check("esportando un 3MF, il popup dice quale profilo c'e' dentro",
      "Bambu Lab A1" in _msg and "PLA Basic" in _msg, _msg[:80])
check("...e come passare alla propria stampante",
      "Printer dropdown" in _msg)
check("...e chiede di non cambiare l'altezza layer, su cui stanno i cambi",
      "layer" in _msg and "colour changes" in _msg)
check("...e non dice piu' che i profili si chiamano come il file",
      "factory values" not in _msg and "named after the file" not in _msg)

# I cambi colore li porta il file: il popup non deve piu' chiedere di
# aggiungerli, e Arachne e' gia' nel profilo.
check("...e dice che i cambi colore sono gia' nel 3MF, non da aggiungere",
      "already in the 3MF" in _msg and "add these layer pauses" not in _msg.lower())
check("...e non consiglia Arachne, che il profilo attiva gia'", "Arachne" not in _msg)

_visto.clear()
_win.on_generate_done(os.path.join(d, "x.stl"), "")
_solo_stl = _visto.get("msg", "")
check("senza 3MF il riquadro del profilo non compare (non c'e' nessun progetto)",
      "profile inside the project" not in _solo_stl)
check("solo STL: cambi colore e Arachne da impostare a mano",
      "set these by hand" in _solo_stl and "Arachne" in _solo_stl)
print()

print("TUTTO OK" if not fails else "FALLITI: " + ", ".join(fails))
sys.exit(1 if fails else 0)

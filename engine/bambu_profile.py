"""Il profilo di stampa dentro il 3MF: quello vero, non cinque chiavi.

Un 3MF con l'intestazione Bambu ma con un project_settings.config minimo si
apre come progetto, ma Bambu costruisce stampante, filamenti e processo
intitolati al file e riempie ogni chiave mancante coi valori di fabbrica
(ugello a 200 gradi, muro esterno a 60 mm/s...). Chi lo scarica deve
riscegliere tutto, e MakerWorld rifiuta un processo che porta il nome del file.
In piu' la riga di comando di Bambu Studio (--slice) va in crash su quei file,
mentre un progetto salvato da Bambu si affetta senza problemi.

Il rimedio e' quello trovato in dnd-tilegen (docs/bambu-3mf.md): mettere nel
file il profilo COMPLETO di un progetto salvato da Bambu Studio, e cambiare
solo quello che decide il generatore. I modelli stanno in assets/bambu/, uno
per ugello, presi da due progetti risalvati in Bambu Studio 02.08.02.61:

    A1_0.4_project_settings.config   A1 0.4, 0.20mm Standard, 4 x PLA Basic
    A1_0.2_project_settings.config   A1 0.2, 0.10mm Standard, 3 x PLA Basic

Nessun dato personale dentro (verificato: niente host, chiavi o percorsi;
`host_type` e' il valore di fabbrica).

Il numero di filamenti non e' fisso (2-4 in MangaRelief), e il profilo ne porta
il conto in molte chiavi. Le regole sono MISURATE confrontando i due modelli
(3 e 4 filamenti, 581 chiavi ciascuno, nessuna chiave in piu' o in meno):

- 142 chiavi hanno un valore per filamento (lunghezza n);
- flush_volumes_matrix e' n x n, flush_volumes_vector 2n,
  filament_dev_ams_drying_* 2n o 4n (blocchi uguali per filamento);
- different_settings_to_system e' n + 2: processo, un posto per filamento,
  stampante.
Fra i filamenti cambiano solo colore, colore "multi", indice e mappa ugello:
tutto il resto e' lo stesso PLA ripetuto, quindi ridimensionare una lista vuol
dire ripeterne il primo elemento (o il primo blocco).
"""

import json
from typing import List, Optional, Tuple

from .resources import asset_path

# ugello -> modello. Sono i due profili che l'autore usa davvero.
_TEMPLATES = {
    0.4: "A1_0.4_project_settings.config",
    0.2: "A1_0.2_project_settings.config",
}
# Le chiavi a blocchi: lunghezza = n * blocco. Il blocco si ricava dal modello.
_BLOCK_KEYS = ("flush_volumes_vector", "filament_dev_ams_drying_ams_limitations",
               "filament_dev_ams_drying_temperature", "filament_dev_ams_drying_time")


def _load(name: str) -> dict:
    with open(asset_path("bambu", name), encoding="utf-8") as f:
        return json.load(f)


def _template_for(nozzle_mm: float) -> Tuple[dict, bool]:
    """Il modello per quell'ugello; se non c'e', il piu' vicino (e False)."""
    best = min(_TEMPLATES, key=lambda n: abs(n - float(nozzle_mm)))
    return _load(_TEMPLATES[best]), abs(best - float(nozzle_mm)) < 1e-6


_PER_FILAMENT = None


def _per_filament_keys() -> set:
    """Le chiavi con un valore per filamento, riconosciute confrontando i
    modelli: lo sono quelle lunghe n in TUTTI e due (n = 4 e n = 3).

    Guardarne uno solo non basta: `printable_area` (i quattro angoli del
    piatto) e `head_wrap_detect_zone` sono lunghe 4 come i filamenti del
    modello da 0,4, e venivano accorciate a 3 — un piatto con tre angoli.
    Con due modelli a numero di filamenti diverso una coincidenza del genere
    dovrebbe ripetersi su entrambi, e non succede.
    """
    global _PER_FILAMENT
    if _PER_FILAMENT is None:
        cfgs = [_load(n) for n in _TEMPLATES.values()]
        keys = set(cfgs[0])
        for c in cfgs:
            n = len(c["filament_colour"])
            keys &= {k for k, v in c.items() if isinstance(v, list) and len(v) == n}
        _PER_FILAMENT = keys
    return _PER_FILAMENT


def _resize(seq: list, old_n: int, new_n: int) -> list:
    """Una lista per filamento portata da old_n a new_n filamenti."""
    block = len(seq) // old_n
    first = seq[:block]
    return [v for i in range(new_n) for v in (seq[i * block:(i + 1) * block] if i < old_n else first)]


def _flush_matrix(seq: list, old_n: int, new_n: int) -> list:
    """La matrice degli spurghi n x n. Ogni coppia nuova prende il valore
    medio fuori diagonale del modello; la diagonale resta zero."""
    vals = [float(seq[r * old_n + c]) for r in range(old_n) for c in range(old_n) if r != c]
    fill = str(int(round(sum(vals) / len(vals)))) if vals else "0"
    out = []
    for r in range(new_n):
        for c in range(new_n):
            if r == c:
                out.append("0")
            elif r < old_n and c < old_n:
                out.append(seq[r * old_n + c])
            else:
                out.append(fill)
    return out


def build_project_settings(palette_hex: List[str], nozzle_mm: float = 0.4,
                           layer_height: Optional[float] = None) -> Tuple[str, str]:
    """project_settings.config per questi filamenti, e la versione di Bambu
    Studio che il modello dichiara (va ripetuta in 3dmodel.model e in
    slice_info.config, come in un progetto vero).

    palette_hex : i colori dei filamenti, dal primo all'ultimo.
    layer_height: se diverso da quello del processo del modello, lo si
                  sovrascrive (con il primo layer, perche' le quote dei cambi
                  colore sono contate su un passo uniforme) e lo si dichiara
                  come modifica al processo: Bambu mostra il processo di
                  serie con l'asterisco, non uno intitolato al file.
    """
    cfg, exact = _template_for(nozzle_mm)
    old_n = len(cfg["filament_colour"])
    new_n = len(palette_hex)
    if new_n < 1:
        raise ValueError("At least one filament is needed.")

    per = _per_filament_keys()
    for k, v in list(cfg.items()):
        if not isinstance(v, list):
            continue
        if k == "flush_volumes_matrix":
            cfg[k] = _flush_matrix(v, old_n, new_n)
        elif k == "different_settings_to_system":
            cfg[k] = [v[0]] + _resize(v[1:1 + old_n], old_n, new_n) + [v[-1]]
        elif k in per or k in _BLOCK_KEYS:
            cfg[k] = _resize(v, old_n, new_n)

    colours = [h if h.startswith("#") else "#" + h for h in palette_hex]
    cfg["filament_colour"] = colours
    cfg["filament_multi_colour"] = list(colours)
    cfg["filament_self_index"] = [str(i + 1) for i in range(new_n)]
    cfg["filament_nozzle_map"] = ["1"] + ["0"] * (new_n - 1)

    changed = []
    if layer_height is not None and abs(float(cfg["layer_height"]) - float(layer_height)) > 1e-6:
        lh = f"{float(layer_height):g}"
        cfg["layer_height"] = lh
        cfg["initial_layer_print_height"] = lh
        changed += ["layer_height", "initial_layer_print_height"]
    if changed:
        prev = [s for s in cfg["different_settings_to_system"][0].split(";") if s]
        cfg["different_settings_to_system"][0] = ";".join(sorted(set(prev + changed)))
    if not exact:
        # Ugello senza un modello suo: si dichiara il diametro vero sulla
        # stampante piu' vicina. Bambu la segnera' come modificata — meglio
        # che un ugello sbagliato.
        cfg["nozzle_diameter"] = [f"{float(nozzle_mm):g}"]
        prev = [s for s in cfg["different_settings_to_system"][-1].split(";") if s]
        cfg["different_settings_to_system"][-1] = ";".join(sorted(set(prev + ["nozzle_diameter"])))

    return json.dumps(cfg, indent=4, ensure_ascii=False), cfg.get("version", "02.08.02.61")

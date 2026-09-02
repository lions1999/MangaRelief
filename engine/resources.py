"""
Risoluzione dei percorsi degli asset usati dal motore (template STL dei deckbox
e delle cover, loghi TCG, preset telefoni).

Il motore non deve sapere nulla di come è impacchettata l'applicazione che lo
ospita: l'app desktop gira sia da sorgente sia congelata da PyInstaller, il
backend web girerà dentro un container con un layout diverso. Per questo la
cartella degli asset è risolta in tre modi, in ordine di priorità:

  1. override esplicito impostato da chi ospita il motore (set_assets_dir)
  2. cartella temporanea di PyInstaller (sys._MEIPASS), per l'eseguibile desktop
  3. <radice del progetto>/assets, cioè il comportamento da sorgente
"""

import os
import sys

_ASSETS_DIR = None


def set_assets_dir(path: str) -> None:
    """Imposta esplicitamente la cartella degli asset del motore.
    Da usare quando il motore viene ospitato fuori dal repo (es. container web)."""
    global _ASSETS_DIR
    _ASSETS_DIR = os.path.abspath(path) if path else None


def assets_dir() -> str:
    """Cartella degli asset attualmente in uso."""
    if _ASSETS_DIR:
        return _ASSETS_DIR
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.join(meipass, "assets")
    # engine/resources.py -> engine/ -> radice del progetto
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def asset_path(*parts: str) -> str:
    """Percorso assoluto di un asset del motore, es. asset_path('phone_presets.json')."""
    return os.path.join(assets_dir(), *parts)

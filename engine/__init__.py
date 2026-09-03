"""
MangaRelief engine — conversione di immagini 2D in mesh 3D a terrazze per la
stampa multicolore.

Il pacchetto non dipende da PyQt: può essere usato dall'app desktop (avvolto in
un QThread) o da un servizio web (avvolto in un job asincrono). L'ingresso
principale è `generate`.

    from engine import GenerationParams, GenerationMode, generate

    params = GenerationParams(mode=GenerationMode.SPOT_COLOR, max_dim=200.0, ...)
    result = generate(image, params, progress=lambda pct, msg: ...)
"""

from .params import GenerationMode, GenerationParams, GenerationResult
from .pipeline import (generate, companion_path_for, standard_heightmap,
                       prepare_source_image, tone_targets,
                       TCG_LOGO_MAP)
from .color_utils import bw_coverage_map, ink_level
from .resources import asset_path, assets_dir, set_assets_dir

__all__ = [
    "GenerationMode",
    "GenerationParams",
    "GenerationResult",
    "generate",
    "companion_path_for",
    "standard_heightmap",
    "prepare_source_image",
    "tone_targets",
    "bw_coverage_map",
    "ink_level",
    "TCG_LOGO_MAP",
    "asset_path",
    "assets_dir",
    "set_assets_dir",
]

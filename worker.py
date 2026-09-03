"""
Adattatore Qt sopra il motore di generazione.

Tutta la logica vive in `engine.pipeline.generate`: qui restano solo il thread e
la traduzione fra i callback del motore e i segnali Qt che la UI ascolta.
"""

import gc

from PyQt6.QtCore import QThread, pyqtSignal

from engine import GenerationParams, generate, companion_path_for, TCG_LOGO_MAP  # noqa: F401


class MeshWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str, str)   # (stl_path, path_3mf)
    finished_err = pyqtSignal(str)

    # Riesportato per compatibilità con il codice che lo leggeva da qui
    TCG_LOGO_MAP = TCG_LOGO_MAP

    def __init__(self, params: GenerationParams, image):
        super().__init__()
        self.params = params
        self.image = image
        self.cancel_requested = False
        self.result = None

    @staticmethod
    def companion_path_for(plate_path: str) -> str:
        """Percorso del bumper/case TPU accanto alla plate."""
        return companion_path_for(plate_path)

    def run(self):
        try:
            result = generate(
                self.image,
                self.params,
                progress=lambda pct, msg: self.progress.emit(pct, msg),
                should_cancel=lambda: self.cancel_requested,
            )

            gc.collect()
            self.result = result   # la UI legge da qui le quote di cambio REALI
            print(f"[Profiling] TOTAL REFACTORED TIME: {result.elapsed_s:.2f}s")

            if self.params.is_deckbox_mode:
                # In deckbox la UI mostra per primo il 3MF della piastra completa
                self.finished_ok.emit(result.mf3_path or "", result.stl_path or "")
            else:
                self.finished_ok.emit(result.stl_path or "", result.mf3_path or "")

        except Exception as e:
            self.finished_err.emit(str(e))

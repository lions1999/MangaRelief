class DeckboxConfig:
    """Standard dimensions for the deckbox templates."""
    WALL_WIDTH = 70.0
    WALL_HEIGHT = 92.0
    DEBOSS_DEPTH = 2.0
    BASE_THICKNESS = 4.0
    MIN_SOLID_WALL_THICKNESS = 1.5

    # Gap between front and lid in the combined full_deckbox plate
    PLATE_GAP_MM = 5.0

    # Lid Logo (Plug & Play)
    PLUG_W = 65.0
    PLUG_H = 31.0
    PLUG_Z = 3.0
    ENGRAVE_FLOOR = 0.4
    NOTCH_Y_OFFSET = 15.0


# Filament slot colours injected into Bambu Studio 3MF metadata (Light → Dark)
SLOT_COLORS_3MF = ["#C8C8C8", "#646464", "#000000", "#1a1a1a"]

# Spot Color mode: ruoli neutri fissi della palette (base chiara e top scuro).
# Gli accenti scelti dall'utente si inseriscono tra questi due estremi.
SPOT_BASE_RGB = (245, 245, 245)
SPOT_TOP_RGB  = (20, 20, 20)

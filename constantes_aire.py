# ══════════════════════════════════════════════════════
#  constantes.py
# ══════════════════════════════════════════════════════

# ── Tipos de celda ─────────────────────────────────────
VACIA     = 'vacia'
AIRE      = 'aire'
OBSTACULO = 'obstaculo'

# ── Intensidades ───────────────────────────────────────
DEBIL  = 1
NORMAL = 2
FUERTE = 3

# ── Direcciones ────────────────────────────────────────
NORTE = 'N'
SUR   = 'S'
ESTE  = 'E'
OESTE = 'O'

DIRS = [NORTE, SUR, ESTE, OESTE]

# para cada dirección, cuál es la contraria
OPUESTO = {
    NORTE: SUR,
    SUR:   NORTE,
    ESTE:  OESTE,
    OESTE: ESTE
}

# para cada dirección, cuáles son las dos perpendiculares
PERPENDICULAR = {
    NORTE: [ESTE, OESTE],
    SUR:   [ESTE, OESTE],
    ESTE:  [NORTE, SUR],
    OESTE: [NORTE, SUR]
}

# desplazamiento (df, dc) para moverse en cada dirección
VECINDAD = {
    NORTE: (-1,  0),
    SUR:   (+1,  0),
    ESTE:  ( 0, +1),
    OESTE: ( 0, -1)
}

# ── Configuración de simulación ────────────────────────
FILAS  = 40
COLS   = 40

# cada cuántos pasos sin interactuar baja un nivel de intensidad
PASOS_DEGRADACION = 7

# ══════════════════════════════════════════════════════
#  celda.py
# ══════════════════════════════════════════════════════

from constantes_aire import VACIA, AIRE, OBSTACULO

def crear_celda_vacia(presion=0):
    return {
        'tipo':                    VACIA,
        'direccion':               None,
        'intensidad':              None,
        'presion':                 presion,
        'presion_base':            presion,
        'pasos_sin_interactuar':   0
    }

def crear_celda_aire(direccion, intensidad, presion=0, presion_base=0, pasos=0):
    return {
        'tipo':                    AIRE,
        'direccion':               direccion,
        'intensidad':              intensidad,
        'presion':                 presion,
        'presion_base':            presion_base,
        'pasos_sin_interactuar':   pasos
    }

def crear_obstaculo():
    return {
        'tipo':                    OBSTACULO,
        'direccion':               None,
        'intensidad':              None,
        'presion':                 0,
        'presion_base':            0,
        'pasos_sin_interactuar':   0
    }

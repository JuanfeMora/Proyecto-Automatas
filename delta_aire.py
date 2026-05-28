# ══════════════════════════════════════════════════════
#  delta_aire.py — función de transición del autómata
# ══════════════════════════════════════════════════════
#
#  delta(celda_actual, vecinos) → celda_nueva
#
#  Recibe el estado de una celda y sus 4 vecinos cardinales
#  en el tiempo t, y devuelve el nuevo estado en t+1.
#
#  ORDEN DE PRIORIDAD (de mayor a menor):
#    1. Obstáculo          → siempre inmutable
#    2. Colisiones         → entre corrientes que se encuentran
#    3. Presión            → atracción/repulsión por gradiente
#    4. Propagación libre  → la corriente avanza sin obstáculos
#    5. Celda vacía        → recibe aire si algún vecino viene hacia acá
# ══════════════════════════════════════════════════════

from constantes_aire import (
    VACIA, AIRE, OBSTACULO,
    DEBIL, NORMAL, FUERTE,
    NORTE, SUR, ESTE, OESTE,
    DIRS, OPUESTO, PERPENDICULAR, VECINDAD,
    PASOS_DEGRADACION
)
from celda_aire import crear_celda_vacia, crear_celda_aire, crear_obstaculo


# ══════════════════════════════════════════════════════
#  UTILIDADES INTERNAS
# ══════════════════════════════════════════════════════

def _intensidad_siguiente(intensidad):
    """Baja un nivel de intensidad. Si ya es DEBIL, retorna None (desaparece)."""
    if intensidad == FUERTE:
        return NORMAL
    if intensidad == NORMAL:
        return DEBIL
    return None  # DEBIL → desaparece



def _viene_hacia_aqui(vecino, direccion_hacia_vecino):
    """
    ¿La corriente del vecino viene en dirección contraria?
    Es decir, ¿está apuntando hacia esta celda?

    Ejemplo: vecino al NORTE que apunta al SUR → viene hacia acá.
    direccion_hacia_vecino = NORTE  →  OPUESTO[NORTE] = SUR
    El vecino viene hacia acá si su dirección == SUR.
    """
    if vecino['tipo'] != AIRE:
        return False
    return vecino['direccion'] == OPUESTO[direccion_hacia_vecino]


def _presion_efectiva(celda):
    """
    Retorna la presión de una celda solo si es > 0.
    Las celdas con presión 0 no participan en las interacciones de presión.
    """
    return celda['presion'] if celda['presion'] > 0 else None


# ══════════════════════════════════════════════════════
#  REGLAS DE COLISIÓN
# ══════════════════════════════════════════════════════

def _resolver_colision_frontal(corriente, intensidad_enemiga, direccion_enemiga):
    """
    Dos corrientes chocan de frente (direcciones opuestas).

    Regla A — misma intensidad:
        Ambas se dispersan perpendicularmente con la misma intensidad.
        Retorna lista de (dirección, intensidad) para las corrientes nuevas.

    Regla B — diferente intensidad:
        Gana la más fuerte, pero pierde un nivel de intensidad.
        Retorna lista con una sola corriente.
    """
    intensidad_propia = corriente['intensidad']
    dir_propia        = corriente['direccion']

    if intensidad_propia == intensidad_enemiga:
        # Regla A: dispersión perpendicular
        perps = PERPENDICULAR[dir_propia]
        return [(d, intensidad_propia) for d in perps]
    elif intensidad_propia > intensidad_enemiga:
        # Regla B: gano yo, pero pierdo un nivel
        nueva_int = _intensidad_siguiente(intensidad_propia)
        if nueva_int is None:
            return []
        return [(dir_propia, nueva_int)]
    else:
        # Regla B: gana el enemigo → esta corriente desaparece
        return []


def _corrientes_que_llegan(celda_actual, vecinos):
    """
    Devuelve una lista de vecinos cuya corriente apunta hacia esta celda.
    Solo incluye vecinos de tipo AIRE.
    """
    llegando = []
    for dir_vecino, vecino in vecinos.items():
        if _viene_hacia_aqui(vecino, dir_vecino):
            llegando.append((dir_vecino, vecino))
    return llegando


# ══════════════════════════════════════════════════════
#  REGLA DE OBSTÁCULO (rebote o dispersión lateral)
# ══════════════════════════════════════════════════════

def _interaccion_con_obstaculo(corriente, vecinos):
    """
    La corriente choca contra un obstáculo (o borde) en su dirección.

    Caso 1 — obstáculo cerrado (sin laterales libres):
        Rebota en dirección contraria, misma intensidad.

    Caso 2 — hay laterales libres:
        Se dispersa en las dos perpendiculares con intensidad - 1.
        Si intensidad era DEBIL, desaparece.
        Si solo uno de los lados está libre, va solo hacia ese.
    """
    dir_corriente = corriente['direccion']
    perps         = PERPENDICULAR[dir_corriente]
    intensidad    = corriente['intensidad']

    # ¿Cuáles laterales están libres (no obstáculo)?
    laterales_libres = [
        d for d in perps
        if vecinos[d]['tipo'] != OBSTACULO
    ]

    if not laterales_libres:
        # Caso 1: rebote
        return [(OPUESTO[dir_corriente], intensidad)]
    else:
        # Caso 2: dispersión lateral con intensidad - 1
        nueva_int = _intensidad_siguiente(intensidad)
        if nueva_int is None:
            return []
        return [(d, nueva_int) for d in laterales_libres]


# ══════════════════════════════════════════════════════
#  REGLA DE PRESIÓN
# ══════════════════════════════════════════════════════

def _interaccion_con_presion(corriente, vecinos):
    """
    Solo se evalúa si la corriente no tuvo colisiones en este paso.

    La celda actual tiene presión P. Se revisan los vecinos:
      - Vecino con presión >= P+2:  la corriente se mueve hacia él.
      - Vecino con presión <= P-2:  actúa como obstáculo (sin perder intensidad).
      - Si todos iguales:           sigue su camino normal.

    Reglas adicionales:
      - Presión 0 NO cuenta para estas interacciones.
      - La diferencia debe ser de 2 o más niveles.

    Retorna:
      - None   si no hay interacción de presión (seguir propagación normal)
      - lista  de (dirección, intensidad) si hay redirección por presión
    """
    presion_actual = corriente['presion']

    # --- Atracción: vecino con presión mucho mayor ---
    mejor_dir      = None
    mejor_presion  = -1
    for dir_vec, vecino in vecinos.items():
        p = _presion_efectiva(vecino)
        if p is not None and p - presion_actual >= 2:
            if p > mejor_presion:
                mejor_presion = p
                mejor_dir     = dir_vec

    if mejor_dir is not None:
        return [(mejor_dir, corriente['intensidad'])]

    # --- Repulsión: vecino adelante con presión mucho menor ---
    dir_corriente  = corriente['direccion']
    vecino_frente  = vecinos[dir_corriente]
    p_frente       = _presion_efectiva(vecino_frente)

    if p_frente is not None and presion_actual - p_frente >= 2:
        # Actúa como obstáculo, pero SIN perder intensidad
        perps          = PERPENDICULAR[dir_corriente]
        laterales_ok   = [
            d for d in perps
            if not (_presion_efectiva(vecinos[d]) is not None
                    and presion_actual - _presion_efectiva(vecinos[d]) >= 2)
            and vecinos[d]['tipo'] != OBSTACULO
        ]

        if not laterales_ok:
            # rebote sin pérdida de intensidad
            return [(OPUESTO[dir_corriente], corriente['intensidad'])]
        else:
            return [(d, corriente['intensidad']) for d in laterales_ok]

    return None   # sin interacción de presión


# ══════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL delta
# ══════════════════════════════════════════════════════

def delta(celda_actual, vecinos):
    """
    Función de transición del autómata.

    Parámetros
    ----------
    celda_actual : dict  — estado de la celda en t
    vecinos      : dict  — {'N': celda, 'S': celda, 'E': celda, 'O': celda}

    Retorna
    -------
    dict — nuevo estado de la celda en t+1
    """

    tipo = celda_actual['tipo']

    # ══════════════════════════════════════════════════
    #  REGLA 0: Obstáculo — siempre inmutable
    # ══════════════════════════════════════════════════
    if tipo == OBSTACULO:
        return crear_obstaculo()

    # ══════════════════════════════════════════════════
    #  Calcular presión nueva de esta celda:
    #  sube 1 si hay aire entrando, baja 1 si el aire se va.
    #  (nunca baja de presion_base, nunca sube de 4)
    # ══════════════════════════════════════════════════
    presion_base   = celda_actual['presion_base']
    presion_actual = celda_actual['presion']

    # ¿Hay corrientes que llegan a esta celda?
    llegando = _corrientes_que_llegan(celda_actual, vecinos)
    hay_aire_entrando = len(llegando) > 0

    # ¿La corriente actual (si existe) se va a mover fuera?
    # (se resuelve más abajo; provisionalmente calculamos la presión base)
    nueva_presion = presion_actual
    if hay_aire_entrando:
        nueva_presion = min(4, presion_actual + 1)

    # ══════════════════════════════════════════════════
    #  REGLA 1: Celda vacía
    #  Solo recibe aire si algún vecino viene hacia acá.
    # ══════════════════════════════════════════════════
    if tipo == VACIA:
        if not llegando:
            # No llega nada: la presión vuelve hacia su base
            nueva_presion = max(presion_base, presion_actual - 1) if hay_aire_entrando is False else presion_actual
            return crear_celda_vacia(presion=max(presion_base, nueva_presion))

        # Si llega más de una corriente, se resuelven entre ellas
        # (la celda es el "ring" donde colisionan)
        if len(llegando) == 1:
            dir_origen, vecino_origen = llegando[0]
            # La corriente viene de dir_origen → avanza en OPUESTO[dir_origen]
            nueva_dir = OPUESTO[dir_origen]
            return crear_celda_aire(
                direccion    = nueva_dir,
                intensidad   = vecino_origen['intensidad'],
                presion      = min(4, nueva_presion),
                presion_base = presion_base,
                pasos        = vecino_origen['pasos_sin_interactuar']
            )

        # Llegan 2, 3 o 4 corrientes simultáneamente → colisión en vacío
        # Tratamos como si fuera la celda de una corriente entrante
        # (tomamos la más fuerte como "ganadora" y la dejamos pasar;
        #  las demás se anulan entre sí)
        llegando_sorted = sorted(llegando, key=lambda x: x[1]['intensidad'], reverse=True)
        if len(llegando) == 2:
            dir_a, vec_a = llegando_sorted[0]
            dir_b, vec_b = llegando_sorted[1]
            # Si son opuestas → colisión frontal
            if OPUESTO[dir_a] == dir_b:
                resultados = _resolver_colision_frontal(
                    {'direccion': OPUESTO[dir_a], 'intensidad': vec_a['intensidad']},
                    vec_b['intensidad'],
                    dir_b
                )
            else:
                # Perpendiculares: gana la más fuerte
                resultados = [(OPUESTO[dir_a], vec_a['intensidad'])]

            if not resultados:
                return crear_celda_vacia(presion=min(4, nueva_presion), presion_base=presion_base)

            dir_nueva, int_nueva = resultados[0]
            return crear_celda_aire(
                direccion    = dir_nueva,
                intensidad   = int_nueva,
                presion      = min(4, nueva_presion),
                presion_base = presion_base
            )

        # 4 corrientes → se anulan (regresan en sentido contrario)
        # Retornar vacío con presión alta momentánea
        return crear_celda_vacia(presion=min(4, nueva_presion), presion_base=presion_base)

    # ══════════════════════════════════════════════════
    #  A partir de aquí: celda_actual es AIRE
    # ══════════════════════════════════════════════════

    dir_actual   = celda_actual['direccion']
    int_actual   = celda_actual['intensidad']
    pasos        = celda_actual['pasos_sin_interactuar']

    # ── Degradación por inactividad ────────────────────
    # Si lleva PASOS_DEGRADACION sin interactuar, baja intensidad
    nueva_int = int_actual
    if pasos >= PASOS_DEGRADACION:
        nueva_int = _intensidad_siguiente(int_actual)
        if nueva_int is None:
            # La corriente se extingue
            return crear_celda_vacia(presion=max(presion_base, presion_actual - 1))

    # ── Vecino en la dirección de avance ──────────────
    vecino_frente = vecinos[dir_actual]

    # ══════════════════════════════════════════════════
    #  REGLA 2: Colisión con obstáculo físico
    # ══════════════════════════════════════════════════
    if vecino_frente['tipo'] == OBSTACULO:
        resultados = _interaccion_con_obstaculo(
            {'direccion': dir_actual, 'intensidad': nueva_int},
            vecinos
        )
        if not resultados:
            return crear_celda_vacia(presion=max(presion_base, presion_actual - 1))

        # Si hay múltiples salidas (dispersión), esta celda toma la primera
        # Las demás se generarán porque las celdas laterales recibirán aire
        dir_nueva, int_nueva = resultados[0]
        return crear_celda_aire(
            direccion    = dir_nueva,
            intensidad   = int_nueva,
            presion      = min(4, nueva_presion),
            presion_base = presion_base
        )

    # ══════════════════════════════════════════════════
    #  REGLA 3: Colisión frontal con otra corriente
    # ══════════════════════════════════════════════════
    if _viene_hacia_aqui(vecino_frente, dir_actual):
        resultados = _resolver_colision_frontal(
            {'direccion': dir_actual, 'intensidad': nueva_int},
            vecino_frente['intensidad'],
            vecino_frente['direccion']
        )
        if not resultados:
            return crear_celda_vacia(presion=max(presion_base, presion_actual - 1))

        dir_nueva, int_nueva = resultados[0]
        return crear_celda_aire(
            direccion    = dir_nueva,
            intensidad   = int_nueva,
            presion      = min(4, nueva_presion),
            presion_base = presion_base
        )

    # ══════════════════════════════════════════════════
    #  REGLA 4: Cuatro corrientes en cruz (rebote total)
    #  Si las 4 direcciones tienen corrientes hacia el centro
    # ══════════════════════════════════════════════════
    corrientes_entrantes = [
        d for d in DIRS
        if _viene_hacia_aqui(vecinos[d], d)
    ]
    if len(corrientes_entrantes) == 4:
        # Todas rebotan → esta celda invierte su dirección
        return crear_celda_aire(
            direccion    = OPUESTO[dir_actual],
            intensidad   = nueva_int,
            presion      = min(4, nueva_presion),
            presion_base = presion_base
        )

    # ══════════════════════════════════════════════════
    #  REGLA 5: Interacción con presión
    #  Solo si no hubo colisión directa
    # ══════════════════════════════════════════════════
    resultado_presion = _interaccion_con_presion(celda_actual, vecinos)
    if resultado_presion is not None:
        if not resultado_presion:
            return crear_celda_vacia(presion=max(presion_base, presion_actual - 1))
        dir_nueva, int_nueva = resultado_presion[0]
        return crear_celda_aire(
            direccion    = dir_nueva,
            intensidad   = int_nueva,
            presion      = min(4, nueva_presion),
            presion_base = presion_base,
            pasos        = 0   # interactuó → resetear contador
        )

    # ══════════════════════════════════════════════════
    #  REGLA 6: Propagación libre
    #  No hay obstáculos ni colisiones → la corriente avanza
    # ══════════════════════════════════════════════════
    # La celda en frente está vacía o es otra corriente que NO viene hacia acá
    # → Esta celda se vacía (la corriente se movió al siguiente paso)
    # La lógica de "recibir la corriente" la maneja la celda destino en REGLA 1.

    nueva_presion_libre = max(presion_base, presion_actual - 1)
    c = crear_celda_vacia(presion=nueva_presion_libre)
    c['presion_base'] = presion_base
    return c
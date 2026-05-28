# ══════════════════════════════════════════════════════
#  main_aire.py
# ══════════════════════════════════════════════════════

from constantes_aire import NORTE, SUR, ESTE, OESTE, DEBIL, NORMAL, FUERTE, FILAS, COLS
from automata_aire import AutomataCelular

# Mapas de texto → constante (para el menú de texto)
_DIR = {'N': NORTE, 'S': SUR, 'E': ESTE, 'O': OESTE}
_INT = {'1': DEBIL, '2': NORMAL, '3': FUERTE}

# ══════════════════════════════════════════════════════
#  API PARA LA INTERFAZ GRÁFICA
#  (sus compañeros importan y llaman estas funciones)
# ══════════════════════════════════════════════════════

def api_poner_aire(automata, fila, col, direccion, intensidad):
    """
    Coloca corriente de aire.
    direccion  → 'N' | 'S' | 'E' | 'O'
    intensidad → 1 (débil) | 2 (normal) | 3 (fuerte)
    """
    dir_real = _DIR.get(direccion, direccion)
    int_real = _INT.get(str(intensidad), intensidad)
    automata.poner_aire(fila, col, dir_real, int_real)

def api_poner_obstaculo(automata, fila, col):
    """Coloca un obstáculo en (fila, col)."""
    automata.poner_obstaculo(fila, col)

def api_poner_presion(automata, fila, col, nivel):
    """Asigna presión base 0-4 a una celda."""
    automata.poner_presion(fila, col, nivel)

def api_borrar_celda(automata, fila, col):
    """Borra la celda, vuelve a vacía."""
    automata.borrar_celda(fila, col)

def api_paso(automata):
    """Avanza una generación. Retorna el número de generación actual."""
    automata.paso()
    return automata.generacion

def api_estado_celda(automata, fila, col):
    """
    Retorna el dict de la celda en (fila, col).
    Campos útiles para dibujar:
        celda['tipo']       → 'vacia' | 'aire' | 'obstaculo'
        celda['direccion']  → 'N' | 'S' | 'E' | 'O' | None
        celda['intensidad'] → 1 | 2 | 3 | None
        celda['presion']    → 0..4
    """
    return automata.get_celda(fila, col)

def api_tablero_completo(automata):
    """
    Retorna lista[fila][col] con todos los dicts del tablero.
    Útil para que la GUI redibuje todo de una vez.
    """
    return [[automata.get_celda(f, c) for c in range(automata.cols)]
            for f in range(automata.filas)]

# ══════════════════════════════════════════════════════
#  HELPERS DEL MENÚ DE TEXTO
# ══════════════════════════════════════════════════════

def _pedir_int(msg, minimo=None, maximo=None):
    while True:
        try:
            v = int(input(msg))
            if minimo is not None and v < minimo:
                print(f"  ✗ Mínimo {minimo}")
                continue
            if maximo is not None and v > maximo:
                print(f"  ✗ Máximo {maximo}")
                continue
            return v
        except ValueError:
            print("  ✗ Ingresa un número.")

def _pedir_coords(automata):
    f = _pedir_int(f"  Fila    (0-{automata.filas-1}): ", 0, automata.filas-1)
    c = _pedir_int(f"  Columna (0-{automata.cols-1}):  ", 0, automata.cols-1)
    return f, c

def _pedir_direccion():
    while True:
        d = input("  Dirección (N/S/E/O): ").strip().upper()
        if d in _DIR:
            return _DIR[d]
        print("  ✗ Usa N, S, E u O.")

def _pedir_intensidad():
    while True:
        i = input("  Intensidad (1=Débil / 2=Normal / 3=Fuerte): ").strip()
        if i in _INT:
            return _INT[i]
        print("  ✗ Usa 1, 2 o 3.")

# ══════════════════════════════════════════════════════
#  MENÚ CONFIGURACIÓN
# ══════════════════════════════════════════════════════

def _menu_configuracion(automata):
    while True:
        print(f"""
──────────────────────────────────
  CONFIGURACIÓN  ({automata.filas}×{automata.cols})
──────────────────────────────────
  1. Agregar corriente de aire
  2. Agregar obstáculo
  3. Asignar presión a celda
  4. Borrar celda
  5. Ver tablero
  6. Limpiar todo
  0. Volver
──────────────────────────────────""")
        op = input("  Opción: ").strip()

        if op == '1':
            f, c = _pedir_coords(automata)
            d = _pedir_direccion()
            i = _pedir_intensidad()
            automata.poner_aire(f, c, d, i)
            print(f"  ✓ Corriente en ({f},{c})")

        elif op == '2':
            f, c = _pedir_coords(automata)
            automata.poner_obstaculo(f, c)
            print(f"  ✓ Obstáculo en ({f},{c})")

        elif op == '3':
            f, c = _pedir_coords(automata)
            n = _pedir_int("  Presión (0-4): ", 0, 4)
            automata.poner_presion(f, c, n)
            print(f"  ✓ Presión {n} en ({f},{c})")

        elif op == '4':
            f, c = _pedir_coords(automata)
            automata.borrar_celda(f, c)
            print(f"  ✓ Celda ({f},{c}) borrada")

        elif op == '5':
            print(automata)

        elif op == '6':
            automata.inicializar_vacio()
            print("  ✓ Tablero limpio")

        elif op == '0':
            break

# ══════════════════════════════════════════════════════
#  MENÚ SIMULACIÓN
# ══════════════════════════════════════════════════════

def _menu_simulacion(automata):
    while True:
        print(f"""
──────────────────────────────────
  SIMULACIÓN — Generación {automata.generacion}
  Corrientes: {automata.contar_corrientes()}
──────────────────────────────────
  1. Avanzar 1 paso
  2. Avanzar N pasos
  3. Ver tablero
  4. Ver celda específica
  0. Volver
──────────────────────────────────""")
        op = input("  Opción: ").strip()

        if op == '1':
            automata.paso()
            print(f"  ✓ Gen {automata.generacion} — Corrientes: {automata.contar_corrientes()}")
            if automata.sin_corrientes():
                print("  ⚠ Sin corrientes. Simulación terminada.")
                break

        elif op == '2':
            n = _pedir_int("  ¿Cuántos pasos? ", 1)
            for _ in range(n):
                automata.paso()
                if automata.sin_corrientes():
                    break
            print(f"  ✓ Gen {automata.generacion} — Corrientes: {automata.contar_corrientes()}")

        elif op == '3':
            print(automata)

        elif op == '4':
            f, c = _pedir_coords(automata)
            celda = automata.get_celda(f, c)
            print(f"\n  Celda ({f},{c}):")
            for k, v in celda.items():
                print(f"    {k}: {v}")

        elif op == '0':
            break

# ══════════════════════════════════════════════════════
#  MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════

def main():
    print("══════════════════════════════════════")
    print("   AUTÓMATA CELULAR — CORRIENTES DE AIRE")
    print("══════════════════════════════════════")

    r = input(f"\n¿Usar tamaño default ({FILAS}×{COLS})? (s/n): ").strip().lower()
    if r == 'n':
        filas = _pedir_int("  Filas  (5-100): ", 5, 100)
        cols  = _pedir_int("  Cols   (5-100): ", 5, 100)
    else:
        filas, cols = FILAS, COLS

    automata = AutomataCelular(filas, cols)
    print(f"  ✓ Tablero {filas}×{cols} creado.")

    while True:
        print("""
══════════════════════════════════════
  MENÚ PRINCIPAL
══════════════════════════════════════
  1. Configurar tablero
  2. Iniciar / continuar simulación
  3. Reiniciar
  0. Salir
──────────────────────────────────────""")
        op = input("  Opción: ").strip()

        if op == '1':
            _menu_configuracion(automata)
        elif op == '2':
            if automata.contar_corrientes() == 0:
                print("  ⚠ Agrega al menos una corriente primero.")
            else:
                _menu_simulacion(automata)
        elif op == '3':
            automata.inicializar_vacio()
            print("  ✓ Reiniciado.")
        elif op == '0':
            print("\n  Hasta luego.\n")
            break

if __name__ == '__main__':
    main()

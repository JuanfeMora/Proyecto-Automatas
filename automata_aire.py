# ══════════════════════════════════════════════════════
#  automata.py — clase principal del autómata de aire
# ══════════════════════════════════════════════════════

import copy
from constantes_aire import (
    VACIA, AIRE, OBSTACULO,
    DEBIL, NORMAL, FUERTE,
    NORTE, SUR, ESTE, OESTE,
    DIRS, OPUESTO, VECINDAD,
    FILAS, COLS
)
from celda_aire import crear_celda_vacia, crear_celda_aire, crear_obstaculo


class AutomataCelular:
    """
    Autómata Celular — Corrientes de aire con obstáculos.

    Quíntupla formal:
        AC = (Q, V, delta, s0, F)

        Q     = {vacia, aire, obstaculo} con atributos
        V     = vecindad de Von Neumann (4 vecinos cardinales)
        delta = reglas de colisión de aire (delta.py)
        s0    = tablero configurado por el usuario
        F     = condición de parada (opcional)
    """

    def __init__(self, filas=FILAS, cols=COLS):
        self.filas      = filas
        self.cols       = cols
        self.generacion = 0

        # s0: todas las celdas vacías al inicio
        self.tablero = self._crear_tablero_vacio()

    # ══════════════════════════════════════════════════
    #  CREACIÓN DEL TABLERO (s0)
    # ══════════════════════════════════════════════════

    def _crear_tablero_vacio(self):
        """
        Crea una matriz de filas × cols donde cada celda
        es una celda vacía con presión 0.
        """
        return [
            [crear_celda_vacia() for _ in range(self.cols)]
            for _ in range(self.filas)
        ]

    def inicializar_vacio(self):
        """
        Resetea el tablero completo a vacío.
        Se usa cuando el usuario quiere empezar de cero.
        """
        self.tablero    = self._crear_tablero_vacio()
        self.generacion = 0

    # ══════════════════════════════════════════════════
    #  CONFIGURACIÓN MANUAL (el usuario coloca celdas)
    # ══════════════════════════════════════════════════

    def poner_obstaculo(self, fila, col):
        """
        Coloca un obstáculo en la posición (fila, col).
        Los obstáculos bloquean el paso de las corrientes.
        """
        if self._dentro_del_tablero(fila, col):
            self.tablero[fila][col] = crear_obstaculo()

    def poner_aire(self, fila, col, direccion, intensidad):
        """
        Coloca una corriente de aire en (fila, col)
        con la dirección e intensidad indicadas.

        direccion : NORTE | SUR | ESTE | OESTE
        intensidad: DEBIL | NORMAL | FUERTE
        """
        if self._dentro_del_tablero(fila, col):
            # conservar la presión que ya tenía esa celda
            presion_actual = self.tablero[fila][col]['presion']
            presion_base   = self.tablero[fila][col]['presion_base']
            self.tablero[fila][col] = crear_celda_aire(
                direccion  = direccion,
                intensidad = intensidad,
                presion    = presion_actual,
                presion_base = presion_base
            )

    def poner_presion(self, fila, col, nivel):
        """
        Asigna un nivel de presión base a una celda vacía.
        nivel debe estar entre 0 y 4.

        Esta presión es fija — la coloca el usuario y
        es el mínimo al que puede bajar esa celda.
        """
        nivel = max(0, min(4, nivel))  # asegurar que esté en rango 0..4
        if self._dentro_del_tablero(fila, col):
            self.tablero[fila][col]['presion']      = nivel
            self.tablero[fila][col]['presion_base'] = nivel

    def borrar_celda(self, fila, col):
        """
        Borra una celda — la devuelve a estado vacío
        conservando su presión base original.
        """
        if self._dentro_del_tablero(fila, col):
            presion_base = self.tablero[fila][col]['presion_base']
            self.tablero[fila][col] = crear_celda_vacia(presion_base)

    # ══════════════════════════════════════════════════
    #  LECTURA DEL TABLERO
    # ══════════════════════════════════════════════════

    def _dentro_del_tablero(self, fila, col):
        """
        Verifica si una posición existe dentro del tablero.
        Usado internamente para evitar errores de índice.
        """
        return 0 <= fila < self.filas and 0 <= col < self.cols

    def get_celda(self, fila, col):
        """
        Retorna la celda en (fila, col).

        Si la posición está fuera del tablero, retorna un
        obstáculo — los bordes del mundo son paredes sólidas.
        Esto evita que las corrientes se salgan del tablero.
        """
        if self._dentro_del_tablero(fila, col):
            return self.tablero[fila][col]
        return crear_obstaculo()

    def get_vecinos(self, fila, col):
        """
        Retorna un diccionario con las 4 celdas vecinas
        en las direcciones cardinales.

        Ejemplo de resultado:
        {
            'N': celda_de_arriba,
            'S': celda_de_abajo,
            'E': celda_de_la_derecha,
            'O': celda_de_la_izquierda
        }

        Usa get_celda para manejar bordes automáticamente —
        si un vecino está fuera del tablero, aparece como obstáculo.
        """
        vecinos = {}
        for direccion, (df, dc) in VECINDAD.items():
            vecinos[direccion] = self.get_celda(fila + df, col + dc)
        return vecinos

    # ══════════════════════════════════════════════════
    #  AVANCE DE LA SIMULACIÓN
    # ══════════════════════════════════════════════════

    def paso(self):
        """
        Avanza el autómata una generación completa: t → t+1.

        Proceso:
            1. Crear un tablero nuevo vacío (nunca modificar
               el actual mientras se está leyendo)
            2. Para cada celda, leer su estado y el de sus
               vecinos del tablero actual (t)
            3. Aplicar delta para calcular el nuevo estado (t+1)
            4. Escribir el resultado en el tablero nuevo
            5. Reemplazar el tablero viejo con el nuevo

        La simultaneidad es crítica: todas las celdas deben
        ver el estado t al mismo tiempo, nunca resultados
        parciales de t+1.
        """
        # importar delta aquí para evitar importación circular
        from delta_aire import delta

        # paso 1: tablero nuevo donde escribiremos t+1
        nuevo_tablero = self._crear_tablero_vacio()

        # paso 2 y 3: recorrer cada celda
        for fila in range(self.filas):
            for col in range(self.cols):

                celda_actual = self.tablero[fila][col]
                vecinos      = self.get_vecinos(fila, col)

                # paso 4: aplicar delta y guardar resultado
                nuevo_tablero[fila][col] = delta(celda_actual, vecinos)

        # paso 5: reemplazar tablero y avanzar generación
        self.tablero    = nuevo_tablero
        self.generacion += 1

    # ══════════════════════════════════════════════════
    #  CONSULTAS
    # ══════════════════════════════════════════════════

    def contar_corrientes(self):
        """Retorna cuántas celdas de tipo AIRE hay en el tablero."""
        return sum(
            1
            for fila in range(self.filas)
            for col  in range(self.cols)
            if self.tablero[fila][col]['tipo'] == AIRE
        )

    def contar_obstaculos(self):
        """Retorna cuántas celdas de tipo OBSTÁCULO hay."""
        return sum(
            1
            for fila in range(self.filas)
            for col  in range(self.cols)
            if self.tablero[fila][col]['tipo'] == OBSTACULO
        )

    def sin_corrientes(self):
        """
        Retorna True si no queda ninguna corriente de aire.
        Se puede usar como condición de parada F.
        """
        return self.contar_corrientes() == 0

    def __str__(self):
        """
        Representación en texto del tablero — útil para debug.
        Muestra cada celda como:
            ░  vacía
            →↑↓←  corriente con su dirección
            █  obstáculo
        """
        simbolos_dir = {
            NORTE: '↑', SUR: '↓',
            ESTE:  '→', OESTE: '←'
        }
        filas_str = []
        for fila in self.tablero:
            fila_str = []
            for celda in fila:
                if celda['tipo'] == OBSTACULO:
                    fila_str.append('█')
                elif celda['tipo'] == AIRE:
                    fila_str.append(simbolos_dir[celda['direccion']])
                else:
                    fila_str.append('░')
            filas_str.append(' '.join(fila_str))
        return f"Generación {self.generacion}:\n" + '\n'.join(filas_str)

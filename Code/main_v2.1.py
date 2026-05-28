import pygame
import sys
import numpy as np

# --- CONSTANTES DE LA INTERFAZ Y MODELO ---
CELL_SIZE = 15
GRID_W = 50
GRID_H = 50
WIDTH = GRID_W * CELL_SIZE
HEIGHT = GRID_H * CELL_SIZE

# Canales del tensor / Direcciones
DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3

# Tipos de material (reemplaza el booleano barriers)
MAT_AIR      = 0   # Aire libre
MAT_WALL     = 1   # Muro sólido – rebote perfecto
MAT_FOLIAGE  = 2   # Follaje / Permeable – resta 1 nivel de intensidad pero permite el paso
MAT_SINK     = 3   # Sumidero / Absorbedor – aniquila viento y presión al contacto

DIR_NAMES = {DIR_UP: "ARRIBA", DIR_DOWN: "ABAJO", DIR_LEFT: "IZQUIERDA", DIR_RIGHT: "DERECHA"}
INTENSITY_NAMES = {1: "DÉBIL (1)", 2: "NORMAL (2)", 3: "FUERTE (3)"}

# ==========================================
# 1. EL MODELO FÍSICO (BACKEND V2)
# ==========================================
class FluidModel:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # --- Tensor principal de viento (sin cambios en forma) ---
        self.grid = np.zeros((width, height, 4), dtype=np.int8)

        # MEJORA 1 ─ Materiales (int8 reemplaza bool)
        self.materials = np.zeros((width, height), dtype=np.int8)

        # MEJORA 2 ─ Presión barométrica estática
        self.static_pressure = np.zeros((width, height), dtype=np.int8)

        # MEJORA 3 ─ Temperatura escalar (float para difusión precisa)
        self.temperature = np.zeros((width, height), dtype=np.float32)

        # MEJORA 4 ─ Trazadores (partículas cinemáticas)
        self.tracers = np.zeros((width, height), dtype=np.int8)

        self._init_boundaries()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def _init_boundaries(self):
        """Bordea el grid con muros sólidos (MAT_WALL)."""
        self.materials[0, :]  = MAT_WALL
        self.materials[-1, :] = MAT_WALL
        self.materials[:, 0]  = MAT_WALL
        self.materials[:, -1] = MAT_WALL

    # ------------------------------------------------------------------
    # Setters públicos (compatibles con el frontend existente)
    # ------------------------------------------------------------------
    def set_wind(self, x: int, y: int, direction: int, intensity: int):
        if 0 <= x < self.width and 0 <= y < self.height and self.materials[x, y] == MAT_AIR:
            self.grid[x, y, :] = 0
            self.grid[x, y, direction] = intensity

    def set_barrier(self, x: int, y: int, mat_type: int = MAT_WALL):
        """Coloca un material en (x, y). Por defecto muro sólido."""
        if 0 < x < self.width - 1 and 0 < y < self.height - 1:
            self.materials[x, y] = mat_type
            self.grid[x, y, :] = 0
            self.tracers[x, y] = 0

    def clear_cell(self, x: int, y: int):
        """Borra pared/corriente con clic central."""
        if 0 < x < self.width - 1 and 0 < y < self.height - 1:
            self.materials[x, y] = MAT_AIR
            self.grid[x, y, :] = 0
            self.static_pressure[x, y] = 0

    def set_pressure(self, x: int, y: int, value: int):
        """Establece presión barométrica estática en una celda."""
        if 0 <= x < self.width and 0 <= y < self.height and self.materials[x, y] == MAT_AIR:
            self.static_pressure[x, y] = np.clip(value, -10, 10)

    def set_temperature(self, x: int, y: int, value: float):
        """Establece temperatura en una celda."""
        if 0 <= x < self.width and 0 <= y < self.height and self.materials[x, y] == MAT_AIR:
            self.temperature[x, y] = np.clip(value, -100.0, 100.0)

    def add_tracer(self, x: int, y: int):
        """Coloca una partícula trazadora."""
        if 0 <= x < self.width and 0 <= y < self.height and self.materials[x, y] == MAT_AIR:
            self.tracers[x, y] = 1

    def reset_fluid(self):
        """Limpia todo el fluido conservando materiales y presión estática."""
        self.grid.fill(0)
        self.temperature.fill(0)
        self.tracers.fill(0)

    def get_pressure_map(self) -> np.ndarray:
        return np.sum(self.grid, axis=2)

    # ------------------------------------------------------------------
    # Paso de simulación principal
    # ------------------------------------------------------------------
    def step(self):
        self._phase_a_streaming()
        self._phase_b_boundaries()
        self._phase_c_collisions()
        self._phase_d_friction()
        self._phase_e_pressure_gradient()   # MEJORA 2
        self._phase_f_temperature()         # MEJORA 3
        self._phase_g_tracers()             # MEJORA 4

    # ------------------------------------------------------------------
    # FASE A – Streaming (sin cambios)
    # ------------------------------------------------------------------
    def _phase_a_streaming(self):
        self.grid[:, :, DIR_UP]    = np.roll(self.grid[:, :, DIR_UP],    shift=-1, axis=1)
        self.grid[:, :, DIR_DOWN]  = np.roll(self.grid[:, :, DIR_DOWN],  shift=1,  axis=1)
        self.grid[:, :, DIR_LEFT]  = np.roll(self.grid[:, :, DIR_LEFT],  shift=-1, axis=0)
        self.grid[:, :, DIR_RIGHT] = np.roll(self.grid[:, :, DIR_RIGHT], shift=1,  axis=0)

    # ------------------------------------------------------------------
    # FASE B – Contornos (MEJORA 1: usa self.materials en lugar de self.barriers)
    # ------------------------------------------------------------------
    def _phase_b_boundaries(self):
        """
        MAT_WALL   → rebote perfecto (intercambia canales opuestos).
        MAT_FOLIAGE → permeable: el viento pasa pero pierde 1 nivel de intensidad.
        MAT_SINK    → absorbedor: aniquila todo viento y presión al contacto.
        """
        # --- Muro sólido: rebote perfecto ---
        wall_mask = (self.materials == MAT_WALL)
        temp_y = self.grid[wall_mask, DIR_UP].copy()
        self.grid[wall_mask, DIR_UP]   = self.grid[wall_mask, DIR_DOWN]
        self.grid[wall_mask, DIR_DOWN] = temp_y

        temp_x = self.grid[wall_mask, DIR_LEFT].copy()
        self.grid[wall_mask, DIR_LEFT]  = self.grid[wall_mask, DIR_RIGHT]
        self.grid[wall_mask, DIR_RIGHT] = temp_x

        # --- Follaje: reduce intensidad en 1 (mínimo 0) ---
        foliage_mask = (self.materials == MAT_FOLIAGE)
        self.grid[foliage_mask] = np.clip(self.grid[foliage_mask].astype(np.int16) - 1, 0, 3).astype(np.int8)

        # --- Sumidero: aniquila todo ---
        sink_mask = (self.materials == MAT_SINK)
        self.grid[sink_mask] = 0
        self.static_pressure[sink_mask] = 0
        self.temperature[sink_mask] = 0.0

    # ------------------------------------------------------------------
    # FASE C – Colisiones (sin cambios funcionales)
    # ------------------------------------------------------------------
    def _phase_c_collisions(self):
        air = (self.materials == MAT_AIR)
        up, down, left, right = (
            self.grid[:, :, 0], self.grid[:, :, 1],
            self.grid[:, :, 2], self.grid[:, :, 3],
        )

        eq_y = air & (up == down) & (up > 0)
        eq_x = air & (left == right) & (left > 0)
        diff_y_up    = air & (up > down)   & (down > 0)
        diff_y_down  = air & (down > up)   & (up > 0)
        diff_x_left  = air & (left > right) & (right > 0)
        diff_x_right = air & (right > left) & (left > 0)

        delta_up    = np.zeros_like(up)
        delta_down  = np.zeros_like(down)
        delta_left  = np.zeros_like(left)
        delta_right = np.zeros_like(right)

        delta_up[eq_y]   -= up[eq_y];   delta_down[eq_y]  -= down[eq_y]
        delta_left[eq_y] += up[eq_y];   delta_right[eq_y] += up[eq_y]

        delta_left[eq_x]  -= left[eq_x]; delta_right[eq_x] -= right[eq_x]
        delta_up[eq_x]    += left[eq_x]; delta_down[eq_x]  += left[eq_x]

        delta_up[diff_y_up]    -= 1;  delta_down[diff_y_up]   -= down[diff_y_up]
        delta_down[diff_y_down] -= 1; delta_up[diff_y_down]   -= up[diff_y_down]

        delta_left[diff_x_left]   -= 1; delta_right[diff_x_left]  -= right[diff_x_left]
        delta_right[diff_x_right] -= 1; delta_left[diff_x_right]  -= left[diff_x_right]

        self.grid[:, :, 0] = np.clip(up    + delta_up,    0, 3)
        self.grid[:, :, 1] = np.clip(down  + delta_down,  0, 3)
        self.grid[:, :, 2] = np.clip(left  + delta_left,  0, 3)
        self.grid[:, :, 3] = np.clip(right + delta_right, 0, 3)

    # ------------------------------------------------------------------
    # FASE D – Fricción / Entropía (sin cambios)
    # ------------------------------------------------------------------
    def _phase_d_friction(self):
        friction_mask    = np.random.random(self.grid.shape) < 0.02
        wind_active_mask = self.grid > 0
        self.grid[friction_mask & wind_active_mask] -= 1

    # ------------------------------------------------------------------
    # FASE E – Gradiente de Presión Barométrica (MEJORA 2)
    #   Vt+1 = Vt − c·∇P   (c = 1, discreta)
    #   Si ΔP ≥ 2 entre vecinos, se inyecta un vector de viento
    #   en dirección opuesta al gradiente (de alta → baja presión).
    # ------------------------------------------------------------------
    def _phase_e_pressure_gradient(self):
        air = (self.materials == MAT_AIR)
        P   = self.static_pressure.astype(np.int16)

        # Gradientes hacia cada dirección (diferencias finitas hacia adelante)
        # dP/dx  ≈ P[x+1,y] - P[x,y]   →  viento hacia la izquierda si positivo
        # dP/dy  ≈ P[x,y+1] - P[x,y]   →  viento hacia arriba si positivo

        # Diferencia en X: presión derecha − izquierda
        dP_x = np.zeros_like(P)
        dP_x[:-1, :] = P[1:, :] - P[:-1, :]   # ∂P/∂x

        # Diferencia en Y: presión abajo − arriba
        dP_y = np.zeros_like(P)
        dP_y[:, :-1] = P[:, 1:] - P[:, :-1]   # ∂P/∂y

        THRESHOLD = 2  # Solo actúa si ΔP ≥ 2

        # Gradiente positivo en X → alta presión a la derecha → viento va a la IZQUIERDA
        mask = air & (dP_x >= THRESHOLD)
        self.grid[:, :, DIR_LEFT][mask] = np.clip(
            self.grid[:, :, DIR_LEFT][mask].astype(np.int16) + 1, 0, 3
        ).astype(np.int8)

        # Gradiente negativo en X → alta presión a la izquierda → viento va a la DERECHA
        mask = air & (dP_x <= -THRESHOLD)
        self.grid[:, :, DIR_RIGHT][mask] = np.clip(
            self.grid[:, :, DIR_RIGHT][mask].astype(np.int16) + 1, 0, 3
        ).astype(np.int8)

        # Gradiente positivo en Y → alta presión abajo → viento va ARRIBA
        mask = air & (dP_y >= THRESHOLD)
        self.grid[:, :, DIR_UP][mask] = np.clip(
            self.grid[:, :, DIR_UP][mask].astype(np.int16) + 1, 0, 3
        ).astype(np.int8)

        # Gradiente negativo en Y → alta presión arriba → viento va ABAJO
        mask = air & (dP_y <= -THRESHOLD)
        self.grid[:, :, DIR_DOWN][mask] = np.clip(
            self.grid[:, :, DIR_DOWN][mask].astype(np.int16) + 1, 0, 3
        ).astype(np.int8)

    # ------------------------------------------------------------------
    # FASE F – Termodinámica / Advección de Temperatura (MEJORA 3)
    #   Tt+1 = Tt − ∇·(V·Tt) + α·∇²Tt
    #   Implementación discreta y vectorizada:
    #     · Advección: cada canal de viento "arrastra" temperatura 1 celda.
    #     · Difusión:  Laplaciano de 5 puntos con α = 0.05.
    # ------------------------------------------------------------------
    def _phase_f_temperature(self):
        T   = self.temperature.astype(np.float32)
        air = (self.materials == MAT_AIR)

        # --- Advección (el viento transporta calor) ---
        # Cada canal indica la dirección; la temperatura de la celda origen
        # se desplaza en esa dirección ponderada por la intensidad normalizada.
        weight = 0.25   # contribución por canal (4 canales → suma 1.0 en max)

        T_adv = T.copy()
        for ch, (shift, axis) in enumerate([(-1, 1), (1, 1), (-1, 0), (1, 0)]):
            intensity = self.grid[:, :, ch].astype(np.float32) / 3.0  # normaliza 0..1
            flux = np.roll(T * intensity, shift=shift, axis=axis) * weight
            T_adv += flux
            T_adv -= T * intensity * weight

        # --- Difusión (Laplaciano discreta 5 puntos, α = 0.05) ---
        alpha = 0.05
        lap = (
            np.roll(T, -1, axis=0) + np.roll(T, 1, axis=0) +
            np.roll(T, -1, axis=1) + np.roll(T, 1, axis=1) -
            4.0 * T
        )
        T_new = T_adv + alpha * lap

        # Solo actualizar celdas de aire; muros/sumideros retienen su temp
        self.temperature[air] = np.clip(T_new[air], -100.0, 100.0)

    # ------------------------------------------------------------------
    # FASE G – Trazadores Cinemáticos (MEJORA 4)
    #   Cada partícula consulta el canal de viento dominante en (x, y)
    #   y se desplaza 1 celda en esa dirección con np.roll.
    # ------------------------------------------------------------------
    def _phase_g_tracers(self):
        t = self.tracers
        if not np.any(t): 
            return

        # 1. Determinar dónde hay viento para mover los trazadores
        wind_sum = np.sum(self.grid, axis=2)
        has_wind = wind_sum > 0

        # 2. Obtener el canal dominante (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) en toda la matriz simultáneamente
        dom = np.argmax(self.grid, axis=2)

        # 3. Crear máscaras lógicas para cada movimiento
        move_up    = (t == 1) & has_wind & (dom == DIR_UP)
        move_down  = (t == 1) & has_wind & (dom == DIR_DOWN)
        move_left  = (t == 1) & has_wind & (dom == DIR_LEFT)
        move_right = (t == 1) & has_wind & (dom == DIR_RIGHT)
        stay       = (t == 1) & (~has_wind) # Se quedan quietos si no hay viento
        
        # 4. Desplazar toda la matriz de trazadores simultáneamente usando OR a nivel de bits
        new_t = np.zeros_like(t)
        new_t |= stay
        new_t |= np.roll(move_up, shift=-1, axis=1)
        new_t |= np.roll(move_down, shift=1, axis=1)
        new_t |= np.roll(move_left, shift=-1, axis=0)
        new_t |= np.roll(move_right, shift=1, axis=0)

        # 5. Los trazadores que caen en materiales sólidos se destruyen
        new_t[self.materials != MAT_AIR] = 0
        self.tracers = new_t


# ==========================================
# 2. EL CONTROLADOR Y VISTA (FRONTEND)
#    Sin cambios respecto a V1 – compatible con el nuevo backend
# ==========================================
def get_bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return points


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LBM-D2Q4 V2 | Simulador de Fluidos")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 24)

    model = FluidModel(GRID_W, GRID_H)

    simulating        = False
    current_direction = DIR_RIGHT
    current_intensity = 3
    last_gx, last_gy  = None, None

    running = True
    while running:
        # --- EVENTOS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulating = not simulating
                if not simulating:
                    if event.key == pygame.K_UP:    current_direction = DIR_UP
                    elif event.key == pygame.K_DOWN:  current_direction = DIR_DOWN
                    elif event.key == pygame.K_LEFT:  current_direction = DIR_LEFT
                    elif event.key == pygame.K_RIGHT: current_direction = DIR_RIGHT
                    elif event.key == pygame.K_1: current_intensity = 1
                    elif event.key == pygame.K_2: current_intensity = 2
                    elif event.key == pygame.K_3: current_intensity = 3
                    elif event.key == pygame.K_r: model.reset_fluid()
                    elif event.key == pygame.K_n: model.step()
                    # Teclas nuevas para probar materiales
                    elif event.key == pygame.K_f:   # Follaje
                        pass  # El frontend puede mapear esto al pincel de follaje
                    elif event.key == pygame.K_s:   # Sumidero
                        pass

        # --- MOUSE ---
        if not simulating:
            mouse_buttons = pygame.mouse.get_pressed()
            if any(mouse_buttons):
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                if last_gx is not None:
                    points_to_draw = get_bresenham_line(last_gx, last_gy, gx, gy)
                else:
                    points_to_draw = [(gx, gy)]
                for px, py in points_to_draw:
                    if 0 <= px < GRID_W and 0 <= py < GRID_H:
                        if mouse_buttons[0]:   model.set_barrier(px, py, MAT_WALL)
                        elif mouse_buttons[2]: model.set_wind(px, py, current_direction, current_intensity)
                        elif mouse_buttons[1]: model.clear_cell(px, py)
                last_gx, last_gy = gx, gy
            else:
                last_gx, last_gy = None, None

        # --- FÍSICA ---
        if simulating:
            model.step()

        # --- RENDERIZADO ---
        screen.fill((10, 10, 15))

        # 1. Dibujar materiales
        for mat_type, color in [
            (MAT_WALL,    (80, 80, 80)),
            (MAT_FOLIAGE, (30, 120, 30)),
            (MAT_SINK,    (120, 20, 20)),
        ]:
            xs, ys = np.nonzero(model.materials == mat_type)
            for x, y in zip(xs, ys):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

        # 2. Fluido (mapa de presión dinámica)
        pressure_map = model.get_pressure_map()
        px_arr, py_arr = np.nonzero(pressure_map > 0)
        for x, y in zip(px_arr, py_arr):
            if model.materials[x, y] == MAT_AIR:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color_val = min(255, 50 + int(pressure_map[x, y]) * 40)
                pygame.draw.rect(screen, (0, int(color_val * 0.8), color_val), rect)

        # 3. Rejilla
        for x in range(GRID_W + 1):
            pygame.draw.line(screen, (30, 30, 30), (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT))
        for y in range(GRID_H + 1):
            pygame.draw.line(screen, (30, 30, 30), (0, y * CELL_SIZE), (WIDTH, y * CELL_SIZE))

        # 4. Flechas de viento
        for channel in range(4):
            xs, ys = np.nonzero(model.grid[:, :, channel] > 0)
            for x, y in zip(xs, ys):
                intensity = model.grid[x, y, channel]
                cx = x * CELL_SIZE + CELL_SIZE // 2
                cy = y * CELL_SIZE + CELL_SIZE // 2
                dx_a, dy_a = 0, 0
                if channel == DIR_UP:    dy_a = -CELL_SIZE // 2
                elif channel == DIR_DOWN:  dy_a =  CELL_SIZE // 2
                elif channel == DIR_LEFT:  dx_a = -CELL_SIZE // 2
                elif channel == DIR_RIGHT: dx_a =  CELL_SIZE // 2
                pygame.draw.line(screen, (0, 255, 200), (cx, cy), (cx + dx_a, cy + dy_a), width=intensity)

        # 5. Trazadores (puntos blancos brillantes) — MEJORA 4
        tx_arr, ty_arr = np.nonzero(model.tracers == 1)
        for x, y in zip(tx_arr, ty_arr):
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, (255, 255, 100), (cx, cy), max(2, CELL_SIZE // 4))

        # 6. HUD
        hud_bg = pygame.Surface((540, 125))
        hud_bg.set_alpha(180)
        hud_bg.fill((0, 0, 0))
        screen.blit(hud_bg, (10, 10))

        state_text    = "SIMULANDO" if simulating else "EDICIÓN (Pausado)"
        brush_text    = f"Pincel: {DIR_NAMES[current_direction]} | Int: {INTENSITY_NAMES[current_intensity]}"
        controls1     = "ESPACIO: Play/Pausa | R: Limpiar Aire | N: Paso a Paso"
        controls2     = "Clic Izq: Pared | Clic Der: Aire | Clic Central: Borrar"
        materials_txt = "Materiales: MAT_WALL=gris | MAT_FOLIAGE=verde | MAT_SINK=rojo"

        screen.blit(font.render(f"Estado: {state_text}", True, (255, 255, 255)), (20, 20))
        screen.blit(font.render(brush_text, True, (255, 255, 255)), (20, 45))
        screen.blit(font.render(controls1, True, (200, 200, 200)), (20, 70))
        screen.blit(font.render(controls2, True, (200, 200, 200)), (20, 90))
        screen.blit(font.render(materials_txt, True, (150, 200, 150)), (20, 110))

        pygame.display.flip()
        clock.tick(15 if simulating else 60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

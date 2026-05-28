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

DIR_NAMES = {DIR_UP: "ARRIBA", DIR_DOWN: "ABAJO", DIR_LEFT: "IZQUIERDA", DIR_RIGHT: "DERECHA"}
INTENSITY_NAMES = {1: "DÉBIL (1)", 2: "NORMAL (2)", 3: "FUERTE (3)"}

# ==========================================
# 1. EL MODELO FÍSICO (BACKEND)
# ==========================================
class FluidModel:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height, 4), dtype=np.int8)
        self.barriers = np.zeros((width, height), dtype=bool)
        self._init_boundaries()

    def _init_boundaries(self):
        self.barriers[0, :] = True
        self.barriers[-1, :] = True
        self.barriers[:, 0] = True
        self.barriers[:, -1] = True

    def set_wind(self, x: int, y: int, direction: int, intensity: int):
        if 0 <= x < self.width and 0 <= y < self.height and not self.barriers[x, y]:
            self.grid[x, y, :] = 0 # Limpia la celda antes de poner el viento nuevo
            self.grid[x, y, direction] = intensity

    def set_barrier(self, x: int, y: int):
        if 0 < x < self.width-1 and 0 < y < self.height-1:
            self.barriers[x, y] = True
            self.grid[x, y, :] = 0
            
    def clear_cell(self, x: int, y: int):
        """Permite borrar una pared o corriente con el clic central"""
        if 0 < x < self.width-1 and 0 < y < self.height-1:
            self.barriers[x, y] = False
            self.grid[x, y, :] = 0

    def get_pressure_map(self) -> np.ndarray:
        return np.sum(self.grid, axis=2)

    def step(self):
        self._phase_a_streaming()
        self._phase_b_boundaries()
        self._phase_c_collisions()

    def _phase_a_streaming(self):
        self.grid[:, :, DIR_UP] = np.roll(self.grid[:, :, DIR_UP], shift=-1, axis=1)
        self.grid[:, :, DIR_DOWN] = np.roll(self.grid[:, :, DIR_DOWN], shift=1, axis=1)
        self.grid[:, :, DIR_LEFT] = np.roll(self.grid[:, :, DIR_LEFT], shift=-1, axis=0)
        self.grid[:, :, DIR_RIGHT] = np.roll(self.grid[:, :, DIR_RIGHT], shift=1, axis=0)

    def _phase_b_boundaries(self):
        b = self.barriers
        temp_y = self.grid[b, 0].copy()
        self.grid[b, 0] = self.grid[b, 1]
        self.grid[b, 1] = temp_y
        
        temp_x = self.grid[b, 2].copy()
        self.grid[b, 2] = self.grid[b, 3]
        self.grid[b, 3] = temp_x

    def _phase_c_collisions(self):
        air = ~self.barriers
        up, down, left, right = self.grid[:, :, 0], self.grid[:, :, 1], self.grid[:, :, 2], self.grid[:, :, 3]
        
        eq_y = air & (up == down) & (up > 0)
        eq_x = air & (left == right) & (left > 0)
        diff_y_up   = air & (up > down) & (down > 0)
        diff_y_down = air & (down > up) & (up > 0)
        diff_x_left  = air & (left > right) & (right > 0)
        diff_x_right = air & (right > left) & (left > 0)

        delta_up, delta_down, delta_left, delta_right = np.zeros_like(up), np.zeros_like(down), np.zeros_like(left), np.zeros_like(right)
        
        delta_up[eq_y] -= up[eq_y]
        delta_down[eq_y] -= down[eq_y]
        delta_left[eq_y] += up[eq_y]
        delta_right[eq_y] += up[eq_y]
        
        delta_left[eq_x] -= left[eq_x]
        delta_right[eq_x] -= right[eq_x]
        delta_up[eq_x] += left[eq_x]
        delta_down[eq_x] += left[eq_x]
        
        delta_up[diff_y_up] -= 1
        delta_down[diff_y_up] -= down[diff_y_up]
        delta_down[diff_y_down] -= 1
        delta_up[diff_y_down] -= up[diff_y_down]
        
        delta_left[diff_x_left] -= 1
        delta_right[diff_x_left] -= right[diff_x_left]
        delta_right[diff_x_right] -= 1
        delta_left[diff_x_right] -= left[diff_x_right]

        self.grid[:, :, 0] = np.clip(up + delta_up, 0, 3)
        self.grid[:, :, 1] = np.clip(down + delta_down, 0, 3)
        self.grid[:, :, 2] = np.clip(left + delta_left, 0, 3)
        self.grid[:, :, 3] = np.clip(right + delta_right, 0, 3)

# ==========================================
# 2. EL CONTROLADOR Y VISTA (FRONTEND)
# ==========================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    model = FluidModel(GRID_W, GRID_H)

    simulating = False
    current_direction = DIR_RIGHT
    current_intensity = 3

    def update_window_title():
        if simulating:
            pygame.display.set_caption("LBM-D2Q4 | [MODO SIMULACIÓN] | Presiona ESPACIO para pausar")
        else:
            d_name = DIR_NAMES[current_direction]
            i_name = INTENSITY_NAMES[current_intensity]
            title = f"LBM-D2Q4 | [EDICIÓN] Pincel: {d_name} | Fza: {i_name} | (Flechas para Dir, 1-3 para Fza)"
            pygame.display.set_caption(title)

    update_window_title()
    running = True

    while running:
        # --- EVENTOS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulating = not simulating
                    update_window_title()
                
                if not simulating:
                    if event.key == pygame.K_UP: current_direction = DIR_UP
                    elif event.key == pygame.K_DOWN: current_direction = DIR_DOWN
                    elif event.key == pygame.K_LEFT: current_direction = DIR_LEFT
                    elif event.key == pygame.K_RIGHT: current_direction = DIR_RIGHT
                    elif event.key == pygame.K_1: current_intensity = 1
                    elif event.key == pygame.K_2: current_intensity = 2
                    elif event.key == pygame.K_3: current_intensity = 3
                    update_window_title()

        # --- MOUSE (PINTAR/BORRAR) ---
        if not simulating:
            mouse_buttons = pygame.mouse.get_pressed()
            # Botones: 0=Izq, 1=Centro, 2=Der
            if mouse_buttons[0] or mouse_buttons[1] or mouse_buttons[2]:  
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    if mouse_buttons[0]: # Clic Izquierdo -> Pared
                        model.set_barrier(gx, gy)
                    elif mouse_buttons[2]: # Clic Derecho -> Viento
                        model.set_wind(gx, gy, current_direction, current_intensity)
                    elif mouse_buttons[1]: # Clic Central -> Borrar celdas
                        model.clear_cell(gx, gy)

        # --- FÍSICA ---
        if simulating:
            model.step()

        # --- RENDERIZADO ---
        screen.fill((10, 10, 15))
        
        # 1. Dibujar Paredes (Optimizado con np.nonzero)
        bx, by = np.nonzero(model.barriers)
        for x, y in zip(bx, by):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (80, 80, 80), rect) # Paredes estáticas

        # 2. Dibujar Fluido basado en Mapa de Presión (Optimizado)
        pressure_map = model.get_pressure_map()
        px, py = np.nonzero(pressure_map > 0)
        for x, y in zip(px, py):
            if not model.barriers[x, y]:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # int() evita posibles desbordamientos de tipos int8 en numpy al multiplicar
                color_val = min(255, 50 + (int(pressure_map[x, y]) * 40))
                pygame.draw.rect(screen, (0, int(color_val*0.8), color_val), rect)

        # 3. Dibujar Rejilla (Optimizado trazando líneas enteras en lugar de rectángulos individuales)
        for x in range(GRID_W + 1):
            pygame.draw.line(screen, (30, 30, 30), (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT))
        for y in range(GRID_H + 1):
            pygame.draw.line(screen, (30, 30, 30), (0, y * CELL_SIZE), (WIDTH, y * CELL_SIZE))

        # 4. Dibujar flechas orientadoras (Solo en modo Edición)
        if not simulating:
            for channel in range(4):
                xs, ys = np.nonzero(model.grid[:, :, channel] > 0)
                for x, y in zip(xs, ys):
                    intensity = model.grid[x, y, channel]
                    cx = x * CELL_SIZE + CELL_SIZE // 2
                    cy = y * CELL_SIZE + CELL_SIZE // 2
                    dx, dy = 0, 0
                    if channel == DIR_UP: dy = -CELL_SIZE // 2
                    elif channel == DIR_DOWN: dy = CELL_SIZE // 2
                    elif channel == DIR_LEFT: dx = -CELL_SIZE // 2
                    elif channel == DIR_RIGHT: dx = CELL_SIZE // 2
                    pygame.draw.line(screen, (0, 255, 200), (cx, cy), (cx + dx, cy + dy), width=intensity)

        pygame.display.flip()
        
        # La física corre a 15fps para ser observable, la edición corre a 60fps para fluidez del pincel
        clock.tick(15 if simulating else 60) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

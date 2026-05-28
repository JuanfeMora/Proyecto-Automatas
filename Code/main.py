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

    def reset_fluid(self):
        """Limpia todo el fluido de forma vectorizada conservando barreras"""
        self.grid.fill(0)

    def get_pressure_map(self) -> np.ndarray:
        return np.sum(self.grid, axis=2)

    def step(self):
        self._phase_a_streaming()
        self._phase_b_boundaries()
        self._phase_c_collisions()
        self._phase_d_friction()

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
        
    def _phase_d_friction(self):
        """
        Fase D: Fricción y Entropía.
        Disipa la energía térmica vectorizadamente para evitar crecimiento infinito en sistema cerrado.
        """
        # Genera una máscara booleana con una probabilidad del 2%
        friction_mask = np.random.random(self.grid.shape) < 0.02
        
        # Solo aplicar fricción si ya existe viento en ese canal de la celda
        wind_active_mask = self.grid > 0
        
        # Restamos 1 solo donde ocurra el 2% y haya intensidad real
        self.grid[friction_mask & wind_active_mask] -= 1

# ==========================================
# 2. EL CONTROLADOR Y VISTA (FRONTEND)
# ==========================================
def get_bresenham_line(x0, y0, x1, y1):
    """Genera las coordenadas intersecadas por el algoritmo de trazado de líneas Bresenham"""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LBM-D2Q4 | Simulador de Fluidos")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont(None, 24)
    model = FluidModel(GRID_W, GRID_H)

    simulating = False
    current_direction = DIR_RIGHT
    current_intensity = 3
    
    # Seguimiento para trazo continuo del mouse
    last_gx, last_gy = None, None

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
                    if event.key == pygame.K_UP: current_direction = DIR_UP
                    elif event.key == pygame.K_DOWN: current_direction = DIR_DOWN
                    elif event.key == pygame.K_LEFT: current_direction = DIR_LEFT
                    elif event.key == pygame.K_RIGHT: current_direction = DIR_RIGHT
                    elif event.key == pygame.K_1: current_intensity = 1
                    elif event.key == pygame.K_2: current_intensity = 2
                    elif event.key == pygame.K_3: current_intensity = 3
                    elif event.key == pygame.K_r: model.reset_fluid()
                    elif event.key == pygame.K_n: model.step()

        # --- MOUSE (PINTAR/BORRAR CON TRAZO CONTINUO) ---
        if not simulating:
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0] or mouse_buttons[1] or mouse_buttons[2]:  
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                
                # Interpolamos si hay rastro anterior para evitar "huecos"
                if last_gx is not None and last_gy is not None:
                    points_to_draw = get_bresenham_line(last_gx, last_gy, gx, gy)
                else:
                    points_to_draw = [(gx, gy)]
                
                for px, py in points_to_draw:
                    if 0 <= px < GRID_W and 0 <= py < GRID_H:
                        if mouse_buttons[0]: # Pared
                            model.set_barrier(px, py)
                        elif mouse_buttons[2]: # Viento
                            model.set_wind(px, py, current_direction, current_intensity)
                        elif mouse_buttons[1]: # Borrar
                            model.clear_cell(px, py)
                
                # Guardar trazo para el siguiente frame
                last_gx, last_gy = gx, gy
            else:
                # Si el mouse se levantó, reiniciamos el trazo
                last_gx, last_gy = None, None

        # --- FÍSICA ---
        if simulating:
            model.step()

        # --- RENDERIZADO ---
        screen.fill((10, 10, 15))
        
        # 1. Dibujar Paredes
        bx, by = np.nonzero(model.barriers)
        for x, y in zip(bx, by):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (80, 80, 80), rect)

        # 2. Dibujar Fluido basado en Mapa de Presión
        pressure_map = model.get_pressure_map()
        px, py = np.nonzero(pressure_map > 0)
        for x, y in zip(px, py):
            if not model.barriers[x, y]:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color_val = min(255, 50 + (int(pressure_map[x, y]) * 40))
                pygame.draw.rect(screen, (0, int(color_val*0.8), color_val), rect)

        # 3. Dibujar Rejilla
        for x in range(GRID_W + 1):
            pygame.draw.line(screen, (30, 30, 30), (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT))
        for y in range(GRID_H + 1):
            pygame.draw.line(screen, (30, 30, 30), (0, y * CELL_SIZE), (WIDTH, y * CELL_SIZE))

        # 4. Dibujar flechas orientadoras (SIEMPRE VISIBLE)
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

        # 5. DIBUJAR HUD SUPERPUESTO
        hud_bg = pygame.Surface((480, 105))
        hud_bg.set_alpha(180)
        hud_bg.fill((0, 0, 0))
        screen.blit(hud_bg, (10, 10))
        
        state_text = "SIMULANDO" if simulating else "EDICIÓN (Pausado)"
        brush_text = f"Pincel: {DIR_NAMES[current_direction]} | Int: {INTENSITY_NAMES[current_intensity]}"
        controls_text1 = "ESPACIO: Play/Pausa | R: Limpiar Aire | N: Paso a Paso"
        controls_text2 = "Clic Izq: Pared | Clic Der: Aire | Clic Central: Borrar"

        screen.blit(font.render(f"Estado: {state_text}", True, (255, 255, 255)), (20, 20))
        screen.blit(font.render(brush_text, True, (255, 255, 255)), (20, 45))
        screen.blit(font.render(controls_text1, True, (200, 200, 200)), (20, 70))
        screen.blit(font.render(controls_text2, True, (200, 200, 200)), (20, 90))

        pygame.display.flip()
        clock.tick(15 if simulating else 60) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

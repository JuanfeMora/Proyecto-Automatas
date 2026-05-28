import pygame
import numpy as np
import sys

# =====================================================================
# 1. CONSTANTES Y CONFIGURACIÓN
# =====================================================================
CELL_SIZE = 10
GRID_W, GRID_H = 80, 60
CANVAS_WIDTH = GRID_W * CELL_SIZE
CANVAS_HEIGHT = GRID_H * CELL_SIZE
HUD_WIDTH = 260
SCREEN_WIDTH = CANVAS_WIDTH + HUD_WIDTH
SCREEN_HEIGHT = CANVAS_HEIGHT

# Materiales (int8)
MAT_AIR = 0
MAT_WALL = 1
MAT_FOLIAGE = 2
MAT_SINK = 3

# Direcciones
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = 0, 1, 2, 3

# =====================================================================
# 2. MODELO DE TENSORES (BACKEND)
# =====================================================================
class TensorModel:
    def __init__(self, w, h):
        self.w, self.h = w, h
        
        # Tensores de entorno y termodinámica
        self.materials = np.zeros((w, h), dtype=np.int8)
        self.static_pressure = np.zeros((w, h), dtype=np.int8)
        self.temperature = np.zeros((w, h), dtype=np.float32)
        self.tracers = np.zeros((w, h), dtype=np.int8)
        
        # Tensores vectoriales de viento (Magnitud 0 a 3)
        self.wind = np.zeros((4, w, h), dtype=np.int8)
        
    def set_material(self, x, y, mat): self.materials[x, y] = mat
    def set_static_pressure(self, x, y, val): self.static_pressure[x, y] = val
    def set_temperature(self, x, y, val): self.temperature[x, y] = val
    def add_tracer(self, x, y): self.tracers[x, y] = 1
    
    def set_wind(self, x, y, direction, intensity):
        self.wind[direction, x, y] = intensity
        
    def clear_cell(self, x, y):
        self.materials[x, y] = MAT_AIR
        self.static_pressure[x, y] = 0
        self.temperature[x, y] = 0.0
        self.tracers[x, y] = 0
        self.wind[:, x, y] = 0

    def update_step(self):
        """Fase Matemática t+1: DOD estricto usando NumPy"""
        
        # 1. GRADIENTE DE PRESIÓN (V = V - c∇P)
        # Inyecta viento si la presión de la celda es mayor que la vecina
        p_up = np.roll(self.static_pressure, 1, axis=1)
        p_down = np.roll(self.static_pressure, -1, axis=1)
        p_left = np.roll(self.static_pressure, 1, axis=0)
        p_right = np.roll(self.static_pressure, -1, axis=0)
        
        self.wind[DIR_UP][self.static_pressure - p_up >= 2] = 1
        self.wind[DIR_DOWN][self.static_pressure - p_down >= 2] = 1
        self.wind[DIR_LEFT][self.static_pressure - p_left >= 2] = 1
        self.wind[DIR_RIGHT][self.static_pressure - p_right >= 2] = 1

        # 2. PROPAGACIÓN VECTORIAL (Streaming)
        new_wind = np.zeros_like(self.wind)
        new_wind[DIR_UP] = np.roll(self.wind[DIR_UP], -1, axis=1)
        new_wind[DIR_DOWN] = np.roll(self.wind[DIR_DOWN], 1, axis=1)
        new_wind[DIR_LEFT] = np.roll(self.wind[DIR_LEFT], -1, axis=0)
        new_wind[DIR_RIGHT] = np.roll(self.wind[DIR_RIGHT], 1, axis=0)

        # 3. FRONTERAS Y COLISIONES (Bordes de pantalla)
        new_wind[DIR_UP, :, 0] = 0
        new_wind[DIR_DOWN, :, -1] = 0
        new_wind[DIR_LEFT, 0, :] = 0
        new_wind[DIR_RIGHT, -1, :] = 0

        # 4. RESOLUCIÓN DE MATERIALES
        # Muros (Rebote perfecto 180 grados)
        walls = (self.materials == MAT_WALL)
        bounced_up = new_wind[DIR_DOWN].copy()
        bounced_down = new_wind[DIR_UP].copy()
        bounced_left = new_wind[DIR_RIGHT].copy()
        bounced_right = new_wind[DIR_LEFT].copy()
        
        new_wind[DIR_UP][walls] = bounced_up[walls]
        new_wind[DIR_DOWN][walls] = bounced_down[walls]
        new_wind[DIR_LEFT][walls] = bounced_left[walls]
        new_wind[DIR_RIGHT][walls] = bounced_right[walls]

        # Follaje (Fricción: resta 1 a la intensidad)
        foliage = (self.materials == MAT_FOLIAGE)
        new_wind[:, foliage] = np.maximum(0, new_wind[:, foliage] - 1)

        # Sumideros (Aniquilación)
        sinks = (self.materials == MAT_SINK)
        new_wind[:, sinks] = 0
        self.static_pressure[sinks] = 0

        self.wind = new_wind

        # 5. ADVECCIÓN CINEMÁTICA (Trazadores y Temperatura)
        # Simplificación: Los trazadores se mueven hacia el vector dominante
        dominant_wind = np.argmax(self.wind, axis=0)
        has_wind = np.max(self.wind, axis=0) > 0
        
        new_tracers = np.zeros_like(self.tracers)
        # Mover trazadores según dirección dominante
        mov_up = (dominant_wind == DIR_UP) & has_wind & (self.tracers == 1)
        new_tracers = np.logical_or(new_tracers, np.roll(mov_up, -1, axis=1))
        
        mov_down = (dominant_wind == DIR_DOWN) & has_wind & (self.tracers == 1)
        new_tracers = np.logical_or(new_tracers, np.roll(mov_down, 1, axis=1))
        
        mov_left = (dominant_wind == DIR_LEFT) & has_wind & (self.tracers == 1)
        new_tracers = np.logical_or(new_tracers, np.roll(mov_left, -1, axis=0))
        
        mov_right = (dominant_wind == DIR_RIGHT) & has_wind & (self.tracers == 1)
        new_tracers = np.logical_or(new_tracers, np.roll(mov_right, 1, axis=0))
        
        # Mantener trazadores estáticos si no hay viento
        new_tracers = np.logical_or(new_tracers, (self.tracers == 1) & ~has_wind)
        new_tracers[walls] = 0 # Destruir contra muros
        self.tracers = new_tracers.astype(np.int8)

    def render(self, surface):
        """Renderizado eficiente iterando la matriz (Frontend 1)"""
        # Calcular presión dinámica para render (suma de los 4 vectores)
        dyn_pressure = np.sum(self.wind, axis=0)

        for x in range(self.w):
            for y in range(self.h):
                px, py = x * CELL_SIZE, y * CELL_SIZE
                mat = self.materials[x, y]
                
                # Fondo basado en presión dinámica (Azul muy oscuro)
                bg_color = (0, 0, min(255, dyn_pressure[x, y] * 30))
                
                if mat == MAT_WALL:
                    color = (100, 100, 100) # Gris
                elif mat == MAT_FOLIAGE:
                    color = (34, 139, 34) # Verde bosque
                elif mat == MAT_SINK:
                    color = (139, 0, 0) # Rojo oscuro
                else:
                    color = bg_color

                # Aplicar Temperatura si es aire
                if mat == MAT_AIR and self.temperature[x, y] > 0:
                    temp_intensity = min(255, int(self.temperature[x, y] * 2.5))
                    color = (temp_intensity, 0, max(0, 255 - temp_intensity))

                pygame.draw.rect(surface, color, (px, py, CELL_SIZE, CELL_SIZE))

                # Dibujar Trazadores
                if self.tracers[x, y] == 1:
                    pygame.draw.rect(surface, (255, 255, 0), (px+2, py+2, CELL_SIZE-4, CELL_SIZE-4))

# =====================================================================
# 3. INTERFAZ GRÁFICA Y CONTROLADOR (FRONTEND 2)
# =====================================================================
class SimulationGUI:
    def __init__(self, model):
        self.model = model
        self.font = pygame.font.SysFont('Consolas', 12, bold=True)
        self.title_font = pygame.font.SysFont('Consolas', 16, bold=True)
        
        self.simulation_running = False
        
        self.tools = [
            'Muro Sólido', 'Follaje', 'Sumidero', 'Borrar (Aire)',
            'Inyectar Viento', 'Pincel Temp', 'Pincel Presión', 'Añadir Trazador'
        ]
        self.active_tool = 'Muro Sólido'
        self.current_value = 50 
        self.wind_dir = 'DERECHA'
        self.dirs = ['ARRIBA', 'DERECHA', 'ABAJO', 'IZQUIERDA']
        
        self.buttons = {}
        self._init_ui_layout()
        self.last_gx, self.last_gy = None, None

    def _init_ui_layout(self):
        start_y = 15
        start_x = CANVAS_WIDTH + 10
        btn_w = HUD_WIDTH - 20
        
        for tool in self.tools:
            self.buttons[tool] = pygame.Rect(start_x, start_y, btn_w, 25)
            start_y += 32
            
        start_y += 10
        self.buttons['TOGGLE_SIM'] = pygame.Rect(start_x, start_y, btn_w, 40)

    def draw_hud(self, surface):
        hud_rect = pygame.Rect(CANVAS_WIDTH, 0, HUD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(surface, (25, 30, 35), hud_rect)
        pygame.draw.line(surface, (0, 255, 255), (CANVAS_WIDTH, 0), (CANVAS_WIDTH, SCREEN_HEIGHT), 2)

        # Título
        title = self.title_font.render("HERRAMIENTAS T=0", True, (0, 255, 255))
        surface.blit(title, (CANVAS_WIDTH + 50, 15))

        # Botones de Pinceles
        for tool, rect in self.buttons.items():
            if tool == 'TOGGLE_SIM': continue
            color = (0, 150, 200) if self.active_tool == tool else (60, 70, 80)
            pygame.draw.rect(surface, color, rect, border_radius=3)
            if self.active_tool == tool:
                pygame.draw.rect(surface, (0, 255, 255), rect, 1, border_radius=3)
                
            text_surf = self.font.render(tool, True, (255, 255, 255))
            surface.blit(text_surf, (rect.x + 10, rect.y + 6))

        # Indicadores Dinámicos
        info_y = self.buttons['TOGGLE_SIM'].y + 60
        pygame.draw.line(surface, (100, 100, 100), (CANVAS_WIDTH+10, info_y), (SCREEN_WIDTH-10, info_y))
        
        info_text = [
            ("Valor (Q/W para +/-):", (200, 200, 200)),
            (f"{self.current_value}", (255, 255, 0)),
            ("Dirección (Flechas):", (200, 200, 200)),
            (f"{self.wind_dir}", (0, 255, 0)),
            ("ESTADO DEL MOTOR:", (200, 200, 200)),
            ("BLOQUEADO (Iterando)" if self.simulation_running else "ABIERTO (Config)", 
             (255, 50, 50) if self.simulation_running else (50, 255, 50))
        ]
        
        info_y += 10
        for text, color in info_text:
            surf = self.font.render(text, True, color)
            surface.blit(surf, (CANVAS_WIDTH + 15, info_y))
            info_y += 20

        # Botón Simulación (Fondo dinámico)
        sim_rect = self.buttons['TOGGLE_SIM']
        sim_color = (200, 50, 50) if self.simulation_running else (50, 200, 50)
        pygame.draw.rect(surface, sim_color, sim_rect, border_radius=5)
        sim_text = "DETENER [ESPACIO]" if self.simulation_running else "INICIAR [ESPACIO]"
        surf = self.title_font.render(sim_text, True, (255, 255, 255))
        surface.blit(surf, (sim_rect.x + 35, sim_rect.y + 10))

    def handle_keydown(self, key):
        if key == pygame.K_SPACE:
            self.simulation_running = not self.simulation_running
        elif not self.simulation_running:
            if key == pygame.K_q: self.current_value = max(-100, self.current_value - 10)
            elif key == pygame.K_w: self.current_value = min(100, self.current_value + 10)
            elif key == pygame.K_UP: self.wind_dir = 'ARRIBA'
            elif key == pygame.K_DOWN: self.wind_dir = 'ABAJO'
            elif key == pygame.K_LEFT: self.wind_dir = 'IZQUIERDA'
            elif key == pygame.K_RIGHT: self.wind_dir = 'DERECHA'

    def handle_click(self, event):
        if event.button == 1:
            for tool, rect in self.buttons.items():
                if rect.collidepoint(event.pos):
                    if tool == 'TOGGLE_SIM':
                        self.simulation_running = not self.simulation_running
                    else:
                        self.active_tool = tool

    def handle_mouse_drag(self):
        if self.simulation_running: return # Bloqueo Causal

        mouse_buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()

        if mx < CANVAS_WIDTH:  
            gx = max(0, min(mx // CELL_SIZE, GRID_W - 1))
            gy = max(0, min(my // CELL_SIZE, GRID_H - 1))
            
            if any(mouse_buttons):
                points = self.get_bresenham_line(self.last_gx, self.last_gy, gx, gy) if self.last_gx is not None else [(gx, gy)]
                
                for px, py in points:
                    if 0 <= px < GRID_W and 0 <= py < GRID_H:
                        if mouse_buttons[0]:
                            self._apply_tool(px, py)
                        elif mouse_buttons[2]:
                            self.model.clear_cell(px, py)
                self.last_gx, self.last_gy = gx, gy
            else:
                self.last_gx, self.last_gy = None, None

    def _apply_tool(self, px, py):
        if self.active_tool == 'Muro Sólido': self.model.set_material(px, py, MAT_WALL)
        elif self.active_tool == 'Follaje': self.model.set_material(px, py, MAT_FOLIAGE)
        elif self.active_tool == 'Sumidero': self.model.set_material(px, py, MAT_SINK)
        elif self.active_tool == 'Borrar (Aire)': self.model.clear_cell(px, py)
        elif self.active_tool == 'Inyectar Viento':
            d_map = {'ARRIBA': DIR_UP, 'ABAJO': DIR_DOWN, 'IZQUIERDA': DIR_LEFT, 'DERECHA': DIR_RIGHT}
            int_viento = 3 if abs(self.current_value) > 40 else (2 if abs(self.current_value) > 10 else 1)
            self.model.set_wind(px, py, d_map[self.wind_dir], int_viento)
        elif self.active_tool == 'Pincel Temp':
            self.model.set_temperature(px, py, self.current_value)
        elif self.active_tool == 'Pincel Presión':
            self.model.set_static_pressure(px, py, self.current_value)
        elif self.active_tool == 'Añadir Trazador':
            self.model.add_tracer(px, py)

    def get_bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

# =====================================================================
# 4. CICLO PRINCIPAL (CONTROLADOR C)
# =====================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lattice Gas Automata - Vectorizado")
    clock = pygame.time.Clock()

    model = TensorModel(GRID_W, GRID_H)
    gui = SimulationGUI(model)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gui.handle_click(event)
            elif event.type == pygame.KEYDOWN:
                gui.handle_keydown(event.key)

        gui.handle_mouse_drag()

        if gui.simulation_running:
            model.update_step()

        # Limpiar y Renderizar
        screen.fill((0, 0, 0))
        model.render(screen)
        gui.draw_hud(screen)

        pygame.display.flip()
        clock.tick(30) # 30 FPS recomendados para observar el autómata

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()

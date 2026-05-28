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
        self.wind[:, x, y] = 0
        self.wind[direction, x, y] = intensity
        
    def clear_cell(self, x, y):
        self.materials[x, y] = MAT_AIR
        self.static_pressure[x, y] = 0
        self.temperature[x, y] = 0.0
        self.tracers[x, y] = 0
        self.wind[:, x, y] = 0

    def reset_fluid(self):
        self.wind.fill(0)
        self.temperature.fill(0)
        self.tracers.fill(0)

    def update_step(self):
        """Fase Matemática t+1: DOD estricto usando NumPy"""
        
        # 1. GRADIENTE DE PRESIÓN (V = V - c∇P)
        air = (self.materials == MAT_AIR)
        p_up = np.roll(self.static_pressure, 1, axis=1)
        p_down = np.roll(self.static_pressure, -1, axis=1)
        p_left = np.roll(self.static_pressure, 1, axis=0)
        p_right = np.roll(self.static_pressure, -1, axis=0)
        
        mask_up = air & (self.static_pressure - p_up >= 2)
        mask_down = air & (self.static_pressure - p_down >= 2)
        mask_left = air & (self.static_pressure - p_left >= 2)
        mask_right = air & (self.static_pressure - p_right >= 2)

        # Evitar el wrap-around del gradiente en los límites de la pantalla
        mask_up[:, 0] = False
        mask_down[:, -1] = False
        mask_left[0, :] = False
        mask_right[-1, :] = False

        self.wind[DIR_UP][mask_up] = np.clip(self.wind[DIR_UP][mask_up] + 1, 0, 3)
        self.wind[DIR_DOWN][mask_down] = np.clip(self.wind[DIR_DOWN][mask_down] + 1, 0, 3)
        self.wind[DIR_LEFT][mask_left] = np.clip(self.wind[DIR_LEFT][mask_left] + 1, 0, 3)
        self.wind[DIR_RIGHT][mask_right] = np.clip(self.wind[DIR_RIGHT][mask_right] + 1, 0, 3)

        # 2. PROPAGACIÓN VECTORIAL (Streaming)
        new_wind = np.zeros_like(self.wind)
        new_wind[DIR_UP] = np.roll(self.wind[DIR_UP], -1, axis=1)
        new_wind[DIR_DOWN] = np.roll(self.wind[DIR_DOWN], 1, axis=1)
        new_wind[DIR_LEFT] = np.roll(self.wind[DIR_LEFT], -1, axis=0)
        new_wind[DIR_RIGHT] = np.roll(self.wind[DIR_RIGHT], 1, axis=0)

        # 3. FRONTERAS SÓLIDAS Y CORRECCIÓN DE WRAP-AROUND
        out_up = new_wind[DIR_UP, :, -1].copy()
        out_down = new_wind[DIR_DOWN, :, 0].copy()
        out_left = new_wind[DIR_LEFT, -1, :].copy()
        out_right = new_wind[DIR_RIGHT, 0, :].copy()
        
        new_wind[DIR_UP, :, -1] = 0
        new_wind[DIR_DOWN, :, 0] = 0
        new_wind[DIR_LEFT, -1, :] = 0
        new_wind[DIR_RIGHT, 0, :] = 0
        
        new_wind[DIR_DOWN, :, 0] = np.maximum(new_wind[DIR_DOWN, :, 0], out_up)
        new_wind[DIR_UP, :, -1] = np.maximum(new_wind[DIR_UP, :, -1], out_down)
        new_wind[DIR_RIGHT, 0, :] = np.maximum(new_wind[DIR_RIGHT, 0, :], out_left)
        new_wind[DIR_LEFT, -1, :] = np.maximum(new_wind[DIR_LEFT, -1, :], out_right)

        # 4. RESOLUCIÓN DE MATERIALES
        walls = (self.materials == MAT_WALL)
        bounced_up = new_wind[DIR_DOWN].copy()
        bounced_down = new_wind[DIR_UP].copy()
        bounced_left = new_wind[DIR_RIGHT].copy()
        bounced_right = new_wind[DIR_LEFT].copy()
        
        new_wind[DIR_UP][walls] = bounced_up[walls]
        new_wind[DIR_DOWN][walls] = bounced_down[walls]
        new_wind[DIR_LEFT][walls] = bounced_left[walls]
        new_wind[DIR_RIGHT][walls] = bounced_right[walls]

        foliage = (self.materials == MAT_FOLIAGE)
        new_wind[:, foliage] = np.maximum(0, new_wind[:, foliage] - 1)

        sinks = (self.materials == MAT_SINK)
        new_wind[:, sinks] = 0
        self.static_pressure[sinks] = 0

        self.wind = new_wind

        # 4.1 FRICCIÓN ESTOCÁSTICA Y EFECTO VACÍO (Entropía y Disipación)
        friction_mask = np.random.random((self.w, self.h)) < 0.02
        self.wind = np.where((self.wind > 0) & friction_mask, self.wind - 1, self.wind).astype(np.int8)

        # Si una celda tiene presión negativa (vacío), funciona como un sumidero parcial
        vacuum = (self.static_pressure < 0)
        if np.any(vacuum):
            self.wind[:, vacuum] = np.maximum(0, self.wind[:, vacuum] - 1)
            
        # 4.2 DECAIMIENTO RÁPIDO DE LA PRESIÓN ESTÁTICA (Shockwave / Impacto)
        # Pierde 2 puntos de presión por frame. Esto evita la "multiplicación infinita" de aire.
        # Una pincelada de presión 10 durará exactamente 5 frames, actuando como una explosión controlada.
        pos_p = self.static_pressure > 0
        neg_p = self.static_pressure < 0
        self.static_pressure[pos_p] = np.maximum(0, self.static_pressure[pos_p] - 2)
        self.static_pressure[neg_p] = np.minimum(0, self.static_pressure[neg_p] + 2)

        # 4.5. COLISIONES DE FLUIDOS (Modelo HPP Vectorizado)
        air = (self.materials == MAT_AIR)
        up, down, left, right = self.wind[DIR_UP], self.wind[DIR_DOWN], self.wind[DIR_LEFT], self.wind[DIR_RIGHT]
        
        # 1. Identificar colisiones frontales (pares de partículas opuestas)
        pairs_y = np.minimum(up, down)
        pairs_x = np.minimum(left, right)
        
        # 2. Calcular espacio disponible en el eje ortogonal (Límite 3)
        space_x = 3 - np.maximum(left, right)
        space_y = 3 - np.maximum(up, down)
        
        # 3. Determinar rotación ortogonal conservando masa y momento
        rotate_y_to_x = np.clip(pairs_y - pairs_x, 0, space_x)
        rotate_x_to_y = np.clip(pairs_x - pairs_y, 0, space_y)
        
        delta = rotate_x_to_y - rotate_y_to_x
        
        # 4. Aplicar dispersión solo donde hay aire
        self.wind[DIR_UP] = up + np.where(air, delta, 0)
        self.wind[DIR_DOWN] = down + np.where(air, delta, 0)
        self.wind[DIR_LEFT] = left - np.where(air, delta, 0)
        self.wind[DIR_RIGHT] = right - np.where(air, delta, 0)

        # 4.6. TERMODINÁMICA
        T = self.temperature
        T_adv = T.copy()
        weight = 0.25
        for ch, (shift, axis) in enumerate([(-1, 1), (1, 1), (-1, 0), (1, 0)]):
            intensity = self.wind[ch].astype(np.float32) / 3.0
            flux = np.roll(T * intensity, shift=shift, axis=axis) * weight
            T_adv += flux
            T_adv -= T * intensity * weight

        lap = np.roll(T, -1, axis=0) + np.roll(T, 1, axis=0) + np.roll(T, -1, axis=1) + np.roll(T, 1, axis=1) - 4.0 * T
        T_new = T_adv + 0.05 * lap
        
        # Enfriamiento natural (ambient cooling) para que el calor no sea una fuente infinita
        T_new *= 0.96 
        self.temperature[air] = np.clip(T_new[air], -100.0, 100.0)

        # 5. ADVECCIÓN CINEMÁTICA REALISTA (Polvo y Convección Térmica)
        # 5.1. Convección: El calor genera corrientes ascendentes, el frío descendentes (ahora con control estocástico de inyección de masa)
        prob_hot = np.random.random((self.w, self.h)) < (np.maximum(0, self.temperature) / 300.0)
        hot = (self.temperature > 5) & air & prob_hot
        
        prob_cold = np.random.random((self.w, self.h)) < (np.maximum(0, -self.temperature) / 300.0)
        cold = (self.temperature < -5) & air & prob_cold
        
        self.wind[DIR_UP][hot] = np.clip(self.wind[DIR_UP][hot] + 1, 0, 3)
        self.wind[DIR_DOWN][hot] = np.maximum(0, self.wind[DIR_DOWN][hot] - 1)
        
        self.wind[DIR_DOWN][cold] = np.clip(self.wind[DIR_DOWN][cold] + 1, 0, 3)
        self.wind[DIR_UP][cold] = np.maximum(0, self.wind[DIR_UP][cold] - 1)
        
        # 5.2. Dinámica del Polvo (Vectores 2D Diagonales)
        vx = self.wind[DIR_RIGHT].astype(np.int8) - self.wind[DIR_LEFT].astype(np.int8)
        vy = self.wind[DIR_DOWN].astype(np.int8) - self.wind[DIR_UP].astype(np.int8)
        
        new_tracers = np.zeros_like(self.tracers)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                mask = (np.sign(vx) == dx) & (np.sign(vy) == dy) & (self.tracers == 1)
                if np.any(mask):
                    moved = np.roll(mask, shift=dx, axis=0)
                    moved = np.roll(moved, shift=dy, axis=1)
                    new_tracers = np.logical_or(new_tracers, moved)
                    
        new_tracers[walls] = 0
        self.tracers = new_tracers.astype(np.int8)

    def render(self, surface):
        """Renderizado eficiente iterando la matriz (Frontend 1)"""
        dyn_pressure = np.sum(self.wind, axis=0)

        for x in range(self.w):
            for y in range(self.h):
                px, py = x * CELL_SIZE, y * CELL_SIZE
                mat = self.materials[x, y]
                
                if mat == MAT_WALL:
                    color = (150, 150, 150) # Gris más claro
                elif mat == MAT_FOLIAGE:
                    color = (34, 139, 34) # Verde bosque
                elif mat == MAT_SINK:
                    color = (139, 0, 0) # Rojo oscuro
                else: # MAT_AIR
                    base_val = min(255, dyn_pressure[x, y] * 40)
                    r, g, b = base_val, base_val, min(255, base_val + 40)
                    
                    # 1. Influencia de la Temperatura (Rojo = Caliente, Cian = Frío)
                    temp = self.temperature[x, y]
                    if temp > 1:
                        r = min(255, r + temp * 2.5)
                        g = max(0, g - temp * 0.5)
                        b = max(0, b - temp * 0.5)
                    elif temp < -1:
                        abs_temp = abs(temp)
                        b = min(255, b + abs_temp * 2.5)
                        g = min(255, g + abs_temp * 1.5)
                        r = max(0, r - abs_temp * 0.5)
                    
                    # 2. Influencia de la Presión Estática (Púrpura = Alta Presión, Verde = Vacío/Baja Presión)
                    p = int(self.static_pressure[x, y])
                    if p > 0:
                        r = min(255, r + p * 18)
                        g = max(0, g - p * 12)
                        b = min(255, b + p * 24)
                    elif p < 0:
                        abs_p = abs(p)
                        r = max(0, r - abs_p * 12)
                        g = min(255, g + abs_p * 20)
                        b = max(0, b - abs_p * 8)
                        
                    color = (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))

                pygame.draw.rect(surface, color, (px, py, CELL_SIZE, CELL_SIZE))

                if self.tracers[x, y] == 1:
                    pygame.draw.rect(surface, (255, 255, 0), (px+2, py+2, CELL_SIZE-4, CELL_SIZE-4))

        for ch in range(4):
            xs, ys = np.nonzero(self.wind[ch] > 0)
            for x, y in zip(xs, ys):
                intensity = self.wind[ch, x, y]
                temp = self.temperature[x, y]
                
                if temp > 1: base_color = (255, max(0, 200 - int(abs(temp)*2)), 0)
                elif temp < -1: base_color = (0, 200, 255)
                else: base_color = (255, 255, 255)
                
                # Oscuridad/Claridad modulada por la intensidad del viento (1=Oscuro, 2=Medio, 3=Brillante)
                factor = [0.0, 0.4, 0.7, 1.0][intensity]
                color_wind = (int(base_color[0]*factor), int(base_color[1]*factor), int(base_color[2]*factor))
                
                cx, cy = x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2
                dx, dy = 0, 0
                if ch == DIR_UP: dy = -CELL_SIZE // 2
                elif ch == DIR_DOWN: dy = CELL_SIZE // 2
                elif ch == DIR_LEFT: dx = -CELL_SIZE // 2
                elif ch == DIR_RIGHT: dx = CELL_SIZE // 2
                pygame.draw.line(surface, color_wind, (cx, cy), (cx + dx, cy + dy), width=intensity)    

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
        
        self.val_wind = 3     
        self.val_temp = 100   
        self.val_press = 5    
        
        self.wind_dir = 'DERECHA'
        self.dirs = ['ARRIBA', 'DERECHA', 'ABAJO', 'IZQUIERDA']
        self.input_buffer = ""
        
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

        title = self.title_font.render("HERRAMIENTAS T=0", True, (0, 255, 255))
        surface.blit(title, (CANVAS_WIDTH + 50, 15))

        for tool, rect in self.buttons.items():
            if tool == 'TOGGLE_SIM': continue
            color = (0, 150, 200) if self.active_tool == tool else (60, 70, 80)
            pygame.draw.rect(surface, color, rect, border_radius=3)
            if self.active_tool == tool:
                pygame.draw.rect(surface, (0, 255, 255), rect, 1, border_radius=3)
                
            text_surf = self.font.render(tool, True, (255, 255, 255))
            surface.blit(text_surf, (rect.x + 10, rect.y + 6))

        info_y = self.buttons['TOGGLE_SIM'].y + 60
        pygame.draw.line(surface, (100, 100, 100), (CANVAS_WIDTH+10, info_y), (SCREEN_WIDTH-10, info_y))
        
        info_y += 10
        surf = self.font.render("Q/W para ajustar valores:", True, (200, 200, 200))
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y))
        
        info_y += 20
        c_wind = (0, 255, 0) if self.active_tool == 'Inyectar Viento' else (150, 150, 150)
        c_temp = (0, 255, 0) if self.active_tool == 'Pincel Temp' else (150, 150, 150)
        c_press = (0, 255, 0) if self.active_tool == 'Pincel Presión' else (150, 150, 150)

        t_wind = self.font.render(f"Intensidad Viento: {self.val_wind}", True, c_wind)
        t_temp = self.font.render(f"Temperatura: {self.val_temp} (C=Frío, H=Calor)", True, c_temp)
        t_press = self.font.render(f"Presión Base: {self.val_press}", True, c_press)
        
        surface.blit(t_wind, (CANVAS_WIDTH + 15, info_y)); info_y += 15
        surface.blit(t_temp, (CANVAS_WIDTH + 15, info_y)); info_y += 15
        surface.blit(t_press, (CANVAS_WIDTH + 15, info_y)); info_y += 25
        
        surf = self.font.render("Dirección Viento (Flechas):", True, (200, 200, 200))
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y)); info_y += 15
        surf = self.font.render(f"{self.wind_dir}", True, (0, 255, 0))
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y)); info_y += 25

        surf = self.font.render("ESTADO DEL MOTOR:", True, (200, 200, 200))
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y)); info_y += 15
        
        state_text = "BLOQUEADO (Iterando)" if self.simulation_running else "ABIERTO (Config)"
        state_color = (255, 50, 50) if self.simulation_running else (50, 255, 50)
        surf = self.font.render(state_text, True, state_color)
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y)); info_y += 25
        
        # Casilla de entrada de texto interactiva
        surf = self.font.render("Ingresar valor exacto:", True, (200, 200, 200))
        surface.blit(surf, (CANVAS_WIDTH + 15, info_y)); info_y += 15
        
        self.input_box_rect = pygame.Rect(CANVAS_WIDTH + 15, info_y, HUD_WIDTH - 30, 25)
        box_color = (0, 255, 255) if getattr(self, 'input_active', False) else (100, 100, 100)
        pygame.draw.rect(surface, (20, 25, 30), self.input_box_rect)
        pygame.draw.rect(surface, box_color, self.input_box_rect, 2, border_radius=3)
        
        txt_surface = self.font.render(self.input_buffer + ("_" if getattr(self, 'input_active', False) else ""), True, (255, 255, 255))
        surface.blit(txt_surface, (self.input_box_rect.x + 8, self.input_box_rect.y + 6))

        sim_rect = self.buttons['TOGGLE_SIM']
        sim_color = (200, 50, 50) if self.simulation_running else (50, 200, 50)
        pygame.draw.rect(surface, sim_color, sim_rect, border_radius=5)
        sim_text = "DETENER [ESPACIO]" if self.simulation_running else "INICIAR [ESPACIO]"
        surf = self.title_font.render(sim_text, True, (255, 255, 255))
        surface.blit(surf, (sim_rect.x + 35, sim_rect.y + 10))

    def draw_cursor(self, surface):
        """Dibuja un cursor fantasma (hover) sobre la celda en la que está posado el ratón"""
        mx, my = pygame.mouse.get_pos()
        if mx < CANVAS_WIDTH:
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                rect = (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, (255, 255, 255), rect, width=1) # Borde blanco

    def handle_keydown(self, event):
        key = event.key
        
        # Si la casilla de texto está activa, capturar teclado y bloquear otros atajos
        if getattr(self, 'input_active', False):
            if key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
                try:
                    val = int(self.input_buffer)
                    if self.active_tool == 'Inyectar Viento':
                        self.val_wind = max(1, min(3, val))
                    elif self.active_tool == 'Pincel Temp':
                        self.val_temp = max(-100, min(100, val))
                    elif self.active_tool == 'Pincel Presión':
                        self.val_press = max(-10, min(10, val))
                except ValueError:
                    pass
                self.input_buffer = ""
                self.input_active = False
            elif key == pygame.K_BACKSPACE:
                self.input_buffer = self.input_buffer[:-1]
            elif key == pygame.K_ESCAPE:
                self.input_active = False
            elif hasattr(event, 'unicode') and (event.unicode.isdigit() or event.unicode == '-'):
                self.input_buffer += event.unicode
            return

        if key == pygame.K_SPACE:
            self.simulation_running = not self.simulation_running
        elif key == pygame.K_r:
            self.model.reset_fluid()
        elif not self.simulation_running:
            if key == pygame.K_q: 
                if self.active_tool == 'Inyectar Viento':
                    self.val_wind = max(1, self.val_wind - 1)
                elif self.active_tool == 'Pincel Temp':
                    self.val_temp = max(-100, self.val_temp - 10)
                elif self.active_tool == 'Pincel Presión':
                    self.val_press = max(-10, self.val_press - 1)
            elif key == pygame.K_w: 
                if self.active_tool == 'Inyectar Viento':
                    self.val_wind = min(3, self.val_wind + 1)
                elif self.active_tool == 'Pincel Temp':
                    self.val_temp = min(100, self.val_temp + 10)
                elif self.active_tool == 'Pincel Presión':
                    self.val_press = min(10, self.val_press + 1)
            elif key == pygame.K_UP: self.wind_dir = 'ARRIBA'
            elif key == pygame.K_DOWN: self.wind_dir = 'ABAJO'
            elif key == pygame.K_LEFT: self.wind_dir = 'IZQUIERDA'
            elif key == pygame.K_RIGHT: self.wind_dir = 'DERECHA'
            # Accesos directos para intensidad durante el trazo
            elif key == pygame.K_1: self.val_wind = 1
            elif key == pygame.K_2: self.val_wind = 2
            elif key == pygame.K_3: self.val_wind = 3
            # Accesos directos rápidos para temperatura
            elif key == pygame.K_c: self.val_temp = -100
            elif key == pygame.K_h: self.val_temp = 100

    def handle_click(self, event):
        if event.button == 1:
            # Primero chequear si clicamos en la casilla de texto
            if hasattr(self, 'input_box_rect') and self.input_box_rect.collidepoint(event.pos):
                self.input_active = True
            else:
                self.input_active = False
                # Evaluar el resto de botones
                for tool, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        if tool == 'TOGGLE_SIM':
                            self.simulation_running = not self.simulation_running
                        else:
                            self.active_tool = tool

    def handle_mouse_drag(self):
        if self.simulation_running: return

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
            self.model.set_wind(px, py, d_map[self.wind_dir], self.val_wind)
        elif self.active_tool == 'Pincel Temp':
            self.model.set_temperature(px, py, self.val_temp)
        elif self.active_tool == 'Pincel Presión':
            self.model.set_static_pressure(px, py, self.val_press)
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
                gui.handle_keydown(event)

        gui.handle_mouse_drag()

        if gui.simulation_running:
            model.update_step()

        screen.fill((0, 0, 0))
        model.render(screen)
        gui.draw_cursor(screen)
        gui.draw_hud(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()

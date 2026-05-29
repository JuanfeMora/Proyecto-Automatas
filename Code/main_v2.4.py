import pygame
import numpy as np
import sys
import time

# =====================================================================
# 1. CONSTANTES Y CONFIGURACIÓN
# =====================================================================
CELL_SIZE = 10
GRID_W, GRID_H = 80, 60
CANVAS_WIDTH  = GRID_W * CELL_SIZE
CANVAS_HEIGHT = GRID_H * CELL_SIZE
HUD_WIDTH     = 280
SCREEN_WIDTH  = CANVAS_WIDTH + HUD_WIDTH
SCREEN_HEIGHT = CANVAS_HEIGHT

# Materiales
MAT_AIR    = 0
MAT_WALL   = 1
MAT_FOLIAGE= 2
MAT_SINK   = 3

# Direcciones
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = 0, 1, 2, 3

# ── Paleta de colores del HUD ─────────────────────────────────────────
C_BG        = (14, 17, 23)
C_PANEL     = (20, 25, 33)
C_BORDER    = (0, 200, 255)
C_ACCENT    = (0, 200, 255)
C_BTN_OFF   = (38, 48, 60)
C_BTN_ON    = (0, 110, 170)
C_TEXT      = (210, 220, 230)
C_DIM       = (100, 115, 130)
C_GREEN     = (50, 220, 100)
C_RED       = (220, 60, 60)
C_YELLOW    = (255, 210, 50)
C_ORANGE    = (255, 140, 30)

# =====================================================================
# 2. MODELO DE TENSORES (BACKEND)
# =====================================================================
class TensorModel:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.materials       = np.zeros((w, h), dtype=np.int8)
        self.static_pressure = np.zeros((w, h), dtype=np.int8)
        self.temperature     = np.zeros((w, h), dtype=np.float32)
        self.tracers         = np.zeros((w, h), dtype=np.int8)
        self.wind            = np.zeros((4, w, h), dtype=np.int8)

    def set_material(self, x, y, mat):        self.materials[x, y] = mat
    def set_static_pressure(self, x, y, val): self.static_pressure[x, y] = val
    def set_temperature(self, x, y, val):     self.temperature[x, y] = val
    def add_tracer(self, x, y):               self.tracers[x, y] = 1

    def set_wind(self, x, y, direction, intensity):
        self.wind[:, x, y] = 0
        self.wind[direction, x, y] = intensity

    def clear_cell(self, x, y):
        self.materials[x, y]       = MAT_AIR
        self.static_pressure[x, y] = 0
        self.temperature[x, y]     = 0.0
        self.tracers[x, y]         = 0
        self.wind[:, x, y]         = 0

    def reset_fluid(self):
        self.wind.fill(0)
        self.temperature.fill(0)
        self.tracers.fill(0)

    def reset_all(self):
        """Limpia absolutamente todo: materiales, fluido, temperatura, trazadores."""
        self.materials.fill(0)
        self.static_pressure.fill(0)
        self.wind.fill(0)
        self.temperature.fill(0)
        self.tracers.fill(0)

    def update_step(self):
        air    = (self.materials == MAT_AIR)
        p_up   = np.roll(self.static_pressure,  1, axis=1)
        p_down = np.roll(self.static_pressure, -1, axis=1)
        p_left = np.roll(self.static_pressure,  1, axis=0)
        p_right= np.roll(self.static_pressure, -1, axis=0)

        mask_up    = air & (self.static_pressure - p_up    >= 2)
        mask_down  = air & (self.static_pressure - p_down  >= 2)
        mask_left  = air & (self.static_pressure - p_left  >= 2)
        mask_right = air & (self.static_pressure - p_right >= 2)

        mask_up[:, 0]   = False;  mask_down[:, -1]  = False
        mask_left[0, :] = False;  mask_right[-1, :] = False

        self.wind[DIR_UP][mask_up]       = np.clip(self.wind[DIR_UP][mask_up]    + 1, 0, 3)
        self.wind[DIR_DOWN][mask_down]   = np.clip(self.wind[DIR_DOWN][mask_down]  + 1, 0, 3)
        self.wind[DIR_LEFT][mask_left]   = np.clip(self.wind[DIR_LEFT][mask_left]  + 1, 0, 3)
        self.wind[DIR_RIGHT][mask_right] = np.clip(self.wind[DIR_RIGHT][mask_right]+ 1, 0, 3)

        new_wind = np.zeros_like(self.wind)
        new_wind[DIR_UP]    = np.roll(self.wind[DIR_UP],   -1, axis=1)
        new_wind[DIR_DOWN]  = np.roll(self.wind[DIR_DOWN],  1, axis=1)
        new_wind[DIR_LEFT]  = np.roll(self.wind[DIR_LEFT], -1, axis=0)
        new_wind[DIR_RIGHT] = np.roll(self.wind[DIR_RIGHT], 1, axis=0)

        out_up    = new_wind[DIR_UP,   :, -1].copy()
        out_down  = new_wind[DIR_DOWN, :,  0].copy()
        out_left  = new_wind[DIR_LEFT, -1, :].copy()
        out_right = new_wind[DIR_RIGHT, 0, :].copy()

        new_wind[DIR_UP,   :, -1] = 0;  new_wind[DIR_DOWN, :,  0] = 0
        new_wind[DIR_LEFT, -1, :] = 0;  new_wind[DIR_RIGHT, 0, :] = 0

        new_wind[DIR_DOWN,  :,  0] = np.maximum(new_wind[DIR_DOWN,  :,  0], out_up)
        new_wind[DIR_UP,    :, -1] = np.maximum(new_wind[DIR_UP,    :, -1], out_down)
        new_wind[DIR_RIGHT, 0,  :] = np.maximum(new_wind[DIR_RIGHT, 0,  :], out_left)
        new_wind[DIR_LEFT, -1,  :] = np.maximum(new_wind[DIR_LEFT, -1,  :], out_right)

        walls = (self.materials == MAT_WALL)
        new_wind[DIR_UP][walls]    = new_wind[DIR_DOWN].copy()[walls]
        new_wind[DIR_DOWN][walls]  = new_wind[DIR_UP].copy()[walls]
        new_wind[DIR_LEFT][walls]  = new_wind[DIR_RIGHT].copy()[walls]
        new_wind[DIR_RIGHT][walls] = new_wind[DIR_LEFT].copy()[walls]

        foliage = (self.materials == MAT_FOLIAGE)
        new_wind[:, foliage] = np.maximum(0, new_wind[:, foliage] - 1)

        sinks = (self.materials == MAT_SINK)
        new_wind[:, sinks] = 0
        self.static_pressure[sinks] = 0

        self.wind = new_wind

        friction_mask = np.random.random((self.w, self.h)) < 0.02
        self.wind = np.where((self.wind > 0) & friction_mask, self.wind - 1, self.wind).astype(np.int8)

        vacuum = (self.static_pressure < 0)
        if np.any(vacuum):
            self.wind[:, vacuum] = np.maximum(0, self.wind[:, vacuum] - 1)

        pos_p = self.static_pressure > 0;  neg_p = self.static_pressure < 0
        self.static_pressure[pos_p] = np.maximum(0, self.static_pressure[pos_p] - 2)
        self.static_pressure[neg_p] = np.minimum(0, self.static_pressure[neg_p] + 2)

        air   = (self.materials == MAT_AIR)
        up, down, left, right = self.wind[DIR_UP], self.wind[DIR_DOWN], self.wind[DIR_LEFT], self.wind[DIR_RIGHT]
        pairs_y   = np.minimum(up, down);     pairs_x   = np.minimum(left, right)
        space_x   = 3 - np.maximum(left, right); space_y = 3 - np.maximum(up, down)
        rotate_y_to_x = np.clip(pairs_y - pairs_x, 0, space_x)
        rotate_x_to_y = np.clip(pairs_x - pairs_y, 0, space_y)
        delta = rotate_x_to_y - rotate_y_to_x
        self.wind[DIR_UP]    = up    + np.where(air, delta, 0)
        self.wind[DIR_DOWN]  = down  + np.where(air, delta, 0)
        self.wind[DIR_LEFT]  = left  - np.where(air, delta, 0)
        self.wind[DIR_RIGHT] = right - np.where(air, delta, 0)

        T = self.temperature;  T_adv = T.copy();  weight = 0.25
        for ch, (shift, axis) in enumerate([(-1,1),(1,1),(-1,0),(1,0)]):
            intensity = self.wind[ch].astype(np.float32) / 3.0
            flux = np.roll(T * intensity, shift=shift, axis=axis) * weight
            T_adv += flux;  T_adv -= T * intensity * weight

        lap   = np.roll(T,-1,axis=0)+np.roll(T,1,axis=0)+np.roll(T,-1,axis=1)+np.roll(T,1,axis=1)-4.0*T
        T_new = T_adv + 0.05 * lap
        T_new *= 0.96
        self.temperature[air] = np.clip(T_new[air], -100.0, 100.0)

        prob_hot  = np.random.random((self.w, self.h)) < (np.maximum(0,  self.temperature) / 300.0)
        prob_cold = np.random.random((self.w, self.h)) < (np.maximum(0, -self.temperature) / 300.0)
        hot  = (self.temperature >  5) & air & prob_hot
        cold = (self.temperature < -5) & air & prob_cold

        self.wind[DIR_UP][hot]    = np.clip(self.wind[DIR_UP][hot]   + 1, 0, 3)
        self.wind[DIR_DOWN][hot]  = np.maximum(0, self.wind[DIR_DOWN][hot]  - 1)
        self.wind[DIR_DOWN][cold] = np.clip(self.wind[DIR_DOWN][cold]+ 1, 0, 3)
        self.wind[DIR_UP][cold]   = np.maximum(0, self.wind[DIR_UP][cold]   - 1)

        vx = self.wind[DIR_RIGHT].astype(np.int8) - self.wind[DIR_LEFT].astype(np.int8)
        vy = self.wind[DIR_DOWN].astype(np.int8)  - self.wind[DIR_UP].astype(np.int8)
        new_tracers = np.zeros_like(self.tracers)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                mask = (np.sign(vx)==dx) & (np.sign(vy)==dy) & (self.tracers==1)
                if np.any(mask):
                    moved = np.roll(np.roll(mask, shift=dx, axis=0), shift=dy, axis=1)
                    new_tracers = np.logical_or(new_tracers, moved)
        new_tracers[walls] = 0
        self.tracers = new_tracers.astype(np.int8)

    # ── Render optimizado con surfarray ─────────────────────────────
    def render(self, surface):
        dyn = np.sum(self.wind, axis=0)  # (w,h)

        # Construir array RGB (w, h, 3) con NumPy
        r = np.zeros((self.w, self.h), dtype=np.float32)
        g = np.zeros((self.w, self.h), dtype=np.float32)
        b = np.zeros((self.w, self.h), dtype=np.float32)

        air_mask  = (self.materials == MAT_AIR)
        wall_mask = (self.materials == MAT_WALL)
        fol_mask  = (self.materials == MAT_FOLIAGE)
        sink_mask = (self.materials == MAT_SINK)

        # Materiales sólidos
        r[wall_mask] = 150;  g[wall_mask] = 155;  b[wall_mask] = 165
        r[fol_mask]  =  34;  g[fol_mask]  = 139;  b[fol_mask]  =  34
        r[sink_mask] = 139;  g[sink_mask] =   0;  b[sink_mask] =   0

        # Aire: base de presión dinámica
        base = np.minimum(255.0, dyn * 40.0)
        r = np.where(air_mask, base,       r)
        g = np.where(air_mask, base,       g)
        b = np.where(air_mask, np.minimum(255.0, base + 40.0), b)

        # Temperatura
        T  = self.temperature
        hot_air  = air_mask & (T >  1)
        cold_air = air_mask & (T < -1)

        r = np.where(hot_air,  np.minimum(255.0, r + T * 2.5),  r)
        g = np.where(hot_air,  np.maximum(0.0,   g - T * 0.5),  g)
        b = np.where(hot_air,  np.maximum(0.0,   b - T * 0.5),  b)

        aT = np.abs(T)
        b = np.where(cold_air, np.minimum(255.0, b + aT * 2.5), b)
        g = np.where(cold_air, np.minimum(255.0, g + aT * 1.5), g)
        r = np.where(cold_air, np.maximum(0.0,   r - aT * 0.5), r)

        # Presión estática
        p  = self.static_pressure.astype(np.float32)
        pp = air_mask & (p > 0);  np_ = air_mask & (p < 0);  ap = np.abs(p)
        r = np.where(pp, np.minimum(255.0, r + p  * 18), r)
        g = np.where(pp, np.maximum(0.0,   g - p  * 12), g)
        b = np.where(pp, np.minimum(255.0, b + p  * 24), b)
        r = np.where(np_, np.maximum(0.0,   r - ap * 12), r)
        g = np.where(np_, np.minimum(255.0, g + ap * 20), g)
        b = np.where(np_, np.maximum(0.0,   b - ap * 8),  b)

        # Ensamblar array (w, h, 3) -> transponer a (h, w, 3) para surfarray
        rgb = np.stack([r, g, b], axis=2).astype(np.uint8)  # (w,h,3)

        # surfarray espera (w, h) con pixel = int32 packed
        # Usamos make_surface que acepta (w,h,3)
        cell_surf = pygame.surfarray.make_surface(rgb)
        # Escalar al tamaño real de la pantalla
        scaled = pygame.transform.scale(cell_surf, (CANVAS_WIDTH, CANVAS_HEIGHT))
        surface.blit(scaled, (0, 0))

        # Trazadores (pequeños cuadrados amarillos) — pocos, loop rápido
        tracer_xs, tracer_ys = np.nonzero(self.tracers)
        for x, y in zip(tracer_xs, tracer_ys):
            px, py = x * CELL_SIZE, y * CELL_SIZE
            pygame.draw.rect(surface, (255, 230, 30), (px+2, py+2, CELL_SIZE-4, CELL_SIZE-4))

        # Vectores de viento
        for ch in range(4):
            xs, ys = np.nonzero(self.wind[ch] > 0)
            for x, y in zip(xs, ys):
                intensity = self.wind[ch, x, y]
                temp = self.temperature[x, y]
                if temp > 1:   base_color = (255, max(0, 200 - int(abs(temp)*2)), 0)
                elif temp < -1: base_color = (0, 200, 255)
                else:           base_color = (255, 255, 255)
                factor = [0.0, 0.4, 0.7, 1.0][intensity]
                color_wind = (int(base_color[0]*factor), int(base_color[1]*factor), int(base_color[2]*factor))
                cx, cy = x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2
                dx2, dy2 = 0, 0
                if ch == DIR_UP:    dy2 = -CELL_SIZE//2
                elif ch == DIR_DOWN: dy2 =  CELL_SIZE//2
                elif ch == DIR_LEFT: dx2 = -CELL_SIZE//2
                elif ch == DIR_RIGHT:dx2 =  CELL_SIZE//2
                pygame.draw.line(surface, color_wind, (cx, cy), (cx+dx2, cy+dy2), width=intensity)


# =====================================================================
# 3. INTERFAZ GRÁFICA MEJORADA (FRONTEND 2)
# =====================================================================
TOOL_META = {
    'Muro Sólido':    {'color': (150,155,165), 'tip': 'Rebota el viento'},
    'Follaje':        {'color': ( 50,180, 50), 'tip': 'Amortigua el flujo'},
    'Sumidero':       {'color': (200, 50, 50), 'tip': 'Absorbe toda energía'},
    'Borrar (Aire)':  {'color': ( 80,100,120), 'tip': 'Limpia la celda'},
    'Inyectar Viento':{'color': (  0,200,255), 'tip': 'Fuerza el flujo'},
    'Pincel Temp':    {'color': (255,140, 30), 'tip': 'Calor o frío'},
    'Pincel Presión': {'color': (180, 80,255), 'tip': 'Onda de choque'},
    'Añadir Trazador':{'color': (255,230, 30), 'tip': 'Polvo visible'},
}

def draw_tool_icon(surface, tool_name, cx, cy, color, size=9):
    """Dibuja un ícono vectorial con primitivas de Pygame según la herramienta."""
    s = size
    if tool_name == 'Muro Sólido':
        # Cuadrado sólido relleno
        pygame.draw.rect(surface, color, (cx - s//2, cy - s//2, s, s))

    elif tool_name == 'Follaje':
        # Tres círculos pequeños como hojas
        pygame.draw.circle(surface, color, (cx,     cy - 3), 4)
        pygame.draw.circle(surface, color, (cx - 4, cy + 2), 3)
        pygame.draw.circle(surface, color, (cx + 4, cy + 2), 3)
        pygame.draw.line(surface, color, (cx, cy + 3), (cx, cy + s//2 + 1), 2)

    elif tool_name == 'Sumidero':
        # Círculo con X dentro
        pygame.draw.circle(surface, color, (cx, cy), s//2, 2)
        d = s//2 - 2
        pygame.draw.line(surface, color, (cx-d, cy-d), (cx+d, cy+d), 2)
        pygame.draw.line(surface, color, (cx+d, cy-d), (cx-d, cy+d), 2)

    elif tool_name == 'Borrar (Aire)':
        # Círculo vacío (solo borde)
        pygame.draw.circle(surface, color, (cx, cy), s//2, 2)

    elif tool_name == 'Inyectar Viento':
        # Flecha apuntando a la derecha
        pygame.draw.line(surface, color, (cx - s//2, cy), (cx + s//2, cy), 2)
        pygame.draw.polygon(surface, color, [
            (cx + s//2,     cy),
            (cx + s//2 - 4, cy - 3),
            (cx + s//2 - 4, cy + 3),
        ])

    elif tool_name == 'Pincel Temp':
        # Termómetro: línea vertical + círculo abajo
        pygame.draw.line(surface, color, (cx, cy - s//2), (cx, cy + 1), 3)
        pygame.draw.circle(surface, color, (cx, cy + s//2 - 1), 4)

    elif tool_name == 'Pincel Presión':
        # Círculo con ondas concéntricas
        pygame.draw.circle(surface, color, (cx, cy), 3)
        pygame.draw.circle(surface, color, (cx, cy), 6, 1)

    elif tool_name == 'Añadir Trazador':
        # Estrella de 4 puntas
        pygame.draw.line(surface, color, (cx, cy - s//2), (cx, cy + s//2), 2)
        pygame.draw.line(surface, color, (cx - s//2, cy), (cx + s//2, cy), 2)
        d2 = s//2 - 2
        pygame.draw.line(surface, color, (cx-d2, cy-d2), (cx+d2, cy+d2), 1)
        pygame.draw.line(surface, color, (cx+d2, cy-d2), (cx-d2, cy+d2), 1)

    elif tool_name == 'Limpiar Todo':
        # Cubo de basura: rectángulo con tapa y líneas internas
        pygame.draw.rect(surface, color, (cx - 4, cy - 2, 9, 7), 2)       # cuerpo
        pygame.draw.line(surface, color, (cx - 5, cy - 2), (cx + 5, cy - 2), 2)  # tapa
        pygame.draw.line(surface, color, (cx - 2, cy - 4), (cx + 2, cy - 4), 2)  # asa
        pygame.draw.line(surface, color, (cx - 1, cy),     (cx - 1, cy + 3), 1)  # línea izq
        pygame.draw.line(surface, color, (cx + 1, cy),     (cx + 1, cy + 3), 1)  # línea der

class Slider:
    """Slider horizontal interactivo."""
    def __init__(self, x, y, w, h, val_min, val_max, value, label, fmt="{:.0f}"):
        self.rect   = pygame.Rect(x, y, w, h)
        self.min    = val_min
        self.max    = val_max
        self.value  = value
        self.label  = label
        self.fmt    = fmt
        self.dragging = False
        self.track_h  = 4
        self.thumb_r  = 7

    @property
    def track_rect(self):
        mid_y = self.rect.centery
        return pygame.Rect(self.rect.x, mid_y - self.track_h//2, self.rect.w, self.track_h)

    def thumb_x(self):
        t = (self.value - self.min) / (self.max - self.min)
        return int(self.rect.x + t * self.rect.w)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            tx = self.thumb_x()
            ty = self.rect.centery
            if abs(event.pos[0] - tx) < self.thumb_r + 4 and abs(event.pos[1] - ty) < self.thumb_r + 4:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel = (event.pos[0] - self.rect.x) / self.rect.w
            self.value = round(self.min + rel * (self.max - self.min))
            self.value = max(self.min, min(self.max, self.value))

    def draw(self, surface, active=False, font=None):
        # Label + valor
        if font:
            lbl = font.render(f"{self.label}: {self.fmt.format(self.value)}", True, C_TEXT)
            surface.blit(lbl, (self.rect.x, self.rect.y - 16))

        # Track
        tr = self.track_rect
        pygame.draw.rect(surface, C_BTN_OFF, tr, border_radius=2)
        # Fill izquierdo
        t   = (self.value - self.min) / (self.max - self.min)
        fill_w = int(t * tr.w)
        if fill_w > 0:
            fill_col = C_ACCENT if active else C_DIM
            pygame.draw.rect(surface, fill_col, (tr.x, tr.y, fill_w, tr.h), border_radius=2)

        # Thumb
        tx = self.thumb_x()
        ty = self.rect.centery
        thumb_col = C_ACCENT if active else (130, 145, 160)
        pygame.draw.circle(surface, thumb_col, (tx, ty), self.thumb_r)
        pygame.draw.circle(surface, C_BG,      (tx, ty), self.thumb_r - 3)


class SimulationGUI:
    def __init__(self, model):
        self.model = model
        self.font       = pygame.font.SysFont('Consolas', 12, bold=True)
        self.title_font = pygame.font.SysFont('Consolas', 15, bold=True)
        self.small_font = pygame.font.SysFont('Consolas', 10)

        self.simulation_running = False
        self.tools       = list(TOOL_META.keys())
        self.tools.append('Limpiar Todo')
        self.active_tool = 'Muro Sólido'

        self.val_wind  = 3
        self.val_temp  = 100
        self.val_press = 5

        self.wind_dir = 'DERECHA'
        self.dirs     = ['ARRIBA', 'DERECHA', 'ABAJO', 'IZQUIERDA']

        self.input_buffer = ""
        self.input_active = False
        self.input_box_rect = pygame.Rect(0, 0, 10, 10)  # placeholder

        self.buttons = {}
        self.sliders = {}
        self._init_ui_layout()

        self.last_gx, self.last_gy = None, None
        self.tooltip_text  = ""
        self.tooltip_timer = 0

        # FPS tracking
        self.fps_history = []
        self.last_time   = time.time()

    def _init_ui_layout(self):
        sx  = CANVAS_WIDTH + 12
        bw  = HUD_WIDTH - 24
        y   = 44  # debajo del título

        for tool in self.tools:
            self.buttons[tool] = pygame.Rect(sx, y, bw, 27)
            y += 31

        y += 6
        self.buttons['TOGGLE_SIM'] = pygame.Rect(sx, y, bw, 36)
        y += 48

        # Sliders
        slider_w = bw - 10
        self.sliders['wind']  = Slider(sx + 5, y + 16, slider_w, 28, 1,   3,   self.val_wind,  "Viento",  "{:.0f}")
        y += 46
        self.sliders['temp']  = Slider(sx + 5, y + 16, slider_w, 28, -100, 100, self.val_temp,  "Temp °C", "{:+.0f}")
        y += 46
        self.sliders['press'] = Slider(sx + 5, y + 16, slider_w, 28, -10,  10,  self.val_press, "Presión", "{:+.0f}")
        y += 54

        self.dir_panel_y = y  # para el panel de dirección

    # ── Dibujar el panel de dirección de viento ──────────────────────
    def _draw_wind_dir_panel(self, surface, px, py):
        """Brújula 3x3 con botones de dirección - íconos dibujados."""
        cell = 22
        dirs_grid = {
            'ARRIBA':    (1, 0),
            'ABAJO':     (1, 2),
            'IZQUIERDA': (0, 1),
            'DERECHA':   (2, 1),
        }

        label = self.small_font.render("DIRECCION VIENTO", True, C_DIM)
        surface.blit(label, (px, py - 14))

        self.dir_buttons = {}
        for name, (gx, gy) in dirs_grid.items():
            rx = px + gx * (cell + 2)
            ry = py + gy * (cell + 2)
            r  = pygame.Rect(rx, ry, cell, cell)
            self.dir_buttons[name] = r
            active = (self.wind_dir == name)
            col    = C_ACCENT if active else C_BTN_OFF
            pygame.draw.rect(surface, col, r, border_radius=4)
            if active:
                pygame.draw.rect(surface, C_BORDER, r, 1, border_radius=4)

            # Flecha dibujada con primitivas
            acx = rx + cell//2;  acy = ry + cell//2
            ac  = (255,255,255) if active else C_DIM
            hw = 5  # half-width de la flecha
            if name == 'ARRIBA':
                pygame.draw.polygon(surface, ac, [(acx, acy-hw),(acx-hw,acy+hw//2),(acx+hw,acy+hw//2)])
            elif name == 'ABAJO':
                pygame.draw.polygon(surface, ac, [(acx, acy+hw),(acx-hw,acy-hw//2),(acx+hw,acy-hw//2)])
            elif name == 'IZQUIERDA':
                pygame.draw.polygon(surface, ac, [(acx-hw, acy),(acx+hw//2,acy-hw),(acx+hw//2,acy+hw)])
            elif name == 'DERECHA':
                pygame.draw.polygon(surface, ac, [(acx+hw, acy),(acx-hw//2,acy-hw),(acx-hw//2,acy+hw)])

        # celda central
        cx = px + 1*(cell+2);  cy = py + 1*(cell+2)
        pygame.draw.rect(surface, (30,38,48), pygame.Rect(cx,cy,cell,cell), border_radius=4)
        pygame.draw.circle(surface, C_DIM, (cx+cell//2, cy+cell//2), 2)

    # ── Dibujar barra FPS ────────────────────────────────────────────
    def _draw_fps_bar(self, surface, x, y, w):
        now = time.time()
        dt  = now - self.last_time;  self.last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 40: self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        lbl  = self.small_font.render(f"FPS  {avg_fps:5.1f}", True, C_DIM)
        surface.blit(lbl, (x, y))

        # mini sparkline
        bar_x = x + 70;  bar_y = y;  bar_w = w - 70;  bar_h = 12
        pygame.draw.rect(surface, C_BTN_OFF, (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        if len(self.fps_history) > 1:
            max_fps = max(self.fps_history) or 1
            step    = bar_w / len(self.fps_history)
            for i, f in enumerate(self.fps_history):
                bh = int((f / max_fps) * bar_h)
                col = C_GREEN if f >= 25 else C_YELLOW if f >= 15 else C_RED
                pygame.draw.rect(surface, col,
                                 (int(bar_x + i * step), bar_y + bar_h - bh,
                                  max(1, int(step)), bh))

    # ── Draw info de celda bajo el cursor ────────────────────────────
    def _draw_cell_info(self, surface, x, y, w):
        mx, my = pygame.mouse.get_pos()
        if mx >= CANVAS_WIDTH:
            return
        gx = mx // CELL_SIZE;  gy = my // CELL_SIZE
        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            return
        m   = self.model
        mat_names = {MAT_AIR:'Aire', MAT_WALL:'Muro', MAT_FOLIAGE:'Follaje', MAT_SINK:'Sumidero'}
        mat = mat_names.get(m.materials[gx, gy], '?')
        t   = m.temperature[gx, gy]
        p   = m.static_pressure[gx, gy]
        dyn = int(np.sum(m.wind[:, gx, gy]))
        info = f"[{gx},{gy}]  {mat}  T:{t:+.0f}  P:{p:+d}  v:{dyn}"
        surf = self.small_font.render(info, True, C_ACCENT)
        surface.blit(surf, (x, y))

    # ── HUD principal ────────────────────────────────────────────────
    def draw_hud(self, surface):
        # Fondo del panel
        hud_rect = pygame.Rect(CANVAS_WIDTH, 0, HUD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(surface, C_BG, hud_rect)
        # Borde izquierdo luminoso
        pygame.draw.line(surface, C_BORDER, (CANVAS_WIDTH, 0), (CANVAS_WIDTH, SCREEN_HEIGHT), 2)

        sx = CANVAS_WIDTH + 12
        bw = HUD_WIDTH - 24

        # ── Título ────────────────────────────────────────────────────
        title = self.title_font.render("LGA  SIMULADOR", True, C_ACCENT)
        surface.blit(title, (sx + 10, 12))
        pygame.draw.line(surface, (30, 40, 55), (sx, 34), (sx + bw, 34))

        # ── Botones de herramienta ─────────────────────────────────────
        mx, my = pygame.mouse.get_pos()
        for tool, rect in self.buttons.items():
            if tool == 'TOGGLE_SIM':
                continue
            is_clear = (tool == 'Limpiar Todo')
            meta   = TOOL_META.get(tool, {'color': (200, 60, 60), 'tip': 'Limpia toda la pantalla'})
            active = (self.active_tool == tool)
            hover  = rect.collidepoint(mx, my)
            tool_color = (200, 60, 60) if is_clear else meta['color']

            # Fondo
            if is_clear:
                col = (100, 30, 30) if hover else (70, 20, 20)
            elif active:
                col = C_BTN_ON
            elif hover:
                col = (50, 62, 78)
            else:
                col = C_BTN_OFF
            pygame.draw.rect(surface, col, rect, border_radius=4)

            # Borde izquierdo de color si activo o hover en limpiar
            if active or (is_clear and hover):
                pygame.draw.rect(surface, tool_color,
                                 pygame.Rect(rect.x, rect.y, 4, rect.h), border_radius=2)
                pygame.draw.rect(surface, tool_color, rect, 1, border_radius=4)

            # Ícono vectorial centrado en el área izquierda del botón
            icon_cx = rect.x + 14
            icon_cy = rect.centery
            draw_tool_icon(surface, tool, icon_cx, icon_cy, tool_color)

            # Texto
            txt_col = (255, 255, 255) if (active or is_clear) else C_TEXT
            txt     = self.font.render(tool, True, txt_col)
            surface.blit(txt, (rect.x + 28, rect.y + 7))

            # Tooltip al hover
            tip = 'Limpia toda la pantalla' if is_clear else meta.get('tip', '')
            if hover and not active and tip:
                self.tooltip_text  = tip
                self.tooltip_timer = 60

        # ── Botón TOGGLE ──────────────────────────────────────────────
        sim_rect = self.buttons['TOGGLE_SIM']
        if self.simulation_running:
            sim_col = (160, 40, 40);  sim_txt = "DETENER  [ESPACIO]"
        else:
            sim_col = (30, 130, 60);  sim_txt = "INICIAR  [ESPACIO]"
        pygame.draw.rect(surface, sim_col, sim_rect, border_radius=6)
        if sim_rect.collidepoint(mx, my):
            pygame.draw.rect(surface, (255,255,255), sim_rect, 1, border_radius=6)
        # Ícono stop/play dibujado con primitivas
        ix = sim_rect.x + 14;  iy = sim_rect.centery
        if self.simulation_running:
            pygame.draw.rect(surface, (255,255,255), (ix-4, iy-5, 4, 10))
            pygame.draw.rect(surface, (255,255,255), (ix+2, iy-5, 4, 10))
        else:
            pygame.draw.polygon(surface, (255,255,255), [(ix-3,iy-5),(ix-3,iy+5),(ix+7,iy)])
        ss = self.title_font.render(sim_txt, True, (255, 255, 255))
        surface.blit(ss, (sim_rect.x + 28,
                          sim_rect.y + sim_rect.h//2 - ss.get_height()//2))

        # ── Sliders ───────────────────────────────────────────────────
        wind_active  = (self.active_tool == 'Inyectar Viento')
        temp_active  = (self.active_tool == 'Pincel Temp')
        press_active = (self.active_tool == 'Pincel Presión')

        self.sliders['wind'].value  = self.val_wind
        self.sliders['temp'].value  = self.val_temp
        self.sliders['press'].value = self.val_press

        self.sliders['wind'].draw(surface,  wind_active,  self.small_font)
        self.sliders['temp'].draw(surface,  temp_active,  self.small_font)
        self.sliders['press'].draw(surface, press_active, self.small_font)

        # ── Panel de dirección ────────────────────────────────────────
        dir_x = sx + 5
        dir_y = self.dir_panel_y
        self._draw_wind_dir_panel(surface, dir_x, dir_y)

        # ── Separador ────────────────────────────────────────────────
        sep_y = dir_y + 3 * 24 + 14
        pygame.draw.line(surface, (30, 40, 55), (sx, sep_y), (sx + bw, sep_y))

        # ── Input de valor exacto ─────────────────────────────────────
        input_y = sep_y + 8
        lbl = self.small_font.render("Valor exacto  [Enter confirma]", True, C_DIM)
        surface.blit(lbl, (sx, input_y))
        input_y += 14
        self.input_box_rect = pygame.Rect(sx, input_y, bw, 22)
        box_col = C_ACCENT if self.input_active else (55, 65, 80)
        pygame.draw.rect(surface, (18, 22, 30), self.input_box_rect)
        pygame.draw.rect(surface, box_col, self.input_box_rect, 1, border_radius=3)
        cursor_str = "_" if self.input_active else ""
        txt_s = self.font.render(self.input_buffer + cursor_str, True, (255, 255, 255))
        surface.blit(txt_s, (self.input_box_rect.x + 6, self.input_box_rect.y + 4))

        # ── Estado del motor ──────────────────────────────────────────
        state_y = input_y + 28
        state_txt = "● ITERANDO" if self.simulation_running else "● CONFIG"
        state_col = C_RED if self.simulation_running else C_GREEN
        ss2 = self.small_font.render(state_txt, True, state_col)
        surface.blit(ss2, (sx, state_y))

        hint = self.small_font.render("[R] Reset fluido", True, C_DIM)
        surface.blit(hint, (sx + 90, state_y))

        # ── FPS sparkline ─────────────────────────────────────────────
        fps_y = state_y + 16
        self._draw_fps_bar(surface, sx, fps_y, bw)

        # ── Info celda bajo cursor ────────────────────────────────────
        info_y = fps_y + 18
        self._draw_cell_info(surface, sx, info_y, bw)

        # ── Tooltip flotante ──────────────────────────────────────────
        if self.tooltip_timer > 0 and self.tooltip_text:
            self.tooltip_timer -= 1
            tip_surf = self.small_font.render(f"  {self.tooltip_text}  ", True, (0,0,0))
            tip_rect = pygame.Rect(mx + 12, my - 20, tip_surf.get_width() + 4, tip_surf.get_height() + 4)
            if tip_rect.right > SCREEN_WIDTH:
                tip_rect.right = SCREEN_WIDTH - 2
            pygame.draw.rect(surface, C_ACCENT, tip_rect, border_radius=3)
            surface.blit(tip_surf, (tip_rect.x + 2, tip_rect.y + 2))

    # ── Cursor fantasma ───────────────────────────────────────────────
    def draw_cursor(self, surface):
        mx, my = pygame.mouse.get_pos()
        if mx < CANVAS_WIDTH:
            gx, gy = mx // CELL_SIZE, my // CELL_SIZE
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                col = TOOL_META.get(self.active_tool, {}).get('color', (255,255,255))
                rect = (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, col, rect, width=1)

    # ── Teclado ───────────────────────────────────────────────────────
    def handle_keydown(self, event):
        key = event.key
        if self.input_active:
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                try:
                    val = int(self.input_buffer)
                    if self.active_tool == 'Inyectar Viento':
                        self.val_wind  = max(1, min(3, val));   self.sliders['wind'].value  = self.val_wind
                    elif self.active_tool == 'Pincel Temp':
                        self.val_temp  = max(-100, min(100, val)); self.sliders['temp'].value  = self.val_temp
                    elif self.active_tool == 'Pincel Presión':
                        self.val_press = max(-10, min(10, val)); self.sliders['press'].value = self.val_press
                except ValueError:
                    pass
                self.input_buffer = ""; self.input_active = False
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
        elif key == pygame.K_DELETE:
            self.model.reset_all()
            self.simulation_running = False
        elif not self.simulation_running:
            if key == pygame.K_q:
                if self.active_tool == 'Inyectar Viento': self.val_wind  = max(1,    self.val_wind  - 1)
                elif self.active_tool == 'Pincel Temp':   self.val_temp  = max(-100, self.val_temp  - 10)
                elif self.active_tool == 'Pincel Presión':self.val_press = max(-10,  self.val_press - 1)
            elif key == pygame.K_w:
                if self.active_tool == 'Inyectar Viento': self.val_wind  = min(3,    self.val_wind  + 1)
                elif self.active_tool == 'Pincel Temp':   self.val_temp  = min(100,  self.val_temp  + 10)
                elif self.active_tool == 'Pincel Presión':self.val_press = min(10,   self.val_press + 1)
            elif key == pygame.K_UP:    self.wind_dir = 'ARRIBA'
            elif key == pygame.K_DOWN:  self.wind_dir = 'ABAJO'
            elif key == pygame.K_LEFT:  self.wind_dir = 'IZQUIERDA'
            elif key == pygame.K_RIGHT: self.wind_dir = 'DERECHA'
            elif key == pygame.K_1: self.val_wind = 1
            elif key == pygame.K_2: self.val_wind = 2
            elif key == pygame.K_3: self.val_wind = 3
            elif key == pygame.K_c: self.val_temp = -100
            elif key == pygame.K_h: self.val_temp = 100

    # ── Click del ratón ───────────────────────────────────────────────
    def handle_click(self, event):
        if event.button != 1:
            return
        # Input box
        if self.input_box_rect.collidepoint(event.pos):
            self.input_active = True;  return
        self.input_active = False

        # Botones de herramienta y toggle
        for tool, rect in self.buttons.items():
            if rect.collidepoint(event.pos):
                if tool == 'TOGGLE_SIM':
                    self.simulation_running = not self.simulation_running
                elif tool == 'Limpiar Todo':
                    self.model.reset_all()
                    self.simulation_running = False
                else:
                    self.active_tool = tool
                return

        # Botones de dirección
        if hasattr(self, 'dir_buttons'):
            for name, rect in self.dir_buttons.items():
                if rect.collidepoint(event.pos):
                    self.wind_dir = name;  return

        # Sliders (click directo)
        for key, slider in self.sliders.items():
            if slider.rect.inflate(0, 20).collidepoint(event.pos):
                slider.handle_event(event)
                self._sync_sliders()
                return

    def handle_mouse_drag(self):
        # Actualizar sliders con drag
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:
            for slider in self.sliders.values():
                if slider.dragging:
                    self._sync_sliders()

        if self.simulation_running:
            return

        mx, my = pygame.mouse.get_pos()
        if mx < CANVAS_WIDTH:
            gx = max(0, min(mx // CELL_SIZE, GRID_W - 1))
            gy = max(0, min(my // CELL_SIZE, GRID_H - 1))
            if any(mouse_buttons):
                points = self.get_bresenham_line(self.last_gx, self.last_gy, gx, gy) if self.last_gx is not None else [(gx, gy)]
                for px, py in points:
                    if 0 <= px < GRID_W and 0 <= py < GRID_H:
                        if mouse_buttons[0]: self._apply_tool(px, py)
                        elif mouse_buttons[2]: self.model.clear_cell(px, py)
                self.last_gx, self.last_gy = gx, gy
            else:
                self.last_gx, self.last_gy = None, None

    def _sync_sliders(self):
        self.val_wind  = int(self.sliders['wind'].value)
        self.val_temp  = int(self.sliders['temp'].value)
        self.val_press = int(self.sliders['press'].value)

    def handle_slider_events(self, event):
        """Propagar eventos a los sliders."""
        for slider in self.sliders.values():
            slider.handle_event(event)
        self._sync_sliders()

    def _apply_tool(self, px, py):
        if self.active_tool == 'Muro Sólido':     self.model.set_material(px, py, MAT_WALL)
        elif self.active_tool == 'Follaje':        self.model.set_material(px, py, MAT_FOLIAGE)
        elif self.active_tool == 'Sumidero':       self.model.set_material(px, py, MAT_SINK)
        elif self.active_tool == 'Borrar (Aire)':  self.model.clear_cell(px, py)
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
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
        return points


# =====================================================================
# 4. CICLO PRINCIPAL
# =====================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lattice Gas Automata  v3")
    clock = pygame.time.Clock()

    model = TensorModel(GRID_W, GRID_H)
    gui   = SimulationGUI(model)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gui.handle_click(event)
                gui.handle_slider_events(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                gui.handle_slider_events(event)
            elif event.type == pygame.MOUSEMOTION:
                gui.handle_slider_events(event)
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

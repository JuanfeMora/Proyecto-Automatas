"""
gui_pyqt6.py — Interfaz gráfica PyQt6 para el simulador de fluidos
Requiere: pip install PyQt6 numpy
Uso: python gui_pyqt6.py
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QButtonGroup,
    QRadioButton, QStatusBar, QSizePolicy, QFrame, QScrollArea,
    QToolButton, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QCursor,
    QIcon, QFont, QPalette, QKeySequence, QShortcut
)

# =====================================================================
# IMPORTAR EL MODELO (mismo archivo o import separado)
# =====================================================================
# from tensor_model import TensorModel, MAT_AIR, MAT_WALL, ...
# Por ahora lo definimos inline para que el archivo sea autónomo:

MAT_AIR, MAT_WALL, MAT_FOLIAGE, MAT_SINK = 0, 1, 2, 3
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT     = 0, 1, 2, 3
GRID_W, GRID_H = 80, 60


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
        self.materials[x, y]        = MAT_AIR
        self.static_pressure[x, y]  = 0
        self.temperature[x, y]      = 0.0
        self.tracers[x, y]          = 0
        self.wind[:, x, y]          = 0

    def reset_fluid(self):
        self.wind.fill(0)
        self.temperature.fill(0)
        self.tracers.fill(0)

    def update_step(self):
        air = (self.materials == MAT_AIR)

        p_up    = np.roll(self.static_pressure,  1, axis=1)
        p_down  = np.roll(self.static_pressure, -1, axis=1)
        p_left  = np.roll(self.static_pressure,  1, axis=0)
        p_right = np.roll(self.static_pressure, -1, axis=0)

        mask_up    = air & (self.static_pressure - p_up    >= 2)
        mask_down  = air & (self.static_pressure - p_down  >= 2)
        mask_left  = air & (self.static_pressure - p_left  >= 2)
        mask_right = air & (self.static_pressure - p_right >= 2)

        mask_up[:, 0]     = False
        mask_down[:, -1]  = False
        mask_left[0, :]   = False
        mask_right[-1, :] = False

        self.wind[DIR_UP][mask_up]       = np.clip(self.wind[DIR_UP][mask_up]       + 1, 0, 3)
        self.wind[DIR_DOWN][mask_down]   = np.clip(self.wind[DIR_DOWN][mask_down]   + 1, 0, 3)
        self.wind[DIR_LEFT][mask_left]   = np.clip(self.wind[DIR_LEFT][mask_left]   + 1, 0, 3)
        self.wind[DIR_RIGHT][mask_right] = np.clip(self.wind[DIR_RIGHT][mask_right] + 1, 0, 3)

        new_wind = np.zeros_like(self.wind)
        new_wind[DIR_UP]    = np.roll(self.wind[DIR_UP],    -1, axis=1)
        new_wind[DIR_DOWN]  = np.roll(self.wind[DIR_DOWN],   1, axis=1)
        new_wind[DIR_LEFT]  = np.roll(self.wind[DIR_LEFT],  -1, axis=0)
        new_wind[DIR_RIGHT] = np.roll(self.wind[DIR_RIGHT],  1, axis=0)

        out_up    = new_wind[DIR_UP,    :, -1].copy()
        out_down  = new_wind[DIR_DOWN,  :,  0].copy()
        out_left  = new_wind[DIR_LEFT,  -1, :].copy()
        out_right = new_wind[DIR_RIGHT,  0, :].copy()

        new_wind[DIR_UP,    :, -1] = 0
        new_wind[DIR_DOWN,  :,  0] = 0
        new_wind[DIR_LEFT,  -1, :] = 0
        new_wind[DIR_RIGHT,  0, :] = 0

        new_wind[DIR_DOWN,  :,  0] = np.maximum(new_wind[DIR_DOWN,  :,  0], out_up)
        new_wind[DIR_UP,    :, -1] = np.maximum(new_wind[DIR_UP,    :, -1], out_down)
        new_wind[DIR_RIGHT,  0, :] = np.maximum(new_wind[DIR_RIGHT,  0, :], out_left)
        new_wind[DIR_LEFT,  -1, :] = np.maximum(new_wind[DIR_LEFT,  -1, :], out_right)

        walls = (self.materials == MAT_WALL)
        new_wind[DIR_UP][walls]    = new_wind[DIR_DOWN].copy()[walls]
        new_wind[DIR_DOWN][walls]  = new_wind[DIR_UP].copy()[walls]
        new_wind[DIR_LEFT][walls]  = new_wind[DIR_RIGHT].copy()[walls]
        new_wind[DIR_RIGHT][walls] = new_wind[DIR_LEFT].copy()[walls]

        foliage = (self.materials == MAT_FOLIAGE)
        new_wind[:, foliage] = np.maximum(0, new_wind[:, foliage] - 1)

        sinks = (self.materials == MAT_SINK)
        new_wind[:, sinks]          = 0
        self.static_pressure[sinks] = 0
        self.wind = new_wind

        friction_mask = np.random.random((self.w, self.h)) < 0.02
        self.wind = np.where((self.wind > 0) & friction_mask, self.wind - 1, self.wind).astype(np.int8)

        vacuum = (self.static_pressure < 0)
        if np.any(vacuum):
            self.wind[:, vacuum] = np.maximum(0, self.wind[:, vacuum] - 1)

        pos_p = self.static_pressure > 0
        neg_p = self.static_pressure < 0
        self.static_pressure[pos_p] = np.maximum(0, self.static_pressure[pos_p] - 2)
        self.static_pressure[neg_p] = np.minimum(0, self.static_pressure[neg_p] + 2)

        air = (self.materials == MAT_AIR)
        up, down, left, right = self.wind[DIR_UP], self.wind[DIR_DOWN], self.wind[DIR_LEFT], self.wind[DIR_RIGHT]
        pairs_y = np.minimum(up, down)
        pairs_x = np.minimum(left, right)
        space_x = 3 - np.maximum(left, right)
        space_y = 3 - np.maximum(up, down)
        rotate_y_to_x = np.clip(pairs_y - pairs_x, 0, space_x)
        rotate_x_to_y = np.clip(pairs_x - pairs_y, 0, space_y)
        delta = rotate_x_to_y - rotate_y_to_x
        self.wind[DIR_UP]    = up    + np.where(air, delta, 0)
        self.wind[DIR_DOWN]  = down  + np.where(air, delta, 0)
        self.wind[DIR_LEFT]  = left  - np.where(air, delta, 0)
        self.wind[DIR_RIGHT] = right - np.where(air, delta, 0)

        T = self.temperature
        T_adv = T.copy()
        weight = 0.25
        for ch, (shift, axis) in enumerate([(-1, 1), (1, 1), (-1, 0), (1, 0)]):
            intensity = self.wind[ch].astype(np.float32) / 3.0
            flux = np.roll(T * intensity, shift=shift, axis=axis) * weight
            T_adv += flux
            T_adv -= T * intensity * weight
        lap = (np.roll(T, -1, axis=0) + np.roll(T, 1, axis=0) +
               np.roll(T, -1, axis=1) + np.roll(T,  1, axis=1) - 4.0 * T)
        T_new = T_adv + 0.05 * lap
        T_new *= 0.96
        self.temperature[air] = np.clip(T_new[air], -100.0, 100.0)

        prob_hot  = np.random.random((self.w, self.h)) < (np.maximum(0,  self.temperature) / 300.0)
        prob_cold = np.random.random((self.w, self.h)) < (np.maximum(0, -self.temperature) / 300.0)
        hot  = (self.temperature >  5) & air & prob_hot
        cold = (self.temperature < -5) & air & prob_cold
        self.wind[DIR_UP][hot]    = np.clip(self.wind[DIR_UP][hot]   + 1, 0, 3)
        self.wind[DIR_DOWN][hot]  = np.maximum(0, self.wind[DIR_DOWN][hot]  - 1)
        self.wind[DIR_DOWN][cold] = np.clip(self.wind[DIR_DOWN][cold] + 1, 0, 3)
        self.wind[DIR_UP][cold]   = np.maximum(0, self.wind[DIR_UP][cold]   - 1)

        vx = self.wind[DIR_RIGHT].astype(np.int8) - self.wind[DIR_LEFT].astype(np.int8)
        vy = self.wind[DIR_DOWN].astype(np.int8)  - self.wind[DIR_UP].astype(np.int8)
        new_tracers = np.zeros_like(self.tracers)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                mask = (np.sign(vx) == dx) & (np.sign(vy) == dy) & (self.tracers == 1)
                if np.any(mask):
                    moved = np.roll(np.roll(mask, shift=dx, axis=0), shift=dy, axis=1)
                    new_tracers = np.logical_or(new_tracers, moved)
        new_tracers[walls] = 0
        self.tracers = new_tracers.astype(np.int8)


# =====================================================================
# CANVAS DE SIMULACIÓN (reemplaza el render de pygame)
# =====================================================================
class SimulationCanvas(QWidget):
    """
    Widget central que muestra la cuadrícula.
    Convierte el array numpy a QImage en cada frame — sin iterar celda a celda.
    Soporta zoom (rueda) y paneo (clic medio / arrastrar con espacio).
    """

    cellPainted = pyqtSignal(int, int, int)   # gx, gy, button (1=izq, 3=der)

    CELL = 10   # tamaño base de celda en px

    def __init__(self, model: TensorModel, parent=None):
        super().__init__(parent)
        self.model  = model
        self.zoom   = 1.0          # factor de zoom actual
        self.offset = [0, 0]       # desplazamiento de paneo en px

        self._pan_active = False
        self._pan_start  = None

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Buffer RGB para no realocarlo cada frame
        self._rgb = np.zeros((model.h, model.w, 3), dtype=np.uint8)

    # ── Renderizado numpy → QPixmap ─────────────────────────────────
    def _build_frame(self) -> QPixmap:
        m = self.model
        rgb = self._rgb

        dyn = np.sum(m.wind, axis=0).T          # (h, w)
        mat = m.materials.T
        temp = m.temperature.T
        sp   = m.static_pressure.T.astype(np.int32)

        # Base: aire
        base = np.clip(dyn * 40, 0, 255).astype(np.uint8)
        rgb[:, :, 0] = base
        rgb[:, :, 1] = base
        rgb[:, :, 2] = np.clip(base.astype(np.int32) + 40, 0, 255).astype(np.uint8)

        # Temperatura (rojo=calor, cian=frío)
        hot_mask  = temp > 1
        cold_mask = temp < -1
        rgb[:, :, 0] = np.where(hot_mask,  np.clip(rgb[:,:,0].astype(np.int32) + temp * 2.5, 0, 255), rgb[:,:,0]).astype(np.uint8)
        rgb[:, :, 1] = np.where(hot_mask,  np.clip(rgb[:,:,1].astype(np.int32) - temp * 0.5, 0, 255), rgb[:,:,1]).astype(np.uint8)
        rgb[:, :, 2] = np.where(hot_mask,  np.clip(rgb[:,:,2].astype(np.int32) - temp * 0.5, 0, 255), rgb[:,:,2]).astype(np.uint8)
        abs_c = np.abs(temp)
        rgb[:, :, 2] = np.where(cold_mask, np.clip(rgb[:,:,2].astype(np.int32) + abs_c * 2.5, 0, 255), rgb[:,:,2]).astype(np.uint8)
        rgb[:, :, 1] = np.where(cold_mask, np.clip(rgb[:,:,1].astype(np.int32) + abs_c * 1.5, 0, 255), rgb[:,:,1]).astype(np.uint8)
        rgb[:, :, 0] = np.where(cold_mask, np.clip(rgb[:,:,0].astype(np.int32) - abs_c * 0.5, 0, 255), rgb[:,:,0]).astype(np.uint8)

        # Presión estática (púrpura=alta, verde=vacío)
        pos_p = sp > 0
        neg_p = sp < 0
        abs_p = np.abs(sp)
        rgb[:, :, 0] = np.where(pos_p, np.clip(rgb[:,:,0].astype(np.int32) + sp * 18,   0, 255), rgb[:,:,0]).astype(np.uint8)
        rgb[:, :, 1] = np.where(pos_p, np.clip(rgb[:,:,1].astype(np.int32) - sp * 12,   0, 255), rgb[:,:,1]).astype(np.uint8)
        rgb[:, :, 2] = np.where(pos_p, np.clip(rgb[:,:,2].astype(np.int32) + sp * 24,   0, 255), rgb[:,:,2]).astype(np.uint8)
        rgb[:, :, 0] = np.where(neg_p, np.clip(rgb[:,:,0].astype(np.int32) - abs_p * 12, 0, 255), rgb[:,:,0]).astype(np.uint8)
        rgb[:, :, 1] = np.where(neg_p, np.clip(rgb[:,:,1].astype(np.int32) + abs_p * 20, 0, 255), rgb[:,:,1]).astype(np.uint8)
        rgb[:, :, 2] = np.where(neg_p, np.clip(rgb[:,:,2].astype(np.int32) - abs_p * 8,  0, 255), rgb[:,:,2]).astype(np.uint8)

        # Materiales sólidos
        rgb[mat == MAT_WALL,    :] = [150, 150, 150]
        rgb[mat == MAT_FOLIAGE, :] = [34,  139,  34]
        rgb[mat == MAT_SINK,    :] = [139,   0,   0]

        # Trazadores (amarillo brillante)
        tr = m.tracers.T.astype(bool)
        rgb[tr, :] = [255, 230, 0]

        # Construir QImage directamente desde el buffer contiguo
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

        # Escalar al tamaño de celda × zoom
        cell = max(1, int(self.CELL * self.zoom))
        scaled = img.scaled(
            w * cell, h * cell,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        return QPixmap.fromImage(scaled)

    def paintEvent(self, event):
        pix = self._build_frame()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 12, 16))
        painter.drawPixmap(self.offset[0], self.offset[1], pix)

        # Grid overlay (solo cuando zoom ≥ 1.5)
        if self.zoom >= 1.5:
            cell = int(self.CELL * self.zoom)
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            for gx in range(self.model.w + 1):
                x = self.offset[0] + gx * cell
                painter.drawLine(x, self.offset[1], x, self.offset[1] + self.model.h * cell)
            for gy in range(self.model.h + 1):
                y = self.offset[1] + gy * cell
                painter.drawLine(self.offset[0], y, self.offset[0] + self.model.w * cell, y)

        painter.end()

    # ── Coordenadas pantalla → celda ────────────────────────────────
    def screen_to_grid(self, sx, sy):
        cell = max(1, int(self.CELL * self.zoom))
        gx = (sx - self.offset[0]) // cell
        gy = (sy - self.offset[1]) // cell
        if 0 <= gx < self.model.w and 0 <= gy < self.model.h:
            return int(gx), int(gy)
        return None, None

    # ── Eventos de ratón ────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = True
            self._pan_start  = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            gx, gy = self.screen_to_grid(int(event.position().x()), int(event.position().y()))
            if gx is not None:
                btn = 1 if event.button() == Qt.MouseButton.LeftButton else 3
                self.cellPainted.emit(gx, gy, btn)
        self._last_mouse = event.position().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._pan_active:
            delta = pos - self._pan_start
            self.offset[0] += delta.x()
            self.offset[1] += delta.y()
            self._pan_start = pos
            self.update()
        elif event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            gx, gy = self.screen_to_grid(pos.x(), pos.y())
            if gx is not None:
                btn = 1 if event.buttons() & Qt.MouseButton.LeftButton else 3
                self.cellPainted.emit(gx, gy, btn)
        self._last_mouse = pos

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self.setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        # Zoom centrado en la posición del cursor
        mx = int(event.position().x())
        my = int(event.position().y())
        old_zoom = self.zoom
        self.zoom = max(0.3, min(8.0, self.zoom * factor))
        scale = self.zoom / old_zoom
        self.offset[0] = int(mx - scale * (mx - self.offset[0]))
        self.offset[1] = int(my - scale * (my - self.offset[1]))
        self.update()


# =====================================================================
# PANEL LATERAL (herramientas + parámetros)
# =====================================================================
class ToolPanel(QWidget):
    """Panel derecho con grupos colapsables de herramientas y parámetros."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setObjectName("ToolPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Herramientas de pintura ──────────────────────────────────
        grp_tools = QGroupBox("Herramienta activa")
        grp_tools.setObjectName("PanelGroup")
        lay_tools = QVBoxLayout(grp_tools)
        lay_tools.setSpacing(4)

        self.tool_group = QButtonGroup(self)
        tools = [
            ("Muro sólido",    "wall"),
            ("Follaje",        "foliage"),
            ("Sumidero",       "sink"),
            ("Borrar (aire)",  "clear"),
            ("Inyectar viento","wind"),
            ("Temperatura",    "temp"),
            ("Presión",        "pressure"),
            ("Trazador",       "tracer"),
        ]
        self.tool_buttons = {}
        for label, key in tools:
            btn = QRadioButton(label)
            btn.setProperty("toolKey", key)
            self.tool_group.addButton(btn)
            lay_tools.addWidget(btn)
            self.tool_buttons[key] = btn
        self.tool_buttons["wall"].setChecked(True)
        root.addWidget(grp_tools)

        # ── Dirección del viento ────────────────────────────────────
        grp_dir = QGroupBox("Dirección del viento")
        grp_dir.setObjectName("PanelGroup")
        lay_dir = QVBoxLayout(grp_dir)

        self.dir_group = QButtonGroup(self)
        dirs = [("↑  Arriba", DIR_UP), ("↓  Abajo", DIR_DOWN),
                ("←  Izquierda", DIR_LEFT), ("→  Derecha", DIR_RIGHT)]
        self.dir_buttons = {}
        for label, key in dirs:
            btn = QRadioButton(label)
            self.dir_group.addButton(btn)
            lay_dir.addWidget(btn)
            self.dir_buttons[key] = btn
        self.dir_buttons[DIR_RIGHT].setChecked(True)
        root.addWidget(grp_dir)

        # ── Parámetros numéricos ────────────────────────────────────
        grp_params = QGroupBox("Parámetros")
        grp_params.setObjectName("PanelGroup")
        lay_params = QVBoxLayout(grp_params)
        lay_params.setSpacing(6)

        # Intensidad viento
        lay_params.addWidget(QLabel("Intensidad viento (1–3):"))
        self.spin_wind = QSpinBox()
        self.spin_wind.setRange(1, 3)
        self.spin_wind.setValue(3)
        lay_params.addWidget(self.spin_wind)

        # Temperatura
        lay_params.addWidget(QLabel("Temperatura (−100 a 100):"))
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(-100, 100)
        self.slider_temp.setValue(50)
        self.slider_temp.setTickInterval(25)
        self.slider_temp.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_temp = QLabel("50 °C")
        self.lbl_temp.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.slider_temp.valueChanged.connect(
            lambda v: self.lbl_temp.setText(f"{v} °C"))
        lay_params.addWidget(self.slider_temp)
        lay_params.addWidget(self.lbl_temp)

        # Presión
        lay_params.addWidget(QLabel("Presión estática (−10 a 10):"))
        self.slider_press = QSlider(Qt.Orientation.Horizontal)
        self.slider_press.setRange(-10, 10)
        self.slider_press.setValue(5)
        self.slider_press.setTickInterval(5)
        self.slider_press.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lbl_press = QLabel("5")
        self.lbl_press.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.slider_press.valueChanged.connect(
            lambda v: self.lbl_press.setText(str(v)))
        lay_params.addWidget(self.slider_press)
        lay_params.addWidget(self.lbl_press)

        root.addWidget(grp_params)

        # ── Controles de simulación ─────────────────────────────────
        grp_sim = QGroupBox("Simulación")
        grp_sim.setObjectName("PanelGroup")
        lay_sim = QVBoxLayout(grp_sim)
        lay_sim.setSpacing(6)

        self.btn_toggle = QPushButton("▶  Iniciar")
        self.btn_toggle.setObjectName("BtnStart")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setMinimumHeight(36)
        lay_sim.addWidget(self.btn_toggle)

        self.btn_reset = QPushButton("↺  Resetear fluido")
        self.btn_reset.setMinimumHeight(32)
        lay_sim.addWidget(self.btn_reset)

        self.btn_clear_all = QPushButton("🗑  Limpiar todo")
        self.btn_clear_all.setMinimumHeight(32)
        lay_sim.addWidget(self.btn_clear_all)

        # FPS target
        lay_sim.addWidget(QLabel("FPS objetivo:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(30)
        lay_sim.addWidget(self.spin_fps)

        root.addWidget(grp_sim)
        root.addStretch()

    # ── Accesores ────────────────────────────────────────────────────
    @property
    def active_tool(self) -> str:
        btn = self.tool_group.checkedButton()
        return btn.property("toolKey") if btn else "wall"

    @property
    def wind_direction(self) -> int:
        btn = self.dir_group.checkedButton()
        for key, b in self.dir_buttons.items():
            if b == btn:
                return key
        return DIR_RIGHT

    @property
    def wind_intensity(self) -> int:
        return self.spin_wind.value()

    @property
    def temperature_value(self) -> int:
        return self.slider_temp.value()

    @property
    def pressure_value(self) -> int:
        return self.slider_press.value()


# =====================================================================
# BARRA DE ESTADO (métricas en tiempo real)
# =====================================================================
class MetricsBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricsBar")

        self.lbl_state  = QLabel("⏸ Detenido")
        self.lbl_fps    = QLabel("FPS: —")
        self.lbl_pmax   = QLabel("P_max: —")
        self.lbl_tmax   = QLabel("T_max: —")
        self.lbl_wmax   = QLabel("V_max: —")
        self.lbl_cursor = QLabel("Celda: —, —")

        for lbl in [self.lbl_state, self.lbl_fps, self.lbl_pmax,
                    self.lbl_tmax, self.lbl_wmax, self.lbl_cursor]:
            lbl.setFixedWidth(130)
            self.addWidget(lbl)

        self._t_last = 0.0

    def update_metrics(self, model: TensorModel, running: bool, fps: float):
        self.lbl_state.setText("▶ Corriendo" if running else "⏸ Detenido")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_pmax.setText(f"P_max: {model.static_pressure.max()}")
        self.lbl_tmax.setText(f"T_max: {model.temperature.max():.1f}")
        self.lbl_wmax.setText(f"V_max: {model.wind.max()}")

    def update_cursor(self, gx, gy):
        if gx is None:
            self.lbl_cursor.setText("Celda: —, —")
        else:
            self.lbl_cursor.setText(f"Celda: {gx}, {gy}")


# =====================================================================
# VENTANA PRINCIPAL
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador de Fluidos — Lattice Gas Automata")
        self.resize(1100, 680)

        self.model   = TensorModel(GRID_W, GRID_H)
        self._frame_count = 0
        self._fps_accum   = 0.0
        self._fps_display = 0.0

        # ── Layout central ───────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self.canvas = SimulationCanvas(self.model)
        self.panel  = ToolPanel()

        h_layout.addWidget(self.canvas, stretch=1)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setObjectName("Separator")
        h_layout.addWidget(separator)
        h_layout.addWidget(self.panel)

        # ── Barra de estado ──────────────────────────────────────────
        self.metrics = MetricsBar()
        self.setStatusBar(self.metrics)

        # ── Timer de simulación ──────────────────────────────────────
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ── Conexiones ───────────────────────────────────────────────
        self.canvas.cellPainted.connect(self._on_cell_painted)
        self.panel.btn_toggle.toggled.connect(self._on_toggle_sim)
        self.panel.btn_reset.clicked.connect(self.model.reset_fluid)
        self.panel.btn_clear_all.clicked.connect(self._clear_all)
        self.panel.spin_fps.valueChanged.connect(self._update_timer_interval)

        # Actualizar cursor en la barra de estado
        self.canvas.mouseMoveEvent = self._wrap_mouse_move(self.canvas.mouseMoveEvent)

        # ── Atajos de teclado ────────────────────────────────────────
        QShortcut(QKeySequence("Space"), self).activated.connect(
            lambda: self.panel.btn_toggle.toggle())
        QShortcut(QKeySequence("R"), self).activated.connect(
            self.model.reset_fluid)
        QShortcut(QKeySequence("1"), self).activated.connect(
            lambda: self.panel.spin_wind.setValue(1))
        QShortcut(QKeySequence("2"), self).activated.connect(
            lambda: self.panel.spin_wind.setValue(2))
        QShortcut(QKeySequence("3"), self).activated.connect(
            lambda: self.panel.spin_wind.setValue(3))
        QShortcut(QKeySequence("Up"),    self).activated.connect(
            lambda: self.panel.dir_buttons[DIR_UP].setChecked(True))
        QShortcut(QKeySequence("Down"),  self).activated.connect(
            lambda: self.panel.dir_buttons[DIR_DOWN].setChecked(True))
        QShortcut(QKeySequence("Left"),  self).activated.connect(
            lambda: self.panel.dir_buttons[DIR_LEFT].setChecked(True))
        QShortcut(QKeySequence("Right"), self).activated.connect(
            lambda: self.panel.dir_buttons[DIR_RIGHT].setChecked(True))

        # ── Timer de métricas (independiente, 4 Hz) ──────────────────
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start(250)

        # ── Estilo ───────────────────────────────────────────────────
        self._apply_stylesheet()

    # ── Lógica del tick ─────────────────────────────────────────────
    def _tick(self):
        import time
        t0 = time.perf_counter()
        self.model.update_step()
        self.canvas.update()
        dt = time.perf_counter() - t0
        self._fps_accum += dt
        self._frame_count += 1

    def _refresh_metrics(self):
        if self._frame_count > 0:
            avg = self._fps_accum / self._frame_count
            self._fps_display = 1.0 / avg if avg > 0 else 0.0
            self._fps_accum   = 0.0
            self._frame_count = 0
        self.metrics.update_metrics(
            self.model,
            self.panel.btn_toggle.isChecked(),
            self._fps_display
        )

    def _on_toggle_sim(self, running: bool):
        if running:
            self.panel.btn_toggle.setText("⏸  Pausar")
            self._update_timer_interval(self.panel.spin_fps.value())
            self.timer.start()
        else:
            self.panel.btn_toggle.setText("▶  Iniciar")
            self.timer.stop()
        self.canvas.update()

    def _update_timer_interval(self, fps: int):
        self.timer.setInterval(max(1, 1000 // fps))

    def _on_cell_painted(self, gx: int, gy: int, button: int):
        if button == 3:
            self.model.clear_cell(gx, gy)
            self.canvas.update()
            return

        tool = self.panel.active_tool
        if   tool == "wall":     self.model.set_material(gx, gy, MAT_WALL)
        elif tool == "foliage":  self.model.set_material(gx, gy, MAT_FOLIAGE)
        elif tool == "sink":     self.model.set_material(gx, gy, MAT_SINK)
        elif tool == "clear":    self.model.clear_cell(gx, gy)
        elif tool == "wind":
            self.model.set_wind(gx, gy, self.panel.wind_direction, self.panel.wind_intensity)
        elif tool == "temp":
            self.model.set_temperature(gx, gy, self.panel.temperature_value)
        elif tool == "pressure":
            self.model.set_static_pressure(gx, gy, self.panel.pressure_value)
        elif tool == "tracer":
            self.model.add_tracer(gx, gy)

        if not self.panel.btn_toggle.isChecked():
            self.canvas.update()

    def _clear_all(self):
        self.model.materials.fill(0)
        self.model.reset_fluid()
        self.canvas.update()

    def _wrap_mouse_move(self, original_fn):
        """Intercepta mouseMoveEvent del canvas para actualizar la celda en status bar."""
        def wrapped(event):
            original_fn(event)
            gx, gy = self.canvas.screen_to_grid(
                int(event.position().x()), int(event.position().y()))
            self.metrics.update_cursor(gx, gy)
        return wrapped

    # ── Hoja de estilos (modo oscuro) ────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f1117;
                color: #d4d4d8;
                font-family: 'Segoe UI', 'Inter', sans-serif;
                font-size: 12px;
            }
            QGroupBox#PanelGroup {
                border: 1px solid #2a2d3a;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 6px;
                font-weight: bold;
                color: #7dd3fc;
            }
            QGroupBox#PanelGroup::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QRadioButton {
                spacing: 6px;
                color: #d4d4d8;
            }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #4b5563;
                border-radius: 7px;
                background: #1e2130;
            }
            QRadioButton::indicator:checked {
                background: #38bdf8;
                border-color: #38bdf8;
            }
            QRadioButton:hover { color: #f0f0f0; }

            QSlider::groove:horizontal {
                height: 4px;
                background: #2a2d3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px;
                margin: -5px 0;
                background: #38bdf8;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #0ea5e9; border-radius: 2px; }

            QSpinBox {
                background: #1e2130;
                border: 1px solid #2a2d3a;
                border-radius: 4px;
                padding: 2px 6px;
                color: #d4d4d8;
            }
            QSpinBox:focus { border-color: #38bdf8; }

            QPushButton {
                background: #1e2130;
                border: 1px solid #2a2d3a;
                border-radius: 5px;
                padding: 5px 10px;
                color: #d4d4d8;
            }
            QPushButton:hover   { background: #252840; border-color: #38bdf8; }
            QPushButton:pressed { background: #0ea5e9; color: #fff; }

            QPushButton#BtnStart {
                background: #064e3b;
                border-color: #059669;
                color: #6ee7b7;
                font-weight: bold;
            }
            QPushButton#BtnStart:checked {
                background: #7f1d1d;
                border-color: #dc2626;
                color: #fca5a5;
            }
            QPushButton#BtnStart:hover { opacity: 0.85; }

            QStatusBar#MetricsBar {
                background: #080a10;
                border-top: 1px solid #1e2130;
                color: #6b7280;
                font-size: 11px;
            }
            QLabel { background: transparent; }
            QFrame#Separator {
                color: #1e2130;
                max-width: 1px;
            }
            QWidget#ToolPanel { background: #0c0e18; }
        """)


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

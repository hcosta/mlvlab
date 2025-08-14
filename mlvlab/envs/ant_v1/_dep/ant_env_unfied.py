# mlvlab/envs/ant/ant_env.py
import gymnasium as gym
from gymnasium import spaces
# Importamos numpy standard para el core del env
import numpy as np
import time
import math


# Arcade se importará solo si se llama al método de renderizado.

# Intentamos importar Numba para habilitar el núcleo acelerado opcional
try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback si no hay Numba
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        # Fallback inofensivo: devuelve la función sin compilar
        def _decorator(f):
            return f

        return _decorator

# Núcleo numérico puro para el paso del entorno, optimizado con Numba.


@njit(cache=True, nogil=True)
def _step_core_numba(
    ant_x: int,
    ant_y: int,
    action: int,
    grid_size: int,
    goal_x: int,
    goal_y: int,
    obstacles_grid: np.ndarray,  # 2D uint8 (1 = obstáculo)
    reward_move: int,
    reward_goal: int,
    reward_obstacle: int,
):
    target_x, target_y = ant_x, ant_y
    if action == 0:
        target_y -= 1
    elif action == 1:
        target_y += 1
    elif action == 2:
        target_x -= 1
    elif action == 3:
        target_x += 1

    # Márgenes como paredes
    if (
        target_x < 0
        or target_x >= grid_size
        or target_y < 0
        or target_y >= grid_size
    ):
        return ant_x, ant_y, reward_obstacle, 0

    # Meta (Hormiguero)
    if target_x == goal_x and target_y == goal_y:
        return target_x, target_y, reward_goal, 1

    # Obstáculo
    if obstacles_grid[target_y, target_x] == 1:
        return ant_x, ant_y, reward_obstacle, 0

    # Movimiento normal
    return target_x, target_y, reward_move, 0


class AntGame:
    """
    Lógica del juego (estado y transición), sin dependencias de UI.
    """

    def __init__(self, grid_size: int, reward_goal: int, reward_obstacle: int, reward_move: int,
                 use_numba_core: bool = False) -> None:
        self.grid_size = int(grid_size)
        self.reward_goal = int(reward_goal)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)
        self.use_numba_core = bool(use_numba_core) and _NUMBA_AVAILABLE
        self.ant_pos = np.zeros(2, dtype=np.int32)
        self.goal_pos = np.zeros(2, dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)
        self._np_random = None
        # Añadido para el renderer Juicy: seguimiento de la última acción y colisión
        self.last_action = 3  # Derecha por defecto
        self.collided = False

    def reset(self, np_random) -> None:
        self._np_random = np_random
        self.generate_scenario(np_random)
        self.place_ant(np_random)
        self.last_action = 3
        self.collided = False

    def generate_scenario(self, np_random) -> None:
        self._np_random = np_random
        # Generar meta y obstáculos de forma determinista con la RNG recibida
        self.goal_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        self.obstacles = {
            tuple(self._np_random.integers(0, self.grid_size, size=2).tolist())
            for _ in range(self.grid_size)
        }
        while tuple(self.goal_pos.tolist()) in self.obstacles:
            self.goal_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.obstacles_grid[oy, ox] = 1

    def place_ant(self, np_random) -> None:
        self._np_random = np_random
        # Colocar hormiga en celda válida distinta de la meta y no obstáculo
        self.ant_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        while tuple(self.ant_pos.tolist()) in self.obstacles or (
            self.ant_pos[0] == self.goal_pos[0] and self.ant_pos[1] == self.goal_pos[1]
        ):
            self.ant_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

    def get_obs(self) -> np.ndarray:
        return np.array((int(self.ant_pos[0]), int(self.ant_pos[1])), dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info: dict = {}
        self.last_action = action
        self.collided = False

        # Ruta acelerada por Numba si está habilitada
        if self.use_numba_core and isinstance(self.obstacles_grid, np.ndarray):
            nx, ny, reward, terminated_int = _step_core_numba(
                ax,
                ay,
                int(action),
                int(self.grid_size),
                int(self.goal_pos[0]),
                int(self.goal_pos[1]),
                self.obstacles_grid,
                int(self.reward_move),
                int(self.reward_goal),
                int(self.reward_obstacle),
            )
            terminated = bool(terminated_int)
            # Detectar colisión: si no se movió y la recompensa es de obstáculo
            if nx == ax and ny == ay and reward == self.reward_obstacle:
                self.collided = True
            self.ant_pos[0], self.ant_pos[1] = nx, ny
            info["collided"] = self.collided
            info["terminated"] = terminated
            return np.array((nx, ny), dtype=np.int32), int(reward), bool(terminated), info

        # Ruta Python (Mantenida igual, pero actualizando collided y last_action)
        prev_ax, prev_ay = ax, ay
        target_ax, target_ay = ax, ay
        if action == 0:
            target_ay -= 1
        elif action == 1:
            target_ay += 1
        elif action == 2:
            target_ax -= 1
        elif action == 3:
            target_ax += 1

        out_of_bounds = (
            target_ax < 0 or target_ax >= self.grid_size or
            target_ay < 0 or target_ay >= self.grid_size
        )
        if out_of_bounds:
            reward = self.reward_obstacle
            ax, ay = prev_ax, prev_ay
            terminated = False
            self.collided = True
        else:
            ax, ay = target_ax, target_ay
            terminated = (
                ax == int(self.goal_pos[0]) and ay == int(self.goal_pos[1]))
            if terminated:
                reward = self.reward_goal
                self.collided = False
            elif (ax, ay) in self.obstacles:
                reward = self.reward_obstacle
                ax, ay = prev_ax, prev_ay
                self.collided = True
            else:
                reward = self.reward_move
                self.collided = False

        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = self.collided
        info["terminated"] = terminated
        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info


# =============================================================================
# JUICY ARCADE RENDERER (Versión Mejorada Visualmente)
# =============================================================================

class ArcadeRenderer:
    """
    Renderer 'Juicy' basado en Arcade. Soporta "human" y "rgb_array".
    Añade animaciones, efectos visuales y un tema naturalista.
    """

    def __init__(self) -> None:
        self.window = None
        self.game = None

        # Configuración visual (Ajustado para ser más grande y detallado)
        self.CELL_SIZE = 50
        self.WIDTH = 0
        self.HEIGHT = 0

        # Paleta de colores (Tema Naturaleza)
        self.COLOR_GRASS = (107, 142, 35)   # Verde hierba
        self.COLOR_ANT = (192, 57, 43)     # Rojo hormiga
        self.COLOR_GOAL = (40, 25, 10)     # Agujero del hormiguero
        self.COLOR_OBSTACLE = (100, 100, 100)  # Gris roca

        # Estado de animación y efectos
        self.ant_prev_pos = None
        self.ant_display_pos = None
        self.last_time = time.time()
        self.flash_time = 0.0

        # Assets y optimización
        self.initialized = False

        # Importamos arcade aquí para retrasar la carga
        self.arcade = None
        self.draw_lbwh_rectangle_filled = None

    def _lazy_import_arcade(self):
        if self.arcade is None:
            try:
                import arcade
                from arcade.draw import draw_lbwh_rectangle_filled
                self.arcade = arcade
                self.draw_lbwh_rectangle_filled = draw_lbwh_rectangle_filled
            except ImportError:
                raise ImportError(
                    "Se requiere 'arcade' para el renderizado. Instálalo con 'pip install arcade'.")

    def _initialize(self, game: AntGame, render_mode: str):
        self._lazy_import_arcade()
        self.game = game
        self.WIDTH = game.grid_size * self.CELL_SIZE
        self.HEIGHT = game.grid_size * self.CELL_SIZE

        # Crear ventana si hace falta
        if self.window is None:
            visible = render_mode == "human"
            try:
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, "Lost Ant Colony - JUICY Edition", visible=visible)
            except TypeError:
                # Fallback para versiones antiguas de Arcade
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, "Lost Ant Colony - JUICY Edition")
                if not visible:
                    try:
                        self.window.set_visible(False)
                    except Exception:
                        pass
            self.arcade.set_background_color(self.COLOR_GRASS)

        # Inicializar posición de la hormiga para la animación
        if self.ant_display_pos is None:
            # Usamos floats para la posición de visualización para permitir movimiento sub-pixel/sub-celda
            self.ant_display_pos = list(game.ant_pos.astype(float))
            self.ant_prev_pos = list(game.ant_pos.astype(float))

        self.initialized = True

    def _cell_to_pixel(self, x_cell: float, y_cell: float):
        # Convierte coordenadas de rejilla (float) a píxeles (centro de la celda)
        x_px = x_cell * self.CELL_SIZE + self.CELL_SIZE / 2
        # Arcade usa Y hacia arriba desde abajo, Gymnasium usa Y hacia abajo desde arriba. Invertimos Y.
        y_px = (self.game.grid_size - 1 - y_cell) * \
            self.CELL_SIZE + self.CELL_SIZE / 2
        return x_px, y_px

    def _update_animations(self, delta_time: float):
        # 1. Movimiento suave de la hormiga (Interpolación con Easing)
        target_pos = list(self.game.ant_pos.astype(float))

        # Detectar si el objetivo ha cambiado desde el último frame de renderizado
        if target_pos != self.ant_prev_pos:
            # Si el objetivo cambió, actualizamos la posición previa a la posición actual visible
            self.ant_prev_pos = list(self.ant_display_pos)

        # Calcular la distancia restante al objetivo actual
        dist_x = target_pos[0] - self.ant_display_pos[0]
        dist_y = target_pos[1] - self.ant_display_pos[1]
        distance = math.sqrt(dist_x**2 + dist_y**2)

        # Movimiento (Easing Exponencial - independiente del framerate)
        if distance > 0.001:
            # 1.0 - exp(-t * k). k=15 es la "velocidad".
            lerp_factor = 1.0 - math.exp(-delta_time * 15.0)

            move_x = dist_x * lerp_factor
            move_y = dist_y * lerp_factor

            self.ant_display_pos[0] += move_x
            self.ant_display_pos[1] += move_y
        else:
            # Si estamos muy cerca, ajustamos directamente al objetivo
            self.ant_display_pos = target_pos
            self.ant_prev_pos = target_pos

        # 2. Efectos (Flash)
        # Solo activamos el efecto si el juego reporta una colisión en el último paso
        if self.game.collided:
            self.flash_time = 0.1  # Duración del flash en segundos

        # Decrementar el tiempo del efecto
        self.flash_time = max(0.0, self.flash_time - delta_time)

    def _draw_static_elements(self):
        # Dibuja los obstáculos y las motas de tierra en cada frame.
        current_hash = hash(frozenset(self.game.obstacles)
                            | frozenset(tuple(self.game.goal_pos)))
        try:
            rng = np.random.default_rng(abs(current_hash) % (2**32))
        except AttributeError:
            # Fallback para versiones antiguas de numpy
            rng = np.random.RandomState(abs(current_hash) % (2**32))

        # Motas de tierra/hojarasca
        for _ in range(self.game.grid_size * self.game.grid_size * 2):
            cx = rng.uniform(0, self.WIDTH)
            cy = rng.uniform(0, self.HEIGHT)
            r = rng.uniform(1, 4)
            try:
                shade = rng.integers(-25, 25)
            except AttributeError:
                shade = rng.randint(-25, 25)

            mote_color = (max(0, min(255, self.COLOR_GRASS[0]+shade)),
                          max(0, min(255, self.COLOR_GRASS[1]+shade)),
                          max(0, min(255, self.COLOR_GRASS[2]+shade)))
            self.arcade.draw_ellipse_filled(
                cx, cy, r, r * rng.uniform(0.5, 1.0), mote_color)

        # Obstáculos (Rocas procedurales)
        for obs_x, obs_y in self.game.obstacles:
            cx, cy = self._cell_to_pixel(obs_x, obs_y)
            self._draw_rock(cx, cy, rng)

    def _draw_rock(self, cx, cy, rng):
        # Crea una roca procedural usando polígonos irregulares
        points = []
        try:
            num_points = rng.integers(7, 12)
        except AttributeError:
            num_points = rng.randint(7, 12)

        base_radius = self.CELL_SIZE * 0.45
        irregularity = self.CELL_SIZE * 0.15

        for i in range(num_points):
            angle = (math.pi * 2 * i) / num_points
            radius = base_radius + rng.uniform(-irregularity, irregularity)
            px = cx + math.cos(angle) * radius
            py = cy + math.sin(angle) * radius
            points.append((px, py))

        # Sombra
        shadow_offset_x, shadow_offset_y = 5, -5
        shadow_points = [(p[0] + shadow_offset_x, p[1] +
                          shadow_offset_y) for p in points]
        self.arcade.draw_polygon_filled(shadow_points, (50, 50, 50, 120))

        # Cuerpo principal
        try:
            shade = rng.integers(-20, 20)
        except AttributeError:
            shade = rng.randint(-20, 20)

        rock_color = (max(0, min(255, self.COLOR_OBSTACLE[0]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[1]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[2]+shade)))
        self.arcade.draw_polygon_filled(points, rock_color)

        # Highlight
        highlight_color = (min(
            255, rock_color[0]+50), min(255, rock_color[1]+50), min(255, rock_color[2]+50))
        start_highlight = int(num_points * (90/360.0))
        end_highlight = int(num_points * (180/360.0))
        highlight_points = points[start_highlight: end_highlight+1]

        if len(highlight_points) > 1:
            self.arcade.draw_line_strip(highlight_points, highlight_color, 4)

    def _draw_heatmap(self, q_table_to_render):
        # Dibuja la visualización de la Q-Table como círculos de luz
        if q_table_to_render is None:
            return

        try:
            max_q = float(np.max(q_table_to_render))
            min_q = float(np.min(q_table_to_render))
        except Exception:
            return

        if (max_q - min_q) > 1e-6:
            for state_index in range(self.game.grid_size * self.game.grid_size):
                x_cell = state_index % self.game.grid_size
                y_cell = state_index // self.game.grid_size
                cx, cy = self._cell_to_pixel(x_cell, y_cell)

                try:
                    q_value = float(np.max(q_table_to_render[state_index, :]))
                except Exception:
                    continue

                norm_q = (q_value - min_q) / (max_q - min_q)

                if norm_q < 0.33:
                    t = norm_q / 0.33
                    r, g, b = int(0*(1-t) + 139*t), int(0 *
                                                        (1-t) + 0*t), int(0*(1-t) + 139*t)
                elif norm_q < 0.66:
                    t = (norm_q - 0.33) / 0.33
                    r, g, b = int(139*(1-t) + 255*t), int(0 *
                                                          (1-t) + 165*t), int(139*(1-t) + 0*t)
                else:
                    t = (norm_q - 0.66) / 0.34
                    r, g, b = int(255*(1-t) + 255*t), int(165 *
                                                          (1-t) + 255*t), int(0*(1-t) + 220*t)

                alpha = int(50 + norm_q * 150)
                heat_color = (r, g, b, alpha)

                # Usamos círculos en lugar de cuadrados para un efecto más suave
                radius = self.CELL_SIZE * 0.15
                self.arcade.draw_circle_filled(cx, cy, radius, heat_color)

    def _draw_anthill(self):
        # Dibuja la entrada del hormiguero
        gx, gy = self.game.goal_pos
        cx, cy = self._cell_to_pixel(gx, gy)

        # Montículo de tierra alrededor del agujero (más grande)
        mound_color = (168, 129, 98)  # Color tierra
        self.arcade.draw_ellipse_filled(
            cx, cy, self.CELL_SIZE * 0.8, self.CELL_SIZE * 0.6, mound_color)
        self.arcade.draw_ellipse_outline(
            cx, cy, self.CELL_SIZE * 0.8, self.CELL_SIZE * 0.6, (0, 0, 0, 50), 2)

        # Agujero oscuro en el centro (más grande)
        hole_color = self.COLOR_GOAL
        self.arcade.draw_ellipse_filled(
            cx, cy, self.CELL_SIZE * 0.4, self.CELL_SIZE * 0.3, hole_color)

    def _draw_ant(self):
        # Dibuja la hormiga animada y procedural
        ax, ay = self.ant_display_pos
        cx, cy = self._cell_to_pixel(ax, ay)

        # Determinar la dirección basada en la última acción tomada por el agente
        action = self.game.last_action
        if action == 0:   # Arriba (Gym) -> Arriba (Arcade)
            target_angle = 90
        elif action == 1:  # Abajo (Gym) -> Abajo (Arcade)
            target_angle = 270
        elif action == 2:  # Izquierda
            target_angle = 180
        elif action == 3:  # Derecha
            target_angle = 0
        else:
            target_angle = 0

        angle = target_angle

        # Definición del cuerpo
        body_color = self.COLOR_ANT
        shadow_color = (
            int(body_color[0]*0.3), int(body_color[1]*0.3), int(body_color[2]*0.3), 180)
        leg_color = (
            max(0, body_color[0]-50), max(0, body_color[1]-50), max(0, body_color[2]-50))

        head_radius = self.CELL_SIZE * 0.16
        thorax_radius_x = self.CELL_SIZE * 0.13
        thorax_radius_y = self.CELL_SIZE * 0.11
        abdomen_radius_x = self.CELL_SIZE * 0.24
        abdomen_radius_y = self.CELL_SIZE * 0.18

        # Función auxiliar para rotar puntos alrededor del centro
        angle_rad = math.radians(angle)

        def rotate(x, y):
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            return rx, ry

        # --- Animación y Movimiento ---
        t = time.time()
        distance_to_target = math.sqrt((self.game.ant_pos[0] - self.ant_display_pos[0])**2 +
                                       (self.game.ant_pos[1] - self.ant_display_pos[1])**2)
        moving = distance_to_target > 0.01

        if moving:
            animation_speed = 25.0
            leg_oscillation_amount = 25
            antenna_oscillation_amount = 10
            bounce = abs(math.sin(t * animation_speed)) * 3
        else:
            animation_speed = 3.0
            leg_oscillation_amount = 3
            antenna_oscillation_amount = 5
            bounce = 0

        cy += bounce
        oscillation = math.sin(t * animation_speed)

        # --- Dibujo de Patas ---
        leg_length = self.CELL_SIZE * 0.22
        leg_thickness = 3

        for side in [-1, 1]:
            for i, offset_angle in enumerate([-40, 0, 40]):
                is_set_1 = (side == 1 and i != 1) or (side == -1 and i == 1)
                current_oscillation = oscillation if is_set_1 else -oscillation
                osc = current_oscillation * leg_oscillation_amount
                end_angle_deg = angle + (90 + offset_angle + osc) * side
                sx, sy = cx, cy
                ex_rel = math.cos(math.radians(end_angle_deg)) * leg_length
                ey_rel = math.sin(math.radians(end_angle_deg)) * leg_length
                self.arcade.draw_line(
                    sx, sy, sx + ex_rel, sy + ey_rel, leg_color, leg_thickness)

        # --- Dibujo del Cuerpo (Encima de las patas) ---
        shadow_offset_x, shadow_offset_y = 3, -3

        # Abdomen
        abd_offset_x = -(thorax_radius_x + abdomen_radius_x*0.85)
        ax_rel, ay_rel = rotate(abd_offset_x, 0)
        self.arcade.draw_ellipse_filled(cx + ax_rel + shadow_offset_x, cy + ay_rel + shadow_offset_y,
                                        abdomen_radius_x, abdomen_radius_y, shadow_color, angle)
        self.arcade.draw_ellipse_filled(
            cx + ax_rel, cy + ay_rel, abdomen_radius_x, abdomen_radius_y, body_color, angle)

        # Tórax
        self.arcade.draw_ellipse_filled(cx + shadow_offset_x, cy + shadow_offset_y,
                                        thorax_radius_x, thorax_radius_y, shadow_color, angle)
        self.arcade.draw_ellipse_filled(
            cx, cy, thorax_radius_x, thorax_radius_y, body_color, angle)

        # Cabeza
        head_offset_x = head_radius*0.85 + thorax_radius_x
        hx_rel, hy_rel = rotate(head_offset_x, 0)
        self.arcade.draw_circle_filled(
            cx + hx_rel + shadow_offset_x, cy + hy_rel + shadow_offset_y, head_radius, shadow_color)
        self.arcade.draw_circle_filled(
            cx + hx_rel, cy + hy_rel, head_radius, body_color)

        # --- Detalles de la Cabeza ---
        eye_radius = head_radius * 0.3
        eye_offset_x = head_radius * 0.4
        eye_offset_y = head_radius * 0.65
        for side in [-1, 1]:
            eox, eoy = rotate(eye_offset_x, eye_offset_y * side)
            self.arcade.draw_circle_filled(
                cx + hx_rel + eox, cy + hy_rel + eoy, eye_radius, (30, 30, 30))

        antenna_length = head_radius * 1.8
        antenna_oscillation = oscillation * antenna_oscillation_amount
        for side in [-1, 1]:
            end_angle = angle + (45 * side) + antenna_oscillation
            start_offset_x = head_radius * 0.9
            start_offset_y = head_radius * 0.4 * side
            asx_rel, asy_rel = rotate(start_offset_x, start_offset_y)
            asx = cx + hx_rel + asx_rel
            asy = cy + hy_rel + asy_rel
            aex_rel = math.cos(math.radians(end_angle)) * antenna_length
            aey_rel = math.sin(math.radians(end_angle)) * antenna_length
            self.arcade.draw_line(asx, asy, asx + aex_rel,
                                  asy + aey_rel, leg_color, 2)

    def draw(self, game: AntGame, q_table_to_render, render_mode: str | None):

        if render_mode is None:
            return None

        # Inicialización perezosa
        if not self.initialized:
            self._initialize(game, render_mode)

        # Cálculo del tiempo delta
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        delta_time = min(delta_time, 0.1)

        # Actualizar animaciones
        self._update_animations(delta_time)

        # --- Proceso de Dibujo ---
        self.window.switch_to()
        self.window.clear()

        # 1. Elementos estáticos (Fondo y Obstáculos)
        self._draw_static_elements()

        # 2. Heatmap (Visualización de Q-Table)
        self._draw_heatmap(q_table_to_render)

        # 3. Elementos dinámicos
        self._draw_anthill()
        self._draw_ant()

        # 4. Post-procesado: Efecto Flash de Colisión
        if self.flash_time > 0:
            intensity = self.flash_time / 0.1
            alpha = int(220 * (intensity**2))
            self.draw_lbwh_rectangle_filled(
                0, 0, self.WIDTH, self.HEIGHT, (255, 120, 120, alpha))

        # Devolvemos las dimensiones para el modo rgb_array
        return self.WIDTH, self.HEIGHT


# =============================================================================
# GYMNASIUM ENVIRONMENT WRAPPER
# (Mantenido estructuralmente idéntico al original, integrando la lógica de AntGame)
# =============================================================================

class LostAntEnv(gym.Env):
    # Aumentamos los FPS a 60 para aprovechar las animaciones fluidas del nuevo renderer.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 reward_goal=100,
                 reward_obstacle=-100,
                 reward_move=-1,
                 use_numba_core: bool = False,
                 ):
        super().__init__()

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        self.REWARD_GOAL = reward_goal
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move
        self.use_numba_core = use_numba_core

        # Espacios de acción y observación
        # 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            # Posición (x, y)
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        # Lógica del juego (Delegada a AntGame)
        self._game = AntGame(
            grid_size=grid_size,
            reward_goal=reward_goal,
            reward_obstacle=reward_obstacle,
            reward_move=reward_move,
            use_numba_core=use_numba_core,
        )
        self._renderer: ArcadeRenderer | None = None

        # Referencias externas para compatibilidad (¡Crucial mantenerlas sincronizadas!)
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

        # Configuración de renderizado
        self.render_mode = render_mode
        self.window = None
        self.q_table_to_render = None  # Almacena la Q-table para el renderer

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad para respawn
        self._respawn_unseeded: bool = True
        try:
            self._respawn_rng = np.random.default_rng()
        except Exception:
            # Fallback para versiones antiguas de numpy
            self._respawn_rng = np.random.RandomState()

    # --- Métodos Auxiliares Privados ---

    def _sync_game_state(self):
        # Asegura que las propiedades públicas del Env reflejan el estado interno de AntGame
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

    def _get_respawn_rng(self):
        # Decide si usar la RNG global (no seeded) o la RNG del entorno (seeded) para el respawn
        if getattr(self, "_respawn_unseeded", False) and self._respawn_rng is not None:
            return self._respawn_rng
        return self.np_random

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {"goal_pos": np.array(self.goal_pos, dtype=np.int32)}

    # Métodos de compatibilidad (si se llaman externamente)
    def _generate_scenario(self):
        self._game.generate_scenario(self.np_random)
        self._sync_game_state()

    def _place_ant(self):
        rng = self._get_respawn_rng()
        self._game.place_ant(rng)
        self._sync_game_state()

    # --- API Pública de Gymnasium ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Resetear el estado del renderer para limpiar animaciones y timers
        if self._renderer:
            self._renderer.initialized = False
            self._renderer.ant_display_pos = None
            self._renderer.ant_prev_pos = None
            self._renderer.last_time = time.time()

        # Lógica de generación de escenario (Seeded vs Reuse)
        scenario_not_ready = (not np.any(self._game.goal_pos)) or (
            not self._game.obstacles)

        if seed is not None or scenario_not_ready:
            self._game.generate_scenario(self.np_random)

        # Recolocar la hormiga siempre al inicio del episodio
        rng = self._get_respawn_rng()
        self._game.place_ant(rng)

        # Sincronizar estado
        self._sync_game_state()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Ejecutar la lógica del paso usando AntGame.
        obs, reward, terminated, game_info = self._game.step(action)

        truncated = False
        info = self._get_info()
        # Fusionamos la info de AntGame (collided, terminated)
        info.update(game_info)

        # Añadir sonidos basados en el resultado
        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        # Sincronizar estado
        self._sync_game_state()

        return obs, reward, terminated, truncated, info

    def render(self):
        # Importación perezosa de arcade para el modo headless
        try:
            import arcade
        except ImportError:
            if self.render_mode in ["human", "rgb_array"]:
                raise ImportError("Se requiere 'arcade' para el renderizado.")
            return None

        # Dibujar el frame
        result = self._render_frame()
        if result is None:
            return None

        width, height = result

        if self.render_mode == "human":
            self._handle_human_render()
        elif self.render_mode == "rgb_array":
            return self._capture_rgb_array(width, height)

    def _render_frame(self):
        # Delegar el dibujo al ArcadeRenderer
        if self._renderer is None:
            self._renderer = ArcadeRenderer()

        # El renderer usa la información de estado directamente de self._game
        result = self._renderer.draw(
            self._game, self.q_table_to_render, self.render_mode)

        # Mantener referencia a la ventana
        if self._renderer is not None:
            self.window = self._renderer.window
        return result

    def _handle_human_render(self):
        # Mostrar en pantalla y regular FPS
        if self.window is not None:
            try:
                self.window.dispatch_events()  # Evitar congelamiento
                self.window.flip()
            except Exception:
                pass  # Ignorar errores de ventana durante el entrenamiento

        # Regulación de FPS (Sleep).
        import time as _time
        target_sleep = 1.0 / float(self.metadata.get("render_fps", 60))
        _time.sleep(target_sleep)

    def _capture_rgb_array(self, width, height):
        # Capturar el framebuffer a un array numpy RGB

        arcade_module = None
        if self._renderer and self._renderer.arcade:
            arcade_module = self._renderer.arcade
        else:
            try:
                import arcade
                arcade_module = arcade
            except ImportError:
                return None

        if arcade_module is None:
            return None

        try:
            image = arcade_module.get_image(0, 0, width, height)
        except TypeError:
            # Fallback para firmas antiguas de get_image()
            try:
                image = arcade_module.get_image()
            except Exception as e:
                print(f"Error al capturar imagen rgb_array: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)

        # Conversión a Numpy RGB para cumplir con el estándar de Gymnasium (3 canales)
        try:
            image = image.convert("RGB")
            frame = np.asarray(image)
        except Exception as e:
            print(f"Error al convertir imagen a array numpy: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        return frame

    def close(self):
        if self.window:
            try:
                self.window.close()
            except Exception:
                try:
                    import arcade
                    arcade.close_window()
                except Exception:
                    pass
            self.window = None
            self._renderer = None

    # --- API Extendida (Para entrenamiento y visualización) ---

    def set_respawn_unseeded(self, flag: bool = True):
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, q_table):
        # Permite pasar la Q-table actualizada al entorno para el renderizado del heatmap
        self.q_table_to_render = q_table

# mlvlab/envs/ant/ant_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
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

# Núcleo numérico puro para el paso del entorno.
# Requiere:
# - arrays y escalares nativos (sin objetos Python)
# - obstáculos como rejilla 2D uint8 (1=obstáculo)


@njit(cache=True, nogil=True)
def _step_core_numba(
    ant_x: int,
    ant_y: int,
    action: int,
    grid_size: int,
    food_x: int,
    food_y: int,
    obstacles_grid: np.ndarray,  # 2D uint8 (1 = obstáculo)
    reward_move: int,
    reward_food: int,
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

    # Comida
    if target_x == food_x and target_y == food_y:
        return target_x, target_y, reward_food, 1

    # Obstáculo
    if obstacles_grid[target_y, target_x] == 1:
        return ant_x, ant_y, reward_obstacle, 0

    # Movimiento normal
    return target_x, target_y, reward_move, 0

# -------------------------------------------------------------
# Nota sobre acelerar con Numba
# -------------------------------------------------------------
#
# ¿Aporta Numba aceleración aquí?
# - El método "hot path" de un entorno Gym suele ser `step(...)` (se llama
#   millones de veces). En este archivo, `step` hace poca aritmética y sí
#   varias operaciones de alto nivel de Python: membership en un `set` de
#   tuplas (`(ax, ay) in self.obstacles`), acceso a atributos del objeto y
#   uso de RNG de Gym (`self.np_random`). Eso limita mucho lo que Numba puede
#   compilar y acelerar directamente.
# - En su forma actual, la mayor parte del coste suele venir del loop Python
#   externo (el agente que llama `env.step(...)`) y del render (cuando está
#   activo). Por ello, aplicar Numba tal cual sobre `step` no tendrá impacto
#   apreciable.
#
# ¿Dónde sí podría ayudar Numba?
# - Si se extrae un núcleo numérico "puro" sin objetos Python (ni `set`, ni RNG
#   de Gym), Numba puede compilarlo a nativo. Para ello, conviene:
#   1) Representar obstáculos como una rejilla `np.ndarray` booleana/uint8
#      (`obstacles_grid[y, x]`), no como `set` de tuplas.
#   2) Pasar al núcleo sólo escalares y arrays NumPy (int32/float32), y devolver
#      valores primitivos.
#   3) Mantener el RNG fuera (la selección de acción ya la hace el agente), o
#      usar números que entren como parámetros.
#
# Esbozo de aplicación (opcional, no activado por defecto):
# - Añadir una rejilla de obstáculos en `_generate_scenario`:
#     self.obstacles_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), np.uint8)
#     for (ox, oy) in self.obstacles:
#         self.obstacles_grid[oy, ox] = 1
# - Extraer un núcleo JIT-able:
#
#     try:
#         from numba import njit
#     except Exception:
#         # Fallback inofensivo si Numba no está instalado
#         def njit(*args, **kwargs):
#             def _decorator(f):
#                 return f
#             return _decorator
#
#     @njit(cache=True, nogil=True)
#     def _step_core_numba(ax, ay, action,
#                          grid_size,
#                          food_x, food_y,
#                          obstacles_grid,  # 2D uint8 (1 = obstáculo)
#                          r_move, r_food, r_obst):
#         # Calcular objetivo
#         tx, ty = ax, ay
#         if action == 0:
#             ty -= 1
#         elif action == 1:
#             ty += 1
#         elif action == 2:
#             tx -= 1
#         elif action == 3:
#             tx += 1
#
#         # Márgenes
#         if tx < 0 or tx >= grid_size or ty < 0 or ty >= grid_size:
#             return ax, ay, r_obst, 0  # terminated=0
#
#         # Comida
#         if tx == food_x and ty == food_y:
#             return tx, ty, r_food, 1  # terminated=1
#
#         # Obstáculo
#         if obstacles_grid[ty, tx] == 1:
#             return ax, ay, r_obst, 0
#
#         # Movimiento normal
#         return tx, ty, r_move, 0
#
# - Uso dentro de `step` (pseudocódigo):
#     if hasattr(self, "obstacles_grid"):
#         ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
#         nx, ny, reward, terminated = _step_core_numba(
#             ax, ay, int(action),
#             int(self.GRID_SIZE),
#             int(self.food_pos[0]), int(self.food_pos[1]),
#             self.obstacles_grid,
#             self.REWARD_MOVE, self.REWARD_FOOD, self.REWARD_OBSTACLE,
#         )
#         self.ant_pos[0], self.ant_pos[1] = nx, ny
#         # `truncated` se seguiría gestionando en Python
#         # y `info` (sonidos/render) también fuera del núcleo.
#     else:
#         # Ruta actual (Python) sin Numba
#         ...
#
# Expectativa de mejora:
# - En un único entorno, el beneficio puede ser modesto por el overhead de
#   cruzar la frontera Python↔nativo en cada `step`. Donde realmente compensa
#   es al procesar lotes (vectorizar múltiples entornos/acciones) dentro de un
#   mismo núcleo JIT. Si en el futuro se añade un `VecEnv` propio o se simulan
#   N entornos en paralelo dentro de un único array, Numba sí aporta ganancias.
# -------------------------------------------------------------


class AntGame:
    """
    Lógica del juego (estado y transición), sin dependencias de UI.
    """

    def __init__(self, grid_size: int, reward_food: int, reward_obstacle: int, reward_move: int,
                 use_numba_core: bool = False) -> None:
        self.grid_size = int(grid_size)
        self.reward_food = int(reward_food)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)
        self.use_numba_core = bool(use_numba_core) and _NUMBA_AVAILABLE
        self.ant_pos = np.zeros(2, dtype=np.int32)
        self.food_pos = np.zeros(2, dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)
        self._np_random = None

    def reset(self, np_random) -> None:
        self._np_random = np_random
        self.generate_scenario(np_random)
        self.place_ant(np_random)

    def generate_scenario(self, np_random) -> None:
        self._np_random = np_random
        # Generar comida y obstáculos de forma determinista con la RNG recibida
        self.food_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        self.obstacles = {
            tuple(self._np_random.integers(0, self.grid_size, size=2).tolist())
            for _ in range(self.grid_size)
        }
        while tuple(self.food_pos.tolist()) in self.obstacles:
            self.food_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                self.obstacles_grid[oy, ox] = 1

    def place_ant(self, np_random) -> None:
        self._np_random = np_random
        # Colocar hormiga en celda válida distinta de comida y no obstáculo
        self.ant_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        while tuple(self.ant_pos.tolist()) in self.obstacles or (
            self.ant_pos[0] == self.food_pos[0] and self.ant_pos[1] == self.food_pos[1]
        ):
            self.ant_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

    def get_obs(self) -> np.ndarray:
        return np.array((int(self.ant_pos[0]), int(self.ant_pos[1])), dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info: dict = {}

        # Ruta acelerada por Numba si está habilitada
        if self.use_numba_core and isinstance(self.obstacles_grid, np.ndarray):
            nx, ny, reward, terminated_int = _step_core_numba(
                ax,
                ay,
                int(action),
                int(self.grid_size),
                int(self.food_pos[0]),
                int(self.food_pos[1]),
                self.obstacles_grid,
                int(self.reward_move),
                int(self.reward_food),
                int(self.reward_obstacle),
            )
            terminated = bool(terminated_int)
            self.ant_pos[0], self.ant_pos[1] = nx, ny
            return np.array((nx, ny), dtype=np.int32), int(reward), bool(terminated), info

        # Ruta Python
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
            collided = True
        else:
            ax, ay = target_ax, target_ay
            terminated = (
                ax == int(self.food_pos[0]) and ay == int(self.food_pos[1]))
            if terminated:
                reward = self.reward_food
                collided = False
            elif (ax, ay) in self.obstacles:
                reward = self.reward_obstacle
                ax, ay = prev_ax, prev_ay
                collided = True
            else:
                reward = self.reward_move
                collided = False

        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = collided
        info["terminated"] = terminated
        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info


class ArcadeRenderer:
    """
    Renderer basado en Arcade. Soporta "human" y "rgb_array".
    """

    def __init__(self) -> None:
        self.window = None

    def draw(self, game: AntGame, q_table_to_render, render_mode: str | None):
        import arcade
        from arcade.draw.rect import draw_lbwh_rectangle_filled
        import time

        if render_mode is None:
            return None

        CELL_SIZE = 30
        WIDTH = HEIGHT = game.grid_size * CELL_SIZE
        COLOR_GRID = (40, 40, 40)
        COLOR_ANT = (255, 64, 64)
        COLOR_FOOD = (64, 255, 128)
        COLOR_OBSTACLE = (120, 120, 120)

        # Crear ventana si hace falta
        if self.window is None and render_mode in ["human", "rgb_array"]:
            visible = render_mode == "human"
            try:
                self.window = arcade.Window(
                    WIDTH, HEIGHT, "Lost Ant Colony", visible=visible)
            except TypeError:
                self.window = arcade.Window(WIDTH, HEIGHT, "Lost Ant Colony")
                if not visible:
                    try:
                        self.window.set_visible(False)
                    except Exception:
                        pass

        # Visibilidad si cambiamos a human
        if self.window is not None and render_mode == "human":
            try:
                self.window.set_visible(True)
            except Exception:
                pass

        def cell_to_pixel(x_cell: int, y_cell: int):
            x_px = x_cell * CELL_SIZE
            y_px = (game.grid_size - 1 - y_cell) * CELL_SIZE
            return x_px, y_px

        # Render base
        self.window.switch_to()
        arcade.set_background_color(COLOR_GRID)
        self.window.clear()

        # Heatmap clásico verde
        if q_table_to_render is not None:
            max_q = float(np.max(q_table_to_render))
            min_q = float(np.min(q_table_to_render))
            if max_q > min_q:
                for state_index in range(game.grid_size * game.grid_size):
                    x_cell = state_index % game.grid_size
                    y_cell = state_index // game.grid_size
                    q_value = float(np.max(q_table_to_render[state_index, :]))
                    norm_q = (q_value - min_q) / (max_q - min_q)
                    intensity = int(40 + norm_q * 180)
                    heat_color = (0, intensity, 0, 255)
                    x_px, y_px = cell_to_pixel(x_cell, y_cell)
                    draw_lbwh_rectangle_filled(
                        x_px, y_px, CELL_SIZE, CELL_SIZE, heat_color)

        # Cuadrícula
        grid_color = (60, 60, 60, 120)
        for i in range(game.grid_size + 1):
            x = i * CELL_SIZE
            y = i * CELL_SIZE
            arcade.draw_line(x, 0, x, HEIGHT, grid_color, 1)
            arcade.draw_line(0, y, WIDTH, y, grid_color, 1)

        # Obstáculos
        for obs in game.obstacles:
            x_px, y_px = cell_to_pixel(obs[0], obs[1])
            draw_lbwh_rectangle_filled(
                x_px + 2, y_px + 2, CELL_SIZE - 2, CELL_SIZE - 2, (90, 90, 90, 255))
            draw_lbwh_rectangle_filled(
                x_px, y_px, CELL_SIZE - 2, CELL_SIZE, COLOR_OBSTACLE)

        # Comida (pulso)
        fx_time = time.time()
        pulse = 0.5 + 0.5 * np.sin(fx_time * 3.0)
        fx_radius = int(6 + 8 * pulse)
        cx_food = game.food_pos[0] * CELL_SIZE + CELL_SIZE // 2
        cy_food = (game.grid_size - 1 -
                   game.food_pos[1]) * CELL_SIZE + CELL_SIZE // 2
        for r, alpha in [(fx_radius * 2, 30), (fx_radius, 60), (fx_radius // 2, 90)]:
            arcade.draw_circle_filled(
                cx_food, cy_food, max(2, r), (64, 255, 128, alpha))
        x_px, y_px = cell_to_pixel(game.food_pos[0], game.food_pos[1])
        draw_lbwh_rectangle_filled(
            x_px + 4, y_px + 4, CELL_SIZE - 8, CELL_SIZE - 8, (*COLOR_FOOD, 255))

        # Hormiga
        ax_px, ay_px = cell_to_pixel(
            int(game.ant_pos[0]), int(game.ant_pos[1]))
        cx_ant = ax_px + CELL_SIZE // 2
        cy_ant = ay_px + CELL_SIZE // 2
        arcade.draw_circle_filled(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (*COLOR_ANT, 255))
        arcade.draw_circle_outline(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (255, 255, 255, 180), 2)
        arcade.draw_line(cx_ant, cy_ant, cx_ant, cy_ant +
                         CELL_SIZE * 0.45, (255, 220, 220, 200), 2)

        return WIDTH, HEIGHT


class LostAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 reward_food=100,
                 reward_obstacle=-100,
                 reward_move=-1,
                 use_numba_core: bool = False,
                 ):
        super().__init__()

        # Los valores ahora vienen de los parámetros, con un valor por defecto
        self.GRID_SIZE = grid_size
        self.REWARD_FOOD = reward_food
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        # Juego y renderer
        self._game = AntGame(
            grid_size=grid_size,
            reward_food=reward_food,
            reward_obstacle=reward_obstacle,
            reward_move=reward_move,
            use_numba_core=use_numba_core,
        )
        self._renderer: ArcadeRenderer | None = None

        # Para compatibilidad con referencias externas
        self.ant_pos = self._game.ant_pos
        self.food_pos = self._game.food_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

        self.render_mode = render_mode
        self.window = None
        self._window_visible = False
        self._last_time = None
        self.q_table_to_render = None  # Para visualización avanzada

        # Activación opcional del núcleo Numba gestionada por AntGame

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Control de respawn aleatorio independiente de la seed de escenario
        self._respawn_unseeded: bool = True
        try:
            self._respawn_rng = np.random.default_rng()
        except Exception:
            self._respawn_rng = None

    def _generate_scenario(self):
        # Compatibilidad: generar un nuevo laberinto (para seed nueva)
        self._game.generate_scenario(self.np_random)
        rng = self._respawn_rng if getattr(
            self, "_respawn_unseeded", False) and self._respawn_rng is not None else self.np_random
        self._game.place_ant(rng)
        self.ant_pos = self._game.ant_pos
        self.food_pos = self._game.food_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

    def _place_ant(self):
        # AntGame ya recoloca la hormiga en reset; mantener compatibilidad
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reglas Gymnasium: si llega una seed nueva o nunca se generó el escenario,
        # crear un mapa nuevo; si no, conservar el laberinto y sólo recolocar la hormiga.
        scenario_not_ready = (self.food_pos is None) or (
            not self.obstacles)  # sin comida o sin rocas
        if seed is not None or scenario_not_ready:
            self._game.generate_scenario(self.np_random)
        rng = self._respawn_rng if getattr(
            self, "_respawn_unseeded", False) and self._respawn_rng is not None else self.np_random
        self._game.place_ant(rng)
        # Exponer referencias compartidas para compatibilidad
        self.ant_pos = self._game.ant_pos
        self.food_pos = self._game.food_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid
        return self._get_obs(), self._get_info()

    # API opcional para controlar el respawn independiente de la seed
    def set_respawn_unseeded(self, flag: bool = True):
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, q_table):
        self.q_table_to_render = q_table

    def _get_obs(self):
        # Devolver una copia para prevenir el "Bug de Observación Mutable"
        return np.array((int(self.ant_pos[0]), int(self.ant_pos[1])), dtype=np.int32)

    def _get_info(self):
        return {"food_pos": self.food_pos}

    def step(self, action):
        # Coordenadas actuales
        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info = self._get_info()
        truncated = False

        # Ruta acelerada por Numba si está habilitada y preparada
        if (
            getattr(self, "use_numba_core", False)
            and hasattr(self, "obstacles_grid")
            and isinstance(self.obstacles_grid, np.ndarray)
        ):
            nx, ny, reward, terminated_int = _step_core_numba(
                ax,
                ay,
                int(action),
                int(self.GRID_SIZE),
                int(self.food_pos[0]),
                int(self.food_pos[1]),
                self.obstacles_grid,
                int(self.REWARD_MOVE),
                int(self.REWARD_FOOD),
                int(self.REWARD_OBSTACLE),
            )
            terminated = bool(terminated_int)
            # Sonido según resultado (equivalente a la ruta Python)
            if terminated:
                info['play_sound'] = {'filename': 'blip.wav', 'volume': 10}
            elif reward == self.REWARD_OBSTACLE and (nx == ax and ny == ay):
                info['play_sound'] = {'filename': 'crash.wav', 'volume': 5}

            self.ant_pos[0], self.ant_pos[1] = nx, ny
            return (
                np.array((nx, ny), dtype=np.int32),
                reward,
                terminated,
                truncated,
                info,
            )

        # Ruta Python original (compatibilidad)
        prev_ax, prev_ay = ax, ay

        # 1) Calcular intento de movimiento (posible salida de límites)
        target_ax, target_ay = ax, ay
        if action == 0:
            target_ay -= 1  # Arriba
        elif action == 1:
            target_ay += 1  # Abajo
        elif action == 2:
            target_ax -= 1  # Izquierda
        elif action == 3:
            target_ax += 1  # Derecha

        # 2) Tratar márgenes como paredes: si se intenta salir, penalizar y no mover
        out_of_bounds = (
            target_ax < 0 or target_ax >= self.GRID_SIZE or
            target_ay < 0 or target_ay >= self.GRID_SIZE
        )
        if out_of_bounds:
            reward = self.REWARD_OBSTACLE
            ax, ay = prev_ax, prev_ay  # queda en la misma celda
            info['play_sound'] = {'filename': 'crash.wav', 'volume': 5}
            terminated = False
        else:
            # 3) Movimiento válido dentro de los límites
            ax, ay = target_ax, target_ay
            terminated = (
                ax == int(self.food_pos[0]) and ay == int(self.food_pos[1]))
            if terminated:
                reward = self.REWARD_FOOD
                info['play_sound'] = {'filename': 'blip.wav', 'volume': 10}
            elif (ax, ay) in self.obstacles:
                reward = self.REWARD_OBSTACLE
                ax, ay = prev_ax, prev_ay  # revertir
                info['play_sound'] = {'filename': 'crash.wav', 'volume': 5}
            else:
                reward = self.REWARD_MOVE

        # 4) Actualizar posición
        self.ant_pos[0], self.ant_pos[1] = ax, ay
        return np.array((ax, ay), dtype=np.int32), reward, terminated, truncated, info

    def _render_frame(self):
        # Delegar al renderer
        if self._renderer is None:
            self._renderer = ArcadeRenderer()
        result = self._renderer.draw(
            self._game, self.q_table_to_render, self.render_mode)
        if self._renderer is not None:
            self.window = self._renderer.window
        return result

    def render(self):
        # Importamos Arcade aquí para que esté disponible en todo el método.
        import arcade

        result = self._render_frame()
        if result is None:
            return None

        width, height = result

        if self.render_mode == "human":
            # Mostrar en pantalla y regular FPS
            if self.window is not None:
                # Procesar eventos de ventana para evitar congelamientos al perder foco
                try:
                    self.window.dispatch_events()
                except Exception:
                    pass
                try:
                    self.window.flip()
                except Exception:
                    pass
            # Pequeña pausa para sincronizar FPS
            import time as _time
            _time.sleep(1.0 / float(self.metadata.get("render_fps", 30)))
        elif self.render_mode == "rgb_array":
            # Capturamos el frame a una imagen y lo convertimos a numpy RGB (H, W, 3)
            try:
                image = arcade.get_image(0, 0, width, height)
            except TypeError:
                # Compatibilidad con firmas anteriores
                image = arcade.get_image()

            # Asegurar RGB sin alfa
            image = image.convert("RGB")
            frame = np.asarray(image)
            return frame

    def close(self):
        if self.window:
            try:
                # Cerrar ventana de Arcade
                self.window.close()
            except Exception:
                try:
                    import arcade
                    arcade.close_window()
                except Exception:
                    pass
            self.window = None

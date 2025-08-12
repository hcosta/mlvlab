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

        self.ant_pos = None
        self.food_pos = None
        self.obstacles = set()

        self.render_mode = render_mode
        self.window = None
        self._window_visible = False
        self._last_time = None
        self.q_table_to_render = None  # Para visualización avanzada

        # Activación opcional del núcleo Numba (solo si está disponible)
        self.use_numba_core = bool(use_numba_core) and _NUMBA_AVAILABLE

        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def _generate_scenario(self):
        """Genera y establece las posiciones de la comida y los obstáculos."""
        self.food_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        # Conjunto de obstáculos para membership O(1)
        self.obstacles = {
            tuple(self.np_random.integers(0, self.GRID_SIZE, size=2).tolist())
            for _ in range(self.GRID_SIZE)
        }
        while tuple(self.food_pos.tolist()) in self.obstacles:
            self.food_pos = self.np_random.integers(
                0, self.GRID_SIZE, size=2, dtype=np.int32)

        # Rejilla de obstáculos para el núcleo Numba (1 = obstáculo)
        self.obstacles_grid = np.zeros(
            (self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8
        )
        for ox, oy in self.obstacles:
            # Seguridad por si aparecen duplicados o valores extremos
            if 0 <= ox < self.GRID_SIZE and 0 <= oy < self.GRID_SIZE:
                self.obstacles_grid[oy, ox] = 1

    def _place_ant(self):
        """Busca una posición inicial aleatoria para la hormiga."""
        self.ant_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        while tuple(self.ant_pos.tolist()) in self.obstacles or (
            self.ant_pos[0] == self.food_pos[0] and self.ant_pos[1] == self.food_pos[1]
        ):
            self.ant_pos = self.np_random.integers(
                0, self.GRID_SIZE, size=2, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Si se proporciona una semilla nueva, O si el escenario nunca se ha generado,
        # creamos un mapa nuevo (comida y obstáculos).
        if seed is not None or self.food_pos is None:
            self._generate_scenario()

        # En CADA reset, la hormiga busca una nueva posición aleatoria en el mapa actual.
        self._place_ant()

        return self._get_obs(), self._get_info()

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
        # --- SECCIÓN VISUAL (AISLADA) ---
        # Si nunca se llama a render(), esta sección nunca se ejecuta.
        import arcade
        from arcade.draw.rect import draw_lbwh_rectangle_filled
        import time

        # Constantes de dibujado (solo existen dentro de este método)
        CELL_SIZE = 30
        WIDTH = HEIGHT = self.GRID_SIZE * CELL_SIZE
        COLOR_GRID = (40, 40, 40)
        COLOR_ANT = (255, 64, 64)
        COLOR_FOOD = (64, 255, 128)
        COLOR_OBSTACLE = (120, 120, 120)

        # Crear ventana de Arcade perezosamente. Para rgb_array la mantenemos oculta.
        if self.window is None and self.render_mode in ["human", "rgb_array"]:
            self._window_visible = self.render_mode == "human"
            # En Arcade, el origen está en la esquina inferior-izquierda.
            # Usamos "visible" para soportar render fuera de pantalla.
            try:
                self.window = arcade.Window(
                    WIDTH, HEIGHT, "Lost Ant Colony", visible=self._window_visible
                )
            except TypeError:
                # Compatibilidad: versiones sin parámetro visible
                self.window = arcade.Window(WIDTH, HEIGHT, "Lost Ant Colony")
                if not self._window_visible:
                    try:
                        self.window.set_visible(False)
                    except Exception:
                        pass

        if self.render_mode is None:
            return None

        # Si cambiamos de modo rgb_array -> human, hacer visible la ventana
        if self.window is not None:
            target_visibility = self.render_mode == "human"
            if target_visibility and not self._window_visible:
                try:
                    self.window.set_visible(True)
                except Exception:
                    pass
                self._window_visible = True

        # Utilidades de coordenadas: convertir celda (x,y) a píxeles (origen abajo-izquierda en Arcade)
        def cell_to_pixel(x_cell: int, y_cell: int):
            x_px = x_cell * CELL_SIZE
            # Invertimos Y para mantener la misma orientación que en PyGame (y hacia abajo)
            y_px = (self.GRID_SIZE - 1 - y_cell) * CELL_SIZE
            return x_px, y_px

        # Renderizado base
        self.window.switch_to()
        arcade.set_background_color(COLOR_GRID)
        # Limpiar el frame usando el color de fondo configurado
        self.window.clear()

        # Dibujar mapa de calor de Q-table si está disponible
        if self.q_table_to_render is not None:
            max_q = float(np.max(self.q_table_to_render))
            min_q = float(np.min(self.q_table_to_render))
            if max_q > min_q:
                for state_index in range(self.GRID_SIZE * self.GRID_SIZE):
                    x_cell = state_index % self.GRID_SIZE
                    y_cell = state_index // self.GRID_SIZE
                    q_value = float(
                        np.max(self.q_table_to_render[state_index, :]))
                    norm_q = (q_value - min_q) / (max_q - min_q)
                    intensity = int(40 + norm_q * 180)
                    heat_color = (0, intensity, 0, 255)
                    x_px, y_px = cell_to_pixel(x_cell, y_cell)
                    draw_lbwh_rectangle_filled(
                        x_px,
                        y_px,
                        CELL_SIZE,
                        CELL_SIZE,
                        heat_color,
                    )

        # Dibujar cuadrícula sutil
        grid_color = (60, 60, 60, 120)
        for i in range(self.GRID_SIZE + 1):
            x = i * CELL_SIZE
            y = i * CELL_SIZE
            arcade.draw_line(x, 0, x, HEIGHT, grid_color, 1)
            arcade.draw_line(0, y, WIDTH, y, grid_color, 1)

        # Dibujar obstáculos
        for obs in self.obstacles:
            x_px, y_px = cell_to_pixel(obs[0], obs[1])
            # Sombra
            draw_lbwh_rectangle_filled(
                x_px + 2,
                y_px + 2,
                CELL_SIZE - 2,
                CELL_SIZE - 2,
                (90, 90, 90, 255),
            )
            # Cara superior con ligero brillo
            draw_lbwh_rectangle_filled(
                x_px,
                y_px,
                CELL_SIZE - 2,
                CELL_SIZE,
                COLOR_OBSTACLE,
            )

        # Dibujar comida con efecto pulso (brillo)
        fx_time = time.time()
        pulse = 0.5 + 0.5 * np.sin(fx_time * 3.0)
        fx_radius = int(6 + 8 * pulse)
        cx_food = self.food_pos[0] * CELL_SIZE + CELL_SIZE // 2
        cy_food = (self.GRID_SIZE - 1 -
                   self.food_pos[1]) * CELL_SIZE + CELL_SIZE // 2
        for r, alpha in [(fx_radius * 2, 30), (fx_radius, 60), (fx_radius // 2, 90)]:
            arcade.draw_circle_filled(
                cx_food, cy_food, max(2, r), (64, 255, 128, alpha))

        x_px, y_px = cell_to_pixel(self.food_pos[0], self.food_pos[1])
        draw_lbwh_rectangle_filled(
            x_px + 4,
            y_px + 4,
            CELL_SIZE - 8,
            CELL_SIZE - 8,
            (*COLOR_FOOD, 255),
        )

        # Dibujar hormiga (cuerpo + borde)
        ax_px, ay_px = cell_to_pixel(
            int(self.ant_pos[0]), int(self.ant_pos[1]))
        cx_ant = ax_px + CELL_SIZE // 2
        cy_ant = ay_px + CELL_SIZE // 2
        arcade.draw_circle_filled(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (*COLOR_ANT, 255))
        arcade.draw_circle_outline(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (255, 255, 255, 180), 2)

        # Dirección indicativa simple (flecha hacia arriba/abajo/izq/der basada en último movimiento no almacenado)
        # Por simplicidad, dibujamos una "antena" hacia arriba
        arcade.draw_line(cx_ant, cy_ant, cx_ant, cy_ant +
                         CELL_SIZE * 0.45, (255, 220, 220, 200), 2)

        # No finalizamos aquí; la finalización la gestiona render() según modo
        return WIDTH, HEIGHT

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

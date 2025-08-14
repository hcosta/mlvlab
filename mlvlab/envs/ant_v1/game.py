# mlvlab/envs/ant/game.py
import numpy as np

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
# Mantenido idéntico al núcleo original.


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
    # Nota: obstacles_grid se indexa como [Y, X]
    if obstacles_grid[target_y, target_x] == 1:
        return ant_x, ant_y, reward_obstacle, 0

    # Movimiento normal
    return target_x, target_y, reward_move, 0


class AntGame:
    """
    Lógica del juego (estado y transición), sin dependencias de UI.
    Mantiene la lógica pura y el estado del gridworld.
    """

    def __init__(self, grid_size: int, reward_goal: int, reward_obstacle: int, reward_move: int,
                 use_numba_core: bool = False) -> None:
        self.grid_size = int(grid_size)
        self.reward_goal = int(reward_goal)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)
        self.use_numba_core = bool(use_numba_core) and _NUMBA_AVAILABLE

        # Estado del juego - Inicialización idéntica al original (0,0)
        self.ant_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.goal_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.obstacles: set[tuple[int, int]] = set()  # {(X, Y), ...}
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)  # Indexado [Y, X]

        self._np_random = None

        # Estado para el renderer (Juicy)
        self.last_action = 3  # Derecha por defecto
        self.collided = False

    def reset(self, np_random) -> None:
        # Mantenido por compatibilidad si se llama internamente.
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
        # Asegurar que la meta no es un obstáculo
        while tuple(self.goal_pos.tolist()) in self.obstacles:
            self.goal_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

        # Actualizar la grid de obstáculos para Numba
        self.obstacles_grid = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.uint8)
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                # Indexado [Y, X]
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
        # Devuelve la observación actual (Posición X, Y)
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

            # Actualizar posición (modificación in-place del array numpy)
            self.ant_pos[0], self.ant_pos[1] = nx, ny
            info["collided"] = self.collided
            info["terminated"] = terminated
            # Devuelve una nueva copia de la observación
            return np.array((nx, ny), dtype=np.int32), int(reward), bool(terminated), info

        # Ruta Python (Mantenida idéntica a la estructura original)
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
            # El original mueve primero...
            ax, ay = target_ax, target_ay
            terminated = (
                ax == int(self.goal_pos[0]) and ay == int(self.goal_pos[1]))
            if terminated:
                reward = self.reward_goal
                self.collided = False
            elif (ax, ay) in self.obstacles:
                # ...y luego revierte si hay obstáculo.
                reward = self.reward_obstacle
                ax, ay = prev_ax, prev_ay
                self.collided = True
            else:
                reward = self.reward_move
                self.collided = False

        # Actualizar posición (modificación in-place del array numpy)
        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = self.collided
        info["terminated"] = terminated
        # Devuelve una nueva copia de la observación
        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info

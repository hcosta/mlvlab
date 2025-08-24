# mlvlab/envs/ant/game.py
import numpy as np


class AntGame:
    """
    Lógica del juego (estado y transición), sin dependencias de UI.
    Mantiene la lógica pura y el estado del gridworld.
    """

    def __init__(self, grid_size: int, reward_obstacle: int, reward_move: int) -> None:
        self.grid_size = int(grid_size)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)

        # Estado del juego
        self.ant_pos = np.zeros(2, dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.is_dead = False  # Flag para el estado de la hormiga

        self._np_random = None

        # Estado para el renderer
        self.last_action = 3
        self.collided = False

    def reset(self, np_random) -> None:
        self._np_random = np_random
        self.generate_scenario(np_random)
        self.place_ant(np_random)
        self.last_action = 3
        self.collided = False
        self.is_dead = False  # Reseteamos el estado de la hormiga

    def generate_scenario(self, np_random) -> None:
        self._np_random = np_random
        self.obstacles = {
            tuple(self._np_random.integers(0, self.grid_size, size=2).tolist())
            for _ in range(self.grid_size)
        }

    def place_ant(self, np_random) -> None:
        self._np_random = np_random
        self.ant_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        while tuple(self.ant_pos.tolist()) in self.obstacles:
            self.ant_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

    def get_obs(self) -> np.ndarray:
        return np.array(self.ant_pos, dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        # En este entorno, la hormiga nunca "termina" por llegar a una meta.
        # Siempre devuelve terminated = False.
        terminated = False

        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info: dict = {}
        self.last_action = action
        self.collided = False

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

        if out_of_bounds or (target_ax, target_ay) in self.obstacles:
            reward = self.reward_obstacle
            ax, ay = prev_ax, prev_ay
            self.collided = True
        else:
            reward = self.reward_move
            ax, ay = target_ax, target_ay
            self.collided = False

        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = self.collided

        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info

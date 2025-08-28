# mlvlab/envs/ant_scout_v1/game.py
import numpy as np


class AntGame:
    """
    Lógica del juego (estado y transición), sin dependencias de UI.
    Mantiene la lógica pura y el estado del gridworld.
    """

    def __init__(self, grid_size: int, reward_goal: int, reward_obstacle: int, reward_move: int) -> None:
        self.grid_size = int(grid_size)
        self.reward_goal = int(reward_goal)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)

        # Estado del juego
        self.ant_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.goal_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.obstacles: set[tuple[int, int]] = set()  # {(X, Y), ...}

        self._np_random = None

        # Estado para el renderer (Juicy)
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
        # Asegurar que la meta no es un obstáculo
        while tuple(self.goal_pos.tolist()) in self.obstacles:
            self.goal_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

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

        # Actualizar posición
        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = self.collided
        info["terminated"] = terminated

        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info

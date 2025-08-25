# mlvlab/envs/ant_lost_v1/game.py
import numpy as np


class AntGame:
    """
    Lógica de juego para AntLost. NO gestiona la terminación.
    Solo se encarga del movimiento, colisiones y estado del mundo.
    """

    def __init__(self, grid_size: int, reward_obstacle: int, reward_move: int) -> None:
        self.grid_size = int(grid_size)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)
        self.ant_pos = np.zeros(2, dtype=np.int32)
        self.obstacles: set[tuple[int, int]] = set()
        self.goal_pos = np.array(
            [-1, -1], dtype=np.int32)  # Para compatibilidad
        self._np_random = None
        self.last_action = 3
        self.collided = False

    def reset(self, np_random) -> None:
        self._np_random = np_random
        self.generate_scenario(np_random)
        self.place_ant(np_random)
        self.last_action = 3
        self.collided = False
        self.goal_pos = np.array([-1, -1], dtype=np.int32)

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

        # La colisión (con roca o límite) solo impide el movimiento. NO termina el juego.
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

        # El juego NUNCA termina por sí mismo. Siempre devuelve terminated = False.
        return self.get_obs(), int(reward), False, info

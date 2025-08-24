# mlvlab/envs/ant_lost_v1/game.py
import numpy as np


class AntLostGame:
    """
    Lógica del juego para AntLost-v1 (Zángano Errante).
    No tiene objetivo (meta) y termina por límite de tiempo.
    """

    def __init__(self, grid_size: int, max_steps: int, reward_obstacle: int, reward_move: int, reward_death: int) -> None:
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.reward_obstacle = int(reward_obstacle)
        self.reward_move = int(reward_move)
        self.reward_death = int(reward_death)

        # Estado del juego
        self.ant_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        # goal_pos eliminado
        self.obstacles: set[tuple[int, int]] = set()  # {(X, Y), ...}
        self.current_step = 0
        self.is_dead = False  # Estado de muerte

        self._np_random = None

        # Estado para el renderer (Juicy)
        self.last_action = 3  # Derecha por defecto
        self.collided = False

    def reset(self, np_random) -> None:
        self._np_random = np_random
        # Aseguramos que el escenario existe
        if not self.obstacles:
            self.generate_scenario(np_random)
        self.place_ant(np_random)
        self.last_action = 3
        self.collided = False
        self.current_step = 0
        self.is_dead = False

    def generate_scenario(self, np_random) -> None:
        self._np_random = np_random
        # Generar pocos obstáculos (piedras decorativas)
        num_obstacles = int(self.grid_size * 0.4)  # Entorno mayormente vacío
        self.obstacles = {
            tuple(self._np_random.integers(0, self.grid_size, size=2).tolist())
            for _ in range(num_obstacles)
        }

    def place_ant(self, np_random) -> None:
        self._np_random = np_random
        # Colocar hormiga en celda válida (no obstáculo)
        self.ant_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        while tuple(self.ant_pos.tolist()) in self.obstacles:
            self.ant_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

    def get_obs(self) -> np.ndarray:
        # Devuelve la observación actual (Posición X, Y)
        return np.array((int(self.ant_pos[0]), int(self.ant_pos[1])), dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        # Si ya está muerta, no procesar más pasos.
        if self.is_dead:
            return self.get_obs(), 0, True, {"is_dead": True, "collided": False}

        self.current_step += 1
        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info: dict = {}
        self.last_action = action
        self.collided = False

        # Calcular movimiento
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

        # 1. Chequeo de colisiones (Bordes y Obstáculos)
        out_of_bounds = (
            target_ax < 0 or target_ax >= self.grid_size or
            target_ay < 0 or target_ay >= self.grid_size
        )

        if out_of_bounds or (target_ax, target_ay) in self.obstacles:
            reward = self.reward_obstacle
            ax, ay = prev_ax, prev_ay  # No se mueve
            self.collided = True
        else:
            reward = self.reward_move
            ax, ay = target_ax, target_ay  # Se mueve
            self.collided = False

        # 2. Chequeo de límite de pasos (Muerte)
        terminated = False
        if self.current_step >= self.max_steps:
            reward += self.reward_death  # Se añade la penalización grande
            terminated = True
            self.is_dead = True

        # Actualizar posición
        self.ant_pos[0], self.ant_pos[1] = ax, ay
        info["collided"] = self.collided
        info["terminated"] = terminated
        info["is_dead"] = self.is_dead

        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info

# mlvlab/envs/ant/ant_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# PyGame se importará solo si se llama al método de renderizado.


class LostAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=10):
        super().__init__()

        # (La inicialización es idéntica a lost_ant_env.py)
        self.GRID_SIZE = grid_size
        self.REWARD_FOOD = 100
        self.REWARD_OBSTACLE = -100
        self.REWARD_MOVE = -1

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        self.ant_pos = None
        self.food_pos = None
        self.obstacles = []

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.q_table_to_render = None  # Para visualización avanzada

        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def _generate_scenario(self):
        """Genera y establece las posiciones de la comida y los obstáculos."""
        self.food_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        self.obstacles = [
            self.np_random.integers(0, self.GRID_SIZE, size=2).tolist() for _ in range(self.GRID_SIZE)
        ]
        while self.food_pos.tolist() in self.obstacles:
            self.food_pos = self.np_random.integers(
                0, self.GRID_SIZE, size=2, dtype=np.int32)

    def _place_ant(self):
        """Busca una posición inicial aleatoria para la hormiga."""
        self.ant_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        while self.ant_pos.tolist() in self.obstacles or np.array_equal(self.ant_pos, self.food_pos):
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
        return np.copy(self.ant_pos)

    def _get_info(self):
        return {"food_pos": self.food_pos}

    def step(self, action):
        # Guardamos la posición anterior por si choca
        old_pos = np.copy(self.ant_pos)
        info = self._get_info()

        # 1. Mueve la hormiga
        if action == 0:
            self.ant_pos[1] -= 1  # Arriba
        elif action == 1:
            self.ant_pos[1] += 1  # Abajo
        elif action == 2:
            self.ant_pos[0] -= 1  # Izquierda
        elif action == 3:
            self.ant_pos[0] += 1  # Derecha

        # 2. Asegura que la hormiga esté dentro de los límites
        self.ant_pos = np.clip(self.ant_pos, 0, self.GRID_SIZE - 1)

        # 3. Evalúa el resultado
        terminated = np.array_equal(self.ant_pos, self.food_pos)
        truncated = False

        if terminated:
            reward = self.REWARD_FOOD
            # Volumen de 0-100, como espera el player.py
            info['play_sound'] = {'filename': 'blip.wav', 'volume': 100}
        elif self.ant_pos.tolist() in self.obstacles:
            reward = self.REWARD_OBSTACLE
            # La hormiga es devuelta a su posición anterior
            self.ant_pos = old_pos
            info['play_sound'] = {'filename': 'crash.wav', 'volume': 50}
        else:
            reward = self.REWARD_MOVE

        return self._get_obs(), reward, terminated, truncated, info

    def _render_frame(self):
        # --- SECCIÓN VISUAL (AISLADA) ---
        # Si nunca se llama a render(), esta sección nunca se ejecuta.
        import pygame

        # Constantes de dibujado (solo existen dentro de este método)
        CELL_SIZE = 30
        WIDTH = HEIGHT = self.GRID_SIZE * CELL_SIZE
        COLOR_GRID = (40, 40, 40)
        COLOR_ANT = (255, 0, 0)
        COLOR_FOOD = (0, 255, 0)
        COLOR_OBSTACLE = (100, 100, 100)

        if self.window is None and self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Lost Ant Colony")
            else:  # "rgb_array"
                self.window = pygame.Surface((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        if self.render_mode is None:
            return

        self.window.fill(COLOR_GRID)
        if self.q_table_to_render is not None:
            max_q, min_q = np.max(self.q_table_to_render), np.min(
                self.q_table_to_render)
            if max_q > min_q:
                for s in range(self.GRID_SIZE * self.GRID_SIZE):
                    x, y = s % self.GRID_SIZE, s // self.GRID_SIZE
                    q_value = np.max(self.q_table_to_render[s, :])
                    norm_q = (q_value - min_q) / (max_q - min_q)
                    heat_color = (0, min(255, int(norm_q * 200)), 0)
                    pygame.draw.rect(
                        self.window, heat_color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for obs in self.obstacles:
            pygame.draw.rect(self.window, COLOR_OBSTACLE,
                             (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.window, COLOR_FOOD, (
            self.food_pos[0] * CELL_SIZE, self.food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.window, COLOR_ANT, (
            self.ant_pos[0] * CELL_SIZE, self.ant_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def render(self):
        # Importamos pygame aquí para que esté disponible en todo el método.
        import pygame

        self._render_frame()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.window), axes=(1, 0, 2))

    def close(self):
        if self.window:
            import pygame
            pygame.quit()
            self.window = None

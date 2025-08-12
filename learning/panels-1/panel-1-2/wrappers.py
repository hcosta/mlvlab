# wrappers.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Wrapper para Recompensas ---


class TimePenaltyWrapper(gym.RewardWrapper):
    """
    Añade una pequeña penalización en cada paso para incentivar al agente
    a encontrar la solución más rápido.
    """

    def __init__(self, env, penalty: float = -0.1):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward: float) -> float:
        """Aplica la penalización a la recompensa original del entorno."""
        return reward + self.penalty


class DirectionToHomeWrapper(gym.ObservationWrapper):
    """Añade a la observación el ángulo en radianes hacia el hormiguero (0,0)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.home_pos = np.array([0, 0])

        # Definimos el NUEVO espacio de observación (x, y, ángulo)
        low = np.append(self.observation_space.low, -np.pi).astype(np.float32)
        high = np.append(self.observation_space.high, np.pi).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ant_pos = obs
        direction_vector = self.home_pos - ant_pos
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        new_obs = np.append(obs, angle).astype(np.float32)
        return new_obs

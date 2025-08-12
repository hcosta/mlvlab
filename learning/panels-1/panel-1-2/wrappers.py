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

# --- Wrapper para Observaciones ---


class FoodVectorObservationWrapper(gym.ObservationWrapper):
    """
    Cambia la observación del agente. En lugar de darle su posición (x, y),
    le da el vector de distancia a la comida (food_x - ant_x, food_y - ant_y).
    Esto es un ejemplo de "feature engineering".
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # El espacio de observación ahora es diferente. Ya no es una posición en
        # la parrilla, sino un vector que puede tener valores negativos.
        grid_size = self.unwrapped.GRID_SIZE
        self.observation_space = gym.spaces.Box(
            low=-(grid_size - 1),
            high=(grid_size - 1),
            shape=(2,),
            dtype=np.int32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Transforma la observación original."""
        # 'obs' es la observación del entorno base (la posición de la hormiga)
        ant_pos = obs

        # Accedemos al entorno original (desenvuelto) para obtener la posición de la comida
        food_pos = self.unwrapped.food_pos

        # Devolvemos el vector como la nueva observación
        return food_pos - ant_pos


# --- Wrapper del Manifiesto: Dirección al "hogar" (0,0) añadido a la observación ---


class DirectionToHomeWrapper(gym.ObservationWrapper):
    """Añade a la observación el ángulo en radianes hacia el hormiguero (0,0)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.home_pos = np.array([0, 0])

        # Definimos el NUEVO espacio de observación (x, y, ángulo)
        low = np.append(self.observation_space.low, -np.pi)
        high = np.append(self.observation_space.high, np.pi)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ant_pos = obs
        direction_vector = self.home_pos - ant_pos
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        new_obs = np.append(obs, angle).astype(np.float32)
        return new_obs

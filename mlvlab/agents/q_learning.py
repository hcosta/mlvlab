from __future__ import annotations

from typing import Any
import numpy as np
import random
import gymnasium as gym

from .base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
    ) -> None:
        super().__init__(observation_space, action_space)

        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)

        # Para compatibilidad con entornos grid (obs=(x,y)), permitimos una función externa
        # de traducción a estado discreto. Por defecto, identidad
        self.extract_state_from_obs = None  # type: ignore

        # Inicialización perezosa de Q-Table dependiendo del tipo de observación
        self._q_table: np.ndarray | None = None

        # Si es Discrete, sabemos el tamaño directamente
        if isinstance(self.observation_space, gym.spaces.Discrete):
            num_states = int(self.observation_space.n)
            self._q_table = np.zeros((num_states, int(self.action_space.n)))

    # Propiedad para exponer la q_table (usada por la UI para el heatmap)
    @property
    def q_table(self) -> np.ndarray:
        if self._q_table is None:
            # Intento de inferencia para entornos grid: usar GRID_SIZE
            grid_size = getattr(self, 'GRID_SIZE', None)
            if grid_size is None:
                # fallback mínimo
                raise RuntimeError(
                    "Q-Table no inicializada y no se puede inferir tamaño de estado.")
            num_states = int(grid_size) * int(grid_size)
            self._q_table = np.zeros((num_states, int(self.action_space.n)))
        return self._q_table

    def _ensure_initialized_from_obs(self, obs: Any) -> int | Any:
        """Convierte la observación a estado discreto y asegura Q-Table lista si es necesario."""
        # 1) Traductor externo si existe
        if callable(self.extract_state_from_obs):
            state = self.extract_state_from_obs(obs)  # type: ignore
        else:
            state = obs
        # 2) Si es un par (x,y) y hay GRID_SIZE, podemos derivar índice
        if isinstance(state, (tuple, list, np.ndarray)) and len(state) >= 2:
            grid_size = getattr(self, 'GRID_SIZE', None)
            if grid_size is not None:
                x, y = int(state[0]), int(state[1])
                state_index = int(y) * int(grid_size) + int(x)
                # Inicializar Q-Table si hace falta
                if self._q_table is None:
                    self._q_table = np.zeros(
                        (int(grid_size) * int(grid_size), int(self.action_space.n)))
                return state_index
        return state

    def act(self, obs: Any) -> int:
        state = self._ensure_initialized_from_obs(obs)
        if random.uniform(0, 1) < float(self.epsilon):
            return int(self.action_space.sample())
        try:
            # type: ignore[index]
            return int(np.argmax(self.q_table[int(state), :]))
        except Exception:
            # fallback si no es discreto indexable
            return int(self.action_space.sample())

    def learn(self, obs: Any, action: int, reward: float, next_obs: Any, done: bool) -> None:
        state = self._ensure_initialized_from_obs(obs)
        next_state = self._ensure_initialized_from_obs(next_obs)

        # Q-learning update
        try:
            old_value = float(self.q_table[int(state), int(action)])
            next_max = float(np.max(self.q_table[int(next_state), :]))
            new_value = old_value + float(self.learning_rate) * (
                float(reward) + float(self.discount_factor) * next_max - old_value)
            self.q_table[int(state), int(action)] = new_value
        except Exception:
            # Si no es indexable, no aprendemos (no compatible con Q-table)
            pass

        if done:
            self.epsilon = max(float(self.min_epsilon), float(
                self.epsilon) * float(self.epsilon_decay))

    def save(self) -> dict:
        """
        Empaqueta todos los datos necesarios del agente en un diccionario.
        """
        return {
            'q_table': self.q_table,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon': self.epsilon,
            }
        }

    def load(self, data: dict) -> None:
        """
        Restaura el estado del agente a partir de un diccionario.
        """
        # Restaurar Q-Table
        self._q_table = data.get('q_table')

        # Cargar hiperparámetros si se proporcionan
        hparams = data.get('hyperparameters', {}) if isinstance(
            data, dict) else {}
        if isinstance(hparams, dict):
            self.learning_rate = float(hparams.get(
                'learning_rate', self.learning_rate))
            self.discount_factor = float(hparams.get(
                'discount_factor', self.discount_factor))
            self.epsilon_decay = float(hparams.get(
                'epsilon_decay', self.epsilon_decay))
            self.epsilon = float(hparams.get('epsilon', self.epsilon))

    def reset(self) -> None:
        """Reinicia el conocimiento del agente poniendo la Q-Table a cero."""
        try:
            self.q_table.fill(0)  # fuerza inicialización si era perezosa
        except Exception:
            pass

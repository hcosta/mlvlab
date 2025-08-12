from __future__ import annotations

from abc import ABC, abstractmethod
import gymnasium as gym


class BaseAgent(ABC):
    """
    Define el contrato que todo agente en mlvlab debe cumplir.
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__()

    @abstractmethod
    def act(self, obs: object) -> int:
        """Dada una observación o estado discreto, devuelve una acción (int)."""
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        obs: object,
        action: int,
        reward: float,
        next_obs: object,
        done: bool,
    ) -> None:
        """
        El agente aprende de una transición de experiencia.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Guarda el estado interno del agente."""
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Carga el estado interno del agente."""
        raise NotImplementedError

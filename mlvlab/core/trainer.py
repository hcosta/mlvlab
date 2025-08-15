from __future__ import annotations

from typing import Callable, Optional, Any
import inspect
import gymnasium as gym

from mlvlab.agents.base import BaseAgent

# Tipos para las funciones que implementarán los alumnos
EpisodeLogicFn = Callable[[gym.Env, BaseAgent], float]
StateFromObsFn = Callable[[Any, gym.Env], Any]


class Trainer:
    """Orquesta el proceso de entrenamiento usando funciones de lógica personalizadas."""

    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        episode_logic: EpisodeLogicFn,
        obs_to_state: Optional[StateFromObsFn] = None,
    ):
        """
        Inicializa el Trainer.

        Args:
            env: El entorno de Gymnasium.
            agent: El agente que aprenderá.
            episode_logic: La función que define el bucle de un episodio.
            obs_to_state: La función que convierte una observación del entorno a un estado.
                          Si no se proporciona, se asume que la observación es el estado.
        """
        self.env = env
        self.agent = agent
        self._logic = episode_logic

        # Guardamos la función de conversión obs->state.
        # Si no se proporciona, usamos una función identidad.
        if obs_to_state and callable(obs_to_state):
            # Creamos un lambda para unificar la firma, pasando el entorno automáticamente.
            self._state_from_obs = lambda obs: obs_to_state(obs, self.env)
        else:
            self._state_from_obs = lambda obs: obs

        # La lógica de episodio es ahora más simple. Se asume que el alumno gestionará
        # la conversión de estado dentro de su propia función episode_logic,
        # usando el adaptador que le proporcionamos.
        # Para mantenerlo simple, la lógica del episodio solo recibe env y agent.
        # El alumno puede usar self.state_from_obs si lo necesita.
        self.run_one_episode = lambda: self._logic(self.env, self.agent)

    @property
    def state_from_obs(self) -> Callable[[Any], Any]:
        """
        Devuelve el adaptador obs->state asociado al Trainer.
        Esta propiedad permite que otras partes del sistema (como AnalyticsView)
        accedan a la función de conversión.
        """
        return self._state_from_obs

# mlvlab/core/trainer.py
from __future__ import annotations
from typing import Callable, Optional, Any, Type
import gymnasium as gym
from mlvlab.agents.base import BaseAgent
# Asegúrate de que InteractiveLogic sea importable, por ejemplo, desde mlvlab.core.logic
from .logic import InteractiveLogic


class Trainer:
    """Orquesta el proceso de entrenamiento usando una clase de lógica interactiva."""

    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        # CAMBIO: Recibimos una clase, no una función
        logic_class: Type[InteractiveLogic],
    ):
        """
        Inicializa el Trainer.
        Args:
            env: El entorno de Gymnasium.
            agent: El agente que aprenderá.
            logic_class: La CLASE (no una instancia) que hereda de InteractiveLogic.
        """
        self.env = env
        self.agent = agent

        # CAMBIO: El Trainer ahora crea y posee la instancia de la lógica.
        # Esto encapsula la lógica del alumno en un objeto manejable.
        self.logic = logic_class(self.env, self.agent)

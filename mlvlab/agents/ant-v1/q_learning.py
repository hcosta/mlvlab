"""Entrenamiento y evaluación baseline para mlvlab/ant-v1 usando Q-Learning genérico.

Este módulo implementa las funciones `train_agent` y `eval_agent` que la CLI
espera encontrar para el entorno `mlvlab/ant-v1` según la configuración BASELINE.
"""

import os
import numpy as np
import random
import gymnasium as gym
from pathlib import Path
from typing import Optional
from rich.progress import track

from mlvlab.helpers.train import train_with_state_adapter
from mlvlab.helpers.eval import evaluate_with_optional_recording
from mlvlab.agents.q_learning import QLearningAgent
from .state import obs_to_state
from pathlib import Path as _Path

# Ruta de fuente para overlay opcional en los vídeos
_FONT_PATH = str(_Path(__file__).parent.parent /
                 "assets" / "fonts" / "Roboto-Regular.ttf")


def _agent_builder(env: gym.Env) -> QLearningAgent:
    """Crea el agente Q-Learning genérico para Ant usando el tamaño de la rejilla."""
    grid_size = int(env.unwrapped.GRID_SIZE)
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
    )
    # Registrar GRID_SIZE para posibles adaptaciones internas
    setattr(agent, 'GRID_SIZE', grid_size)
    return agent

# --- Función de Entrenamiento Estandarizada (Contrato para la CLI) ---


def train_agent(
    env_id: str,
    config: dict,
    run_dir: Path,
    seed: int | None = None,
    render: bool = False
):
    """
    Entrena un agente Q-Learning y guarda la Q-Table en la carpeta del 'run'.
    """
    TOTAL_EPISODES = int(config['episodes'])
    alpha = float(config['alpha'])
    gamma = float(config['gamma'])
    epsilon_decay = float(config['epsilon_decay'])
    min_epsilon = float(config['min_epsilon'])

    def state_adapter(obs, env: gym.Env) -> int:
        grid = env.unwrapped.GRID_SIZE
        return obs_to_state(int(obs[0]), int(obs[1]), int(grid))

    def on_render(env: gym.Env, agent: QLearningAgent) -> None:
        try:
            if hasattr(env.unwrapped, "set_render_data"):
                env.unwrapped.set_render_data(q_table=agent.q_table)
        except Exception:
            pass

    def builder_with_hparams(env: gym.Env) -> QLearningAgent:
        agent = _agent_builder(env)
        agent.learning_rate = alpha
        agent.discount_factor = gamma
        agent.epsilon_decay = epsilon_decay
        agent.min_epsilon = min_epsilon
        agent.epsilon = 1.0
        return agent

    # Entrena usando helper reutilizable
    train_with_state_adapter(
        env_id=env_id,
        run_dir=run_dir,
        total_episodes=TOTAL_EPISODES,
        agent_builder=builder_with_hparams,
        state_adapter=state_adapter,
        seed=seed,
        render=render,
        on_render_frame=on_render,
    )


def eval_agent(
    env_id: str,
    run_dir: Path,
    episodes: int,
    seed: Optional[int] = None,
    cleanup: bool = True,
    video: bool = False,
):
    """
    Carga una Q-Table de un 'run' y evalúa al agente, guardando un vídeo fijo.
    """
    # Reutiliza helper de evaluación/grabación
    def builder(env: gym.Env) -> QLearningAgent:
        agent = _agent_builder(env)
        # Cargar Q-Table si existe (el helper también lo intentará)
        q_table_file = run_dir / "q_table.npy"
        try:
            agent.load(str(q_table_file))
        except Exception:
            try:
                # type: ignore[attr-defined]
                agent._q_table = np.load(q_table_file)
            except Exception:
                pass
        return agent

    evaluate_with_optional_recording(
        env_id=env_id,
        run_dir=run_dir,
        episodes=int(episodes),
        agent_builder=builder,
        seed=seed,
        record=video,
        cleanup=cleanup,
        overlay_font_path=_FONT_PATH,
    )

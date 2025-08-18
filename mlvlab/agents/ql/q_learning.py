"""Entrenamiento y evaluación baseline para mlv/ql/ant-v1 usando Q-Learning genérico.

Implementa `train_agent` y `eval_agent`, que la CLI invoca según la configuración BASELINE
del entorno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.helpers.train import train_with_state_adapter
from mlvlab.helpers.eval import evaluate_with_optional_recording


def _agent_builder(env: gym.Env) -> QLearningAgent:
    grid_size = int(env.unwrapped.GRID_SIZE)
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
    )
    setattr(agent, "GRID_SIZE", grid_size)
    return agent


def train_agent(
    env_id: str,
    config: dict,
    run_dir: Path,
    seed: int | None = None,
    render: bool = False,
):
    total_episodes = int(config["episodes"])
    alpha = float(config["alpha"])
    gamma = float(config["gamma"])
    epsilon_decay = float(config["epsilon_decay"])
    min_epsilon = float(config["min_epsilon"])

    def state_adapter(obs, env: gym.Env) -> int:
        grid = int(env.unwrapped.GRID_SIZE)
        return int(obs[1]) * grid + int(obs[0])

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

    train_with_state_adapter(
        env_id=env_id,
        run_dir=run_dir,
        total_episodes=total_episodes,
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
    video: bool = False,
):
    def builder(env: gym.Env) -> QLearningAgent:
        agent = _agent_builder(env)
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
        record=video
    )

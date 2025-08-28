# mlvlab/algorithms/ql/plugin.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym

from mlvlab.algorithms.registry import register_algorithm
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.helpers.train import train_with_state_adapter
from mlvlab.helpers.eval import evaluate_with_optional_recording


class QLearningPlugin:
    def key(self) -> str:
        return "ql"

    def build_agent(self, env: gym.Env, hparams: Dict[str, Any]) -> QLearningAgent:
        grid_size = int(getattr(env.unwrapped, "GRID_SIZE", 0))
        obs_space = env.observation_space
        # Si el env es de grid clásico, discretizar estados
        if grid_size and not isinstance(obs_space, gym.spaces.Discrete):
            obs_space = gym.spaces.Discrete(grid_size * grid_size)
        agent = QLearningAgent(
            observation_space=obs_space,
            action_space=env.action_space,
        )
        # Aplicar hiperparámetros si existen
        agent.learning_rate = float(hparams.get(
            "alpha", getattr(agent, "learning_rate", 0.1)))
        agent.discount_factor = float(hparams.get(
            "gamma", getattr(agent, "discount_factor", 0.99)))
        agent.epsilon_decay = float(hparams.get(
            "epsilon_decay", getattr(agent, "epsilon_decay", 0.999)))
        agent.min_epsilon = float(hparams.get(
            "min_epsilon", getattr(agent, "min_epsilon", 0.01)))
        agent.epsilon = float(hparams.get(
            "epsilon", getattr(agent, "epsilon", 1.0)))
        if grid_size:
            setattr(agent, "GRID_SIZE", grid_size)
        return agent

    def _resolve_state_adapter(self, env_id: str):
        """Busca adaptador declarado por convención y devuelve la función (obs, env)->state."""
        try:
            pkg = env_id.split('/')[-1]
            module_path = f"mlvlab.envs.{pkg.replace('-', '_')}.adapters"
            import importlib
            mod = importlib.import_module(module_path)
            fn = getattr(mod, 'obs_to_state', None)
            if callable(fn):
                return fn  # firma (obs, env)
        except Exception:
            pass
        return None

    def train(self, env_id: str, config: Dict[str, Any], run_dir: Path, seed: Optional[int] = None, render: bool = False) -> None:
        def agent_builder(env: gym.Env) -> QLearningAgent:
            return self.build_agent(env, config)

        def on_render(env: gym.Env, agent: QLearningAgent) -> None:
            try:
                if hasattr(env.unwrapped, "set_render_data"):
                    env.unwrapped.set_render_data(q_table=agent.q_table)
            except Exception:
                pass

        # Intentar adaptador declarado por convención (firma (obs, env))
        state_adapter = self._resolve_state_adapter(env_id)

        train_with_state_adapter(
            env_id=env_id,
            run_dir=run_dir,
            total_episodes=int(config.get("episodes", 1000)),
            agent_builder=agent_builder,
            state_adapter=(state_adapter or (lambda obs: obs)),
            seed=seed,
            render=render,
            on_render_frame=on_render,
        )

    def eval(self, env_id: str, run_dir: Path, **kwargs: Any) -> Optional[str]:
        def builder(env: gym.Env) -> QLearningAgent:
            return self.build_agent(env, {})

        # TRADUCTOR DE PARÁMETRO: de 'video' a 'record' ---
        # Comprobamos si el argumento 'video' viene en los kwargs
        if 'video' in kwargs:
            # Creamos la clave 'record' con el valor de 'video' y eliminamos la clave 'video'.
            kwargs['record'] = kwargs.pop('video')
        # ----------------------------------------------------

        # Ahora pasamos los argumentos corregidos a la función final
        evaluate_with_optional_recording(
            env_id=env_id,
            run_dir=run_dir,
            agent_builder=builder,
            **kwargs
        )
        return None


# Registrar plugin automáticamente al importar el módulo
register_algorithm(QLearningPlugin())

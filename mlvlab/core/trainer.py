from __future__ import annotations

from typing import Callable, Optional, Any
import inspect
import importlib
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import gymnasium as gym

from mlvlab.agents.base import BaseAgent

EpisodeLogicFn = Callable[[gym.Env, BaseAgent,
                           Optional[Callable[[Any], Any]]], float]


def _resolve_adapter_for_env(env: gym.Env) -> Optional[Callable[[Any], Any]]:
    """Resuelve un adaptador obs->state específico del entorno, si existe.

    Orden de búsqueda:
      1) `mlvlab.envs.<env_pkg_us>.adapters: obs_to_state(obs, env)`
      2) Compat: `mlvlab.agents.<env_pkg>.state: obs_to_state` o `obs_to_state_<env_pkg_us>`
    """
    try:
        env_id: str = getattr(getattr(env, 'spec', None), 'id', '')
        env_pkg = env_id.split('/')[-1] if '/' in env_id else env_id
        env_pkg_us = env_pkg.replace('-', '_')

        # 1) Preferido: mlvlab.envs.<pkg_us>.adapters
        mod = None
        try:
            module_path_env = f"mlvlab.envs.{env_pkg_us}.adapters"
            mod = importlib.import_module(module_path_env)
        except Exception:
            mod = None

        # 2) Compat: mlvlab.agents.<pkg>.state
        if mod is None:
            try:
                module_path_agents = f"mlvlab.agents.{env_pkg}.state"
                mod = importlib.import_module(module_path_agents)
            except Exception:
                # Fallback por ruta directa (paquetes con '-')
                base_dir = Path(__file__).resolve(
                ).parents[1] / 'agents' / env_pkg
                file_path = base_dir / 'state.py'
                if file_path.exists():
                    spec = spec_from_file_location(
                        "mlvlab_env_state_module", str(file_path))
                    if spec is not None and spec.loader is not None:
                        mod = module_from_spec(spec)
                        spec.loader.exec_module(mod)  # type: ignore
        if mod is None:
            return None

        fn = getattr(mod, 'obs_to_state', None)
        if callable(fn):
            return lambda obs: fn(obs, env)
        fn2 = getattr(mod, f"obs_to_state_{env_pkg_us}", None)
        if callable(fn2):
            return lambda obs: fn2(obs, env)
    except Exception:
        return None
    return None


def default_episode_logic(env: gym.Env, agent: BaseAgent, state_from_obs: Optional[Callable[[Any], Any]] = None) -> float:
    """Lógica por defecto para ejecutar un episodio sin adaptadores internos.

    Si `state_from_obs` no se proporciona, intentará resolverse automáticamente uno
    específico del entorno. Si no hay, se usa identidad.
    """
    adapter = state_from_obs or _resolve_adapter_for_env(
        env) or (lambda obs: obs)

    obs, info = env.reset()
    state = adapter(obs)

    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = adapter(next_obs)
        agent.learn(state, action, float(reward), next_state,
                    bool(terminated or truncated))
        state = next_state
        total_reward += float(reward)
        done = bool(terminated or truncated)

    return total_reward


class Trainer:
    """Orquesta el proceso de entrenamiento, permitiendo lógica de episodio personalizable."""

    def __init__(
        self,
        env: gym.Env,
        agent: BaseAgent,
        logic: EpisodeLogicFn = default_episode_logic,
        state_from_obs: Optional[Callable[[Any], Any]] = None,
    ):
        self.env = env
        self.agent = agent

        # Si se pasa la clase en lugar de una instancia, la instanciamos aquí.
        if inspect.isclass(logic):
            self._logic = logic()
        else:
            self._logic = logic
        # --- FIN DE LA MODIFICACIÓN ---

        # Si no se pasó state_from_obs, intentar extraerlo de logic.obs_to_state
        if state_from_obs is None:
            try:
                candidate = getattr(logic, 'obs_to_state', None)
                if callable(candidate):
                    self._state_from_obs = lambda obs: candidate(obs, self.env)
                else:
                    self._state_from_obs = None
            except Exception:
                self._state_from_obs = None
        else:
            self._state_from_obs = state_from_obs

        # Respetar firma de logic (2 o 3 parámetros)
        try:
            sig = inspect.signature(self._logic)
            num_params = len(sig.parameters)
        except Exception:
            num_params = 3

        if num_params >= 3:
            self.run_one_episode = lambda: self._logic(
                self.env, self.agent, self._state_from_obs)
        else:
            self.run_one_episode = lambda: self._logic(
                self.env, self.agent)

    def train(self, num_episodes: int) -> None:
        print(f"Iniciando entrenamiento para {num_episodes} episodios...")
        for episode in range(num_episodes):
            episode_reward = self.run_one_episode()
            if (episode + 1) % 1000 == 0:
                print(
                    f"Episodio {episode + 1}: Recompensa Total = {episode_reward}")
        print("Entrenamiento finalizado.")
        self.env.close()

    @property
    def state_from_obs(self) -> Optional[Callable[[Any], Any]]:
        """Devuelve el adaptador obs->state asociado al Trainer (si existe)."""
        return self._state_from_obs

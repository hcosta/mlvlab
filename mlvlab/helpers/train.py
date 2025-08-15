from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from rich.progress import track


def train_with_state_adapter(
    env_id: str,
    run_dir: Path,
    total_episodes: int,
    agent_builder: Callable[[gym.Env], object],
    state_adapter: Callable[[object, gym.Env], int] | Callable[[object], int],
    seed: Optional[int] = None,
    render: bool = False,
    on_render_frame: Optional[Callable[[gym.Env, object], None]] = None,
) -> Path:
    """
    Entrena un agente... (ver docstring original)
    """
    env = gym.make(env_id, render_mode=("human" if render else None))
    if render:
        env.unwrapped.debug_mode = True

    # --- AJUSTE DE ALEATORIEDAD PARA ENTRENAMIENTO ---
    # Para Q-Learning 1:1, necesitamos que el mapa sea fijo (determinado por la semilla)
    # pero que la posición inicial sea aleatoria en cada episodio para explorar el espacio de estados.
    # Si el entorno lo soporta, activamos el modo de respawn aleatorio (unseeded).
    try:
        if hasattr(env.unwrapped, "set_respawn_unseeded"):
            env.unwrapped.set_respawn_unseeded(True)
            print("ℹ️  Configurado respawn aleatorio (unseeded) para entrenamiento.")
    except Exception as e:
        print(
            f"⚠️ Advertencia: No se pudo configurar el respawn aleatorio: {e}")
    # --------------------------------------------------

    # Construir agente
    agent = agent_builder(env)

    # Reset inicial con semilla si se proporciona (Fija el mapa)
    obs, info = env.reset(seed=seed)

    # Parámetros por defecto: obtenidos del propio agente si existen
    alpha = float(getattr(agent, "learning_rate", 0.1))
    gamma = float(getattr(agent, "discount_factor", 0.99))
    epsilon = float(getattr(agent, "epsilon", 1.0))
    epsilon_decay = float(getattr(agent, "epsilon_decay", 0.999))
    min_epsilon = float(getattr(agent, "min_epsilon", 0.01))

    grid_size = getattr(env.unwrapped, "GRID_SIZE", None)

    def adapt(o):
        try:
            # state_adapter puede aceptar (obs, env) o (obs)
            # type: ignore[attr-defined]
            if state_adapter.__code__.co_argcount >= 2:
                return state_adapter(o, env)
            return state_adapter(o)  # type: ignore[misc]
        except Exception:
            # fallback: si env es grid, derivar índice (x,y) -> idx
            if grid_size is not None and hasattr(o, "__getitem__") and len(o) >= 2:
                return int(o[1]) * int(grid_size) + int(o[0])
            return o

    for episode in track(range(total_episodes), description="Entrenando..."):
        if episode > 0:
            # En episodios subsiguientes, reset() reutiliza el mapa pero randomiza la posición (gracias al ajuste inicial)
            obs, info = env.reset()

        terminated, truncated = False, False
        while not (terminated or truncated):
            if render:
                if callable(on_render_frame):
                    try:
                        on_render_frame(env, agent)
                    except Exception:
                        pass
                try:
                    env.render()
                except Exception:
                    pass

            state = adapt(obs)
            # Explorar/Explotar (si el agente soporta act con epsilon interno)
            try:
                action = int(agent.act(state))
            except Exception:
                # Compatibilidad: choose_action(state, epsilon)
                # type: ignore[attr-defined]
                action = int(agent.choose_action(state, float(epsilon)))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = adapt(next_obs)

            # Aprendizaje (preferimos API nueva)
            try:
                agent.learn(state, action, float(reward),
                            next_state, bool(terminated or truncated))
            except Exception:
                agent.update(state, action, float(reward), next_state, float(
                    alpha), float(gamma))  # type: ignore[attr-defined]

            obs = next_obs

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
            try:
                setattr(agent, "epsilon", float(epsilon))
            except Exception:
                pass

    env.close()

    # Guardar Q-Table si existe
    q_path = run_dir / "q_table.npy"
    try:
        import numpy as np
        q_table = getattr(agent, "q_table", None)
        if q_table is not None:
            np.save(q_path, q_table)
            print(f"✅ Entrenamiento completado. Q-Table guardada en: {q_path}")
            return q_path
    except Exception:
        pass
    print("⚠️ Entrenamiento completado, pero el agente no expone q_table. No se guardó archivo.")
    return q_path

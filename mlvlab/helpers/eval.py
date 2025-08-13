from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from rich.progress import track

from .video import merge_videos_with_counter


def evaluate_with_optional_recording(
    env_id: str,
    run_dir: Path,
    episodes: int,
    agent_builder: Callable[[gym.Env], object],
    seed: Optional[int] = None,
    record: bool = False,
    cleanup: bool = True,
) -> Optional[Path]:
    """
    Evalúa un agente construido por `agent_builder` en `env_id` durante `episodes` episodios.
    Si `record=True`, graba cada episodio en `run_dir/evaluation_videos_temp` y genera un
    vídeo final `run_dir/evaluation.mp4`. Devuelve la ruta al vídeo si se genera.

    Requisitos del agente:
      - Método `act(obs_or_state) -> int`
      - (Opcional) atributo `epsilon` (para forzar política greedy)
      - (Opcional) método `load(filepath)` para cargar estado (p.ej., Q-Table)
      - (Opcional) propiedad `q_table` para overlay de heatmap
    """
    temp_folder = run_dir / "evaluation_videos_temp"
    final_video_path = run_dir / "evaluation.mp4"

    # Crear entorno según modo
    if record:
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, str(temp_folder),
                          episode_trigger=lambda x: True)
    else:
        env = gym.make(env_id, render_mode="human")

    # Construcción del agente (el builder debe configurarlo para este env)
    agent = agent_builder(env)

    # Intentar cargar estado del agente desde run_dir estandarizado
    q_table_file = run_dir / "q_table.npy"
    try:
        if hasattr(agent, "load") and q_table_file.exists():
            agent.load(str(q_table_file))  # type: ignore[attr-defined]
            print(f"🧠 Cargado estado del agente desde {q_table_file}.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar el estado del agente: {e}")

    # Forzar política greedy en evaluación si el agente soporta epsilon
    try:
        if hasattr(agent, "epsilon"):
            setattr(agent, "epsilon", 0.0)
    except Exception:
        pass

    # Bucle de evaluación
    for ep in track(range(episodes), description="Evaluando..."):
        current_seed = seed if ep == 0 else None
        obs, info = env.reset(seed=current_seed)

        terminated, truncated = False, False
        while not (terminated or truncated):
            # Overlay de Q-Table si existe
            try:
                q_table = getattr(agent, "q_table", None)
                if q_table is not None and hasattr(env.unwrapped, "set_render_data"):
                    env.unwrapped.set_render_data(q_table=q_table)
            except Exception:
                pass

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if not record:
                env.render()

    env.close()

    # Manejo de vídeo
    if not record:
        print("✅ Evaluación completada en modo interactivo (sin grabación).")
        return None

    ok = merge_videos_with_counter(
        str(temp_folder),
        str(final_video_path),
        font_path=None,
        cleanup=cleanup,
    )
    if cleanup:
        if os.path.exists(temp_folder):
            try:
                import shutil
                shutil.rmtree(temp_folder)
                print("🗑️ Archivos temporales eliminados.")
            except Exception:
                pass
    else:
        print(f"ℹ️ Se conservan los archivos temporales en: {temp_folder}")

    if ok:
        print(
            f"✅ Evaluación completada. Vídeo guardado en: {final_video_path}")
        return final_video_path
    else:
        return None

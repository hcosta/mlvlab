# mlvlab/evaluation/eval.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional
import time
import shutil
import sys

import gymnasium as gym
import imageio  # Necesario para la grabación manual
from rich.progress import track
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    """Un gestor de contexto para suprimir temporalmente la salida a stderr."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Asumiendo que merge_videos_with_counter está disponible, aunque no lo usaremos en este flujo
try:
    from .video import merge_videos_with_counter
except ImportError:
    def merge_videos_with_counter(*args, **kwargs):
        print("⚠️ Utilidad de vídeo no encontrada.")
        return False


def evaluate_with_optional_recording(
    env_id: str,
    run_dir: Path,
    episodes: int,
    agent_builder: Callable[[gym.Env], object],
    seed: Optional[int] = None,
    record: bool = False,
    speed: float = 1.0,
) -> Optional[Path]:
    """
    Evalúa un agente y, si record=True, graba un vídeo de la evaluación de forma manual
    para asegurar la compatibilidad con renderers cinemáticos dependientes del tiempo.
    """
    final_video_path = run_dir / "evaluation.mp4"

    # --- LÓGICA DE RENDERIZADO Y GRABACIÓN RECTIFICADA ---

    render_mode = "human" if not record else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)

    # Activar el modo debug si está disponible, para visualizaciones extra
    if hasattr(env.unwrapped, "debug_mode"):
        env.unwrapped.debug_mode = True

    # Configuración de respawn determinista para una evaluación consistente
    if hasattr(env.unwrapped, "set_respawn_unseeded"):
        try:
            env.unwrapped.set_respawn_unseeded(False)
            print("ℹ️  Configurado respawn determinista (seeded) para evaluación.")
        except Exception as e:
            print(
                f"⚠️ Advertencia: No se pudo configurar el respawn determinista: {e}")

    # Construcción del agente
    agent = agent_builder(env)

    # Carga del estado del agente (p.ej., Tabla Q)
    agent_file = run_dir / "q_table.npy"  # Asumiendo un nombre estándar
    if hasattr(agent, "load") and agent_file.exists():
        try:
            agent.load(str(agent_file))
            print(f"🧠 Cargado estado del agente desde {agent_file}.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el estado del agente: {e}")

    # Forzar modo explotación (sin acciones aleatorias)
    if hasattr(agent, "epsilon"):
        setattr(agent, "epsilon", 0.0)

    # --- BUCLE DE EVALUACIÓN ---

    frames = []  # Lista para guardar los fotogramas si estamos grabando

    for ep in track(range(episodes), description="Evaluando..."):
        current_seed = seed if ep == 0 else None
        obs, info = env.reset(seed=current_seed)

        terminated, truncated = False, False
        while not (terminated or truncated):

            # Pasar datos de renderizado al entorno si es necesario (p.ej. Q-Table)
            q_table = getattr(agent, "q_table", None)
            if q_table is not None and hasattr(env.unwrapped, "set_render_data"):
                env.unwrapped.set_render_data(q_table=q_table)

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if record:
                # Grabación manual del frame
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            else:
                # Renderizado en tiempo real para el modo humano
                env.render()

            # PAUSA CRUCIAL para dar tiempo al renderer cinemático
            target_fps = env.metadata.get("render_fps", 60)
            effective_speed = max(speed, 0.01)
            delay = (1.0 / target_fps) / effective_speed
            time.sleep(delay)

    # --- GRABACIÓN DE LA ESCENA FINAL CINEMÁTICA (SOLO EN MODO RECORD) ---
    if record:
        print("Grabando escena final cinemática...")
        # Grabamos durante 0.5 segundos extra para capturar la animación completa del renderer
        num_final_frames = int(env.metadata.get("render_fps", 60) * 0.5)
        for _ in range(num_final_frames):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            time.sleep(1/env.metadata.get("render_fps", 60))

        # Guardar el vídeo usando imageio
        if frames:
            effective_speed = max(speed, 0.01)
            playback_fps = target_fps * effective_speed

            print(
                f"Guardando vídeo a {playback_fps:.2f} FPS (velocidad x{speed})...")
            try:
                # Aquí podrías usar tu supresor de warnings si quieres
                imageio.mimsave(str(final_video_path),
                                frames, fps=playback_fps)
                print(
                    f"✅ Evaluación completada. Vídeo guardado en: {final_video_path}")
            except Exception as e:
                print(f"❌ Error al guardar el vídeo con imageio: {e}")
                final_video_path = None
        else:
            print("⚠️ No se generaron frames para el vídeo.")
            final_video_path = None

    env.close()

    if not record:
        print("✅ Evaluación completada en modo interactivo.")
        return None

    return final_video_path

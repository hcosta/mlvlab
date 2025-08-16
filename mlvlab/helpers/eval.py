# mlvlab/evaluation/eval.py

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import cv2  # Para añadir texto a los fotogramas
import gymnasium as gym
import imageio
import numpy as np
from rich.progress import track


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

# =============================================================================
# FUNCIÓN DE UTILIDAD PARA EL SELLO DE TEXTO
# =============================================================================


def add_stamp_to_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Añade un sello de texto a un fotograma usando OpenCV."""
    stamped_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # Blanco
    shadow_color = (0, 0, 0)      # Negro para la sombra

    position = (15, 30)
    shadow_position = (17, 32)

    # Dibujar la sombra primero para mejor legibilidad
    cv2.putText(stamped_frame, text, shadow_position, font,
                font_scale, shadow_color, font_thickness, cv2.LINE_AA)
    # Dibujar el texto principal
    cv2.putText(stamped_frame, text, position, font, font_scale,
                text_color, font_thickness, cv2.LINE_AA)

    return stamped_frame

# =============================================================================


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
    Evalúa un agente y, si record=True, graba un vídeo de forma manual,
    compatible con renderers cinemáticos y con un sello de texto.
    """
    final_video_path = run_dir / "evaluation.mp4"
    render_mode = "human" if not record else "rgb_array"
    env = gym.make(env_id, render_mode=render_mode)

    if hasattr(env.unwrapped, "debug_mode"):
        env.unwrapped.debug_mode = True

    if hasattr(env.unwrapped, "set_respawn_unseeded"):
        try:
            env.unwrapped.set_respawn_unseeded(False)
            print("ℹ️  Configurado respawn determinista (seeded) para evaluación.")
        except Exception as e:
            print(
                f"⚠️ Advertencia: No se pudo configurar el respawn determinista: {e}")

    agent = agent_builder(env)
    agent_file = run_dir / "q_table.npy"
    if hasattr(agent, "load") and agent_file.exists():
        try:
            agent.load(str(agent_file))
            print(f"🧠 Cargado estado del agente desde {agent_file}.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el estado del agente: {e}")

    if hasattr(agent, "epsilon"):
        setattr(agent, "epsilon", 0.0)

    frames = []
    target_fps = env.metadata.get("render_fps", 60)

    for ep in track(range(episodes), description="Evaluando..."):
        current_seed = seed if ep == 0 else None
        obs, info = env.reset(seed=current_seed)
        terminated, truncated = False, False

        stamp_text = f"Episodio: {ep + 1}/{episodes}"

        while not (terminated or truncated):
            q_table = getattr(agent, "q_table", None)
            if q_table is not None and hasattr(env.unwrapped, "set_render_data"):
                env.unwrapped.set_render_data(q_table=q_table)

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if record:
                frame = env.render()
                if frame is not None:
                    frame_with_stamp = add_stamp_to_frame(frame, stamp_text)
                    frames.append(frame_with_stamp)
            else:
                env.render()

            delay = (1.0 / target_fps) / max(speed, 0.01)
            time.sleep(delay)

    if record:
        print("Grabando escena final cinemática...")
        # 0.25 segundos de animación final
        num_final_frames = int(target_fps * .25)
        for _ in range(num_final_frames):
            frame = env.render()
            if frame is not None:
                frame_with_stamp = add_stamp_to_frame(frame, stamp_text)
                frames.append(frame_with_stamp)
            time.sleep(1 / target_fps)

        if frames:
            playback_fps = target_fps * max(speed, 0.01)
            print(
                f"Guardando vídeo a {playback_fps:.2f} FPS (velocidad x{speed})...")
            try:
                with suppress_stderr():
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

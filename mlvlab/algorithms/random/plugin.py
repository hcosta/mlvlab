from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import time
import gymnasium as gym
from tqdm import tqdm
from mlvlab.i18n.core import i18n
from mlvlab.algorithms.registry import register_algorithm
import sys  # 1. Importar el módulo 'sys'


class RandomAgent:
    """Un agente simple que solo tiene un método: tomar una acción aleatoria."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation: Any) -> Any:
        return self.action_space.sample()


class RandomPlugin:
    """
    Implementación del protocolo AlgorithmPlugin para un agente
    que ejecuta acciones aleatorias. No aprende ni guarda nada.
    """

    def key(self) -> str:
        return "random"

    def build_agent(self, env: gym.Env, hparams: Dict[str, Any]) -> RandomAgent:
        """Crea nuestro agente simple."""
        return RandomAgent(env.action_space)

    def train(self, env_id: str, config: Dict[str, Any], run_dir: Path, seed: Optional[int] = None, render: bool = False) -> None:
        """Simula un 'entrenamiento' ejecutando episodios con acciones aleatorias."""
        print(i18n.t("cli.messages.random_training_start"))
        episodes = int(config.get('episodes', 10))
        env = gym.make(env_id, render_mode="human" if render else None)
        agent = self.build_agent(env, {})

        for _ in tqdm(range(episodes), desc=i18n.t("common.training"), ncols=50, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                    time.sleep(0.05)  # Pequeña pausa para poder ver algo

        env.close()
        print(i18n.t("cli.messages.random_training_end"))

    def eval(self, env_id: str, run_dir: Path, **kwargs: Any) -> Optional[str]:
        """Simula una 'evaluación' ejecutando episodios con acciones aleatorias."""
        print(i18n.t("cli.messages.random_eval_start"))
        env = gym.make(env_id, render_mode="human")
        agent = self.build_agent(env, {})

        episodes = int(kwargs.get("episodes", 5))
        seed = kwargs.get("seed")
        speed = float(kwargs.get("speed", 1.0))
        # Ajustamos el frame_delay base para que sea más lento por defecto
        # Usamos 30 FPS en lugar de 60 FPS para una velocidad más natural
        base_fps = 30.0
        frame_delay = (1.0 / base_fps) / speed if speed > 0 else 0

        # --- 2. LÓGICA DE PROGRESIÓN MODIFICADA ---
        for episode in range(episodes):
            # Construimos el texto que queremos mostrar
            progress_text = f"{i18n.t('common.evaluating_episode')}: {episode + 1}/{episodes}"
            # Escribimos en la salida estándar sin saltar de línea y volvemos al principio
            sys.stdout.write(progress_text + "\r")
            sys.stdout.flush()

            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if frame_delay > 0:
                    time.sleep(frame_delay)

        # 3. Añadimos un print() vacío al final para saltar de línea y no sobreescribir la última
        print()
        # --- FIN DE LA MODIFICACIÓN ---

        env.close()
        print(i18n.t("cli.messages.random_eval_end"))
        return None


# Registrar el plugin automáticamente al importar el módulo
register_algorithm(RandomPlugin())

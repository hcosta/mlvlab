# mlvlab/agents/q_learning.py
import os
import numpy as np
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from pathlib import Path
from typing import Optional
from rich.progress import track

FONT_PATH = str(Path(__file__).parent.parent /
                "assets" / "fonts" / "Roboto-Regular.ttf")


def _merge_videos(video_folder: str, output_filename: str, cleanup: bool = True):
    """
    Une todos los vídeos, añadiendo un overlay de texto usando la sintaxis
    correcta para MoviePy 2.2.1.
    """
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    video_files.sort()

    if not video_files:
        print("ℹ️ No se encontraron vídeos para unir.")
        return

    clips_originales = []
    clips_con_texto = []

    try:
        total_episodes = len(video_files)
        print(f"📹 Procesando {total_episodes} vídeos para añadir texto...")

        for i, filename in enumerate(video_files, 1):
            filepath = os.path.join(video_folder, filename)
            try:
                clip = VideoFileClip(filepath)
                clips_originales.append(clip)

                texto = f"{i}/{total_episodes}  \n"

                txt_clip = TextClip(text=texto, font=FONT_PATH,
                                    font_size=24, color='white')

                # --- CORRECCIÓN FINAL ---
                # Cambiamos .set_position por .with_position y .set_duration por .with_duration
                txt_clip = txt_clip.with_position(
                    ('right', 'bottom')).with_duration(clip.duration)

                video_con_texto = CompositeVideoClip([clip, txt_clip])
                clips_con_texto.append(video_con_texto)
            except Exception as e:
                print(f"- {filename:25} | ❌ Error al procesar clip: {e}")

        if not clips_con_texto:
            print("🛑 No se pudieron procesar los vídeos.")
            return

        print(
            f"🎞️ Uniendo {len(clips_con_texto)} vídeos en '{output_filename}'...")
        final_clip = concatenate_videoclips(clips_con_texto, method="compose")
        final_clip.write_videofile(output_filename, logger='bar')

    except Exception as e:
        print(f"🛑 Error durante la creación del vídeo final: {e}")
    finally:
        for clip in clips_con_texto:
            clip.close()
        for clip in clips_originales:
            clip.close()

    if os.path.exists(output_filename):
        print(f"✅ Vídeo unificado y con texto creado con éxito.")
        # Solo borra los ficheros si cleanup es True
        if cleanup:
            for f in video_files:
                os.remove(os.path.join(video_folder, f))
    else:
        print("❌ Error: El vídeo final no se pudo crear.")


def get_state_from_pos(x, y, grid_size):
    """
    Convierte coordenadas (x, y) a un estado único en la tabla Q.
    Es una función de utilidad que pertenece conceptualmente al agente,
    ya que es él quien necesita traducir la observación del entorno a un índice.
    """
    # Programación defensiva: Comprobamos si las coordenadas son válidas.
    # Esto te ayudará a encontrar errores en el futuro.
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        raise ValueError(
            f"Coordenadas ({x}, {y}) fuera de los límites para una rejilla de {grid_size}x{grid_size}")

    # La fórmula correcta
    return y * grid_size + x


class QLearningAgent:
    """
    Un agente que aprende a tomar decisiones usando el algoritmo Q-Learning.
    """

    def __init__(self, num_states, num_actions):
        """
        Inicializa la Q-Table, que almacenará los valores de cada par estado-acción.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state, epsilon):
        """
        Decide qué acción tomar usando una política epsilon-greedy.
        - Con probabilidad epsilon, elige una acción al azar (exploración).
        - Con probabilidad 1-epsilon, elige la mejor acción conocida (explotación).
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state, alpha, gamma):
        """
        Actualiza la Q-Table usando la ecuación de Bellman.
        - alpha: Tasa de aprendizaje (learning rate).
        - gamma: Factor de descuento (discount factor).
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])

        # La fórmula central de Q-Learning.
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.q_table[state, action] = new_value

    def reset(self):
        """Reinicia el conocimiento del agente poniendo la Q-Table a cero."""
        self.q_table.fill(0)

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
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    obs, info = env.reset(seed=seed)

    GRID_SIZE = env.unwrapped.GRID_SIZE
    TOTAL_EPISODES = config['episodes']

    agent = QLearningAgent(num_states=GRID_SIZE * GRID_SIZE,
                           num_actions=env.action_space.n)

    alpha = config['alpha']
    gamma = config['gamma']
    epsilon = 1.0
    epsilon_decay = config['epsilon_decay']
    min_epsilon = config['min_epsilon']

    for episode in track(range(TOTAL_EPISODES), description="Entrenando..."):
        if episode > 0:
            obs, info = env.reset()

        terminated, truncated = False, False
        while not terminated and not truncated:
            if render:
                if hasattr(env.unwrapped, "set_render_data"):
                    env.unwrapped.set_render_data(q_table=agent.q_table)
                env.render()

            state = get_state_from_pos(obs[0], obs[1], GRID_SIZE)
            action = agent.choose_action(state, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = get_state_from_pos(
                next_obs[0], next_obs[1], GRID_SIZE)
            agent.update(state, action, reward, next_state, alpha, gamma)
            obs = next_obs

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

    env.close()

    output_file = run_dir / "q_table.npy"
    np.save(output_file, agent.q_table)

    print(f"✅ Entrenamiento completado. Q-Table guardada en: {output_file}")


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
    q_table_file = run_dir / "q_table.npy"
    video_temp_folder = run_dir / "evaluation_videos_temp"
    final_video_path = run_dir / "evaluation.mp4"

    # Si se solicita grabar vídeo, usamos render_mode="rgb_array" con RecordVideo.
    # En caso contrario, mostramos una ventana interactiva (render_mode="human").
    if video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, str(video_temp_folder),
                          episode_trigger=lambda x: True)
    else:
        env = gym.make(env_id, render_mode="human")

    GRID_SIZE = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(num_states=GRID_SIZE * GRID_SIZE,
                           num_actions=env.action_space.n)

    try:
        agent.q_table = np.load(q_table_file)
        print(f"🧠 Q-Table cargada desde {q_table_file}.")
    except Exception as e:
        print(f"🛑 Error al cargar la Q-Table: {e}")
        env.close()
        return

    # Bucle de evaluación
    for episode in track(range(episodes), description="Evaluando..."):
        # Si no se proporciona semilla para la evaluación, se usará una aleatoria para el mapa.
        # Si se proporciona, se usará solo en el primer episodio para fijar el mapa.
        current_seed = seed if episode == 0 else None
        obs, info = env.reset(seed=current_seed)

        terminated, truncated = False, False
        while not terminated and not truncated:
            state = get_state_from_pos(obs[0], obs[1], GRID_SIZE)
            action = agent.choose_action(state, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            # En modo ventana interactiva, aseguramos el render frame a frame
            if not video:
                env.render()

    env.close()

    # Si no estamos grabando vídeo, terminamos aquí tras la visualización.
    if not video:
        print("✅ Evaluación completada en modo interactivo (sin grabación).")
        return

    # Post-procesado del vídeo cuando se ha solicitado --video
    _merge_videos(str(video_temp_folder), str(
        final_video_path), cleanup=cleanup)

    if cleanup:
        if os.path.exists(video_temp_folder):
            import shutil
            shutil.rmtree(video_temp_folder)
            print("🗑️ Archivos temporales eliminados.")
    else:
        print(
            f"ℹ️ Se conservan los archivos temporales en: {video_temp_folder}")

    print(f"✅ Evaluación completada. Vídeo guardado en: {final_video_path}")

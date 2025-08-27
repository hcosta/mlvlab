# mlvlab/envs/ant_maze_v1/view.py
import gymnasium as gym
import numpy as np

# Asumimos que mlvlab está instalado para la vista interactiva
try:
    from mlvlab.agents.q_learning import QLearningAgent
    from mlvlab.core.logic import InteractiveLogic
    from mlvlab.core.trainer import Trainer
    from mlvlab.ui import AnalyticsView
    from mlvlab import ui
except ImportError:
    print("ADVERTENCIA: mlvlab no detectado. La vista interactiva requiere mlvlab.")
    # Definiciones dummy para evitar errores si solo se importa el archivo
    QLearningAgent = None
    InteractiveLogic = object
    Trainer = None
    AnalyticsView = None
    ui = None


class MazeLogic(InteractiveLogic):
    """
    Lógica interactiva para AntMaze.
    Implementa el manejo del bloqueo de Q-Table para AntShift (Punto 7).
    """

    def _obs_to_state(self, obs):
        grid_size = self.env.unwrapped.GRID_SIZE
        try:
            x = np.clip(obs[0], 0, grid_size - 1)
            y = np.clip(obs[1], 0, grid_size - 1)
            return y * grid_size + x
        except (IndexError, TypeError):
            return 0

    def step(self, state):
        # 1. Verificar si la Q-Table está bloqueada (AntShift - Punto 7)
        is_locked = getattr(self.env.unwrapped, "is_q_table_locked", False)

        # 2. El agente decide la acción
        # Si está bloqueada (AntShift), forzamos Epsilon=0 para explotación pura.
        if is_locked:
            # Guardamos el epsilon actual para poder restaurarlo en las métricas de la UI
            original_epsilon = self.agent.epsilon
            # Forzamos explotación
            self.agent.epsilon = 0.0
            action = self.agent.act(state)
            # Restauramos el valor original para que la UI muestre el valor configurado por el usuario
            self.agent.epsilon = original_epsilon
        else:
            action = self.agent.act(state)

        # 3. El entorno ejecuta la acción
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = self._obs_to_state(obs)
        done = bool(terminated or truncated)

        # 4. El agente aprende (Punto 3 y Punto 7)
        # El aprendizaje ocurre aquí usando los hiperparámetros actuales (Alpha y Gamma).
        # Solo aprendemos si la Q-Table NO está bloqueada.
        if not is_locked:
            self.agent.learn(state, action, reward, next_state, done)

        self.total_reward += reward
        return next_state, reward, done, info


def main(enable_ant_shift=False):
    if AnalyticsView is None:
        return

    # Configuración del entorno
    # Asumiendo registro como "mlv/AntMaze-v1"
    env_name = "mlv/AntMaze-v1"
    grid_size = 11

    try:
        env = gym.make(env_name,
                       render_mode="rgb_array",
                       grid_size=grid_size)
    except gym.error.NameNotFound:
        print(
            f"ERROR: Entorno '{env_name}' no encontrado. Asegúrate de que esté registrado.")
        return
    except Exception as e:
        print(f"Error al crear el entorno: {e}")
        return

    # Configuración inicial del agente (Valores intuitivos)
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.2,    # Alpha (Moderado)
        discount_factor=0.95,  # Gamma (Alto)
        epsilon=1.0,
        epsilon_decay=0.999,  # Decae lentamente
        min_epsilon=0.05
    )
    setattr(agent, "GRID_SIZE", grid_size)

    trainer = Trainer(env, agent, MazeLogic, 1)

    # Configuración de la vista interactiva (Punto 3)
    view = AnalyticsView(
        trainer=trainer,
        title="Dungeons & Pheromones (AntMaze)",
        left_panel=[
            ui.SimulationControls(
                includes=["speed", "turbo"],
                buttons=["play", "reset", "sound", "debug"]
            ),
            # Panel de Hiperparámetros con nombres tematizados
            ui.AgentHyperparameters(
                params={
                    "learning_rate", "discount_factor", "epsilon_decay"
                },
            ),
            ui.ModelPersistence(default_filename="dungeon_pheromones_qt.npz"),
        ],
        right_panel=[
            ui.MetricsDashboard(
                metrics=["epsilon", "current_reward",
                         "episodes_completed", "steps_per_second", "seed"],
            ),
            ui.RewardChart(history_size=500),
        ],
    )

    view.run()


if __name__ == "__main__":
    # Por defecto ejecutamos AntMaze (sin AntShift).
    # Para probar AntShift (Punto 7), ejecutar con main(enable_ant_shift=True)
    main()

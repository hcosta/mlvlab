import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab.core.logic import InteractiveLogic  # <-- Importamos la clase base
from mlvlab import ui
from mlvlab.ui import AnalyticsView

# --- Wrappers (sin cambios) ---


class TimePenaltyWrapper(gym.RewardWrapper):
    """
    Añade una pequeña penalización en cada paso para incentivar al agente
    a encontrar la solución más rápido.
    """

    def __init__(self, env, penalty: float = -0.1):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward: float) -> float:
        """Aplica la penalización a la recompensa original del entorno."""
        return reward + self.penalty


class DirectionToHomeWrapper(gym.ObservationWrapper):
    """Añade a la observación el ángulo en radianes hacia el hormiguero."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # El wrapper asume que el hormiguero está en la posición que devuelve el entorno
        # self.home_pos = np.array([0, 0]) # Esto sería incorrecto si la meta no es (0,0)

        # Definimos el NUEVO espacio de observación (x, y, ángulo)
        low = np.append(self.observation_space.low, -np.pi).astype(np.float32)
        high = np.append(self.observation_space.high, np.pi).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ant_pos = obs
        # Usamos la posición real del hormiguero del entorno base
        goal_pos = self.env.unwrapped.goal_pos
        direction_vector = goal_pos - ant_pos
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        new_obs = np.append(obs, angle).astype(np.float32)
        return new_obs


# --- Lógica de Episodio Adaptada ---

class AntWrapperLogic(InteractiveLogic):
    """
    Implementación de la lógica que funciona con los wrappers.
    """

    def _obs_to_state(self, obs):
        """
        La observación ahora es (x, y, ángulo), pero para nuestro agente Q-Learning
        simple, seguimos usando solo las coordenadas (x, y) para definir el estado.
        Ignoramos el ángulo, que un agente más avanzado podría usar.
        """
        grid_size = self.env.unwrapped.GRID_SIZE
        # obs[0] es X, obs[1] es Y. Ignoramos obs[2] que es el ángulo.
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def step(self, state):
        """
        Lógica de un único paso, adaptada del bucle original.
        """
        action = self.agent.act(state)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        next_state = self._obs_to_state(next_obs)
        self.agent.learn(state, action, reward, next_state, done)
        self.total_reward += reward
        return next_state, reward, done


def main():
    base_env = gym.make("mlv/ant-v1", render_mode="rgb_array")

    # Aplicamos los wrappers al entorno base
    env = DirectionToHomeWrapper(base_env)
    env = TimePenaltyWrapper(env, penalty=-0.1)
    env = TimeLimit(env, max_episode_steps=300)

    grid_size = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_decay=0.999,
    )

    # Instanciamos el Trainer con la nueva clase de lógica y activamos la animación final
    trainer = Trainer(
        env,
        agent,
        logic_class=AntWrapperLogic,
        use_end_scene_animation=True
    )

    # La configuración de la vista es la misma
    view = AnalyticsView(
        trainer=trainer,
        dark=True,
        title="Ant Q-Learning con Wrappers",
        subtitle="Observación aumentada y penalización por tiempo",
        left_panel_components=[
            ui.SimulationControls(),
            ui.AgentHyperparameters(
                trainer.agent, params=[
                    'learning_rate',
                    'discount_factor',
                    'epsilon_decay']
            ),
        ],
        right_panel_components=[
            ui.MetricsDashboard(),
            ui.RewardChart(history_size=100),
        ],
    )
    view.run()


if __name__ == "__main__":
    main()

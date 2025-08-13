
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab import ui

# --- Wrapper para Recompensas ---


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
    """Añade a la observación el ángulo en radianes hacia el hormiguero (0,0)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.home_pos = np.array([0, 0])

        # Definimos el NUEVO espacio de observación (x, y, ángulo)
        low = np.append(self.observation_space.low, -np.pi).astype(np.float32)
        high = np.append(self.observation_space.high, np.pi).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        ant_pos = obs
        direction_vector = self.home_pos - ant_pos
        angle = np.arctan2(direction_vector[1], direction_vector[0])
        new_obs = np.append(obs, angle).astype(np.float32)
        return new_obs


class EpisodeLogicAnt:
    def __init__(self, preserve_seed: bool = False) -> None:
        # Si True, no regeneramos escenario entre episodios (seed estable) salvo reset manual
        self.preserve_seed = bool(preserve_seed)

    def obs_to_state(self, obs, env):
        grid_size = env.unwrapped.GRID_SIZE
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def __call__(self, env, agent):
        # Si preservamos seed entre episodios, no forzamos nueva seed aquí; sólo recolocamos
        if self.preserve_seed:
            # Mantener escenario; respawn controlado por el propio env
            obs, info = env.reset()
        else:
            # Crear nueva seed por episodio para escenarios distintos
            import random
            new_seed = random.randint(0, 1_000_000)
            try:
                obs, info = env.reset(seed=new_seed)
            except TypeError:
                obs, info = env.reset()
        state = self.obs_to_state(obs, env)

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.obs_to_state(next_obs, env)
            agent.learn(state, action, reward, next_state,
                        terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        return total_reward


def main():
    base_env = gym.make("mlv/ant-v1", render_mode="rgb_array")

    # Demostración de extensibilidad con wrappers:
    # 1) Añadimos una característica a la observación (ángulo hacia origen)
    # 2) Penalizamos cada paso para favorecer trayectorias cortas
    env = DirectionToHomeWrapper(base_env)
    env = TimePenaltyWrapper(env, penalty=-0.1)
    if TimeLimit is not None:
        env = TimeLimit(env, max_episode_steps=300)

    grid = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid * grid),
        action_space=env.action_space,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_decay=0.999,
    )

    logic = EpisodeLogicAnt(preserve_seed=True)
    trainer = Trainer(env, agent, episode_logic=logic)

    view = ui.AnalyticsView(
        trainer=trainer,
        subtitle="Q-Learning con Wrappers (Obs + Penalización)",
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
        title="Ant Q-Learning con Wrappers",
    )

    view.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()


class EpisodeLogicAnt:
    def obs_to_state(self, obs, env):
        grid_size = env.unwrapped.GRID_SIZE
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def __call__(self, env, agent):
        obs, info = env.reset()
        state = self.obs_to_state(obs, env)

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.obs_to_state(next_obs, env)
            agent.learn(state, action, reward, next_state,
                        terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        return total_reward


def main():
    base_env = gym.make("mlv/ant-v1", render_mode="rgb_array")
    # Aplicamos wrappers para demostrar extensibilidad de Gymnasium
    env = DirectionToHomeWrapper(base_env)
    env = TimePenaltyWrapper(env, penalty=-0.1)
    env = TimeLimit(env, max_episode_steps=300)
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(
            env.unwrapped.GRID_SIZE * env.unwrapped.GRID_SIZE),
        action_space=env.action_space,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_decay=0.999
    )
    logic = EpisodeLogicAnt()
    trainer = Trainer(env, agent, episode_logic=logic)
    view = ui.AnalyticsView(
        trainer=trainer,
        subtitle="Q-Learning con Lógica de Episodio Personalizada",
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
        title="Ant Q-Learning Custom Logic",
    )

    view.run()


# Para permitir multiprocessing y acelerar las demos
if __name__ in {"__main__", "__mp_main__"}:
    main()

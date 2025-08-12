from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab import ui

# Import robusto de wrappers: soporta ejecución como script o como módulo
try:
    from .wrappers import TimePenaltyWrapper, DirectionToHomeWrapper
except Exception:  # ejecución directa como script
    from wrappers import TimePenaltyWrapper, DirectionToHomeWrapper  # type: ignore


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
    base_env = gym.make("mlvlab/ant-v1", render_mode="rgb_array")

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

    logic = EpisodeLogicAnt()
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
    base_env = gym.make("mlvlab/ant-v1", render_mode="rgb_array")
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

import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab import ui


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
    env = gym.make("mlvlab/ant-v1", render_mode="rgb_array")
    grid = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid * grid),
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

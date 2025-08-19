import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.logic import InteractiveLogic
from mlvlab.core.trainer import Trainer
from mlvlab.ui import AnalyticsView
from mlvlab import ui


class AntLogic(InteractiveLogic):
    """
    Lógica interactiva para el entorno de la hormiga.
    """

    def _obs_to_state(self, obs):
        grid_size = self.env.unwrapped.GRID_SIZE
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def step(self, state):
        action = self.agent.act(state)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        next_state = self._obs_to_state(next_obs)
        self.agent.learn(state, action, float(reward), next_state, done)
        self.total_reward += float(reward)
        return next_state, float(reward), done, info


def main():
    env = gym.make("mlv/AntScout-v1", render_mode="rgb_array")
    grid_size = int(env.unwrapped.GRID_SIZE)
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_decay=0.99,
    )
    setattr(agent, "GRID_SIZE", grid_size)

    trainer = Trainer(env, agent, AntLogic, True)
    view = AnalyticsView(
        dark=True,
        trainer=trainer,
        left_panel=[
            ui.SimulationControls(
                includes=["speed", "turbo"],
                buttons=["play", "reset", "sound", "debug"],
            ),
            ui.AgentHyperparameters(
                params=["learning_rate", "discount_factor", "epsilon_decay"],
            ),
            ui.ModelPersistence(default_filename="ant_q_table.npz"),
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
    main()

import gymnasium as gym
from mlvlab.core.logic import InteractiveLogic
from mlvlab.core.trainer import Trainer
from mlvlab.ui import AnalyticsView
from mlvlab import ui


class AntLogic(InteractiveLogic):
    """
    Lógica interactiva para el entorno de la hormiga.
    """

    def _obs_to_state(self, obs):
        return None

    def step(self, state):
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        # Guardamos los flags para que la clase base los use, esto deberia hacerlo automáticamente InteractiveLogic mediante un wrapper o algo
        self.last_terminated = terminated
        self.last_truncated = truncated
        # Fin del código que debería hacer InteractiveLogic

        return next_obs, reward, done, info


def main():
    # max_episode_steps=20, grid_size=25
    env = gym.make("mlv/AntLost-v1", render_mode="rgb_array")
    trainer = Trainer(env, 0, AntLogic, 1)
    view = AnalyticsView(
        trainer=trainer,
        left_panel=[
            ui.SimulationControls(
                includes=["speed", "turbo"],
                buttons=["play", "reset", "sound"],
            ),
        ],
        right_panel=[
            ui.MetricsDashboard(
                metrics=["current_reward", "episodes_completed",
                         "steps_per_second", "seed"],
            )
        ],
    )
    view.run()


if __name__ == "__main__":
    main()

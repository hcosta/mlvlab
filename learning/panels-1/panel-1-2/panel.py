import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.logic import InteractiveLogic
from mlvlab.core.trainer import Trainer
from mlvlab.ui import AnalyticsView
from mlvlab import ui


class AntLogic(InteractiveLogic):
    """
    Implementación de la lógica para el entorno de la hormiga.
    """

    def _obs_to_state(self, obs):
        """
        Implementación obligatoria de la conversión de observación a estado.
        Para Ant-v1, la observación (x,y) se mapea a un índice único.
        """
        grid_size = self.env.unwrapped.GRID_SIZE
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def step(self, state):
        """
        Implementación obligatoria de la lógica de un paso.
        """
        # 1. El agente elige una acción.
        action = self.agent.act(state)
        # 2. El entorno ejecuta la acción.
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # 3. Convertimos la nueva observación a un estado usando nuestra implementación.
        next_state = self._obs_to_state(next_obs)
        # 4. El agente aprende.
        self.agent.learn(state, action, reward, next_state, done)
        # 5. Acumulamos la recompensa.
        self.total_reward += reward
        # 6. Devolvemos los resultados.
        return next_state, reward, done


def main():
    env = gym.make("mlv/ant-v1", render_mode="rgb_array")
    grid_size = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.99
    )
    trainer = Trainer(env, agent, AntLogic, True)
    view = AnalyticsView(
        dark=True,
        trainer=trainer,
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
            ui.RewardChart(history_size=500),
        ],
    )
    view.run()


if __name__ == "__main__":
    main()

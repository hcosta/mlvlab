import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab.ui import AnalyticsView
from mlvlab import ui
import random


class EpisodeLogicAnt:

    def obs_to_state(self, obs, env):
        # Método imprescindible para el entrenamiento usado por el trainer
        grid_size = env.unwrapped.GRID_SIZE
        return int(obs[1]) * int(grid_size) + int(obs[0])

    def __call__(self, env, agent):
        # Preservamos seed entre episodios, no forzamos nueva seed aquí; sólo recolocamos
        obs, info = env.reset()  # Seed random => env.reset(seed=random.randint(0, 9_999))
        # Traducimos la observación a un estado discreto
        state = self.obs_to_state(obs, env)
        # Lógica de episodio
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            next_state = self.obs_to_state(next_obs, env)
            agent.learn(state, action, reward, next_state,
                        terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
        return total_reward


def main():
    env = gym.make("mlv/ant-v1", render_mode="rgb_array")
    # Respawn independiente de seed de escenario: episodios no idénticos al iniciar
    # env.unwrapped.set_respawn_unseeded(True) # Ya es así por defecto en esta simulación
    grid_size = env.unwrapped.GRID_SIZE
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.995
    )
    # Por defecto, preservamos la seed entre episodios (mismo laberinto) salvo reset manual
    trainer = Trainer(env, agent, EpisodeLogicAnt())
    view = AnalyticsView(
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

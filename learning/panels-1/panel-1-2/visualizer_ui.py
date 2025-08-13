import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab import ui


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
    env = gym.make("mlvlab/ant-v1", render_mode="rgb_array")
    # Respawn independiente de seed de escenario: episodios no idénticos al iniciar
    try:
        env.unwrapped.set_respawn_unseeded(True)
    except Exception:
        pass
    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(
            env.unwrapped.GRID_SIZE * env.unwrapped.GRID_SIZE),
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_decay=0.995
    )
    # Por defecto, preservamos la seed entre episodios (mismo laberinto) salvo reset manual
    logic = EpisodeLogicAnt(preserve_seed=True)
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

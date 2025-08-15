import gymnasium as gym
from mlvlab.agents.q_learning import QLearningAgent
from mlvlab.core.trainer import Trainer
from mlvlab.ui import AnalyticsView
from mlvlab import ui

# 1. FUNCIÓN PARA CONVERTIR OBSERVACIÓN EN ESTADO
# Esta función es específica del entorno y el agente.


def obs_to_state_ant(obs, env):
    """Convierte la observación (coordenadas) de Ant-v1 a un estado discreto."""
    grid_size = env.unwrapped.GRID_SIZE
    # La observación es una tupla (x, y), la convertimos en un índice único.
    return int(obs[1]) * int(grid_size) + int(obs[0])


# 2. FUNCIÓN PARA GOBERNAR LA LÓGICA DE UN EPISODIO
# Esta función define el bucle principal de entrenamiento para un episodio.
def episode_logic_ant(env, agent):
    """
    Ejecuta un episodio completo de entrenamiento para el agente en el entorno.
    """
    # Se podría obtener la función de conversión del propio trainer si se pasara,
    # pero es más limpio usar la función directamente.
    obs, info = env.reset()
    state = obs_to_state_ant(obs, env)

    done = False
    total_reward = 0.0
    while not done:
        # El agente elige una acción basada en el estado actual.
        action = agent.act(state)
        # El entorno ejecuta la acción.
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Convertimos la nueva observación a un estado.
        next_state = obs_to_state_ant(next_obs, env)
        # El agente aprende de la transición.
        agent.learn(state, action, reward, next_state, terminated or truncated)

        # Actualizamos el estado y la recompensa total.
        state = next_state
        total_reward += reward
        done = terminated or truncated

    return total_reward


def main():
    env = gym.make("mlv/ant-v1", render_mode="rgb_array")
    grid_size = env.unwrapped.GRID_SIZE

    agent = QLearningAgent(
        observation_space=gym.spaces.Discrete(grid_size * grid_size),
        action_space=env.action_space,
        learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.9925
    )

    # ---------------------------------------------------------------------
    # USO DE LA NUEVA API DE TRAINER
    # Pasamos el entorno, el agente y las dos funciones de lógica.
    # ---------------------------------------------------------------------
    trainer = Trainer(
        env=env,
        agent=agent,
        episode_logic=episode_logic_ant,
        obs_to_state=obs_to_state_ant
    )
    # ---------------------------------------------------------------------

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

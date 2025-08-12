# visualizer_ui.py: Demo declarativa usando mlvlab.ui

import gymnasium as gym
import mlvlab as m

from q_learning import QLearningAgent, get_state_from_pos


def main():
    grid_size = 15
    env = gym.make(
        "mlvlab/ant-v1",
        render_mode="rgb_array",
        grid_size=grid_size,
        reward_food=500,
        reward_obstacle=-50,
        reward_move=0,
    )

    agent = QLearningAgent(
        num_states=grid_size * grid_size,
        num_actions=env.action_space.n,
    )

    # Asignamos el estado de la observación del entorno al agente
    # Patrón recomendado para alumnos: asignar la función directamente
    agent.extract_state_from_obs = get_state_from_pos

    view = m.AnalyticsView(
        env=env,
        agent=agent,
        subtitle="Q-Learning con entorno Ant usando mlvlab.ui",
        left_panel_components=[
            m.ui.SimulationControls(),
            m.ui.AgentHyperparameters(
                agent, params=['learning_rate', 'discount_factor', 'epsilon_decay']),
        ],
        right_panel_components=[
            m.ui.MetricsDashboard(),
            m.ui.RewardChart(history_size=100),
        ],
        title="Ant Q-Learning",
        history_size=100,
        dark=False,
    )

    view.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()

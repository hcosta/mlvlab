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

    agent = QLearningAgent(num_states=grid_size * grid_size,
                           num_actions=env.action_space.n)

    # Proveer conversión de observación->estado al runner de forma opcional
    agent.extract_state_from_obs = lambda obs: get_state_from_pos(
        obs[0], obs[1], grid_size)  # type: ignore[attr-defined]

    view = m.AnalyticsView(
        env=env,
        agent=agent,
        subtitle="Q-Learning con entorno Ant (mlvlab.ui)",
        agent_hparams_defaults={
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon_decay': 0.99,
            'epsilon': 1.0,
            'min_epsilon': 0.1,
        },
        left_panel_components=[
            m.ui.SimulationControls(),
            m.ui.AgentHyperparameters(
                agent, params=['learning_rate', 'discount_factor', 'epsilon_decay']),
        ],
        right_panel_components=[
            m.ui.MetricsDashboard(),
            m.ui.RewardChart(history_size=100),
        ],
        title="mlvlab.ui - Ant Q-Learning",
        history_size=100,
        dark=False,
    )

    view.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()

# mlvlab/__init__.py
from gymnasium.envs.registration import register

register(
    id="mlv/AntLost-v1",
    entry_point="mlvlab.envs.ant_lost_v1.env:LostAntEnv",
    max_episode_steps=5,
    kwargs={'grid_size': 10}
)

# Registramos con namespace "mlv" y mantenemos compatibilidad con el ID antiguo
register(
    id="mlv/AntScout-v1",
    entry_point="mlvlab.envs.ant_scout_v1.env:ScoutAntEnv",
    max_episode_steps=500,
    kwargs={'grid_size': 10}  # Argumentos por defecto
)

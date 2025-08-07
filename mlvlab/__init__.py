# mlvlab/__init__.py
from gymnasium.envs.registration import register

# Registramos la versión base del entorno de la Hormiga.
# Usamos el namespace "MLVLab" como se especifica en el documento.
register(
    id="mlvlab/ant-v1",
    entry_point="mlvlab.envs.ant.ant_env:LostAntEnv",
    max_episode_steps=500,
    kwargs={'grid_size': 15}  # Argumentos por defecto
)

# mlvlab/__init__.py
from gymnasium.envs.registration import register

# Registro del entorno de ejemplo (Ant)
try:
    register(
        id="mlvlab/ant-v1",
        entry_point="mlvlab.envs.ant.ant_env:LostAntEnv",
    )
except Exception:
    # Evitar errores si se registra dos veces en entornos interactivos
    pass

# mlvlab/__init__.py
from gymnasium.envs.registration import register

# Registramos con namespace "mlv" y mantenemos compatibilidad con el ID antiguo
register(
    id="mlv/AntScout-v1",
    entry_point="mlvlab.envs.ant_scout_v1.env:LostAntEnv",
    max_episode_steps=500,
    kwargs={'grid_size': 10}  # Argumentos por defecto
)

# # Exponer AnalyticsView en el nivel superior para la API propuesta
# try:
#     from .ui import AnalyticsView  # noqa: F401
#     # Exponer submódulo ui como atributo (permite usar mlvlab.ui.*)
#     from . import ui as ui  # type: ignore # noqa: F401
# except Exception:
#     # Permite importar el paquete aunque nicegui no esté instalado todavía
#     pass

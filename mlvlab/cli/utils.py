# mlvlab/cli/utils.py
from rich.console import Console
import importlib
import gymnasium as gym

console = Console()


def get_env_config(env_id: str) -> dict:
    """
    Intenta cargar el módulo de configuración (config.py) para un entorno específico.
    """
    try:
        # Obtenemos la especificación del entorno registrado
        spec = gym.spec(env_id)
        entry_point = spec.entry_point  # e.g., "mlvlab.envs.ant.ant_env:LostAntEnv"

        # Extraemos el path del módulo (e.g., "mlvlab.envs.ant.ant_env")
        module_path = entry_point.split(':')[0]

        # Derivamos el path de configuración (e.g. "mlvlab.envs.ant.config")
        path_parts = module_path.split('.')
        if len(path_parts) > 1:
            # Reemplazamos el último segmento (e.g., 'ant_env') por 'config'
            config_module_path = ".".join(path_parts[:-1] + ["config"])
        else:
            return {}

        # Importamos el módulo de configuración dinámicamente
        config_module = importlib.import_module(config_module_path)

        # Devolvemos la configuración si existe
        return {
            "KEY_MAP": getattr(config_module, 'KEY_MAP', None),
            "DESCRIPTION": getattr(config_module, 'DESCRIPTION', None),
            "BASELINE": getattr(config_module, 'BASELINE', None),
        }

    except (ImportError, AttributeError, gym.error.NameNotFound):
        # Si falla algo (el entorno no existe, el config no existe), devolvemos un diccionario vacío.
        return {}

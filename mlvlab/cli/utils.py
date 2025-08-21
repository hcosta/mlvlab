# mlvlab/cli/utils.py
from rich.console import Console
import importlib
import gymnasium as gym
from mlvlab.i18n import i18n

console = Console()


def get_env_config(env_id: str) -> dict:
    """
    Intenta cargar el módulo de configuración (config.py) para un entorno específico.
    """
    try:
        # Obtenemos la especificación del entorno registrado
        spec = gym.spec(env_id)
        entry_point = spec.entry_point  # e.g., "mlvlab.envs.ant.env:ScoutAntEnv"

        # Extraemos el path del módulo (e.g., "mlvlab.envs.ant.env")
        module_path = entry_point.split(':')[0]

        # Derivamos el path de configuración (e.g. "mlvlab.envs.ant.config")
        path_parts = module_path.split('.')
        if len(path_parts) > 1:
            # Reemplazamos el último segmento (e.g., 'env') por 'config'
            config_module_path = ".".join(path_parts[:-1] + ["config"])
        else:
            return {}

        # Importamos el módulo de configuración dinámicamente
        config_module = importlib.import_module(config_module_path)

        # Obtenemos la configuración base
        config = {
            "KEY_MAP": getattr(config_module, 'KEY_MAP', None),
            "DESCRIPTION": getattr(config_module, 'DESCRIPTION', None),
            "BASELINE": getattr(config_module, 'BASELINE', None),
            "UNIT": getattr(config_module, 'UNIT', None),
            "ALGORITHM": getattr(config_module, 'ALGORITHM', None),
        }

        # Intentamos obtener la descripción traducida
        env_key = env_id.split('/')[-1].replace('-', '_').lower()
        # Removemos la versión (_v1) para obtener la clave base
        env_key = env_key.replace('_v1', '').replace('_v0', '')
        # Convertimos camelCase a snake_case si es necesario
        if 'scout' in env_key:
            env_key = 'ant_scout'
        try:
            translated_desc = i18n.t(f"environments.descriptions.{env_key}")
            config["DESCRIPTION"] = translated_desc
        except:
            # Si no hay traducción, mantenemos la descripción original
            pass

        return config

    except (ImportError, AttributeError, gym.error.NameNotFound):
        # Fallbacks: derivar por ID del entorno con -/_
        try:
            pkg = env_id.split('/')[-1]
            pkg_us = pkg.replace('-', '_')
            # Intentar envs.<pkg_us>.config
            config_module = importlib.import_module(
                f"mlvlab.envs.{pkg_us}.config")

            config = {
                "KEY_MAP": getattr(config_module, 'KEY_MAP', None),
                "DESCRIPTION": getattr(config_module, 'DESCRIPTION', None),
                "BASELINE": getattr(config_module, 'BASELINE', None),
                "UNIT": getattr(config_module, 'UNIT', None),
                "ALGORITHM": getattr(config_module, 'ALGORITHM', None),
            }

            # Intentamos obtener la descripción traducida
            env_key = pkg_us.lower()
            # Removemos la versión (_v1) para obtener la clave base
            env_key = env_key.replace('_v1', '').replace('_v0', '')
            # Convertimos camelCase a snake_case si es necesario
            if 'scout' in env_key:
                env_key = 'ant_scout'
            try:
                translated_desc = i18n.t(
                    f"environments.descriptions.{env_key}")
                config["DESCRIPTION"] = translated_desc
            except:
                # Si no hay traducción, mantenemos la descripción original
                pass

            return config
        except Exception:
            return {}

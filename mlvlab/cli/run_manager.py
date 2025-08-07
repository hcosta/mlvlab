# mlvlab/cli/run_manager.py
import os
from pathlib import Path

DATA_DIR = Path("data")


def get_env_data_dir(env_id: str) -> Path:
    """Obtiene la carpeta de datos para un entorno, creándola si no existe."""
    env_name = env_id.replace("/", "_")
    env_dir = DATA_DIR / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    return env_dir


def get_run_dir(env_id: str, seed: int) -> Path:
    """Obtiene/Crea el directorio para un 'run' basado en una semilla específica."""
    env_dir = get_env_data_dir(env_id)
    run_dir = env_dir / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run_dir(env_id: str) -> Path | None:
    """Encuentra el directorio de 'run' (seed-*) modificado más recientemente."""
    env_dir = get_env_data_dir(env_id)

    # Filtra solo directorios que sigan el patrón "seed-*"
    seed_dirs = [d for d in env_dir.iterdir() if d.is_dir()
                 and d.name.startswith("seed-")]

    if not seed_dirs:
        return None

    # Devuelve el directorio modificado más recientemente
    latest_dir = max(seed_dirs, key=os.path.getmtime)
    return latest_dir

# mlvlab/cli/player_entry.py
from mlvlab.i18n.core import i18n
from mlvlab.core.player import play_interactive
from mlvlab.cli.utils import get_env_config
import argparse
import sys
from pathlib import Path
from typing import Optional

# Añadimos la raíz del proyecto al path para que los imports funcionen
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def normalize_env_id(env_id: str) -> str:
    """
    Helper to normalize env_id, ensuring it has the 'mlv/' prefix.
    """
    if not env_id.startswith("mlv/"):
        return f"mlv/{env_id}"
    return env_id


def main():
    """
    Entry point for running an interactive play session in a separate process.
    This script is called by the 'play' command from the main CLI.
    """
    # La inicialización de i18n es automática al importar.

    parser = argparse.ArgumentParser(description="MLV-Lab Interactive Player")
    parser.add_argument("env_id", type=str, help="The environment ID to play.")
    parser.add_argument("--seed", type=int, default=None,
                        help="The random seed for the environment.")

    args = parser.parse_args()

    normalized_env_id = normalize_env_id(args.env_id)

    config = get_env_config(normalized_env_id)
    key_map = config.get("KEY_MAP")

    if key_map is None:
        print(i18n.t('cli.entry.no_keymap',
                     env_id=normalized_env_id), file=sys.stderr)
        sys.exit(1)

    play_interactive(normalized_env_id, key_map=key_map, seed=args.seed)


if __name__ == "__main__":
    main()

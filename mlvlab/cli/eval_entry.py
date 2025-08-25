# mlvlab/cli/eval_entry.py
from mlvlab.i18n.core import i18n
from mlvlab.algorithms.registry import get_algorithm
from mlvlab.cli.utils import get_env_config
from mlvlab.cli.run_manager import get_run_dir, find_latest_run_dir
import argparse
import sys
from pathlib import Path

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
    Entry point for running an evaluation session in a separate process.
    """
    # La inicialización de i18n es automática al importar.

    parser = argparse.ArgumentParser(description="MLV-Lab Evaluator")
    parser.add_argument("env_id", type=str,
                        help="The environment ID to evaluate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="The specific run seed to evaluate.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier.")
    parser.add_argument("--record", action='store_true',
                        help="Record a video of the evaluation.")

    args = parser.parse_args()

    normalized_env_id = normalize_env_id(args.env_id)
    config = get_env_config(normalized_env_id)
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT")

    if not algorithm_key:
        print(i18n.t('cli.entry.no_algorithm',
              env_id=normalized_env_id), file=sys.stderr)
        sys.exit(1)

    # --- LÓGICA MODIFICADA ---
    run_dir = None
    eval_seed = args.seed

    # Si el algoritmo NO es 'random', necesitamos buscar y cargar un entrenamiento.
    if algorithm_key != 'random':
        if args.seed is not None:
            run_dir = get_run_dir(normalized_env_id, args.seed)
        else:
            print(i18n.t("cli.messages.searching_last_training"))
            run_dir = find_latest_run_dir(normalized_env_id)

        if not run_dir or not (run_dir / "q_table.npy").exists():
            print(i18n.t('cli.entry.no_valid_training',
                  env_id=normalized_env_id), file=sys.stderr)
            sys.exit(1)

        print(i18n.t("cli.messages.evaluating_from", run_dir=str(run_dir)))
        try:
            # Si no se especifica una semilla para eval, la tomamos del entrenamiento
            if eval_seed is None:
                eval_seed = int(run_dir.name.split('-')[1])
        except (IndexError, ValueError):
            eval_seed = None  # Fallback
    else:
        # Si es random, no buscamos directorio, solo usamos la semilla si se proveyó
        print(i18n.t("cli.messages.evaluating_random"))
    # --- FIN DE LA LÓGICA MODIFICADA ---

    try:
        algo = get_algorithm(algorithm_key)
        algo.eval(
            normalized_env_id,
            run_dir=run_dir,  # Será None para el agente random
            episodes=args.episodes,
            seed=eval_seed,
            video=args.record,
            speed=args.speed
        )
    except Exception as e:
        print(i18n.t('cli.entry.eval_error', error=e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

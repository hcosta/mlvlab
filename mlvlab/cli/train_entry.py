# mlvlab/cli/train_entry.py
from mlvlab.i18n.core import i18n
from mlvlab.algorithms.registry import get_algorithm
from mlvlab.cli.utils import get_env_config
from mlvlab.cli.run_manager import get_run_dir
import argparse
import sys
import random
from pathlib import Path
import traceback

# Añadimos la raíz del proyecto al path para que los imports funcionen
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def normalize_env_id(env_id: str) -> str:
    if not env_id.startswith("mlv/"):
        return f"mlv/{env_id}"
    return env_id


def main():
    """
    Punto de entrada para ejecutar una sesión de entrenamiento en un proceso separado.
    """
    parser = argparse.ArgumentParser(description="MLV-Lab Trainer")
    parser.add_argument("env_id", type=str,
                        help="El ID del entorno a entrenar.")
    parser.add_argument("--seed", type=int, default=None,
                        help="La semilla aleatoria para el entrenamiento.")
    parser.add_argument("--eps", type=int, default=None,
                        help="Número de episodios para entrenar.")
    parser.add_argument("--render", action='store_true',
                        help="Renderizar el entorno durante el entrenamiento.")

    args = parser.parse_args()

    normalized_env_id = normalize_env_id(args.env_id)

    config = get_env_config(normalized_env_id)
    baseline = config.get("BASELINE", {})
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT")
    train_config = baseline.get("config", {}).copy()

    # Sobrescribir el número de episodios si se pasa como argumento
    if args.eps is not None:
        train_config['episodes'] = args.eps

    if not algorithm_key:
        print(i18n.t('cli.entry.no_algorithm',
              env_id=normalized_env_id), file=sys.stderr)
        sys.exit(1)

    run_seed = args.seed
    if run_seed is None:
        run_seed = random.randint(0, 10000)
        print(i18n.t("cli.messages.no_seed_random", seed=run_seed))

    run_dir = None
    # Solo creamos un directorio si el algoritmo no es 'random'
    if algorithm_key != 'random':
        run_dir = get_run_dir(normalized_env_id, run_seed)
        print(i18n.t("cli.messages.working_dir", run_dir=str(run_dir)))

    try:
        algo = get_algorithm(algorithm_key)
        # La llamada a train es ahora simple y universal para todos los algoritmos
        algo.train(
            normalized_env_id,
            train_config,
            run_dir=run_dir,  # Será None para 'random', lo cual es correcto
            seed=run_seed,
            render=args.render
        )
    except Exception as e:
        print(i18n.t('cli.entry.train_error', error=e), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

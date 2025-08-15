# mlvlab/cli/main.py
import random
import typer
import sys
import gymnasium as gym
import importlib.util
import importlib
from pathlib import Path
from gymnasium.error import NameNotFound

# Asegurar registro de entornos al cargar la CLI
try:
    import mlvlab as _mlvlab  # noqa: F401
except Exception:
    pass
from typing import Optional
# Importamos la librería correcta para descubrir entry points
if sys.version_info < (3, 10):
    # Para Python < 3.10
    from importlib_metadata import entry_points
else:
    # Para Python >= 3.10
    from importlib.metadata import entry_points

# Importamos las nuevas utilidades de gestión de 'runs'
from mlvlab.cli.run_manager import get_run_dir, find_latest_run_dir
from mlvlab.cli.utils import console, get_env_config
from mlvlab.core.player import play_interactive

app = typer.Typer(
    rich_markup_mode="rich",
    help="[bold green]MLV-Lab CLI[/bold green]: Ecosistema de aprendizaje de IA Visual.",
    no_args_is_help=True
)

# Comandos Principales ---


def complete_env_id(incomplete: str):
    """
    Función de autocompletado que devuelve los IDs de entorno que coinciden.
    """
    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    for env_id in all_env_ids:
        if env_id.startswith(incomplete):
            yield env_id


def complete_unit_id(incomplete: str):
    """
    Función de autocompletado que devuelve las unidades (colecciones) disponibles.
    """
    # Reutilizamos la misma lógica que ya tienes en el comando 'list'
    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    env_id_to_unit: dict[str, str] = {}
    for env_id in all_env_ids:
        cfg = get_env_config(env_id)
        unit = cfg.get("UNIT") or cfg.get("unit") or "otros"
        env_id_to_unit[env_id] = str(unit)

    unidades = sorted(set(env_id_to_unit.values()))

    for unidad in unidades:
        if unidad.startswith(incomplete):
            yield unidad


@app.command(name="list")
def list_environments(
    unidad: Optional[str] = typer.Argument(
        None,
        help="Filtra por unidad (ej.: ql, sarsa, dqn, dqnp)",
        autocompletion=complete_unit_id
    )
):
    """Lista unidades disponibles o los entornos de una unidad concreta."""
    from rich.table import Table

    # Detectar entornos por el namespace "mlv" (IDs sin unidad)
    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    # Mapa env_id -> unidad desde su config
    env_id_to_unit: dict[str, str] = {}
    for env_id in all_env_ids:
        cfg = get_env_config(env_id)
        unit = cfg.get("UNIT") or cfg.get("unit") or "otros"
        env_id_to_unit[env_id] = str(unit)

    # Si no pasan unidad, listamos las unidades disponibles
    if unidad is None:
        unidades = sorted(set(env_id_to_unit.values()))
        table = Table("ID de la Unidad", "Descripción")
        for u in unidades:
            desc = {
                'ql': 'Simulaciones de Q-Learning',
                'sarsa': 'Simulaciones de SARSA',
                'dqn': 'Simulaciones de DQN',
                'dqnp': 'Simulaciones DQN Plus (avanzado)'
            }.get(u, 'Unidad personalizada')
            table.add_row(f"[cyan]{u}[/cyan]", desc)
        console.print(table)
        console.print(
            "Usa: [bold]mlv list <unidad>[/bold] para ver entornos de una unidad (ej. mlv list ql)")
        return

    # Con unidad proporcionada: filtrar por config
    unit_envs = [env_id for env_id, u in env_id_to_unit.items() if u == unidad]

    table = Table("ID del Entorno", "Descripción", "Baseline Agent")
    for env_id in sorted(unit_envs):
        config = get_env_config(env_id)
        desc = config.get("DESCRIPTION", "N/D")
        baseline = config.get("BASELINE", {}).get("agent", "[red]N/A[/red]")
        table.add_row(f"[cyan]{env_id}[/cyan]", desc, baseline)
    console.print(table)


@app.command(name="play")
def play(
    env_id: str = typer.Argument(...,
                                 help="ID del entorno a jugar (e.g., mlv/ant-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Semilla para reproducibilidad del mapa.")
):
    """(Paso 1) Juega interactivamente a un entorno (render_mode=human)."""
    console.print(f"🚀 Lanzando [bold cyan]{env_id}[/bold cyan]...")

    try:
        # Obtenemos la configuración específica (mapeo de teclas)
        config = get_env_config(env_id)
        key_map = config.get("KEY_MAP", None)

        if key_map is None:
            console.print(
                f"❌ [bold red]Error:[/bold red] No se encontró un mapeo de teclas para {env_id}.")
            raise typer.Exit(code=1)

        # Usamos el player genérico
        play_interactive(env_id, key_map=key_map, seed=seed)

    except NameNotFound:
        console.print(
            f"❌ [bold red]Error:[/bold red] Entorno '{env_id}' no encontrado.")
        raise typer.Exit(code=1)


@app.command(name="train")
def train(
    env_id: str = typer.Argument(...,
                                 help="ID del entorno a entrenar(e.g., mlv/ant-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Semilla para el entrenamiento (si no se da, se genera una)."),
    eps: Optional[int] = typer.Option(
        None, "--eps", "-e", help="Sobrescribir el número de episodios."),
    render: bool = typer.Option(
        False, "--render", "-r", help="Renderizar el entrenamiento en tiempo real (lento).")
):
    """Entrena un agente. Si no se especifica una semilla, se genera una aleatoria."""
    config = get_env_config(env_id)
    baseline = config.get("BASELINE", {})
    # Nuevo flujo: seleccionar algoritmo desde ALGORITHM/UNIT
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT", None)
    train_config = baseline.get("config", {}).copy()

    if not algorithm_key:
        console.print(
            f"❌ Error: No hay algoritmo de referencia definido para {env_id}.")
        raise typer.Exit(code=1)

    if eps:
        train_config['episodes'] = eps

    # LÓGICA DE SEMILLA ALEATORIA ---
    run_seed = seed
    if run_seed is None:
        run_seed = random.randint(0, 10000)
        console.print(
            f"🌱 No se especificó semilla. Usando una aleatoria: [bold yellow]{run_seed}[/bold yellow]")

    # Obtener/crear el directorio para este 'run'
    run_dir = get_run_dir(env_id, run_seed)
    console.print(
        f"📂 Trabajando en el directorio: [bold yellow]{run_dir}[/bold yellow]")

    # Nuevo: usar registro de algoritmos (asegurar carga de plugins integrados)
    try:
        import mlvlab.algorithms as _mlv_algos  # activa auto-registro de plugins
        from mlvlab.algorithms.registry import get_algorithm
        algo = get_algorithm(str(algorithm_key))
        algo.train(env_id, train_config, run_dir=run_dir,
                   seed=run_seed, render=render)
    except Exception as e:
        console.print(
            f"❌ Error al entrenar con algoritmo '{algorithm_key}': {e}")
        raise typer.Exit(code=1)


@app.command(name="eval")
def evaluate(
    env_id: str = typer.Argument(...,
                                 help="ID del entorno a evaluar(e.g., mlv/ant-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Semilla del 'run' a evaluar (por defecto, la última)."),
    episodes: int = typer.Option(
        5, "--eps", "-e", help="Número de episodios a ejecutar."),
    # <--- 1. AÑADE LA NUEVA OPCIÓN DE VELOCIDAD AQUÍ ---
    speed: float = typer.Option(
        1.0, "--speed", "-sp", help="Multiplicador de velocidad (e.g., 0.5 para mitad de velocidad)."),
    # ---------------------------------------------------
    no_cleanup: bool = typer.Option(
        False, "--no-cleanup", "-nc", help="Conserva los vídeos temporales de cada episodio (solo si --record)."),
    record: bool = typer.Option(
        False, "--rec", "-r", help="Graba y genera un vídeo de la evaluación en lugar de solo visualizar.")
):
    """Evalúa un 'run'. Por defecto se visualiza en modo interactivo; usa --record para grabar."""
    run_dir = None
    if seed is not None:
        run_dir = get_run_dir(env_id, seed)
    else:
        console.print(
            "ℹ️ No se especificó semilla. Buscando el último entrenamiento...")
        run_dir = find_latest_run_dir(env_id)

    if not run_dir or not (run_dir / "q_table.npy").exists():
        console.print(
            f"❌ Error: No se encontró un entrenamiento válido. Ejecuta 'mlv train' primero.")
        raise typer.Exit(code=1)

    console.print(f"🎬 Evaluando desde [bold yellow]{run_dir}[/bold yellow]...")

    # Extraer la semilla del nombre de la carpeta para la reproducibilidad del mapa
    try:
        eval_seed = int(run_dir.name.split('-')[1])
    except (IndexError, ValueError):
        console.print(
            f"⚠️ No se pudo extraer la semilla del nombre de la carpeta '{run_dir.name}'. La evaluación usará un mapa aleatorio.")
        eval_seed = None

    config = get_env_config(env_id)
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT", None)

    # Nuevo: usar registro de algoritmos (asegurar carga de plugins)
    try:
        import mlvlab.algorithms as _mlv_algos
        from mlvlab.algorithms.registry import get_algorithm
        algo = get_algorithm(str(algorithm_key))
        algo.eval(
            env_id,
            run_dir=run_dir,
            episodes=episodes,
            seed=eval_seed,
            cleanup=not no_cleanup,
            video=record,
            speed=speed
        )
    except Exception as e:
        console.print(
            f"❌ Error al evaluar con algoritmo '{algorithm_key}': {e}")
        raise typer.Exit(code=1)


@app.command(name="help")
def help_env(
    env_id: str = typer.Argument(...,
                                 help="ID del entorno a inspeccionar (e.g., mlv/ant-v1).",
                                 autocompletion=complete_env_id),
):
    """Muestra la ficha técnica y un enlace a la documentación del entorno."""
    try:
        env = gym.make(env_id)
        spec = gym.spec(env_id)

        console.print(
            f"\n[bold underline]Ficha Técnica de {env_id}[/bold underline]\n")
        console.print(
            f"[bold cyan]Observation Space:[/bold cyan]\n{env.observation_space}\n")
        console.print(
            f"[bold cyan]Action Space:[/bold cyan]\n{env.action_space}\n")

        # LÓGICA PARA CONSTRUIR LA URL DINÁMICA AL README ---
        try:
            # 1. Define la URL base de tu repositorio
            base_repo_url = "https://github.com/hcosta/mlvlab/tree/master"

            # 2. Extrae la ruta del módulo del entry point
            #    Ej: "mlvlab.envs.ant.env:LostAntEnv" -> "mlvlab.envs.ant.env"
            entry_point = spec.entry_point
            module_path_str = entry_point.split(':')[0]

            # 3. Convierte la ruta del módulo a una ruta de directorio
            #    Ej: "mlvlab.envs.ant.env" -> "mlvlab/envs/ant"
            path_parts = module_path_str.split('.')
            # Nos quedamos con todo menos el último elemento (el nombre del fichero _env.py)
            relative_path = "/".join(path_parts[:-1])

            # 4. Construye la URL final
            readme_url = f"{base_repo_url}/{relative_path}/README.md"

            console.print(
                f"[bold cyan]For more details, check the README:[/bold cyan]\n[green]{readme_url}[/green]\n")

        except Exception:
            # Si algo falla, simplemente no mostramos el enlace
            pass

        env.close()

    except NameNotFound:
        console.print(
            f"❌ [bold red]Error:[/bold red] Entorno '{env_id}' no encontrado.")
        raise typer.Exit(code=1)


# Cargador de Plugins (Nivel 3: Arquitecto) ---

def load_plugins(application: typer.Typer):
    """Descubre y carga plugins registrados mediante 'mlvlab.plugins' entry points."""
    try:
        discovered_plugins = entry_points(group='mlvlab.plugins')
    except Exception:
        return

    for plugin in discovered_plugins:
        try:
            # Carga el objeto (generalmente otra aplicación Typer)
            plugin_app = plugin.load()
            # Añade los comandos del plugin a la aplicación principal
            if isinstance(plugin_app, typer.Typer):
                application.add_typer(plugin_app, name=plugin.name)
            else:
                application.command(name=plugin.name)(plugin_app)
            # console.print(f"🔌 Plugin cargado: [green]{plugin.name}[/green]")
        except Exception as e:
            console.print(
                f"❌ [red]Error al cargar plugin '{plugin.name}':[/red] {e}")

# Función principal que se ejecuta cuando se llama a 'mlv'


def run_app():
    load_plugins(app)
    app()


if __name__ == "__main__":
    run_app()

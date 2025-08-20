# mlvlab/cli/main.py

from __future__ import annotations

import importlib
import random
from typing import Optional

import gymnasium as gym
import typer
from gymnasium.error import NameNotFound
from rich.console import Console

# Importamos las nuevas utilidades de gestión de 'runs'
from mlvlab.cli.run_manager import get_run_dir, find_latest_run_dir
from mlvlab.cli.utils import console, get_env_config
from mlvlab.core.player import play_interactive

# Importamos el sistema de internacionalización
from mlvlab.i18n.core import i18n

app = typer.Typer(
    rich_markup_mode="rich",
    help=i18n.t("cli.help.main"),
    no_args_is_help=True
)

# =============================================================================
# FUNCIONES DE AUTOCOMPLETADO
# =============================================================================


def normalize_env_id(env_id: str) -> str:
    """
    Normaliza un ID de entorno para que siempre tenga el prefijo 'mlv/'.
    Si el ID no tiene el prefijo, lo añade.
    """
    if not env_id.startswith("mlv/"):
        return f"mlv/{env_id}"
    return env_id


def complete_env_id(incomplete: str):
    """
    Función de autocompletado que devuelve los IDs de entorno que coinciden.
    """
    # Importar automáticamente todos los entornos para asegurar que estén registrados
    try:
        import mlvlab.envs
    except ImportError:
        pass

    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    # Si el usuario está escribiendo sin el prefijo mlv/, mostrar opciones sin prefijo
    if not incomplete.startswith("mlv/"):
        for env_id in all_env_ids:
            short_name = env_id.replace("mlv/", "")
            if short_name.startswith(incomplete):
                yield short_name
    else:
        # Si el usuario está escribiendo con el prefijo, mostrar opciones completas
        for env_id in all_env_ids:
            if env_id.startswith(incomplete):
                yield env_id


def complete_unit_id(incomplete: str):
    """
    Función de autocompletado que devuelve las unidades (colecciones) disponibles.
    """
    # Importar automáticamente todos los entornos para asegurar que estén registrados
    try:
        import mlvlab.envs
    except ImportError:
        pass

    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]
    env_id_to_unit: dict[str, str] = {}
    for env_id in all_env_ids:
        cfg = get_env_config(env_id)
        unit = cfg.get("UNIT") or cfg.get("unit") or i18n.t("common.other")
        env_id_to_unit[env_id] = str(unit)

    unidades = sorted(set(env_id_to_unit.values()))
    for unidad in unidades:
        if unidad.startswith(incomplete):
            yield unidad


# =============================================================================
# COMANDOS PRINCIPALES
# =============================================================================


@app.command(name="config")
def config_command(
    action: str = typer.Argument(...,
                                 help="Action to perform: get, set, reset"),
    key: Optional[str] = typer.Argument(
        None, help="Configuration key (e.g., locale)"),
    value: Optional[str] = typer.Argument(None, help="Value to set"),
):
    """Manage MLV-Lab configuration."""
    from pathlib import Path
    import json

    config_dir = Path.home() / '.mlvlab'
    config_file = config_dir / 'config.json'

    if action == "get":
        if not config_file.exists():
            console.print("No configuration file found. Using defaults.")
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            if key:
                if key in config:
                    console.print(f"{key}: {config[key]}")
                else:
                    console.print(f"Key '{key}' not found in configuration.")
            else:
                console.print("Current configuration:")
                for k, v in config.items():
                    console.print(f"  {k}: {v}")
        except Exception as e:
            console.print(f"Error reading configuration: {e}")

    elif action == "set":
        if not key or not value:
            console.print("Both key and value are required for 'set' action.")
            raise typer.Exit(1)

        # Validate locale setting
        if key == "locale" and value not in ["en", "es"]:
            console.print(
                "Invalid locale. Use 'en' for English or 'es' for Spanish.")
            raise typer.Exit(1)

        try:
            config_dir.mkdir(exist_ok=True)

            # Load existing config or create new
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            config[key] = value

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            console.print(f"Configuration updated: {key} = {value}")

            # Reload i18n if locale was changed
            if key == "locale":
                from mlvlab.i18n.core import i18n
                i18n.set_locale(value)
                console.print(f"Language changed to: {value}")

        except Exception as e:
            console.print(f"Error updating configuration: {e}")
            raise typer.Exit(1)

    elif action == "reset":
        if config_file.exists():
            config_file.unlink()
            console.print("Configuration reset to defaults.")
        else:
            console.print("No configuration file to reset.")

    else:
        console.print(f"Unknown action: {action}. Use: get, set, or reset")
        raise typer.Exit(1)


@app.command(name="list")
def list_environments(
    unidad: Optional[str] = typer.Argument(
        None,
        help=i18n.t("cli.options.unit_filter"),
        autocompletion=complete_unit_id
    )
):
    """List available units or environments from a specific unit."""
    from rich.table import Table

    # Recargar configuración del i18n para detectar cambios
    i18n._detect_locale()
    # Recargar traducciones si el locale cambió
    if i18n.current_locale != i18n._last_detected_locale:
        i18n.translations = i18n._load_translations()
        i18n._last_detected_locale = i18n.current_locale

    # Verificar configuración del usuario directamente
    try:
        from pathlib import Path
        import json
        config_file = Path.home() / '.mlvlab' / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                user_locale = user_config.get('locale')
                if user_locale and user_locale in ['en', 'es'] and user_locale != i18n.current_locale:
                    i18n.current_locale = user_locale
                    i18n.translations = i18n._load_translations()
    except:
        pass

    # Detectar entornos por el namespace "mlv" (IDs sin unidad)
    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    # Mapa env_id -> unidad desde su config
    env_id_to_unit: dict[str, str] = {}
    configs: dict[str, dict] = {}
    for env_id in all_env_ids:
        cfg = get_env_config(env_id)
        configs[env_id] = cfg
        unit = cfg.get("UNIT") or cfg.get("unit") or i18n.t("common.other")
        env_id_to_unit[env_id] = str(unit)

    # Si no pasan unidad, listamos las unidades disponibles
    if unidad is None:
        unidades = sorted(set(env_id_to_unit.values()))
        # Primera columna sin título para mostrar un emoji por unidad (🐜 para 'ants')
        table = Table("", i18n.t("cli.tables.unit_id"), i18n.t(
            "cli.tables.collection"), i18n.t("cli.tables.description"))
        for u in unidades:
            emoji = i18n.t(f"cli.units.{u}") if u in [
                'ants', 'ql', 'sarsa', 'dqn', 'dqnp'] else ''
            coleccion = i18n.t(f"cli.unit_names.{u}") if u in [
                'ants', 'ql', 'sarsa', 'dqn', 'dqnp'] else i18n.t("common.dash")
            desc = i18n.t(f"cli.unit_descriptions.{u}") if u in [
                'ants', 'ql', 'sarsa', 'dqn', 'dqnp'] else i18n.t("common.custom_unit")
            table.add_row(emoji, f"[cyan]{u}[/cyan]", coleccion, desc)
        console.print(table)
        console.print(i18n.t("cli.messages.use_list_unit"))
        return

    # Con unidad proporcionada: filtrar por config
    unit_envs = [env_id for env_id, u in env_id_to_unit.items() if u == unidad]

    if unidad == 'ants':
        # Vista extendida sin columna de emoji
        table = Table(i18n.t("cli.tables.name_mlv"), i18n.t("cli.tables.id_gymnasium"),
                      i18n.t("cli.tables.description"), i18n.t("cli.tables.baseline_agent"))
        for env_id in sorted(unit_envs):
            config = configs.get(env_id) or get_env_config(env_id)
            desc = config.get("DESCRIPTION", i18n.t("common.no_description"))
            baseline = config.get("BASELINE", {}).get(
                "agent", f"[red]{i18n.t('common.not_available')}[/red]")
            short_name = env_id.split('/')[-1]
            table.add_row(f"[cyan]{short_name}[/cyan]",
                          f"[cyan]{env_id}[/cyan]", desc, baseline)
        console.print(table)
    else:
        table = Table(i18n.t("cli.tables.name_mlv"), i18n.t("cli.tables.id_gymnasium"), i18n.t("cli.tables.collection"),
                      i18n.t("cli.tables.description"), i18n.t("cli.tables.baseline_agent"))
        for env_id in sorted(unit_envs):
            config = configs.get(env_id) or get_env_config(env_id)
            desc = config.get("DESCRIPTION", i18n.t("common.no_description"))
            baseline = config.get("BASELINE", {}).get(
                "agent", f"[red]{i18n.t('common.not_available')}[/red]")
            short_name = env_id.split('/')[-1]
            coleccion = i18n.t("cli.unit_names.ants") if (
                config.get("UNIT") == 'ants') else i18n.t("common.dash")
            table.add_row(f"[cyan]{short_name}[/cyan]",
                          f"[cyan]{env_id}[/cyan]", coleccion, desc, baseline)
        console.print(table)


@app.command(name="play")
def play_command(
    env_id: str = typer.Argument(..., help="Environment ID to play (e.g., AntScout-v1 or mlv/AntScout-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help=i18n.t("cli.options.seed")),
):
    """Play interactively in an environment (render_mode=human)."""
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    console.print(i18n.t("cli.messages.launching", env_id=normalized_env_id))

    try:
        # Obtenemos la configuración específica (mapeo de teclas)
        config = get_env_config(normalized_env_id)
        key_map = config.get("KEY_MAP", None)

        if key_map is None:
            console.print(i18n.t("cli.messages.error_no_keymap",
                          env_id=normalized_env_id))
            raise typer.Exit(code=1)

        # Usamos el player genérico
        play_interactive(normalized_env_id, key_map=key_map, seed=seed)

    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)


@app.command(name="view")
def view_command(
    env_id: str = typer.Argument(..., help="Environment ID to open view (e.g., AntScout-v1 or mlv/AntScout-v1).",
                                 autocompletion=complete_env_id),
):
    """Launch the interactive view associated with an environment."""
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    console.print(
        f"🖥️  Abriendo view para [bold cyan]{normalized_env_id}[/bold cyan]...")
    try:
        # Resolver el módulo de view a partir del entry_point de Gym para evitar errores de mayúsculas
        spec = gym.spec(normalized_env_id)
        # p.ej., mlvlab.envs.ant_scout_v1.env
        entry_point = spec.entry_point.split(':')[0]
        # mlvlab.envs.ant_scout_v1
        base_pkg = '.'.join(entry_point.split('.')[:-1])
        module_path = f"{base_pkg}.view"
        mod = importlib.import_module(module_path)
        if hasattr(mod, "main"):
            mod.main()
        else:
            console.print(
                i18n.t("cli.messages.error_no_main_function", module_path=module_path))
            raise typer.Exit(code=1)
    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            i18n.t("cli.messages.error_cannot_start_view", error=str(e)))
        raise typer.Exit(code=1)


@app.command(name="train")
def train_command(
    env_id: str = typer.Argument(..., help="Environment ID to train (e.g., AntScout-v1 or mlv/AntScout-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help=i18n.t("cli.options.seed")),
    eps: Optional[int] = typer.Option(
        None, "--eps", "-e", help=i18n.t("cli.options.episodes")),
    render: bool = typer.Option(
        False, "--render", "-r", help=i18n.t("cli.options.render")),
):
    """Train an agent. If no seed is specified, a random one is generated."""
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    config = get_env_config(normalized_env_id)
    baseline = config.get("BASELINE", {})
    # Nuevo flujo: seleccionar algoritmo desde ALGORITHM/UNIT
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT", None)
    train_config = baseline.get("config", {}).copy()

    if not algorithm_key:
        console.print(i18n.t("cli.messages.error_no_algorithm",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)

    # LÓGICA DE SEMILLA ALEATORIA ---
    run_seed = seed
    if run_seed is None:
        run_seed = random.randint(0, 10000)
        console.print(i18n.t("cli.messages.no_seed_random", seed=run_seed))

    # Obtener/crear el directorio para este 'run'
    run_dir = get_run_dir(normalized_env_id, run_seed)
    console.print(i18n.t("cli.messages.working_dir", run_dir=str(run_dir)))

    # Nuevo: usar registro de algoritmos (asegurar carga de plugins integrados)
    from mlvlab.algorithms.registry import get_algorithm

    try:
        algo = get_algorithm(algorithm_key)
        algo.train(normalized_env_id, train_config, run_dir=run_dir,
                   seed=run_seed, render=render)
    except Exception as e:
        console.print(i18n.t("cli.messages.error_training",
                      algorithm_key=algorithm_key, error=str(e)))
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_command(
    env_id: str = typer.Argument(..., help="Environment ID to evaluate (e.g., AntScout-v1 or mlv/AntScout-v1).",
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help=i18n.t("cli.options.seed")),
    episodes: int = typer.Option(
        5, "--eps", "-e", help=i18n.t("cli.options.episodes")),
    speed: float = typer.Option(
        1.0, "--speed", "-sp", help=i18n.t("cli.options.speed")),
    record: bool = typer.Option(
        False, "--rec", "-r", help=i18n.t("cli.options.record")),
):
    """Evaluate an agent interactively by default; use --rec to record."""
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    run_dir = None
    if seed is not None:
        run_dir = get_run_dir(normalized_env_id, seed)
    else:
        console.print(i18n.t("cli.messages.searching_last_training"))
        run_dir = find_latest_run_dir(normalized_env_id)

    if not run_dir or not (run_dir / "q_table.npy").exists():
        console.print(i18n.t("cli.messages.error_no_training"))
        raise typer.Exit(code=1)

    console.print(i18n.t("cli.messages.evaluating_from", run_dir=str(run_dir)))

    # Extraer la semilla del nombre de la carpeta para la reproducibilidad del mapa
    try:
        eval_seed = int(run_dir.name.split('-')[1])
    except (IndexError, ValueError):
        console.print(
            i18n.t("cli.messages.error_cannot_extract_seed", folder_name=run_dir.name))
        eval_seed = None

    # Obtener algoritmo desde la configuración del entorno
    config = get_env_config(normalized_env_id)
    algorithm_key = config.get("ALGORITHM") or config.get("UNIT", None)

    if not algorithm_key:
        console.print(i18n.t("cli.messages.error_no_algorithm",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)

    from mlvlab.algorithms.registry import get_algorithm

    try:
        algo = get_algorithm(algorithm_key)
        algo.eval(
            normalized_env_id,
            run_dir=run_dir,
            episodes=episodes,
            seed=eval_seed,
            video=record,
            speed=speed
        )
    except Exception as e:
        console.print(i18n.t("cli.messages.error_evaluation",
                      algorithm_key=algorithm_key, error=str(e)))
        raise typer.Exit(code=1)


@app.command(name="docs")
def docs_command(
    env_id: str = typer.Argument(..., help="Environment ID to show documentation (e.g., AntScout-v1 or mlv/AntScout-v1).",
                                 autocompletion=complete_env_id),
):
    """Show technical specifications and open documentation in browser."""
    import webbrowser
    from pathlib import Path

    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)

    try:
        env = gym.make(normalized_env_id)

        # Obtener configuración del entorno
        config = get_env_config(normalized_env_id)

        # Construir URL de documentación basada en el idioma configurado
        base_url = "https://github.com/hcosta/mlvlab/blob/master"
        env_name = normalized_env_id.replace("mlv/", "").lower()

        # Mapear nombres de entorno a nombres de directorio
        env_dir_mapping = {
            "antscout-v1": "ant_scout_v1",
            # Agregar más mapeos según sea necesario
        }

        # Usar el mapeo si existe, sino convertir a snake_case
        dir_name = env_dir_mapping.get(env_name, env_name.replace("-", "_"))

        # Determinar el idioma para la documentación
        locale = i18n.current_locale
        if locale == "es":
            docs_url = f"{base_url}/mlvlab/envs/{dir_name}/README_es.md"
        else:
            docs_url = f"{base_url}/mlvlab/envs/{dir_name}/README.md"

        # Intentar abrir en el navegador
        try:
            webbrowser.open(docs_url)
            console.print(i18n.t("cli.messages.docs_browser_opened"))
        except Exception as e:
            console.print(
                i18n.t("cli.messages.docs_browser_error", error=str(e)))

        # Mostrar información en terminal
        console.print(
            f"\n[bold cyan]{i18n.t('cli.messages.docs_full_documentation')}[/bold cyan]")
        console.print(f"{docs_url}\n")
        console.print(
            f"[bold cyan]{i18n.t('cli.messages.docs_observation_space')}[/bold cyan]")
        console.print(f"{env.observation_space}\n")
        console.print(
            f"[bold cyan]{i18n.t('cli.messages.docs_action_space')}[/bold cyan]")
        console.print(f"{env.action_space}\n")

        env.close()
    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)


# =============================================================================
# PLUGINS
# =============================================================================


def _load_plugins():
    """Carga plugins registrados en entry_points."""
    import pkg_resources
    plugin_names = set()

    for entry_point in pkg_resources.iter_entry_points("mlvlab.plugins"):
        try:
            plugin = entry_point.load()
            if hasattr(plugin, "app"):
                app.add_typer(plugin.app, name=entry_point.name)
                plugin_names.add(entry_point.name)
        except Exception as e:
            console.print(i18n.t("cli.messages.error_plugin_load",
                          plugin_name=entry_point.name, error=str(e)))
    return plugin_names


if __name__ == "__main__":
    _load_plugins()
    app()

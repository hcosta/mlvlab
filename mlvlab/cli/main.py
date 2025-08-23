# mlvlab/cli/main.py

from __future__ import annotations

import os
import sys
import typer
import importlib
import subprocess
import importlib.util
import gymnasium as gym
from pathlib import Path
from typing import Optional
from gymnasium.error import NameNotFound
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import NestedCompleter

# Importamos las nuevas utilidades de gestión de 'runs'
from mlvlab.cli.utils import console, get_env_config

# Importamos el sistema de internacionalización
from mlvlab.i18n.core import i18n


def clear_screen(command=None):
    """Limpia la pantalla del terminal conservando solo la última línea del comando."""
    # Limpiar toda la pantalla
    os.system('cls' if os.name == 'nt' else 'clear')

    # Mostrar el comando que se está ejecutando con colores usando HTML como en el shell
    if command:
        # Usar el mismo formato HTML que funciona en el shell interactivo
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit import print_formatted_text

        prompt_html = HTML(
            f"<skyblue><b>MLV-Lab</b></skyblue><b>></b> {command}")
        print_formatted_text(prompt_html)
    else:
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit import print_formatted_text

        prompt_html = HTML(
            "<skyblue><b>MLV-Lab</b></skyblue><b>></b> [command]")
        print_formatted_text(prompt_html)


app = typer.Typer(
    rich_markup_mode="rich",
    help=i18n.t("cli.help.main"),
    no_args_is_help=True,
    add_completion=False  # Oculta las opciones de autocompletado --install-completion
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


@app.command(name="config", help=i18n.t("cli.help.config"))
def config_command(
    action: str = typer.Argument(...,
                                 help="Action to perform: get, set, reset"),
    key: Optional[str] = typer.Argument(
        None, help="Configuration key (e.g., locale)"),
    value: Optional[str] = typer.Argument(None, help="Value to set"),
):
    # Construir el comando completo para mostrarlo
    command_parts = ["config", action]
    if key:
        command_parts.append(key)
    if value:
        command_parts.append(value)
    command_str = " ".join(command_parts)

    clear_screen(command_str)
    from pathlib import Path
    import json
    from mlvlab.i18n.core import i18n

    config_dir = Path.home() / '.mlvlab'
    config_file = config_dir / 'config.json'

    if action == "get":
        if not config_file.exists():
            console.print(i18n.t("cli.config.no_config_file"))
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            if key:
                if key in config:
                    console.print(f"{key}: {config[key]}")
                else:
                    console.print(
                        i18n.t("cli.config.key_not_found", config_key=key))
            else:
                console.print(i18n.t("cli.config.current_configuration"))
                for k, v in config.items():
                    console.print(f"  {k}: {v}")
        except Exception as e:
            console.print(
                i18n.t("cli.config.error_reading_config", error=str(e)))

    elif action == "set":
        if not key or not value:
            console.print(i18n.t("cli.config.key_value_required"))
            raise typer.Exit(1)

        # Validate locale setting
        if key == "locale" and value not in ["en", "es"]:
            console.print(i18n.t("cli.config.invalid_locale"))
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

            console.print(
                i18n.t("cli.config.config_updated", config_key=key, value=value))

            # Reload i18n if locale was changed
            if key == "locale":
                from mlvlab.i18n.core import i18n
                i18n.set_locale(value)
                console.print(
                    i18n.t("cli.config.language_changed", value=value))

        except Exception as e:
            console.print(
                i18n.t("cli.config.error_updating_config", error=str(e)))
            raise typer.Exit(1)

    elif action == "reset":
        if config_file.exists():
            config_file.unlink()
            console.print(i18n.t("cli.config.config_reset"))
        else:
            console.print(i18n.t("cli.config.no_config_to_reset"))

    else:
        console.print(i18n.t("cli.config.unknown_action", action=action))
        raise typer.Exit(1)


@app.command(name="list", help=i18n.t("cli.help.list"))
def list_environments(
    unidad: Optional[str] = typer.Argument(
        None,
        help=i18n.t("cli.options.unit_filter"),
        autocompletion=complete_unit_id
    )
):
    # --- Parte 1: Preparación y Lógica General ---
    command_parts = ["list"]
    if unidad:
        command_parts.append(unidad)
    command_str = " ".join(command_parts)
    clear_screen(command_str)

    from rich.table import Table

    # Recarga la configuración de i18n para cualquier cambio en vivo
    # (Esta lógica de recarga se puede mantener como está)
    try:
        from pathlib import Path
        import json
        i18n._detect_locale()
        config_file = Path.home() / '.mlvlab' / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                user_locale = user_config.get('locale')
                if user_locale and user_locale in ['en', 'es'] and user_locale != i18n.current_locale:
                    i18n.current_locale = user_locale
                    i18n.translations = i18n._load_translations()
    except Exception:
        pass

    all_env_ids = [env_id for env_id in gym.envs.registry.keys()
                   if env_id.startswith("mlv/")]

    env_id_to_unit: dict[str, str] = {}
    configs: dict[str, dict] = {}
    for env_id in all_env_ids:
        cfg = get_env_config(env_id)
        configs[env_id] = cfg
        unit = cfg.get("UNIT") or cfg.get("unit") or i18n.t("common.other")
        env_id_to_unit[env_id] = str(unit)

    # --- Parte 2: Si no se especifica unidad, mostrar las unidades disponibles ---
    if unidad is None:
        unidades = sorted(set(env_id_to_unit.values()))
        table = Table("", i18n.t("cli.tables.unit_id"), i18n.t(
            "cli.tables.collection"), i18n.t("cli.tables.description"))
        for u in unidades:
            # Esta lógica se puede mejorar en el futuro, pero la dejamos como está por ahora
            emoji = i18n.t(f"cli.units.{u}", default='')
            coleccion = i18n.t(f"cli.unit_names.{u}", default=u.capitalize())
            desc = i18n.t(f"cli.unit_descriptions.{u}", default=i18n.t(
                "common.custom_unit"))
            table.add_row(emoji, f"[cyan]{u}[/cyan]", coleccion, desc)
        console.print(table)
        console.print(i18n.t("cli.messages.use_list_unit"))
        return

    # --- Parte 3: Lógica Unificada y Refactorizada para mostrar entornos de una unidad ---
    unit_envs = [env_id for env_id, u in env_id_to_unit.items() if u == unidad]

    # Define los encabezados de la tabla (con el nombre primero)
    table = Table(
        i18n.t("cli.tables.environment"),
        i18n.t("cli.tables.id_command"),
        i18n.t("cli.tables.objective"),
        i18n.t("cli.tables.baseline_agent")
    )

    # Un único bucle para todos los entornos de la unidad
    for env_id in sorted(unit_envs):
        config = configs.get(env_id) or get_env_config(env_id)
        short_name = env_id.split('/')[-1]
        i18n_key = short_name.replace('-', '_').lower()

        # Carga los nombres descriptivos desde los archivos i18n
        env_name = i18n.t(f"environments.{i18n_key}.name", default=short_name)
        objective = i18n.t(f"environments.{i18n_key}.description", default=i18n.t(
            "common.no_description"))

        baseline_config = config.get("BASELINE", {})
        baseline = baseline_config.get(
            "agent", f"[red]{i18n.t('common.not_available')}[/red]")

        # Añade la fila con el orden preferido
        table.add_row(
            env_name,
            f"[cyan]{short_name}[/cyan]",
            objective,
            baseline.capitalize()
        )

    console.print(table)
    console.print(f"[dim]{i18n.t('cli.messages.dev_note')}[/dim]\n")


@app.command(name="play", help=i18n.t("cli.help.play"))
def play_command(
    env_id: str = typer.Argument(..., help=i18n.t("cli.args.env_id_play"),
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help=i18n.t("cli.options.seed")),
):
    # Construir el comando completo para mostrarlo
    command_parts = ["play", env_id]
    if seed is not None:
        command_parts.extend(["--seed", str(seed)])
    command_str = " ".join(command_parts)

    clear_screen(command_str)
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    console.print(i18n.t("cli.messages.launching", env_id=normalized_env_id))

    try:
        # 1. Encontrar la ruta del script auxiliar del player
        player_entry_script = Path(__file__).parent / "player_entry.py"
        if not player_entry_script.exists():
            console.print(
                f"[red]{i18n.t('cli.errors.player_entry_not_found', path=player_entry_script)}[/red]")
            raise typer.Exit(code=1)

        # 2. Construir y ejecutar el comando en un subproceso
        python_executable = sys.executable
        command = [python_executable, str(player_entry_script), env_id]
        if seed is not None:
            command.extend(["--seed", str(seed)])

        subprocess.run(command)

    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            f"[red]{i18n.t('cli.errors.play_launch_failed', error=e)}[/red]")
        raise typer.Exit(code=1)


@app.command(name="view", help=i18n.t("cli.help.view"))
def view_command(
    env_id: str = typer.Argument(..., help=i18n.t("cli.args.env_id_view"),
                                 autocompletion=complete_env_id),
):
    # Construir el comando completo para mostrarlo
    command_str = f"view {env_id}"

    clear_screen(command_str)
    # Normalizar el ID del entorno
    normalized_env_id = normalize_env_id(env_id)
    console.print(
        i18n.t("cli.view.opening_view", env_id=normalized_env_id))
    try:
        # 1. Encontrar la ruta del script de la vista
        spec = gym.spec(normalized_env_id)
        entry_point = spec.entry_point.split(':')[0]
        base_pkg = '.'.join(entry_point.split('.')[:-1])
        module_name = f"{base_pkg}.view"

        view_spec = importlib.util.find_spec(module_name)
        if view_spec is None or view_spec.origin is None:
            console.print(
                f"[red]{i18n.t('cli.errors.view_script_not_found', module_name=module_name)}[/red]")
            raise typer.Exit(code=1)

        view_script_path = view_spec.origin

        # 2. Construir y ejecutar el comando en un subproceso
        python_executable = sys.executable
        command = [python_executable, view_script_path]

        # subprocess.run bloqueará hasta que la vista se cierre, que es el comportamiento deseado.
        subprocess.run(command)

    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            i18n.t("cli.messages.error_cannot_start_view", error=str(e)))
        raise typer.Exit(code=1)


@app.command(name="train", help=i18n.t("cli.help.train"))
def train_command(
    env_id: str = typer.Argument(..., help=i18n.t("cli.args.env_id_train"),
                                 autocompletion=complete_env_id),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help=i18n.t("cli.options.seed")),
    eps: Optional[int] = typer.Option(
        None, "--eps", "-e", help=i18n.t("cli.options.episodes")),
    render: bool = typer.Option(
        False, "--render", "-r", help=i18n.t("cli.options.render")),
):
    # Construir el comando completo para mostrarlo
    command_parts = ["train", env_id]
    if seed is not None:
        command_parts.extend(["--seed", str(seed)])
    if eps is not None:
        command_parts.extend(["--eps", str(eps)])
    if render:
        command_parts.append("--render")
    command_str = " ".join(command_parts)

    clear_screen(command_str)
    try:
        # 1. Encontrar la ruta del script auxiliar de entrenamiento
        train_entry_script = Path(__file__).parent / "train_entry.py"
        if not train_entry_script.exists():
            console.print(
                f"[red]{i18n.t('cli.errors.train_entry_not_found', path=train_entry_script)}[/red]")
            raise typer.Exit(code=1)

        # 2. Construir y ejecutar el comando en un subproceso
        python_executable = sys.executable
        command = [python_executable, str(train_entry_script), env_id]

        # Añadir argumentos opcionales si se proporcionan
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if eps is not None:
            command.extend(["--eps", str(eps)])
        if render:
            command.append("--render")

        subprocess.run(command)

    except Exception as e:
        console.print(
            f"[red]{i18n.t('cli.errors.train_launch_failed', error=e)}[/red]")
        raise typer.Exit(code=1)


@app.command(name="eval", help=i18n.t("cli.help.eval"))
def eval_command(
    env_id: str = typer.Argument(..., help=i18n.t("cli.args.env_id_eval"),
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
    # Construir el comando completo para mostrarlo
    command_parts = ["eval", env_id]
    if seed is not None:
        command_parts.extend(["--seed", str(seed)])
    if episodes != 5:  # Solo mostrar si no es el valor por defecto
        command_parts.extend(["--eps", str(episodes)])
    if speed != 1.0:  # Solo mostrar si no es el valor por defecto
        command_parts.extend(["--speed", str(speed)])
    if record:
        command_parts.append("--rec")
    command_str = " ".join(command_parts)

    clear_screen(command_str)
    try:
        # 1. Encontrar la ruta del script auxiliar de evaluación
        eval_entry_script = Path(__file__).parent / "eval_entry.py"
        if not eval_entry_script.exists():
            console.print(
                f"[red]{i18n.t('cli.errors.eval_entry_not_found', path=eval_entry_script)}[/red]")
            raise typer.Exit(code=1)

        # 2. Construir y ejecutar el comando en un subproceso
        python_executable = sys.executable
        command = [python_executable, str(eval_entry_script), env_id]

        # Añadir argumentos opcionales si se proporcionan
        if seed is not None:
            command.extend(["--seed", str(seed)])
        if episodes != 5:  # Solo añadir si no es el valor por defecto
            command.extend(["--episodes", str(episodes)])
        if speed != 1.0:  # Solo añadir si no es el valor por defecto
            command.extend(["--speed", str(speed)])
        if record:
            command.append("--record")

        subprocess.run(command)

    except Exception as e:
        console.print(
            f"[red]{i18n.t('cli.errors.eval_launch_failed', error=e)}[/red]")
        raise typer.Exit(code=1)


@app.command(name="shell", help=i18n.t("cli.help.shell"), hidden=True)
def shell_command():
    # Inicializar el estado si no existe
    if not hasattr(shell_command, '_in_shell'):
        shell_command._in_shell = False

    # Verificar si ya estamos en una shell para evitar recursión
    if shell_command._in_shell:
        console.print(
            f"[yellow]{i18n.t('cli.help.in_shell')}[/yellow]")
        return

    shell_command._in_shell = True
    clear_screen("shell")
    console.print(f"[bold green]{i18n.t('cli.repl.welcome')}[/bold green]")
    console.print(i18n.t('cli.repl.exit_tip'))

    # --- 1. Configuración del Autocompletado ---

    # Obtenemos todos los nombres de los comandos de la app
    command_names = [cmd.name for cmd in app.registered_commands if cmd.name]

    # Obtenemos todos los IDs de entorno para autocompletar en play, train, etc.
    try:
        import mlvlab.envs
    except ImportError:
        pass
    all_env_ids = [env_id.replace(
        "mlv/", "") for env_id in gym.envs.registry.keys() if env_id.startswith("mlv/")]

    # Obtenemos todas las unidades
    all_unit_ids = list(complete_unit_id(""))

    # Creamos un diccionario para el autocompletado anidado
    completer_map = {
        command: None for command in command_names
    }
    for command in ["play", "view", "train", "eval", "docs"]:
        completer_map[command] = {env_id: None for env_id in all_env_ids}
    completer_map["list"] = {unit_id: None for unit_id in all_unit_ids}

    # Palabras clave para salir, ayuda y limpiar pantalla
    completer_map["exit"] = None
    completer_map["quit"] = None
    completer_map["cls"] = None
    completer_map["clear"] = None
    completer_map["help"] = {cmd: None for cmd in command_names}

    completer = NestedCompleter.from_nested_dict(completer_map)

    # --- 2. Configuración de la Sesión de Prompt ---

    history_file = Path.home() / '.mlvlab' / 'history.txt'
    history_file.parent.mkdir(exist_ok=True)

    session = PromptSession(
        history=FileHistory(str(history_file)),
        completer=completer,
        complete_while_typing=True,
    )

    # --- 3. Bucle Principal (REPL) ---
    while True:
        try:
            text = session.prompt(
                HTML("<skyblue><b>MLV-Lab</b></skyblue><b>></b> "))
            args = text.split()

            if not args:
                continue

            # Manejar comandos especiales de la shell
            command = args[0].lower()

            if command in ["quit", "exit"]:
                break

            if command in ["cls", "clear"]:
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            if command == 'help':
                # Limpiar pantalla y mostrar el comando
                help_command = " ".join(args)
                clear_screen(help_command)

                help_args = args[1:] + \
                    ['--help'] if len(args) > 1 else ['--help']
                try:
                    app(args=help_args, prog_name="mlv")
                except SystemExit:
                    pass
                continue

            # Ejecutar comando normal
            try:
                app(args=args, prog_name="mlv")
            except SystemExit:
                pass
            except typer.Exit:
                pass
            except Exception as e:
                console.print(f"[bold red]Error inesperado:[/bold red] {e}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    # Limpiar el estado de la shell
    shell_command._in_shell = False
    console.print(f"[yellow]{i18n.t('cli.repl.goodbye')}[/yellow]")


@app.command(name="docs", help=i18n.t("cli.help.docs"))
def docs_command(
    env_id: str = typer.Argument(..., help=i18n.t("cli.args.env_id_docs"),
                                 autocompletion=complete_env_id),
):
    # Construir el comando completo para mostrarlo
    command_str = f"docs {env_id}"

    clear_screen(command_str)
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

        # Intentar abrir en el navegador usando subprocess para evitar bloqueo
        try:
            # Usar subprocess para abrir el navegador de forma no bloqueante
            if os.name == 'nt':  # Windows
                subprocess.Popen(['start', docs_url], shell=True)
            elif os.name == 'posix':  # macOS y Linux
                subprocess.Popen(['xdg-open', docs_url])  # Linux
                # Para macOS también intentamos con 'open'
                if sys.platform == 'darwin':
                    subprocess.Popen(['open', docs_url])
            else:
                # Fallback a webbrowser si no se puede usar subprocess
                webbrowser.open(docs_url)

            console.print("\n" + i18n.t("cli.messages.docs_browser_opened"))
        except Exception as e:
            console.print(
                i18n.t("cli.messages.docs_browser_error", error=str(e)))

        # Mostrar información en terminal
        console.print(f"{docs_url}")
        console.print(
            f"\n[bold yellow]{i18n.t('cli.messages.docs_observation_space')}[/bold yellow]\t{env.observation_space}")
        console.print(
            f"[bold yellow]{i18n.t('cli.messages.docs_action_space')}[/bold yellow]\t\t{env.action_space}\n")

        env.close()
    except NameNotFound:
        console.print(i18n.t("cli.messages.error_env_not_found",
                      env_id=normalized_env_id))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # Flujo normal de la CLI
    app()

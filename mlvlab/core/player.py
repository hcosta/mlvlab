# mlvlab/core/player.py
import gymnasium as gym
import importlib.util
from pathlib import Path
from typing import Optional
import time
import arcade
import pyglet
import random
from mlvlab.i18n.core import i18n


def find_asset_path(env: gym.Env, asset_name: str) -> Optional[Path]:
    """
    Busca la ruta de un asset (e.g., sonido) dentro del paquete del entorno de forma robusta.
    """
    env_module = type(getattr(env, 'unwrapped', env)).__module__
    try:
        module_spec = importlib.util.find_spec(env_module)
        if module_spec and module_spec.origin:
            env_dir = Path(module_spec.origin).parent
            candidate = env_dir / "assets" / asset_name
            if candidate.exists():
                return candidate
            try:
                pkg_dir = env_dir.parent
                if pkg_dir.name and '-' in pkg_dir.name:
                    us_dir = pkg_dir.parent / pkg_dir.name.replace('-', '_')
                    cand2 = us_dir / 'assets' / asset_name
                    if cand2.exists():
                        return cand2
            except Exception:
                pass
    except Exception:
        pass
    return None


def play_interactive(env_id: str, key_map: dict, seed: Optional[int] = None):
    """
    Ejecuta un entorno en modo interactivo usando Arcade/pyglet.
    """
    env = gym.make(env_id, render_mode="human")

    # Estado de ejecución
    running = True
    pending_action = None
    # Estado para controlar la animación final ---
    waiting_for_end_scene = False

    # Sonidos cacheados
    cached_sounds: dict[str, Optional[pyglet.media.Source]] = {}

    # Reset inicial
    obs, info = env.reset(seed=seed)
    env.render()

    window: Optional[arcade.Window] = getattr(env.unwrapped, "window", None)
    if window is None:
        env.render()
        window = getattr(env.unwrapped, "window", None)
    if window is None:
        raise RuntimeError(
            "No se pudo acceder a la ventana de Arcade del entorno.")

    print(i18n.t("cli.messages.game_started"))

    def on_key_press(symbol: int, modifiers: int):
        nonlocal pending_action, running
        if symbol == arcade.key.ESCAPE:
            running = False
            return
        # Ignorar input si estamos en la animación final
        if not waiting_for_end_scene and symbol in key_map:
            pending_action = key_map[symbol]

    def on_close():
        nonlocal running
        running = False

    window.push_handlers(on_key_press=on_key_press, on_close=on_close)

    target_dt = 1.0 / 60.0
    terminated = False
    truncated = False
    while running:
        window.dispatch_events()

        # LÓGICA MODIFICADA PARA LA ANIMACIÓN FINAL
        if waiting_for_end_scene:
            # 1. Si estamos esperando, comprobamos si la animación ha terminado.
            if env.unwrapped.is_end_scene_animation_finished():
                # MODIFICACIÓN: Añadimos AntMaze a la condición para que genere
                # un nuevo mapa en cada episodio del modo de juego interactivo.
                if env_id.startswith("mlv/AntScout") or env_id.startswith("mlv/AntMaze"):
                    # 2. Generamos una nueva semilla aleatoria en cada reseteo.
                    new_seed = random.randint(0, 1_000_000)
                    obs, info = env.reset(seed=new_seed)
                else:
                    # Para otros entornos, mantenemos el comportamiento de reset por defecto.
                    obs, info = env.reset()
                # Reseteamos las flags del episodio para el nuevo comienzo.
                terminated, truncated = False, False
                # Le decimos al bucle que ya no estamos esperando.
                waiting_for_end_scene = False
        elif terminated or truncated:
            # 3. Si el episodio acaba de terminar, activamos la animación y el modo espera.
            if hasattr(env.unwrapped, "trigger_end_scene"):
                env.unwrapped.trigger_end_scene(terminated, truncated)
                waiting_for_end_scene = True
            else:
                # 4. Fallback: si el entorno no tiene la animación, reseteamos como antes.
                time.sleep(0.75)
                obs, info = env.reset()
                terminated, truncated = False, False
        elif pending_action is not None:
            # 5. Si no pasa nada de lo anterior, ejecutamos la acción del jugador.
            obs, reward, terminated, truncated, info = env.step(pending_action)
            pending_action = None

            # Gestión de sonido (sin cambios)
            if 'play_sound' in info:
                sound_data = info['play_sound']
                filename = sound_data.get('filename')
                if filename:
                    if filename not in cached_sounds:
                        asset_path = find_asset_path(env, filename)
                        source = pyglet.media.load(
                            str(asset_path), streaming=False) if asset_path and asset_path.exists() else None
                        cached_sounds[filename] = source
                    source = cached_sounds.get(filename)
                    if source is not None:
                        player = pyglet.media.Player()
                        player.volume = float(
                            sound_data.get('volume', 100)) / 100.0
                        player.queue(source)
                        player.play()

        # Renderizamos en cada fotograma para que la animación se vea fluida
        env.render()
        time.sleep(target_dt)

    try:
        window.pop_handlers()
    except Exception:
        pass
    env.close()

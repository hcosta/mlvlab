# mlvlab/core/player.py
import gymnasium as gym
import importlib.util
from pathlib import Path
from typing import Optional
import time
import arcade
import pyglet


def find_asset_path(env: gym.Env, asset_name: str) -> Optional[Path]:
    """
    Busca la ruta de un asset (e.g., sonido) dentro del paquete del entorno de forma robusta.
    """
    # Obtenemos el módulo donde está definida la clase del entorno (desenvolviendo wrappers si los hay)
    env_module = type(env.unwrapped).__module__
    try:
        # Intentamos encontrar la ruta base del módulo
        module_spec = importlib.util.find_spec(env_module)
        if module_spec and module_spec.origin:
            # Asumimos que los assets están en una carpeta 'assets' junto al archivo .py del entorno.
            env_dir = Path(module_spec.origin).parent
            asset_path = env_dir / "assets" / asset_name
            if asset_path.exists():
                return asset_path
    except Exception:
        pass
    return None


def play_interactive(env_id: str, key_map: dict, seed: Optional[int] = None):
    """
    Ejecuta un entorno en modo interactivo usando Arcade/pyglet.

    - `key_map`: diccionario que mapea códigos de tecla de `arcade.key.*` a acciones del entorno.
                 Ejemplo: {arcade.key.UP: 0, arcade.key.DOWN: 1, arcade.key.LEFT: 2, arcade.key.RIGHT: 3,
                           arcade.key.W: 0, arcade.key.S: 1, arcade.key.A: 2, arcade.key.D: 3}
    """
    env = gym.make(env_id, render_mode="human")

    # Estado de ejecución
    running = True
    pending_action = None

    # Sonidos cacheados (pyglet Source)
    cached_sounds: dict[str, Optional[pyglet.media.Source]] = {}

    # Reset inicial y forzar creación de ventana en el entorno
    obs, info = env.reset(seed=seed)
    env.render()

    # Intentar obtener la ventana de Arcade creada por el entorno
    window: Optional[arcade.Window] = getattr(env.unwrapped, "window", None)
    if window is None:
        # Si no existe aún, intentar render otra vez
        env.render()
        window = getattr(env.unwrapped, "window", None)

    if window is None:
        raise RuntimeError(
            "No se pudo acceder a la ventana de Arcade del entorno.")

    print("Juego iniciado. Usa las teclas definidas (Flechas/WASD). ESC para salir.")

    # Handlers de teclado/cierre
    def on_key_press(symbol: int, modifiers: int):
        nonlocal pending_action, running
        if symbol == arcade.key.ESCAPE:
            running = False
            return
        if symbol in key_map:
            pending_action = key_map[symbol]

    def on_close():
        nonlocal running
        running = False

    # Registrar handlers en la ventana de Arcade (pyglet)
    window.push_handlers(on_key_press=on_key_press, on_close=on_close)

    # Bucle principal
    target_dt = 1.0 / 30.0
    while running:
        # Procesar eventos de ventana
        window.dispatch_events()

        # Ejecutar acción si hay pendiente
        if pending_action is not None:
            obs, reward, terminated, truncated, info = env.step(pending_action)
            pending_action = None

            # Reproducción de sonido (pyglet)
            if 'play_sound' in info:
                sound_data = info['play_sound']
                filename = sound_data.get('filename')
                if filename:
                    if filename not in cached_sounds:
                        asset_path = find_asset_path(env, filename)
                        if asset_path and asset_path.exists():
                            try:
                                source = pyglet.media.load(
                                    str(asset_path), streaming=False)
                            except Exception as e:
                                print(
                                    f"Error al cargar sonido {filename}: {e}")
                                source = None
                        else:
                            source = None
                        cached_sounds[filename] = source
                    source = cached_sounds.get(filename)
                    if source is not None:
                        player = pyglet.media.Player()
                        player.volume = float(
                            sound_data.get('volume', 100)) / 100.0
                        player.queue(source)
                        player.play()

            if terminated or truncated:
                obs, info = env.reset()

        # Render del entorno
        env.render()

        # Pequeña pausa para controlar FPS
        time.sleep(target_dt)

    # Limpieza
    try:
        window.pop_handlers()
    except Exception:
        pass
    env.close()

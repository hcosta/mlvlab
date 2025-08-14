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
    env_module = type(getattr(env, 'unwrapped', env)).__module__
    try:
        # Intentamos encontrar la ruta base del módulo
        module_spec = importlib.util.find_spec(env_module)
        if module_spec and module_spec.origin:
            # Asumimos que los assets están en una carpeta 'assets' junto al archivo .py del entorno.
            env_dir = Path(module_spec.origin).parent
            # 1) Ruta por defecto: junto al módulo del entorno
            candidate = env_dir / "assets" / asset_name
            if candidate.exists():
                return candidate
            # 2) Fallback: si el módulo está envs/<pkg>/ant_env.py y el ID contiene '-', probar underscore
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
    target_dt = 1.0 / 60.0
    terminated = False
    truncated = False
    while running:
        # 1. Procesar eventos de ventana (input del usuario)
        window.dispatch_events()

        # 2. Si el episodio anterior terminó, reiniciamos y volvemos al inicio del bucle.
        if terminated or truncated:
            # print("Episodio terminado. Reiniciando automáticamente...")
            time.sleep(.75)
            obs, info = env.reset()
            terminated, truncated = False, False  # Reseteamos las banderas
            continue  # Saltamos al siguiente ciclo del bucle

        # 3. Si hay una acción del usuario, la ejecutamos.
        if pending_action is not None:
            obs, reward, terminated, truncated, info = env.step(pending_action)
            pending_action = None

            # Tu código de sonido original va aquí.
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

        # 4. Si el entorno está en su animación final, le damos un "tick" para que continúe.
        #    Esto es un pequeño "hack" para no cambiar el entorno. Usamos una acción cualquiera (e.g., 0)
        #    porque en este estado, el entorno la ignorará y solo avanzará la animación.
        elif getattr(env.unwrapped, '_logical_terminated', False):
            obs, reward, terminated, truncated, info = env.step(
                0)  # Le damos un "tick"

        # 5. Renderizamos siempre para que la ventana y las animaciones fluyan.
        env.render()

        # Pausa para controlar FPS
        time.sleep(target_dt)

    # Limpieza
    try:
        window.pop_handlers()
    except Exception:
        pass
    env.close()

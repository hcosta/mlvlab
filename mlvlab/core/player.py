# mlvlab/core/player.py
import gymnasium as gym
import pygame
import importlib.util
from pathlib import Path


def find_asset_path(env: gym.Env, asset_name: str) -> Path | None:
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


def play_interactive(env_id: str, key_map: dict, seed: int | None = None):
    """
    Ejecuta un entorno en modo interactivo usando PyGame.
    """
    env = gym.make(env_id, render_mode="human")
    pygame.init()

    # Inicializar el mixer
    try:
        pygame.mixer.init()
        mixer_available = True
    except pygame.error:
        print("Advertencia: PyGame mixer no disponible. El juego será silencioso.")
        mixer_available = False

    sounds = {}
    obs, info = env.reset(seed=seed)
    env.render()

    running = True
    print("Juego iniciado. Usa las teclas definidas (Flechas/WASD). ESC para salir.")

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    action = key_map[event.key]

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)

            # Lógica de sonido dinámica y robusta
            if mixer_available and 'play_sound' in info:
                sound_data = info['play_sound']
                filename = sound_data['filename']

                # 1. Comprobar caché
                if filename not in sounds:
                    asset_path = find_asset_path(env, filename)
                    if asset_path:
                        try:
                            # 2. Cargar y guardar
                            sound_object = pygame.mixer.Sound(str(asset_path))
                            sounds[filename] = sound_object
                        except pygame.error as e:
                            print(f"Error al cargar {filename}: {e}")
                            sounds[filename] = None
                    else:
                        sounds[filename] = None

                # 3. Reproducir
                sound_to_play = sounds.get(filename)
                if sound_to_play:
                    volume = sound_data.get('volume', 100) / 100.0
                    sound_to_play.set_volume(volume)
                    sound_to_play.play()

            if terminated or truncated:
                obs, info = env.reset()

            env.render()

    env.close()
    pygame.quit()

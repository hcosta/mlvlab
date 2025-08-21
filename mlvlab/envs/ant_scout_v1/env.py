# mlvlab/envs/ant/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading
import os
import platform
import sys

# Parche para arreglar el bug de HeadlessScreen en Pyglet


def _patch_pyglet_headless():
    """Parche para arreglar el bug de HeadlessScreen en Pyglet cuando se usa con Arcade."""
    try:
        import pyglet.display.headless

        # Crear una implementación concreta de HeadlessScreen si no tiene los métodos requeridos
        class PatchedHeadlessScreen(pyglet.display.headless.HeadlessScreen):
            def get_display_id(self):
                return 0

            def get_monitor_name(self):
                return "Headless"

        # Sobrescribir la clase original
        pyglet.display.headless.HeadlessScreen = PatchedHeadlessScreen

        # También parchear HeadlessDisplay para evitar el error de _screens
        original_headless_display = pyglet.display.headless.HeadlessDisplay

        class PatchedHeadlessDisplay(original_headless_display):
            def __init__(self):
                super().__init__()
                if not hasattr(self, '_screens'):
                    self._screens = [PatchedHeadlessScreen(
                        self, 0, 0, 1920, 1080)]

        pyglet.display.headless.HeadlessDisplay = PatchedHeadlessDisplay

    except Exception:
        pass  # Si no podemos parchear, continuamos


# Aplicar el parche al importar
_patch_pyglet_headless()

# Importamos las clases modularizadas
try:
    from .game import AntGame
    from .renderer import ArcadeRenderer
except ImportError:
    # Fallback para ejecución directa o si la estructura del paquete falla
    from game import AntGame
    from renderer import ArcadeRenderer


class ScoutAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 reward_goal=100,
                 reward_obstacle=-100,
                 reward_move=-1,
                 ):
        super().__init__()

        # Detección del entorno
        self._is_colab = 'google.colab' in sys.modules
        self._is_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules

        # Configuración para renderizado rgb_array
        self._headless_enabled = False
        self._supports_headless = False

        if render_mode == "rgb_array":
            # Configuración especial para Google Colab y notebooks
            if self._is_colab or self._is_notebook:
                # Forzar modo headless con el parche aplicado
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                os.environ["ARCADE_HEADLESS"] = "True"
                # En Colab, a veces necesitamos esto
                if self._is_colab:
                    os.environ["SDL_VIDEODRIVER"] = "dummy"
                self._headless_enabled = True
            elif platform.system() != "Windows":
                # En Linux/Mac, intentar modo headless
                try:
                    import pyglet.libs.egl.egl
                    os.environ["ARCADE_HEADLESS"] = "True"
                    self._headless_enabled = True
                    self._supports_headless = True
                except (ImportError, OSError):
                    # Si EGL no está disponible, usar modo offscreen
                    os.environ["SDL_VIDEODRIVER"] = "dummy"
                    if "DISPLAY" not in os.environ:
                        os.environ["DISPLAY"] = ":99"
                    self._headless_enabled = False
            else:
                # Windows no soporta headless, pero podemos intentar crear ventana invisible
                self._headless_enabled = False

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        self.REWARD_GOAL = reward_goal
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        # Lógica del juego (Delegada a AntGame)
        self._game = AntGame(
            grid_size=grid_size,
            reward_goal=reward_goal,
            reward_obstacle=reward_obstacle,
            reward_move=reward_move,
        )
        self._renderer: ArcadeRenderer | None = None

        # Referencias externas para compatibilidad
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles

        # Configuración de renderizado
        self.render_mode = render_mode
        self.window = None
        self.q_table_to_render = None
        self.debug_mode = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad para respawn
        self._respawn_unseeded: bool = False
        try:
            self._respawn_rng = np.random.default_rng()
        except Exception:
            self._respawn_rng = np.random.RandomState()

        # Sistema de animación de fin de escena
        self._state_store = None
        self._end_scene_state = "IDLE"
        self._end_scene_finished_event = threading.Event()
        self._simulation_speed = 1.0

    def _sync_game_state(self):
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles

    def _get_respawn_rng(self):
        if getattr(self, "_respawn_unseeded", False) and self._respawn_rng is not None:
            return self._respawn_rng
        return self.np_random

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {"goal_pos": np.array(self.goal_pos, dtype=np.int32)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._end_scene_state = "IDLE"

        if self._renderer:
            self._renderer.reset()

        scenario_not_ready = (not np.any(self._game.goal_pos)) or (
            not self._game.obstacles)

        if seed is not None or scenario_not_ready:
            self._game.generate_scenario(self.np_random)

        rng = self._get_respawn_rng()
        self._game.place_ant(rng)
        self._sync_game_state()
        # Render inmediato en modo human para abrir/actualizar la ventana
        if self.render_mode == "human":
            try:
                self.render()
            except Exception:
                pass
        return self._get_obs(), self._get_info()

    def step(self, action):
        obs, reward, terminated, game_info = self._game.step(action)
        truncated = False
        info = self._get_info()
        info.update(game_info)

        if info.get("collided", False) and self.render_mode in ["human", "rgb_array"]:
            self._lazy_init_renderer()
            if self._renderer:
                self._renderer._spawn_collision_particles()

        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        self._sync_game_state()
        # Render inmediato en modo human para reflejar el frame actual
        if self.render_mode == "human":
            try:
                self.render()
            except Exception:
                pass
        return obs, reward, terminated, truncated, info

    def _lazy_init_renderer(self):
        if self._renderer is None:
            try:
                # Configuración adicional para rgb_array sin headless real
                if self.render_mode == "rgb_array" and not self._headless_enabled:
                    os.environ["SDL_VIDEODRIVER"] = "dummy"
                    if "DISPLAY" not in os.environ:
                        os.environ["DISPLAY"] = ":99"

                import arcade

                # Verificar versión de Arcade para compatibilidad
                arcade_version = tuple(
                    map(int, arcade.version.VERSION.split('.')))
                if arcade_version >= (3, 0, 0):
                    # Arcade 3.x tiene mejor soporte para headless
                    pass

            except ImportError:
                if self.render_mode in ["human", "rgb_array"]:
                    raise ImportError(
                        "Se requiere 'arcade' para el renderizado. Instálalo con 'pip install arcade'")
                return None

            # Usar tu renderer original
            self._renderer = ArcadeRenderer()
            self._renderer._headless_mode = (
                self.render_mode == "rgb_array" and not self._headless_enabled)

    def render(self):
        self._lazy_init_renderer()
        if self._renderer is None:
            return None
        result = self._render_frame()
        if result is None:
            return None
        width, height = result
        if self.render_mode == "human":
            self._handle_human_render()
        elif self.render_mode == "rgb_array":
            return self._capture_rgb_array(width, height)

    def _render_frame(self):
        if self._renderer is not None:
            if self._end_scene_state == "REQUESTED":
                self._renderer.start_success_transition()
                self._end_scene_state = "RUNNING"
            if self._end_scene_state == "RUNNING":
                if not self._renderer.is_in_success_transition():
                    self._end_scene_state = "IDLE"
                    self._end_scene_finished_event.set()
            self._renderer.debug_mode = self.debug_mode

        result = self._renderer.draw(
            self._game, self.q_table_to_render, self.render_mode,
            simulation_speed=self._simulation_speed
        )

        if self._renderer is not None:
            self.window = self._renderer.window
        return result

    def _handle_human_render(self):
        if self.window is not None:
            try:
                self.window.dispatch_events()
                self.window.flip()
            except Exception:
                pass
        target_sleep = 1.0 / float(self.metadata.get("render_fps", 60))
        time.sleep(target_sleep)

    def _capture_rgb_array(self, width, height):
        arcade_module = None
        if self._renderer and self._renderer.arcade:
            arcade_module = self._renderer.arcade
        else:
            try:
                import arcade
                arcade_module = arcade
            except ImportError:
                return None
        if arcade_module is None:
            return None

        try:
            # Aseguramos que el contexto esté activo
            if self.window:
                self.window.switch_to()

            # Intentamos diferentes métodos para capturar la imagen
            image = None

            # Para Arcade 3.x, usar el método correcto
            try:
                # Método para Arcade 3.x
                image = arcade_module.get_image(0, 0, width, height)
            except (TypeError, AttributeError):
                try:
                    # Método alternativo sin parámetros
                    image = arcade_module.get_image()
                    # Redimensionar si es necesario
                    if image and hasattr(image, 'size'):
                        if image.size != (width, height):
                            image = image.resize((width, height))
                except (TypeError, AttributeError):
                    # Último intento: capturar del framebuffer directamente
                    try:
                        from PIL import Image
                        import array

                        # Leer pixels del framebuffer
                        buffer = (arcade_module.gl.GLubyte *
                                  (width * height * 4))()
                        arcade_module.gl.glReadPixels(0, 0, width, height,
                                                      arcade_module.gl.GL_RGBA,
                                                      arcade_module.gl.GL_UNSIGNED_BYTE,
                                                      buffer)

                        # Convertir a imagen PIL
                        image = Image.frombytes(
                            'RGBA', (width, height), buffer, 'raw', 'RGBA', 0, -1)
                        image = image.convert('RGB')
                    except Exception:
                        pass

            if image is None:
                # Si todo falla, crear un frame negro
                return np.zeros((height, width, 3), dtype=np.uint8)

        except Exception as e:
            print(f"Error al capturar imagen rgb_array: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        try:
            # Convertir a RGB si no lo está
            if hasattr(image, 'convert'):
                image = image.convert("RGB")

            # Convertir a numpy array
            frame = np.asarray(image)

            # Asegurar que tiene las dimensiones correctas
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return frame
            else:
                print(f"Formato de imagen inesperado: {frame.shape}")
                return np.zeros((height, width, 3), dtype=np.uint8)

        except Exception as e:
            print(f"Error al convertir imagen a array numpy: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        if self.window:
            try:
                self.window.close()
            except Exception:
                try:
                    import arcade
                    arcade.close_window()
                except Exception:
                    pass
            self.window = None
            self._renderer = None

    # API Extendida ---
    def set_simulation_speed(self, speed: float):
        self._simulation_speed = speed

    def set_respawn_unseeded(self, flag: bool = True):
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, **kwargs):
        self.q_table_to_render = kwargs.get('q_table')

    def set_state_store(self, state_store):
        self._state_store = state_store

    def trigger_end_scene(self):
        if self.render_mode in ["human", "rgb_array"]:
            self._end_scene_state = "REQUESTED"

    def is_end_scene_animation_finished(self) -> bool:
        if self._renderer is None:
            return True
        return self._end_scene_state == "IDLE"

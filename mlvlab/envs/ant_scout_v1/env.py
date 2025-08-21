# mlvlab/envs/ant/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading
import os
import platform

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

        # Configuración para renderizado rgb_array
        # Intentamos usar modo headless solo en sistemas compatibles
        self._headless_enabled = False
        self._supports_headless = False
        if render_mode == "rgb_array":
            # Intentamos detectar si el sistema soporta headless
            try:
                # En Windows y algunos sistemas, EGL no está disponible
                if platform.system() != "Windows":
                    # Probamos si podemos importar las dependencias headless
                    import pyglet.libs.egl.egl
                    self._supports_headless = True
                    if "ARCADE_HEADLESS" not in os.environ:
                        os.environ["ARCADE_HEADLESS"] = "True"
                    self._headless_enabled = True
            except (ImportError, OSError):
                # Si no podemos usar headless, usaremos un renderer alternativo
                self._supports_headless = False
                self._headless_enabled = False
        # =================================================================

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
            # --- INICIO DE LA SOLUCIÓN PARA ARCADE 3.3.2 ---
            if self.render_mode == "rgb_array":
                # Este truco de bajo nivel fuerza un driver de video "falso"
                # ANTES de que arcade se importe, evitando el bug de Pyglet en Colab.
                import os
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            # --- FIN DE LA SOLUCIÓN ---

            try:
                import arcade
            except ImportError:
                if self.render_mode in ["human", "rgb_array"]:
                    raise ImportError(
                        "Se requiere 'arcade' para el renderizado.")
                return None

            # Pasamos información sobre el modo headless al renderer
            self._renderer = ArcadeRenderer()
            # Esta línea ya no es crítica, pero la dejamos por compatibilidad
            self._renderer._headless_mode = (self.render_mode == "rgb_array")

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
            try:
                # Método preferido: especificar coordenadas
                image = arcade_module.get_image(0, 0, width, height)
            except (TypeError, AttributeError):
                try:
                    # Método alternativo: capturar toda la ventana
                    image = arcade_module.get_image()
                except (TypeError, AttributeError):
                    try:
                        # Método de fallback usando PIL si está disponible
                        import PIL.ImageGrab
                        if self.window and hasattr(self.window, 'get_size'):
                            # Capturar solo el área de la ventana
                            image = PIL.ImageGrab.grab()
                            if image:
                                image = image.resize((width, height))
                    except Exception:
                        pass

            if image is None:
                print("No se pudo capturar imagen, devolviendo frame negro")
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
                # No voltear - el renderer ya maneja las coordenadas Y correctamente
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

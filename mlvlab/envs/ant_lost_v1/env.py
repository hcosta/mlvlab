# mlvlab/envs/ant_lost_v1/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading
import os
import platform

# Importamos las clases modularizadas
try:
    from .game import AntLostGame
    from .renderer import ArcadeRenderer
except ImportError:
    # Fallback para ejecución directa
    from game import AntLostGame
    from renderer import ArcadeRenderer

# =============================================================================
# GESTOR AUTOMÁTICO DE DISPLAY VIRTUAL (Mantenido de AntScout)
# =============================================================================


class _VirtualDisplayManager:
    _display = None
    _is_active = False

    @classmethod
    def start_if_needed(cls):
        if cls._is_active:
            return
        is_linux = platform.system() == "Linux"
        is_headless = "DISPLAY" not in os.environ
        if is_linux and is_headless:
            # print("INFO: Entorno headless detectado. Iniciando display virtual (Xvfb)...")
            try:
                from pyvirtualdisplay import Display
                cls._display = Display(visible=0, size=(1024, 768))
                cls._display.start()
                cls._is_active = True
                # print("INFO: Display virtual iniciado con éxito.")
            except ImportError:
                print("ADVERTENCIA: 'pyvirtualdisplay' no está instalado.")
            except Exception as e:
                # print(f"ERROR: No se pudo iniciar el display virtual: {e}")
                pass

    @classmethod
    def stop(cls):
        if cls._is_active and cls._display:
            try:
                cls._display.stop()
            except Exception:
                pass
            finally:
                cls._is_active = False
                cls._display = None

# =============================================================================


class LostAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 # Límite de pasos antes de morir (100-200)
                 max_steps=150,
                 reward_obstacle=-10,   # Castigo por chocar
                 reward_move=-1,        # Coste por moverse
                 # Recompensa por morir (grande y negativa)
                 reward_death=-200,
                 ):
        super().__init__()

        if render_mode in ["human", "rgb_array"]:
            _VirtualDisplayManager.start_if_needed()

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        self.MAX_STEPS = max_steps
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move
        self.REWARD_DEATH = reward_death

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        # Lógica del juego (Delegada a AntLostGame)
        self._game = AntLostGame(
            grid_size=grid_size,
            max_steps=max_steps,
            reward_obstacle=reward_obstacle,
            reward_move=reward_move,
            reward_death=reward_death,
        )
        self._renderer: ArcadeRenderer | None = None

        # Referencias externas para compatibilidad
        self.ant_pos = self._game.ant_pos
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

        # Sistema de animación de fin de escena (Muerte)
        self._state_store = None
        self._end_scene_state = "IDLE"
        self._end_scene_finished_event = threading.Event()
        self._simulation_speed = 1.0

    def _sync_game_state(self):
        self.ant_pos = self._game.ant_pos
        self.obstacles = self._game.obstacles

    def _get_respawn_rng(self):
        if getattr(self, "_respawn_unseeded", False) and self._respawn_rng is not None:
            return self._respawn_rng
        return self.np_random

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {"steps": self._game.current_step}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._end_scene_state = "IDLE"

        if self._renderer:
            self._renderer.reset()

        # Controlamos la generación del escenario
        if seed is not None or not self._game.obstacles:
            self._game.generate_scenario(self.np_random)

        rng = self._get_respawn_rng()
        # Reseteamos el estado del juego (posición, pasos, etc.)
        self._game.reset(rng)
        self._sync_game_state()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        # El agente (alumno o aleatorio) provee la acción.
        obs, reward, terminated, game_info = self._game.step(action)

        # En este caso, max_steps ES la condición de terminación (muerte), por lo que usamos terminated.
        truncated = False
        info = self._get_info()
        info.update(game_info)

        if info.get("collided", False) and self.render_mode in ["human", "rgb_array"]:
            self._lazy_init_renderer()
            if self._renderer:
                self._renderer._spawn_collision_particles()

        # Gestión de sonidos
        if terminated and info.get("is_dead", False):
            # La hormiga ha muerto (Sonido de fallo)
            info['play_sound'] = {'filename': 'fail.wav', 'volume': 10}
        elif info.get("collided", False):
            # La hormiga ha chocado (Sonido de golpe)
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        self._sync_game_state()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _lazy_init_renderer(self):
        if self._renderer is None:
            try:
                import arcade
            except ImportError:
                if self.render_mode in ["human", "rgb_array"]:
                    raise ImportError(
                        "Se requiere 'arcade' para el renderizado.")
                return

            self._renderer = ArcadeRenderer()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self._renderer is None:
            self._lazy_init_renderer()

        if self._renderer is None:
            return None

        result = self._render_frame()
        if result is None:
            return None

        width, height = result
        if self.render_mode == "human":
            self._handle_human_render()
            return None
        elif self.render_mode == "rgb_array":
            return self._capture_rgb_array(width, height)

    def _render_frame(self):
        if self._renderer is not None:
            # Gestión de la escena final (Muerte)
            # Se activa si se solicita externamente O si el juego reporta la muerte.
            if self._end_scene_state == "REQUESTED" or self._game.is_dead:
                if self._end_scene_state != "RUNNING":
                    self._renderer.start_death_transition()
                    self._end_scene_state = "RUNNING"

            if self._end_scene_state == "RUNNING":
                # Esperamos a que la animación de muerte termine
                if not self._renderer.is_in_death_transition():
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

    # Métodos _handle_human_render, _capture_rgb_array, close y API Extendida se mantienen igual

    def _handle_human_render(self):
        if self.window is not None:
            self.window.dispatch_events()
            self.window.flip()
        time.sleep(1.0 / self.metadata["render_fps"])

    def _capture_rgb_array(self, width, height):
        if not self._renderer or not self._renderer.arcade:
            return np.zeros((height, width, 3), dtype=np.uint8)

        arcade_module = self._renderer.arcade
        try:
            if self.window:
                self.window.switch_to()
            image = arcade_module.get_image(0, 0, width, height)
            return np.asarray(image.convert("RGB"))
        except Exception as e:
            print(f"Error al capturar imagen rgb_array: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        if self.window:
            try:
                self.window.close()
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
        # Se llama cuando el episodio termina (la hormiga muere) para iniciar la animación
        if self.render_mode in ["human", "rgb_array"]:
            self._end_scene_state = "REQUESTED"

    def is_end_scene_animation_finished(self) -> bool:
        if self._renderer is None:
            return True
        return self._end_scene_state == "IDLE"

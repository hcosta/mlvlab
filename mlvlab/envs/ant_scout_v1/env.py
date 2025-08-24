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

# =============================================================================
# GESTOR AUTOMÁTICO DE DISPLAY VIRTUAL PARA COMPATIBILIDAD MÁXIMA
# =============================================================================


class _VirtualDisplayManager:
    """
    Clase interna para gestionar un display virtual (Xvfb).
    Detecta si es necesario y lo inicia/detiene automáticamente.
    """
    _display = None
    _is_active = False

    @classmethod
    def start_if_needed(cls):
        """
        Inicia un display virtual si estamos en un entorno sin GUI (como Colab).
        """
        # Si ya está activo, no hacemos nada.
        if cls._is_active:
            return

        # La heurística principal: si la variable de entorno DISPLAY no existe en Linux,
        # es casi seguro que estamos en un entorno headless.
        is_linux = platform.system() == "Linux"
        is_headless = "DISPLAY" not in os.environ

        if is_linux and is_headless:
            print("INFO: Entorno headless detectado. Iniciando display virtual (Xvfb)...")
            try:
                from pyvirtualdisplay import Display
                # Creamos una instancia de display virtual en memoria
                cls._display = Display(visible=0, size=(1024, 768))
                cls._display.start()
                cls._is_active = True
                print("INFO: Display virtual iniciado con éxito.")
            except ImportError:
                print(
                    "ADVERTENCIA: 'pyvirtualdisplay' no está instalado. El renderizado puede fallar.")
                print("             Instálalo con 'pip install pyvirtualdisplay'")
            except Exception as e:
                print(f"ERROR: No se pudo iniciar el display virtual: {e}")
                print(
                    "       Asegúrate de que 'xvfb' está instalado en tu sistema (sudo apt-get install xvfb)")

    @classmethod
    def stop(cls):
        """
        Detiene el display virtual si se había iniciado.
        """
        # Esta función se mantiene por si se necesita en el futuro, pero no se usa
        # activamente para evitar problemas de reinicio de sesión en Colab.
        if cls._is_active and cls._display:
            try:
                cls._display.stop()
            except Exception as e:
                print(
                    f"ADVERTENCIA: No se pudo detener el display virtual limpiamente: {e}")
            finally:
                cls._is_active = False
                cls._display = None

# =============================================================================


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

        # =================================================================
        # INICIAMOS EL GESTOR DE DISPLAY VIRTUAL SI ES NECESARIO
        # Esto se ejecuta solo una vez cuando se crea el primer entorno.
        # =================================================================
        if render_mode in ["human", "rgb_array"]:
            _VirtualDisplayManager.start_if_needed()

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
        if self.render_mode == "human":
            self.render()
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
        """
        Versión renovada del método render, compatible con el renderer unificado
        y la lógica del AnalyticsView.
        """
        if self.render_mode is None:
            return

        self._lazy_init_renderer()
        if self._renderer is None:
            return None

        # _render_frame llama a draw y devuelve el resultado final (array o dimensiones)
        render_result = self._render_frame()

        if self.render_mode == "human":
            if self.window:
                self.window.flip()
            return None

        elif self.render_mode == "rgb_array":
            # Devolvemos directamente el resultado, que ya es el array de la imagen.
            return render_result

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
        # Detenemos el renderer
        if self.window:
            try:
                self.window.close()
            except Exception:
                pass
        self.window = None
        self._renderer = None

        # =================================================================
        # CORRECCIÓN: NO DETENEMOS EL DISPLAY VIRTUAL.
        # Lo dejamos activo durante toda la sesión para evitar fugas de
        # recursos y fallos al reiniciar la celda. El proceso se
        # limpiará automáticamente cuando la sesión de Colab termine.
        # =================================================================
        # _VirtualDisplayManager.stop()

    # API Extendida ---
    def set_simulation_speed(self, speed: float):
        self._simulation_speed = speed

    def set_respawn_unseeded(self, flag: bool = True):
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, **kwargs):
        self.q_table_to_render = kwargs.get('q_table')

    def set_state_store(self, state_store):
        self._state_store = state_store

    def trigger_end_scene(self, terminated: bool, truncated: bool):
        if self.render_mode in ["human", "rgb_array"]:
            self._end_scene_state = "REQUESTED"

    def is_end_scene_animation_finished(self) -> bool:
        if self._renderer is None:
            return True
        return self._end_scene_state == "IDLE"

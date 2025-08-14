# mlvlab/envs/ant/env.py

"""
Para este entorno, se ha implementado un sistema de respawn aleatorio para el entrenamiento y determinista para la evaluación manejado por la variable _respawn_unseeded (misma seed, mismo comportamiento en todos los episodios, pocición de la hormiga, las rocas, etc): env.set_respawn_unseeded(True) <- False por defecto.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# Importamos las clases modularizadas
try:
    from .game import AntGame
    from .renderer import ArcadeRenderer
except ImportError:
    # Fallback para ejecución directa o si la estructura del paquete falla
    from game import AntGame
    from renderer import ArcadeRenderer

# =============================================================================
# GYMNASIUM ENVIRONMENT WRAPPER
# =============================================================================


class LostAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 reward_goal=100,
                 reward_obstacle=-100,
                 reward_move=-1,
                 use_numba_core: bool = False,
                 ):
        super().__init__()

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        self.REWARD_GOAL = reward_goal
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move
        self.use_numba_core = use_numba_core

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
            use_numba_core=use_numba_core,
        )
        self._renderer: ArcadeRenderer | None = None

        # Estado interno para manejar la transición de éxito
        self._logical_terminated = False

        # Referencias externas para compatibilidad
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

        # Configuración de renderizado
        self.render_mode = render_mode
        self.window = None
        self.q_table_to_render = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad para respawn
        # Se establece a False por defecto para asegurar comportamiento determinista (Gymnasium standard).
        # train.py lo cambiará a True explícitamente para entrenamiento.
        self._respawn_unseeded: bool = False
        try:
            self._respawn_rng = np.random.default_rng()
        except Exception:
            self._respawn_rng = np.random.RandomState()

    # --- Métodos Auxiliares Privados ---

    def _sync_game_state(self):
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

    def _get_respawn_rng(self):
        # Decide si usar la RNG global (no seeded) o la RNG del entorno (seeded)
        if getattr(self, "_respawn_unseeded", False) and self._respawn_rng is not None:
            # Usa RNG no seedada (Aleatorio - Entrenamiento)
            return self._respawn_rng
        # Usa RNG seedada por Gymnasium (Determinista - Evaluación)
        return self.np_random

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {"goal_pos": np.array(self.goal_pos, dtype=np.int32)}

    # Métodos de compatibilidad
    def _generate_scenario(self):
        self._game.generate_scenario(self.np_random)
        self._sync_game_state()

    def _place_ant(self):
        rng = self._get_respawn_rng()
        self._game.place_ant(rng)
        self._sync_game_state()

    # --- API Pública de Gymnasium ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._logical_terminated = False

        if self._renderer:
            self._renderer.reset()

        # Lógica de generación de escenario (Seeded vs Reuse)
        scenario_not_ready = (not np.any(self._game.goal_pos)) or (
            not self._game.obstacles)

        # Importante: Solo regeneramos el mapa si se pasa una semilla explícitamente
        # o si es la primera vez. Esto permite mantener el mismo mapa.
        if seed is not None or scenario_not_ready:
            # Usamos la RNG seedada por Gymnasium (self.np_random) para el mapa
            self._game.generate_scenario(self.np_random)

        # Recolocar la hormiga siempre al inicio del episodio
        # Usará la RNG correcta (seeded o unseeded) según la configuración (self._respawn_unseeded).
        rng = self._get_respawn_rng()
        self._game.place_ant(rng)

        self._sync_game_state()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Flujo de control robusto para manejar la transición de éxito.

        if self._logical_terminated:
            # Si el juego ya terminó lógicamente, solo actualizamos la transición visual.
            return self._step_transition()

        # Si no ha terminado, ejecutamos la lógica del juego.
        obs, reward, terminated, game_info = self._game.step(action)

        truncated = False
        info = self._get_info()
        info.update(game_info)

        # Disparamos el efecto de partículas como un evento si hay colisión.
        if info.get("collided", False) and self.render_mode in ["human", "rgb_array"]:
            self._lazy_init_renderer()
            if self._renderer:
                # Llamamos directamente a la función que genera las partículas.
                self._renderer._spawn_collision_particles()

        # Añadir sonidos basados en el resultado
        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        self._sync_game_state()

        if terminated:
            self._logical_terminated = True

            # Informar al renderer para que inicie la animación.
            if self.render_mode in ["human", "rgb_array"]:
                self._lazy_init_renderer()
                if self._renderer:
                    self._renderer.start_success_transition()

            # El episodio NO termina para el exterior todavía, empezamos la transición visual.
            return self._step_transition(initial_reward=reward, initial_info=info)

        # Paso normal
        return obs, reward, False, truncated, info

    def _step_transition(self, initial_reward=0, initial_info=None):
        # Maneja los pasos (frames) durante la animación de éxito.
        obs = self._get_obs()
        info = initial_info if initial_info is not None else self._get_info()
        truncated = False
        reward = initial_reward

        # Comprobamos si la animación ha terminado.
        animation_finished = True

        if self.render_mode in ["human", "rgb_array"] and self._renderer:
            if self._renderer.is_in_success_transition():
                animation_finished = False

        if animation_finished:
            # Ahora sí, el episodio termina para el exterior (Gymnasium loop).
            return obs, reward, True, truncated, info
        else:
            # La animación continúa.
            return obs, reward, False, truncated, info

    def _lazy_init_renderer(self):
        if self._renderer is None:
            try:
                import arcade
            except ImportError:
                if self.render_mode in ["human", "rgb_array"]:
                    raise ImportError(
                        "Se requiere 'arcade' para el renderizado.")
                return None
            self._renderer = ArcadeRenderer()

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
        result = self._renderer.draw(
            self._game, self.q_table_to_render, self.render_mode)

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
        # Capturar el framebuffer a un array numpy RGB

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
            image = arcade_module.get_image(0, 0, width, height)
        except TypeError:
            # Fallback para firmas antiguas de get_image()
            try:
                image = arcade_module.get_image()
            except Exception as e:
                print(f"Error al capturar imagen rgb_array: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)

        # Conversión a Numpy RGB para cumplir con el estándar de Gymnasium (3 canales)
        try:
            image = image.convert("RGB")
            frame = np.asarray(image)
        except Exception as e:
            print(f"Error al convertir imagen a array numpy: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        return frame

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

    # --- API Extendida (Para entrenamiento y visualización) ---

    def set_respawn_unseeded(self, flag: bool = True):
        # Permite activar/desactivar el respawn aleatorio.
        # Usado por train.py (True) y eval.py (False).
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, q_table):
        self.q_table_to_render = q_table

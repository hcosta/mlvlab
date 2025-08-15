# mlvlab/envs/ant/env.py

"""
Para este entorno, se ha implementado un sistema de respawn aleatorio para el entrenamiento y determinista para la evaluación manejado por la variable _respawn_unseeded (misma seed, mismo comportamiento en todos los episodios, pocición de la hormiga, las rocas, etc): env.set_respawn_unseeded(True) <- False por defecto.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading  # <-- AÑADIR: Importar threading

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

        # --- NUEVO: Atributo para controlar el modo de depuración ---
        self.debug_mode = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad para respawn
        # Se establece a False por defecto para asegurar comportamiento determinista (Gymnasium standard).
        # train.py lo cambiará a True explícitamente para entrenamiento.
        self._respawn_unseeded: bool = False
        try:
            self._respawn_rng = np.random.default_rng()
        except Exception:
            self._respawn_rng = np.random.RandomState()

        # --- AÑADIDO: Sistema de animación de fin de escena ---
        self._state_store = None  # Referencia al StateStore para leer speed/turbo
        self._end_scene_state = "IDLE"  # "IDLE", "REQUESTED", "RUNNING"
        self._end_scene_finished_event = threading.Event()
        # --- FIN AÑADIDO ---

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
        self._end_scene_state = "IDLE"  # <-- AÑADIDO: Resetear estado de animación

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
        """
        Ejecuta un paso en el entorno.
        Esta versión simplificada elimina cualquier lógica de animación o retraso
        para garantizar que el bucle de aprendizaje reciba la información
        (recompensa, terminación) de forma inmediata.
        """
        # 1. Ejecutamos un paso en la lógica pura del juego.
        obs, reward, terminated, game_info = self._game.step(action)

        # 2. Preparamos la información adicional estándar de Gymnasium.
        truncated = False
        info = self._get_info()
        info.update(game_info)

        # 3. Gestionamos los efectos visuales y de sonido (no interfieren con la lógica).
        if info.get("collided", False) and self.render_mode in ["human", "rgb_array"]:
            self._lazy_init_renderer()
            if self._renderer:
                self._renderer._spawn_collision_particles()

        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        # 4. Sincronizamos el estado para que el renderer tenga la última posición.
        self._sync_game_state()

        # 5. Devolvemos los resultados directamente.
        #    Esto es crucial para que el agente aprenda del resultado real del paso.
        return obs, reward, terminated, truncated, info

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
        # --- AÑADIDO: Lógica de la máquina de estados para la animación ---
        if self._renderer is not None:
            # 1. Si se ha solicitado una animación, la iniciamos y cambiamos de estado.
            if self._end_scene_state == "REQUESTED":
                self._renderer.start_success_transition()
                self._end_scene_state = "RUNNING"

            # 2. Si la animación se está ejecutando, comprobamos si ha terminado.
            if self._end_scene_state == "RUNNING":
                # La animación termina cuando el flag interno del renderer vuelve a False.
                if not self._renderer.is_in_success_transition():
                    self._end_scene_state = "IDLE"
                    # Avisamos al otro hilo.
                    self._end_scene_finished_event.set()
        # --- FIN AÑADIDO ---

        # --- MODIFICADO: Pasamos el estado de debug_mode al renderer antes de cada dibujado. ---
        if self._renderer is not None:
            self._renderer.debug_mode = self.debug_mode

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

    # --- AÑADIDO: Nuevos métodos para la animación de fin de escena ---

    def set_state_store(self, state_store):
        """Permite al entorno acceder al estado de la UI (para speed/turbo)."""
        self._state_store = state_store

    def end_scene(self):
        """
        Bloquea la ejecución para renderizar la animación de fin de episodio.
        Se salta la animación si el modo turbo está activo o la velocidad es alta.
        """
        # Si no hay renderizado, no hay nada que hacer.
        if self.render_mode not in ["human", "rgb_array"]:
            return

        # Bypass si el usuario está en modo turbo o alta velocidad.
        if self._state_store:
            try:
                speed = self._state_store.get(["sim", "speed_multiplier"], 1)
                turbo = self._state_store.get(["sim", "turbo_mode"], False)
                if speed > 1 or turbo:
                    return  # Salimos sin bloquear ni animar
            except Exception:
                pass  # Si falla la lectura, continuamos con la animación

        # 1. Pone el flag para que el hilo de renderizado lo vea.
        self._end_scene_state = "REQUESTED"
        # 2. Limpiamos el evento por si se usó antes.
        self._end_scene_finished_event.clear()
        # 3. Esperamos aquí (bloquea) hasta que el hilo de render llame a .set().
        #    Añadimos un timeout para evitar un bloqueo infinito si algo va mal.
        self._end_scene_finished_event.wait(timeout=1)
    # --- FIN AÑADIDO ---

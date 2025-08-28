# mlvlab/envs/ant_maze_v1/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading
import os
import platform

# Importamos las clases modularizadas
try:
    from .game import MazeGame
    from .renderer import MazeRenderer
except ImportError:
    # Fallback para ejecución directa
    from game import MazeGame
    from renderer import MazeRenderer

# =============================================================================
# GESTOR AUTOMÁTICO DE DISPLAY VIRTUAL (Se mantiene igual que en AntScout)
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
            print("INFO: Entorno headless detectado. Iniciando display virtual (Xvfb)...")
            try:
                from pyvirtualdisplay import Display
                cls._display = Display(visible=0, size=(1024, 768))
                cls._display.start()
                cls._is_active = True
                print("INFO: Display virtual iniciado con éxito.")
            except ImportError:
                print("ADVERTENCIA: 'pyvirtualdisplay' no está instalado.")
            except Exception as e:
                print(f"ERROR: No se pudo iniciar el display virtual: {e}")

    @classmethod
    def stop(cls):
        # No se detiene activamente para evitar problemas en entornos compartidos.
        pass

# =============================================================================


class AntMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 grid_size=15,  # Aumentamos el tamaño por defecto para el laberinto
                 reward_goal=100,
                 reward_wall=-10,  # La penalización por chocar es menor, ya que es esperable
                 reward_move=-1,
                 # Flag para activar la funcionalidad AntShift (Punto 7)
                 enable_ant_shift=False
                 ):
        super().__init__()

        if render_mode in ["human", "rgb_array"]:
            _VirtualDisplayManager.start_if_needed()

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        self.REWARD_GOAL = reward_goal
        self.REWARD_WALL = reward_wall
        self.REWARD_MOVE = reward_move
        self.enable_ant_shift = enable_ant_shift

        # Espacios de acción y observación
        # Si AntShift está activado (Punto 7), añadimos una acción (4) para el cambio de mapa.
        self.num_movement_actions = 4
        self.action_space = spaces.Discrete(
            self.num_movement_actions + (1 if enable_ant_shift else 0))
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        # Lógica del juego (Delegada a MazeGame)
        self._game = MazeGame(
            grid_size=grid_size,
            reward_goal=reward_goal,
            reward_wall=reward_wall,
            reward_move=reward_move,
        )
        self._renderer: MazeRenderer | None = None

        # Configuración de renderizado
        self.render_mode = render_mode
        self.window = None
        self.q_table_to_render = None
        self.debug_mode = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad interna (necesaria para AntShift aleatorio)
        try:
            self._internal_rng = np.random.default_rng()
        except Exception:
            self._internal_rng = np.random.RandomState()

        # Sistema de animación
        self._state_store = None
        self._end_scene_state = "IDLE"
        self._end_scene_finished_event = threading.Event()
        self._simulation_speed = 1.0

        # CAMBIO: Añadimos seguimiento de pasos (Importado de AntLost)
        self._elapsed_steps = 0
        self._max_episode_steps = None

        # Estado para AntShift (Punto 7). Controla si el agente debe seguir aprendiendo.
        self.is_q_table_locked = False
        self._sync_game_state()

    def _sync_game_state(self):
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.walls = self._game.walls
        self.obstacles = self._game.walls  # Compatibilidad
        self.start_pos = self._game.start_pos

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {
            "goal_pos": np.array(self.goal_pos, dtype=np.int32),
            "start_pos": np.array(self.start_pos, dtype=np.int32),
            "is_q_table_locked": self.is_q_table_locked  # Info sobre el estado de AntShift
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._end_scene_state = "IDLE"
        self._elapsed_steps = 0

        if hasattr(self, 'np_random') and self.np_random is not None:
            self._game._np_random = self.np_random

        # Un mapa nuevo se genera si:
        # 1. Se pasa una nueva seed.
        # 2. O si el escenario no se ha generado nunca (detectado por la falta de una meta).
        scenario_not_ready = not np.any(self._game.goal_pos)
        map_changed = seed is not None or scenario_not_ready

        # LÓGICA DE REINICIO DE Q-TABLE ---
        if map_changed and not self.enable_ant_shift:
            # ... (código para reiniciar la q-table) ...
            self.q_table_to_render = None
            if self._state_store and hasattr(self._state_store, 'q_table'):
                print("Nuevo mapa detectado: Reiniciando Q-Table...")
                num_states = self.GRID_SIZE * self.GRID_SIZE
                num_actions = self.action_space.n
                self._state_store.q_table = np.zeros((num_states, num_actions))
                self.q_table_to_render = self._state_store.q_table

        # LÓGICA DE GENERACIÓN Y RESETEO DEL JUEGO ---
        if map_changed:
            self._game.generate_scenario(self.np_random)
            if options is None or not options.get("keep_q_table_locked", False):
                self.is_q_table_locked = False
            self._game.reset(self.np_random, hard=True)
        else:
            self._game.reset(self.np_random, hard=False)

        # ... (el resto de la función sigue igual) ...
        self._sync_game_state()
        if self.render_mode and self._renderer is None:
            self._lazy_init_renderer()
        if self._renderer:
            self._renderer.reset(full_reset=map_changed)
        if self.render_mode == "human":
            self.render()
        info = self._get_info()
        info['map_changed'] = map_changed
        return self._get_obs(), info

    def step(self, action):

        # CAMBIO: Inicializamos _max_episode_steps si es necesario (replicando AntLost).
        if self._max_episode_steps is None:
            # Intentamos obtener el límite del spec del entorno (ej. si está envuelto en TimeLimit).
            try:
                # Buscamos en el spec actual o en el del entorno sin wrappers.
                if self.spec and self.spec.max_episode_steps:
                    self._max_episode_steps = self.spec.max_episode_steps
                elif hasattr(self.unwrapped, 'spec') and self.unwrapped.spec and self.unwrapped.spec.max_episode_steps:
                    self._max_episode_steps = self.unwrapped.spec.max_episode_steps
                else:
                    # Fallback si no se encuentra límite
                    self._max_episode_steps = float('inf')
            except Exception:
                self._max_episode_steps = float('inf')

        # Manejo de la acción especial para AntShift (Punto 7)
        if self.enable_ant_shift and action == self.num_movement_actions:
            self._handle_ant_shift_action()
            # La acción de shift no cuenta como un paso de movimiento.
            obs = self._get_obs()
            reward = 0
            terminated = False
            truncated = False
            info = self._get_info()
            info.update({"collided": False, "terminated": False,
                        "ant_shift_triggered": True})

            if self.render_mode == "human":
                self.render()
            return obs, reward, terminated, truncated, info

        # Paso normal de la simulación
        obs, reward, terminated, game_info = self._game.step(action)

        # CAMBIO: Incrementamos el contador de pasos.
        self._elapsed_steps += 1

        truncated = False
        info = self._get_info()
        info.update(game_info)

        if info.get("collided", False) and self.render_mode in ["human", "rgb_array"]:
            self._lazy_init_renderer()
            if self._renderer:
                self._renderer._spawn_collision_particles()

        # Sonidos
        # CAMBIO: Añadimos la lógica para el sonido de fallo al alcanzar el límite.
        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif self._elapsed_steps >= self._max_episode_steps:
            # Se alcanzó el límite de pasos (Truncation inminente)
            info['play_sound'] = {'filename': 'fail.wav', 'volume': 10}
        elif info.get("collided", False):
            # Usamos un sonido para chocar contra muros
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 7}

        self._sync_game_state()
        if self.render_mode == "human":
            self.render()

        # Nota: El wrapper TimeLimit de Gymnasium se encargará de poner truncated=True si se cumple la condición.
        return obs, reward, terminated, truncated, info

    def _handle_ant_shift_action(self):
        """
        Implementación de la lógica de AntShift (Punto 7).
        Genera un nuevo mapa en vivo y bloquea la QTable.
        """
        print("AntShift Activado: Generando nuevo laberinto y bloqueando Q-Table.")

        # 1. Bloquear la Q-Table
        self.is_q_table_locked = True

        # 2. Generar un nuevo mapa. Usamos la RNG interna (no la de Gymnasium)
        # para asegurar que el nuevo mapa es diferente e impredecible respecto al entrenamiento.
        self._game.generate_scenario(self._internal_rng)

        # 3. Posicionar la hormiga en el punto de salida del nuevo mapa
        self._game.place_ant()

        # 4. Reiniciar el estado del renderer para el nuevo mapa
        if self._renderer:
            self._renderer.reset(full_reset=True)

        self._sync_game_state()

    # Métodos de Renderizado ---

    def _lazy_init_renderer(self):
        if self._renderer is None:
            try:
                import arcade
            except ImportError:
                if self.render_mode in ["human", "rgb_array"]:
                    raise ImportError(
                        "Se requiere 'arcade' para el renderizado.")
                return
            self._renderer = MazeRenderer()

    def render(self):
        if self.render_mode is None:
            return

        self._lazy_init_renderer()
        if self._renderer is None:
            return None

        render_result = self._render_frame()

        if self.render_mode == "human":
            if self.window:
                self.window.dispatch_events()  # Procesar eventos para interactividad (ej. tecla L)
                try:
                    self.window.flip()
                except Exception:
                    pass
            return None

        elif self.render_mode == "rgb_array":
            return render_result

    def _render_frame(self):
        if self._renderer is not None:
            # CAMBIO: Máquina de estados expandida para manejar Éxito y Muerte.

            # 1. Secuencia de Éxito (Success) ---
            # (Lógica original de AntMaze)
            if self._end_scene_state == "SUCCESS_REQUESTED":
                if hasattr(self._renderer, 'start_success_transition'):
                    self._renderer.start_success_transition()
                self._end_scene_state = "SUCCESS_RUNNING"

            elif self._end_scene_state == "SUCCESS_RUNNING":
                is_running = False
                if hasattr(self._renderer, 'is_in_success_transition'):
                    is_running = self._renderer.is_in_success_transition()

                if not is_running:
                    self._end_scene_state = "IDLE"
                    self._end_scene_finished_event.set()

            # 2. Secuencia de Muerte (Death) ---
            # (Lógica importada de AntLost: REQUESTED -> DELAY_FRAME -> RUNNING)

            elif self._end_scene_state == "DEATH_REQUESTED":
                # Pasamos a un estado de espera (DELAY_FRAME) para asegurar que
                # el renderer procese el último movimiento antes de iniciar la muerte.
                self._end_scene_state = "DEATH_DELAY_FRAME"

            elif self._end_scene_state == "DEATH_DELAY_FRAME":
                # Solicitamos al renderer iniciar la transición de muerte.
                # Esto requiere que MazeRenderer tenga el método start_death_transition().
                if hasattr(self._renderer, 'start_death_transition'):
                    self._renderer.start_death_transition()
                self._end_scene_state = "DEATH_RUNNING"

            elif self._end_scene_state == "DEATH_RUNNING":
                # Esperamos a que el renderer nos indique que la animación ha terminado.
                # Esto requiere que MazeRenderer tenga el método is_in_death_transition().
                is_running = False
                if hasattr(self._renderer, 'is_in_death_transition'):
                    is_running = self._renderer.is_in_death_transition()

                if not is_running:
                    self._end_scene_state = "IDLE"
                    # Notificamos que la animación ha terminado.
                    self._end_scene_finished_event.set()

            # (Se mantiene igual)
            self._renderer.debug_mode = self.debug_mode

        # (Se mantiene igual, pero aseguramos que el renderer exista antes de usarlo)
        result = None
        if self._renderer:
            result = self._renderer.draw(
                self._game, self.q_table_to_render, self.render_mode,
                simulation_speed=self._simulation_speed
            )
            self.window = self._renderer.window

        return result

    def close(self):
        if self.window:
            try:
                self.window.close()
            except Exception:
                pass
        self.window = None
        self._renderer = None

    # API Extendida (Se mantiene igual) ---
    def set_simulation_speed(self, speed: float):
        self._simulation_speed = speed

    def set_respawn_unseeded(self, flag: bool = True):
        # Esta funcionalidad es menos relevante en AntMaze ya que el respawn es determinista.
        pass

    def set_render_data(self, **kwargs):
        self.q_table_to_render = kwargs.get('q_table')

    def set_state_store(self, state_store):
        self._state_store = state_store

    def trigger_end_scene(self, terminated: bool, truncated: bool):
        # CAMBIO: Activamos la animación correspondiente al motivo de finalización.
        if self.render_mode in ["human", "rgb_array"]:

            # Evitar solicitudes si ya estamos en una animación
            if self._end_scene_state != "IDLE":
                return

            if terminated:
                # Finalización exitosa -> Animación de éxito
                self._end_scene_state = "SUCCESS_REQUESTED"
            elif truncated:
                # Truncamiento (Timeout) -> Animación de muerte
                self._end_scene_state = "DEATH_REQUESTED"

                # Opcional: Si tu MazeRenderer utiliza el flag 'is_dead' (como AntLostRenderer),
                # debes añadir este atributo a MazeGame y activarlo aquí.
                if hasattr(self._game, 'is_dead'):
                    self._game.is_dead = True

    def is_end_scene_animation_finished(self) -> bool:
        if self._renderer is None:
            return True
        return self._end_scene_state == "IDLE"

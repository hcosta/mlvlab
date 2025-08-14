# mlvlab/envs/ant/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# Importar los componentes separados
try:
    # Intento de importación relativa (paquete)
    from .game import AntGame
    from .renderer import ArcadeRenderer
except ImportError:
    # Fallback para importación local (mismo directorio)
    try:
        from game import AntGame
        from renderer import ArcadeRenderer
    except ImportError:
        print("Error Crítico: No se pudo importar AntGame o ArcadeRenderer.")
        AntGame = None
        ArcadeRenderer = None


# =============================================================================
# GYMNASIUM ENVIRONMENT WRAPPER
# =============================================================================

class LostAntEnv(gym.Env):
    # Mantenemos 60 FPS para animaciones fluidas.
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

        if AntGame is None:
            raise RuntimeError(
                "LostAntEnv no puede inicializarse porque faltan dependencias (AntGame/ArcadeRenderer).")

        # Parámetros del entorno
        self.GRID_SIZE = grid_size
        # ... (Resto de parámetros) ...

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

        # Referencias externas para compatibilidad (¡Crucial mantenerlas sincronizadas!)
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

        # Configuración de renderizado
        self.render_mode = render_mode
        self.window = None
        self.q_table_to_render = None

        # Feature 4: Estado para gestionar el retraso de la animación de éxito
        self._animation_pending = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Gestión de aleatoriedad (Simplificado, confiamos en la RNG de gym.Env)
        self._respawn_unseeded = False  # Manejo de comportamiento obsoleto

    # --- Métodos Auxiliares Privados ---

    def _sync_game_state(self):
        # Asegura que las propiedades públicas del Env reflejan el estado interno de AntGame
        self.ant_pos = self._game.ant_pos
        self.goal_pos = self._game.goal_pos
        self.obstacles = self._game.obstacles
        self.obstacles_grid = self._game.obstacles_grid

    def _get_obs(self):
        return self._game.get_obs()

    def _get_info(self):
        return {"goal_pos": np.array(self.goal_pos, dtype=np.int32)}

    # --- API Pública de Gymnasium ---

    def reset(self, seed=None, options=None):
        # Inicializa la RNG si se proporciona una semilla
        super().reset(seed=seed)

        self._animation_pending = False

        # Resetear el juego usando la RNG inicializada por Gym
        # Esto regenera el escenario y la posición de la hormiga.
        self._game.reset(self.np_random)

        # Sincronizar estado
        self._sync_game_state()

        # Resetear el estado del renderer (se hace en el primer render posterior)
        if self._renderer:
            self._renderer.initialized = False

        if self.render_mode:
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):

        # Feature 4: Prevenir lógica de juego si la animación está pendiente (seguridad)
        if self._animation_pending:
            # Devolvemos terminated=True porque el episodio ya terminó lógicamente.
            return self._get_obs(), 0, True, False, self._get_info()

        # Ejecutar la lógica del paso usando AntGame.
        obs, reward, terminated, game_info = self._game.step(action)

        truncated = False
        info = self._get_info()
        info.update(game_info)

        # Añadir sonidos
        if terminated:
            info['play_sound'] = {'filename': 'success.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        # Sincronizar estado
        self._sync_game_state()

        # Feature 4: Manejar la transición de éxito si el renderizado está habilitado
        if terminated and self.render_mode in ["human", "rgb_array"]:
            self._animation_pending = True
            # Ejecutamos el bucle de animación DENTRO de step().
            # Esto bloquea la ejecución hasta que la animación termine, asegurando que se visualice.
            self._run_success_animation()

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, info

    def _run_success_animation(self):
        # Seguir renderizando hasta que el renderer informe que la animación ha terminado
        animation_finished = False
        while not animation_finished:
            # Llamamos a render, que internamente llama a _render_frame y _handle_human_render
            result = self.render()

            # En modo human, render() no devuelve el estado de animación, debemos consultarlo.
            if self.render_mode == "human":
                if self._renderer:
                    is_terminated = getattr(self._game, 'is_terminated', False)
                    animation_finished = is_terminated and self._renderer.success_animation_time >= self._renderer.SUCCESS_ANIMATION_DURATION
                else:
                    break  # Seguridad si el renderer falla

            # En modo rgb_array, render() devuelve el frame. Debemos interpretar el resultado de _render_frame.
            # Esto es complicado porque render() no devuelve animation_finished directamente.
            # Para simplificar y asegurar compatibilidad, confiamos en el estado del renderer.
            elif self.render_mode == "rgb_array":
                if self._renderer:
                    is_terminated = getattr(self._game, 'is_terminated', False)
                    animation_finished = is_terminated and self._renderer.success_animation_time >= self._renderer.SUCCESS_ANIMATION_DURATION
                else:
                    break

            # Comprobación de seguridad si la ventana se cierra
            if self.window is None:
                break

        self._animation_pending = False

    def render(self):
        # Chequeo de dependencias
        if self.render_mode in ["human", "rgb_array"]:
            if ArcadeRenderer is None:
                raise ImportError(
                    "Se requiere 'ArcadeRenderer', pero no pudo ser importado.")
            try:
                import arcade
            except ImportError:
                raise ImportError(
                    "Se requiere la librería 'arcade'. Instálalo con 'pip install arcade'.")

        # Dibujar el frame
        result = self._render_frame()
        if result is None or result[0] is None:
            return None

        dimensions, animation_finished = result
        width, height = dimensions

        if self.render_mode == "human":
            self._handle_human_render()
            return None
        elif self.render_mode == "rgb_array":
            return self._capture_rgb_array(width, height)

    def _render_frame(self):
        # Delegar el dibujo al ArcadeRenderer
        if self._renderer is None:
            if ArcadeRenderer is None:
                return None
            self._renderer = ArcadeRenderer()

        # Devuelve (dimensions), animation_finished
        result = self._renderer.draw(
            self._game, self.q_table_to_render, self.render_mode)

        # Mantener referencia a la ventana
        if self._renderer is not None:
            self.window = self._renderer.window

        return result

    def _handle_human_render(self):
        # Mostrar en pantalla y regular FPS
        if self.window is not None:
            try:
                self.window.dispatch_events()  # Procesar eventos OS (e.g., cerrar ventana)
                self.window.flip()             # Actualizar pantalla
            except Exception as e:
                # Si ocurre un error (e.g., el usuario cerró la ventana), cerramos el entorno.
                self.close()

        # Regulación de FPS (Sleep). Solo si la ventana sigue abierta.
        if self.window:
            target_sleep = 1.0 / float(self.metadata.get("render_fps", 60))
            time.sleep(target_sleep)

    def _capture_rgb_array(self, width, height):
        # Capturar el framebuffer a un array numpy RGB
        arcade_module = None
        if self._renderer and self._renderer.arcade:
            arcade_module = self._renderer.arcade
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)

        try:
            # Asegurarse de que estamos en el contexto de la ventana correcta
            if self.window:
                self.window.switch_to()
            image = arcade_module.get_image(0, 0, width, height)
        except Exception as e:
            print(f"Error during rgb_array capture: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Conversión a Numpy RGB
        try:
            image = image.convert("RGB")
            frame = np.asarray(image)
        except Exception as e:
            print(f"Error converting image to numpy array: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

        return frame

    def close(self):
        # Cleanup resources
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
        self.render_mode = None  # Desactivar renderizado futuro

    # --- API Extendida (Para entrenamiento y visualización) ---

    def set_respawn_unseeded(self, flag: bool = True):
        """DEPRECATED: Use the 'seed' parameter in reset() for determinism."""
        print("Warning: set_respawn_unseeded is deprecated and behavior might differ.")
        self._respawn_unseeded = bool(flag)

    def set_render_data(self, q_table):
        self.q_table_to_render = q_table

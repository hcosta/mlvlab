# mlvlab/envs/ant_lost_v1/env.py
from arcade.key import P
import gymnasium as gym
import numpy as np
import pprint

# Importamos la clase base ScoutAntEnv y el renderer local
from mlvlab.envs.ant_scout_v1.env import ScoutAntEnv
from .renderer import ArcadeRenderer
from .game import AntGame


class LostAntEnv(ScoutAntEnv):
    """
    Entorno de la hormiga perdida. Hereda de ScoutAntEnv y se adapta
    para ser autosuficiente y compatible con el framework de UI.
    """

    def __init__(self, render_mode=None, grid_size=10):
        super().__init__(render_mode=render_mode, grid_size=grid_size,
                         reward_goal=0, reward_obstacle=-1, reward_move=0)

        # Ahora, sobrescribimos el self._game del padre con nuestra propia
        # lógica de juego de AntLost, que sí tiene terminación.
        self._game = AntGame(
            grid_size=grid_size,
            reward_obstacle=-1,
            reward_move=0
        )

        # Ya no intentamos acceder a self.spec aquí.
        self._max_episode_steps = None
        self._elapsed_steps = 0
        self._end_scene_state = "IDLE"
        self._simulation_speed = 1.0

    def reset(self, seed=None, options=None):
        """
        Esta es la versión definitiva y desacoplada.
        No llamamos a super().reset() para evitar la lógica conflictiva
        del padre y garantizar un inicio idéntico en todos los modos.
        """
        # 1. Reseteamos el generador de números aleatorios (importante).
        super().reset(seed=seed)

        # 2. Reseteamos la lógica del juego base.
        self._game.reset(self.np_random)

        # 3. Anulamos el objetivo, como es específico de LostAntEnv.
        self._game.goal_pos = np.array([-1, -1])

        # 4. Sincronizamos los atributos del entorno.
        self._sync_game_state()

        # 5. Reseteamos el renderer si ya existe.
        if self._renderer:
            self._renderer.reset()

        # Nos aseguramos de que el contador de pasos empiece en 0.
        self._elapsed_steps = 0

        # Añadir esta línea para resetear el estado de la animación:
        self._end_scene_state = "IDLE"

        # 7. Devolvemos la observación y la info inicial.
        return self._get_obs(), self._get_info()

    def set_simulation_speed(self, speed: float):
        """Permite al runner controlar la velocidad de simulación."""
        self._simulation_speed = speed

    def is_end_scene_animation_finished(self) -> bool:
        """Comprueba si la animación de la escena final ha terminado (usado por el runner)."""
        if not hasattr(self, '_end_scene_state'):
            return True
        # La animación se considera terminada cuando el estado vuelve a IDLE (gestionado en _render_frame).
        return self._end_scene_state == "IDLE"

    def step(self, action):
        if self._max_episode_steps is None:
            self.spec = gym.spec(self.unwrapped.spec.id)
            self._max_episode_steps = self.spec.max_episode_steps or float(
                'inf')

        # Ejecutamos la acción en la lógica del juego.
        # 'terminated' de aquí siempre será False, como debe ser.
        obs, reward, terminated, game_info = self._game.step(action)

        self._elapsed_steps += 1

        info = self._get_info()
        info.update(game_info)

        # La lógica de sonidos y sincronización no cambia.
        if self._elapsed_steps == self._max_episode_steps:
            info['play_sound'] = {'filename': 'fail.wav', 'volume': 10}
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 7}

        self._sync_game_state()

        # El entorno en sí no decide la terminación ni el truncamiento.
        # Devuelve los valores del juego (terminated=False) y deja que
        # el wrapper TimeLimit de Gymnasium gestione el truncamiento.
        return obs, reward, terminated, False, info

    def _render_frame(self):
        """
        Añadimos la línea que asigna la ventana del renderer
        al atributo 'window' del entorno.
        """
        if self._renderer is None:
            return None

        # Lógica de la animación de muerte (sin cambios)
        if self._end_scene_state == "REQUESTED":
            self._end_scene_state = "DELAY_FRAME"
        elif self._end_scene_state == "DELAY_FRAME":
            self._renderer.start_death_transition()
            self._end_scene_state = "RUNNING"
        elif self._end_scene_state == "RUNNING":
            if not self._renderer.is_in_death_transition():
                self._end_scene_state = "IDLE"

        # Llamamos al método de dibujo.
        result = self._renderer.draw(
            game=self._game,
            render_mode=self.render_mode,
            simulation_speed=self._simulation_speed
        )

        # Asignamos la ventana al entorno para que el player pueda acceder a ella.
        if self._renderer is not None:
            self.window = self._renderer.window

        return result

    def render(self):
        """
        Implementamos nuestro propio método render para manejar
        correctamente la salida de nuestro renderer simplificado, evitando el
        error de desempaquetado del método padre.
        """
        if self.render_mode is None:
            return

        self._lazy_init_renderer()
        if self._renderer is None:
            return None

        render_result = self._render_frame()

        if self.render_mode == "human":
            # Simplemente comprobamos si la ventana existe antes de usarla.
            if self._renderer.window:
                self._renderer.window.flip()
            return None

        elif self.render_mode == "rgb_array":
            return render_result

    def trigger_end_scene(self, terminated: bool, truncated: bool):
        """
        Al detectar truncamiento, iniciamos la secuencia de la escena final
        poniendo el estado en 'REQUESTED'.
        """
        if truncated and self.render_mode:
            # Este es el estado inicial correcto que _render_frame espera.
            self._end_scene_state = "REQUESTED"
            self._game.is_dead = True

    def _lazy_init_renderer(self):
        """
        Nos aseguramos de que se importe y cree el renderer
        correcto para LostAntEnv (el simplificado), y no el de la clase padre.
        """
        if self._renderer is None:
            self._renderer = ArcadeRenderer()

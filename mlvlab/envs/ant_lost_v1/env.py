# mlvlab/envs/ant_lost_v1/env.py
import gymnasium as gym
import numpy as np
import pprint

# Importamos la clase base ScoutAntEnv y el renderer local
from mlvlab.envs.ant_scout_v1.env import ScoutAntEnv
from .renderer import ArcadeRenderer


class LostAntEnv(ScoutAntEnv):
    """
    Entorno de la hormiga perdida. Hereda de ScoutAntEnv y se adapta
    para ser autosuficiente y compatible con el framework de UI.
    """

    def __init__(self, render_mode=None, grid_size=10):
        super().__init__(render_mode=render_mode, grid_size=grid_size,
                         reward_goal=0, reward_obstacle=-1, reward_move=0)
        # Ya no intentamos acceder a self.spec aquí.
        self._max_episode_steps = None
        self._elapsed_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._game.goal_pos = np.array([-1, -1])
        self._sync_game_state()

        # Al resetear, los pasos vuelven a ser 0.
        self._elapsed_steps = 0

        return obs, self._get_info()

    def step(self, action):
        # Si es la primera vez que se llama a step, obtenemos el valor.
        if self._max_episode_steps is None:
            self.spec = gym.spec(self.unwrapped.spec.id)
            self._max_episode_steps = self.spec.max_episode_steps or float(
                'inf')

        # Ejecutamos la acción en la lógica del juego.
        obs, reward, terminated, game_info = self._game.step(action)

        # Preparamos el diccionario de información que devolveremos.
        info = self._get_info()
        info.update(game_info)

        # Lógica de partículas si hay colisión.
        if info.get("collided", False) and self.render_mode:
            self._lazy_init_renderer()
            if self._renderer:
                if hasattr(self._renderer, 'spawn_collision_particles'):
                    self._renderer.spawn_collision_particles()
                else:
                    self._renderer._spawn_collision_particles()

        # Comprobamos si es el último paso para añadir el sonido de muerte.
        info['truncated'] = False
        if self._elapsed_steps == self._max_episode_steps - 1:
            info['truncated'] = True
            info['play_sound'] = {'filename': 'fail.wav', 'volume': 8}
        # Si no es el último, comprobamos si ha habido una colisión.
        elif info.get("collided", False):
            info['play_sound'] = {'filename': 'bump.wav', 'volume': 5}

        # Sincronizamos el estado y devolvemos los resultados.
        self._sync_game_state()

        # Incrementamos nuestro contador de pasos interno.
        self._elapsed_steps += 1

        return obs, reward, terminated, info['truncated'], info

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

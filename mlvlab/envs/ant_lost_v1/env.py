# mlvlab/envs/ant_lost_v1/env.py
import gymnasium as gym
import numpy as np
import time

# Importamos la clase base ScoutAntEnv y el renderer local
try:
    from ..ant_scout_v1.env import ScoutAntEnv
    from .renderer import ArcadeRenderer
except (ImportError, ValueError):
    from mlvlab.envs.ant_scout_v1.env import ScoutAntEnv
    from renderer import ArcadeRenderer


class LostAntEnv(ScoutAntEnv):
    """
    Entorno de la hormiga perdida. Hereda de ScoutAntEnv y se adapta
    para ser autosuficiente y compatible con el framework de UI.
    """

    def __init__(self, render_mode=None, grid_size=10):
        """
        Llamamos al init del padre con recompensas neutrales
        """
        super().__init__(render_mode=render_mode, grid_size=grid_size,
                         reward_goal=0, reward_obstacle=-1, reward_move=0)

        # El entorno obtiene su propio límite de pasos al crearse
        try:
            self.spec = gym.spec(self.unwrapped.spec.id)
            self._max_episode_steps = self.spec.max_episode_steps
        except Exception:
            self._max_episode_steps = float('inf')
        self._elapsed_steps = 0

    def reset(self, seed=None, options=None):
        """
        Reseteamos el contador de pasos y anulamos la meta
        """
        obs, info = super().reset(seed=seed, options=options)
        self._game.goal_pos = np.array([-1, -1])
        self._sync_game_state()
        self._elapsed_steps = 0
        return obs, self._get_info()

    def step(self, action):
        """
        En cada paso, incrementamos el contador
        """
        self._elapsed_steps += 1
        return super().step(action)

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
            if self._renderer.window and not self._renderer.window.is_closing:
                self._renderer.window.flip()
            return None

        elif self.render_mode == "rgb_array":
            return render_result

    def trigger_end_scene(self, terminated: bool, truncated: bool):
        """
        Esta versión acepta los argumentos que le pasa la clase Logic.
        """
        # Inicia la animación de muerte solo si el episodio terminó por truncamiento.
        if truncated and self.render_mode:
            self._end_scene_state = "REQUESTED"
            self._game.is_dead = True

    def _lazy_init_renderer(self):
        """
        Nos aseguramos de que se importe y cree el renderer
        correcto para LostAntEnv (el simplificado), y no el de la clase padre.
        """
        if self._renderer is None:
            self._renderer = ArcadeRenderer()

    def _render_frame(self):
        """
        Implementa la lógica de la animación de MUERTE.
        """
        if self._renderer is not None:
            if self._end_scene_state == "REQUESTED":
                self._renderer.start_death_transition()
                self._end_scene_state = "RUNNING"

            if self._end_scene_state == "RUNNING":
                if not self._renderer.is_in_death_transition():
                    self._end_scene_state = "IDLE"

        return self._renderer.draw(
            game=self._game,
            render_mode=self.render_mode,
            simulation_speed=self._simulation_speed
        )

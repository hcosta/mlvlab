# mlvlab/core/logic.py

class InteractiveLogic:
    """
    Clase base para definir una lógica de episodio compatible con la vista interactiva.
    El alumno debe heredar de esta clase e implementar los métodos abstractos.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.total_reward = 0.0
        self.state = None

    def on_episode_start(self):
        """
        Llamado al inicio de cada episodio. Usa la implementación del alumno
        de _obs_to_state para devolver el estado inicial.
        """
        obs, info = self.env.reset()
        # Esta línea ahora llamará al método del alumno
        self.state = self._obs_to_state(obs)
        self.total_reward = 0.0
        return self.state

    def step(self, state):
        """
        Ejecuta un único paso (acción, observación, aprendizaje).
        Debe devolver una tupla: (next_state, reward, done)
        """
        raise NotImplementedError(
            "El alumno debe implementar el método 'step'.")

    def on_episode_end(self):
        """
        Llamado al final de cada episodio, después de que step() devuelva done=True.
        """
        if hasattr(self.env.unwrapped, 'trigger_end_scene'):
            self.env.unwrapped.trigger_end_scene()

    def _obs_to_state(self, obs):
        """
        Convierte una observación del entorno a un estado discreto.
        Este método DEBE ser implementado por el alumno en su clase hija.
        """
        # CAMBIO: En lugar de dar la solución, forzamos la implementación.
        raise NotImplementedError(
            "El alumno debe implementar el método '_obs_to_state'.")

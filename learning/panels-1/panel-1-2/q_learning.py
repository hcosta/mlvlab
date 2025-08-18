import gymnasium as gym
import numpy as np
import random


class QLearningAgent():
    """
    Implementación custom de Q-Learning desde cero.
    Totalmente desacoplada de mlvlab.ui pero no de Gymnasium.
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.99):

        self.action_space = action_space

        # Estos atributos deben ser públicos y tener estos nombres para que
        # mlvlab.ui pueda leerlos y modificarlos interactivamente.
        self.learning_rate = float(learning_rate)       # Alpha
        self.discount_factor = float(discount_factor)   # Gamma
        # Epsilon inicial (siempre exploramos al inicio)
        self.epsilon = 1.0
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = 0.01

        # Verificación de espacios (Q-Learning Tabular requiere espacios discretos)
        if not isinstance(observation_space, gym.spaces.Discrete) or \
           not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(
                "Esta implementación requiere espacios de observación y acción discretos.")

        # Inicialización de la Q-Table (El "Cerebro")
        # CRUCIAL: Nombrarla 'q_table' permite que el modo debug integrado la visualice.
        num_states = observation_space.n
        num_actions = action_space.n
        self.q_table = np.zeros((num_states, num_actions))

    def act(self, state: int) -> int:
        """Política Epsilon-Greedy."""
        # self.epsilon es actualizado en tiempo real por la UI si se modifica el slider
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: acción aleatoria
            return self.action_space.sample()
        else:
            # Explotación: la mejor acción conocida para este estado
            return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Actualización basada en la Ecuación de Bellman."""

        # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]

        # 1. Valor actual Q(s,a)
        old_value = self.q_table[state, action]

        # 2. Mejor valor futuro max(Q(s',a'))
        next_max = np.max(self.q_table[next_state])

        # 3. Calcular el nuevo valor usando los hiperparámetros actuales
        # (self.learning_rate y self.discount_factor son actualizados en tiempo real por la UI)
        td_target = reward + self.discount_factor * next_max
        td_error = td_target - old_value
        new_value = old_value + self.learning_rate * td_error

        # 4. Actualizar la tabla
        self.q_table[state, action] = new_value

        # 5. Decaimiento de Epsilon (normalmente al finalizar el episodio)
        if done:
            self.epsilon = max(
                self.min_epsilon, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        """Reinicia el conocimiento (Q-Table). Útil para el botón Reset de la UI."""
        self.q_table.fill(0)
        self.epsilon = 1.0

    def save(self, filepath: str) -> None:
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        self._q_table = np.load(filepath)


class GenericQLearningAgent:
    """
    Implementación pura de Q-Learning. Independiente de Gymnasium y mlvlab.ui.
    Asume estados discretos (0 a N-1) y acciones discretas (0 a M-1).

    ¿Cómo utilizar este agente genérico por ejemplo en ant?

    # 1. Calcular el número de estados. 
    # Esto se basa en cómo AntLogic abstrae el estado (_obs_to_state).
    grid_size = env.unwrapped.GRID_SIZE
    num_states = grid_size * grid_size

    # 2. Obtener el número de acciones y verificar el tipo de espacio.
    # (La verificación que antes estaba en el agente, ahora está aquí).
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        raise TypeError("Se requiere un espacio de acciones discreto para este agente.")

    # 3. Instanciar el agente genérico con las dimensiones calculadas.
    agent = GenericQLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.99
    )

    # 4. Alternativa rápida:
    agent = GenericQLearningAgent(
        num_states= env.unwrapped.GRID_SIZE * env.unwrapped.GRID_SIZE,
        num_actions=env.action_space.n,
        learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.99
    )

    """

    def __init__(self, num_states: int, num_actions: int,
                 learning_rate=0.1, discount_factor=0.9, epsilon_decay=0.99):

        # Guardamos el número de acciones para la exploración
        self.num_actions = num_actions

        # Hiperparámetros (Nombres clave para que mlvlab.ui los detecte)
        self.learning_rate = float(learning_rate)       # Alpha
        self.discount_factor = float(discount_factor)   # Gamma
        self.epsilon = 1.0
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = 0.01

        # Inicialización de la Q-Table (El "Cerebro")
        # CRUCIAL: Nombrarla 'q_table' permite que el modo debug la visualice.
        self.q_table = np.zeros((num_states, num_actions))

    def act(self, state: int) -> int:
        """Política Epsilon-Greedy."""
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: Selección de acción aleatoria desacoplada
            # Usamos random.randint en lugar de action_space.sample()
            return random.randint(0, self.num_actions - 1)
        else:
            # Explotación: la mejor acción conocida
            return int(np.argmax(self.q_table[state]))

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Actualización basada en la Ecuación de Bellman."""
        # (La lógica de aprendizaje no cambia, ya era genérica)

        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        td_target = reward + self.discount_factor * next_max
        td_error = td_target - old_value
        new_value = old_value + self.learning_rate * td_error

        self.q_table[state, action] = new_value

        # Decaimiento de Epsilon
        if done:
            self.epsilon = max(
                self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self) -> dict:
        """
        Empaqueta todos los datos necesarios del agente en un diccionario.
        """
        return {
            'q_table': self.q_table,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'epsilon': self.epsilon,
            }
        }

    def load(self, data: dict) -> None:
        """
        Restaura el estado del agente a partir de un diccionario.
        """
        self.q_table = data.get('q_table')

        # Carga los hiperparámetros
        hparams = data.get('hyperparameters', {})
        self.learning_rate = hparams.get('learning_rate', self.learning_rate)
        self.discount_factor = hparams.get(
            'discount_factor', self.discount_factor)
        self.epsilon_decay = hparams.get('epsilon_decay', self.epsilon_decay)
        self.epsilon = hparams.get('epsilon', self.epsilon)

    def reset(self) -> None:
        self.q_table.fill(0)
        self.epsilon = 1.0

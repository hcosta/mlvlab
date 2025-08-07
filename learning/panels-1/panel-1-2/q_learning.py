# q-learning-visualizer/q_learning.py
# Este archivo contiene únicamente la lógica del agente Q-Learning.
# Está completamente separado del entorno y de la interfaz gráfica.

import numpy as np
import random


def get_state_from_pos(x, y, grid_size):
    """
    Convierte coordenadas (x, y) a un estado único en la tabla Q.
    Es una función de utilidad que pertenece conceptualmente al agente,
    ya que es él quien necesita traducir la observación del entorno a un índice.
    """
    # Programación defensiva: Comprobamos si las coordenadas son válidas.
    # Esto te ayudará a encontrar errores en el futuro.
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        raise ValueError(
            f"Coordenadas ({x}, {y}) fuera de los límites para una rejilla de {grid_size}x{grid_size}")

    # La fórmula correcta
    return y * grid_size + x


class QLearningAgent:
    """
    Un agente que aprende a tomar decisiones usando el algoritmo Q-Learning.
    """

    def __init__(self, num_states, num_actions):
        """
        Inicializa la Q-Table, que almacenará los valores de cada par estado-acción.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state, epsilon):
        """
        Decide qué acción tomar usando una política epsilon-greedy.
        - Con probabilidad epsilon, elige una acción al azar (exploración).
        - Con probabilidad 1-epsilon, elige la mejor acción conocida (explotación).
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state, alpha, gamma):
        """
        Actualiza la Q-Table usando la ecuación de Bellman.
        - alpha: Tasa de aprendizaje (learning rate).
        - gamma: Factor de descuento (discount factor).
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])

        # La fórmula central de Q-Learning.
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.q_table[state, action] = new_value

    def reset(self):
        """Reinicia el conocimiento del agente poniendo la Q-Table a cero."""
        self.q_table.fill(0)

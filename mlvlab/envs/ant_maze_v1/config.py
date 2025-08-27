import arcade

DESCRIPTION = "Dungeons & Pheromones (AntMaze-v1): Guía a la hormiga a través de la mazmorra ajustando sus instintos."

# Mapeo de teclas de Arcade a acciones del entorno (Action Space)
KEY_MAP = {
    arcade.key.UP: 0,
    arcade.key.DOWN: 1,
    arcade.key.LEFT: 2,
    arcade.key.RIGHT: 3,
    # Añadimos WASD como alternativa
    arcade.key.W: 0,
    arcade.key.S: 1,
    arcade.key.A: 2,
    arcade.key.D: 3,
    # Acción para AntShift (Punto 7) - 'L' para Lock/Shift.
    # Esta acción (4) solo estará disponible si el entorno se inicia con enable_ant_shift=True.
    arcade.key.L: 4,
}

# Configuración del agente de referencia para 'train'
# Ajustes optimizados para la resolución de laberintos
BASELINE = {
    "agent": "q_learning",
    "config": {
        "episodes": 1500,
        "alpha": 0.1,
        "gamma": 0.95,  # Gamma alto es crucial para la visión a largo plazo en laberintos
        "epsilon_decay": 0.9999,
        "min_epsilon": 0.05,
    }
}

# Unidad pedagógica a la que pertenece este entorno (para list)
UNIT = "ants"
ALGORITHM = "ql"

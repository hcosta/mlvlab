# mlvlab/envs/ant/config.py
import arcade

DESCRIPTION = "Encuentra la colonia perdida evitando los obstáculos. (GridWorld)"

# Mapeo de teclas de Arcade a acciones del entorno (Action Space)
# Esto permite que el Player genérico funcione con arcade/pyglet.
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
}

# Configuración del agente de referencia para 'mlv train'
BASELINE = {
    "agent": "q_learning",  # Debe coincidir con el nombre del módulo en mlvlab/agents/
    "config": {
        "episodes": 1000,
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon_decay": 0.99995,
        "min_epsilon": 0.01,
        # Nota: grid_size se obtiene automáticamente del entorno.
    }
}

# Unidad pedagógica a la que pertenece este entorno (para mlv list)
UNIT = "ql"

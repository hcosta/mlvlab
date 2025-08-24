# mlvlab/envs/ant_lost_v1/config.py
import arcade

DESCRIPTION = "Zángano Errante (AntLost-v1): Una hormiga sin rumbo fijo. El caos antes de la inteligencia."

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
}

# Configuración del agente de referencia.
# Usamos un agente aleatorio para demostrar el comportamiento errante.
BASELINE = {
    "agent": "random_agent",
    "config": {
        "episodes": 10,  # Solo para demostración
    }
}

# Unidad pedagógica a la que pertenece este entorno
UNIT = "ants"
ALGORITHM = "random"

# mlvlab/envs/ant/config.py
import pygame

DESCRIPTION = "Encuentra la colonia perdida evitando los obstáculos. (GridWorld)"

# Mapeo de teclas de PyGame a acciones del entorno (Action Space)
# Esto permite que el Player genérico funcione.
KEY_MAP = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    # Añadimos WASD como alternativa
    pygame.K_w: 0,
    pygame.K_s: 1,
    pygame.K_a: 2,
    pygame.K_d: 3,
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

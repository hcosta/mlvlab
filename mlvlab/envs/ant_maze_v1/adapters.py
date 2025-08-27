from __future__ import annotations
import numpy as np
from typing import Any


def obs_to_state(obs: Any, env) -> int:
    """Adaptador estándar para entornos grid (x,y)-> idx."""
    try:
        if isinstance(obs, (int,)):
            return int(obs)
        grid = int(getattr(env.unwrapped, 'GRID_SIZE',
                   getattr(env, 'GRID_SIZE', 0)))

        if grid > 0:
            # Añadimos clipping por seguridad, asegurando que las coordenadas están dentro del grid
            x = int(np.clip(obs[0], 0, grid - 1))
            y = int(np.clip(obs[1], 0, grid - 1))
        else:
            x, y = int(obs[0]), int(obs[1])

        return y * grid + x
    except Exception:
        return 0

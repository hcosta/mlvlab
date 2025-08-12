from __future__ import annotations

from typing import Any


def obs_to_state(obs: Any, env) -> int:
    """Adaptador canónico para Ant: (x,y) -> y*GRID_SIZE + x. Si ya es int, retorna tal cual."""
    try:
        if isinstance(obs, (int,)):
            return int(obs)
        grid = int(getattr(env.unwrapped, 'GRID_SIZE',
                   getattr(env, 'GRID_SIZE', 0)))
        return int(obs[1]) * grid + int(obs[0])
    except Exception:
        # Fallback defensivo
        return int(obs) if isinstance(obs, (int,)) else 0

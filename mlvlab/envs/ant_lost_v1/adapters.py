from __future__ import annotations

from typing import Any


def obs_to_state(obs: Any, env) -> int:
    """Adaptador estándar para entornos grid (x,y)-> idx."""
    try:
        if isinstance(obs, (int,)):
            return int(obs)
        grid = int(getattr(env.unwrapped, 'GRID_SIZE',
                   getattr(env, 'GRID_SIZE', 0)))
        x, y = int(obs[0]), int(obs[1])
        return y * grid + x
    except Exception:
        return 0

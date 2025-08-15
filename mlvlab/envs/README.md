## Guía de entornos en MLVLab: cómo crear nuevas simulaciones que funcionen con `mlv`

Este documento explica la arquitectura basada en Env · Algorithm · Agent · Adapter y describe, paso a paso, cómo crear y versionar entornos para que funcionen con la CLI (`mlv play`, `mlv train`, `mlv eval`).

### Conceptos clave

- **Env (Gymnasium Env)**: lógica del entorno (estado, transición, recompensas, render). Vive en `mlvlab/envs/<env_pkg>/`.
- **Algorithm (plugin)**: cómo se entrena/evalúa (QL, SARSA, DQN, …). Vive en `mlvlab/algorithms/<algo_key>/plugin.py` y se selecciona con `ALGORITHM = "ql"` (u otro).
- **Agent**: implementación concreta (p. ej., `QLearningAgent`) que el plugin crea y configura.
- **Adapter (obs_to_state)**: función que traduce observaciones del entorno al formato que el algoritmo necesita (p. ej., índice discreto para Q-Table). Convención: `mlvlab.envs.<env_pkg>.adapters:obs_to_state(obs, env)`.

Flujo de la CLI:
1) `mlv` resuelve el entorno por su ID (`mlv/<env-name>`).
2) Carga `config.py` del entorno y obtiene `ALGORITHM` y sus hiperparámetros.
3) Obtiene el plugin del algoritmo y ejecuta `train` o `eval`.
4) El plugin construye el `Agent` y, si aplica, usa `adapters.obs_to_state`.

### Estructura por entorno

Ejemplo para `ant_v1` (estándar actual):

```
mlvlab/
  envs/
    ant_v1/
      __init__.py
      env.py        # clase Gym (p. ej. LostAntEnv)
      adapters.py       # obs_to_state(obs, env)
      config.py         # DESCRIPTION, ALGORITHM, BASELINE, KEY_MAP (opcional)
      assets/
        blip.wav
        crash.wav
```

- El paquete del entorno usa underscore: `<env>_v<version>` → `ant_v1`, `minotaur_v1`, etc.
- El ID público del entorno usa guion: `mlv/ant-v1`, `mlv/minotaur-v1`, etc.

### Registro del entorno (Gym)

Añade la entrada en `mlvlab/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id="mlv/ant-v1",
    entry_point="mlvlab.envs.ant_v1.env:LostAntEnv",
    max_episode_steps=500,
    kwargs={"grid_size": 15},
)
```

### Configuración del entorno (`config.py`)

```python
# mlvlab/envs/minotaur_v1/config.py
DESCRIPTION = "Minotaur v1 (grid-based)."

# Plugin de algoritmo a usar por la CLI
ALGORITHM = "ql"  # valores posibles: "ql", "sarsa", "dqn", ...

# Hiperparámetros por defecto para el baseline
BASELINE = {
    "config": {
        "episodes": 1000,
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon_decay": 0.9995,
        "min_epsilon": 0.01,
    },
}

# Opcional: mapa de teclas para mlv play (Arcade)
try:
    import arcade
    KEY_MAP = {
        arcade.key.UP: 0,
        arcade.key.DOWN: 1,
        arcade.key.LEFT: 2,
        arcade.key.RIGHT: 3,
        arcade.key.W: 0,
        arcade.key.S: 1,
        arcade.key.A: 2,
        arcade.key.D: 3,
    }
except Exception:
    KEY_MAP = None

# Compatibilidad con código antiguo (si ALGORITHM no está):
UNIT = "ql"
```

### Adaptador de estado (`adapters.py`)

- Convención: `mlvlab.envs.<env_pkg>.adapters:obs_to_state(obs, env)`.
- Para algoritmos tabulares (QL/SARSA) suele mapear `(x, y) -> idx`.

```python
# mlvlab/envs/minotaur_v1/adapters.py
from __future__ import annotations
from typing import Any

def obs_to_state(obs: Any, env) -> int:
    try:
        if isinstance(obs, (int,)):
            return int(obs)
        grid = int(getattr(env.unwrapped, 'GRID_SIZE', getattr(env, 'GRID_SIZE', 0)))
        x, y = int(obs[0]), int(obs[1])
        return y * grid + x
    except Exception:
        return 0
```

### Implementación del entorno (`env.py`)

- Define una clase Gym (`gym.Env`) con `observation_space`, `action_space`, `reset`, `step`, `render`, `close`.
- Si quieres superponer un heatmap de Q-Table durante `render`, implementa en tu entorno:
  - `set_render_data(q_table)` en el `unwrapped`.
- Para sonidos en `mlv play`, desde `step(...)` emite en `info`:

```python
info['play_sound'] = { 'filename': 'blip.wav', 'volume': 10 }
```

Coloca los WAV en `assets/` junto al entorno.

### Plugins de algoritmo

- Registro central en `mlvlab/algorithms/registry.py`.
- Cada plugin expone:
  - `key() -> str` (p. ej., `"ql"`)
  - `build_agent(env, hparams) -> agent`
  - `train(env_id, config, run_dir, seed=None, render=False)`
  - `eval(env_id, run_dir, episodes, seed=None, cleanup=True, video=False)`
- Q-Learning está en `mlvlab/algorithms/ql/plugin.py` y usa `mlvlab/agents/q_learning.py`.

Para añadir un algoritmo (ej. SARSA):

```
mlvlab/
  algorithms/
    sarsa/
      plugin.py       # misma interfaz que QL
      agent.py        # implementación del agente
```

En `plugin.py`:

```python
from mlvlab.algorithms.registry import register_algorithm

class SarsaPlugin:
    def key(self) -> str: return "sarsa"
    # build_agent/train/eval ...

register_algorithm(SarsaPlugin())
```

Cualquier `config.py` con `ALGORITHM = "sarsa"` quedará soportado por `mlv`.

### Flujo de la CLI (`mlv`)

- `mlv list`: lista entornos registrados y su config.
- `mlv play <env_id>`: carga `KEY_MAP` y ejecuta `play_interactive`.
- `mlv train <env_id>`: lee `ALGORITHM`, resuelve el plugin y ejecuta `train`.
- `mlv eval <env_id>`: idem; con `--record` crea `evaluation.mp4`.

### Checklist: entorno nuevo (ej. `minotaur_v1`)

1) Crear paquete:
  - `mlvlab/envs/minotaur_v1/`
  - `env.py` (clase Gym)
  - `adapters.py` (`obs_to_state`)
  - `config.py` (`ALGORITHM`, `BASELINE.config`, `DESCRIPTION`, `KEY_MAP`)
  - `assets/` (opcional)
2) Registrar en `mlvlab/__init__.py`:
  - `id="mlv/minotaur-v1"`, `entry_point="mlvlab.envs.minotaur_v1.env:MinotaurEnv"`
3) Probar:
  - `mlv list` (o `mlv list ql`)
  - `mlv play mlv/minotaur-v1`
  - `mlv train mlv/minotaur-v1`
  - `mlv eval mlv/minotaur-v1 --record`

### Checklist: nueva versión (ej. `ant_v2`)

1) Duplicar paquete:
  - `mlvlab/envs/ant_v2/env.py` con `LostAntEnvV2`
  - `mlvlab/envs/ant_v2/adapters.py` (si cambia)
  - `mlvlab/envs/ant_v2/config.py`
  - `assets/` (si cambia)
2) Registrar `mlv/ant-v2` en `mlvlab/__init__.py`.
3) Probar con `mlv`.

### Buenas prácticas

- Mantén `obs_to_state` determinista y consistente con el espacio de estados.
- Reutiliza utilidades y reexporta clases para evitar duplicación entre versiones.
- Documenta `DESCRIPTION` y mantén `ALGORITHM` actualizado en `config.py`.
- Emite sonidos solo en eventos clave (choque, comida) y colócalos en `assets/`.

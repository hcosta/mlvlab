# Entorno: Ant (LostAntEnv)

Este fichero documenta el entorno `mlv/ant-v1`, también conocido como "El Hormiguero Perdido".

<img src="../../../docs/ant_v1/mode_view.jpg" alt="modo view" width="100%">

## Descripción

En este entorno, un agente (la hormiga) se encuentra en una rejilla de 15x15. El objetivo de la hormiga es encontrar su hormiguero (la meta) en el menor número de pasos posible, mientras evita los obstáculos repartidos por el mapa.

Este es un problema clásico de **navegación en un Grid World**, diseñado para enseñar los fundamentos del aprendizaje por refuerzo tabular.

---

## Ficha Técnica

### Observation Space

El espacio de observación define lo que el agente "ve" en cada paso.
```
Box(0, 14, (2,), int32)
```
* **Significado:** La observación es un vector con 2 números enteros, que representan la posición `[x, y]` de la hormiga en la rejilla.
* **Límites:** Cada coordenada va de 0 a 14, correspondiendo a una rejilla de 15x15.
* **Total de Estados:** $15 \times 15 = 225$ estados únicos posibles.

### Action Space

El espacio de acciones define qué movimientos puede realizar el agente.
```
Discrete(4)
```
* **Significado:** El agente puede elegir una de 4 acciones discretas, representadas por un número entero:
    * `0`: Moverse **Arriba** (disminuye la coordenada `y`)
    * `1`: Moverse **Abajo** (aumenta la coordenada `y`)
    * `2`: Moverse a la **Izquierda** (disminuye la coordenada `x`)
    * `3`: Moverse a la **Derecha** (aumenta la coordenada `x`)

---

## Dinámica del Entorno

### Recompensas (Rewards)

El agente recibe una señal (recompensa) después de cada acción para guiar su aprendizaje:
* **`+100`**: Por llegar al hormiguero (la meta).
* **`-100`**: Por chocar contra un obstáculo.
* **`-1`**: Por cada paso que da. Esto incentiva al agente a encontrar la ruta más corta.

### Fin del Episodio (Termination & Truncation)

Un "episodio" (un intento de encontrar el hormiguero) termina bajo las siguientes condiciones:
* **`terminated = True`**: El agente llega al hormiguero. El episodio termina con éxito.
* **`truncated = True`**: El agente alcanza el límite máximo de pasos (`max_episode_steps=500`) sin encontrar el hormiguero. Esto evita que el agente vague indefinidamente.

**Nota importante:** Si la hormiga choca contra un obstáculo, recibe la penalización de `-100` pero **el episodio no termina**. En su lugar, la hormiga es devuelta a la casilla en la que estaba antes de chocar.

---

## Información Adicional (Diccionario `info`)

Las funciones `reset()` y `step()` devuelven un diccionario `info` que contiene datos útiles para depuración, pero que no deben ser usados directamente para el entrenamiento.
* `info['food_pos']`: Devuelve un array con las coordenadas `[x, y]` del hormiguero.

---

## Estrategia de Entrenamiento Recomendada

### Algoritmo: Q-Learning (tabular)

La combinación de un **espacio de estados discreto y pequeño (225 estados)** y un **espacio de acciones discreto (4 acciones)** hace que este entorno sea un candidato perfecto para algoritmos tabulares como **Q-Learning**.

Este método aprende creando una "tabla de consulta" (la Q-Table) que almacena el valor esperado para cada acción en cada una de las 225 casillas, permitiendo al agente determinar la política óptima.

---

## Ejemplos de Uso con la CLI

```bash
# Jugar interactivamente en el entorno
mlv ant-v1 play

# Entrenar un agente para una semilla específica (p. ej. 42)
mlv ant-v1 train --seed 42

# Entrenar con una semilla aleatoria
mlv ant-v1 train

# Evaluar el último entrenamiento en modo ventana
mlv ant-v1 eval

# Evaluar un entrenamiento de una semilla específica
mlv ant-v1 eval --seed 42

# Evaluar un entrenamiento en modo headless grabando un video de 100 episodios
mlv ant-v1 eval --rec --eps 100

# Lanza una vista interactiva para manipular el entorno usando controles
mlv ant-v1 view

# Ver esta ficha técnica desde la terminal
mlv ant-v1 help
```

---

## Compatibilidad con Notebooks

Puedes experimentar con este entorno directamente desde Jupyter o Google Colab.

Ejemplos rápidos para cuadernos:

```bash
# (Opcional) Instalación si estás en Colab
pip install -U git+https://github.com/hcosta/mlvlab
```

```python
# 1) Crear el entorno y ejecutar un episodio aleatorio
import gymnasium as gym
import mlvlab  # registra los entornos "mlv/..."

env = gym.make("mlv/ant-v1", render_mode="human")
obs, info = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

```python
# 2) Mini-entrenamiento tabular (Q-Table) simplificado
import numpy as np
import gymnasium as gym
import mlvlab

env = gym.make("mlv/ant-v1")
GRID = int(env.unwrapped.GRID_SIZE)
N_S, N_A = GRID * GRID, env.action_space.n
Q = np.zeros((N_S, N_A), dtype=np.float32)

def obs_to_state(obs):
    x, y = int(obs[0]), int(obs[1])
    return y * GRID + x

alpha, gamma, eps = 0.1, 0.9, 1.0
for ep in range(100):
    obs, info = env.reset(seed=123)
    s = obs_to_state(obs)
    done = False
    while not done:
        a = np.random.randint(N_A) if np.random.rand() < eps else int(Q[s].argmax())
        obs2, r, term, trunc, info = env.step(a)
        s2 = obs_to_state(obs2)
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * Q[s2].max())
        s = s2
        done = term or trunc
    eps = max(0.05, eps * 0.995)
env.close()
```

Sugerencia: guarda y carga la Q-Table/pesos para reutilizarlos entre sesiones. También puedes entrenar desde la CLI y evaluar en notebook, o al revés.
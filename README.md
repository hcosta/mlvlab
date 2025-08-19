# MLV-Lab: Ecosistema Visual para Aprender RL

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.47-darkred)](https://pypi.org/project/mlvlab/)

> **La Misión:** Democratizar y concienciar sobre el desarrollo de la Inteligencia Artificial a través de la experimentación visual e interactiva.

MLV-Lab es un ecosistema pedagógico diseñado para explorar los conceptos fundamentales de la IA sin necesidad de conocimientos matemáticas avanzados. Nuestra filosofía es **"Show, don't tell"**: pasamos de la teoría abstracta a la práctica concreta y visual.

Este proyecto tiene dos audiencias principales:
1.  **Entusiastas de la IA:** Una herramienta para jugar, entrenar y observar agentes inteligentes resolviendo problemas complejos desde la terminal.
2.  **Desarrolladores de IA:** Un *sandbox* con entornos estándar (compatibles con [Gymnasium](https://gymnasium.farama.org/)) para diseñar, entrenar y analizar agentes desde cero.

---

## 🚀 Uso Rápido (CLI)

MLV-Lab se controla a través del comando `mlv`. El flujo de trabajo está diseñado para ser intuitivo.

**Requisito:** Python 3.9+

### 1. Instalación
```bash
pip install -U git+https://github.com/hcosta/mlvlab
```

### 2. Flujo de Trabajo Básico

```bash
# 1. Descubre las unidades disponibles o lista por unidad
mlv list
mlv list ql

# 2. Juega para entender el objetivo (usa Flechas/WASD)
mlv ant-v1 play

# 3. Entrena un agente con una semilla específica (ej. 123)
#    (Se ejecuta rápido y guarda los "pesos" en data/mlv_ql_ant-v1/seed-123/)
mlv ant-v1 train --seed 123

# 4. Evalúa el entrenamiento visualmente (modo interactivo por defecto)
#    (Carga los pesos de la semilla 123 y abre la ventana con el agente usando esos pesos)
mlv ant-v1 eval --seed 123

# 4b. Si quieres grabar un vídeo (en lugar de solo visualizar), añade --record
mlv ant-v1 eval --seed 123 --record

# 5. Consulta la ficha técnica y la documentación de un entorno
mlv ant-v1 help
```
---

## 📦 Entornos disponibles

| Entorno    | ID (mlv, gym)                | Baseline    | Detalles |  |
|-----------|-----------------------------|------------|----------------|--------------|
| Lost Ant  | `AntScout-v1`,<br>`mlv/AntScout-v1` | Q-Learning | [README.md](./mlvlab/envs/ant_scout_v1/README.md) | <a href="./mlvlab/envs/ant_scout_v1/README.md"><img src="./docs/ant_scout_v1/mode_play.jpg" alt="modo play" width="100px"></a> |



---

## 💻 Desarrollo de Agentes (API)

Puedes usar los entornos de MLV-Lab en tus propios proyectos de Python como cualquier otra librería de Gymnasium.

### 1. Instalación en tu Proyecto

```bash
# Crea tu entorno virtual y luego instala las dependencias
pip install -U git+https://github.com/hcosta/mlvlab
```

### 2. Uso en tu Código

```python
import gymnasium as gym
import mlvlab  # ¡Importante! Esto registra los entornos "mlv/..." y mantiene compatibilidad con los antiguos

# Crea el entorno como lo harías normalmente con Gymnasium
env = gym.make("mlv/ant-v1", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    # Aquí es donde va tu lógica para elegir una acción
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
<!--
---

## 🏛️ Extender MLV-Lab (Plugins)

Para usuarios avanzados, MLV-Lab puede ser extendido con nuevos comandos a través de un sistema de plugins. Esto te permite integrar tus propias herramientas (ej. una vista personalizada) directamente en la CLI `mlv`.

### Ejemplo: Crear un comando `mlv view`

1.  **Crea tu herramienta** con Typer.
2.  **Declara un "entry point"** en el `pyproject.toml` de tu herramienta para que MLV-Lab lo descubra:

```toml
# pyproject.toml de tu plugin
[project.entry-points."mlvlab.plugins"]
view = "mi_visualizador.cli:app"
```

3.  **Instala tu herramienta** (`pip install -e .`).

Ahora, tu nuevo comando estará disponible:
`mlv view mi-comando --argumentos`
-->
---

## 🛠️ Contribuir a MLV-Lab

Si quieres añadir nuevos entornos o funcionalidades al núcleo de MLV-Lab:

1.  Clona el repositorio.
2.  Crea un entorno virtual.
   
    ```bash
    python -m venv .venv
    ``` 

3.  Activa tu entorno virtual.

    * macOS/Linux: `source .venv/bin/activate`
    * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

4.  Instala el proyecto en modo editable con las dependencias de desarrollo:

    ```bash
    pip install -e ".[dev]"
    ```

Esto instala `mlvlab` (modo editable) y también las herramientas del grupo `[dev]`.

---

## ⚙️ Opciones de la CLI: list, play, train, eval, view

### Modo lista: `mlv list`

Devuelve un listado de las categorías de entornos disponibles o

- **Uso básico**: `mlv list`
- **Opciones**: ID de la categoría a filtrar (ej. `mlv list ql`).

Ejemplos:

```bash
mlv list
mlv list ql
```

### Modo juego: `mlv <env-id> play`

Ejecuta el entorno en modo interactivo (humano) para probar el control manual.

- **Uso básico**: `mlv <env-id> play`
- **Parámetros**:
  - **env_id**: No del entorno (ej. `ant-v1`).
  - **--seed, -s**: Semilla para reproducibilidad del mapa. Si no se especifica, se usa la predeterminada del entorno.

Ejemplo:

```bash
mlv ant-v1 play --seed 42
```

### Modo entrenamiento: `mlv <env-id> train`

Entrena el agente baseline del entorno y guarda los pesos/artefactos en `data/<env>/<seed-XYZ>/`.

- **Uso básico**: `mlv <env-id> train`
- **Parámetros**:
  - **env_id**: ID del entorno.
  - **--seed, -s**: Semilla del entrenamiento. Si no se indica, se genera una aleatoria y se muestra por pantalla.
  - **--eps, -e**: Número de episodios (sobrescribe el valor de la configuración baseline del entorno).
  - **--render, -r**: Renderiza el entrenamiento en tiempo real. Nota: esto puede ralentizar significativamente el entrenamiento.

Ejemplo:

```bash
mlv train mlv/ant-v1 --seed 123 --eps 500 --render
```

### Modo evaluación: `mlv <env-id> eval`

Evalúa un entrenamiento existente cargando la Q-Table/pesos desde el directorio de `run` correspondiente. Por defecto, se abre la ventana (modo `human`) y se visualiza al agente usando sus pesos. Para grabar un vídeo en disco, añade `--record`.

- **Uso básico**: `mlv <env-id> eval [opciones]`
- **Parámetros**:
  - **env_id**: ID del entorno.
  - **--seed, -s**: Semilla del `run` a evaluar. Si no se indica, se usa el último `run` disponible para ese entorno.
  - **--eps, -e**: Número de episodios a ejecutar durante la evaluación. Por defecto: 5.
  - **--rec, -r**: Graba y genera un vídeo de la evaluación (en `evaluation.mp4` dentro del directorio del `run`). Si no se especifica, solo se muestra la ventana interactiva y no se guardan vídeos.
  - **--speed, -sp**: Factor de multiplicación de velocidad, por defecto es `1.0`, para verlo a la mitad poner `.5`.

Ejemplos:

```bash
# Visualizar el agente usando los pesos del último entrenamiento
mlv ant-v1 eval

# Visualizar un entrenamiento concreto y grabar vídeo
mlv ant-v1 eval --seed 123 --record

# Evaluar 10 episodios
mlv ant-v1 eval --seed 123 --eps 10 --record
```

### Modo vista interactiva: `mlv <env-id> view`

Lanza la vista interactiva (Analytics View) del entorno con controles de simulación, métricas y gestión de modelos.

- Uso básico: `mlv <env-id> view`

Ejemplo:

```bash
mlv ant-v1 view
```
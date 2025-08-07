# MLV-Lab: Ecosistema para el Aprendizaje Visual de IA

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-brightgreen)](https://pypi.org/project/mlvlab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

> **La Misión:** Democratizar y concienciar sobre el desarrollo de la Inteligencia Artificial a través de la experimentación visual e interactiva.

MLV-Lab es un ecosistema pedagógico diseñado para explorar los conceptos fundamentales de la IA. Nuestra filosofía es **"Show, don't tell"**: pasamos de la teoría abstracta a la práctica concreta y visual.

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
# 1. Descubre qué simulaciones hay disponibles
mlv list

# 2. Juega para entender el objetivo (usa Flechas/WASD)
mlv play mlvlab/ant-v1

# 3. Entrena un agente con una semilla específica (ej. 123)
#    (Se ejecuta rápido y guarda los "pesos" en data/mlvlab_ant-v1/seed-123/)
mlv train mlvlab/ant-v1 --seed 123

# 4. Evalúa el entrenamiento y graba un vídeo
#    (Carga los pesos de la semilla 123 y crea un vídeo en la misma carpeta)
mlv eval mlvlab/ant-v1 --seed 123

# 5. Consulta la ficha técnica y la documentación de un entorno
mlv help mlvlab/ant-v1
```

---

## 💻 Desarrollo de Agentes (API)

Puedes usar los entornos de MLV-Lab en tus propios proyectos de Python como cualquier otra librería de Gymnasium.

### 1. Instalación en tu Proyecto

```bash
# Crea tu entorno virtual y luego instala las dependencias
pip install mlvlab numpy
```

### 2. Uso en tu Código

```python
import gymnasium as gym
import mlvlab  # ¡Importante! Esto registra los entornos "mlvlab/..."

# Crea el entorno como lo harías normalmente con Gymnasium
env = gym.make("mlvlab/ant-v1", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    # Aquí es donde va tu lógica para elegir una acción
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## 🏛️ Extender MLV-Lab (Plugins)

Para usuarios avanzados, MLV-Lab puede ser extendido con nuevos comandos a través de un sistema de plugins. Esto te permite integrar tus propias herramientas (ej. un panel de visualización) directamente en la CLI `mlv`.

### Ejemplo: Crear un comando `mlv panel`

1.  **Crea tu herramienta** con Typer.
2.  **Declara un "entry point"** en el `pyproject.toml` de tu herramienta para que MLV-Lab lo descubra:

```toml
# pyproject.toml de tu plugin
[project.entry-points."mlvlab.plugins"]
panel = "mi_visualizador.cli:app"
```

3.  **Instala tu herramienta** (`pip install -e .`).

Ahora, tu nuevo comando estará disponible:
`mlv panel mi-comando --argumentos`

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
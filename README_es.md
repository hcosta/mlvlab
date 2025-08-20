# MLV-Lab: Ecosistema Visual para Aprender RL

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-darkred)](https://pypi.org/project/mlvlab/)

<div style="position: absolute; top: 15px; right: 15px;">
    <a href="./README.md">
        <img src="https://flagicons.lipis.dev/flags/4x3/gb.svg" alt="English version" width="25"/>
    </a>
    <a href="./README_es.md" style="margin-left: 8px;">
        <img src="https://flagicons.lipis.dev/flags/4x3/es.svg" alt="Versión en español" width="25"/>
    </a>
</div>

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
mlv list ants

# 2. Juega para entender el objetivo (usa Flechas/WASD)
mlv play AntScout-v1

# 3. Entrena un agente con una semilla específica (ej. 123)
#    (Se ejecuta rápido y guarda los "pesos" en data/mlv_AntScout-v1/seed-123/)
mlv train AntScout-v1 --seed 123

# 4. Evalúa el entrenamiento visualmente (modo interactivo por defecto)
#    (Carga los pesos de la semilla 123 y abre la ventana con el agente usando esos pesos)
mlv eval AntScout-v1 --seed 123

# 4b. Si quieres grabar un vídeo (en lugar de solo visualizar), añade --rec
mlv eval AntScout-v1 --seed 123 --rec

# 5. Crea una vista interactiva de la simulación
mlv view AntScout-v1

# 6. Consulta la ficha técnica y la documentación de un entorno
mlv docs AntScout-v1
```
---

## 📦 Entornos disponibles

| Saga | Entorno    | ID (Gym)                | Baseline    | Detalles |  |
|------|-----------|-----------------------------|------------|----------------|--------------|
| 🐜 Hormigas | Vigía Rastreadora | `mlv/AntScout-v1` | Q-Learning | [README_es.md](./mlvlab/envs/ant_scout_v1/README_es.md) | <a href="./mlvlab/envs/ant_scout_v1/README_es.md"><img src="./docs/ant_scout_v1/mode_play.jpg" alt="modo play" width="75px"></a> |

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
env = gym.make("mlv/AntScout-v1", render_mode="human")
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

## ⚙️ Opciones de la CLI: list, config, play, train, eval, view, docs

### Modo lista: `mlv list`

Devuelve un listado de las categorías de entornos disponibles o

- **Uso básico**: `mlv list`
- **Opciones**: ID de la categoría a filtrar (ej. `mlv list ants`).

Ejemplos:

```bash
mlv list
mlv list ants
```

### Modo configuración: `mlv config`

Gestiona la configuración de MLV-Lab incluyendo la configuración del idioma.

- **Uso básico**: `mlv config <acción> [clave] [valor]`
- **Acciones**:
  - **get**: Mostrar configuración actual o clave específica
  - **set**: Establecer un valor de configuración
  - **reset**: Restablecer configuración a valores predeterminados
- **Claves comunes**:
  - **locale**: Configuración del idioma (`en` para inglés, `es` para español)

Ejemplos:

```bash
# Mostrar configuración actual
mlv config get

# Mostrar configuración específica
mlv config get locale

# Establecer idioma a español
mlv config set locale es

# Restablecer a valores predeterminados
mlv config reset
```

### Modo juego: `mlv play <env-id>`

Ejecuta el entorno en modo interactivo (humano) para probar el control manual.

- **Uso básico**: `mlv play <env-id>`
- **Parámetros**:
  - **env_id**: ID del entorno (ej. `mlv/AntScout-v1`).
  - **--seed, -s**: Semilla para reproducibilidad del mapa. Si no se especifica, se usa la predeterminada del entorno.

Ejemplo:

```bash
mlv play AntScout-v1 --seed 42
```

### Modo entrenamiento: `mlv train <env-id>`

Entrena el agente baseline del entorno y guarda los pesos/artefactos en `data/<env>/<seed-XYZ>/`.

- **Uso básico**: `mlv train <env-id>`
- **Parámetros**:
  - **env_id**: ID del entorno.
  - **--seed, -s**: Semilla del entrenamiento. Si no se indica, se genera una aleatoria y se muestra por pantalla.
  - **--eps, -e**: Número de episodios (sobrescribe el valor de la configuración baseline del entorno).
  - **--render, -r**: Renderiza el entrenamiento en tiempo real. Nota: esto puede ralentizar significativamente el entrenamiento.

Ejemplo:

```bash
mlv train AntScout-v1 --seed 123 --eps 500 --render
```

### Modo evaluación: `mlv eval <env-id>`

Evalúa un entrenamiento existente cargando la Q-Table/pesos desde el directorio de `run` correspondiente. Por defecto, se abre la ventana (modo `human`) y se visualiza al agente usando sus pesos. Para grabar un vídeo en disco, añade `--rec`.

- **Uso básico**: `mlv eval <env-id> [opciones]`
- **Parámetros**:
  - **env_id**: ID del entorno.
  - **--seed, -s**: Semilla del `run` a evaluar. Si no se indica, se usa el último `run` disponible para ese entorno.
  - **--eps, -e**: Número de episodios a ejecutar durante la evaluación. Por defecto: 5.
  - **--rec, -r**: Graba y genera un vídeo de la evaluación (en `evaluation.mp4` dentro del directorio del `run`). Si no se especifica, solo se muestra la ventana interactiva y no se guardan vídeos.
  - **--speed, -sp**: Factor de multiplicación de velocidad, por defecto es `1.0`, para verlo a la mitad poner `.5`.

Ejemplos:

```bash
# Visualizar el agente usando los pesos del último entrenamiento
mlv eval AntScout-v1

# Visualizar un entrenamiento concreto y grabar vídeo
mlv eval AntScout-v1 --seed 123 --record

# Evaluar 10 episodios
mlv eval AntScout-v1 --seed 123 --eps 10 --rec
```

### Modo vista interactiva: `mlv view <env-id>`

Lanza la vista interactiva (Analytics View) del entorno con controles de simulación, métricas y gestión de modelos.

- Uso básico: `mlv view <env-id>`

Ejemplo:

```bash
mlv view AntScout-v1
```

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

## 🌍 Internacionalización

MLV-Lab soporta múltiples idiomas. El idioma por defecto es inglés, y el español está completamente soportado como idioma alternativo.

### Configuración de Idioma

Puedes establecer el idioma de varias formas:

1. **Variable de Entorno:**
   ```bash
   export MLVLAB_LOCALE=es  # Español
   export MLVLAB_LOCALE=en  # Inglés (por defecto)
   ```

2. **Archivo de Configuración del Usuario:**
   ```bash
   # Crear ~/.mlvlab/config.json
   echo '{"locale": "es"}' > ~/.mlvlab/config.json
   ```

3. **Detección Automática:**
   El sistema detecta automáticamente el idioma de tu sistema y usa español si está disponible, de lo contrario usa inglés por defecto.

### Idiomas Disponibles

- **Inglés (en)**: Idioma por defecto
- **Español (es)**: Alternativa completamente traducida

---

## 📄 Documentación en Múltiples Idiomas

- **Inglés**: [README.md](./README.md)
- **Español**: [README_es.md](./README_es.md) (este archivo)
# MLV-Lab: Visual AI Learning Ecosystem

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/badge/PyPI-v0.2-blue)](https://pypi.org/project/mlvlab/)
&nbsp;&nbsp;&nbsp;&nbsp;
[![en](https://img.shields.io/badge/Lang-EN-red.svg)](./README.md)
[![es](https://img.shields.io/badge/Lang-ES-lightgrey.svg)](./docs/README_es.md)

> **Our Mission:** Democratize and raise awareness about Artificial Intelligence development through visual and interactive experimentation.

MLV-Lab is a pedagogical ecosystem designed to explore the fundamental concepts of AI without requiring advanced mathematical knowledge. Our philosophy is **"Show, don't tell"**: we move from abstract theory to concrete, visual practice.

This project has two main audiences:
1. **AI Enthusiasts:** A tool to play, train, and observe intelligent agents solving complex problems from the terminal.
2. **AI Developers:** A *sandbox* with standard environments (compatible with [Gymnasium](https://gymnasium.farama.org/)) to design, train, and analyze agents from scratch.

---

## 🚀 Quick Start (CLI)

MLV-Lab is controlled through the `mlv` command. The workflow is designed to be intuitive.

**Requirement:** Python 3.10+

### 1. Installation
```bash
pip install -U mlvlab
mlv --install-completion  # Optional for command autocompletion
```

### 2. Basic Workflow

```bash
# 1. Discover available units or list by unit
mlv list
mlv list ants

# 2. Play to understand the objective (use Arrow keys/WASD)
mlv play AntScout-v1

# 3. Train an agent with a specific seed (e.g., 123)
#    (Runs quickly and saves "weights" in data/mlv_AntScout-v1/seed-123/)
mlv train AntScout-v1 --seed 123

# 4. Evaluate training visually (interactive mode by default)
#    (Loads weights from seed 123 and opens window with agent using those weights)
mlv eval AntScout-v1 --seed 123

# 4b. If you want to record a video (instead of just viewing), add --rec
mlv eval AntScout-v1 --seed 123 --rec

# 5. Create an interactive view of the simulation
mlv view AntScout-v1

# 6. Check technical specifications and environment documentation
mlv docs AntScout-v1
```

---

## 📦 Available Environments

| Saga | Environment | ID (Gym) | Baseline | Details |  |
|------|-----------|-----------------------------|------------|----------------|--------------|
| 🐜 Ants | Scout Ant | `mlv/AntScout-v1` | Q-Learning | [README.md](./mlvlab/envs/ant_scout_v1/README.md) | <a href="./mlvlab/envs/ant_scout_v1/README.md"><img src="./docs/ant_scout_v1/mode_play.jpg" alt="play mode" width="75px"></a> |

---

## 💻 Agent Development (API)

You can use MLV-Lab environments in your own Python projects like any other Gymnasium library.

### 1. Installation in your Project

```bash
# Create your virtual environment and then install dependencies
pip install -U git+https://github.com/hcosta/mlvlab
```

### 2. Usage in your Code

```python
import gymnasium as gym
import mlvlab  # Important! This registers "mlv/..." environments and maintains compatibility with old ones

# Create environment as you would normally with Gymnasium
env = gym.make("mlv/AntScout-v1", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    # This is where your logic goes to choose an action
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## ⚙️ CLI Options: list, play, train, eval, view, docs, config

### List mode: `mlv list`

Returns a listing of available environment categories or

- **Basic usage**: `mlv list`
- **Options**: ID of category to filter (e.g., `mlv list ants`).

Examples:

```bash
mlv list
mlv list ants
```

### Play mode: `mlv play <env-id>`

Runs the environment in interactive mode (human) to test manual control.

- **Basic usage**: `mlv play <env-id>`
- **Parameters**:
  - **env_id**: Environment ID (e.g., `mlv/AntScout-v1`).
  - **--seed, -s**: Seed for map reproducibility. If not specified, uses environment default.

Example:

```bash
mlv play AntScout-v1 --seed 42
```

### Training mode: `mlv train <env-id>`

Trains the environment's baseline agent and saves weights/artifacts in `data/<env>/<seed-XYZ>/`.

- **Basic usage**: `mlv train <env-id>`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Training seed. If not indicated, generates a random one and displays it.
  - **--eps, -e**: Number of episodes (overrides environment baseline configuration value).
  - **--render, -r**: Render training in real time. Note: this can significantly slow down training.

Example:

```bash
mlv train AntScout-v1 --seed 123 --eps 500 --render
```

### Evaluation mode: `mlv eval <env-id>`

Evaluates an existing training by loading Q-Table/weights from the corresponding `run` directory. By default, opens window (human mode) and visualizes agent using its weights. To record video to disk, add `--rec`.

- **Basic usage**: `mlv eval <env-id> [options]`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Seed of `run` to evaluate. If not indicated, uses latest `run` available for that environment.
  - **--eps, -e**: Number of episodes to run during evaluation. Default: 5.
  - **--rec, -r**: Record and generate evaluation video (in `evaluation.mp4` within `run` directory). If not specified, only shows interactive window and doesn't save videos.
  - **--speed, -sp**: Speed multiplication factor, default is `1.0`, to see at half speed put `.5`.

Examples:

```bash
# Visualize agent using weights from latest training
mlv eval AntScout-v1

# Visualize specific training and record video
mlv eval AntScout-v1 --seed 123 --record

# Evaluate 10 episodes
mlv eval AntScout-v1 --seed 123 --eps 10 --rec
```

### Interactive view mode: `mlv view <env-id>`

Launches the interactive view (Analytics View) of the environment with simulation controls, metrics, and model management.

- Basic usage: `mlv view <env-id>`

Example:

```bash
mlv view AntScout-v1
```

### Documentation mode: `mlv docs`

Opens a browser with the `README.md` file associated with the environment, providing full details.
It also displays a summary in the terminal in the configured language:

- **Basic usage**: `mlv docs <env-id>`

Example:

```bash
mlv docs AntScout-v1
```

### Configuration mode: `mlv config`

Manages MLV-Lab configuration including language settings (the package detects the system language automatically):

- **Basic usage**: `mlv config <action> [key] [value]`
- **Actions**:
  - **get**: Show current configuration or specific key
  - **set**: Set a configuration value
  - **reset**: Reset configuration to defaults
- **Common keys**:
  - **locale**: Language setting (`en` for English, `es` for Spanish)

Examples:

```bash
# Show current configuration
mlv config get

# Show specific setting
mlv config get locale

# Set language to Spanish
mlv config set locale es

# Reset to defaults
mlv config reset
```

---

## 🛠️ Contributing to MLV-Lab

If you want to add new environments or functionality to MLV-Lab core:

1. Clone the repository.
2. Create a virtual environment.
   
   ```bash
   python -m venv .venv
   ``` 

3. Activate your virtual environment.

   * macOS/Linux: `source .venv/bin/activate`
   * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

4. Install the project in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

This installs `mlvlab` (editable mode) and also the tools from the `[dev]` group.

---

## 🌍 Internationalization

MLV-Lab supports multiple languages. The default language is English, and Spanish is fully supported as an alternative language.

### Language Configuration

You can set the language in several ways:

1. **Environment Variable:**
   ```bash
   export MLVLAB_LOCALE=es  # Spanish
   export MLVLAB_LOCALE=en  # English (default)
   ```

2. **User Configuration File:**
   ```bash
   # Create ~/.mlvlab/config.json
   echo '{"locale": "es"}' > ~/.mlvlab/config.json
   ```

3. **Automatic Detection:**
   The system automatically detects your system language and uses Spanish if available, otherwise defaults to English.

### Available Languages

- **English (en)**: Default language
- **Spanish (es)**: Fully translated alternative
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

## 🚀 Quick Start (Interactive Shell)

MLV-Lab is controlled through an interactive shell called `MLVisual`. The workflow is designed to be intuitive and user-friendly.

**Requirement:** Python 3.10+

### 1. Installation with uv

```bash
# Install uv package manager
pip install uv

# Create a dedicated virtual environment
uv venv

# Install mlvlab in the virtual environment
uv pip install mlvlab

# For development (local installation)
uv pip install -e ".[dev]"

# Launch the interactive shell
uv run mlv shell
```

### 2. Interactive Shell Workflow

Once you're in the `(MLVisual)>` shell:

```sh
MLVLab> list                    # Discover available units
MLVLab> list ants               # List environments from a specific unit
MLVLab> play <env>              # Play to understand the objective (use Arrow keys/WASD)
MLVLab> train <env>             # Train an agent with a specific seed
MLVLab> eval <env>              # Evaluate training visually (interactive mode by default)
MLVLab> view <env>              # Create an interactive view of the simulation
MLVLab> docs <env>              # Check technical specifications and documentation
MLVLab> config <args>           # Manage configuration settings
MLVLab> clear                   # Reset the terminal logs
MLVLab> exit                    # Exit the shell (or use 'quit')
```

**Example session:**
```sh
MLVLab> play AntScout-v1
MLVLab> train AntScout-v1 --seed 123
MLVLab> eval AntScout-v1 --seed 123
MLVLab> view AntScout-v1
MLVLab> docs AntScout-v1
MLVLab> exit
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
pip install -U mlvlab
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

## ⚙️ Shell Commands: list, play, train, eval, view, docs, config

### List command: `list [unit]`

Returns a listing of available environment categories or environments from a specific unit.

- **Basic usage**: `list`
- **Options**: ID of category to filter (e.g., `list ants`).

Examples:

```bash
MLVLab> list
MLVLab> list ants
```

### Play command: `play <env-id> [options]`

Runs the environment in interactive mode (human) to test manual control.

- **Basic usage**: `play <env-id>`
- **Parameters**:
  - **env_id**: Environment ID (e.g., `AntScout-v1`).
  - **--seed, -s**: Seed for map reproducibility. If not specified, uses environment default.

Example:

```bash
MLVLab> play AntScout-v1 --seed 42
```

### Training command: `train <env-id> [options]`

Trains the environment's baseline agent and saves weights/artifacts in `data/<env>/<seed-XYZ>/`.

- **Basic usage**: `train <env-id>`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Training seed. If not indicated, generates a random one and displays it.
  - **--eps, -e**: Number of episodes (overrides environment baseline configuration value).
  - **--render, -r**: Render training in real time. Note: this can significantly slow down training.

Example:

```bash
MLVLab> train AntScout-v1 --seed 123 --eps 500 --render
```

### Evaluation command: `eval <env-id> [options]`

Evaluates an existing training by loading Q-Table/weights from the corresponding `run` directory. By default, opens window (human mode) and visualizes agent using its weights. To record video to disk, add `--rec`.

- **Basic usage**: `eval <env-id> [options]`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Seed of `run` to evaluate. If not indicated, uses latest `run` available for that environment.
  - **--eps, -e**: Number of episodes to run during evaluation. Default: 5.
  - **--rec, -r**: Record and generate evaluation video (in `evaluation.mp4` within `run` directory). If not specified, only shows interactive window and doesn't save videos.
  - **--speed, -sp**: Speed multiplication factor, default is `1.0`, to see at half speed put `.5`.

Examples:

```bash
# Visualize agent using weights from latest training
MLVLab> eval AntScout-v1

# Visualize specific training and record video
MLVLab> eval AntScout-v1 --seed 123 --rec

# Evaluate 10 episodes
MLVLab> eval AntScout-v1 --seed 123 --eps 10 --rec
```

### Interactive view command: `view <env-id>`

Launches the interactive view (Analytics View) of the environment with simulation controls, metrics, and model management.

- Basic usage: `view <env-id>`

Example:

```bash
MLVLab> view AntScout-v1
```

### Documentation command: `docs <env-id>`

Opens a browser with the `README.md` file associated with the environment, providing full details.
It also displays a summary in the terminal in the configured language:

- **Basic usage**: `docs <env-id>`

Example:

```bash
MLVLab> docs AntScout-v1
```

### Configuration command: `config <action> [key] [value]`

Manages MLV-Lab configuration including language settings (the package detects the system language automatically):

- **Basic usage**: `config <action> [key] [value]`
- **Actions**:
  - **get**: Show current configuration or specific key
  - **set**: Set a configuration value
  - **reset**: Reset configuration to defaults
- **Common keys**:
  - **locale**: Language setting (`en` for English, `es` for Spanish)

Examples:

```bash
# Show current configuration
MLVLab> config get

# Show specific setting
MLVLab> config get locale

# Set language to Spanish
MLVLab> config set locale es

# Reset to defaults
MLVLab> config reset
```

---

## 🛠️ Contributing to MLV-Lab

If you want to add new environments or functionality to MLV-Lab core:

1. Clone the repository.
2. Create a virtual environment with uv.
   
   ```bash
   uv venv
   ``` 

3. Install the project in editable mode with development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

4. Launch the development shell:

   ```bash
   uv run mlv shell
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
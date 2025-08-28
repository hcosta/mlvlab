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

## 📦 Available Environments

| Name | Environment | Saga | Baseline | Details | Preview |
| -----| ----------- | ---- | -------- | ------- | :-----: |
| `AntLost-v1`<br><sup>`mlv/AntLost-v1`</sup> | Errant Drone  | 🐜 Ants | Random | [README.md](/mlvlab/envs/ant_lost_v1/README.md) | <a href="/mlvlab/envs/ant_lost_v1/README.md"><img src="./docs/ant_lost_v1/mode_play.jpg" alt="play mode" width="50px"></a> |
| `AntScout-v1`<br><sup>`mlv/AntScout-v1`</sup> | Lookout Scout  | 🐜 Ants | Q-Learning | [README.md](/mlvlab/envs/ant_scout_v1/README.md) | <a href="/mlvlab/envs/ant_scout_v1/README.md"><img src="./docs/ant_scout_v1/mode_play.jpg" alt="play mode" width="50px"></a> |
| `AntMaze-v1`<br><sup>`mlv/AntMaze-v1`</sup> | Dungeons & Pheromones | 🐜 Ants | Q-Learning | [README.md](/mlvlab/envs/ant_maze_v1/README.md) | <a href="/mlvlab/envs/ant_maze_v1/README.md"><img src="./docs/ant_maze_v1/mode_play.jpg" alt="play mode" width="50px"></a> |

---

## 🚀 Quick Start (Interactive Shell)

MLV-Lab is controlled through an interactive shell called `MLVisual`. The workflow is designed to be intuitive and user-friendly.

**Requirement:** Python 3.10+

### 1. Installation with uv

```bash
# Install uv package manager inside the virtual environment
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

Once inside the `MLV-Lab>` shell, we recommend following this logical flow to get acquainted with an environment. The philosophy is to explore, play, train, and finally, watch the artificial intelligence in action.

1.  🗺️ **Discover (`list`)**: Start by seeing what worlds you can explore. The `list` command will show you the available environment sagas.
2.  🕹️ **Play (`play`)**: Once you choose an environment, play it in manual mode to understand its mechanics, controls, and objective.
3.  🤖 **Train (`train`)**: Now, let the AI learn how to solve it. The `train` command will start the training process for the baseline agent.
4.  🎬 **Evaluate (`eval`)**: Watch the agent you just trained apply what it has learned. The `eval` command loads the training result and displays it visually.
5.  📚 **Learn (`docs`)**: If you want to dive deeper into the technical details of the environment, the `docs` command will open the full documentation for you.

This cycle of **play -> train -> evaluate** is the heart of the **MLV-Lab** experience.

### 3. Complete Example Session

Here is a concrete example that follows the recommended flow, with comments explaining each step.

```bash
# Launch the interactive shell
uv run mlv shell

# 1. Discover what environments are in the "Ants" category
MLV-Lab> list ants

# 2. Play to understand the objective of AntScout-v1
MLV-Lab> play AntScout-v1

# 3. Train an agent with a specific seed (so it can be repeated)
MLV-Lab> train AntScout-v1 --seed 123

# 4. Evaluate the result of that specific in a live simulation
MLV-Lab> eval AntScout-v1 --seed 123

# 6. Check the documentation to learn more
MLV-Lab> docs AntScout-v1

# Exit the session
MLV-Lab> exit
```

---

## 💻 Agent Development (as a Library)

You can use MLV-Lab environments in your own Python projects, just like any other Gymnasium-compatible library.

### 1. Installation in Your Project

This workflow assumes you want to write your own Python scripts that `import` the `mlvlab` package.

```bash
# Create a dedicated virtual environment for your project (if you don't already have one)
uv venv

# Install mlvlab inside that virtual environment
uv pip install mlvlab
```

### 2. Usage in your Code

First, create a file (for example, `my_agent.py`) with your code:

```python
import gymnasium as gym
import mlvlab  # Important! This "magic" line registers the "mlv/..." environments in Gymnasium

# Create the environment as you normally would
env = gym.make("mlv/AntScout-v1", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    # Here is where your logic for selecting an action goes
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Next, run the script using `uv run`, which will ensure it uses the Python from your virtual environment:

```bash
uv run python my_agent.py
```

**Note**: In editors like Visual Studio Code, you can automate this last step. Simply select the Python interpreter located inside your virtual environment (the path will be something like `.venv/Scripts/python.exe`) as the interpreter for your project. That way, when you press the "Run" button, the editor will automatically use the correct environment.

---

## ⚙️ Shell Commands: list, play, train, eval, view, docs, config

### List command: `list [unit]`

Returns a listing of available environment categories or environments from a specific unit.

- **Basic usage**: `list`
- **Options**: ID of category to filter (e.g., `list ants`).

Examples:

```bash
list
list ants
```

### Play command: `play <env-id> [options]`

Runs the environment in interactive mode (human) to test manual control.

- **Basic usage**: `play <env-id>`
- **Parameters**:
  - **env_id**: Environment ID (e.g., `AntScout-v1`).
  - **--seed, -s**: Seed for map reproducibility. If not specified, uses environment default.

Example:

```bash
play AntScout-v1 --seed 42
```

### Training command: `train <env-id> [options]`

Trains the environment's baseline agent and saves weights/artifacts in `data/<env-id>/<seed-XYZ>/`.

- **Basic usage**: `train <env-id>`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Training seed. If not indicated, generates a random one and displays it.
  - **--eps, -e**: Number of episodes (overrides environment baseline configuration value).
  - **--render, -r**: Render training in real time. Note: this can significantly slow down training.

Example:

```bash
train AntScout-v1 --seed 123 --eps 500 --render
```

### Evaluation command: `eval <env-id> [options]`

Evaluates an existing training by loading Q-Table/weights from the corresponding `run` directory. By default, opens window (human mode) and visualizes agent using its weights.

- **Basic usage**: `eval <env-id> [options]`
- **Parameters**:
  - **env_id**: Environment ID.
  - **--seed, -s**: Seed of `run` to evaluate. If not indicated, uses latest `run` available for that environment.
  - **--eps, -e**: Number of episodes to run during evaluation. Default: 5.
  - **--speed, -sp**: Speed multiplication factor, default is `1.0`, to see at half speed put `.5`.

Examples:

```bash
# Visualize agent using weights from latest training
eval AntScout-v1

# Visualize specific training 
eval AntScout-v1 --seed 123

# Evaluate 10 episodes
eval AntScout-v1 --seed 123 --eps 10
```

### Interactive view command: `view <env-id>`

Launches the interactive view (Analytics View) of the environment with simulation controls, metrics, and model management.

- Basic usage: `view <env-id>`

Example:

```bash
view AntScout-v1
```

### Documentation command: `docs <env-id>`

Opens a browser with the `README.md` file associated with the environment, providing full details.
It also displays a summary in the terminal in the configured language:

- **Basic usage**: `docs <env-id>`

Example:

```bash
docs AntScout-v1
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
config get

# Show specific setting
config get locale

# Set language to Spanish
config set locale es

# Reset to defaults
config reset
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

MLV-Lab supports multiple languages. The default language is English `en`, and Spanish `es` is fully supported as an alternative language.

### Language Configuration

The language can be configured in two ways:

1. **Automatic Detection:**
   The system automatically detects your system language and uses Spanish if available, otherwise defaults to English.

2. **Manual Language Change:**
   The desired language can be forced if it does not match the user's prefences:

   ```bash
   # Launch the interactive shell
   uv run mlv shell

   # Set language to English
   config set locale en

   # Set language to Spanish
   config set locale es
   ```

### Available Languages

- **English (`en`)**: Default language.
- **Spanish (`es`)**: Fully translated alternative.
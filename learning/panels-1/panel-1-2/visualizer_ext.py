import random
import time
from threading import Lock, Thread

from nicegui import ui, app
import gymnasium as gym

# Agente y utilidades
from q_learning import QLearningAgent, get_state_from_pos

# Wrappers definidos por defecto
from wrappers import TimePenaltyWrapper, FoodVectorObservationWrapper

# Helpers NiceGUI
from mlvlab.helpers.ng import frame_to_base64_src


# --- Configuración ---
GRID_SIZE = 15
USE_FOOD_VECTOR = True  # Activa el ObservationWrapper para vector comida-hormiga


# --- Crear entorno base y aplicar wrappers ---
env = gym.make(
    "mlvlab/ant-v1",
    render_mode="rgb_array",
    grid_size=GRID_SIZE,
    reward_food=500,
    reward_obstacle=-50,
    reward_move=0,
)

# Wrapper de recompensa (penalización por tiempo)
env = TimePenaltyWrapper(env, penalty=-0.05)

# Wrapper de observación (vector hacia la comida)
if USE_FOOD_VECTOR:
    env = FoodVectorObservationWrapper(env)


# --- Adaptación del espacio de estados para el agente ---
if USE_FOOD_VECTOR:
    # dx, dy en [-(GRID-1), ..., +(GRID-1)] → total (2*GRID-1)^2 estados
    SPAN = (2 * GRID_SIZE - 1)
    NUM_STATES = SPAN * SPAN

    def obs_to_state(obs):
        dx = int(obs[0])
        dy = int(obs[1])
        offset = GRID_SIZE - 1
        return (dy + offset) * SPAN + (dx + offset)
else:
    # Observación original (x, y) → estado con util de agente
    NUM_STATES = GRID_SIZE * GRID_SIZE

    def obs_to_state(obs):
        return get_state_from_pos(int(obs[0]), int(obs[1]), GRID_SIZE)


agent = QLearningAgent(num_states=NUM_STATES, num_actions=env.action_space.n)


# --- Estado global y sincronización ---
SIM = {
    'initialized': False,
    'obs': None,
    'info': None,
    'total_steps': 0,
    'stop': False,
    'thread': None,
}
ENV_LOCK = Lock()

app_state = {
    'command': 'run',
    'speed_multiplier': 1,  # pasos/seg lineales
    'epsilon': 1.0,
    'epsilon_decay': 0.99,
    'min_epsilon': 0.1,
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'steps_per_second': 0,
    'preview_image': None,
}


def startup_handler():
    # Reset inicial con semilla fija opcional
    if not SIM['initialized']:
        with ENV_LOCK:
            SIM['obs'], SIM['info'] = env.reset(
                seed=random.randint(0, 1_000))
        SIM['initialized'] = True

    def simulation_loop():
        last_check = time.perf_counter()
        step_accum = 0.0
        while not SIM['stop']:
            if app_state['command'] == 'pause':
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            dt = now - last_check
            last_check = now

            # Token bucket lineal
            step_accum += app_state['speed_multiplier'] * dt
            steps_to_do = min(int(step_accum), 20000)
            if steps_to_do <= 0:
                time.sleep(0.0005)
                continue

            for _ in range(steps_to_do):
                step_accum -= 1.0
                with ENV_LOCK:
                    obs = SIM['obs']
                    state = obs_to_state(obs)
                    action = agent.choose_action(state, app_state['epsilon'])
                    next_obs, reward, terminated, truncated, info = env.step(
                        action)
                    SIM['obs'] = next_obs
                    SIM['info'] = info
                next_state = obs_to_state(next_obs)
                agent.update(state, action, reward, next_state,
                             app_state['learning_rate'], app_state['discount_factor'])
                SIM['total_steps'] += 1
                if terminated or truncated:
                    with ENV_LOCK:
                        SIM['obs'], SIM['info'] = env.reset()
                    if app_state['epsilon'] > app_state['min_epsilon']:
                        app_state['epsilon'] *= app_state['epsilon_decay']
                    break

            time.sleep(0.0005)

    SIM['stop'] = False
    SIM['thread'] = Thread(target=simulation_loop, daemon=True)
    SIM['thread'].start()

    @app.on_shutdown
    def _stop_simulation():
        SIM['stop'] = True
        t = SIM.get('thread')
        if t and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass


@ui.page('/')
def main_page():
    loop_state = {
        'last_check_time': time.time(),
        'steps_at_last_check': 0,
    }

    ui.label('Visualizer EXT: entorno envuelto con Gymnasium Wrappers').classes(
        'text-xl font-semibold mt-2')

    with ui.card().classes('w-full max-w-[900px]'):
        ui.label().bind_text_from(
            app_state, 'speed_multiplier', lambda v: f'Velocidad: {v}x')
        ui.slider(min=1, max=50, step=1).bind_value(
            app_state, 'speed_multiplier')
        with ui.row():
            def toggle():
                app_state['command'] = 'pause' if app_state['command'] == 'run' else 'run'
            btn = ui.button('Pausar', on_click=toggle)
            btn.bind_text_from(
                app_state, 'command', lambda cmd: 'Pausar' if cmd == 'run' else 'Reanudar')
            ui.button('Cerrar', on_click=lambda: app.shutdown(), color='red')

        ui.image().props(
            'no-transition').classes('max-w-[600px]').bind_source_from(app_state, 'preview_image')
        ui.label().bind_text_from(app_state, 'steps_per_second',
                                  lambda v: f'Pasos/seg: {v:,d}')

    def render_tick():
        with ENV_LOCK:
            frame = env.render()
        app_state['preview_image'] = frame_to_base64_src(frame)

        now = time.time()
        elapsed = now - loop_state['last_check_time']
        if elapsed > 0.5:
            steps = SIM['total_steps'] - loop_state['steps_at_last_check']
            app_state['steps_per_second'] = int(
                steps / elapsed) if elapsed > 0 else 0
            loop_state['last_check_time'] = now
            loop_state['steps_at_last_check'] = SIM['total_steps']

    ui.timer(1/15, render_tick)


app.on_startup(startup_handler)
ui.run(title='Visualizer EXT con Wrappers',
       dark=False, reload=True, show=False)

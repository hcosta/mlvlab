# train_visualizer.py
import random
import time
import ast  # Necesario para la lógica de audio robusta
from nicegui import ui, app
import gymnasium as gym

# --- PASO 1: Importar el entorno, el agente y los helpers ---
try:
    import mlvlab
    from q_learning import QLearningAgent, get_state_from_pos
    # Asumiendo que los helpers están en la ubicación recomendada
    from mlvlab.helpers.ng import setup_audio, create_reward_chart, frame_to_base64_src
except ImportError:
    print("Error: El paquete 'mlvlab', sus helpers o 'q_learning.py' no se encontraron.")
    exit()

# --- PASO 2: Configuración Inicial ---

# CORRECCIÓN 1: Asegurar que GRID_SIZE coincida con la configuración del entorno.
GRID_SIZE = 15

env = gym.make(
    "mlvlab/ant-v1",
    render_mode="rgb_array",
    grid_size=15,          # Un mapa más grande
    reward_food=500,       # Una recompensa por comida mucho mayor
    reward_obstacle=-50,   # Un castigo por obstáculo menos severo
    reward_move=0,         # Sin castigo por moverse
)

agent = QLearningAgent(num_states=GRID_SIZE * GRID_SIZE,
                       num_actions=env.action_space.n)

# Helper de mlvlab para reproducir sonidos en NiceGUI
play_sound = None

# El estado de la aplicación contiene ahora los valores por defecto de los hiperparámetros
app_state = {
    "learning_rate": 0.1,
    "discount_factor": 0.9,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "min_epsilon": 0.1,
    "command": "run",
    "speed_multiplier": 1,
    "episodes_completed": 0,
    "current_episode_reward": 0,
    "reward_history": [],
    "chart_reward_number": 100,
    "sound_enabled": False,
    "chart_visible": True,
    "preview_image": None,
    "steps_per_second": 0,
}
reward_chart = None


def startup_handler():
    """Inicializa recursos globales como el audio una sola vez."""
    global play_sound
    play_sound = setup_audio()


@ui.page('/')
def main_interface():
    global reward_chart

    # El estado del bucle interno se mantiene igual
    loop_state = {
        'obs': None,
        'info': None,
        'total_steps': 0,
        'last_check_time': time.time(),
        'steps_at_last_check': 0,
        'last_step_time': time.time(),
        'current_seed': None,
    }

    def reset_simulation():
        """Reinicia toda la simulación, generando un NUEVO mapa."""
        loop_state['current_seed'] = random.randint(0, 1_000_000_000)
        obs, info = env.reset(seed=loop_state['current_seed'])
        loop_state['obs'] = obs
        loop_state['info'] = info
        agent.reset()
        loop_state['total_steps'] = 0
        loop_state['last_check_time'] = time.time()
        loop_state['steps_at_last_check'] = 0
        loop_state['last_step_time'] = time.time()
        app_state.update({
            "episodes_completed": 0, "reward_history": [], "epsilon": 1.0,
            "current_episode_reward": 0, "steps_per_second": 0, "command": "run"
        })
        if reward_chart is not None:
            reward_chart.options['series'][0]['data'] = []
            reward_chart.update()
        print(
            f"Simulación reiniciada con un nuevo mapa (Seed: {loop_state['current_seed']}).")

    # Guarda una lista para los temporizadores de este cliente
    active_timers = []

    # --- NUEVA LÓGICA DE DIÁLOGO Y CIERRE ---
    with ui.dialog() as dialog, ui.card():
        ui.label('¿Estás seguro de que quieres cerrar la aplicación?')
        with ui.row().classes('w-full justify-end'):
            # El botón "Sí" ejecuta la lógica de cierre que teníamos antes
            def do_shutdown():
                print("Confirmado. Iniciando cierre natural...")
                for timer in active_timers:
                    timer.cancel()
                ui.run_javascript('window.close()')
                app.shutdown()

            ui.button('Sí, cerrar', on_click=do_shutdown, color='red')
            ui.button('No, cancelar', on_click=dialog.close)

    # Reemplaza tu función handle_shutdown con esta:
    async def handle_shutdown():
        """
        Muestra un diálogo de confirmación en el navegador y, si se acepta,
        cierra la aplicación de forma limpia.
        """
        # 1. Ejecuta el confirm() en el navegador y espera la respuesta.
        confirmacion = await ui.run_javascript('return confirm("¿Estás seguro de que quieres cerrar la aplicación?");')

        # 2. Continúa solo si el usuario hizo clic en "Aceptar" (confirmacion es true).
        if confirmacion:
            print("Confirmado. Iniciando cierre natural...")

            # Detiene los temporizadores para un cierre limpio
            for timer in active_timers:
                timer.cancel()

            # Pide a esta pestaña que se cierre
            ui.run_javascript('window.close()')

            # Pide al servidor que se apague
            app.shutdown()
        else:
            print("Cierre cancelado por el usuario.")

    # Iniciar la simulación la primera vez para este cliente
    reset_simulation()

    # --- Definición de la Interfaz de Usuario ---
    ui.label('Project 1.2: The Lost Ant Colony (Arquitectura Original Corregida)').classes(
        'w-full text-2xl font-bold text-center mt-4 mb-2')

    with ui.element('div').classes('w-full flex justify-center'):
        with ui.element('div').classes('w-full max-w-[1400px] flex flex-col lg:flex-row'):
            with ui.column().classes('w-full lg:w-1/4 pb-4 lg:pr-2 lg:pb-0'):
                with ui.card().classes('w-full mb-4'):
                    ui.label('Controles de Simulación').classes(
                        'text-lg font-semibold text-center w-full')
                    ui.label().bind_text_from(app_state, 'speed_multiplier',
                                              lambda v: f'Velocidad Simulación: {v}x')
                    ui.slider(min=1, max=1000, step=1).bind_value(
                        app_state, 'speed_multiplier')

                    with ui.row().classes('w-full justify-around mt-3'):
                        def toggle_simulation():
                            app_state['command'] = "pause" if app_state['command'] == "run" else "run"
                        with ui.button(on_click=toggle_simulation).props('outline'):
                            # CORRECCIÓN 2: Añadir un icono inicial para evitar TypeError
                            ui.icon('pause').bind_name_from(
                                app_state, 'command', lambda cmd: 'pause' if cmd == 'run' else 'play_arrow')

                        ui.button(on_click=lambda: app_state.update(
                            {"command": "reset"})).props('icon=refresh outline')

                        with ui.button(on_click=lambda: app_state.update({'sound_enabled': not app_state['sound_enabled']})).props('outline'):
                            # CORRECCIÓN 2: Añadir un icono inicial para evitar TypeError
                            ui.icon('volume_up').bind_name_from(
                                app_state, 'sound_enabled', lambda enabled: 'volume_up' if enabled else 'volume_off')

                        ui.button(on_click=dialog.open).props(
                            'icon=close outline color="red"')

                with ui.card().classes('w-full'):
                    ui.label('Configuración del Agente').classes(
                        'text-lg font-semibold text-center w-full mb-3')
                    with ui.grid(columns=3).classes('w-full gap-x-2 items-center'):
                        ui.label('Tasa de Aprendizaje (α):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['learning_rate'], format='%.2f', step=0.01, min=0, max=1).bind_value(
                            app_state, 'learning_rate').bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')
                        ui.label('Factor de Descuento (γ):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['discount_factor'], format='%.2f', step=0.01, min=0, max=1).bind_value(
                            app_state, 'discount_factor').bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')
                        ui.label('Decaimiento de Epsilon (ε):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['epsilon_decay'], format='%.5f', step=0.00001, min=0, max=1).bind_value(
                            app_state, 'epsilon_decay').bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')

            with ui.column().classes('w-full lg:w-2/4 pb-4 lg:pb-0 lg:px-2 items-center'):
                with ui.card().classes('w-full p-0 bg-black flex justify-center items-center'):
                    ui.image().props(
                        'no-transition').classes('max-w-[550px]').bind_source_from(app_state, 'preview_image')

            with ui.column().classes('w-full lg:w-1/4 lg:pl-2'):
                with ui.card().classes('w-full mb-4'):
                    ui.label('Métricas en Tiempo Real').classes(
                        'text-lg font-semibold text-center w-full')
                    ui.label().bind_text_from(app_state, 'epsilon',
                                              lambda v: f'Epsilon (Exploración): {v:.3f}')
                    ui.label().bind_text_from(app_state, 'current_episode_reward',
                                              lambda v: f'Recompensa Actual: {v}')
                    ui.label().bind_text_from(app_state, 'episodes_completed',
                                              lambda v: f'Episodios Completados: {v}')
                    ui.label().bind_text_from(app_state, 'steps_per_second',
                                              lambda v: f'Pasos/seg: {v:,d}')
                    ui.button('Mostrar/Esconder Gráfico', on_click=lambda: app_state.update(
                        {"chart_visible": not app_state["chart_visible"]})).props('icon=bar_chart outline w-full')

                with ui.card().classes('w-full flex-grow') as chart_card:
                    chart_card.bind_visibility_from(app_state, 'chart_visible')
                    reward_chart = create_reward_chart(
                        chart_card, number=app_state['chart_reward_number'])

    def simulation_tick():
        if app_state["command"] == "reset":
            reset_simulation()
            return
        if app_state["command"] != "run" or loop_state.get('obs') is None:
            return

        now = time.time()
        if now - loop_state['last_step_time'] < 1.0 / app_state['speed_multiplier']:
            return
        loop_state['last_step_time'] = now

        obs = loop_state['obs']
        state = get_state_from_pos(obs[0], obs[1], GRID_SIZE)
        action = agent.choose_action(state, app_state["epsilon"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        # CORRECCIÓN 3: Lógica de audio robusta para manejar la inconsistencia del entorno.
        if app_state['sound_enabled'] and 'play_sound' in info:
            try:
                sound_info = info['play_sound']
                sound_data_to_play = {}
                if isinstance(sound_info, dict):
                    sound_data_to_play = sound_info
                elif isinstance(sound_info, str):
                    sound_data_to_play = ast.literal_eval(
                        sound_info) if sound_info.startswith('{') else {'filename': sound_info}
                if sound_data_to_play:
                    play_sound(sound_data_to_play)
            except Exception as e:
                print(f"Error al procesar o reproducir el audio: {e}")

        next_state = get_state_from_pos(next_obs[0], next_obs[1], GRID_SIZE)
        agent.update(state, action, reward, next_state,
                     app_state["learning_rate"], app_state["discount_factor"])

        current_reward = app_state["current_episode_reward"] + reward
        app_state["current_episode_reward"] = round(current_reward, 2)
        loop_state['obs'] = next_obs
        loop_state['total_steps'] += 1

        if terminated or truncated:
            app_state["episodes_completed"] += 1
            app_state["reward_history"].append(current_reward)
            if len(app_state["reward_history"]) > app_state["chart_reward_number"]:
                app_state["reward_history"].pop(0)
            loop_state['obs'], _ = env.reset()
            app_state["current_episode_reward"] = 0
            if app_state["epsilon"] > app_state["min_epsilon"]:
                app_state["epsilon"] *= app_state["epsilon_decay"]

    def render_tick():
        env.unwrapped.set_render_data(q_table=agent.q_table)
        frame = env.render()
        app_state["preview_image"] = frame_to_base64_src(frame)
        if reward_chart is not None and reward_chart.visible:
            if list(reward_chart.options['series'][0]['data']) != app_state['reward_history']:
                reward_chart.options['series'][0]['data'] = app_state['reward_history'].copy(
                )
                reward_chart.update()

        now = time.time()
        elapsed = now - loop_state['last_check_time']
        if elapsed > 0.5:
            steps_this_interval = loop_state['total_steps'] - \
                loop_state['steps_at_last_check']
            sps = int(steps_this_interval / elapsed) if elapsed > 0 else 0
            app_state["steps_per_second"] = sps
            loop_state['last_check_time'] = now
            loop_state['steps_at_last_check'] = loop_state['total_steps']

    sim_timer = ui.timer(0.001, simulation_tick)
    render_timer = ui.timer(1/30, render_tick)
    active_timers.extend([sim_timer, render_timer])


app.on_startup(startup_handler)

# --- Ejecutar la Aplicación ---
ui.run(title="Visualizador de Q-Learning", dark=False, reload=True, show=False)

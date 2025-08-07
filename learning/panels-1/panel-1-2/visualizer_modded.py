# visualizer_modded.py
import random
import time
from nicegui import ui, app
import gymnasium as gym

# --- PASO 1: Importar el entorno, el agente y los helpers ---
try:
    import mlvlab
    from q_learning import QLearningAgent, get_state_from_pos
    from mlvlab.helpers.ng import setup_audio, create_reward_chart, frame_to_base64_src

    # NUEVO: Importamos los wrappers que acabamos de crear
    from wrappers import TimePenaltyWrapper, FoodVectorObservationWrapper

except ImportError:
    print("Error: El paquete 'mlvlab', sus wrappers, helpers o 'q_learning.py' no se encontraron.")
    exit()

# --- PASO 2: Configuración Inicial ---
GRID_SIZE = 10
# Ya no creamos 'env' aquí. Lo haremos en cada reset para poder cambiar los wrappers.
base_env = gym.make(
    "mlvlab/ant-v1", render_mode="rgb_array", grid_size=GRID_SIZE)
agent = QLearningAgent(num_states=GRID_SIZE * GRID_SIZE,
                       num_actions=base_env.action_space.n)
play_sound = setup_audio()

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
    "chart_visible": True,
    "chart_reward_number": 100,
    "sound_enabled": False,
    "preview_image": None,
    "steps_per_second": 0,

    # NUEVO: Estado para controlar qué wrappers están activos
    "use_penalty_wrapper": False,
    "use_vector_obs_wrapper": False,
}
reward_chart = None
env = None  # Se inicializará en reset_simulation


@ui.page('/')
def main_interface():
    global reward_chart, env

    loop_state = {  # ... (sin cambios)
        'obs': None, 'info': None, 'total_steps': 0, 'last_check_time': time.time(),
        'steps_at_last_check': 0, 'last_step_time': time.time(), 'current_seed': None,
    }

    def reset_simulation():
        """
        Reinicia la simulación. Ahora también reconstruye el entorno
        con los wrappers seleccionados en la UI.
        """
        global env

        # NUEVO: Construcción del entorno y los wrappers
        # 1. Siempre empezamos con el entorno base
        env = gym.make("mlvlab/ant-v1", render_mode="rgb_array",
                       grid_size=GRID_SIZE)

        # 2. Condicionalmente, envolvemos el entorno con los wrappers seleccionados
        if app_state["use_penalty_wrapper"]:
            env = TimePenaltyWrapper(env, penalty=-0.1)
            print("INFO: Wrapper de penalización por tiempo ACTIVADO.")

        if app_state["use_vector_obs_wrapper"]:
            env = FoodVectorObservationWrapper(env)
            print("INFO: Wrapper de observación de vector a comida ACTIVADO.")

        # El resto de la función de reinicio es casi idéntica
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
        print(f"Simulación reiniciada (Seed: {loop_state['current_seed']}).")

    reset_simulation()

    # --- Definición de la Interfaz de Usuario ---
    ui.label('Project 1.2: The Lost Ant Colony (Wrappers Demo)').classes(
        'w-full text-2xl font-bold text-center mt-4 mb-2')

    with ui.element('div').classes('w-full flex justify-center'):
        with ui.element('div').classes('w-full max-w-[1400px] flex flex-col lg:flex-row'):

            # --- COLUMNA IZQUIERDA: Controles y Configuración ---
            with ui.column().classes('w-full lg:w-1/4 pb-4 lg:pr-2 lg:pb-0'):

                # ... (Tarjeta de Controles de Simulación sin cambios) ...
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
                            ui.icon('play_arrow').bind_name_from(
                                app_state, 'command', lambda cmd: 'pause' if cmd == 'run' else 'play_arrow')
                        ui.button(on_click=lambda: app_state.update(
                            {"command": "reset"})).props('icon=refresh outline')
                        with ui.button(on_click=lambda: app_state.update({'sound_enabled': not app_state['sound_enabled']})).props('outline'):
                            ui.icon('volume_up').bind_name_from(
                                app_state, 'sound_enabled', lambda enabled: 'volume_up' if enabled else 'volume_off')

                # NUEVO: Tarjeta para seleccionar los wrappers
                with ui.card().classes('w-full mb-4'):
                    ui.label('Wrappers del Entorno').classes(
                        'text-lg font-semibold text-center w-full')
                    ui.markdown(
                        'Activa los wrappers y pulsa **Reiniciar** para aplicarlos.')
                    ui.switch('Penalización por tiempo (-0.1/paso)', value=False) \
                        .bind_value(app_state, 'use_penalty_wrapper')
                    ui.switch('Obs: Vector a comida', value=False) \
                        .bind_value(app_state, 'use_vector_obs_wrapper')

                # ... (Tarjeta de Configuración del Agente sin cambios) ...
                with ui.card().classes('w-full'):
                    ui.label('Configuración del Agente').classes(
                        'text-lg font-semibold text-center w-full mb-3')
                    with ui.grid(columns=3).classes('w-full gap-x-2 items-center'):
                        ui.label('Tasa de Aprendizaje (α):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['learning_rate'], format='%.2f', step=0.01, min=0, max=1) \
                            .bind_value(app_state, 'learning_rate') \
                            .bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')
                        ui.label('Factor de Descuento (γ):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['discount_factor'], format='%.2f', step=0.01, min=0, max=1) \
                            .bind_value(app_state, 'discount_factor') \
                            .bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')
                        ui.label('Decaimiento de Epsilon (ε):').classes(
                            'col-span-2 justify-self-start')
                        ui.number(value=app_state['epsilon_decay'], format='%.5f', step=0.00001, min=0, max=1) \
                            .bind_value(app_state, 'epsilon_decay') \
                            .bind_enabled_from(app_state, 'command', lambda cmd: cmd != 'run')

            # --- COLUMNA CENTRAL Y DERECHA (sin cambios) ---
            with ui.column().classes('w-full lg:w-2/4 pb-4 lg:pb-0 lg:px-2 items-center'):
                with ui.card().classes('w-full p-0 bg-black flex justify-center items-center'):
                    preview_image = ui.image().props(
                        'no-transition').classes('max-w-[550px]')
                    preview_image.bind_source_from(app_state, 'preview_image')
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

    # --- Bucle de Simulación ---
    def simulation_tick():
        if app_state["command"] == "reset":
            reset_simulation()
            return
        if app_state["command"] != "run" or env is None:
            return

        # ... El resto del bucle de simulación tiene una pequeña modificación ...
        obs = loop_state['obs']

        # IMPORTANTE: Aquí hay una incompatibilidad deliberada para enseñar un concepto.
        # Nuestro QLearningAgent espera un estado discreto (un número de 0 a 99) que
        # obtiene de la posición (x,y). Pero el FoodVectorObservationWrapper devuelve un
        # vector continuo. ¡No son compatibles!
        # Un agente más avanzado (como una Red Neuronal) podría manejar este vector.
        # Por ahora, si el wrapper de vector está activado, el agente no funcionará correctamente.
        if app_state["use_vector_obs_wrapper"]:
            # Si estuviéramos usando un agente avanzado, usaríamos 'obs' directamente.
            # Como no lo tenemos, el agente se comportará de forma aleatoria porque
            # no sabrá cómo interpretar este nuevo tipo de estado.
            state = -1  # Estado inválido para forzar exploración
        else:
            # Con el wrapper desactivado, todo funciona como antes.
            state = get_state_from_pos(obs[0], obs[1], GRID_SIZE)

        action = agent.choose_action(state, app_state["epsilon"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        if app_state['sound_enabled'] and 'play_sound' in info:
            play_sound(info['play_sound'])

        if not app_state["use_vector_obs_wrapper"]:
            next_state = get_state_from_pos(
                next_obs[0], next_obs[1], GRID_SIZE)
            agent.update(state, action, reward, next_state,
                         app_state["learning_rate"], app_state["discount_factor"])

        # ... (el resto de la lógica de actualización de recompensas y estado no cambia) ...
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

    # --- Bucle de Renderizado (sin cambios) ---
    def render_tick():
        if env is None:
            return
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

    ui.timer(0.001, simulation_tick)
    ui.timer(1/30, render_tick)


# --- Ejecutar la Aplicación ---
ui.run(title="Visualizador de Q-Learning con Wrappers", dark=False, reload=True)

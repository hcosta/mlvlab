# train_visualizer.py
import random
import time
import ast  # Necesario para la lógica de audio robusta
from threading import Lock, Thread
from nicegui import ui, app
import asyncio
from starlette.websockets import WebSocket
try:
    # Para tipado y callbacks de desconexión por cliente
    from nicegui import Client  # type: ignore
except Exception:
    Client = object  # fallback
import gymnasium as gym

# --- PASO 1: Importar el entorno, el agente y los helpers ---
try:
    import mlvlab
    from q_learning import QLearningAgent, get_state_from_pos
    # Asumiendo que los helpers están en la ubicación recomendada
    from mlvlab.helpers.ng import setup_audio, create_reward_chart, frame_to_webp_bytes
except ImportError:
    print("Error: El paquete 'mlvlab', sus helpers o 'q_learning.py' no se encontraron.")
    exit()

# --- PASO 2: Configuración Inicial ---

# Estado global del servidor para una única seed/mapa compartido
GLOBAL_SEED = None
SIM_INITIALIZED = False

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
    "turbo_mode": False,
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

# Simulación global compartida por todos los clientes
SIM = {
    'initialized': False,
    "seed": None,
    'obs': None,
    'info': None,
    'total_steps': 0,
    'last_check_time': time.time(),
    'steps_at_last_check': 0,
    'last_step_time': time.time(),
    'step_accum': 0.0,
    'current_episode_reward': 0.0,
    'last_sound': None,
    'last_turbo': False,
    'stop': False,
    'thread': None,
}
ENV_LOCK = Lock()


def startup_handler():
    """Inicializa recursos globales como el audio una sola vez."""
    global play_sound
    play_sound = setup_audio()
    # Inicializar simulación global si no está
    if not SIM['initialized']:
        SIM['seed'] = SIM['seed'] or random.randint(0, 1_000)
        with ENV_LOCK:
            SIM['obs'], SIM['info'] = env.reset(seed=SIM['seed'])
        SIM['initialized'] = True
        SIM['last_check_time'] = time.time()
        SIM['steps_at_last_check'] = 0
        SIM['last_step_time'] = time.time()
    # Bucle global de simulación (único) en hilo dedicado

    def simulation_loop():
        # usar perf_counter para más precisión en dt
        SIM['last_step_time'] = time.perf_counter()
        while not SIM['stop']:
            cmd = app_state.get('command')
            if cmd == 'pause':
                time.sleep(0.001)
                continue
            if cmd == 'reset':
                new_seed = random.randint(0, 1_000)
                SIM['seed'] = new_seed
                with ENV_LOCK:
                    SIM['obs'], SIM['info'] = env.reset(seed=new_seed)
                SIM['current_episode_reward'] = 0.0
                SIM['total_steps'] = 0
                SIM['step_accum'] = 0.0
                agent.reset()
                app_state.update({
                    'episodes_completed': 0,
                    'reward_history': [],
                    'steps_per_second': 0,
                    'epsilon': 1.0,
                    'current_episode_reward': 0,
                    'command': 'run',
                })
                continue

            now = time.perf_counter()
            dt = now - SIM['last_step_time']
            turbo = bool(app_state.get('turbo_mode'))
            if turbo != SIM['last_turbo']:
                SIM['step_accum'] = 0.0
                SIM['last_step_time'] = now
                SIM['last_turbo'] = turbo
                continue
            SIM['last_step_time'] = now

            spm = max(0, int(app_state['speed_multiplier']))
            if turbo:
                steps_to_do = max(
                    1, min(120000, int(30000 * dt) if dt > 0 else 10000))
            else:
                SIM['step_accum'] += spm * dt
                steps_to_do = int(SIM['step_accum'])
            if steps_to_do <= 0:
                # ceder CPU para no monopolizar
                time.sleep(0.0005)
                continue
            steps_to_do = min(steps_to_do, 40000)

            # micro-batching
            for _ in range(steps_to_do):
                if not turbo:
                    SIM['step_accum'] -= 1.0
                with ENV_LOCK:
                    obs = SIM['obs']
                    state = get_state_from_pos(obs[0], obs[1], GRID_SIZE)
                    action = agent.choose_action(state, app_state['epsilon'])
                    next_obs, reward, terminated, truncated, info = env.step(
                        action)
                    SIM['obs'] = next_obs
                    SIM['info'] = info
                next_state = get_state_from_pos(
                    next_obs[0], next_obs[1], GRID_SIZE)
                agent.update(state, action, reward, next_state,
                             app_state['learning_rate'], app_state['discount_factor'])
                SIM['current_episode_reward'] = round(
                    SIM['current_episode_reward'] + reward, 2)
                SIM['total_steps'] += 1
                if 'play_sound' in info and (app_state['speed_multiplier'] <= 50) and not turbo:
                    SIM['last_sound'] = info['play_sound']
                if terminated or truncated:
                    app_state['episodes_completed'] += 1
                    app_state['reward_history'].append(
                        SIM['current_episode_reward'])
                    if len(app_state['reward_history']) > app_state['chart_reward_number']:
                        app_state['reward_history'].pop(0)
                    with ENV_LOCK:
                        SIM['obs'], SIM['info'] = env.reset()
                    SIM['current_episode_reward'] = 0.0
                    if app_state['epsilon'] > app_state['min_epsilon']:
                        app_state['epsilon'] *= app_state['epsilon_decay']
                    break
            # breve sleep para dar aire a la UI
            time.sleep(0.0005)

    SIM['stop'] = False
    SIM['thread'] = Thread(target=simulation_loop, daemon=True)
    SIM['thread'].start()

    # Parada limpia al cerrar servidor
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
def main_interface(client: Client):
    global reward_chart

    # Estado mínimo por cliente
    loop_state = {
        'disconnected': False,
        'last_check_time': time.time(),
        'steps_at_last_check': SIM['total_steps'],
    }

    def reset_simulation():
        """Reinicia la simulación sin cambiar la seed global una vez fijada."""
        global GLOBAL_SEED, SIM_INITIALIZED
        if not SIM_INITIALIZED:
            if GLOBAL_SEED is None:
                GLOBAL_SEED = random.randint(0, 1_000)
            obs, info = env.reset(seed=GLOBAL_SEED)
            loop_state['current_seed'] = GLOBAL_SEED
            SIM_INITIALIZED = True
        else:
            # Reset sin semilla para preservar el mapa actual
            obs, info = env.reset()
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

    # Cancelar timers y marcar desconexión cuando el cliente se vaya
    def _on_disconnect():
        loop_state['disconnected'] = True
        for timer in active_timers:
            try:
                timer.cancel()
            except Exception:
                pass

    try:
        client.on_disconnect(_on_disconnect)
    except Exception:
        pass

    # --- NUEVA LÓGICA DE DIÁLOGO Y CIERRE ---
    with ui.dialog() as dialog, ui.card():
        ui.label('¿Estás seguro de que quieres cerrar la aplicación?')
        with ui.row().classes('w-full justify-end'):
            # El botón "Sí" ejecuta la lógica de cierre que teníamos antes
            def do_shutdown():
                print("Confirmado. Iniciando cierre natural...")

                # 1) Cancelar timers de esta pestaña
                for timer in active_timers:
                    try:
                        timer.cancel()
                    except Exception:
                        pass

                # 2) Señal cross-tab: BroadcastChannel + localStorage
                ui.run_javascript("""
                (() => {
                  const nonce = String(Date.now()) + '-' + Math.random().toString(36).slice(2);
                  try { new BroadcastChannel('mlvlab-shutdown').postMessage('shutdown'); } catch (e) {}
                  try {
                    localStorage.setItem('mlvlab_shutdown_signal', nonce);
                    setTimeout(() => { try { localStorage.removeItem('mlvlab_shutdown_signal'); } catch (e) {} }, 200);
                  } catch (e) {}
                  try { window.close(); } catch (e) {}
                  try { location.replace('about:blank'); } catch (e) {}
                })();
                """)

                # 3) Detener bucle de simulación global
                try:
                    SIM['stop'] = True
                    t = SIM.get('thread')
                    if t and t.is_alive():
                        t.join(timeout=1.0)
                except Exception:
                    pass

                # 4) Apagar el servidor tras breve margen para que la señal llegue
                ui.timer(0.3, lambda: app.shutdown())

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

    # No reiniciar la simulación al abrir nueva pestaña; la simulación es global

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
                    with ui.grid(columns=2).classes('w-full items-center gap-2'):
                        ui.slider(min=1, max=50, step=1).bind_value(
                            app_state, 'speed_multiplier').classes('w-full')
                        ui.switch('Turbo Mode').bind_value(
                            app_state, 'turbo_mode').classes('w-full')

                    with ui.row().classes('w-full justify-around mt-3 items-center'):
                        def toggle_simulation():
                            app_state['command'] = "pause" if app_state['command'] == "run" else "run"
                        with ui.button(on_click=toggle_simulation).props('outline'):
                            ui.icon('pause').bind_name_from(
                                app_state, 'command', lambda cmd: 'pause' if cmd == 'run' else 'play_arrow')

                        ui.button(on_click=lambda: app_state.update(
                            {"command": "reset"})).props('icon=refresh outline')

                        with ui.button(on_click=lambda: app_state.update({'sound_enabled': not app_state['sound_enabled']})).props('outline'):
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
                    # Canvas para recibir frames por WebSocket binario (WebP)
                    canvas = ui.element('canvas').classes(
                        'max-w-[550px]').style('width: 100%; height: auto;')
                    canvas.props('id=viz_canvas width=900 height=900')

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

    # Eliminamos simulación por cliente: solo render

    def render_tick():
        if loop_state.get('disconnected'):
            return
        env.unwrapped.set_render_data(q_table=agent.q_table)
        with ENV_LOCK:
            # Ya no empujamos PNG/base64 por app_state: los frames irán por WS binario
            pending_sound = SIM.get('last_sound')
            SIM['last_sound'] = None
        # Actualizar gráfico si es necesario
        if reward_chart is not None and reward_chart.visible:
            if list(reward_chart.options['series'][0]['data']) != app_state['reward_history']:
                reward_chart.options['series'][0]['data'] = app_state['reward_history'].copy(
                )
                reward_chart.update()
        # Reproducir sonido si aplica
        if pending_sound and app_state['sound_enabled'] and app_state['speed_multiplier'] <= 50:
            if client and getattr(client, 'connected', True):
                try:
                    play_sound(pending_sound)
                except Exception:
                    pass

        now = time.time()
        elapsed = now - loop_state['last_check_time']
        if elapsed > 0.5:
            steps_this_interval = SIM['total_steps'] - \
                loop_state['steps_at_last_check']
            sps = int(steps_this_interval / elapsed) if elapsed > 0 else 0
            app_state["steps_per_second"] = sps
            loop_state['last_check_time'] = now
            loop_state['steps_at_last_check'] = SIM['total_steps']

    # Reducimos la frecuencia de render a 15 Hz para el resto de UI; frames irán por WS
    render_timer = ui.timer(1/15, render_tick)
    active_timers.extend([render_timer])

    # WebSocket binario para enviar frames WebP a cada cliente
    route = f"/ws/frame/{id(client)}"

    async def frame_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                with ENV_LOCK:
                    env.unwrapped.set_render_data(q_table=agent.q_table)
                    frame = env.render()
                webp = frame_to_webp_bytes(frame, quality=100)
                await websocket.send_bytes(webp)
                await asyncio.sleep(1/25)  # ~25 FPS
        except Exception:
            pass

    try:
        app.add_websocket_route(route, frame_ws)
    except Exception:
        # Fallback por compatibilidad
        pass

    # JS para consumir el WS y pintar en canvas
    ui.run_javascript(f"""
    (() => {{
      const canvas = document.getElementById('viz_canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      let url = (location.origin.replace(/^http/, 'ws')) + '{route}';
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      ws.onmessage = async (ev) => {{
        const blob = new Blob([ev.data], {{ type: 'image/webp' }});
        const img = new Image();
        img.onload = () => {{
          try {{
            canvas.width = img.width; canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
          }} catch (e) {{}}
        }};
        img.src = URL.createObjectURL(blob);
      }};
    }})();
    """)

    ui.run_javascript("""
(() => {
  if (window.__mlvlabShutdownInit) return;
  window.__mlvlabShutdownInit = true;

  const closeSelf = () => {
    try { window.close(); } catch (e) {}
    try { location.replace('about:blank'); } catch (e) {}
    try { location.href = 'about:blank'; } catch (e) {}
  };

  // BroadcastChannel
  try {
    const bc = new BroadcastChannel('mlvlab-shutdown');
    bc.onmessage = ev => { if (ev && ev.data === 'shutdown') closeSelf(); };
  } catch (e) {}

  // Evento de localStorage (solo evento, nada de leer valor inicial)
  window.addEventListener('storage', (e) => {
    if (e.key === 'mlvlab_shutdown_signal') closeSelf();
  });
})();
""")


app.on_startup(startup_handler)

# --- Ejecutar la Aplicación ---
ui.run(title="Visualizador de Q-Learning", dark=False, reload=True, show=False)

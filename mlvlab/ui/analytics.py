# mlvlab/ui/analytics.py

from __future__ import annotations

from typing import Any, List, Optional, Callable, Dict
import time
import threading
import asyncio
from pathlib import Path
import importlib.util
import sys

# CAMBIO CLAVE: Importar webview y otras utilidades
from nicegui import ui, app, Client
import numpy as np
from starlette.responses import StreamingResponse
from starlette.requests import Request
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import QUrl

# Asumo que estos imports son de tu proyecto y correctos
from .state import StateStore
from .runtime import SimulationRunner
from mlvlab.core.trainer import Trainer
from .components.base import ComponentContext, UIComponent
from mlvlab.helpers.ng import setup_audio, encode_frame_fast_jpeg, create_reward_chart

# Variables globales para gestionar los hilos y el apagado
_ACTIVE_THREADS: Dict[str, Any] = {
    "renderer": None,
    "runner": None,
    "stream_tasks": {},
}
# Evento para sincronizar el inicio del servidor y la ventana
_server_started = threading.Event()


class FrameBuffer:
    """Buffer Thread-Safe para pasar frames desde el hilo de renderizado (Thread) al servidor (Asyncio)."""

    def __init__(self):
        self.current_frame = b""
        self.lock = threading.Lock()
        self.new_frame_event = asyncio.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def update_frame(self, frame_bytes: bytes):
        with self.lock:
            self.current_frame = frame_bytes
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.new_frame_event.set)

    async def get_next_frame(self):
        """Espera y obtiene el siguiente fotograma (ejecutado en el bucle de eventos Asyncio)."""
        await self.new_frame_event.wait()
        self.new_frame_event.clear()
        with self.lock:
            return self.current_frame


class RenderingThread(threading.Thread):
    """Hilo dedicado para renderizar el entorno a la tasa de FPS objetivo."""

    def __init__(self, env, agent, env_lock, buffer: FrameBuffer, state: StateStore, target_fps=60):
        super().__init__(daemon=True)
        self.env = env
        self.agent = agent
        self.env_lock = env_lock
        self.buffer = buffer
        self.state = state
        self.target_fps = max(1, target_fps)
        self.interval = 1.0 / self.target_fps
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        print(f"▶️ Hilo de Renderizado [ID: {self.ident}] iniciado.")
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            try:
                # Sincronizar el debug_mode desde el estado de la UI
                debug_is_on = bool(self.state.get(["ui", "debug_mode"]))

                with self.env_lock:
                    # --- INICIO: CÓDIGO MODIFICADO PARA SER AGNÓSTICO ---
                    env_unwrapped = self.env.unwrapped

                    # 1. Establecer modo debug solo si el entorno lo soporta
                    if hasattr(env_unwrapped, 'debug_mode'):
                        setattr(env_unwrapped, 'debug_mode', debug_is_on)

                    # 2. Pasar datos de renderizado solo si el método existe
                    if hasattr(env_unwrapped, 'set_render_data'):
                        # Preparamos un diccionario de datos para ser flexible
                        render_data = {}

                        # Añadir q_table solo si el agente la tiene
                        if hasattr(self.agent, 'q_table'):
                            render_data['q_table'] = getattr(
                                self.agent, 'q_table', None)

                        # (Ejemplo futuro) Puedo añadir datos de una red neuronal aquí
                        # if hasattr(self.agent, 'policy_network'):
                        #     render_data['policy'] = self.agent.policy_network

                        # Llamar al método solo si hay datos que pasar
                        if render_data:
                            try:
                                # Usamos ** para pasar los datos como argumentos nombrados
                                env_unwrapped.set_render_data(**render_data)
                            except Exception:
                                # Fallback por si el método no acepta kwargs
                                env_unwrapped.set_render_data(
                                    render_data.get('q_table'))

                    frame_np = self.env.render()

                    try:
                        last_step = int(self.state.get(
                            ["sim", "total_steps"]) or 0)
                        self.state.set(["ui", "last_frame_step"], last_step)
                    except Exception:
                        pass

                # A 100 de calidad, aunque parezca estúpido, es lo más rapido posible
                frame_bytes = encode_frame_fast_jpeg(frame_np, quality=100)
                if frame_bytes:
                    self.buffer.update_frame(frame_bytes)

            except Exception as e:
                print(
                    f"⚠️ Error en el hilo de renderizado [ID: {self.ident}]: {e}")
                self._stop_event.wait(0.1)
                continue

            elapsed = time.perf_counter() - start_time
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

        print(f"⏹️ Hilo de Renderizado [ID: {self.ident}] detenido.")


class AnalyticsView:
    """Vista principal declarativa que arma el panel de análisis estándar."""

    def __init__(
        self,
        env: Any | None = None, agent: Any | None = None, trainer: Trainer | None = None,
        left_panel_components: Optional[List[UIComponent]] = None,
        right_panel_components: Optional[List[UIComponent]] = None,
        title: str = "", history_size: int = 100, dark: bool = False,
        subtitle: Optional[str] = None, state_from_obs: Optional[Callable[..., Any]] = None,
        agent_hparams_defaults: Optional[dict] = None,
    ) -> None:
        self._trainer: Trainer | None = trainer
        if self._trainer is not None:
            self.env, self.agent = self._trainer.env, self._trainer.agent
        else:
            if env is None or agent is None:
                raise ValueError(
                    "Debes proporcionar 'trainer' o bien 'env' y 'agent'.")
            self.env, self.agent = env, agent

        self.left_components = left_panel_components or []
        self.right_components = right_panel_components or []
        self.title = title
        self.history_size = history_size
        self.dark = dark
        self.subtitle = subtitle
        self.env_lock = threading.Lock()
        self.user_hparams = agent_hparams_defaults or {}

        agent_defaults = {
            'epsilon': float(getattr(self.agent, 'epsilon', 1.0) or 1.0),
            'epsilon_decay': float(getattr(self.agent, 'epsilon_decay', 0.99) or 0.99),
            'min_epsilon': float(getattr(self.agent, 'min_epsilon', 0.1) or 0.1),
            'learning_rate': float(getattr(self.agent, 'learning_rate', 0.1) or 0.1),
            'discount_factor': float(getattr(self.agent, 'discount_factor', 0.9) or 0.9),
        }

        self.state = StateStore(
            defaults={
                "sim": {"command": "run", "speed_multiplier": 1, "turbo_mode": False, "total_steps": 0, "current_episode_reward": 0.0},
                "agent": {**agent_defaults, **{k: float(v) for k, v in self.user_hparams.items()}},
                "metrics": {"episodes_completed": 0, "reward_history": [], "steps_per_second": 0, "chart_reward_number": history_size},
                "ui": {"sound_enabled": True, "chart_visible": True, "debug_mode": False},
            }
        )

        trainer_adapter = getattr(
            self._trainer, 'state_from_obs', None) if self._trainer is not None else None

        def _auto_resolve_adapter():
            try:
                env_id = getattr(getattr(self.env, 'spec', None), 'id', '')
                env_pkg = env_id.split('/')[-1] if '/' in env_id else env_id
                env_pkg_us = env_pkg.replace('-', '_')
                mod = None
                try:
                    module_path_env = f"mlvlab.envs.{env_pkg_us}.adapters"
                    mod = importlib.import_module(module_path_env)
                except Exception:
                    mod = None
                if mod is None:
                    try:
                        module_path_agents = f"mlvlab.agents.{env_pkg}.state"
                        mod = importlib.import_module(module_path_agents)
                    except Exception:
                        base_dir = Path(__file__).resolve(
                        ).parents[1] / 'agents' / env_pkg
                        file_path = base_dir / 'state.py'
                        if not file_path.exists():
                            return None
                        spec = importlib.util.spec_from_file_location(
                            "mlvlab_env_state_module_ui", str(file_path))
                        if spec is None or spec.loader is None:
                            return None
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                fn = getattr(mod, 'obs_to_state', None)
                if callable(fn):
                    return lambda obs: fn(obs, self.env)
            except Exception:
                return None
            return None
        raw_fn = trainer_adapter if callable(trainer_adapter) else (
            state_from_obs if callable(state_from_obs) else _auto_resolve_adapter())
        if not callable(raw_fn):
            raw_fn = getattr(self.agent, 'extract_state_from_obs', None)
        adapted_fn = self._build_state_from_obs_adapter(
            raw_fn) if callable(raw_fn) else None

        self.runner = SimulationRunner(
            env=self.env, agent=self.agent, state=self.state, env_lock=self.env_lock, state_from_obs=adapted_fn)
        self.frame_buffer = FrameBuffer()
        self.target_fps = getattr(
            self.env, 'metadata', {}).get("render_fps", 60)
        self._reward_chart = None

    def _build_state_from_obs_adapter(self, fn: Callable[..., Any]) -> Callable[[Any], Any]:
        try:
            from inspect import signature
            sig = signature(fn)
            num_params = len(sig.parameters)
        except Exception:
            num_params = 1
        grid_size = getattr(getattr(self.env, 'unwrapped',
                            self.env), 'GRID_SIZE', None)

        def adapter(obs: Any) -> Any:
            try:
                if isinstance(obs, (int, np.integer)):
                    return int(obs)
                if num_params <= 1:
                    return fn(obs)
                if hasattr(obs, '__getitem__'):
                    try:
                        if len(obs) >= 2:
                            x, y = int(obs[0]), int(obs[1])
                        else:
                            return obs
                    except Exception:
                        return obs
                else:
                    return obs
                if num_params == 2:
                    return fn(x, y)
                if num_params >= 3 and grid_size is not None:
                    return fn(x, y, int(grid_size))
                return fn(obs)
            except Exception:
                return obs
        return adapter

    def _build_page(self) -> None:
        @app.get('/video_feed/{client_id}')
        async def video_feed(request: Request, client_id: str):
            async def mjpeg_stream_generator():
                print(f"🔌 Cliente {client_id} conectado al stream de video.")
                task = asyncio.current_task()
                _ACTIVE_THREADS["stream_tasks"][client_id] = task
                try:
                    while True:
                        if await request.is_disconnected():
                            print(
                                f"🔌 Cliente {client_id} desconectado (detectado por el servidor).")
                            break

                        try:
                            frame = await asyncio.wait_for(
                                self.frame_buffer.get_next_frame(),
                                timeout=1.0
                            )
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' +
                                   frame + b'\r\n')
                            await asyncio.sleep(0.001)
                        except asyncio.TimeoutError:
                            continue
                except asyncio.CancelledError:
                    print(
                        f"🔌 Stream task para cliente {client_id} fue cancelada explícitamente.")
                finally:
                    _ACTIVE_THREADS["stream_tasks"].pop(client_id, None)
                    print(
                        f"🔌 Stream de video para cliente {client_id} detenido limpiamente.")
            return StreamingResponse(mjpeg_stream_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

        @ui.page("/")
        def main_page(client: Client):
            video_endpoint = f'/video_feed/{client.id}'

            app.storage.client['reconnect_timeout'] = 3.0
            context = ComponentContext(
                state=self.state, env_lock=self.env_lock)
            play_sound = setup_audio(self.env)

            if self.title:
                ui.label("MLVLab - " + self.title).classes(
                    'w-full text-2xl font-bold text-center mt-4 mb-1')
            if self.subtitle:
                ui.label(self.subtitle).classes(
                    'w-full text-base text-center mb-2 opacity-80')

            with ui.element('div').classes('w-full flex justify-center'):
                with ui.element('div').classes('w-full max-w-[1400px] flex flex-col lg:flex-row'):
                    with ui.column().classes('w-full lg:w-1/4 pb-4 lg:pr-2 lg:pb-0'):
                        for component in self.left_components:
                            component.render(self.state, context)
                    with ui.column().classes('w-full lg:w-2/4 pb-4 lg:pb-0 lg:px-2 items-center'):
                        ui.image(video_endpoint).classes(
                            'w-full h-auto border rounded shadow-lg bg-gray-200')
                    with ui.column().classes('w-full lg:w-1/4 lg:pl-2'):
                        for component in self.right_components:
                            component.render(self.state, context)

            def render_tick() -> None:
                try:
                    pending_sound = self.state.get(["sim", "last_sound"])
                    if pending_sound and self.state.get(["ui", "sound_enabled"]):
                        evt_step = int(pending_sound.get("step", -1))
                        last_frame_step = int(self.state.get(
                            ["ui", "last_frame_step"]) or -1)
                        if evt_step < 0 or evt_step <= last_frame_step:
                            play_sound(pending_sound)
                            self.state.set(["sim", "last_sound"], None)
                except Exception:
                    pass
            context.register_timer(ui.timer(1/15, render_tick))

        _ = create_reward_chart

    # =========================================================================
    # MÉTODO RUN REFACTORIZADO CON PYSIDE6
    # =========================================================================
    def run(self, host='127.0.0.1', port=8181) -> None:
        """Arranca la app NiceGUI en un hilo y la muestra en una ventana nativa con PySide6."""

        def run_nicegui():
            """Función que se ejecutará en un hilo para correr el servidor NiceGUI."""
            self._build_page()
            ui.run(
                host=host,
                port=port,
                title=self.title,
                dark=self.dark,
                reload=False,
                show=False,  # No queremos que NiceGUI abra un navegador
                native=False,
            )

        # 1. Registrar manejadores de ciclo de vida de NiceGUI (sin cambios)
        @app.on_startup
        def startup_handler():
            print("\n--- INICIANDO APLICACIÓN (on_startup) ---")
            try:
                loop = asyncio.get_running_loop()
                self.frame_buffer.set_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            print("▶️ Creando nuevos hilos de simulación/renderizado...")
            _ACTIVE_THREADS["renderer"] = RenderingThread(
                env=self.env, agent=self.agent, env_lock=self.env_lock,
                buffer=self.frame_buffer, state=self.state, target_fps=self.target_fps
            )
            _ACTIVE_THREADS["runner"] = self.runner
            _ACTIVE_THREADS["renderer"].start()
            _ACTIVE_THREADS["runner"].start()
            print("✅ Startup de hilos completado.")
            _server_started.set()

        @app.on_shutdown
        def shutdown_handler():
            """Función robusta para detener los hilos y tareas de forma segura."""
            print("--- DETENIENDO APLICACIÓN (on_shutdown) ---")

            # Define la corrutina que debe ejecutarse en el bucle de eventos
            async def cancel_streams():
                tasks_to_cancel = list(
                    _ACTIVE_THREADS["stream_tasks"].values())
                if tasks_to_cancel:
                    print(
                        f"... Cancelando {len(tasks_to_cancel)} tarea(s) de stream...")
                    for task in tasks_to_cancel:
                        task.cancel()
                    # Espera a que todas las tareas de cancelación finalicen
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                _ACTIVE_THREADS["stream_tasks"].clear()

            # Usa el método seguro para ejecutar una corrutina desde un contexto síncrono
            try:
                loop = asyncio.get_running_loop()
                # Envía la corrutina al bucle en ejecución y obtiene un 'futuro'
                future = asyncio.run_coroutine_threadsafe(
                    cancel_streams(), loop)
                # Espera a que el futuro se complete con un timeout
                future.result(timeout=2.0)
                print("... Tareas de stream canceladas de forma segura.")
            except (RuntimeError, TimeoutError) as e:
                # Si no se puede obtener el bucle o hay un timeout, informa del error
                print(
                    f"⚠️ No se pudieron cancelar las tareas de stream limpiamente: {e}")

            # Ahora, el resto del código síncrono se ejecuta como antes
            renderer = _ACTIVE_THREADS.get("renderer")
            runner = _ACTIVE_THREADS.get("runner")

            print("... Deteniendo hilos de renderizado y simulación...")
            if isinstance(renderer, threading.Thread) and renderer.is_alive():
                renderer.stop()
            if isinstance(runner, threading.Thread) and hasattr(runner, 'stop') and runner.is_alive():
                runner.stop()

            if isinstance(renderer, threading.Thread) and renderer.is_alive():
                renderer.join(timeout=1.0)
            if isinstance(runner, threading.Thread) and runner.is_alive():
                runner.join(timeout=1.0)

            _ACTIVE_THREADS["renderer"] = None
            _ACTIVE_THREADS["runner"] = None
            print("✅ Limpieza de shutdown completada.")

        # 2. Iniciar el servidor NiceGUI en un hilo separado (sin cambios)
        nicegui_thread = threading.Thread(target=run_nicegui, daemon=True)
        nicegui_thread.start()

        # 3. Esperar a que el servidor esté listo (sin cambios)
        _server_started.wait()

        # 4. Crear y mostrar la ventana nativa con PySide6
        qt_app = QApplication(sys.argv)
        url = QUrl(f"http://{host}:{port}")

        class MainWindow(QMainWindow):
            def closeEvent(self, event):
                """Sobrescribe el evento de cierre para apagar NiceGUI."""
                print(
                    "Ventana nativa cerrándose. Solicitando apagado de la aplicación...")
                app.shutdown()
                event.accept()

        window = MainWindow()
        window.setWindowTitle("MLVLab Analytics Panel - " + self.env.spec.id)
        window.setGeometry(100, 100, 1280, 800)  # x, y, ancho, alto

        # Crear el widget de vista web
        web_view = QWebEngineView()
        # Permitir la reproducción de audio sin interacción del usuario
        web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        web_view.setUrl(url)

        # Establecer la vista web como el widget central de la ventana
        window.setCentralWidget(web_view)
        window.showMaximized()

        print(
            f"🚀 Mostrando ventana nativa con PySide6. Cargando {url.toString()}...")

        # 5. Ejecutar la aplicación Qt (esto es un bucle bloqueante)
        sys.exit(qt_app.exec())

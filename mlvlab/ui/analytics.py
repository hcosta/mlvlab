# mlvlab/ui/analytics.py

from __future__ import annotations

from typing import Any, List, Optional, Callable, Dict
import time
import threading
import asyncio
from pathlib import Path
import importlib.util
import sys

from nicegui import ui, app, Client
import numpy as np
from starlette.responses import StreamingResponse
from starlette.requests import Request
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtCore import QUrl

from .state import StateStore
from .runtime import SimulationRunner
from mlvlab.core.trainer import Trainer
from .components.base import ComponentContext, UIComponent
from mlvlab.helpers.ng import setup_audio, encode_frame_fast_jpeg, create_reward_chart

_ACTIVE_THREADS: Dict[str, Any] = {
    "renderer": None,
    "runner": None,
    "stream_tasks": {},
}
_server_started = threading.Event()


class FrameBuffer:
    """Buffer Thread-Safe para pasar frames desde el hilo de renderizado al servidor."""

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
        await self.new_frame_event.wait()
        self.new_frame_event.clear()
        with self.lock:
            return self.current_frame


class RenderingThread(threading.Thread):
    """Hilo dedicado para renderizar el entorno tan rápido como sea posible."""

    def __init__(self, env, agent, env_lock, buffer: FrameBuffer, state: StateStore):
        super().__init__(daemon=True)
        self.env = env
        self.agent = agent
        self.env_lock = env_lock
        self.buffer = buffer
        self.state = state
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        print(f"▶️ Hilo de Renderizado [ID: {self.ident}] iniciado.")
        while not self._stop_event.is_set():
            # El bucle ahora intentará ejecutarse tan rápido como pueda,
            # limitado únicamente por el tiempo que tarda en renderizar un fotograma.
            try:
                debug_is_on = bool(self.state.get(["ui", "debug_mode"]))
                with self.env_lock:
                    env_unwrapped = self.env.unwrapped
                    if hasattr(env_unwrapped, 'debug_mode'):
                        setattr(env_unwrapped, 'debug_mode', debug_is_on)
                    if hasattr(env_unwrapped, 'set_render_data'):
                        render_data = {}
                        if hasattr(self.agent, 'q_table'):
                            render_data['q_table'] = getattr(
                                self.agent, 'q_table', None)
                        if render_data:
                            try:
                                env_unwrapped.set_render_data(**render_data)
                            except Exception:
                                env_unwrapped.set_render_data(
                                    render_data.get('q_table'))
                    frame_np = self.env.render()
                    try:
                        last_step = int(self.state.get(
                            ["sim", "total_steps"]) or 0)
                        self.state.set(["ui", "last_frame_step"], last_step)
                    except Exception:
                        pass

                frame_bytes = encode_frame_fast_jpeg(frame_np, quality=100)
                if frame_bytes:
                    self.buffer.update_frame(frame_bytes)

            except Exception as e:
                print(
                    f"⚠️ Error en el hilo de renderizado [ID: {self.ident}]: {e}")
                # Pequeña pausa en caso de error continuo
                self._stop_event.wait(0.1)
                continue

            # Damos una mínima oportunidad a otros hilos de ejecutarse para no saturar la CPU al 100%
            time.sleep(0.001)

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

        chart_component = next((c for c in (left_panel_components or [
        ]) + (right_panel_components or []) if hasattr(c, 'history_size')), None)
        effective_history_size = chart_component.history_size if chart_component else history_size

        self.state = StateStore(
            defaults={
                "sim": {"command": "run", "speed_multiplier": 2, "turbo_mode": False, "total_steps": 0, "current_episode_reward": 0.0},
                "agent": {**agent_defaults, **{k: float(v) for k, v in self.user_hparams.items()}},
                "metrics": {"episodes_completed": 0, "reward_history": [], "steps_per_second": 0, "chart_reward_number": effective_history_size},
                "ui": {"sound_enabled": True, "chart_visible": True, "debug_mode": False, "dark_mode": dark},
            }
        )

        if hasattr(self.env.unwrapped, 'set_state_store'):
            self.env.unwrapped.set_state_store(self.state)

        self.runner = SimulationRunner(
            trainer=self._trainer, state=self.state,  env_lock=self.env_lock,
        )
        self.frame_buffer = FrameBuffer()
        self._reward_chart = None

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
                            break
                        try:
                            frame = await asyncio.wait_for(self.frame_buffer.get_next_frame(), timeout=1.0)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' +
                                   frame + b'\r\n')
                            await asyncio.sleep(0.001)
                        except asyncio.TimeoutError:
                            continue
                except asyncio.CancelledError:
                    pass
                finally:
                    _ACTIVE_THREADS["stream_tasks"].pop(client_id, None)
            return StreamingResponse(mjpeg_stream_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

        @ui.page("/")
        def main_page(client: Client):
            video_endpoint = f'/video_feed/{client.id}'
            context = ComponentContext(
                state=self.state, env_lock=self.env_lock)
            play_sound = setup_audio(self.env)

            if self.title:
                ui.label(
                    "MLVLab - " + self.title).classes('w-full text-2xl font-bold text-center mt-4 mb-1')
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
                            'w-full h-auto rounded shadow-lg bg-gray-200')
                    with ui.column().classes('w-full lg:w-1/4 lg:pl-2'):
                        for component in self.right_components:
                            component.render(self.state, context)

            def render_tick() -> None:
                try:
                    pending_sound = self.state.get(["sim", "last_sound"])
                    if pending_sound and self.state.get(["ui", "sound_enabled"]):
                        play_sound(pending_sound)
                        self.state.set(["sim", "last_sound"], None)
                except Exception:
                    pass
            context.register_timer(ui.timer(1/15, render_tick))

    def run(self, host='127.0.0.1', port=8181) -> None:
        def run_nicegui():
            self._build_page()
            ui.run(host=host, port=port, title=self.title,
                   dark=self.dark, reload=False, show=False, native=False)

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
                buffer=self.frame_buffer, state=self.state
            )
            _ACTIVE_THREADS["runner"] = self.runner
            _ACTIVE_THREADS["renderer"].start()
            _ACTIVE_THREADS["runner"].start()
            print("✅ Startup de hilos completado.")
            _server_started.set()

        @app.on_shutdown
        def shutdown_handler():
            print("--- DETENIENDO APLICACIÓN (on_shutdown) ---")

            async def cancel_streams():
                tasks_to_cancel = list(
                    _ACTIVE_THREADS["stream_tasks"].values())
                if tasks_to_cancel:
                    for task in tasks_to_cancel:
                        task.cancel()
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                _ACTIVE_THREADS["stream_tasks"].clear()
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    cancel_streams(), loop)
                future.result(timeout=2.0)
            except (RuntimeError, TimeoutError):
                pass

            renderer = _ACTIVE_THREADS.get("renderer")
            runner = _ACTIVE_THREADS.get("runner")
            if isinstance(renderer, threading.Thread):
                renderer.stop()
            if hasattr(runner, 'stop'):
                runner.stop()
            if isinstance(renderer, threading.Thread):
                renderer.join(timeout=1.0)
            if hasattr(runner, 'join'):
                runner.join(timeout=1.0)

            _ACTIVE_THREADS["renderer"] = None
            _ACTIVE_THREADS["runner"] = None
            print("✅ Limpieza de shutdown completada.")

        nicegui_thread = threading.Thread(target=run_nicegui, daemon=True)
        nicegui_thread.start()
        _server_started.wait()

        qt_app = QApplication(sys.argv)
        url = QUrl(f"http://{host}:{port}")

        class MainWindow(QMainWindow):
            def closeEvent(self, event):
                app.shutdown()
                event.accept()

        window = MainWindow()
        window.setWindowTitle("MLVLab Analytics Panel - " + self.env.spec.id)
        window.setGeometry(100, 100, 1280, 800)
        web_view = QWebEngineView()
        web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        web_view.setUrl(url)
        window.setCentralWidget(web_view)
        window.showMaximized()
        print(
            f"🚀 Mostrando ventana nativa con PySide6. Cargando {url.toString()}...")
        sys.exit(qt_app.exec())

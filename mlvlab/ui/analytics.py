from __future__ import annotations

from typing import Any, List, Optional
import time
from threading import Lock

from nicegui import ui, app
import asyncio
try:
    from nicegui import Client  # type: ignore
except Exception:  # pragma: no cover - compatibilidad
    Client = object  # type: ignore
from starlette.websockets import WebSocket

from .state import StateStore
from .runtime import SimulationRunner
from .components.base import ComponentContext, UIComponent
from mlvlab.helpers.ng import setup_audio, frame_to_webp_bytes, create_reward_chart


_ACTIVE_RUNNER = None  # runner activo en memoria (no persistente)


class AnalyticsView:
    """
    Vista principal declarativa que arma el panel de análisis estándar (3 columnas).

    Uso:
        view = AnalyticsView(env, agent, left_panel_components=[...], right_panel_components=[...])
        view.run()
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        left_panel_components: Optional[List[UIComponent]] = None,
        right_panel_components: Optional[List[UIComponent]] = None,
        title: str = "MLVLab Analytics",
        history_size: int = 100,
        dark: bool = False,
        subtitle: Optional[str] = None,
        agent_hparams_defaults: Optional[dict] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.left_components = left_panel_components or []
        self.right_components = right_panel_components or []
        self.title = "MLVLab - " + title
        self.history_size = history_size
        self.dark = dark
        self.subtitle = subtitle

        self.env_lock = Lock()
        # Defaults de hiperparámetros (permitidos por el alumno)
        self.user_hparams = agent_hparams_defaults or {}

        self.state = StateStore(
            defaults={
                "sim": {
                    "command": "run",
                    "speed_multiplier": 1,
                    "turbo_mode": False,
                    "total_steps": 0,
                    "current_episode_reward": 0.0,
                },
                "agent": {
                    # Inicialización por defecto sensible
                    **{
                        'epsilon': 1.0,
                        'epsilon_decay': 0.99,
                        'min_epsilon': 0.1,
                        'learning_rate': 0.1,
                        'discount_factor': 0.9,
                    },
                    **{k: float(v) for k, v in self.user_hparams.items()},
                },
                "metrics": {
                    "episodes_completed": 0,
                    "reward_history": [],
                    "steps_per_second": 0,
                    "chart_reward_number": history_size,
                },
                "ui": {
                    "sound_enabled": False,
                    "chart_visible": True,
                },
            }
        )

        self.runner = SimulationRunner(
            env=self.env,
            agent=self.agent,
            state=self.state,
            env_lock=self.env_lock,
        )

        self._reward_chart = None  # type: ignore

    # -------------------------- Página principal -------------------------- #
    def _build_page(self) -> None:
        route = "/"

        @ui.page(route)
        def main(client: Client):  # type: ignore[override]
            # Contexto por cliente: timers, bindings y utilidades
            context = ComponentContext(
                state=self.state,
                env_lock=self.env_lock,
            )

            # Controles de cierre y audio
            play_sound = setup_audio()

            # Layout 3 columnas, con centro reservado al visualizador
            ui.label(self.title).classes(
                'w-full text-2xl font-bold text-center mt-4 mb-1')
            if self.subtitle:
                ui.label(self.subtitle).classes(
                    'w-full text-base text-center mb-2 opacity-80')

            with ui.element('div').classes('w-full flex justify-center'):
                with ui.element('div').classes('w-full max-w-[1400px] flex flex-col lg:flex-row'):
                    # Columna izquierda
                    with ui.column().classes('w-full lg:w-1/4 pb-4 lg:pr-2 lg:pb-0'):
                        for component in self.left_components:
                            component.render(self.state, context)

                    # Columna central: environment viewer (canvas + WS)
                    with ui.column().classes('w-full lg:w-2/4 pb-4 lg:pb-0 lg:px-2 items-center'):
                        from .components.environment_viewer import EnvironmentViewer

                        viewer = EnvironmentViewer()
                        viewer.render(self.state, context)
                        # Registrar WS de frames por cliente
                        self._register_frame_ws_route(client)

                    # Columna derecha
                    with ui.column().classes('w-full lg:w-1/4 lg:pl-2'):
                        for component in self.right_components:
                            component.render(self.state, context)

            # Timer de render/metricas ~15 Hz
            def render_tick() -> None:
                # Actualizar serie del chart si cambia
                # Reproducir audio si hay señal y el sonido está activado
                try:
                    pending_sound = self.state.get(["sim", "last_sound"])
                    if pending_sound and self.state.get(["ui", "sound_enabled"]):
                        play_sound(pending_sound)
                        self.state.set(["sim", "last_sound"], None)
                except Exception:
                    pass

                # Calcular SPS cada ~0.5 s ya lo hace el runner; aquí no duplicamos

            context.register_timer(ui.timer(1/15, render_tick))

            # Inyectar receptor de señal de cierre multi-pestaña
            ui.run_javascript(
                """
                (() => {
                  if (window.__mlvlabShutdownInit) return;
                  window.__mlvlabShutdownInit = true;
                  const closeSelf = () => {
                    try { window.close(); } catch (e) {}
                    try { location.replace('about:blank'); } catch (e) {}
                    try { location.href = 'about:blank'; } catch (e) {}
                  };
                  try {
                    const bc = new BroadcastChannel('mlvlab-shutdown');
                    bc.onmessage = ev => { if (ev && ev.data === 'shutdown') closeSelf(); };
                  } catch (e) {}
                  window.addEventListener('storage', (e) => {
                    if (e.key === 'mlvlab_shutdown_signal') closeSelf();
                  });
                })();
                """
            )

        # Evitar herramienta no usada
        _ = create_reward_chart

    # type: ignore[override]
    def _register_frame_ws_route(self, client: Client) -> None:
        """Registra un WebSocket que envía frames WebP a este cliente."""
        ws_route = f"/ws/frame/{id(client)}"

        async def frame_ws(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    with self.env_lock:
                        # Permite overlays (Q-table, etc.) si el entorno lo soporta
                        try:
                            self.env.unwrapped.set_render_data(q_table=getattr(
                                self.agent, 'q_table', None))  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        frame = self.env.render()
                    webp = frame_to_webp_bytes(frame, quality=100)
                    await websocket.send_bytes(webp)
                    await asyncio.sleep(1/25)  # ~25 FPS
            except Exception:
                pass

        try:
            app.add_websocket_route(ws_route, frame_ws)
        except Exception:
            # Compatibilidad: si ya existe, ignorar
            pass

        # JS para consumir el WS y pintar en canvas#viz_canvas si existe
        ui.run_javascript(
            f"""
            (() => {{
              const canvas = document.getElementById('viz_canvas');
              if (!canvas) return;
              const ctx = canvas.getContext('2d');
              let url = (location.origin.replace(/^http/, 'ws')) + '{ws_route}';
              const ws = new WebSocket(url);
              ws.binaryType = 'arraybuffer';
              ws.onmessage = async (ev) => {{
                const blob = new Blob([ev.data], {{ type: 'image/webp' }});
                const img = new Image();
                img.onload = () => {{
                  try {{ canvas.width = img.width; canvas.height = img.height; ctx.drawImage(img, 0, 0); }} catch (e) {{}}
                }};
                img.src = URL.createObjectURL(blob);
              }};
            }})();
            """
        )

    # ------------------------------ Ciclo de vida ------------------------------ #
    def run(self) -> None:
        """Arranca la app NiceGUI y la simulación."""

        def on_startup() -> None:
            # Si hay un runner previo (por live-reload en el mismo proceso), detenerlo limpiamente
            global _ACTIVE_RUNNER
            try:
                if _ACTIVE_RUNNER and _ACTIVE_RUNNER is not self.runner:
                    stop = getattr(_ACTIVE_RUNNER, 'stop', None)
                    if callable(stop):
                        stop()
            except Exception:
                pass
            _ACTIVE_RUNNER = self.runner
            self.runner.start()

        app.on_startup(on_startup)

        @app.on_shutdown
        def _shutdown_cleanup() -> None:  # pragma: no cover
            global _ACTIVE_RUNNER
            try:
                self.runner.stop()
            except Exception:
                pass
            _ACTIVE_RUNNER = None

        self._build_page()
        ui.run(title=self.title,
               dark=self.dark, reload=True, show=False)

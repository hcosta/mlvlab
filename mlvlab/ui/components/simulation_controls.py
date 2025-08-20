# mlvlab/ui/components/simulation_controls.py

from __future__ import annotations
from typing import List, Optional
from nicegui import app, ui
from .base import UIComponent, ComponentContext
from mlvlab.i18n.core import i18n


class SimulationControls(UIComponent):
    DEFAULT_INCLUDES = ["speed", "turbo"]
    DEFAULT_BUTTONS = ["play", "reset", "sound", "debug"]

    def __init__(
        self,
        includes: Optional[List[str]] = None,
        buttons: Optional[List[str]] = None
    ):
        super().__init__()
        self.includes = includes if includes is not None else self.DEFAULT_INCLUDES
        self.buttons = buttons if buttons is not None else self.DEFAULT_BUTTONS

    def render(self, state, context: ComponentContext) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(i18n.t("ui.components.simulation_controls.close_confirm"))
            with ui.row().classes('w-full justify-end'):
                def do_shutdown_dialog():
                    ui.run_javascript("try { window.close(); } catch (e) {}")
                    app.shutdown()
                ui.button(i18n.t("ui.components.simulation_controls.yes_close"), on_click=do_shutdown_dialog, color='red')
                ui.button(i18n.t("ui.components.simulation_controls.no_cancel"), on_click=dialog.close)

        with ui.card().classes('w-full mb-1'):
            ui.label(i18n.t("ui.components.simulation_controls.title")).classes(
                'text-lg font-semibold text-center w-full')

            if any(item in self.includes for item in ["speed", "turbo"]):
                with ui.row().classes('w-full items-center no-wrap gap-x-0'):
                    if "speed" in self.includes:
                        width_class = 'w-2/3' if "turbo" in self.includes else 'w-full'
                        with ui.row().classes(f'{width_class} items-center gap-x-2 no-wrap'):
                            ui.label().bind_text_from(state.full(), 'sim',
                                                      lambda s: i18n.t("ui.components.simulation_controls.speed_multiplier", speed=s.get('speed_multiplier', 1))).classes('w-36')
                            slider = ui.slider(
                                min=0, max=200, step=5).classes('flex-grow')
                            slider.bind_value(
                                state.full()['sim'], 'speed_multiplier')
                    if "turbo" in self.includes:
                        width_class = 'w-1/3' if "speed" in self.includes else 'w-full'
                        with ui.row().classes(f'{width_class} justify-end'):
                            switch = ui.switch(i18n.t("ui.components.simulation_controls.turbo"))
                            switch.bind_value(
                                state.full()['sim'], 'turbo_mode')

                def _normalize_types():
                    try:
                        sim = state.full().get('sim', {})
                        if "speed" in self.includes and 'speed_multiplier' in sim:
                            sim['speed_multiplier'] = max(
                                1, min(200, int(sim.get('speed_multiplier') or 1)))
                        if "turbo" in self.includes and 'turbo_mode' in sim:
                            sim['turbo_mode'] = bool(sim.get('turbo_mode'))
                    except Exception:
                        pass
                ui.timer(0.5, _normalize_types)

            if self.buttons:
                with ui.row().classes('w-full justify-around mt-3 items-center'):
                    if "play" in self.buttons:
                        def toggle_simulation():
                            cmd = state.get(['sim', 'command']) or 'run'
                            state.set(['sim', 'command'],
                                      'pause' if cmd == 'run' else 'run')
                        with ui.button(on_click=toggle_simulation).props('outline'):
                            ui.icon('pause').bind_name_from(state.full(), 'sim', lambda s: 'pause' if s.get(
                                'command') == 'run' else 'play_arrow')

                    if "reset" in self.buttons:
                        # Ahora este botón se encarga de todo el reinicio.
                        def on_reset_click():
                            state.set(['sim', 'command'], 'reset')
                            state.set(['ui', 'clear_upload'], True)
                            state.set(['sim', 'active_model_name'],
                                      'Ninguno (Nuevo)')
                        ui.button(on_click=on_reset_click).props(
                            'icon=refresh outline')

                    if "sound" in self.buttons:
                        def toggle_sound():
                            state.set(['ui', 'sound_enabled'], not bool(
                                state.get(['ui', 'sound_enabled'])))
                        with ui.button(on_click=toggle_sound).props('outline'):
                            ui.icon('volume_up').bind_name_from(state.full(
                            ), 'ui', lambda s: 'volume_up' if s.get('sound_enabled') else 'volume_off')

                    if "debug" in self.buttons:
                        def toggle_debug_mode():
                            state.set(['ui', 'debug_mode'], not bool(
                                state.get(['ui', 'debug_mode'])))
                        with ui.button(on_click=toggle_debug_mode).props('outline'):
                            ui.icon('visibility').bind_name_from(state.full(
                            ), 'ui', lambda s: 'visibility' if s.get('debug_mode') else 'visibility_off')

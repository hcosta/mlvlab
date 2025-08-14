from __future__ import annotations

from nicegui import ui, app

from .base import UIComponent, ComponentContext


class SimulationControls(UIComponent):
    def render(self, state, context: ComponentContext) -> None:
        # Diálogo de confirmación de cierre
        with ui.dialog() as dialog, ui.card():
            ui.label('¿Estás seguro de que quieres cerrar la aplicación?')
            with ui.row().classes('w-full justify-end'):
                def do_shutdown_dialog():
                    ui.run_javascript(
                        """
                        (() => {
                          const nonce = String(Date.now()) + '-' + Math.random().toString(36).slice(2);
                          try { new BroadcastChannel('mlvlab-shutdown').postMessage('shutdown'); } catch (e) {}
                          try { localStorage.setItem('mlvlab_shutdown_signal', nonce); setTimeout(() => { try { localStorage.removeItem('mlvlab_shutdown_signal'); } catch (e) {} }, 200); } catch (e) {}
                          try { window.close(); } catch (e) {}
                          try { location.replace('about:blank'); } catch (e) {}
                        })();
                        """
                    )
                    app.shutdown()
                ui.button('Sí, cerrar', on_click=do_shutdown_dialog, color='red')
                ui.button('No, cancelar', on_click=dialog.close)

        with ui.card().classes('w-full mb-4'):
            ui.label('Controles de Simulación').classes(
                'text-lg font-semibold text-center w-full')
            ui.label().bind_text_from(state.full(), 'sim',
                                      lambda s: f"Velocidad Simulación: {s.get('speed_multiplier', 1)}x")
            with ui.grid(columns=2).classes('w-full items-center gap-2'):
                # Slider velocidad (binding directo al dict interno del estado)
                slider = ui.slider(min=1, max=50, step=1).classes('w-full')
                slider.bind_value(state.full()['sim'], 'speed_multiplier')

                # Switch turbo (binding directo evita ambigüedades de eventos)
                switch = ui.switch('Turbo Mode').classes('w-full')
                switch.bind_value(state.full()['sim'], 'turbo_mode')

                # Protección ante live-reload: forzar tipos correctos
                def _normalize_types():  # pragma: no cover
                    try:
                        sim = state.full().get('sim', {})
                        spd = int(sim.get('speed_multiplier') or 1)
                        sim['speed_multiplier'] = max(1, min(50, spd))
                        sim['turbo_mode'] = bool(sim.get('turbo_mode'))
                    except Exception:
                        pass
                ui.timer(0.5, _normalize_types)

            with ui.row().classes('w-full justify-around mt-3 items-center'):
                # Play/Pause
                def toggle_simulation():
                    cmd = state.get(['sim', 'command']) or 'run'
                    state.set(['sim', 'command'],
                              'pause' if cmd == 'run' else 'run')

                with ui.button(on_click=toggle_simulation).props('outline'):
                    ui.icon('pause').bind_name_from(state.full(), 'sim', lambda s: 'pause' if s.get(
                        'command') == 'run' else 'play_arrow')

                # Reset
                ui.button(on_click=lambda: state.set(
                    ['sim', 'command'], 'reset')).props('icon=refresh outline')

                # Sonido
                def toggle_sound():
                    enabled = bool(state.get(['ui', 'sound_enabled']))
                    state.set(['ui', 'sound_enabled'], not enabled)

                with ui.button(on_click=toggle_sound).props('outline'):
                    ui.icon('volume_up').bind_name_from(state.full(
                    ), 'ui', lambda s: 'volume_up' if s.get('sound_enabled') else 'volume_off')

                # Cierre
                # ui.button(on_click=dialog.open).props(
                #     'icon=close outline color="red"')

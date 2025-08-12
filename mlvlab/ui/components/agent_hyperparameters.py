from __future__ import annotations

from typing import List
from nicegui import ui

from .base import UIComponent, ComponentContext


def _pretty(name: str) -> str:
    return name.replace('_', ' ').strip().title()


class AgentHyperparameters(UIComponent):
    def __init__(self, agent: object, params: List[str]) -> None:
        self.agent = agent
        self.params = params

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full'):
            ui.label('Configuración del Agente').classes(
                'text-lg font-semibold text-center w-full mb-3')

            with ui.grid(columns=3).classes('w-full gap-x-2 items-center'):
                for name in self.params:
                    ui.label(_pretty(name)).classes(
                        'col-span-2 justify-self-start')

                    # preferencia: state.agent -> attr del agente -> defaults
                    value_from_state = state.get(['agent', name])
                    if value_from_state is not None:
                        initial_value = float(value_from_state)
                    else:
                        attr_val = getattr(self.agent, name, None)
                        if attr_val is not None:
                            initial_value = float(attr_val)
                        else:
                            defaults = {
                                'learning_rate': 0.1,
                                'discount_factor': 0.9,
                                'epsilon_decay': 0.99,
                                'epsilon': 1.0,
                                'min_epsilon': 0.1,
                            }
                            initial_value = float(defaults.get(name, 0.0))
                        state.set(['agent', name], initial_value)

                    num = ui.number(value=initial_value,
                                    format='%.5f', step=0.00001, min=0, max=1)
                    # Deshabilitar edición cuando la simulación está en marcha (pureza)

                    def _is_running():
                        try:
                            return (state.get(['sim', 'command']) or 'run') == 'run'
                        except Exception:
                            return True
                    num.bind_enabled_from(state.full(), 'sim', lambda sim: (
                        sim or {}).get('command') != 'run')

                    def _on_change(e, attr_name=name):
                        try:
                            val = float(e.args) if e.args is not None else 0.0
                        except Exception:
                            val = 0.0
                        state.set(['agent', attr_name], val)
                        if hasattr(self.agent, attr_name):
                            try:
                                setattr(self.agent, attr_name, val)
                            except Exception:
                                pass

                    num.on('update:model-value', _on_change)

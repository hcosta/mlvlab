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
                    initial_value = getattr(self.agent, name, None)
                    if initial_value is None:
                        defaults = {
                            'learning_rate': 0.1,
                            'discount_factor': 0.9,
                            'epsilon_decay': 0.99,
                        }
                        initial_value = defaults.get(name, 0.0)

                    num = ui.number(value=float(initial_value),
                                    format='%.5f', step=0.00001, min=0, max=1)

                    def _make_setter(attr_name: str):
                        def setter(v):
                            try:
                                val = float(v) if v is not None else 0.0
                            except Exception:
                                val = 0.0
                            setattr(self.agent, attr_name, val)
                            if attr_name in ("learning_rate", "discount_factor", "epsilon_decay"):
                                state.set(['agent', attr_name], val)
                        return setter

                    if name in ("learning_rate", "discount_factor", "epsilon_decay"):
                        state.set(['agent', name], float(initial_value))

                    num.on('update:model-value', _make_setter(name))

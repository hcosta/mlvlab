from __future__ import annotations

from typing import List
from nicegui import ui

from .base import UIComponent, ComponentContext
from mlvlab.i18n.core import i18n


def _pretty(name: str) -> str:
    return name.replace('_', ' ').strip().title()


def _pretty(name: str) -> str:
    return name.replace('_', ' ').strip().title()


class AgentHyperparameters(UIComponent):
    # El 'agent' se ha eliminado del constructor.
    def __init__(self, params: List[str]) -> None:
        self.params = params

    def render(self, state, context: ComponentContext) -> None:
        # Obtenemos el agente desde el contexto, no desde 'self'.
        agent = context.agent

        with ui.card().classes('w-full mb-1'):
            ui.label(i18n.t("ui.components.agent_hyperparameters.title")).classes(
                'text-lg font-semibold text-center w-full mb-0')

            with ui.grid(columns=3).classes('w-full gap-x-2 items-center'):
                for name in self.params:
                    ui.label(_pretty(name)).classes(
                        'col-span-2 justify-self-start')

                    value_from_state = state.get(['agent', name])
                    if value_from_state is not None:
                        initial_value = float(value_from_state)
                    else:
                        # Usamos el 'agent' del contexto.
                        attr_val = getattr(agent, name, None)
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

                    num.bind_enabled_from(state.full(), 'sim', lambda sim: (
                        sim or {}).get('command') != 'run')

                    def _on_change(e, attr_name=name):
                        try:
                            val = float(e.args) if e.args is not None else 0.0
                        except Exception:
                            val = 0.0
                        state.set(['agent', attr_name], val)
                        # Actualizamos el 'agent' del contexto.
                        if hasattr(agent, attr_name):
                            try:
                                setattr(agent, attr_name, val)
                            except Exception:
                                pass

                    num.on('update:model-value', _on_change)

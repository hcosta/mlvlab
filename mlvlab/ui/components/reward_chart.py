from __future__ import annotations

from nicegui import ui

from .base import UIComponent, ComponentContext
from mlvlab.helpers.ng import create_reward_chart


class RewardChart(UIComponent):
    def __init__(self, history_size: int = 100) -> None:
        self.history_size = history_size
        self._chart = None
        self._card = None

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full') as chart_card:
            self._card = chart_card
            number = int(
                state.get(['metrics', 'chart_reward_number']) or self.history_size)
            self._chart = create_reward_chart(chart_card, number=number)

            def tick():
                desired = list(state.get(['metrics', 'reward_history']) or [])
                current = list(
                    self._chart.options['series'][0]['data']) if self._chart else []
                if self._chart and current != desired:
                    self._chart.options['series'][0]['data'] = desired
                    self._chart.update()
                visible = bool(state.get(['ui', 'chart_visible']))
                if self._card:
                    if visible:
                        self._card.classes(remove='hidden')
                    else:
                        self._card.classes(add='hidden')

            context.register_timer(ui.timer(1/5, tick))

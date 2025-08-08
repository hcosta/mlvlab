from __future__ import annotations

from nicegui import ui

from .base import UIComponent, ComponentContext


class MetricsDashboard(UIComponent):
    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full mb-4'):
            ui.label('Métricas en Tiempo Real').classes(
                'text-lg font-semibold text-center w-full')
            ui.label().bind_text_from(state.full(), 'agent',
                                      lambda a: f"Epsilon (Exploración): {float(a.get('epsilon', 1.0)):.3f}")
            ui.label().bind_text_from(state.full(), 'sim',
                                      lambda s: f"Recompensa Actual: {s.get('current_episode_reward', 0)}")
            ui.label().bind_text_from(state.full(), 'metrics',
                                      lambda m: f"Episodios Completados: {m.get('episodes_completed', 0)}")
            ui.label().bind_text_from(state.full(), 'metrics',
                                      lambda m: f"Pasos/seg: {m.get('steps_per_second', 0):,d}")
            ui.button('Mostrar/Esconder Gráfico', on_click=lambda: state.set(['ui', 'chart_visible'], not bool(
                state.get(['ui', 'chart_visible'])))).props('icon=bar_chart outline w-full')

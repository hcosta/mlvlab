from __future__ import annotations
from nicegui import ui
from typing import List, Optional
from .base import UIComponent, ComponentContext


class MetricsDashboard(UIComponent):
    """
    Un componente de UI que muestra métricas en tiempo real de la simulación,
    con opciones para personalizar las métricas visibles.
    """
    # Define las métricas válidas y su orden por defecto
    DEFAULT_METRICS = ["epsilon", "current_reward",
                       "episodes_completed", "steps_per_second"]

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Args:
            metrics: Lista opcional de métricas a mostrar.
                     Los valores posibles son: "epsilon", "current_reward",
                     "episodes_completed", "steps_per_second".
                     Si es None, se muestran todas las métricas por defecto.
        """
        super().__init__()
        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS

    def render(self, state, context: ComponentContext) -> None:
        # Si no hay métricas para mostrar, no renderizar nada.
        if not self.metrics:
            return

        with ui.card().classes('w-full mb-1'):
            ui.label('Métricas en Tiempo Real').classes(
                'text-lg font-semibold text-center w-full')

            # Renderizar condicionalmente cada métrica si está en la lista
            if "epsilon" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'agent',
                    lambda a: f"Epsilon (Exploración): {float(a.get('epsilon', 1.0)):.3f}"
                )

            if "current_reward" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'sim',
                    lambda s: f"Recompensa Actual: {s.get('current_episode_reward', 0)}"
                )

            if "episodes_completed" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'metrics',
                    lambda m: f"Episodios Completados: {m.get('episodes_completed', 0)}"
                )

            if "steps_per_second" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'metrics',
                    lambda m: f"Acciones por Segundo: {m.get('steps_per_second', 0):,d}"
                )

            # El botón original está comentado, se mantiene así.
            # ui.button('Mostrar/Esconder Gráfico', on_click=lambda: state.set(['ui', 'chart_visible'], not bool(
            #     state.get(['ui', 'chart_visible'])))).props('icon=bar_chart outline w-full')

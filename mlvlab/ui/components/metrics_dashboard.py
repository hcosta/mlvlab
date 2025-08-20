from __future__ import annotations
from nicegui import ui
from typing import List, Optional
from .base import UIComponent, ComponentContext
from mlvlab.i18n.core import i18n


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
                     Valores posibles: "epsilon", "current_reward",
                     "episodes_completed", "steps_per_second", "seed".
                     Si es None, se muestran todas las métricas por defecto.
        """
        super().__init__()
        self.metrics = metrics if metrics is not None else self.DEFAULT_METRICS

    def render(self, state, context: ComponentContext) -> None:
        # Si no hay métricas para mostrar, no renderizar nada.
        if not self.metrics:
            return

        with ui.card().classes('w-full mb-1'):
            ui.label(i18n.t("ui.components.metrics_dashboard.title")).classes(
                'text-lg font-semibold text-center w-full')

            # Renderizar condicionalmente cada métrica si está en la lista
            if "epsilon" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'agent',
                    lambda a: i18n.t("ui.components.metrics_dashboard.epsilon_exploration",
                                     epsilon=f"{float(a.get('epsilon', 1.0)):.3f}")
                )

            if "current_reward" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'sim',
                    lambda s: i18n.t("ui.components.metrics_dashboard.current_reward", reward=s.get(
                        'current_episode_reward', 0))
                )

            if "episodes_completed" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'metrics',
                    lambda m: i18n.t("ui.components.metrics_dashboard.episodes_completed", episodes=m.get(
                        'episodes_completed', 0))
                )

            if "steps_per_second" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'metrics',
                    lambda m: i18n.t("ui.components.metrics_dashboard.steps_per_second",
                                     steps=f"{m.get('steps_per_second', 0):,d}")
                )

            if "seed" in self.metrics:
                ui.label().bind_text_from(
                    state.full(), 'sim',
                    lambda s: i18n.t(
                        "ui.components.metrics_dashboard.map_seed", seed=s.get('seed', 'N/D'))
                )

            # El botón original está comentado, se mantiene así.
            # ui.button('Mostrar/Esconder Gráfico', on_click=lambda: state.set(['ui', 'chart_visible'], not bool(
            #     state.get(['ui', 'chart_visible'])))).props('icon=bar_chart outline w-full')

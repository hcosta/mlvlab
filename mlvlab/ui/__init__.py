"""
mlvlab.ui

Paquete de UI declarativa para simulaciones RL con NiceGUI.

Expone:
- AnalyticsView: vista principal que arma el layout de 3 columnas
- Widgets esenciales: EnvironmentViewer, SimulationControls, AgentHyperparameters,
  MetricsDashboard, RewardChart, LiveMetric, DynamicTable, InfoPanel
"""

from .analytics import AnalyticsView

# Widgets
from .components.environment_viewer import EnvironmentViewer
from .components.simulation_controls import SimulationControls
from .components.agent_hyperparameters import AgentHyperparameters
from .components.metrics_dashboard import MetricsDashboard
from .components.reward_chart import RewardChart
from .components.live_metric import LiveMetric
from .components.dynamic_table import DynamicTable
from .components.info_panel import InfoPanel

__all__ = [
    "AnalyticsView",
    # Widgets
    "EnvironmentViewer",
    "SimulationControls",
    "AgentHyperparameters",
    "MetricsDashboard",
    "RewardChart",
    "LiveMetric",
    "DynamicTable",
    "InfoPanel",
]

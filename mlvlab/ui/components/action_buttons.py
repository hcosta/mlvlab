# mlvlab/ui/components/action_buttons.py

from __future__ import annotations
from typing import Dict, Optional
from nicegui import ui

# Asumiendo imports estándar del framework
try:
    from .base import UIComponent, ComponentContext
    from mlvlab.i18n.core import i18n
except ImportError:
    # Fallbacks para
    class UIComponent:
        pass

    class ComponentContext:
        pass

    class i18n_fallback:
        def t(self, key, default="[Missing]", **kwargs):
            return default if default != "[Missing]" else f"[{key}]"
    i18n = i18n_fallback()


class ActionButtons(UIComponent):
    """
    Widget que muestra botones de acción personalizados.
    Al hacer clic, señalan una acción pendiente en el StateStore para que el SimulationRunner la ejecute.
    """

    # Define iconos para acciones conocidas, fallback a 'tune' (ajustes).
    DEFAULT_ICONS = {
        "shift": "shuffle",  # Icono de aleatorio/cambio
        "map": "map",
        "reset": "refresh",
        "default": "tune"  # Icono de acción/ajuste genérico
    }

    def __init__(
        self,
        # Formato: {"nombre_metodo": "Título Descriptivo"}
        actions: Dict[str, str],
        title: Optional[str] = None,
        icons: Optional[Dict[str, str]] = None
    ):
        super().__init__()
        self.actions = actions
        self.title = title or i18n.t(
            "ui.components.action_buttons.title", default="Environment Actions")
        self.custom_icons = icons or {}

    def render(self, state, context: ComponentContext) -> None:
        if not self.actions:
            return

        with ui.card().classes('w-full mb-1'):
            ui.label(self.title).classes(
                'text-lg font-semibold text-center w-full')

            with ui.column().classes('w-full gap-2 mt-2'):
                for action_name, display_title in self.actions.items():
                    self._render_action_button(
                        state, context, action_name, display_title)

    def _render_action_button(self, state, context, action_name: str, display_title: str):
        """Renderiza una fila de botón de acción."""

        def on_action_click():
            # 1. Verificar si el entorno soporta la acción
            if not hasattr(context.env.unwrapped, action_name):
                ui.notify(
                    f"Error: Acción '{action_name}' no encontrada en el entorno.", type='negative')
                return

            # 2. Verificar si ya hay otra acción pendiente
            if state.get(['sim', 'pending_action']):
                ui.notify(i18n.t("ui.components.action_buttons.action_pending",
                          default="Waiting, another action is in progress."), type='warning', timeout=1.5)
                return

            # 3. Establecer la acción pendiente en el StateStore (Patrón Command).
            # Usamos un diccionario para flexibilidad futura (argumentos).
            state.set(['sim', 'pending_action'], {"name": action_name})
            # print(f"UI: Solicitando ejecución de acción '{action_name}'.")
            ui.notification(
                i18n.t("ui.components.action_buttons.action_requested",
                       default="Requesting: {display_title}...").format(display_title=display_title),
                type='info', timeout=1.5)

        # Determinar el icono
        icon = self.custom_icons.get(action_name)
        if not icon:
            # Buscar en iconos por defecto basados en palabras clave
            for keyword, default_icon in self.DEFAULT_ICONS.items():
                if keyword in action_name:
                    icon = default_icon
                    break
            else:
                icon = self.DEFAULT_ICONS["default"]

        # Fila: Descripción a la izquierda, Botón a la derecha.
        with ui.row().classes('w-full items-center justify-between no-wrap py-1'):
            # Descripción e Icono
            with ui.row().classes('items-center gap-2'):
                ui.icon(icon, size='sm').classes('opacity-70')
                ui.label(display_title).classes('flex-grow mr-2')

            # Botón de activación (Usamos 'play_arrow' para indicar ejecución)
            ui.button(on_click=on_action_click).props(
                f'icon=play_arrow outline dense size=sm')

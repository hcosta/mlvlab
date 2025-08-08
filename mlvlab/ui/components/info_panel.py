from __future__ import annotations

from nicegui import ui

from .base import UIComponent, ComponentContext


class InfoPanel(UIComponent):
    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full mb-2'):
            ui.label(self.title).classes('text-lg font-semibold')
            ui.label(self.text).classes('text-sm')

from __future__ import annotations

from nicegui import ui

from .base import UIComponent, ComponentContext


class EnvironmentViewer(UIComponent):
    """
    Canvas central donde se renderizan los frames recibidos por WebSocket binario.
    """

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full p-0 bg-black flex justify-center items-center'):
            canvas = ui.element('canvas').classes(
                'max-w-[550px]').style('width: 100%; height: auto;')
            canvas.props('id=viz_canvas width=900 height=900')

from __future__ import annotations

from typing import Callable, List, Dict, Any, Optional
from nicegui import ui

from .base import UIComponent, ComponentContext


class DynamicTable(UIComponent):
    def __init__(self, title: str, source: Optional[Callable[[dict], List[Dict[str, Any]]]] = None) -> None:
        self.title = title
        self.source = source

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full mb-2'):
            ui.label(self.title).classes('text-lg font-semibold')
            table = ui.table(columns=[], rows=[]).classes('w-full')

            def tick():
                rows = []
                try:
                    rows = self.source(state.full()) if self.source else []
                except Exception:
                    rows = []
                # Actualizar columnas dinámicamente según claves
                cols = []
                if rows:
                    keys = list(rows[0].keys())
                    cols = [{'name': k, 'label': k, 'field': k} for k in keys]
                table.options['columns'] = cols
                table.options['rows'] = rows
                table.update()

            context.register_timer(ui.timer(0.5, tick))

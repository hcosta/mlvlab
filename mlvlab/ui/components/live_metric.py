from __future__ import annotations

from typing import Callable, Optional, Any
from nicegui import ui

from .base import UIComponent, ComponentContext


def _resolve_source(state_dict: dict, path: str) -> Any:
    cur: Any = state_dict
    for part in path.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class LiveMetric(UIComponent):
    def __init__(self, name: str, source: Optional[Callable[[dict], Any]] = None, path: Optional[str] = None, fmt: str = "{}") -> None:
        self.name = name
        self.source = source
        self.path = path
        self.fmt = fmt

    def render(self, state, context: ComponentContext) -> None:
        with ui.card().classes('w-full mb-2'):
            label = ui.label(f"{self.name}: ...")

            def tick():
                value = None
                if self.source is not None:
                    try:
                        value = self.source(state.full())
                    except Exception:
                        value = None
                elif self.path:
                    value = _resolve_source(state.full(), self.path)
                label.text = f"{self.name}: {self.fmt.format(value)}"

            context.register_timer(ui.timer(0.2, tick))

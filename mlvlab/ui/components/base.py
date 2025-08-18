from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field

from nicegui import ui
from threading import Lock

from ..state import StateStore


@dataclass
class ComponentContext:
    state: StateStore
    env_lock: Lock
    agent: Any
    env: Any
    _timers: list = field(default_factory=list, init=False)
    # 'init=False' es una buena práctica aquí

    def register_timer(self, timer: Any) -> None:
        self._timers.append(timer)


class UIComponent:
    """Base para todos los widgets."""

    def render(self, state: StateStore, context: ComponentContext) -> None:  # pragma: no cover - abstracto
        raise NotImplementedError

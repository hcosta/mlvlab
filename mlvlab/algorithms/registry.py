from __future__ import annotations

from typing import Callable, Dict, Protocol, Any, Optional


class AlgorithmPlugin(Protocol):
    def key(self) -> str: ...

    def build_agent(self, env: Any, hparams: Dict[str, Any]) -> Any: ...

    def train(self, env_id: str, config: Dict[str, Any], run_dir,
              seed: Optional[int] = None, render: bool = False) -> None: ...

    def eval(self, env_id: str, run_dir, episodes: int,
             seed: Optional[int] = None, cleanup: bool = True, video: bool = False) -> Optional[str]: ...


_ALGORITHMS: Dict[str, AlgorithmPlugin] = {}


def register_algorithm(plugin: AlgorithmPlugin) -> None:
    k = plugin.key()
    _ALGORITHMS[k] = plugin


def get_algorithm(key: str) -> AlgorithmPlugin:
    if key not in _ALGORITHMS:
        raise KeyError(f"Algoritmo no registrado: {key}")
    return _ALGORITHMS[key]


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS.keys())

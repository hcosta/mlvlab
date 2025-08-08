from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import copy


class StateStore:
    """
    Almacén de estado jerárquico con utilidades de acceso seguro.

    Estructura sugerida:
    - sim: control de simulación y métricas inmediatas
    - agent: hiperparámetros y estado del agente
    - metrics: métricas agregadas (historial recompensa, episodios, sps)
    - ui: flags de UI (sonido, visibilidad de chart)
    """

    def __init__(self, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._state: Dict[str, Any] = copy.deepcopy(
            defaults) if defaults else {}

    # --- API de acceso --- #
    def get(self, path: Union[str, List[Union[str, int]]]) -> Any:
        keys = self._coerce_path(path)
        ref: Any = self._state
        for key in keys:
            if isinstance(ref, dict):
                ref = ref.get(key)
            elif isinstance(ref, list) and isinstance(key, int) and 0 <= key < len(ref):
                ref = ref[key]
            else:
                return None
        return ref

    def set(self, path: Union[str, List[Union[str, int]]], value: Any) -> None:
        keys = self._coerce_path(path)
        if not keys:
            return
        ref: Any = self._state
        for key in keys[:-1]:
            if isinstance(ref, dict):
                if key not in ref or not isinstance(ref[key], (dict, list)):
                    ref[key] = {}
                ref = ref[key]
            elif isinstance(ref, list) and isinstance(key, int):
                while key >= len(ref):
                    ref.append({})
                ref = ref[key]
            else:
                raise KeyError("Ruta inválida para set")
        last = keys[-1]
        if isinstance(ref, dict):
            ref[last] = value
        elif isinstance(ref, list) and isinstance(last, int):
            while last >= len(ref):
                ref.append(None)
            ref[last] = value
        else:
            raise KeyError("Ruta inválida para set (final)")

    def update(self, path: Union[str, List[Union[str, int]]], delta: Dict[str, Any]) -> None:
        current = self.get(path) or {}
        if not isinstance(current, dict):
            current = {}
        merged = {**current, **delta}
        self.set(path, merged)

    def full(self) -> Dict[str, Any]:
        return self._state

    # --- helpers --- #
    @staticmethod
    def _coerce_path(path: Union[str, List[Union[str, int]]]) -> List[Union[str, int]]:
        if isinstance(path, str):
            return [p for p in path.split('.') if p]
        return list(path)

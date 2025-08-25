from __future__ import annotations

# Inicializa el registro importando plugins integrados
try:
    from . import registry as registry  # re-export
    # Importar plugin Q-Learning integrado para autorregistro
    from .ql import plugin as _ql_plugin  # noqa: F401
    from .random import plugin as _random_plugin  # noqa: F401
except Exception:
    # Permitir que el paquete se importe aun si faltan dependencias opcionales
    pass

__all__ = [
    "registry",
]

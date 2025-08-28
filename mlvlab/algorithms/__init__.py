# mlvlab/algorithms/__init__.py
from __future__ import annotations
import importlib
from pathlib import Path

# Re-exportar el registro para que sea accesible desde mlvlab.algorithms.registry
try:
    from . import registry as registry
except ImportError:
    pass

# Detección automática de plugins ---
# Busca en todas las subcarpetas de este directorio...
package_dir = Path(__file__).parent
for subdir in package_dir.iterdir():
    # Si es una carpeta y contiene un archivo 'plugin.py'...
    if subdir.is_dir() and (subdir / "plugin.py").exists():
        try:
            # ...intenta importarlo. La importación ejecutará 'register_algorithm()'.
            module_name = f"{__name__}.{subdir.name}.plugin"
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            # Permite que el paquete se importe aun si faltan dependencias opcionales
            pass

__all__ = ["registry"]

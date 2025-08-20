# # Importar automáticamente todos los entornos disponibles
# import os
# import importlib

# # Obtener todos los directorios de entornos
# env_dirs = [d for d in os.listdir(os.path.dirname(__file__))
#             if os.path.isdir(os.path.join(os.path.dirname(__file__), d))
#             and not d.startswith('__') and not d.startswith('.')]

# # Importar cada entorno
# for env_dir in env_dirs:
#     try:
#         importlib.import_module(f"mlvlab.envs.{env_dir}")
#     except ImportError as e:
#         # Silenciar errores de importación para entornos que no están completamente implementados
#         pass


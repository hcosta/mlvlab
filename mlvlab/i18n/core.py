"""
Sistema central de internacionalización para MLV-Lab.
"""

import os
import json
import locale
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class I18nManager:
    """
    Gestor de internacionalización que maneja la carga de traducciones
    y la detección automática del idioma del usuario.
    """

    def __init__(self):
        self.current_locale = self._detect_locale()
        self.translations = self._load_translations()
        self._fallback_translations = self._load_fallback_translations()

    def _detect_locale(self) -> str:
        """
        Detecta el idioma del sistema o usa configuración del usuario.
        Prioridad: MLVLAB_LOCALE > config file > system locale > 'en'
        """
        # 1. Variable de entorno MLVLAB_LOCALE
        env_locale = os.getenv('MLVLAB_LOCALE')
        if env_locale and env_locale in ['en', 'es']:
            return env_locale

        # 2. Configuración del usuario
        config_file = Path.home() / '.mlvlab' / 'config.json'
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                user_locale = config.get('locale')
                if user_locale in ['en', 'es']:
                    return user_locale
            except (json.JSONDecodeError, IOError):
                pass

        # 3. Idioma del sistema
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale and system_locale.startswith('es'):
                return 'es'
        except (locale.Error, AttributeError):
            pass

        return 'en'  # Fallback a inglés

    def _load_translations(self) -> Dict[str, Any]:
        """Carga las traducciones para el idioma actual"""
        try:
            locale_file = Path(__file__).parent / 'locales' / \
                self.current_locale / 'mlvlab.json'
            if locale_file.exists():
                with open(locale_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return {}

    def _load_fallback_translations(self) -> Dict[str, Any]:
        """Carga las traducciones en inglés como fallback"""
        try:
            fallback_file = Path(__file__).parent / \
                'locales' / 'en' / 'mlvlab.json'
            if fallback_file.exists():
                with open(fallback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return {}

    def t(self, key: str, **kwargs) -> str:
        """
        Traduce una clave con interpolación de variables.

        Args:
            key: Clave de traducción (ej: 'cli.help.main')
            **kwargs: Variables para interpolación

        Returns:
            Texto traducido o la clave original si no se encuentra
        """
        # Buscar en traducciones del idioma actual
        translation = self._get_nested_value(self.translations, key)

        # Si no se encuentra, usar fallback en inglés
        if translation is None:
            translation = self._get_nested_value(
                self._fallback_translations, key)

        # Si aún no se encuentra, devolver la clave
        if translation is None:
            return key

        # Interpolar variables si las hay
        try:
            return translation.format(**kwargs)
        except (KeyError, ValueError):
            return translation

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Obtiene un valor anidado usando notación de punto"""
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current if isinstance(current, str) else None

    def get_current_locale(self) -> str:
        """Retorna el idioma actual"""
        return self.current_locale

    def set_locale(self, locale: str) -> bool:
        """
        Cambia el idioma del sistema.

        Args:
            locale: Código de idioma ('en' o 'es')

        Returns:
            True si se cambió exitosamente, False en caso contrario
        """
        if locale not in ['en', 'es']:
            return False

        if locale != self.current_locale:
            self.current_locale = locale
            self.translations = self._load_translations()
            return True

        return False


# Instancia global del gestor de i18n
i18n = I18nManager()


def t(key: str, **kwargs) -> str:
    """
    Función de conveniencia para traducción.
    Equivalente a i18n.t(key, **kwargs)
    """
    return i18n.t(key, **kwargs)

import io
import cv2
import base64
import numpy as np
from PIL import Image
import importlib.resources
import importlib.util
from pathlib import Path
from typing import Optional, Any
from nicegui import ui, app


def setup_audio(env: Optional[Any] = None):
    """
    Configura NiceGUI para encontrar y reproducir los sonidos del paquete mlvlab.

    Esta función se encarga de:
    1. Encontrar la ruta a los recursos de audio dentro del paquete.
    2. Servir estáticamente esa carpeta para que el navegador pueda acceder a ella.
    3. Devolver una función 'play_sound' lista para usar.

    Retorna:
        Una función que puede ser llamada con el diccionario de sonido
        del entorno (ej: info['play_sound']).
    """
    # Resolver carpeta de assets (orden de preferencia):
    # 1) Si se proporciona env, buscar junto al módulo del entorno
    # 2) Si el paquete contiene '-', probar con underscore
    # 3) Si hay env.spec.id, inferir paquete y buscar mlvlab/envs/<pkg_us>/assets
    assets_dir: Optional[Path] = None
    attempted: list[str] = []

    if env is not None:
        try:
            mod_name = type(getattr(env, 'unwrapped', env)).__module__
            spec = importlib.util.find_spec(mod_name)
            if spec and spec.origin:
                env_dir = Path(spec.origin).parent
                candidate = env_dir / 'assets'
                attempted.append(str(candidate))
                if candidate.exists():
                    assets_dir = candidate
        except Exception:
            assets_dir = None

    # Si el módulo del entorno es mlvlab.envs.<pkg>.* y <pkg> contiene '-', probar underscore
    if assets_dir is None and env is not None:
        try:
            mod_name = type(getattr(env, 'unwrapped', env)).__module__
            parts = mod_name.split('.')
            if len(parts) >= 3 and parts[0] == 'mlvlab' and parts[1] == 'envs':
                pkg = parts[2]
                if '-' in pkg:
                    pkg_us = pkg.replace('-', '_')
                    spec2 = importlib.util.find_spec(f"mlvlab.envs.{pkg_us}")
                    if spec2 and spec2.origin:
                        env_dir2 = Path(spec2.origin).parent
                        cand = env_dir2 / 'assets'
                        attempted.append(str(cand))
                        if cand.exists():
                            assets_dir = cand
        except Exception:
            pass

    # Inferir por env_id si existe
    if assets_dir is None:
        try:
            env_id = getattr(getattr(env, 'spec', None), 'id',
                             '') if env is not None else ''
            if env_id:
                pkg = env_id.split('/')[-1]
                pkg_us = pkg.replace('-', '_')
                cand = importlib.resources.files(
                    'mlvlab') / 'envs' / pkg_us / 'assets'
                attempted.append(str(cand))
                if cand.exists():
                    assets_dir = Path(cand)
        except Exception:
            pass

    if assets_dir is None:
        try:
            dbg = " | intentados: " + "; ".join(attempted) if attempted else ""
        except Exception:
            dbg = ""
        print(
            f"⚠️ Advertencia: No se encontró carpeta de assets para audio. El sonido no funcionará.{dbg}")

        def no_sound(_):
            pass

        return no_sound

    # Servir carpeta encontrada bajo '/assets'
    try:
        app.add_static_files('/assets', str(assets_dir))
        print(f"✅ Recursos de audio servidos desde: {assets_dir}")
    except Exception:
        pass

    # Si todo fue bien, definimos y devolvemos la función real
    def play_sound_from_info(sound_data: dict):
        """
        Lee un diccionario con datos de sonido y lo reproduce en el navegador.
        Esta función es generada y devuelta por setup_nicegui_audio.
        """
        filename = sound_data.get('filename')
        if not filename:
            return

        volume_percent = sound_data.get('volume', 100)
        try:
            volume_percent = max(0, min(100, int(volume_percent)))
        except Exception:
            volume_percent = 100
        volume_js = volume_percent / 100.0

        js_command = f"""
            (() => {{
                const sound = new Audio('/assets/{filename}');
                try {{ sound.volume = {volume_js}; }} catch (e) {{}}
                sound.muted = {volume_js} <= 0;
                sound.play().catch(e => console.error("Error al reproducir audio:", e.name, e.message));
            }})()
        """
        ui.run_javascript(js_command)

    return play_sound_from_info


def create_reward_chart(container, number=None) -> ui.highchart:
    """
    Crea y devuelve un gráfico estándar para mostrar la recompensa por episodio.
    El título cambia si se proporciona un número.
    """
    # 1. Determina el texto del título basado en el parámetro 'number'
    if number is not None:
        title_text = f'Últimas {number} recompensas'
    else:
        title_text = 'Últimas recompensas'

    # 2. Usa la variable title_text para construir las opciones del gráfico
    chart_options = {
        'title': {'text': title_text},
        'chart': {'type': 'line'},
        'series': [{'name': 'Recompensa', 'data': []}],
        'credits': {'enabled': False},
        'xAxis': {'title': {'text': 'Episodio'}},
        'yAxis': {'title': {'text': None}},
        'accessibility': {'enabled': False},
        'legend': {'enabled': False}
    }
    with container:
        chart = ui.highchart(chart_options).classes('w-full h-64')
    return chart


def frame_to_base64_src(frame: np.ndarray) -> str:
    """Convierte un frame (NumPy array) a una cadena de texto Base64 para una imagen en HTML."""
    pil_img = Image.fromarray(frame)
    with io.BytesIO() as buffered:
        # PNG más rápido: sin optimización y compresión mínima
        pil_img.save(buffered, format="PNG", optimize=False, compress_level=0)
        b64_img = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{b64_img}"


def frame_to_webp_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    """Convierte un frame (NumPy array RGB) a bytes WebP (mucho más ligero que PNG)."""
    pil_img = Image.fromarray(frame)
    with io.BytesIO() as buffered:
        pil_img.save(buffered, format="WEBP",
                     quality=int(max(1, min(100, quality))))
        return buffered.getvalue()


def encode_frame_fast_jpeg(frame_np: np.ndarray, quality: int = 80) -> bytes:
    """Codifica un frame (array de numpy RGB) a bytes JPEG usando OpenCV."""
    # OpenCV trabaja con el formato de color BGR por defecto, mientras que los
    # entornos de Gym/renderizado suelen dar RGB. La conversión es crucial.
    if frame_np.ndim == 3 and frame_np.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame_np  # Asumir que ya está en el formato correcto si no es RGB

    # Parámetros de codificación, incluyendo la calidad del JPEG (0-100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Codificar la imagen a un formato de memoria
    result, encoded_image = cv2.imencode('.jpg', frame_bgr, encode_param)

    if result:
        # Convertir el array de numpy resultante a bytes
        return encoded_image.tobytes()
    else:
        # Devolver bytes vacíos si la codificación falla
        return b""

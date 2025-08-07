# mlvlab/utils.py

import io
import base64
import numpy as np
from PIL import Image
import importlib.resources
from nicegui import ui, app


def setup_audio():
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
    try:
        # Encuentra la ruta a la carpeta de assets de forma robusta
        resources_path = importlib.resources.files(
            'mlvlab') / 'envs' / 'ant' / 'assets'

        # Le dice a NiceGUI que sirva esa carpeta bajo la URL '/assets'
        app.add_static_files('/assets', str(resources_path))
        print(
            f"✅ Recursos de audio de 'mlvlab' servidos desde: {resources_path}")

    except ModuleNotFoundError:
        print("⚠️ Advertencia: No se pudo encontrar el directorio de recursos de 'mlvlab'. El sonido no funcionará.")
        # Si no encontramos los sonidos, devolvemos una función que no hace nada

        def no_sound(_):
            pass
        return no_sound

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
        volume_js = volume_percent / 100.0

        js_command = f"""
            (() => {{
                const sound = new Audio('/assets/{filename}');
                sound.volume = {volume_js};
                sound.play().catch(e => console.error("Error al reproducir audio:", e));
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
        'legend': {'enabled': False}
    }
    with container:
        chart = ui.highchart(chart_options).classes('w-full h-64')
    return chart


def frame_to_base64_src(frame: np.ndarray) -> str:
    """Convierte un frame (NumPy array) a una cadena de texto Base64 para una imagen en HTML."""
    pil_img = Image.fromarray(frame)
    with io.BytesIO() as buffered:
        pil_img.save(buffered, format="PNG")
        b64_img = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{b64_img}"

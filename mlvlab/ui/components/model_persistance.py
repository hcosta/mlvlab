# mlvlab/ui/components/model_persistence.py

from __future__ import annotations
import io
import time
from pathlib import Path
import numpy as np
from nicegui import ui
import base64
from .base import UIComponent, ComponentContext
from ..state import StateStore


class ModelPersistence(UIComponent):
    def __init__(self, default_filename: str = "agent_model.npz"):
        super().__init__()
        self.default_filename = f"{Path(default_filename).stem}.npz"

    def render(self, state: StateStore, context: ComponentContext) -> None:
        with ui.card().classes('w-full mb-1'):
            ui.label('Gestión del Modelo').classes(
                'text-lg font-semibold text-center w-full')

            # La lógica de 'save_model' no cambia
            def save_model():
                agent = context.agent
                env = context.env
                if not hasattr(agent, 'save'):
                    ui.notify("El agente no tiene el método 'save'.",
                              type='negative')
                    return
                try:
                    data_to_save = agent.save()
                    if not isinstance(data_to_save, dict):
                        ui.notify(
                            "El método 'save' debe devolver un diccionario.", type='warning')
                        return
                except Exception as ex:
                    ui.notify(f"Error en agent.save(): {ex}", type='negative')
                    return
                full_metadata = {
                    'env_id': getattr(env.spec, 'id', 'unknown_env'),
                    'agent_data': data_to_save,
                    'simulation_state': {
                        'episodes_completed': state.get(['metrics', 'episodes_completed']),
                        'total_steps': state.get(['sim', 'total_steps']),
                        'seed': state.get(['sim', 'seed']),
                    }, 'timestamp': time.time(),
                }
                with io.BytesIO() as buffer:
                    np.savez_compressed(
                        buffer, **{'metadata': np.array(full_metadata)})
                    buffer.seek(0)
                    file_bytes = buffer.getvalue()
                file_base64 = base64.b64encode(file_bytes).decode('utf-8')
                js_command = f"saveModelWithNativeDialog('{file_base64}', '{self.default_filename}')"
                ui.run_javascript(js_command)
                ui.notify("Abriendo diálogo de guardado...", type='info')

            # La lógica de 'load_model' no cambia
            async def load_model(e: ui.UploadEventArguments):
                agent = context.agent
                if not hasattr(agent, 'load'):
                    ui.notify("El agente no tiene método 'load'.",
                              type='negative')
                    return
                try:
                    content = e.content.read()
                    with io.BytesIO(content) as buffer:
                        data = np.load(buffer, allow_pickle=True)
                        metadata = data['metadata'].item()
                    agent_data = metadata.get('agent_data', {})
                    agent.load(agent_data)
                    sim_state = metadata.get('simulation_state', {})
                    episodes = sim_state.get('episodes_completed', 0)
                    steps = sim_state.get('total_steps', 0)
                    seed = sim_state.get('seed')
                    agent_hparams = agent_data.get('hyperparameters', {})
                    state.update('agent', agent_hparams)
                    state.set(['metrics', 'episodes_completed'], episodes)
                    state.set(['sim', 'total_steps'], steps)
                    state.set(['sim', 'active_model_name'], e.name)
                    if seed:
                        state.set(['sim', 'seed'], seed)
                    state.set(['sim', 'current_episode_reward'], 0.0)
                    state.set(['metrics', 'reward_history'], [])
                    state.set(['sim', 'command'], 'pause')
                    with context.env_lock:
                        context.env.reset(seed=seed)
                    state.set(['sim', 'command'], 'run')
                    ui.notify(
                        f"Modelo '{e.name}' cargado y sincronizado.", type='positive')
                except Exception as ex:
                    ui.notify(
                        f"Error al cargar el modelo: {ex}", type='negative')

            # Se elimina la función 'clear_model' y el botón 'Nuevo'.
            # Se elimina el 'label' de "Modelo Activo".
            with ui.column().classes('w-full items-stretch gap-y-2'):
                ui.button('GUARDAR', on_click=save_model,
                          icon='save').props('color=primary')

                upload_widget = ui.upload(on_upload=load_model, label="CARGAR", auto_upload=True).props(
                    'icon=upload color=secondary').classes('w-full')

            def check_and_clear_upload():
                if state.get(['ui', 'clear_upload']):
                    upload_widget.clear()
                    upload_widget.reset()
                    state.set(['ui', 'clear_upload'], False)
            ui.timer(0.5, check_and_clear_upload)

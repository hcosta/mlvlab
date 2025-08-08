from __future__ import annotations

# mlvlab/web/app.py
from typing import Optional, Dict, Any
import time
import random
import json
import uuid
import gymnasium as gym
import numpy as np
from nicegui import ui, app

from mlvlab.helpers.ng import setup_audio, create_reward_chart, setup_keyboard


def _build_colors():
    return {
        'grid': 0x222222,
        'ant': 0xff3333,
        'food': 0x33ff33,
        'obstacle': 0x777777,
        'heat_low': 0x003300,
        'heat_high': 0x00cc00,
    }


def _state_to_heat_color(q_table: np.ndarray, s_index: int, low_color: int, high_color: int) -> int:
    if q_table is None:
        return low_color
    qs = q_table[s_index]
    qv = float(np.max(qs))
    q_min = float(np.min(q_table))
    q_max = float(np.max(q_table))
    if q_max <= q_min:
        t = 0.0
    else:
        t = (qv - q_min) / (q_max - q_min)
    # Interpolar de low a high en RGB
    def _ch(c, s):
        return (c >> s) & 0xFF
    r = int(_ch(low_color, 16) + t * (_ch(high_color, 16) - _ch(low_color, 16)))
    g = int(_ch(low_color, 8) + t * (_ch(high_color, 8) - _ch(low_color, 8)))
    b = int(_ch(low_color, 0) + t * (_ch(high_color, 0) - _ch(low_color, 0)))
    return (r << 16) | (g << 8) | b


class AntPixi2D:
    """Renderer 2D usando Pixi.js embebido en NiceGUI."""
    def __init__(self, grid_size: int, size_px: int = 700):
        self.grid_size = grid_size
        self.size_px = size_px
        self.colors = _build_colors()
        self.dom_id = f"pixi_{uuid.uuid4().hex}"

        # Inyectar Pixi.js en el head (una vez)
        ui.add_head_html('<script src="https://cdnjs.cloudflare.com/ajax/libs/pixi.js/7.4.0/pixi.min.js" crossorigin="anonymous"></script>')

        # Contenedor
        self.container = ui.html(f'<div id="{self.dom_id}" style="width:{size_px}px;height:{size_px}px;"></div>')

        # JS helper para inicializar y actualizar
        init_js = f"""
        (function(){{
            if (!window._mlv_pixi) window._mlv_pixi = {{}};
            if (window._mlv_pixi['{self.dom_id}']) return; // ya inicializado
            const root = document.getElementById('{self.dom_id}');
            const app = new PIXI.Application({{ width: {size_px}, height: {size_px}, background: 0x000000, antialias: true }});
            root.innerHTML = '';
            root.appendChild(app.view);
            const state = {{}};
            const cell = Math.floor({size_px} / {self.grid_size});

            // capas
            const heatLayer = new PIXI.Container();
            const obstacleLayer = new PIXI.Container();
            const entityLayer = new PIXI.Container();
            app.stage.addChild(heatLayer, obstacleLayer, entityLayer);

            // celdas calor
            state.heatRects = [];
            for (let y=0; y<{self.grid_size}; y++) {{
                const row = [];
                for (let x=0; x<{self.grid_size}; x++) {{
                    const g = new PIXI.Graphics();
                    g.rect(0,0,cell-2,cell-2).fill({self.colors['grid']});
                    g.x = x*cell + 1; g.y = y*cell + 1;
                    heatLayer.addChild(g);
                    row.push(g);
                }}
                state.heatRects.push(row);
            }}

            // entidades
            const ant = new PIXI.Graphics().rect(0,0,cell-2,cell-2).fill({self.colors['ant']});
            entityLayer.addChild(ant);
            const food = new PIXI.Graphics().rect(0,0,cell-2,cell-2).fill({self.colors['food']});
            entityLayer.addChild(food);

            state.ant = ant; state.food = food; state.obstacles = [];

            // util: color int -> hex string
            function toHex(c){{return '#' + ('000000'+c.toString(16)).slice(-6);}}

            window._mlv_pixi['{self.dom_id}'] = {{ app, state, cell, heatLayer, obstacleLayer, entityLayer, toHex }};
        }})();
        """
        ui.run_javascript(init_js)

    def update(self, state: Dict[str, Any]):
        # Precalcular colores de calor si hay q_table
        gs = int(state['grid_size'])
        q_table = np.array(state['q_table']) if state.get('q_table') is not None else None
        heat_colors: list[int] = []
        if q_table is not None:
            for s in range(gs*gs):
                heat_colors.append(_state_to_heat_color(q_table, s, self.colors['heat_low'], self.colors['heat_high']))
        # Enviar estado a JS
        payload = {
            'grid_size': gs,
            'ant': state['ant'],
            'food': state['food'],
            'obstacles': state['obstacles'],
            'heat_colors': heat_colors,
        }
        js = f"""
        (function(){{
            const ctx = window._mlv_pixi['{self.dom_id}']; if(!ctx) return;
            const s = {json.dumps(payload)};
            const cell = ctx.cell; const st = ctx.state;
            // calor
            if (s.heat_colors && s.heat_colors.length === s.grid_size*s.grid_size) {{
                for (let y=0; y<s.grid_size; y++) {{
                    for (let x=0; x<s.grid_size; x++) {{
                        const idx = y*s.grid_size + x;
                        const col = s.heat_colors[idx];
                        const g = st.heatRects[y][x];
                        g.clear(); g.rect(0,0,cell-2,cell-2).fill(col);
                    }}
                }}
            }}
            // obstáculos
            ctx.obstacleLayer.removeChildren(); st.obstacles = [];
            for (const [ox, oy] of s.obstacles) {{
                const ob = new PIXI.Graphics().rect(0,0,cell-2,cell-2).fill({self.colors['obstacle']});
                ob.x = ox*cell + 1; ob.y = oy*cell + 1; ctx.obstacleLayer.addChild(ob);
                st.obstacles.push(ob);
            }}
            // ant & food
            if (s.ant) {{ st.ant.x = s.ant[0]*cell + 1; st.ant.y = s.ant[1]*cell + 1; }}
            if (s.food) {{ st.food.x = s.food[0]*cell + 1; st.food.y = s.food[1]*cell + 1; }}
        }})();
        """
        ui.run_javascript(js)


class AntScene3D:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.colors = _build_colors()
        with ui.scene(width=900, height=700) as scene:
            self.scene = scene
            self.scene.lights()
            self.camera = scene.camera
            self.camera.move(0, -grid_size*1.5, grid_size*1.2)
            self.camera.look_at([grid_size/2, grid_size/2, 0])
            grid = scene.add_plane(width=grid_size, height=grid_size, color=self.colors['grid'])
            grid.rotation.set(-np.pi/2, 0, 0)
            grid.position.set(grid_size/2, 0, grid_size/2)
            self.cells: list[Any] = []
            for y in range(grid_size):
                row = []
                for x in range(grid_size):
                    tile = scene.add_box(1, 1, 0.1, color=self.colors['grid'])
                    tile.position.set(x+0.5, y+0.5, 0.05)
                    row.append(tile)
                self.cells.append(row)
            self.ant = scene.add_sphere(0.45, color=self.colors['ant'])
            self.ant.position.set(0.5, 0.5, 0.45)
            self.food = scene.add_sphere(0.45, color=self.colors['food'])
            self.food.position.set(0.5, 0.5, 0.45)
            self.obstacles: list[Any] = []

    def update(self, state: Dict[str, Any]):
        gs = state['grid_size']
        ant = state['ant']
        food = state['food']
        obstacles = state['obstacles']
        q_table = np.array(state['q_table']) if state.get('q_table') is not None else None
        if q_table is not None:
            for s in range(gs*gs):
                x = s % gs
                y = s // gs
                color = _state_to_heat_color(q_table, s, self.colors['heat_low'], self.colors['heat_high'])
                self.cells[y][x].material.color = color
        for o in self.obstacles:
            o.remove()
        self.obstacles.clear()
        for (ox, oy) in obstacles:
            cube = self.scene.add_box(1, 1, 0.6, color=self.colors['obstacle'])
            cube.position.set(ox+0.5, oy+0.5, 0.3)
            self.obstacles.append(cube)
        if ant:
            self.ant.position.set(ant[0]+0.5, ant[1]+0.5, 0.45)
        if food:
            self.food.position.set(food[0]+0.5, food[1]+0.5, 0.45)


def run_web(env_id: str, seed: Optional[int] = None, mode: str = "2d"):
    """Lanza NiceGUI con una escena 2D (Pixi.js) o 3D (Three.js) y control del entorno Gym."""
    # Preparar env
    env = gym.make(env_id, render_mode=None)
    app_state = {
        'running': True,
        'last_step_time': time.time(),
        'speed': 15.0,  # steps/sec
        'accum_reward': 0.0,
        'eps': 0,
    }

    obs, info = env.reset(seed=seed if seed is not None else random.randint(0, 1_000_000))

    # Audio
    play_sound = setup_audio()

    # UI
    @ui.page('/')
    def main_page():
        ui.label(f'Visualizador WebGL para {env_id}').classes('text-2xl font-bold my-2')
        with ui.row().classes('items-start w-full gap-4'):
            # Panel izquierdo: controles
            with ui.column().classes('w-1/4'):
                ui.switch('Ejecutando').bind_value(app_state, 'running')
                ui.slider(min=1, max=60, value=app_state['speed']).bind_value(app_state, 'speed').props('label=Velocidad (steps/s)')
                with ui.card():
                    ui.label('Controles: WASD / Flechas (mlv play usa PyGame como antes)')
                    ui.label('Botones: ← ↓ ↑ →')
                    with ui.row():
                        ui.button('←', on_click=lambda: _apply_action(2))
                        ui.button('↓', on_click=lambda: _apply_action(1))
                        ui.button('↑', on_click=lambda: _apply_action(0))
                        ui.button('→', on_click=lambda: _apply_action(3))
                with ui.card():
                    ui.label('Estadísticas')
                    ui.label().bind_text_from(app_state, 'eps', lambda v: f'Episodios: {v}')
                    ui.label().bind_text_from(app_state, 'accum_reward', lambda v: f'Recompensa episodio: {v:.1f}')
            # Panel central: escena
            with ui.column().classes('w-2/4 items-center'):
                nonlocal_scene_container = {'obj': None}
                gs = int(env.unwrapped.GRID_SIZE)
                if mode.lower() == '3d':
                    nonlocal_scene_container['obj'] = AntScene3D(gs)
                else:
                    nonlocal_scene_container['obj'] = AntPixi2D(gs)
                scene_obj = nonlocal_scene_container['obj']
                # primer render
                scene_obj.update(env.unwrapped.get_render_state())
            # Panel derecho: gráfico
            with ui.column().classes('w-1/4'):
                chart = create_reward_chart(ui.card(), number=100)

        # Teclado (opcional en web)
        def on_key(key: str):
            mapping = {
                'ArrowUp': 0, 'w': 0, 'W': 0,
                'ArrowDown': 1, 's': 1, 'S': 1,
                'ArrowLeft': 2, 'a': 2, 'A': 2,
                'ArrowRight': 3, 'd': 3, 'D': 3,
            }
            if key in mapping:
                _apply_action(mapping[key])
        setup_keyboard(on_key)

        # Timers
        def sim_tick():
            if not app_state['running']:
                return
            now = time.time()
            if now - app_state['last_step_time'] < 1.0 / max(1.0, float(app_state['speed'])):
                return
            app_state['last_step_time'] = now
            # política aleatoria para demo (se puede sustituir por agente)
            action = getattr(main_page, '_pending_action', None)
            if action is None:
                action = env.action_space.sample()
            setattr(main_page, '_pending_action', None)
            obs, reward, terminated, truncated, info = env.step(action)
            app_state['accum_reward'] += reward
            if 'play_sound' in info and info['play_sound']:
                try:
                    play_sound(info['play_sound'])
                except Exception:
                    pass
            if terminated or truncated:
                env.reset()
                app_state['eps'] += 1
                app_state['accum_reward'] = 0.0

        def render_tick():
            # Actualizar escena
            scene_obj.update(env.unwrapped.get_render_state())

        ui.timer(0.001, sim_tick)
        ui.timer(1/30, render_tick)

        def _apply_action(a: int):
            setattr(main_page, '_pending_action', a)

    ui.run(title=f"MLV-Lab WebGL: {env_id}", dark=False, reload=False, show=True)
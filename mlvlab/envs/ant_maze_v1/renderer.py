# mlvlab/envs/ant_maze_v1/renderer.py
import time
import math
import numpy as np
import random
from pathlib import Path

try:
    from .game import MazeGame
except ImportError:
    from game import MazeGame


class ParticleFX:
    def __init__(self, x, y, dx, dy, lifespan, size, color, p_type="dust", gravity=0.2):
        self.x, self.y, self.dx, self.dy = x, y, dx, dy
        self.lifespan, self.age = lifespan, 0.0
        self.size = size
        self.color = color if len(color) == 4 else color + (255,)
        self.p_type, self.gravity = p_type, gravity

    def update(self, delta_time):
        self.age += delta_time
        if self.age >= self.lifespan:
            return
        self.dy -= self.gravity * delta_time * 60
        self.x += self.dx * delta_time * 60
        self.y += self.dy * delta_time * 60


class MazeRenderer:
    def __init__(self) -> None:
        self.window = None
        self.game: MazeGame | None = None
        self.CELL_SIZE = 40
        self.WIDTH, self.HEIGHT = 0, 0
        self.COLOR_FLOOR = (139, 119, 90)
        self.COLOR_ANT = (192, 57, 43)
        self.COLOR_GOAL = (40, 25, 10)
        self.COLOR_WALL = (89, 69, 40)
        self.COLOR_PARTICLE_DUST = (210, 180, 140)

        # --- INICIO: NUEVAS PROPIEDADES PARA LA VARIACIÓN DE LA HORMIGA ---
        self.randomized_ant_color = self.COLOR_ANT
        self.ant_size_multipliers = {
            'head': 1.0, 'thorax': 1.0, 'abdomen': 1.0}
        # --- FIN: NUEVAS PROPIEDADES PARA LA VARIACIÓN DE LA HORMIGA ---

        self.ant_prev_pos, self.ant_display_pos = None, None
        self.ant_current_angle, self.ant_scale = 0.0, 1.0
        self.last_time = time.time()
        self.particles: list[ParticleFX] = []
        self.anthill_hole_visual_center = None
        self.was_colliding_last_frame = False
        self._q_value_text_objects: list = []

        # Animación de Éxito (Success)
        self.in_success_transition, self.success_transition_time = False, 0.0
        self.SUCCESS_TRANSITION_DURATION = 1.5

        # CAMBIO: Lógica de animación de Muerte (Importado de AntLost)
        self.in_death_transition = False
        # Usado para sincronizar el último movimiento antes de morir
        self.death_pending_completion = False
        self.death_transition_time = 0.0
        self.DEATH_TRANSITION_DURATION = 2.0  # Duración exacta de AntLost
        self.DEATH_PAUSE_DURATION = 0.01      # Pausa exacta de AntLost
        self.ant_vertical_flip = False
        # Controla la transparencia (255=opaco, 0=invisible)
        self.ant_alpha = 255

        self.initialized, self.debug_mode = False, False
        try:
            self.rng_visual = np.random.default_rng()
        except AttributeError:
            self.rng_visual = np.random.RandomState()
        self.arcade, self._headless_mode = None, False
        self.wall_sprite_list: "arcade.SpriteList" | None = None
        self.ASSETS_PATH = Path(__file__).parent / "assets"

    def _lazy_import_arcade(self):
        if self.arcade is None:
            try:
                import arcade
                self.arcade = arcade
            except ImportError:
                raise ImportError("Se requiere 'arcade' para el renderizado.")

    def _get_angle_from_action(self, action):
        return {0: 90, 1: 270, 2: 180, 3: 0}.get(action, 0)

    def _initialize(self, game: MazeGame, render_mode: str):
        self._lazy_import_arcade()
        self.game = game
        if game.grid_size * self.CELL_SIZE > 800:
            self.CELL_SIZE = 800 // game.grid_size
        self.WIDTH, self.HEIGHT = game.grid_size * \
            self.CELL_SIZE, game.grid_size * self.CELL_SIZE
        if self.window is None:
            visible = render_mode == "human"
            title = "Ants Saga - Dungeons & Pheromones - MLVisual®"
            try:
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, title, visible=visible)
                if (self._headless_mode or render_mode == "rgb_array") and visible:
                    self.window.set_visible(False)
            except Exception:
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, title)
            self.arcade.set_background_color(self.COLOR_FLOOR)
        if self.ant_display_pos is None:
            self.ant_display_pos = list(game.ant_pos.astype(float))
            self.ant_prev_pos = list(game.ant_pos.astype(float))
            self.ant_scale = 1.0
            self.ant_current_angle = self._get_angle_from_action(
                game.last_action)
        self._setup_static_elements()
        self.initialized = True

    def reset(self, full_reset=False):
        if full_reset:
            self.initialized = False
            self.wall_sprite_list = None

        # --- INICIO: LÓGICA DE VARIACIÓN DE APARIENCIA DE LA HORMIGA ---
        # (Este bloque se mantiene igual que en tu código original)
        try:
            r_var = self.rng_visual.integers(-20, 21)
            g_var = self.rng_visual.integers(-20, 21)
            b_var = self.rng_visual.integers(-20, 21)
        except AttributeError:  # Fallback para versiones antiguas de numpy
            r_var = self.rng_visual.randint(-20, 21)
            g_var = self.rng_visual.randint(-20, 21)
            b_var = self.rng_visual.randint(-20, 21)

        r = max(0, min(255, self.COLOR_ANT[0] + r_var))
        g = max(0, min(255, self.COLOR_ANT[1] + g_var))
        b = max(0, min(255, self.COLOR_ANT[2] + b_var))
        self.randomized_ant_color = (r, g, b)

        # Generar multiplicadores de tamaño para las partes del cuerpo (90% a 115%).
        self.ant_size_multipliers = {
            'head': self.rng_visual.uniform(0.65, 1.35),
            'thorax': self.rng_visual.uniform(0.7, 1.35),
            'abdomen': self.rng_visual.uniform(0.65, 1.35)
        }
        # --- FIN: LÓGICA DE VARIACIÓN DE APARIENCIA ---

        # Siempre reiniciamos el estado visual de la hormiga
        # al inicio de CADA episodio.

        # Aseguramos que tenga un valor inicial (Comprobamos si self.game existe)
        if self.game:
            self.ant_display_pos = list(self.game.ant_pos.astype(float))
            self.ant_prev_pos = list(self.game.ant_pos.astype(float))
            self.ant_current_angle = self._get_angle_from_action(
                self.game.last_action)
        else:
            # Mantenemos None si el juego no está inicializado aún
            pass

        # Aseguramos que sea visible
        self.ant_scale = 1.0

        self.last_time = time.time()
        self.particles = []
        self._q_value_text_objects = []
        self.anthill_hole_visual_center = None
        self.was_colliding_last_frame = False

        # Reset de Éxito
        self.in_success_transition = False
        self.success_transition_time = 0.0

        # CAMBIO: Reset de Muerte
        self.in_death_transition = False
        self.death_pending_completion = False
        self.death_transition_time = 0.0
        self.ant_vertical_flip = False
        self.ant_alpha = 255

    def _cell_to_pixel(self, x_cell: float, y_cell: float):
        x_px = x_cell * self.CELL_SIZE + self.CELL_SIZE / 2
        y_px = (self.game.grid_size - 1 - y_cell) * \
            self.CELL_SIZE + self.CELL_SIZE / 2
        return x_px, y_px

    def _setup_static_elements(self):
        """
        Crea la SpriteList para los muros, asignando y rotando tiles de
        forma aleatoria pero determinista para cada mapa.
        """
        wall_tile_paths = list(self.ASSETS_PATH.glob("tile_wall_*.png"))
        if not wall_tile_paths:
            raise FileNotFoundError(f"No se encontraron imágenes de muros en '{self.ASSETS_PATH}'. "
                                    f"Asegúrate de ejecutar el script create_wall_asset.py primero.")
        wall_tile_paths.sort()
        map_hash = hash(frozenset(self.game.walls))
        seeded_rng = random.Random(map_hash)
        self.wall_sprite_list = self.arcade.SpriteList()
        for wall_x, wall_y in self.game.walls:
            cx, cy = self._cell_to_pixel(wall_x, wall_y)
            random_tile_path = seeded_rng.choice(wall_tile_paths)
            random_angle = seeded_rng.choice([0, 90, 180, 270])
            wall_sprite = self.arcade.Sprite(
                random_tile_path,
                center_x=cx,
                center_y=cy,
                angle=random_angle
            )
            self.wall_sprite_list.append(wall_sprite)

    def _pixel_to_cell(self, x_px: float, y_px: float):
        x_cell = (x_px-self.CELL_SIZE/2)/self.CELL_SIZE
        y_cell = self.game.grid_size-1-(y_px-self.CELL_SIZE/2)/self.CELL_SIZE
        return x_cell, y_cell

    def start_success_transition(self):
        # CAMBIO: Comprobamos que la animación de muerte no esté activa o pendiente.
        if not self.in_success_transition and not self.is_in_death_transition():
            self.in_success_transition, self.success_transition_time = True, 0.0

    # (Se mantiene igual)
    def is_in_success_transition(self) -> bool:
        return self.in_success_transition

    def start_death_transition(self):
        """Solicita el inicio de la animación de muerte. Usa el sistema de 'pending'
           para sincronizar con el último movimiento."""
        # No iniciar si ya estamos en otra animación (muerte o éxito)
        if not self.in_death_transition and not self.death_pending_completion and not self.in_success_transition:
            self.death_pending_completion = True

    def is_in_death_transition(self) -> bool:
        """Indica si la animación de muerte está activa o pendiente de sincronización."""
        return self.in_death_transition or self.death_pending_completion

    def _update_rotation(self, delta_time, target_angle):
        diff = target_angle-self.ant_current_angle
        while diff < -180:
            diff += 360
        while diff > 180:
            diff -= 360
        if abs(diff) > 0.1:
            self.ant_current_angle += diff*(1.0-math.exp(-delta_time*25.0))
        else:
            self.ant_current_angle = target_angle
        self.ant_current_angle %= 360

    def _update_success_transition(self, delta_time: float):
        if not self.in_success_transition:
            return
        self.success_transition_time += delta_time
        progress = self.success_transition_time / self.SUCCESS_TRANSITION_DURATION
        if progress >= 1.0:
            # CAMBIO: Aseguramos que desaparezca completamente (scale 0 y alpha 0).
            self.in_success_transition, self.ant_scale = False, 0.0
            self.ant_alpha = 0
            return

        if self.anthill_hole_visual_center:
            target_x_px, target_y_px = self.anthill_hole_visual_center
            # CAMBIO: Usamos un ajuste relativo al tamaño de celda en vez de 11px fijos.
            vertical_adjustment = self.CELL_SIZE * \
                0.275  # (11px / 40px = 0.275)
            target_x_cell, target_y_cell = self._pixel_to_cell(
                target_x_px, target_y_px - vertical_adjustment)
            target_pos = [target_x_cell, target_y_cell]
        else:
            target_pos = list(self.game.goal_pos.astype(float))

        lerp = 1.0 - math.exp(-delta_time * 10.0)
        self.ant_display_pos[0] += (target_pos[0] -
                                    self.ant_display_pos[0]) * lerp
        self.ant_display_pos[1] += (target_pos[1] -
                                    self.ant_display_pos[1]) * lerp

        def easeInOutCubic(t): return 4 * t * t * \
            t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
        self.ant_scale = 1.0 - easeInOutCubic(progress)

    def _update_animations(self, delta_time: float):
        if self.ant_display_pos is None:
            return

        # CAMBIO: 1. Priorizamos las animaciones de fin de escena.
        if self.in_success_transition:
            self._update_success_transition(delta_time)
            return

        if self.in_death_transition:
            self._update_death_transition(delta_time)
            return

        # 2. Movimiento normal y sincronización
        target_pos = list(self.game.ant_pos.astype(float))
        if target_pos != self.ant_prev_pos:
            self.ant_prev_pos = list(self.ant_display_pos)

        dx, dy = target_pos[0] - \
            self.ant_display_pos[0], target_pos[1] - self.ant_display_pos[1]

        # CAMBIO: Calculamos si estamos en el objetivo (necesario para sincronización).
        distance = math.sqrt(dx**2 + dy**2)
        is_at_target = distance < 0.001

        # Actualización de posición
        if not is_at_target:
            lerp = 1.0 - math.exp(-delta_time * 15.0)
            self.ant_display_pos[0] += dx * lerp
            self.ant_display_pos[1] += dy * lerp
        else:
            self.ant_display_pos, self.ant_prev_pos = list(
                target_pos), list(target_pos)

        # Actualización de rotación
        target_angle = self._get_angle_from_action(self.game.last_action)
        if self.game.last_action in [0, 1, 2, 3]:
            self._update_rotation(
                delta_time, target_angle)

        # CAMBIO: 3. Sincronización de la muerte pendiente (Importado de AntLost)
        # Si la muerte está pendiente Y la hormiga ha terminado su último paso (is_at_target).
        if self.death_pending_completion and is_at_target:
            self.death_pending_completion = False
            self.in_death_transition = True
            self.death_transition_time = 0.0
            # Reiniciar estado visual para la muerte
            self.ant_vertical_flip = False
            self.ant_alpha = 255
            self.ant_scale = 1.0
            # Asegurar el ángulo final correcto
            self.ant_current_angle = target_angle

        # Colisiones (Se mantiene igual)
        if self.game.collided and not self.was_colliding_last_frame:
            self._spawn_collision_particles()
        self.was_colliding_last_frame = self.game.collided

    def _update_death_transition(self, delta_time: float):
        """Actualiza la lógica de la animación de muerte (temblor, volteo, fade)."""
        if not self.in_death_transition:
            return

        self.death_transition_time += delta_time
        progress = min(1.0, self.death_transition_time /
                       self.DEATH_TRANSITION_DURATION)

        # Lógica exacta de AntLost para el timing:

        # Fase 1: Temblor (0% a 25%) - El efecto visual se gestiona en _draw_ant

        # Fase 2: Volteo (inicia al 25%)
        if progress >= 0.25:
            self.ant_vertical_flip = True

        # Fase 3: Desvanecimiento (35% a 100%)
        if progress >= 0.35:
            fade_progress = (progress - 0.35) / 0.65
            min_alpha = 60  # Mantenemos el mínimo de AntLost
            self.ant_alpha = int(255 - (255 - min_alpha) * fade_progress)

        # Finalización
        total_duration = self.DEATH_TRANSITION_DURATION + self.DEATH_PAUSE_DURATION
        if self.death_transition_time >= total_duration:
            self.in_death_transition = False
            self.ant_alpha = 0  # Invisible al final
            self.ant_scale = 0.0  # Aseguramos consistencia

    def _update_particles(self, delta_time: float):
        for p in self.particles:
            p.update(delta_time)
        self.particles = [p for p in self.particles if p.age < p.lifespan]

    def _spawn_collision_particles(self):
        if not self.ant_display_pos:
            return
        cx, cy = self._cell_to_pixel(*self.ant_display_pos)
        action, offset = self.game.last_action, self.CELL_SIZE*0.3
        iv, sx, sy = [0, 0], cx, cy
        if action == 0:
            iv, sy = [0, -1], cy+offset
        elif action == 1:
            iv, sy = [0, 1], cy-offset
        elif action == 2:
            iv, sx = [1, 0], cx-offset
        elif action == 3:
            iv, sx = [-1, 0], cx+offset
        for _ in range(15):
            s, ao = self.rng_visual.uniform(
                0.5, 2.5), self.rng_visual.uniform(-0.8, 0.8)
            dx, dy = (iv[0]+ao)*s, (iv[1]+abs(ao))*s
            p = ParticleFX(sx, sy, dx, dy, self.rng_visual.uniform(
                1.5, 3.0), self.rng_visual.uniform(2, 6), self.COLOR_PARTICLE_DUST, gravity=0.1)
            self.particles.append(p)

    def _get_scenario_rng(self):
        h = hash(frozenset(self.game.walls) |
                 frozenset(tuple(self.game.goal_pos)))
        try:
            return np.random.default_rng(abs(h) % (2**32))
        except AttributeError:
            return np.random.RandomState(abs(h) % (2**32))

    def _draw_floor_texture(self, rng):
        density = (self.CELL_SIZE/40.0)**2
        num = int(self.game.grid_size**2*3*density)
        for _ in range(num):
            cx, cy = rng.uniform(0, self.WIDTH), rng.uniform(0, self.HEIGHT)
            r = rng.uniform(1, 3)
            try:
                shade = rng.integers(-30, 30)
            except AttributeError:
                shade = rng.randint(-30, 30)
            c = tuple(max(0, min(255, v+shade)) for v in self.COLOR_FLOOR)
            self.arcade.draw_ellipse_filled(
                cx, cy, r, r*rng.uniform(0.7, 1.0), c)

    def _draw_pheromones(self, q_table):
        if not self.debug_mode or q_table is None:
            return
        try:
            q_mov = q_table[:, :4]
            max_q, min_q = float(np.max(q_mov)), float(np.min(q_mov))
            q_range = max_q - min_q
            if q_range < 1e-6:
                return
        except Exception:
            return
        SQUARE_SIZE = self.CELL_SIZE * 0.85
        for idx in range(self.game.grid_size**2):
            x, y = idx % self.game.grid_size, idx // self.game.grid_size
            if (x, y) in self.game.walls:
                continue
            cx, cy = self._cell_to_pixel(x, y)
            try:
                q_val = float(np.max(q_table[idx, :4]))
            except Exception:
                continue
            nq = (q_val - min_q) / q_range
            r, g, b = 255, int(220 * (1 - nq) + 105 *
                               nq), int(230 * (1 - nq) + 180 * nq)
            alpha = int(40 + (nq**0.5) * 160)
            left = cx - SQUARE_SIZE / 2
            right = cx + SQUARE_SIZE / 2
            bottom = cy - SQUARE_SIZE / 2
            top = cy + SQUARE_SIZE / 2
            self.arcade.draw_lrbt_rectangle_filled(
                left, right, bottom, top, (r, g, b, alpha))

    def _draw_ant_q_values(self, q_table):
        if not self.debug_mode or q_table is None or not self.game:
            if self._q_value_text_objects:
                self._q_value_text_objects = []
            return
        x, y = self.game.ant_pos
        try:
            state_idx = int(y)*self.game.grid_size+int(x)
        except:
            return
        cx, cy = self._cell_to_pixel(*self.ant_display_pos)
        try:
            q_values = q_table[state_idx, :4]
        except:
            return
        font_size = max(6, int(self.CELL_SIZE*0.22))
        if not self._q_value_text_objects:
            font_c, shadow_c = (255, 255, 255, 240), (0, 0, 0, 200)
            for i in range(4):
                s = self.arcade.Text(
                    "", 0, 0, shadow_c, font_size, anchor_x='center', anchor_y='center')
                m = self.arcade.Text(
                    "", 0, 0, font_c, font_size, anchor_x='center', anchor_y='center')
                self._q_value_text_objects.append((s, m))
        offsets = {0: (0, 0.3), 1: (0, -0.4), 2: (-0.3, 0), 3: (0.3, 0)}
        for action, q_val in enumerate(q_values):
            s, m = self._q_value_text_objects[action]
            text = f"{q_val:.1f}"
            if m.text != text:
                m.text, s.text = text, text
            if m.font_size != font_size:
                m.font_size, s.font_size = font_size, font_size
            ox, oy = offsets[action]
            m.x, m.y = cx+ox*self.CELL_SIZE, cy+oy*self.CELL_SIZE
            s.x, s.y = m.x+1, m.y-1
            s.draw()
            m.draw()

    def _draw_anthill(self, rng):
        cx, cy = self._cell_to_pixel(*self.game.goal_pos)
        base, hole_c = (168, 139, 108), self.COLOR_GOAL
        rx, ry, max_h = self.CELL_SIZE*1.1, self.CELL_SIZE*0.8, self.CELL_SIZE*0.3
        shadow_offset = self.CELL_SIZE*0.1
        self.arcade.draw_ellipse_filled(
            cx+shadow_offset, cy-shadow_offset, rx, ry, (50, 50, 50, 80))
        for i in range(5):
            p, s = i/4, 1.0-(i/4*0.3)
            c = tuple(min(255, v+p*50) for v in base)
            self.arcade.draw_ellipse_filled(cx, cy+p*max_h, rx*s, ry*s, c)
        for _ in range(60):
            a, d = rng.uniform(0, 2*math.pi), rng.uniform(0, 1)**2
            px, py = cx+math.cos(a)*d*rx*0.8, cy + \
                math.sin(a)*d*ry*0.8+max_h*(1.0-d)*0.9
            try:
                shade = rng.integers(-20, 20)
            except:
                shade = np.random.randint(-20, 20)
            grain_c = tuple(max(0, min(255, c+shade+40)) for c in base)
            self.arcade.draw_circle_filled(
                px, py, rng.uniform(1.5, 3.0), grain_c)
        hole_cy = cy+max_h*0.95
        self.arcade.draw_ellipse_filled(
            cx, hole_cy, self.CELL_SIZE*0.3, self.CELL_SIZE*0.18, hole_c)
        self.anthill_hole_visual_center = (cx, hole_cy+self.CELL_SIZE*0.26)

    def _draw_ant(self):
        if self.ant_display_pos is None:
            return

        # CAMBIO: Comprobamos visibilidad basada en escala (éxito) Y transparencia (muerte).
        if self.ant_scale <= 0.01 or self.ant_alpha <= 0:
            return

        ax, ay = self.ant_display_pos
        base_cx, base_cy = self._cell_to_pixel(ax, ay)
        S, angle, t = self.ant_scale, self.ant_current_angle, time.time()

        # CAMBIO: Variables para gestionar el dibujo durante la muerte (Importado de AntLost).
        draw_cx, draw_cy = base_cx, base_cy
        vertical_flip_multiplier = 1

        # Determinar si la muerte es horizontal (Necesario para replicar el volteo y pose exactos de AntLost)
        is_horizontal_death_flag = self.in_death_transition and self.game.last_action in [
            2, 3]

        if self.in_death_transition:
            progress = self.death_transition_time / self.DEATH_TRANSITION_DURATION

            # 1. Temblor (Fase inicial 0-25%)
            if progress < 0.25:
                tremor = 4.0 * (1.0 - (progress / 0.25))
                draw_cx += self.rng_visual.uniform(-tremor, tremor)
                draw_cy += self.rng_visual.uniform(-tremor, tremor)

            # 2. Volteo (Aplicar SÓLO si la muerte NO es horizontal, como en AntLost)
            if self.ant_vertical_flip and not is_horizontal_death_flag:
                vertical_flip_multiplier = -1

        # --- INICIO: USAR EL COLOR Y TAMAÑO ALEATORIZADOS + TRANSPARENCIA ---
        # CAMBIO: Aplicamos self.ant_alpha a los colores.
        base_color = self.randomized_ant_color

        # Calculamos el alpha final (combina escala para éxito y alpha para muerte)
        final_alpha = int(self.ant_alpha * S)

        body_c = (*base_color, final_alpha)
        leg_c = (*(max(0, c - 50) for c in base_color), final_alpha)

        # La sombra también se desvanece proporcionalmente a la transparencia general.
        shadow_alpha = int(180 * (final_alpha / 255.0))
        shadow_c = (*(int(c * 0.3) for c in base_color), shadow_alpha)
        eye_c = (30, 30, 30, final_alpha)

        # Aplicar multiplicadores de tamaño (Se mantiene igual)
        m_head = self.ant_size_multipliers['head']
        m_thorax = self.ant_size_multipliers['thorax']
        m_abdomen = self.ant_size_multipliers['abdomen']

        hr = self.CELL_SIZE * 0.16 * S * m_head
        trx = self.CELL_SIZE * 0.21 * S * m_thorax
        trya = self.CELL_SIZE * 0.18 * S * m_thorax
        arx = self.CELL_SIZE * 0.28 * S * m_abdomen
        ary = self.CELL_SIZE * 0.22 * S * m_abdomen
        # --- FIN: USAR EL COLOR Y TAMAÑO ALEATORIZADOS + TRANSPARENCIA ---

        rad = math.radians(angle)

        def rotate(x, y): return x * math.cos(rad) - y * \
            math.sin(rad), x * math.sin(rad) + y * math.cos(rad)

        dist = math.sqrt(
            (self.game.ant_pos[0] - ax)**2 + (self.game.ant_pos[1] - ay)**2)

        # CAMBIO: Consideramos movimiento si estamos en CUALQUIER transición o si la distancia es > 0.01
        moving = self.in_success_transition or self.in_death_transition or dist > 0.01

        anim_s = self.CELL_SIZE / 40.0

        # CAMBIO: Definimos los parámetros de animación (Velocidad, Oscilación, Rebote) según el estado.
        if self.in_death_transition:
            # Muerte: Oscilación lenta, sin rebote (Valores de AntLost)
            speed, leg_o, ant_o, bounce = 3.0, 3, 5, 0
        elif moving and not self.in_success_transition:
            # Movimiento normal: Rápido y con rebote
            speed, leg_o, ant_o = 25.0, 25, 10
            bounce = abs(math.sin(t * 25.0)) * 3 * S * anim_s
        else:
            # Quieto o en éxito: Lento y sin rebote
            speed, leg_o, ant_o, bounce = 3.0, 3, 5, 0

        # Aplicamos el rebote solo si NO estamos muriendo (el temblor ya maneja el movimiento)
        if not self.in_death_transition:
            draw_cy += bounce

        osc = math.sin(t * speed)

        # --- DIBUJO ---

        # Patas
        ll, lt = self.CELL_SIZE * 0.28 * S, max(1, int(3 * S * anim_s))
        for side in [-1, 1]:
            for i, off_a in enumerate([-40, 0, 40]):
                co = osc if (side == 1 and i != 1) or \
                    (side == -1 and i == 1) else -osc
                end_a = angle + (90 + off_a + co * leg_o) * side
                ex, ey = math.cos(math.radians(end_a)) * \
                    ll, math.sin(math.radians(end_a)) * ll
                # CAMBIO: Aplicamos vertical_flip_multiplier al eje Y. Usamos draw_cx/cy.
                self.arcade.draw_line(
                    draw_cx, draw_cy, draw_cx + ex, draw_cy + ey * vertical_flip_multiplier, leg_c, lt)

        # Cuerpo (Usamos draw_cx/cy y colores con alpha)
        sx, sy = 3 * S * anim_s, -3 * S * anim_s
        ax_r, ay_r = rotate(-(trx + arx * 0.5), 0)
        self.arcade.draw_ellipse_filled(
            draw_cx + ax_r + sx, draw_cy + ay_r + sy, arx, ary, shadow_c, angle)
        self.arcade.draw_ellipse_filled(
            draw_cx + ax_r, draw_cy + ay_r, arx, ary, body_c, angle)
        self.arcade.draw_ellipse_filled(
            draw_cx + sx, draw_cy + sy, trx, trya, shadow_c, angle)
        self.arcade.draw_ellipse_filled(
            draw_cx, draw_cy, trx, trya, body_c, angle)

        # Cabeza (Usamos draw_cx/cy)
        hx_r, hy_r = rotate(hr * 0.85 + trx, 0)
        self.arcade.draw_circle_filled(
            draw_cx + hx_r + sx, draw_cy + hy_r + sy, hr, shadow_c)
        self.arcade.draw_circle_filled(
            draw_cx + hx_r, draw_cy + hy_r, hr, body_c)

        # Ojos
        er, eox, eoy = hr * 0.3, hr * 0.4, hr * 0.65
        for side in [-1, 1]:
            ex_r, ey_r = rotate(eox, eoy * side)
            # CAMBIO: Usamos el color de ojos con alpha (eye_c).
            self.arcade.draw_circle_filled(
                draw_cx + hx_r + ex_r, draw_cy + hy_r + ey_r, er, eye_c)

        # Antenas
        al, at = hr * 1.8, max(1, int(2 * S * anim_s))
        ant_o_val = osc * ant_o

        # CAMBIO: Lógica de antenas adaptada de AntLost para gestionar el volteo y el plegado horizontal.
        # Debemos manejar las antenas individualmente para replicar la pose exacta.

        # --- ANTENA IZQUIERDA (side = 1) ---
        side_L = 1
        # Lógica de AntLost: Plegar antenas si la muerte es horizontal
        if is_horizontal_death_flag:
            local_angle_L = 135.0
        else:
            local_angle_L = (45 * side_L) + ant_o_val

        end_a_L = angle + local_angle_L
        asx_r_L, asy_r_L = rotate(hr * 0.9, hr * 0.4 * side_L)
        asx_L, asy_L = draw_cx + hx_r + asx_r_L, draw_cy + hy_r + asy_r_L
        aex_L = math.cos(math.radians(end_a_L)) * al
        aey_L = math.sin(math.radians(end_a_L)) * al
        # Aplicamos vertical_flip_multiplier
        self.arcade.draw_line(asx_L, asy_L, asx_L + aex_L,
                              asy_L + aey_L * vertical_flip_multiplier, leg_c, at)

        # --- ANTENA DERECHA (side = -1) ---
        side_R = -1
        if is_horizontal_death_flag:
            local_angle_R = 225.0  # Ángulo para plegar la antena derecha
        else:
            local_angle_R = (45 * side_R) + ant_o_val

        end_a_R = angle + local_angle_R
        asx_r_R, asy_r_R = rotate(hr * 0.9, hr * 0.4 * side_R)
        asx_R, asy_R = draw_cx + hx_r + asx_r_R, draw_cy + hy_r + asy_r_R
        aex_R = math.cos(math.radians(end_a_R)) * al
        aey_R = math.sin(math.radians(end_a_R)) * al
        # Aplicamos vertical_flip_multiplier
        self.arcade.draw_line(asx_R, asy_R, asx_R + aex_R,
                              asy_R + aey_R * vertical_flip_multiplier, leg_c, at)

    def _draw_particles(self):
        if not self.arcade:
            return
        for p in self.particles:
            if p.age >= p.lifespan:
                continue
            progress = min(1.0, p.age/p.lifespan)
            fade = math.exp(-progress*4)
            alpha = int(p.color[3]*fade)
            if alpha <= 1:
                continue
            color = p.color[:3]+(alpha,)
            size = p.size*fade
            if size > 0.1:
                self.arcade.draw_circle_filled(p.x, p.y, size, color)

    def draw(self, game: MazeGame, q_table_to_render, render_mode: str | None, simulation_speed: float = 1.0):
        if render_mode is None:
            return None
        if not self.initialized:
            self._initialize(game, render_mode)
        if not self.window:
            return None

        current_time = time.time()
        delta_time = min(current_time-self.last_time, 0.1)*simulation_speed
        self.last_time = current_time
        try:
            self.window.switch_to()
            self.window.clear()
        except Exception:
            return None
        scenario_rng = self._get_scenario_rng()
        self._draw_floor_texture(scenario_rng)
        self._draw_pheromones(q_table_to_render)
        if self.wall_sprite_list:
            self.wall_sprite_list.draw()
        self._draw_anthill(scenario_rng)
        self._update_animations(delta_time)
        self._update_particles(delta_time)
        self._draw_ant()
        self._draw_particles()
        self._draw_ant_q_values(q_table_to_render)
        if render_mode == "rgb_array":
            try:
                return np.asarray(self.arcade.get_image().convert("RGB"))
            except Exception as e:
                print(f"Error capturando rgb_array: {e}")
                return np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        return self.WIDTH, self.HEIGHT

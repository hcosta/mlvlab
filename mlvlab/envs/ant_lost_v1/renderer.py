# mlvlab/envs/ant_lost_v1/renderer.py
import time
import math
import numpy as np

# Importación relativa para mantener la estructura del paquete
from .game import AntGame


class ParticleFX:
    """Clase para manejar partículas visuales de polvo con física básica."""

    def __init__(self, x, y, dx, dy, lifespan, size, color, gravity=0.1):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.lifespan = lifespan
        self.age = 0.0
        self.size = size
        self.color = (*color, 255) if len(color) == 3 else color
        self.gravity = gravity

    def update(self, delta_time):
        self.age += delta_time
        if self.age >= self.lifespan:
            return
        # Movimiento basado en framerate de 60fps
        dt_factor = delta_time * 60
        self.dy -= self.gravity * dt_factor
        self.x += self.dx * dt_factor
        self.y += self.dy * dt_factor


class ArcadeRenderer:
    """
    Renderer simplificado para LostAntEnv.
    Solo contiene la lógica de animación de muerte y colisiones.
    """

    def __init__(self) -> None:
        self.window = None
        self.game: AntGame | None = None
        self.CELL_SIZE = 50
        self.WIDTH = 0
        self.HEIGHT = 0

        # Paleta de colores
        self.COLOR_GRASS = (107, 142, 35)
        self.COLOR_ANT = (137, 48, 43)
        self.COLOR_OBSTACLE = (100, 100, 100)
        self.COLOR_PARTICLE_DUST = (210, 180, 140)

        # Estado de animación
        self.ant_display_pos = None
        self.ant_current_angle = 0.0
        self.last_time = time.time()
        self.particles: list[ParticleFX] = []
        self.was_colliding_last_frame = False

        # Lógica de animación de muerte
        self.in_death_transition = False
        self.death_pending_completion = False
        self.death_transition_time = 0.0
        self.DEATH_TRANSITION_DURATION = 2
        self.DEATH_PAUSE_DURATION = 0.01
        self.ant_vertical_flip = False
        self.ant_alpha = 255

        self.initialized = False
        try:
            self.rng_visual = np.random.default_rng()
        except AttributeError:
            self.rng_visual = np.random.RandomState()

        self.arcade = None

    def _lazy_import_arcade(self):
        if self.arcade is None:
            try:
                import arcade
                self.arcade = arcade
            except ImportError:
                raise ImportError(
                    "Se requiere 'arcade' para el renderizado. Instálalo con 'pip install arcade'.")

    def _initialize(self, game: AntGame, render_mode: str):
        self._lazy_import_arcade()
        self.game = game
        self.WIDTH = game.grid_size * self.CELL_SIZE
        self.HEIGHT = game.grid_size * self.CELL_SIZE
        if self.window is None:
            visible = render_mode == "human"
            self.window = self.arcade.Window(
                self.WIDTH, self.HEIGHT, "Ant Lost", visible=visible)
            if not visible:
                self.window.set_visible(False)
            self.arcade.set_background_color(self.COLOR_GRASS)

        self.ant_display_pos = list(game.ant_pos.astype(float))
        self.ant_current_angle = 0
        self.initialized = True

    def reset(self):
        self.initialized = False
        self.ant_display_pos = None
        self.last_time = time.time()
        self.particles = []
        self.was_colliding_last_frame = False
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

    def start_death_transition(self):
        if not self.in_death_transition and not self.death_pending_completion:
            self.death_pending_completion = True

    def is_in_death_transition(self) -> bool:
        return self.in_death_transition or self.death_pending_completion

    def _update_death_transition(self, delta_time: float):
        if not self.in_death_transition:
            return
        self.death_transition_time += delta_time
        progress = min(1.0, self.death_transition_time /
                       self.DEATH_TRANSITION_DURATION)
        if progress >= 0.25:
            self.ant_vertical_flip = True
        if progress >= 0.35:
            fade_progress = (progress - 0.35) / 0.65
            min_alpha = 60
            self.ant_alpha = int(255 - (255 - min_alpha) * fade_progress)
        total_duration = self.DEATH_TRANSITION_DURATION + self.DEATH_PAUSE_DURATION
        if self.death_transition_time >= total_duration:
            self.in_death_transition = False
            self.ant_alpha = 0

    def _get_angle_from_action(self, action):
        return {0: 90, 1: 270, 2: 180, 3: 0}.get(action, self.ant_current_angle)

    def _update_animations(self, delta_time: float):
        if self.in_death_transition:
            self._update_death_transition(delta_time)
            return

        target_pos = list(self.game.ant_pos.astype(float))
        dist_x = target_pos[0] - self.ant_display_pos[0]
        dist_y = target_pos[1] - self.ant_display_pos[1]
        is_at_target = math.sqrt(dist_x**2 + dist_y**2) < 0.01

        if not is_at_target:
            lerp_factor = 1.0 - math.exp(-delta_time * 15.0)
            self.ant_display_pos[0] += dist_x * lerp_factor
            self.ant_display_pos[1] += dist_y * lerp_factor
        else:
            self.ant_display_pos = target_pos

        target_angle = self._get_angle_from_action(self.game.last_action)
        diff = (target_angle - self.ant_current_angle + 180) % 360 - 180
        if abs(diff) > 0.1:
            lerp_factor = 1.0 - math.exp(-delta_time * 25.0)
            self.ant_current_angle += diff * lerp_factor
            self.ant_current_angle %= 360

        if self.death_pending_completion and is_at_target:
            self.death_pending_completion = False
            self.in_death_transition = True
            self.death_transition_time = 0.0
            self.ant_vertical_flip = False
            self.ant_alpha = 255
            self.ant_current_angle = self._get_angle_from_action(
                self.game.last_action)

        if self.game.collided and not self.was_colliding_last_frame:
            self._spawn_collision_particles()
        self.was_colliding_last_frame = self.game.collided

    def _spawn_collision_particles(self):
        if self.ant_display_pos is None:
            return
        cx, cy = self._cell_to_pixel(*self.ant_display_pos)

        impact_vectors = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
        impact_vector = impact_vectors.get(self.game.last_action, (0, 0))

        for _ in range(15):
            speed = self.rng_visual.uniform(0.5, 2.5)
            angle_offset = self.rng_visual.uniform(-0.8, 0.8)
            dx = (impact_vector[0] + angle_offset) * speed
            dy = (impact_vector[1] + abs(angle_offset)) * speed
            lifespan = self.rng_visual.uniform(1.5, 3.0)
            size = self.rng_visual.uniform(2, 6)
            self.particles.append(ParticleFX(
                cx, cy, dx, dy, lifespan, size, self.COLOR_PARTICLE_DUST))

    def _update_and_draw_particles(self, delta_time):
        if not self.particles:
            return

        to_keep = []
        for p in self.particles:
            p.update(delta_time)
            if p.age < p.lifespan:
                progress = p.age / p.lifespan
                fade_alpha = math.exp(-progress * 4)
                alpha = int(p.color[3] * fade_alpha)
                if alpha > 1:
                    color = (*p.color[:3], alpha)
                    size = p.size * fade_alpha
                    self.arcade.draw_circle_filled(p.x, p.y, size, color)
                    to_keep.append(p)
        self.particles = to_keep

    def _draw_static_elements(self):
        current_hash = hash(frozenset(self.game.obstacles))
        try:
            rng = np.random.default_rng(abs(current_hash) % (2**32))
        except AttributeError:
            rng = np.random.RandomState(abs(current_hash) % (2**32))

        for _ in range(self.game.grid_size * self.game.grid_size * 2):
            cx = rng.uniform(0, self.WIDTH)
            cy = rng.uniform(0, self.HEIGHT)
            r = rng.uniform(1, 4)
            try:
                shade = rng.integers(-25, 25)
            except AttributeError:
                shade = rng.randint(-25, 25)

            mote_color = (max(0, min(255, self.COLOR_GRASS[0]+shade)),
                          max(0, min(255, self.COLOR_GRASS[1]+shade)),
                          max(0, min(255, self.COLOR_GRASS[2]+shade)))
            self.arcade.draw_ellipse_filled(
                cx, cy, r, r * rng.uniform(0.5, 1.0), mote_color)

        for obs_x, obs_y in self.game.obstacles:
            cx, cy = self._cell_to_pixel(obs_x, obs_y)
            self._draw_rock(cx, cy, rng)

    def _draw_rock(self, cx, cy, rng):
        points = []
        try:
            num_points = rng.integers(7, 12)
        except AttributeError:
            num_points = rng.randint(7, 12)

        base_radius = self.CELL_SIZE * 0.55
        irregularity = self.CELL_SIZE * 0.18
        cy_visual = cy + self.CELL_SIZE * 0.1

        for i in range(num_points):
            angle = (math.pi * 2 * i) / num_points
            radius = base_radius + rng.uniform(-irregularity, irregularity)
            px = cx + math.cos(angle) * radius
            py = cy_visual + math.sin(angle) * radius
            points.append((px, py))

        shadow_offset_x, shadow_offset_y = 6, -6
        shadow_points = [(p[0] + shadow_offset_x, p[1] - (cy_visual - cy) +
                          shadow_offset_y) for p in points]
        self.arcade.draw_polygon_filled(shadow_points, (50, 50, 50, 120))

        try:
            shade = rng.integers(-20, 20)
        except AttributeError:
            shade = rng.randint(-20, 20)

        rock_color = (max(0, min(255, self.COLOR_OBSTACLE[0]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[1]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[2]+shade)))
        self.arcade.draw_polygon_filled(points, rock_color)

        highlight_color = (min(
            255, rock_color[0]+50), min(255, rock_color[1]+50), min(255, rock_color[2]+50))
        start_highlight = int(num_points * (110/360.0))
        end_highlight = int(num_points * (160/360.0))
        highlight_points = points[start_highlight: end_highlight+1]

        if len(highlight_points) > 1:
            self.arcade.draw_line_strip(highlight_points, highlight_color, 4)

    def _draw_ant(self):
        if self.ant_alpha <= 0:
            return

        base_cx, base_cy = self._cell_to_pixel(*self.ant_display_pos)
        draw_cx, draw_cy = base_cx, base_cy
        vertical_flip_multiplier = 1

        is_moving = math.sqrt((self.game.ant_pos[0] - self.ant_display_pos[0])**2 +
                              (self.game.ant_pos[1] - self.ant_display_pos[1])**2) > 0.01

        # --- CORRECCIÓN 1: Determinar el tipo de muerte aquí ---
        is_horizontal_death_flag = self.in_death_transition and self.game.last_action in [
            2, 3]

        if self.in_death_transition:
            progress = self.death_transition_time / self.DEATH_TRANSITION_DURATION
            if progress < 0.25:
                tremor = 4.0 * (1.0 - (progress / 0.25))
                draw_cx += self.rng_visual.uniform(-tremor, tremor)
                draw_cy += self.rng_visual.uniform(-tremor, tremor)

            # --- CORRECCIÓN 2: Aplicar el volteo SÓLO si la muerte NO es horizontal ---
            if self.ant_vertical_flip and not is_horizontal_death_flag:
                vertical_flip_multiplier = -1
        else:
            if is_moving:
                draw_cy += abs(math.sin(time.time() * 25.0)) * 3

        size_multiplier = 1.5
        angle = self.ant_current_angle
        body_color = (*self.COLOR_ANT, self.ant_alpha)
        shadow_color = (*(int(c * 0.3) for c in self.COLOR_ANT),
                        int(180 * (self.ant_alpha / 255)))
        leg_color = (*(max(0, c - 50) for c in self.COLOR_ANT), self.ant_alpha)

        head_radius = self.CELL_SIZE * 0.16 * size_multiplier
        thorax_radius_x = self.CELL_SIZE * 0.21 * size_multiplier
        thorax_radius_y = self.CELL_SIZE * 0.18 * size_multiplier
        abdomen_radius_x = self.CELL_SIZE * 0.28 * size_multiplier
        abdomen_radius_y = self.CELL_SIZE * 0.22 * size_multiplier

        angle_rad = math.radians(angle)

        def rotate(x, y):
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            return rx, ry

        # Se mantiene la oscilación original, ya que el problema no era este.
        oscillation = 0.0
        if is_moving or self.in_death_transition:
            oscillation = math.sin(
                time.time() * (3.0 if self.in_death_transition else 25.0))

        leg_osc_amount = 3 if self.in_death_transition else 25
        ant_osc_amount = 5 if self.in_death_transition else 10

        # Patas
        leg_length = self.CELL_SIZE * 0.28 * size_multiplier
        leg_thickness = 3 * size_multiplier
        for side in [-1, 1]:
            for i, offset_angle in enumerate([-40, 0, 40]):
                is_set_1 = (side == 1 and i != 1) or (side == -1 and i == 1)
                osc = (oscillation if is_set_1 else -
                       oscillation) * leg_osc_amount
                end_angle_deg = angle + (90 + offset_angle + osc) * side
                ex_rel = math.cos(math.radians(end_angle_deg)) * leg_length
                ey_rel = math.sin(math.radians(end_angle_deg)) * leg_length
                self.arcade.draw_line(draw_cx, draw_cy, draw_cx + ex_rel, draw_cy +
                                      ey_rel * vertical_flip_multiplier, leg_color, leg_thickness)

        # Cuerpo
        shadow_offset_x, shadow_offset_y = 3, -3
        abd_offset_x = -(thorax_radius_x + abdomen_radius_x * 0.5)
        ax_rel, ay_rel = rotate(abd_offset_x, 0)
        self.arcade.draw_ellipse_filled(draw_cx + ax_rel + shadow_offset_x, draw_cy +
                                        ay_rel + shadow_offset_y, abdomen_radius_x, abdomen_radius_y, shadow_color, angle)
        self.arcade.draw_ellipse_filled(
            draw_cx + ax_rel, draw_cy + ay_rel, abdomen_radius_x, abdomen_radius_y, body_color, angle)
        self.arcade.draw_ellipse_filled(draw_cx + shadow_offset_x, draw_cy +
                                        shadow_offset_y, thorax_radius_x, thorax_radius_y, shadow_color, angle)
        self.arcade.draw_ellipse_filled(
            draw_cx, draw_cy, thorax_radius_x, thorax_radius_y, body_color, angle)
        head_offset_x = head_radius * 0.85 + thorax_radius_x
        hx_rel, hy_rel = rotate(head_offset_x, 0)
        self.arcade.draw_circle_filled(
            draw_cx + hx_rel + shadow_offset_x, draw_cy + hy_rel + shadow_offset_y, head_radius, shadow_color)
        self.arcade.draw_circle_filled(
            draw_cx + hx_rel, draw_cy + hy_rel, head_radius, body_color)

        # Ojos
        eye_radius = head_radius * 0.3
        eye_color = (*(30, 30, 30), self.ant_alpha)
        for side in [-1, 1]:
            eox, eoy = rotate(head_radius * 0.4, head_radius * 0.65 * side)
            self.arcade.draw_circle_filled(
                draw_cx + hx_rel + eox, draw_cy + hy_rel + eoy, eye_radius, eye_color)

        # Antenas
        antenna_length = head_radius * 1.8
        antenna_thickness = 2 * size_multiplier

        # --- ANTENA IZQUIERDA (side = 1) ---
        side_L = 1
        local_angle_L = 0.0
        if is_horizontal_death_flag:
            local_angle_L = 135.0
        else:
            local_angle_L = (45 * side_L) + (oscillation * ant_osc_amount)

        final_world_angle_L = angle + local_angle_L

        asx_rel_L, asy_rel_L = rotate(
            head_radius * 0.9, head_radius * 0.4 * side_L)
        asx_L, asy_L = draw_cx + hx_rel + asx_rel_L, draw_cy + hy_rel + asy_rel_L
        aex_rel_L = math.cos(math.radians(
            final_world_angle_L)) * antenna_length
        aey_rel_L = math.sin(math.radians(
            final_world_angle_L)) * antenna_length
        self.arcade.draw_line(asx_L, asy_L, asx_L + aex_rel_L, asy_L +
                              aey_rel_L * vertical_flip_multiplier, leg_color, antenna_thickness)

        # --- ANTENA DERECHA (side = -1) ---
        side_R = -1
        local_angle_R = 0.0
        if is_horizontal_death_flag:
            local_angle_R = 225.0
        else:
            local_angle_R = (45 * side_R) + (oscillation * ant_osc_amount)

        final_world_angle_R = angle + local_angle_R

        asx_rel_R, asy_rel_R = rotate(
            head_radius * 0.9, head_radius * 0.4 * side_R)
        asx_R, asy_R = draw_cx + hx_rel + asx_rel_R, draw_cy + hy_rel + asy_rel_R
        aex_rel_R = math.cos(math.radians(
            final_world_angle_R)) * antenna_length
        aey_rel_R = math.sin(math.radians(
            final_world_angle_R)) * antenna_length
        self.arcade.draw_line(asx_R, asy_R, asx_R + aex_rel_R, asy_R +
                              aey_rel_R * vertical_flip_multiplier, leg_color, antenna_thickness)

    def draw(self, game: AntGame, render_mode: str, simulation_speed: float = 1.0):
        if not self.initialized:
            self._initialize(game, render_mode)

        current_time = time.time()
        delta_time = min(current_time - self.last_time, 0.1) * simulation_speed
        self.last_time = current_time

        if self.window:
            self.window.switch_to()
            self.window.clear()

        self._draw_static_elements()
        self._update_animations(delta_time)
        self._draw_ant()
        self._update_and_draw_particles(delta_time)

        if render_mode == "rgb_array":
            image_data = self.arcade.get_image(0, 0, self.WIDTH, self.HEIGHT)
            rgb_image = image_data.convert("RGB")
            return np.asarray(rgb_image)

        return self.WIDTH, self.HEIGHT

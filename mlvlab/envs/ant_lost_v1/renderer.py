# mlvlab/envs/ant_lost_v1/renderer.py
import time
import math
import numpy as np
# Importación relativa para mantener la estructura del paquete
try:
    from .game import AntLostGame
except ImportError:
    # Fallback si la importación relativa falla
    from game import AntLostGame


# =============================================================================
# JUICY ARCADE RENDERER (Adaptado para AntLost-v1 con animación de muerte)
# =============================================================================

class ParticleFX:
    """Clase para manejar partículas visuales (Polvo) con física básica."""

    def __init__(self, x, y, dx, dy, lifespan, size, color, p_type="dust", gravity=0.2):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.lifespan = lifespan
        self.age = 0.0
        self.size = size
        if len(color) == 3:
            self.color = (color[0], color[1], color[2], 255)
        else:
            self.color = color
        self.p_type = p_type
        self.gravity = gravity

    def update(self, delta_time):
        self.age += delta_time
        if self.age >= self.lifespan:
            return

        # Física básica (independiente del framerate)
        self.dy -= self.gravity * delta_time * 60
        self.x += self.dx * delta_time * 60
        self.y += self.dy * delta_time * 60


class ArcadeRenderer:
    """
    Renderer 'Juicy' basado en Arcade.
    """

    def __init__(self) -> None:
        self.window = None
        self.game: AntLostGame | None = None

        # Configuración visual
        self.CELL_SIZE = 50
        self.WIDTH = 0
        self.HEIGHT = 0

        # Paleta de colores
        self.COLOR_GRASS = (107, 142, 35)
        self.COLOR_ANT = (192, 57, 43)
        self.COLOR_OBSTACLE = (100, 100, 100)
        self.COLOR_PARTICLE_DUST = (210, 180, 140)

        # Estado de animación y efectos
        self.ant_prev_pos = None
        self.ant_display_pos = None
        self.ant_current_angle = 0.0
        self.ant_scale = 1.0
        # Para animación de muerte (boca arriba)
        self.ant_vertical_flip = False
        self.ant_alpha = 255  # Para desvanecimiento

        self.last_time = time.time()
        self.particles: list[ParticleFX] = []
        self.was_colliding_last_frame = False
        self._q_value_text_objects: list = []

        # Estado de transición de muerte (Reemplaza la transición de éxito de AntScout)
        self.in_death_transition = False
        self.death_transition_time = 0.0
        self.DEATH_TRANSITION_DURATION = 2.0  # Duración de la animación de muerte

        # Assets y optimización
        self.initialized = False
        self.debug_mode = False
        try:
            self.rng_visual = np.random.default_rng()
        except AttributeError:
            self.rng_visual = np.random.RandomState()

        # Importamos arcade aquí para retrasar la carga
        self.arcade = None
        self.draw_lbwh_rectangle_filled = None
        self._headless_mode = False

    def _lazy_import_arcade(self):
        if self.arcade is None:
            try:
                import arcade
                from arcade.draw import draw_lbwh_rectangle_filled
                self.arcade = arcade
                self.draw_lbwh_rectangle_filled = draw_lbwh_rectangle_filled
            except ImportError:
                raise ImportError(
                    "Se requiere 'arcade' para el renderizado. Instálalo con 'pip install arcade'.")

    def _get_angle_from_action(self, action):
        if action == 0:
            return 90   # Arriba
        if action == 1:
            return 270  # Abajo
        if action == 2:
            return 180  # Izquierda
        if action == 3:
            return 0    # Derecha
        return 0

    def _initialize(self, game: AntLostGame, render_mode: str):
        self._lazy_import_arcade()
        self.game = game
        self.WIDTH = game.grid_size * self.CELL_SIZE
        self.HEIGHT = game.grid_size * self.CELL_SIZE

        # Crear ventana si hace falta
        if self.window is None:
            visible = render_mode == "human"
            title = "Ants Saga - Zángano Errante (AntLost-v1) - MLVisual®"

            # (Lógica de creación de ventana idéntica a AntScout)
            if self._headless_mode or render_mode == "rgb_array":
                try:
                    self.window = self.arcade.Window(
                        self.WIDTH, self.HEIGHT, title, visible=False)
                except Exception:
                    try:
                        self.window = self.arcade.Window(
                            self.WIDTH, self.HEIGHT, title)
                        self.window.set_visible(False)
                    except Exception:
                        self.window = self.arcade.Window(
                            self.WIDTH, self.HEIGHT, title)
            else:
                try:
                    self.window = self.arcade.Window(
                        self.WIDTH, self.HEIGHT, title, visible=visible)
                except TypeError:
                    self.window = self.arcade.Window(
                        self.WIDTH, self.HEIGHT, title)
                    if not visible:
                        try:
                            self.window.set_visible(False)
                        except Exception:
                            pass

            try:
                self.arcade.set_background_color(self.COLOR_GRASS)
            except Exception:
                pass

        # Inicializar posición y ángulo de la hormiga
        if self.ant_display_pos is None:
            self.ant_display_pos = list(game.ant_pos.astype(float))
            self.ant_prev_pos = list(game.ant_pos.astype(float))
            self.ant_scale = 1.0
            self.ant_alpha = 255
            self.ant_vertical_flip = False
            self.ant_current_angle = self._get_angle_from_action(
                game.last_action)

        self.initialized = True

    def reset(self):
        # Limpia el estado del renderer
        self.initialized = False
        self.ant_display_pos = None
        self.ant_prev_pos = None
        self.ant_scale = 1.0
        self.ant_alpha = 255
        self.ant_vertical_flip = False
        self.last_time = time.time()
        self.particles = []
        self.in_death_transition = False
        self.death_transition_time = 0.0

    def _cell_to_pixel(self, x_cell: float, y_cell: float):
        x_px = x_cell * self.CELL_SIZE + self.CELL_SIZE / 2
        y_px = (self.game.grid_size - 1 - y_cell) * \
            self.CELL_SIZE + self.CELL_SIZE / 2
        return x_px, y_px

    def _pixel_to_cell(self, x_px: float, y_px: float):
        x_cell = (x_px - self.CELL_SIZE / 2) / self.CELL_SIZE
        y_cell = self.game.grid_size - 1 - \
            (y_px - self.CELL_SIZE / 2) / self.CELL_SIZE
        return x_cell, y_cell

    # Gestión de la Transición de Muerte ---

    def start_death_transition(self):
        if not self.in_death_transition:
            self.in_death_transition = True
            self.death_transition_time = 0.0

    def is_in_death_transition(self) -> bool:
        return self.in_death_transition

    def _update_rotation(self, delta_time, target_angle):
        # Interpola suavemente el ángulo actual hacia el objetivo (Maneja el cruce 360->0)
        current_angle = self.ant_current_angle
        diff = target_angle - current_angle
        while diff < -180:
            diff += 360
        while diff > 180:
            diff -= 360

        if abs(diff) > 0.1:
            # k=25 es la velocidad de rotación
            lerp_factor = 1.0 - math.exp(-delta_time * 25.0)
            self.ant_current_angle += diff * lerp_factor
        else:
            self.ant_current_angle = target_angle

        # Normalizar el ángulo
        self.ant_current_angle = self.ant_current_angle % 360

    def _update_death_transition(self, delta_time: float):
        if not self.in_death_transition:
            return

        self.death_transition_time += delta_time
        progress = min(1.0, self.death_transition_time /
                       self.DEATH_TRANSITION_DURATION)

        # Fases de la animación de muerte (Duración total 2.0s):
        # Fase 1: Temblor (0.0 - 0.5). El temblor visual se aplica en _draw_ant.

        # Fase 2: Volteo (Inicia en 0.5).
        if progress >= 0.5:
            self.ant_vertical_flip = True

        # Fase 3: Desvanecimiento (Inicia en 0.7).
        if progress >= 0.7:
            fade_progress = (progress - 0.7) / 0.3
            # self.ant_alpha va de 255 a 0
            self.ant_alpha = int(255 * (1.0 - fade_progress))

        if progress >= 1.0:
            self.in_death_transition = False
            self.ant_alpha = 0
            return

    # Actualización de Animaciones y Efectos ---

    def _update_animations(self, delta_time: float):
        if self.in_death_transition:
            self._update_death_transition(delta_time)
            # Durante la transición de muerte, el movimiento lógico se detiene.
            return

        # Si la hormiga está muerta (y la transición ya terminó), no actualizamos nada.
        if self.game.is_dead:
            return

        # 1. Movimiento suave
        target_pos = list(self.game.ant_pos.astype(float))

        if target_pos != self.ant_prev_pos:
            self.ant_prev_pos = list(self.ant_display_pos)

        dist_x = target_pos[0] - self.ant_display_pos[0]
        dist_y = target_pos[1] - self.ant_display_pos[1]
        distance = math.sqrt(dist_x**2 + dist_y**2)

        if distance > 0.001:
            # k=15 es la "velocidad".
            lerp_factor = 1.0 - math.exp(-delta_time * 15.0)
            self.ant_display_pos[0] += dist_x * lerp_factor
            self.ant_display_pos[1] += dist_y * lerp_factor
        else:
            self.ant_display_pos = list(target_pos)
            self.ant_prev_pos = list(target_pos)

        # 2. Rotación suave
        target_angle = self._get_angle_from_action(self.game.last_action)
        self._update_rotation(delta_time, target_angle)

        # 3. Efectos (Partículas de Colisión)
        is_colliding_now = self.game.collided
        if is_colliding_now and not self.was_colliding_last_frame:
            self._spawn_collision_particles()

        # Actualizamos el estado para el siguiente frame.
        self.was_colliding_last_frame = is_colliding_now

    def _update_particles(self, delta_time: float):
        # Actualizar todas las partículas
        for particle in self.particles:
            particle.update(delta_time)

        # Eliminar partículas muertas (cuando age >= lifespan)
        self.particles = [p for p in self.particles if p.age < p.lifespan]

    def _spawn_collision_particles(self):
        # Guarda para evitar un error si se llama antes de la inicialización.
        if self.ant_display_pos is None:
            return
        # Genera partículas de polvo y tierra en el punto de colisión.
        ax, ay = self.ant_display_pos
        cx, cy = self._cell_to_pixel(ax, ay)

        # Determinar la dirección del impacto
        action = self.game.last_action
        impact_vector = [0, 0]
        spawn_x, spawn_y = cx, cy

        # (Lógica de impacto idéntica a AntScout)
        if action == 0:   # Arriba (Gym) -> Impacto desde arriba (Arcade Y es inverso)
            impact_vector = [0, -1]
            spawn_y += self.CELL_SIZE * 0.3
        elif action == 1:  # Abajo (Gym) -> Impacto desde abajo
            impact_vector = [0, 1]
            spawn_y -= self.CELL_SIZE * 0.3
        elif action == 2:  # Izquierda -> Impacto desde la izquierda
            impact_vector = [1, 0]
            spawn_x -= self.CELL_SIZE * 0.3
        elif action == 3:  # Derecha -> Impacto desde la derecha
            impact_vector = [-1, 0]
            spawn_x += self.CELL_SIZE * 0.3

        # Partículas de Polvo/Tierra
        for _ in range(15):
            speed = self.rng_visual.uniform(0.5, 2.5)
            angle_offset = self.rng_visual.uniform(-0.8, 0.8)
            dx = (impact_vector[0] + angle_offset) * speed
            dy = (impact_vector[1] + abs(angle_offset)) * speed
            lifespan = self.rng_visual.uniform(1.5, 3.0)
            size = self.rng_visual.uniform(2, 6)

            p = ParticleFX(spawn_x, spawn_y, dx, dy, lifespan, size,
                           self.COLOR_PARTICLE_DUST, gravity=0.1)
            self.particles.append(p)

    # Funciones de Dibujo ---

    def _draw_static_elements(self):
        # Dibuja los obstáculos y las motas de tierra en cada frame.
        # Usamos el hash del escenario para generar una semilla determinista.
        # En AntLost, no hay goal_pos, solo obstáculos.
        current_hash = hash(frozenset(self.game.obstacles))
        try:
            rng = np.random.default_rng(abs(current_hash) % (2**32))
        except AttributeError:
            rng = np.random.RandomState(abs(current_hash) % (2**32))

        # Motas de tierra/hojarasca (Textura del fondo)
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

        # Obstáculos (Rocas procedurales)
        for obs_x, obs_y in self.game.obstacles:
            cx, cy = self._cell_to_pixel(obs_x, obs_y)
            self._draw_rock(cx, cy, rng)

    def _draw_rock(self, cx, cy, rng):
        # (Implementación de roca idéntica a AntScout)
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

        # Sombra
        shadow_offset_x, shadow_offset_y = 6, -6
        shadow_points = [(p[0] + shadow_offset_x, p[1] - (cy_visual - cy) +
                          shadow_offset_y) for p in points]
        self.arcade.draw_polygon_filled(shadow_points, (50, 50, 50, 120))

        # Cuerpo principal
        try:
            shade = rng.integers(-20, 20)
        except AttributeError:
            shade = rng.randint(-20, 20)

        rock_color = (max(0, min(255, self.COLOR_OBSTACLE[0]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[1]+shade)),
                      max(0, min(255, self.COLOR_OBSTACLE[2]+shade)))
        self.arcade.draw_polygon_filled(points, rock_color)

        # Highlight
        highlight_color = (min(
            255, rock_color[0]+50), min(255, rock_color[1]+50), min(255, rock_color[2]+50))
        start_highlight = int(num_points * (110/360.0))
        end_highlight = int(num_points * (160/360.0))
        highlight_points = points[start_highlight: end_highlight+1]

        if len(highlight_points) > 1:
            self.arcade.draw_line_strip(highlight_points, highlight_color, 4)

    # --- Funciones de Heatmap y Q-Values (Mantenidas para depuración si es necesario) ---

    def _draw_ant_q_values(self, q_table):
        """Dibuja los 4 valores Q solo para la celda actual de la hormiga."""
        if not self.debug_mode or q_table is None or self.game is None:
            if self._q_value_text_objects:
                self._q_value_text_objects = []
            return

        # (Código idéntico a AntScout)
        ant_x_logic, ant_y_logic = self.game.ant_pos
        state_index = int(ant_y_logic) * self.game.grid_size + int(ant_x_logic)
        cx, cy = self._cell_to_pixel(*self.ant_display_pos)

        try:
            q_values = q_table[state_index, :]
        except (IndexError, TypeError):
            # Si no hay Q-table válida (e.g. agente aleatorio), no dibujamos nada
            return

        if not self._q_value_text_objects:
            font_size = 9
            font_color = (255, 255, 255, 220)
            shadow_color = (0, 0, 0, 180)
            for i in range(4):
                shadow_obj = self.arcade.Text(
                    "", x=0, y=0, color=shadow_color, font_size=font_size, anchor_x='center', anchor_y='center')
                main_obj = self.arcade.Text(
                    "", x=0, y=0, color=font_color, font_size=font_size, anchor_x='center', anchor_y='center')
                self._q_value_text_objects.append((shadow_obj, main_obj))

        offsets = {0: (0, self.CELL_SIZE*0.3), 1: (0, -self.CELL_SIZE*0.4),
                   2: (-self.CELL_SIZE*0.3, 0), 3: (self.CELL_SIZE*0.3, 0)}

        for action, q_value in enumerate(q_values):
            shadow_obj, main_obj = self._q_value_text_objects[action]
            new_text = f"{q_value:.1f}"
            if main_obj.text != new_text:
                main_obj.text = new_text
                shadow_obj.text = new_text

            offset_x, offset_y = offsets[action]
            pos_x, pos_y = cx + offset_x, cy + offset_y

            main_obj.x, main_obj.y = pos_x, pos_y
            shadow_obj.x, shadow_obj.y = pos_x + 1, pos_y - 1

            shadow_obj.draw()
            main_obj.draw()

    def _draw_heatmap(self, q_table_to_render):
        # Dibuja la visualización de la Q-Table solo si el modo debug está activo.
        if not self.debug_mode or q_table_to_render is None:
            return

        # (Código idéntico a AntScout)
        try:
            max_q = float(np.max(q_table_to_render))
            min_q = float(np.min(q_table_to_render))
        except (Exception, TypeError):
            return

        q_range = max_q - min_q
        if q_range < 1e-6:
            return

        SQUARE_SIZE = self.CELL_SIZE * 0.75

        for state_index in range(self.game.grid_size * self.game.grid_size):
            x_cell = state_index % self.game.grid_size
            y_cell = state_index // self.game.grid_size
            cx, cy = self._cell_to_pixel(x_cell, y_cell)

            try:
                q_value = float(np.max(q_table_to_render[state_index, :]))
            except Exception:
                continue

            norm_q = (q_value - min_q) / q_range

            # Gradiente "MAGMA-LIKE"
            if norm_q < 0.5:
                t = norm_q * 2
                r = int(10 * (1 - t) + 252 * t)
                g = int(8 * (1 - t) + 80 * t)
                b = int(40 * (1 - t) + 50 * t)
            else:
                t = (norm_q - 0.5) * 2
                r = int(252 * (1 - t) + 252 * t)
                g = int(80 * (1 - t) + 250 * t)
                b = int(50 * (1 - t) + 100 * t)

            base_alpha = 50
            value_alpha = norm_q * 180
            final_alpha = int(base_alpha + value_alpha)

            heat_color = (r, g, b, final_alpha)

            left = cx - SQUARE_SIZE / 2
            bottom = cy - SQUARE_SIZE / 2

            if self.draw_lbwh_rectangle_filled:
                self.draw_lbwh_rectangle_filled(
                    left, bottom, SQUARE_SIZE, SQUARE_SIZE, heat_color)

    # NOTA: _draw_anthill se elimina porque no hay hormiguero en AntLost-v1.

    def _draw_ant(self):
        # Dibuja la hormiga animada y procedural.

        # Si el alpha es muy bajo, no dibujamos nada.
        if self.ant_alpha <= 1:
            return

        ax, ay = self.ant_display_pos
        cx, cy = self._cell_to_pixel(ax, ay)

        # Aplicar temblor durante la fase inicial de la muerte
        if self.in_death_transition:
            progress = min(1.0, self.death_transition_time /
                           self.DEATH_TRANSITION_DURATION)
            # Temblor en la primera fase (0.0 a 0.5)
            if progress < 0.5:
                # Intensidad disminuye
                shake_intensity = 4.0 * (1.0 - progress*2)
                cx += math.sin(self.death_transition_time *
                               60) * shake_intensity
                cy += math.cos(self.death_transition_time *
                               60) * shake_intensity

        # Aplicar escala global
        SCALE = self.ant_scale
        ALPHA = self.ant_alpha

        # Usamos el ángulo interpolado suavemente
        angle = self.ant_current_angle

        # Definición del cuerpo (Ajustado por escala y Alpha)
        # Aplicamos el alpha a todos los colores
        body_color = (
            self.COLOR_ANT[0], self.COLOR_ANT[1], self.COLOR_ANT[2], ALPHA)

        # La sombra se desvanece
        shadow_alpha = max(0, int(180 * (ALPHA/255.0)))
        shadow_color = (
            int(self.COLOR_ANT[0]*0.3), int(self.COLOR_ANT[1]*0.3), int(self.COLOR_ANT[2]*0.3), shadow_alpha)

        leg_base_color = (max(
            0, self.COLOR_ANT[0]-50), max(0, self.COLOR_ANT[1]-50), max(0, self.COLOR_ANT[2]-50))
        leg_color = (leg_base_color[0],
                     leg_base_color[1], leg_base_color[2], ALPHA)
        eye_color = (30, 30, 30, ALPHA)

        head_radius = self.CELL_SIZE * 0.16 * SCALE
        thorax_radius_x = self.CELL_SIZE * 0.21 * SCALE
        thorax_radius_y = self.CELL_SIZE * 0.18 * SCALE
        abdomen_radius_x = self.CELL_SIZE * 0.28 * SCALE
        abdomen_radius_y = self.CELL_SIZE * 0.22 * SCALE

        # Función auxiliar para rotar puntos alrededor del centro
        angle_rad = math.radians(angle)

        # Control del flip vertical para la animación de muerte
        flip_multiplier = -1 if self.ant_vertical_flip else 1

        def rotate(x, y):
            # Aplicamos el flip antes de la rotación
            y *= flip_multiplier
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            return rx, ry

        # Animación y Movimiento ---
        t = time.time()

        # Detectar movimiento. Si está muriendo, detenemos la animación de caminar.
        if self.in_death_transition:
            moving = False
        else:
            distance_to_target = math.sqrt((self.game.ant_pos[0] - self.ant_display_pos[0])**2 +
                                           (self.game.ant_pos[1] - self.ant_display_pos[1])**2)
            moving = distance_to_target > 0.01

        if moving:
            animation_speed = 25.0
            leg_oscillation_amount = 25
            antenna_oscillation_amount = 10
            # El rebote también se escala
            bounce = abs(math.sin(t * animation_speed)) * 3 * SCALE
        else:
            # Animación idle o durante la muerte (movimiento lento)
            animation_speed = 3.0
            leg_oscillation_amount = 3
            antenna_oscillation_amount = 5
            bounce = 0

        cy += bounce
        oscillation = math.sin(t * animation_speed)

        # Dibujo de Patas (Ajustado por escala) ---
        leg_length = self.CELL_SIZE * 0.28 * SCALE
        leg_thickness = max(1, int(3 * SCALE))  # Grosor mínimo de 1px

        for side in [-1, 1]:
            # Si está volteada, las patas se dibujan hacia arriba
            visual_side = side * flip_multiplier

            for i, offset_angle in enumerate([-40, 0, 40]):
                # Lógica para alternar el movimiento de las patas
                is_set_1 = (side == 1 and i != 1) or (side == -1 and i == 1)
                current_oscillation = oscillation if is_set_1 else -oscillation
                osc = current_oscillation * leg_oscillation_amount

                # Calculamos el ángulo base de la pata
                base_leg_angle = 90 + offset_angle

                # Calculamos el ángulo final en grados, considerando el lado y la oscilación
                end_angle_deg = angle + (base_leg_angle + osc) * visual_side

                sx, sy = cx, cy
                ex_rel = math.cos(math.radians(end_angle_deg)) * leg_length
                ey_rel = math.sin(math.radians(end_angle_deg)) * leg_length
                self.arcade.draw_line(
                    sx, sy, sx + ex_rel, sy + ey_rel, leg_color, leg_thickness)

        # Dibujo del Cuerpo (Encima de las patas) ---
        shadow_offset_x, shadow_offset_y = 3 * SCALE, -3 * SCALE

        # Abdomen
        abd_offset_x = -(thorax_radius_x + abdomen_radius_x*0.5)
        ax_rel, ay_rel = rotate(abd_offset_x, 0)
        # Sombra
        self.arcade.draw_ellipse_filled(cx + ax_rel + shadow_offset_x, cy + ay_rel + shadow_offset_y,
                                        abdomen_radius_x, abdomen_radius_y, shadow_color, angle)
        # Cuerpo
        self.arcade.draw_ellipse_filled(
            cx + ax_rel, cy + ay_rel, abdomen_radius_x, abdomen_radius_y, body_color, angle)

        # Tórax
        # Sombra
        self.arcade.draw_ellipse_filled(cx + shadow_offset_x, cy + shadow_offset_y,
                                        thorax_radius_x, thorax_radius_y, shadow_color, angle)
        # Cuerpo
        self.arcade.draw_ellipse_filled(
            cx, cy, thorax_radius_x, thorax_radius_y, body_color, angle)

        # Cabeza
        head_offset_x = head_radius*0.85 + thorax_radius_x
        hx_rel, hy_rel = rotate(head_offset_x, 0)

        # Sombra
        # Usamos draw_ellipse_filled para asegurar la compatibilidad con el alpha
        self.arcade.draw_ellipse_filled(
            cx + hx_rel + shadow_offset_x, cy + hy_rel + shadow_offset_y, head_radius, head_radius, shadow_color)
        # Cuerpo
        self.arcade.draw_ellipse_filled(
            cx + hx_rel, cy + hy_rel, head_radius, head_radius, body_color)

        # Detalles de la Cabeza (Ajustado por escala) ---
        eye_radius = head_radius * 0.3
        eye_offset_x = head_radius * 0.4
        eye_offset_y = head_radius * 0.65

        for side in [-1, 1]:
            eox, eoy = rotate(eye_offset_x, eye_offset_y * side)
            self.arcade.draw_ellipse_filled(
                cx + hx_rel + eox, cy + hy_rel + eoy, eye_radius, eye_radius, eye_color)

        # Antenas
        antenna_length = head_radius * 1.8
        antenna_thickness = max(1, int(2 * SCALE))
        antenna_oscillation = oscillation * antenna_oscillation_amount
        for side in [-1, 1]:
            visual_side = side * flip_multiplier
            base_angle = 45

            end_angle = angle + (base_angle * visual_side) + \
                antenna_oscillation

            start_offset_x = head_radius * 0.9
            # Usamos side original aquí para el punto de inicio
            start_offset_y = head_radius * 0.4 * side

            asx_rel, asy_rel = rotate(start_offset_x, start_offset_y)
            asx = cx + hx_rel + asx_rel
            asy = cy + hy_rel + asy_rel
            aex_rel = math.cos(math.radians(end_angle)) * antenna_length
            aey_rel = math.sin(math.radians(end_angle)) * antenna_length
            self.arcade.draw_line(asx, asy, asx + aex_rel,
                                  asy + aey_rel, leg_color, antenna_thickness)

    def _draw_particles(self):
        if not self.arcade:
            return

        for p in self.particles:
            if p.age >= p.lifespan:
                continue

            progress = min(1.0, p.age / p.lifespan)

            # Easing para el fade out
            if p.p_type == "dust":
                # Decaimiento exponencial para polvo (rápido al inicio)
                fade_alpha = math.exp(-progress * 4)

            # El alpha inicial se toma del cuarto componente del color guardado
            initial_alpha = p.color[3]
            alpha = int(initial_alpha * fade_alpha)

            if alpha <= 1:
                continue

            color = (p.color[0], p.color[1], p.color[2], alpha)
            # El tamaño también se reduce
            size = p.size * fade_alpha

            if size > 0.1:
                # Usamos draw_circle_filled que es compatible.
                self.arcade.draw_circle_filled(p.x, p.y, size, color)

    def draw(self, game: AntLostGame, q_table_to_render, render_mode: str | None, simulation_speed: float = 1.0):
        """ Función principal de dibujo del frame. """

        # Si el juego está en un estado terminal (muerte o colisión),
        # forzamos a que la posición visual sea idéntica a la posición real.
        if game.collided or game.is_dead:
            # Si acaba de morir, iniciamos la transición aquí también como seguridad
            if game.is_dead and not self.in_death_transition and self.ant_alpha > 0:
                self.start_death_transition()

            # Solo forzamos la posición si no estamos en la animación de muerte (para permitir el temblor)
            if not self.in_death_transition:
                self.ant_display_pos = list(game.ant_pos.astype(float))

        if render_mode is None:
            return None

        # Inicialización perezosa
        if not self.initialized:
            self._initialize(game, render_mode)

        # Cálculo del tiempo delta
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        delta_time = min(delta_time, 0.1)

        # Escalamos el delta_time ---
        scaled_delta_time = delta_time * simulation_speed

        # Proceso de Dibujo y Actualización Coordinada ---

        if self.window:
            self.window.switch_to()
            self.window.clear()

        # 1. Heatmap (si aplica)
        self._draw_heatmap(q_table_to_render)

        # 2. Elementos estáticos (Rocas, textura del suelo)
        self._draw_static_elements()

        # Actualizamos las animaciones (movimiento, rotación, muerte, partículas) usando scaled_delta_time
        self._update_animations(scaled_delta_time)
        self._update_particles(scaled_delta_time)

        # 3. Dibujamos la hormiga (usando la posición, ángulo, escala, flip y alpha actualizados)
        self._draw_ant()

        # 4. Debug mode (Q-values)
        self._draw_ant_q_values(q_table_to_render)

        # 5. Post-procesado: Partículas
        self._draw_particles()

        return self.WIDTH, self.HEIGHT

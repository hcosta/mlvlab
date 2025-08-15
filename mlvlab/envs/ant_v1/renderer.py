# mlvlab/envs/ant/renderer.py
import time
import math
import numpy as np
# Importación relativa para mantener la estructura del paquete
try:
    from .game import AntGame
except ImportError:
    # Fallback si la importación relativa falla
    from game import AntGame


# =============================================================================
# JUICY ARCADE RENDERER (Modularizado, Mejorado y Restaurado)
# =============================================================================

class ParticleFX:
    """Clase para manejar partículas visuales (Polvo y Estrellas) con física básica."""

    def __init__(self, x, y, dx, dy, lifespan, size, color, p_type="dust", gravity=0.2):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.lifespan = lifespan
        self.age = 0.0  # Tiempo transcurrido desde la creación
        self.size = size
        # Aseguramos que el color tenga 4 componentes (RGBA) para el alpha inicial
        if len(color) == 3:
            self.color = (color[0], color[1], color[2], 255)
        else:
            self.color = color
        self.p_type = p_type  # Tipo de partícula ("dust" o "star")
        self.gravity = gravity

    def update(self, delta_time):
        self.age += delta_time
        if self.age >= self.lifespan:
            return

        # Física básica (independiente del framerate)
        # Asumiendo 60 FPS como base para la gravedad original.
        self.dy -= self.gravity * delta_time * 60

        self.x += self.dx * delta_time * 60
        self.y += self.dy * delta_time * 60

    # El dibujo se maneja externamente en _draw_particles.


class ArcadeRenderer:
    """
    Renderer 'Juicy' basado en Arcade.
    """

    def __init__(self) -> None:
        self.window = None
        self.game: AntGame | None = None

        # Configuración visual
        self.CELL_SIZE = 50
        self.WIDTH = 0
        self.HEIGHT = 0

        # Paleta de colores
        self.COLOR_GRASS = (107, 142, 35)
        self.COLOR_ANT = (192, 57, 43)
        self.COLOR_GOAL = (40, 25, 10)
        self.COLOR_OBSTACLE = (100, 100, 100)
        self.COLOR_PARTICLE_DUST = (210, 180, 140)
        self.COLOR_PARTICLE_STAR = (255, 223, 0)

        # Estado de animación y efectos
        self.ant_prev_pos = None
        self.ant_display_pos = None  # Coordenadas de Grid (float)
        self.ant_current_angle = 0.0  # Ángulo actual (para rotación suave)
        self.ant_scale = 1.0
        self.last_time = time.time()
        self.particles: list[ParticleFX] = []
        self.anthill_hole_visual_center = None  # (x_px, y_px) del agujero

        # Estado de transición de éxito
        self.in_success_transition = False
        self.success_transition_time = 0.0
        self.SUCCESS_TRANSITION_DURATION = 1.5  # Segundos

        # Assets y optimización
        self.initialized = False
        # Estado para controlar el modo de depuración ---
        self.debug_mode = False
        try:
            self.rng_visual = np.random.default_rng()
        except AttributeError:
            self.rng_visual = np.random.RandomState()

        # Importamos arcade aquí para retrasar la carga
        self.arcade = None
        self.draw_lbwh_rectangle_filled = None

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

    def _initialize(self, game: AntGame, render_mode: str):
        self._lazy_import_arcade()
        self.game = game
        self.WIDTH = game.grid_size * self.CELL_SIZE
        self.HEIGHT = game.grid_size * self.CELL_SIZE

        # Crear ventana si hace falta
        if self.window is None:
            visible = render_mode == "human"
            title = "Lost Ant Colony - JUICY"
            try:
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, title, visible=visible)
            except TypeError:
                # Fallback para versiones antiguas de Arcade
                self.window = self.arcade.Window(
                    self.WIDTH, self.HEIGHT, title)
                if not visible:
                    try:
                        self.window.set_visible(False)
                    except Exception:
                        pass
            self.arcade.set_background_color(self.COLOR_GRASS)

        # Inicializar posición y ángulo de la hormiga
        if self.ant_display_pos is None:
            self.ant_display_pos = list(game.ant_pos.astype(float))
            self.ant_prev_pos = list(game.ant_pos.astype(float))
            self.ant_scale = 1.0
            # Inicializar el ángulo basado en la acción inicial
            self.ant_current_angle = self._get_angle_from_action(
                game.last_action)

        self.initialized = True

    def reset(self):
        # Limpia el estado del renderer
        self.initialized = False
        self.ant_display_pos = None
        self.ant_prev_pos = None
        # No reseteamos el ángulo aquí, se hace en _initialize basado en la nueva posición/acción
        self.ant_scale = 1.0
        self.last_time = time.time()
        self.particles = []
        self.in_success_transition = False
        self.success_transition_time = 0.0
        self.anthill_hole_visual_center = None

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

    # Gestión de la Transición de Éxito ---

    def start_success_transition(self):
        if not self.in_success_transition:
            self.in_success_transition = True
            self.success_transition_time = 0.0

    def is_in_success_transition(self) -> bool:
        return self.in_success_transition

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

    def _update_success_transition(self, delta_time: float):
        if not self.in_success_transition:
            return

        self.success_transition_time += delta_time

        if self.success_transition_time >= self.SUCCESS_TRANSITION_DURATION:
            self.in_success_transition = False
            self.ant_scale = 0.0
            return

        # Calcular el progreso (0.0 a 1.0)
        progress = self.success_transition_time / self.SUCCESS_TRANSITION_DURATION

        # 1. Determinar el objetivo (Centro visual del agujero)
        if self.anthill_hole_visual_center:
            target_x_px, target_y_px = self.anthill_hole_visual_center
            target_x_cell, target_y_cell = self._pixel_to_cell(
                target_x_px, target_y_px)
            target_pos = [target_x_cell, target_y_cell]
        else:
            # Fallback al centro lógico
            target_pos = list(self.game.goal_pos.astype(float))

        # 2. Mover suavemente hacia el objetivo
        # k=10 es la velocidad de acercamiento
        lerp_factor_move = 1.0 - math.exp(-delta_time * 10.0)

        dist_x = target_pos[0] - self.ant_display_pos[0]
        dist_y = target_pos[1] - self.ant_display_pos[1]

        self.ant_display_pos[0] += dist_x * lerp_factor_move
        self.ant_display_pos[1] += dist_y * lerp_factor_move

        # 3. Rotar suavemente para mirar hacia el objetivo
        # Calculamos el ángulo en el espacio de píxeles para precisión visual
        current_x_px, current_y_px = self._cell_to_pixel(*self.ant_display_pos)

        if self.anthill_hole_visual_center:
            target_x_px, target_y_px = self.anthill_hole_visual_center
            dx = target_x_px - current_x_px
            dy = target_y_px - current_y_px
            target_angle = math.degrees(math.atan2(dy, dx))
            self._update_rotation(delta_time, target_angle)

        # 4. Escalar la hormiga hacia abajo (Easing "EaseInOutCubic")

        def easeInOutCubic(t):
            if t < 0.5:
                return 4 * t * t * t
            else:
                return 1 - pow(-2 * t + 2, 3) / 2

        eased_progress = easeInOutCubic(progress)
        self.ant_scale = 1.0 - eased_progress

        # La animación termina si se completa la duración O si la hormiga es invisible.
        if self.success_transition_time >= self.SUCCESS_TRANSITION_DURATION or self.ant_scale <= 0.01:
            self.in_success_transition = False
            self.ant_scale = 0.0  # Forzamos la escala a 0 para asegurar que no se dibuje

    # Actualización de Animaciones y Efectos ---

    def _update_animations(self, delta_time: float):
        if self.in_success_transition:
            self._update_success_transition(delta_time)
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

        # Eliminar partículas muertas
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _spawn_collision_particles(self):
        # Guarda para evitar un error si se llama antes de la inicialización.
        if self.ant_display_pos is None:
            return
        # Genera partículas de polvo y tierra en el punto de colisión.
        ax, ay = self.ant_display_pos
        cx, cy = self._cell_to_pixel(ax, ay)

        # Determinar la dirección del impacto para lanzar las partículas en la dirección opuesta
        action = self.game.last_action
        impact_vector = [0, 0]
        spawn_x, spawn_y = cx, cy

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

        # Partículas de Polvo/Tierra (Ajustadas) ---

        # Un poco más de partículas para un "poof" más denso
        for _ in range(15):
            # 1. Velocidad más lenta para que parezcan motas de polvo a la deriva
            speed = self.rng_visual.uniform(0.5, 2.5)
            # Un cono de dispersión un poco más amplio
            angle_offset = self.rng_visual.uniform(-0.8, 0.8)
            dx = (impact_vector[0] + angle_offset) * speed
            dy = (impact_vector[1] + abs(angle_offset)) * speed

            # 2. Aumentamos significativamente la vida de las partículas
            lifespan = self.rng_visual.uniform(1.5, 3.0)

            # Ligeramente más pequeñas para el efecto polvo
            size = self.rng_visual.uniform(2, 6)

            # 3. Reducimos la gravedad para que floten más tiempo en el aire
            p = ParticleFX(spawn_x, spawn_y, dx, dy, lifespan, size,
                           self.COLOR_PARTICLE_DUST, gravity=0.1)
            self.particles.append(p)

        # La sección de partículas de "Estrella" se ha eliminado para un efecto
        # puramente de tierra/polvo, como solicitaste.

    # Funciones de Dibujo ---
    def _draw_static_elements(self):
        # Dibuja los obstáculos y las motas de tierra en cada frame.
        # Usamos el hash del escenario para generar una semilla determinista para la apariencia visual.
        current_hash = hash(frozenset(self.game.obstacles)
                            | frozenset(tuple(self.game.goal_pos)))
        try:
            rng = np.random.default_rng(abs(current_hash) % (2**32))
        except AttributeError:
            # Fallback para versiones antiguas de numpy
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
        # Crea una roca procedural usando polígonos irregulares.
        # Se ha mejorado para que ocupe más espacio visualmente (Volumen).
        points = []
        try:
            num_points = rng.integers(7, 12)
        except AttributeError:
            num_points = rng.randint(7, 12)

        # Aumentamos el tamaño base
        base_radius = self.CELL_SIZE * 0.55
        irregularity = self.CELL_SIZE * 0.18

        # Desplazamos el centro visual ligeramente hacia arriba para dar volumen
        cy_visual = cy + self.CELL_SIZE * 0.1

        for i in range(num_points):
            angle = (math.pi * 2 * i) / num_points
            radius = base_radius + rng.uniform(-irregularity, irregularity)
            px = cx + math.cos(angle) * radius
            py = cy_visual + math.sin(angle) * radius
            points.append((px, py))

        # Sombra (usando el centro real de la celda para la base)
        shadow_offset_x, shadow_offset_y = 6, -6
        # Proyectamos la sombra desde la forma visual al suelo real (cy)
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

        # Highlight (para dar brillo en la parte superior izquierda)
        highlight_color = (min(
            255, rock_color[0]+50), min(255, rock_color[1]+50), min(255, rock_color[2]+50))
        # Ángulos correspondientes a la parte superior izquierda (aprox 110 a 160 grados)
        start_highlight = int(num_points * (110/360.0))
        end_highlight = int(num_points * (160/360.0))
        highlight_points = points[start_highlight: end_highlight+1]

        if len(highlight_points) > 1:
            self.arcade.draw_line_strip(highlight_points, highlight_color, 4)

    def _draw_heatmap(self, q_table_to_render):
        # Dibuja la visualización de la Q-Table solo si el modo debug está activo.
        if not self.debug_mode or q_table_to_render is None:
            return

        try:
            max_q = float(np.max(q_table_to_render))
            min_q = float(np.min(q_table_to_render))
        except Exception:
            return

        q_range = max_q - min_q
        if q_range < 1e-6:
            return

        # Mantenemos el tamaño del 75% que se ve bien
        SQUARE_SIZE = self.CELL_SIZE * 0.75

        for state_index in range(self.game.grid_size * self.game.grid_size):
            x_cell = state_index % self.game.grid_size
            y_cell = state_index // self.game.grid_size
            cx, cy = self._cell_to_pixel(x_cell, y_cell)

            try:
                # Volvemos a usar el valor MÁXIMO de Q para colorear la celda
                q_value = float(np.max(q_table_to_render[state_index, :]))
            except Exception:
                continue

            norm_q = (q_value - min_q) / q_range

            # --- NUEVO GRADIENTE "VIRIDIS-LIKE" (Azul -> Verde -> Amarillo) ---
            if norm_q < 0.5:
                # Interpola de Azul oscuro (44, 1, 105) a Verde brillante (30, 250, 100)
                t = norm_q * 2
                r = int(44 * (1 - t) + 30 * t)
                g = int(1 * (1 - t) + 250 * t)
                b = int(105 * (1 - t) + 100 * t)
            else:
                # Interpola de Verde brillante a Amarillo Fuego (253, 231, 37)
                t = (norm_q - 0.5) * 2
                r = int(30 * (1 - t) + 253 * t)
                g = int(250 * (1 - t) + 231 * t)
                b = int(100 * (1 - t) + 37 * t)

            # Alpha ajustado para mayor contraste
            base_alpha = 40
            value_alpha = norm_q * 190
            final_alpha = int(base_alpha + value_alpha)

            heat_color = (r, g, b, final_alpha)

            left = cx - SQUARE_SIZE / 2
            bottom = cy - SQUARE_SIZE / 2

            if self.draw_lbwh_rectangle_filled:
                self.draw_lbwh_rectangle_filled(
                    left, bottom, SQUARE_SIZE, SQUARE_SIZE, heat_color)

    def _draw_anthill(self):
        # RESTAURADO: Dibuja la entrada del hormiguero con volumen y altura (Estilo Terraced).
        gx, gy = self.game.goal_pos
        cx, cy = self._cell_to_pixel(gx, gy)

        # Semilla para detalles procedurales consistentes
        current_hash = hash(tuple(self.game.goal_pos))
        try:
            rng = np.random.default_rng(abs(current_hash) % (2**32))
        except AttributeError:
            rng = np.random.RandomState(abs(current_hash) % (2**32))

        mound_color_base = (168, 129, 98)  # Color tierra base
        hole_color = self.COLOR_GOAL

        # 1. Configuración del montículo
        base_radius_x = self.CELL_SIZE * 1.1
        base_radius_y = self.CELL_SIZE * 0.8
        max_height = self.CELL_SIZE * 0.3

        # Sombra proyectada en el suelo
        self.arcade.draw_ellipse_filled(
            cx + 5, cy - 5, base_radius_x, base_radius_y, (50, 50, 50, 80))

        # 2. Dibujar el montículo usando capas para simular altura (Estilo Terraced)
        layers = 5
        for i in range(layers):
            progress = i / (layers - 1)
            scale = 1.0 - (progress * 0.3)
            lightness = progress * 50

            color = (
                min(255, mound_color_base[0] + lightness),
                min(255, mound_color_base[1] + lightness),
                min(255, mound_color_base[2] + lightness)
            )

            offset_y = progress * max_height

            self.arcade.draw_ellipse_filled(
                cx, cy + offset_y, base_radius_x * scale, base_radius_y * scale, color)

        # 3. Textura (Granos de arena/tierra)
        for _ in range(60):
            angle = rng.uniform(0, 2 * math.pi)
            dist_factor = rng.uniform(0, 1)**2
            distance_x = dist_factor * base_radius_x * 0.8
            distance_y = dist_factor * base_radius_y * 0.8

            px = cx + math.cos(angle) * distance_x
            py = cy + math.sin(angle) * distance_y + \
                max_height * (1.0 - dist_factor) * 0.9

            try:
                shade = rng.integers(-20, 20)
            except (AttributeError, TypeError):
                shade = rng.randint(-20, 20)

            grain_color = (max(0, min(255, mound_color_base[0]+shade+40)),
                           max(0, min(255, mound_color_base[1]+shade+40)),
                           max(0, min(255, mound_color_base[2]+shade+40)))
            # Usamos draw_circle_filled compatible.
            self.arcade.draw_circle_filled(
                px, py, rng.uniform(1.5, 3.0), grain_color)

        # 4. Agujero oscuro en la cima
        hole_center_y = cy + max_height * 0.95
        self.arcade.draw_ellipse_filled(
            cx, hole_center_y, self.CELL_SIZE * 0.30, self.CELL_SIZE * 0.18, hole_color)

        # Guardamos la posición del agujero (en Píxeles) para la animación de éxito
        self.anthill_hole_visual_center = (cx, hole_center_y)

    def _draw_ant(self):
        # Dibuja la hormiga animada y procedural.
        # Si la escala es 0 (fin de la transición), no dibujamos nada.
        if self.ant_scale <= 0.01:
            return

        ax, ay = self.ant_display_pos
        cx, cy = self._cell_to_pixel(ax, ay)

        # Aplicar escala global para la animación de éxito
        SCALE = self.ant_scale

        # Determinar la dirección basada en la última acción tomada por el agente
        action = self.game.last_action
        if action == 0:   # Arriba (Gym) -> Arriba (Arcade)
            target_angle = 90
        elif action == 1:  # Abajo (Gym) -> Abajo (Arcade)
            target_angle = 270
        elif action == 2:  # Izquierda
            target_angle = 180
        elif action == 3:  # Derecha
            target_angle = 0
        else:
            target_angle = 0

        angle = target_angle

        # Definición del cuerpo (Ajustado por escala)
        body_color = self.COLOR_ANT
        shadow_color = (
            int(body_color[0]*0.3), int(body_color[1]*0.3), int(body_color[2]*0.3), 180)
        leg_color = (
            max(0, body_color[0]-50), max(0, body_color[1]-50), max(0, body_color[2]-50))

        head_radius = self.CELL_SIZE * 0.16 * SCALE
        thorax_radius_x = self.CELL_SIZE * 0.21 * SCALE
        thorax_radius_y = self.CELL_SIZE * 0.18 * SCALE
        abdomen_radius_x = self.CELL_SIZE * 0.28 * SCALE
        abdomen_radius_y = self.CELL_SIZE * 0.22 * SCALE

        # Función auxiliar para rotar puntos alrededor del centro
        angle_rad = math.radians(angle)

        def rotate(x, y):
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            return rx, ry

        # Animación y Movimiento ---
        t = time.time()

        # Detectar movimiento: Si estamos en transición de éxito, consideramos que estamos moviéndonos.
        if self.in_success_transition:
            moving = True
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
            for i, offset_angle in enumerate([-40, 0, 40]):
                # Lógica para alternar el movimiento de las patas
                is_set_1 = (side == 1 and i != 1) or (side == -1 and i == 1)
                current_oscillation = oscillation if is_set_1 else -oscillation
                osc = current_oscillation * leg_oscillation_amount
                end_angle_deg = angle + (90 + offset_angle + osc) * side
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
        self.arcade.draw_circle_filled(
            cx + hx_rel + shadow_offset_x, cy + hy_rel + shadow_offset_y, head_radius, shadow_color)
        # Cuerpo
        self.arcade.draw_circle_filled(
            cx + hx_rel, cy + hy_rel, head_radius, body_color)

        # Detalles de la Cabeza (Ajustado por escala) ---
        eye_radius = head_radius * 0.3
        eye_offset_x = head_radius * 0.4
        eye_offset_y = head_radius * 0.65
        for side in [-1, 1]:
            eox, eoy = rotate(eye_offset_x, eye_offset_y * side)
            self.arcade.draw_circle_filled(
                cx + hx_rel + eox, cy + hy_rel + eoy, eye_radius, (30, 30, 30))

        # Antenas
        antenna_length = head_radius * 1.8
        antenna_thickness = max(1, int(2 * SCALE))
        antenna_oscillation = oscillation * antenna_oscillation_amount
        for side in [-1, 1]:
            end_angle = angle + (45 * side) + antenna_oscillation
            start_offset_x = head_radius * 0.9
            start_offset_y = head_radius * 0.4 * side
            asx_rel, asy_rel = rotate(start_offset_x, start_offset_y)
            asx = cx + hx_rel + asx_rel
            asy = cy + hy_rel + asy_rel
            aex_rel = math.cos(math.radians(end_angle)) * antenna_length
            aey_rel = math.sin(math.radians(end_angle)) * antenna_length
            self.arcade.draw_line(asx, asy, asx + aex_rel,
                                  asy + aey_rel, leg_color, antenna_thickness)

    def _draw_particles(self):
        # RESTAURADO: Dibuja todas las partículas activas usando easing y draw_circle_filled.

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
            else:  # Estrella
                # Decaimiento lineal para estrellas
                fade_alpha = (1.0 - progress)

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

    def draw(self, game: AntGame, q_table_to_render, render_mode: str | None, simulation_speed: float = 1.0):
        # Función principal de dibujo del frame.

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

        # CAMBIO CLAVE: Escalamos el delta_time ---
        scaled_delta_time = delta_time * simulation_speed

        # Proceso de Dibujo y Actualización Coordinada ---

        if self.window:
            self.window.switch_to()
            self.window.clear()

        # 3. Heatmap
        self._draw_heatmap(q_table_to_render)

        # 11. Hormiguero debajo de todo
        self._draw_anthill()

        # 2. Elementos estáticos
        self._draw_static_elements()

        # Actualizamos las animaciones (movimiento, rotación, partículas) usando scaled_delta_time
        self._update_animations(scaled_delta_time)
        self._update_particles(scaled_delta_time)

        # Dibujamos la hormiga (usando la posición y ángulo actualizados)
        self._draw_ant()

        # 4. Post-procesado: Partículas
        self._draw_particles()

        return self.WIDTH, self.HEIGHT

# mlvlab/envs/ant/ant_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Arcade se importará solo si se llama al método de renderizado.


class LostAntEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 grid_size=10,
                 reward_food=100,
                 reward_obstacle=-100,
                 reward_move=-1
                 ):
        super().__init__()

        # Los valores ahora vienen de los parámetros, con un valor por defecto
        self.GRID_SIZE = grid_size
        self.REWARD_FOOD = reward_food
        self.REWARD_OBSTACLE = reward_obstacle
        self.REWARD_MOVE = reward_move

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.GRID_SIZE - 1, shape=(2,), dtype=np.int32
        )

        self.ant_pos = None
        self.food_pos = None
        self.obstacles = []

        self.render_mode = render_mode
        self.window = None
        self._window_visible = False
        self._last_time = None
        self.q_table_to_render = None  # Para visualización avanzada

        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def _generate_scenario(self):
        """Genera y establece las posiciones de la comida y los obstáculos."""
        self.food_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        self.obstacles = [
            self.np_random.integers(0, self.GRID_SIZE, size=2).tolist() for _ in range(self.GRID_SIZE)
        ]
        while self.food_pos.tolist() in self.obstacles:
            self.food_pos = self.np_random.integers(
                0, self.GRID_SIZE, size=2, dtype=np.int32)

    def _place_ant(self):
        """Busca una posición inicial aleatoria para la hormiga."""
        self.ant_pos = self.np_random.integers(
            0, self.GRID_SIZE, size=2, dtype=np.int32)
        while self.ant_pos.tolist() in self.obstacles or np.array_equal(self.ant_pos, self.food_pos):
            self.ant_pos = self.np_random.integers(
                0, self.GRID_SIZE, size=2, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Si se proporciona una semilla nueva, O si el escenario nunca se ha generado,
        # creamos un mapa nuevo (comida y obstáculos).
        if seed is not None or self.food_pos is None:
            self._generate_scenario()

        # En CADA reset, la hormiga busca una nueva posición aleatoria en el mapa actual.
        self._place_ant()

        return self._get_obs(), self._get_info()

    def set_render_data(self, q_table):
        self.q_table_to_render = q_table

    def _get_obs(self):
        # Devolver una copia para prevenir el "Bug de Observación Mutable"
        return np.copy(self.ant_pos)

    def _get_info(self):
        return {"food_pos": self.food_pos}

    def step(self, action):
        # Guardamos la posición anterior por si choca
        old_pos = np.copy(self.ant_pos)
        info = self._get_info()

        # 1. Mueve la hormiga
        if action == 0:
            self.ant_pos[1] -= 1  # Arriba
        elif action == 1:
            self.ant_pos[1] += 1  # Abajo
        elif action == 2:
            self.ant_pos[0] -= 1  # Izquierda
        elif action == 3:
            self.ant_pos[0] += 1  # Derecha

        # 2. Asegura que la hormiga esté dentro de los límites
        self.ant_pos = np.clip(self.ant_pos, 0, self.GRID_SIZE - 1)

        # 3. Evalúa el resultado
        terminated = np.array_equal(self.ant_pos, self.food_pos)
        truncated = False

        if terminated:
            reward = self.REWARD_FOOD
            # Volumen de 0-100, como espera el player.py
            info['play_sound'] = {'filename': 'blip.wav', 'volume': 10}
        elif self.ant_pos.tolist() in self.obstacles:
            reward = self.REWARD_OBSTACLE
            # La hormiga es devuelta a su posición anterior
            self.ant_pos = old_pos
            info['play_sound'] = {'filename': 'crash.wav', 'volume': 5}
        else:
            reward = self.REWARD_MOVE

        return self._get_obs(), reward, terminated, truncated, info

    def _render_frame(self):
        # --- SECCIÓN VISUAL (AISLADA) ---
        # Si nunca se llama a render(), esta sección nunca se ejecuta.
        import arcade
        from arcade.draw.rect import draw_lbwh_rectangle_filled
        import time

        # Constantes de dibujado (solo existen dentro de este método)
        CELL_SIZE = 30
        WIDTH = HEIGHT = self.GRID_SIZE * CELL_SIZE
        COLOR_GRID = (40, 40, 40)
        COLOR_ANT = (255, 64, 64)
        COLOR_FOOD = (64, 255, 128)
        COLOR_OBSTACLE = (120, 120, 120)

        # Crear ventana de Arcade perezosamente. Para rgb_array la mantenemos oculta.
        if self.window is None and self.render_mode in ["human", "rgb_array"]:
            self._window_visible = self.render_mode == "human"
            # En Arcade, el origen está en la esquina inferior-izquierda.
            # Usamos "visible" para soportar render fuera de pantalla.
            try:
                self.window = arcade.Window(
                    WIDTH, HEIGHT, "Lost Ant Colony", visible=self._window_visible
                )
            except TypeError:
                # Compatibilidad: versiones sin parámetro visible
                self.window = arcade.Window(WIDTH, HEIGHT, "Lost Ant Colony")
                if not self._window_visible:
                    try:
                        self.window.set_visible(False)
                    except Exception:
                        pass

        if self.render_mode is None:
            return None

        # Si cambiamos de modo rgb_array -> human, hacer visible la ventana
        if self.window is not None:
            target_visibility = self.render_mode == "human"
            if target_visibility and not self._window_visible:
                try:
                    self.window.set_visible(True)
                except Exception:
                    pass
                self._window_visible = True

        # Utilidades de coordenadas: convertir celda (x,y) a píxeles (origen abajo-izquierda en Arcade)
        def cell_to_pixel(x_cell: int, y_cell: int):
            x_px = x_cell * CELL_SIZE
            # Invertimos Y para mantener la misma orientación que en PyGame (y hacia abajo)
            y_px = (self.GRID_SIZE - 1 - y_cell) * CELL_SIZE
            return x_px, y_px

        # Renderizado base
        self.window.switch_to()
        arcade.set_background_color(COLOR_GRID)
        # Limpiar el frame usando el color de fondo configurado
        self.window.clear()

        # Dibujar mapa de calor de Q-table si está disponible
        if self.q_table_to_render is not None:
            max_q = float(np.max(self.q_table_to_render))
            min_q = float(np.min(self.q_table_to_render))
            if max_q > min_q:
                for state_index in range(self.GRID_SIZE * self.GRID_SIZE):
                    x_cell = state_index % self.GRID_SIZE
                    y_cell = state_index // self.GRID_SIZE
                    q_value = float(
                        np.max(self.q_table_to_render[state_index, :]))
                    norm_q = (q_value - min_q) / (max_q - min_q)
                    intensity = int(40 + norm_q * 180)
                    heat_color = (0, intensity, 0, 255)
                    x_px, y_px = cell_to_pixel(x_cell, y_cell)
                    draw_lbwh_rectangle_filled(
                        x_px,
                        y_px,
                        CELL_SIZE,
                        CELL_SIZE,
                        heat_color,
                    )

        # Dibujar cuadrícula sutil
        grid_color = (60, 60, 60, 120)
        for i in range(self.GRID_SIZE + 1):
            x = i * CELL_SIZE
            y = i * CELL_SIZE
            arcade.draw_line(x, 0, x, HEIGHT, grid_color, 1)
            arcade.draw_line(0, y, WIDTH, y, grid_color, 1)

        # Dibujar obstáculos
        for obs in self.obstacles:
            x_px, y_px = cell_to_pixel(obs[0], obs[1])
            # Sombra
            draw_lbwh_rectangle_filled(
                x_px + 2,
                y_px + 2,
                CELL_SIZE - 2,
                CELL_SIZE - 2,
                (90, 90, 90, 255),
            )
            # Cara superior con ligero brillo
            draw_lbwh_rectangle_filled(
                x_px,
                y_px,
                CELL_SIZE - 2,
                CELL_SIZE,
                COLOR_OBSTACLE,
            )

        # Dibujar comida con efecto pulso (brillo)
        fx_time = time.time()
        pulse = 0.5 + 0.5 * np.sin(fx_time * 3.0)
        fx_radius = int(6 + 8 * pulse)
        cx_food = self.food_pos[0] * CELL_SIZE + CELL_SIZE // 2
        cy_food = (self.GRID_SIZE - 1 -
                   self.food_pos[1]) * CELL_SIZE + CELL_SIZE // 2
        for r, alpha in [(fx_radius * 2, 30), (fx_radius, 60), (fx_radius // 2, 90)]:
            arcade.draw_circle_filled(
                cx_food, cy_food, max(2, r), (64, 255, 128, alpha))

        x_px, y_px = cell_to_pixel(self.food_pos[0], self.food_pos[1])
        draw_lbwh_rectangle_filled(
            x_px + 4,
            y_px + 4,
            CELL_SIZE - 8,
            CELL_SIZE - 8,
            (*COLOR_FOOD, 255),
        )

        # Dibujar hormiga (cuerpo + borde)
        ax_px, ay_px = cell_to_pixel(
            int(self.ant_pos[0]), int(self.ant_pos[1]))
        cx_ant = ax_px + CELL_SIZE // 2
        cy_ant = ay_px + CELL_SIZE // 2
        arcade.draw_circle_filled(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (*COLOR_ANT, 255))
        arcade.draw_circle_outline(
            cx_ant, cy_ant, CELL_SIZE * 0.35, (255, 255, 255, 180), 2)

        # Dirección indicativa simple (flecha hacia arriba/abajo/izq/der basada en último movimiento no almacenado)
        # Por simplicidad, dibujamos una "antena" hacia arriba
        arcade.draw_line(cx_ant, cy_ant, cx_ant, cy_ant +
                         CELL_SIZE * 0.45, (255, 220, 220, 200), 2)

        # No finalizamos aquí; la finalización la gestiona render() según modo
        return WIDTH, HEIGHT

    def render(self):
        # Importamos Arcade aquí para que esté disponible en todo el método.
        import arcade

        result = self._render_frame()
        if result is None:
            return None

        width, height = result

        if self.render_mode == "human":
            # Mostrar en pantalla y regular FPS
            if self.window is not None:
                try:
                    self.window.flip()
                except Exception:
                    pass
            # Pequeña pausa para sincronizar FPS
            import time as _time
            _time.sleep(1.0 / float(self.metadata.get("render_fps", 30)))
        elif self.render_mode == "rgb_array":
            # Capturamos el frame a una imagen y lo convertimos a numpy RGB (H, W, 3)
            try:
                image = arcade.get_image(0, 0, width, height)
            except TypeError:
                # Compatibilidad con firmas anteriores
                image = arcade.get_image()

            # Asegurar RGB sin alfa
            image = image.convert("RGB")
            frame = np.asarray(image)
            return frame

    def close(self):
        if self.window:
            try:
                # Cerrar ventana de Arcade
                self.window.close()
            except Exception:
                try:
                    import arcade
                    arcade.close_window()
                except Exception:
                    pass
            self.window = None

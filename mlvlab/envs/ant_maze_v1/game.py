# mlvlab/envs/ant_maze_v1/game.py
import numpy as np
import collections


class MazeGame:
    """
    Lógica del juego (estado y transición) para AntMaze.
    Mantiene la lógica del laberinto y el estado del agente.
    """

    def __init__(self, grid_size: int, reward_goal: int, reward_wall: int, reward_move: int) -> None:
        self.grid_size = int(grid_size)
        self.reward_goal = int(reward_goal)
        self.reward_wall = int(reward_wall)
        self.reward_move = int(reward_move)

        # Estado del juego
        self.ant_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.goal_pos = np.zeros(2, dtype=np.int32)  # [X, Y]
        self.walls: set[tuple[int, int]] = set()  # {(X, Y), ...}
        self.start_pos = np.zeros(2, dtype=np.int32)  # [X, Y]

        self._np_random = None

        # Estado para el renderer (Juicy)
        self.last_action = 3  # Derecha por defecto
        self.collided = False

        # Registras las celdas visitadas
        self.visited_cells = set()

    # Compatibilidad con código que espere 'obstacles'
    @property
    def obstacles(self):
        return self.walls

    def reset(self, np_random, hard: bool = False) -> None:
        """
        Reinicia el estado del juego.
        - hard=False (reintento): Solo reposiciona la hormiga.
        - hard=True (mapa nuevo): Limpia las celdas visitadas y reposiciona la hormiga.
        """
        self._np_random = np_random
        self.place_ant()
        self.last_action = 3
        self.collided = False

        # LÓGICA MODIFICADA ---
        if hard:
            # En un reinicio "duro", borramos el historial de exploración.
            self.visited_cells.clear()

        # Siempre añadimos la posición actual (sea la de inicio o la nueva).
        # Si no se limpió, esto no tiene efecto. Si se limpió, es la primera.
        self.visited_cells.add(tuple(self.ant_pos))

    def generate_scenario(self, np_random) -> None:
        """Genera el laberinto o un escenario simple si el grid es pequeño."""
        self._np_random = np_random
        self.walls = set()

        # Punto 4: Si el grid es menor que 10, generamos un escenario simple
        if self.grid_size < 10:
            self._generate_simple_scenario(np_random)
            return

        # 1. Generación Procedural (Recursive Backtracking)
        # 1 = Muro, 0 = Camino
        grid = np.ones((self.grid_size, self.grid_size), dtype=np.int8)

        def carve_passages_from(cx, cy, grid, rng):
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            # Usamos el RNG proporcionado por Gymnasium para determinismo
            try:
                # shuffle funciona tanto para default_rng como RandomState si directions es una lista
                rng.shuffle(directions)
            except (AttributeError, TypeError):
                # Fallback muy antiguo
                import random
                random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                # Aseguramos estar dentro de los límites (excluyendo el perímetro - Punto 5)
                if 0 < nx < self.grid_size - 1 and 0 < ny < self.grid_size - 1 and grid[ny, nx] == 1:
                    # Quitamos el muro entre la celda actual y la nueva
                    grid[cy + dy//2, cx + dx//2] = 0
                    grid[ny, nx] = 0
                    carve_passages_from(nx, ny, grid, rng)

        # Empezar desde una celda (1, 1)
        start_x, start_y = 1, 1
        grid[start_y, start_x] = 0
        carve_passages_from(start_x, start_y, grid, np_random)

        # 2. Definir Entrada y Salida (Punto 6) usando BFS para encontrar el punto más lejano
        self.start_pos = np.array([1, 1], dtype=np.int32)

        # BFS para encontrar la celda accesible más lejana (garantiza camino válido y largo)
        q = collections.deque([(1, 1, 0)])  # (x, y, distance)
        visited = {(1, 1)}
        farthest_cell = (1, 1)
        max_dist = 0

        while q:
            cx, cy, dist = q.popleft()

            if dist > max_dist:
                max_dist = dist
                farthest_cell = (cx, cy)

            # Explorar vecinos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                # Comprobamos límites válidos dentro del grid
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if grid[ny, nx] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny, dist + 1))

        self.goal_pos = np.array(farthest_cell, dtype=np.int32)

        # 3. Convertir la matriz numpy a un set de muros
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if grid[y, x] == 1:
                    self.walls.add((x, y))

    def _generate_simple_scenario(self, np_random):
        """Genera un escenario simple con obstáculos aleatorios (para GRID_SIZE < 10)."""
        # Colocamos la meta y el inicio aleatoriamente
        self.goal_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)

        self.start_pos = self._np_random.integers(
            0, self.grid_size, size=2, dtype=np.int32)
        while (self.start_pos[0] == self.goal_pos[0] and self.start_pos[1] == self.goal_pos[1]):
            self.start_pos = self._np_random.integers(
                0, self.grid_size, size=2, dtype=np.int32)

        # Generamos muros (20% de ocupación)
        num_walls = int(self.grid_size * self.grid_size * 0.2)
        self.walls = set()
        attempts = 0
        while len(self.walls) < num_walls and attempts < num_walls * 5:
            attempts += 1
            wall = tuple(self._np_random.integers(
                0, self.grid_size, size=2).tolist())
            if wall != tuple(self.start_pos.tolist()) and wall != tuple(self.goal_pos.tolist()):
                self.walls.add(wall)

    def place_ant(self, np_random=None) -> None:
        # En AntMaze, la hormiga siempre se posiciona en el punto de comienzo definido (Punto 6).
        self.ant_pos = np.copy(self.start_pos)

    def get_obs(self) -> np.ndarray:
        return np.array((int(self.ant_pos[0]), int(self.ant_pos[1])), dtype=np.int32)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        ax, ay = int(self.ant_pos[0]), int(self.ant_pos[1])
        info: dict = {}

        # Ignoramos acciones no válidas (como la acción 4 si no está activa)
        if action not in [0, 1, 2, 3]:
            # No hacemos nada y no penalizamos
            return self.get_obs(), 0, False, info

        self.last_action = action
        self.collided = False

        prev_ax, prev_ay = ax, ay
        target_ax, target_ay = ax, ay
        if action == 0:
            target_ay -= 1
        elif action == 1:
            target_ay += 1
        elif action == 2:
            target_ax -= 1
        elif action == 3:
            target_ax += 1

        # Comprobación de límites del grid
        out_of_bounds = (
            target_ax < 0 or target_ax >= self.grid_size or
            target_ay < 0 or target_ay >= self.grid_size
        )

        # Comprobación de muros
        is_wall = (target_ax, target_ay) in self.walls

        if out_of_bounds or is_wall:
            # Colisión contra muro o límite
            reward = self.reward_wall
            ax, ay = prev_ax, prev_ay  # La hormiga no se mueve
            terminated = False
            self.collided = True
        else:
            # Movimiento válido
            ax, ay = target_ax, target_ay
            terminated = (
                ax == int(self.goal_pos[0]) and ay == int(self.goal_pos[1]))

            if terminated:
                reward = self.reward_goal
            else:
                reward = self.reward_move

        # Actualizar posición
        self.ant_pos[0], self.ant_pos[1] = ax, ay

        # NUEVA LÍNEA: Registrar la celda visitada ---
        self.visited_cells.add(tuple(self.ant_pos))

        info["collided"] = self.collided
        info["terminated"] = terminated

        return np.array((ax, ay), dtype=np.int32), int(reward), bool(terminated), info

# create_wall_asset.py
import arcade
import numpy as np
from PIL import Image
from pathlib import Path

# Configuración de los Tiles ---
CELL_SIZE = 40
COLOR_WALL = (89, 69, 40)
# MODIFICACIÓN: Definimos cuántos tiles únicos queremos crear
NUM_TILES_TO_CREATE = 10
OUTPUT_FILENAME_PATTERN = "tile_wall_{i}.png"


def draw_wall(cx, cy, rng):
    """Dibuja un muro de tierra. Esta es la misma lógica que tenías antes."""
    left, bottom, base_color = cx - CELL_SIZE / 2, cy - CELL_SIZE / 2, COLOR_WALL
    arcade.draw_lrbt_rectangle_filled(
        left, left + CELL_SIZE, bottom, bottom + CELL_SIZE, base_color)
    num_motes = int(CELL_SIZE * CELL_SIZE / 50)
    for _ in range(num_motes):
        offset_x = rng.uniform(-CELL_SIZE / 2, CELL_SIZE / 2)
        offset_y = rng.uniform(-CELL_SIZE / 2, CELL_SIZE / 2)
        r = rng.uniform(1.5, 4)
        shade = rng.integers(-40, 40)
        mote_color = tuple(max(0, min(255, val + shade)) for val in base_color)
        arcade.draw_circle_filled(cx + offset_x, cy + offset_y, r, mote_color)
    border_color = tuple(max(0, val - 20) for val in base_color)
    arcade.draw_lrbt_rectangle_outline(
        left, left + CELL_SIZE, bottom, bottom + CELL_SIZE, border_color, 1)


def main():
    """Función principal para generar y guardar múltiples imágenes de muro."""
    print(
        f"Generando {NUM_TILES_TO_CREATE} tiles de muro de {CELL_SIZE}x{CELL_SIZE} píxeles...")
    window = arcade.Window(CELL_SIZE, CELL_SIZE,
                           "Generador de Assets", visible=False)
    ctx = window.ctx
    texture = ctx.texture((CELL_SIZE, CELL_SIZE))
    framebuffer = ctx.framebuffer(color_attachments=[texture])
    script_dir = Path(__file__).resolve().parent

    # MODIFICACIÓN: Bucle para crear cada tile
    for i in range(NUM_TILES_TO_CREATE):
        framebuffer.use()
        framebuffer.clear(color=(0, 0, 0, 0))

        # Usamos el índice del bucle `i` como 'semilla' (seed) para que cada muro sea visualmente único
        rng = np.random.default_rng(i)
        draw_wall(CELL_SIZE / 2, CELL_SIZE / 2, rng)

        image = Image.frombytes(
            "RGBA", (CELL_SIZE, CELL_SIZE), framebuffer.read(components=4))

        # Generamos el nombre del archivo para esta iteración
        output_filename = OUTPUT_FILENAME_PATTERN.format(i=i)
        output_path = script_dir / output_filename
        image.save(output_path)
        print(
            f" -> Tile {i+1}/{NUM_TILES_TO_CREATE} guardado en '{output_path}'")

    print(f"\n✅ ¡Proceso completado!")
    window.close()


if __name__ == "__main__":
    main()

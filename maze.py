import random
import numpy as np
import matplotlib.pyplot as plt

MAZEWIDTH, MAZEHEIGHT = 11,11 
DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

def generate_maze(width, height):
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1
    maze = np.ones((height, width), dtype=int)
    start_x, start_y = random.randrange(1, width - 1, 2), random.randrange(1, height - 1, 2)
    maze[start_y, start_x] = 0
    walls = []
    def add_walls(x, y):
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 1:
                walls.append((nx, ny, x, y)) 

    add_walls(start_x, start_y)
    while walls:
        idx = random.randint(0, len(walls) - 1)
        wx, wy, px, py = walls.pop(idx) 

        if maze[wy, wx] == 1:
            maze[wy, wx] = 0 
            maze[(wy + py) // 2, (wx + px) // 2] = 0 

            add_walls(wx, wy)

    return maze


if __name__ ==  "__main__":
    ODD_WIDTH = 11
    ODD_HEIGHT = 11
    maze = generate_maze(ODD_WIDTH, ODD_HEIGHT)
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()

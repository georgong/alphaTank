import numpy as np
import random

class MazeGenerator:
    def __init__(self, mazewidth, mazeheight, grid_size, use_octagon=True):
        self.mazewidth = mazewidth
        self.mazeheight = mazeheight
        self.GRID_SIZE = grid_size
        self.USE_OCTAGON = use_octagon

    def construct_wall(self):
        """Creates a battlefield-style map with cover instead of a random maze."""
        maze = np.ones((self.mazeheight, self.mazewidth), dtype=int) if self.USE_OCTAGON else np.zeros((self.mazeheight, self.mazewidth), dtype=int)

        if self.USE_OCTAGON:
            maze[1:-1, 1:-1] = 0
        else:
            # Border walls
            maze[0, :] = 1
            maze[-1, :] = 1
            maze[:, 0] = 1
            maze[:, -1] = 1

            # Central cross
            cx, cy = self.mazewidth // 2, self.mazeheight // 2
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                maze[cy + dy, cx + dx] = 1

            # Side covers
            cover_positions = [
                (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
                (self.mazewidth - 3, 3), (self.mazewidth - 3, 4), (self.mazewidth - 3, 5),
                (self.mazewidth - 3, 6), (self.mazewidth - 3, 7),
                (4, 8), (5, 8), (6, 8),
                (4, 2), (5, 2), (6, 2),
            ]
            for r, c in cover_positions:
                maze[r, c] = 1

        #empty_space = self._get_empty_space(maze)
        return maze#, empty_space

    def construct_wall_2(self):
        """Creates a map with a smaller central play area and limited movement space."""
        maze = np.ones((self.mazeheight, self.mazewidth), dtype=int)

        if self.USE_OCTAGON:
            maze[2:-2, 2:-2] = 0
        else:
            playable_min = 2
            playable_max = self.mazewidth - 3
            maze[playable_min:playable_max, playable_min:playable_max] = 0

            # Side barriers
            for i in range(playable_min, playable_max):
                maze[playable_min + 1, i] = 1
                maze[playable_max - 1, i] = 1
                maze[i, playable_min + 1] = 1
                maze[i, playable_max - 1] = 1

            # Central covers
            cx, cy = self.mazewidth // 2, self.mazeheight // 2
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                maze[cy + dy, cx + dx] = 1

            # Side dodging covers
            side_covers = [
                (playable_min + 2, playable_min + 3), (playable_min + 2, playable_max - 3),
                (playable_max - 2, playable_min + 3), (playable_max - 2, playable_max - 3),
            ]
            for r, c in side_covers:
                maze[r, c] = 1

        #empty_space = self._get_empty_space(maze)
        return maze#, empty_space

    def generate_maze(self, width, height):
        """Generates a random maze using recursive backtracking."""
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        maze = np.ones((height, width), dtype=int)
        start_x = random.randrange(1, width - 1, 2)
        start_y = random.randrange(1, height - 1, 2)
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

        #empty_space = self._get_empty_space(maze, width, height)
        return maze#, empty_space

    def _get_empty_space(self, maze, w=None, h=None):
        """Extracts empty positions (0s) and converts to pixel coordinates."""
        empty_space = []
        h = h if h is not None else self.mazeheight
        w = w if w is not None else self.mazewidth
        for row in range(h):
            for col in range(w):
                if maze[row, col] == 0:
                    empty_space.append((col * self.GRID_SIZE, row * self.GRID_SIZE))
        return empty_space



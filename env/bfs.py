
from collections import deque
import pygame
from env.config import *

def bfs_path(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    start = tuple(start)
    goal = tuple(goal)
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        print("the start agent not on the map")
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        print("the end agent not on the map")
        return None
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("the agent starts in the wall.")
        return None

    visited = [[False]*cols for _ in range(rows)]
    parent = dict()

    queue = deque([start])
    visited[start[0]][start[1]] = True
    found = False

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue and not found:
        r, c = queue.popleft()
        if (r, c) == goal:
            found = True
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr][nc] and grid[nr][nc] == 0:
                    visited[nr][nc] = True
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

    if not found:
        print("not found")
        return None

    # Retrace the path
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


def get_bfs_recommended_action(current_pos, next_pos):
    """
    Returns a tuple (rotate, move, shoot) consistent with BFS direction.
    We'll ignore tank orientation for simplicity, just the raw direction:
      - up = (row-1, col)
      - down = (row+1, col)
      - left = (row, col-1)
      - right = (row, col+1)
    
    rotate: +1 => left, -1 => right, 0 => no rotation
    move: +1 => forward, -1 => backward, 0 => stay
    shoot: 0 => don't shoot, 1 => shoot
    
    This is a simplified approach that won't necessarily match your 
    rotation-based logic perfectly, but demonstrates the idea.
    """
    (r, c) = current_pos
    (nr, nc) = next_pos
    
    # We'll just define a naive approach: 
    dr, dc = nr - r, nc - c
    
    # We'll assume "forward" = +1 if up, 
    # "backward" = -1 if down, etc., ignoring rotation for now.
    if dr == -1 and dc == 0:
        # BFS says "go up"
        return (0, 1, 0)  # (no rotate, forward, no shoot)
    elif dr == 1 and dc == 0:
        # BFS says "go down"
        return (0, -1, 0) # (no rotate, backward, no shoot)
    elif dr == 0 and dc == -1:
        # BFS says "go left"
        return (1, 0, 0)  # e.g. rotate left, but not move? 
                          # This is tricky if your logic requires rotation.
    elif dr == 0 and dc == 1:
        # BFS says "go right"
        return (-1, 0, 0) # rotate right
    else:
        # If BFS next step is diagonal or something else,
        # or no BFS next step is found
        return (0, 0, 0)



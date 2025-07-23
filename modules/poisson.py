import random  # type: ignore
import math
import numpy as np

'''
Poisson disk sampling algorithm, returns a list of (x, y) coordinates to be used for instancing positions.
'''
class PoissonClass:

    @staticmethod
    def generate_poisson_points(size, world_x, world_y, r_min=0.2, r_max=1, k=1000):
        cell_size = r_min / math.sqrt(2)
        grid_width = int(math.ceil(size / cell_size))
        grid_height = int(math.ceil(size / cell_size))
        
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        points = []
        spawn_points = []

        def get_cell_coords(pt):
            local_x = pt[0] - world_x
            local_y = pt[1] - world_y
            return int(local_x / cell_size), int(local_y / cell_size)

        def is_valid(pt, r_local):
            gx, gy = get_cell_coords(pt)
            for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
                for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                    neighbor = grid[x][y]
                    if neighbor is not None:
                        dist = math.dist(pt, neighbor)
                        if dist < r_local:
                            return False
            return True

        # === Generate initial point in world coordinates ===
        initial_pt = (
            random.uniform(world_x, world_x + size),
            random.uniform(world_y, world_y + size)
        )
        points.append(initial_pt)
        spawn_points.append(initial_pt)
        gx, gy = get_cell_coords(initial_pt)
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            grid[gx][gy] = initial_pt
        else:
            print(f"[WARN] Initial point grid index out of bounds: gx={gx}, gy={gy}")

        # === Sampling loop ===
        while spawn_points:
            idx = random.randint(0, len(spawn_points) - 1)
            spawn_center = spawn_points[idx]
            found = False
            for _ in range(k):
                angle = random.uniform(0, 2 * math.pi)
                rad = random.uniform(r_min, r_max)
                r_local = rad  # allows localized clustering
                new_x = spawn_center[0] + math.cos(angle) * rad
                new_y = spawn_center[1] + math.sin(angle) * rad
                candidate = (new_x, new_y)

                # Must be within the world-local tile boundaries
                if (world_x <= new_x < world_x + size and
                    world_y <= new_y < world_y + size and
                    is_valid(candidate, r_local)):
                    
                    points.append(candidate)
                    spawn_points.append(candidate)
                    gx, gy = get_cell_coords(candidate)
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        grid[gx][gy] = candidate
                    else:
                        print(f"[WARN] Candidate point grid index out of bounds: gx={gx}, gy={gy}")
                    found = True
                    break
            if not found:
                spawn_points.pop(idx)

        print(f"Poisson scattered {len(points)} points.")
        return points

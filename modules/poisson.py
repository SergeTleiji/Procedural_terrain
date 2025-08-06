"""
poisson.py
----------
Generates Poisson disk sample points for vegetation scattering, such as trees,
ensuring a minimum spacing between points to avoid clustering.

Purpose in Pipeline:
    - Step 8 (tree scattering) in `main.py`.
    - Produces evenly spaced vegetation placement points for sparser objects
      like trees, where uniform randomness would cause unrealistic clustering.
    - Stores results in a per-tile dictionary for efficient streaming.

Workflow:
    1. If a cached `.npy` file exists, load and return it.
    2. Initialize a sampling grid based on minimum spacing.
    3. Seed the grid with an initial random point.
    4. Iteratively:
        - Pick a random "spawn point" from the active list.
        - Generate up to `k` random candidates within [r_min, r_max] distance.
        - Accept candidates that are inside the area and far enough from neighbors.
        - Add accepted candidates to both the final point list and spawn list.
    5. Convert final point set into tile-based dictionary.
    6. Save results to `.npy` for caching.

Inputs:
    - output_dir (str): Folder to save/load cached results.
    - size (float): Width/height of the square sampling area (meters).
    - r_min (float): Minimum allowed distance between points.
    - r_max (float): Maximum distance when generating candidates.
    - k (int): Number of candidates to try per spawn point before discarding it.
    - tile_size (float): Tile size in meters for bucketing points.
    - seed (int or None): RNG seed for deterministic output.

Outputs:
    - dict: {(tile_x, tile_y): [(x, y), ...]} — per-tile point dictionary.
    - `.npy` file containing the same dictionary for reuse.

Dependencies:
    - numpy
    - math
    - random
    - os
    - Called by `main.py` → `PoissonClass.generate_poisson_points()`.

Example:
    trees = PoissonClass.generate_poisson_points(
        output_dir="output",
        size=500,
        r_min=5.0,
        r_max=7.0,
        k=1000,
        tile_size=10,
        seed=42
    )
"""

import random
import math
import numpy as np
import os


class PoissonClass:
    """
    Implements Poisson disk sampling for vegetation scattering.
    """

    @staticmethod
    def generate_poisson_points(
        output_dir,
        size=500,  # Size of area in meters (square)
        r_min=0.2,  # Minimum spacing between points
        r_max=1.0,  # Maximum spacing for candidate generation
        k=1000,  # Candidates per spawn point
        tile_size=10,  # Tile size for dictionary keys
        seed=None,  # RNG seed for reproducibility
    ):
        """
        Generates Poisson disk points over a square region.

        Args:
            output_dir (str): Output directory for cached points.
            size (float): Square side length (meters).
            r_min (float): Minimum distance between points.
            r_max (float): Max distance for candidate point generation.
            k (int): Candidates per spawn point before removal.
            tile_size (float): Tile size for per-tile bucketing.
            seed (int or None): Seed for deterministic output.

        Returns:
            dict: Keys are (tile_x, tile_y), values are lists of (x, y) points.
        """
        save_path = os.path.join(output_dir, "poisson.npy")

        # === Load from cache if available ===
        if os.path.exists(save_path):
            print("Loaded cached Poisson points")
            return np.load(save_path, allow_pickle=True).item()

        # === Set RNG seeds for reproducibility ===
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # === Grid initialization ===
        cell_size = r_min / math.sqrt(2)
        grid_width = int(math.ceil(size / cell_size))
        grid_height = int(math.ceil(size / cell_size))
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

        points = []
        spawn_points = []

        def get_cell_coords(pt):
            return int(pt[0] / cell_size), int(pt[1] / cell_size)

        def is_valid(pt, r_local):
            gx, gy = get_cell_coords(pt)
            for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
                for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                    neighbor = grid[x][y]
                    if neighbor is not None and math.dist(pt, neighbor) < r_local:
                        return False
            return True

        # === Seed with initial point ===
        initial_pt = (random.uniform(0, size), random.uniform(0, size))
        points.append(initial_pt)
        spawn_points.append(initial_pt)
        gx, gy = get_cell_coords(initial_pt)
        grid[gx][gy] = initial_pt

        # === Main Poisson sampling loop ===
        while spawn_points:
            idx = random.randint(0, len(spawn_points) - 1)
            spawn_center = spawn_points[idx]
            found = False

            for _ in range(k):
                angle = random.uniform(0, 2 * math.pi)
                rad = random.uniform(r_min, r_max)
                new_x = spawn_center[0] + math.cos(angle) * rad
                new_y = spawn_center[1] + math.sin(angle) * rad
                candidate = (new_x, new_y)

                if 0 <= new_x < size and 0 <= new_y < size and is_valid(candidate, rad):
                    points.append(candidate)
                    spawn_points.append(candidate)
                    gx, gy = get_cell_coords(candidate)
                    grid[gx][gy] = candidate
                    found = True
                    break

            if not found:
                spawn_points.pop(idx)

        print(f"[✓] Poisson scattered {len(points)} points (seed={seed}).")

        # === Convert point list to tile-bucketed dict ===
        grid_dict = {}
        for x, y in points:
            tile_x = int(x // tile_size)
            tile_y = int(y // tile_size)
            grid_dict.setdefault((tile_x, tile_y), []).append((x, y))

        # Save to cache
        np.save(save_path, grid_dict)
        return grid_dict

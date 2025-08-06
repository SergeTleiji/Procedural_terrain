"""
random_generation.py
--------------------
Generates dense random scatter points for vegetation placement, such as grass.
Produces a similar output format to Poisson disk sampling but uses uniform random
generation for speed.

Purpose in Pipeline:
    - Step 8 (grass scattering) in `main.py`.
    - Ideal for dense vegetation (e.g., grass) where Poisson disk sampling
      would be too slow.
    - Produces per-tile point buckets for efficient streaming and instancing.

Workflow:
    1. If a cached `.npy` file of points exists, load and return it.
    2. If not, generate uniform random (x, y) positions within the terrain bounds.
    3. Bucket points into grid cells based on tile size.
    4. Save the resulting dictionary to disk for reuse.

Inputs:
    - output_dir (str): Where to store/load the cached `.npy` file.
    - size (int): Terrain width in meters.
    - density (int): Number of points per square meter.
    - tile_size (int): Size of a tile in meters for bucketing.
    - seed (int or None): Optional random seed for reproducibility.

Outputs:
    - dict: Keys are (tile_x, tile_y), values are lists of (x, y) point tuples.
    - `.npy` file containing the dictionary for caching.

Dependencies:
    - numpy
    - random
    - os
    - Called by `main.py` → `RandomScatterClass.generate_random_points()`.

Example:
    grass_points = RandomScatterClass.generate_random_points(
        output_dir="output",
        size=500,
        density=20,
        tile_size=10,
        seed=42
    )
"""

import random
import numpy as np
import os


class RandomScatterClass:
    """
    Generates dense vegetation scatter points using uniform random distribution.
    """

    @staticmethod
    def generate_random_points(output_dir, size, density=100, tile_size=10, seed=None):
        """
        Generates and buckets random (x, y) vegetation points.

        Args:
            output_dir (str): Folder for caching the generated points.
            size (int): Terrain width in meters.
            density (int): Points per square meter.
            tile_size (int): Tile size in meters (used for bucketing).
            seed (int or None): Random seed for reproducibility.

        Returns:
            dict: {(tile_x, tile_y): [(x1, y1), (x2, y2), ...]}
        """
        num_points = density * (size**2)
        save_path = os.path.join(output_dir, "random.npy")

        # === Load from cache if available ===
        if os.path.exists(save_path):
            print("Loaded cached random points")
            return np.load(save_path, allow_pickle=True).item()

        # === Set RNG seed if provided ===
        if seed is not None:
            random.seed(seed)

        # === Generate uniform random points ===
        points = [
            (random.uniform(0, size), random.uniform(0, size))
            for _ in range(num_points)
        ]
        print("Generated new random point set with seed", seed)

        # === Organize points into tile buckets ===
        grid_dict = {}
        for x, y in points:
            tile_x = int(x // tile_size)
            tile_y = int(y // tile_size)
            key = (tile_x, tile_y)
            grid_dict.setdefault(key, []).append((x, y))

        # === Save for caching ===
        np.save(save_path, grid_dict)
        return grid_dict

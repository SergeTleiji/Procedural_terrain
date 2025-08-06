"""
generate_semantic_map.py
------------------------
Generates a semantic map for vegetation placement by assigning vegetation
types and scale classes to each pixel in the terrain.

Purpose in Pipeline:
    - Step 7 in `main.py`: determines where and how vegetation will be placed.
    - Uses procedural noise to vary vegetation type and density across the map.
    - Encodes vegetation model ID and scale class into a 2-channel map.

Workflow:
    1. Generate two layers of value noise (base and detail) to simulate natural variation.
    2. Combine and normalize noise to form a density map.
    3. Assign vegetation model IDs based on provided probability weights.
    4. Compute scale classes (0–3) based on local density using a convolution filter.
    5. Store results in a semantic map (channels: model_id, scale_class).
    6. Save both a `.npy` binary map and a `.png` visual representation.

Inputs:
    - probabilities: List of probabilities for each vegetation type (sum should be 1.0).
    - map_size: Output resolution (pixels per side).
    - seed: Random seed for reproducibility.

Outputs:
    - semantic_map.npy: Binary map containing vegetation model IDs and scale classes.
    - semantic_map_visual.png: Visual preview (R = model_id, G = scale class).

Dependencies:
    - numpy
    - scipy.ndimage.convolve
    - Pillow (PIL)
    - Called by `main.py` → `SemanticMap.build()` and `.save()`.

Example:
    semantic = SemanticMap([0.2, 0.3, 0.5], map_size=1024, seed=42)
    semantic.build()
    semantic.save("output/semantic_map.npy", "output/semantic_map_visual.png")
"""

import numpy as np
from scipy.ndimage import convolve
from PIL import Image


class SemanticMap:
    """
    Builds a semantic vegetation map with type and scale variation.
    """

    def __init__(self, probabilities, map_size=1024, seed=42):
        self.map_size = map_size
        self.seed = seed
        np.random.seed(self.seed)
        self.semantic_map = None
        self.model_id = None
        self.scale_class = None
        self.probabilities = probabilities

    def generate_value_noise(self, freq):
        """
        Generates 2D value noise using bilinear interpolation.

        Args:
            freq (int): Frequency of the base grid (higher = more variation).

        Returns:
            np.ndarray: Value noise array of shape (map_size, map_size).
        """
        size = self.map_size
        grid_size = (size // freq) + 2
        grid = np.random.rand(grid_size, grid_size)
        result = np.zeros((size, size))

        for y in range(size):
            for x in range(size):
                gx = x / freq
                gy = y / freq
                x0 = int(gx)
                y0 = int(gy)
                tx = gx - x0
                ty = gy - y0

                # Bilinear interpolation between four grid points
                v00 = grid[y0, x0]
                v10 = grid[y0, x0 + 1]
                v01 = grid[y0 + 1, x0]
                v11 = grid[y0 + 1, x0 + 1]
                a = (1 - tx) * v00 + tx * v10
                b = (1 - tx) * v01 + tx * v11
                result[y, x] = (1 - ty) * a + ty * b

        return result

    def build(self):
        """
        Creates the semantic map by assigning vegetation model IDs
        and scale classes based on procedural noise.
        """
        # === Generate density noise ===
        base_noise = self.generate_value_noise(freq=30)
        detail_noise = self.generate_value_noise(freq=10) * 0.5
        density_noise = base_noise + detail_noise
        density_noise -= density_noise.min()
        density_noise /= density_noise.max()

        # === Assign vegetation model IDs ===
        bins = np.cumsum([0.0] + self.probabilities)
        self.model_id = np.digitize(density_noise, bins=bins) - 1

        # === Assign scale classes based on local density ===
        self.scale_class = np.zeros_like(self.model_id)
        kernel = np.ones((5, 5), dtype=np.uint8)

        for i in range(len(self.probabilities)):
            mask = (self.model_id == i).astype(np.uint8)
            local_density = convolve(mask, kernel, mode="constant")
            self.scale_class[self.model_id == i] = np.digitize(
                local_density[self.model_id == i], bins=[2, 6, 10]
            )  # Scale classes: 0–3

        # === Assemble semantic map (2 channels: model_id, scale_class) ===
        self.semantic_map = np.zeros((self.map_size, self.map_size, 2), dtype=np.uint8)
        self.semantic_map[:, :, 0] = self.model_id
        self.semantic_map[:, :, 1] = self.scale_class

    def save(self, path_npy, path_png):
        """
        Saves the semantic map to disk as both .npy and .png.

        Args:
            path_npy (str): Path to save binary semantic map.
            path_png (str): Path to save visual representation.
        """
        if self.semantic_map is None:
            raise ValueError("Semantic map not generated. Call build() first.")

        # Save binary semantic map
        np.save(path_npy, self.semantic_map)
        print(f"✅ Saved: {path_npy}")

        # Create visual preview (R = model_id, G = scale class)
        visual_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        visual_img[:, :, 0] = (self.model_id * 50) % 255
        visual_img[:, :, 1] = (self.scale_class * 85) % 255
        Image.fromarray(visual_img).save(path_png)
        print(f"✅ Saved: {path_png}")

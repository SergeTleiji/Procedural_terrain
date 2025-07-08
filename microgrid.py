from perlin_noise import PerlinNoise
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class PerlinNoiseGenerator:
    def __init__(self, seed=0, tile_res=1024, tile_size_m=5.0):
        self.seed = seed
        self.tile_res = tile_res
        self.tile_size_m = tile_size_m
        self.m_per_pixel = tile_size_m / tile_res

        # Create layered Perlin noise generators
        self.noise_micro = PerlinNoise(octaves=4, seed=self.seed + 20)

    def get_height(self, x_m, y_m):
        micro = self.noise_micro([x_m / 2.5, y_m / 2.5]) * .2           

        height = (micro) / 2.0            # *********** NO MICRO FOR NOW (+ MICRO)
        return np.clip((height + 1.0) / 2.0, 0, 1)

    def generate_tile(self, row, col):
        x_start_m = col * self.tile_size_m
        y_start_m = row * self.tile_size_m
        return self.generate_area(x_start_m, y_start_m, self.tile_size_m, self.tile_res)

    def generate_area(self, x_start_m, y_start_m, width_m, res):
        tile = np.zeros((res, res), dtype=np.float32)
        m_per_pixel = width_m / res

        for i in range(res):
            for j in range(res):
                x_m = x_start_m + i * m_per_pixel
                y_m = y_start_m + j * m_per_pixel
                tile[j, i] = self.get_height(x_m, y_m)
        return tile

    def save_heightmap(self, heightmap, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.imsave(path, heightmap, cmap='gray', vmin=0, vmax=1)
        print(f"Saved heightmap to: {path}")

    def generate_and_save_tile(self, row, col, output_path, upscale_to=None):
        heightmap = self.generate_tile(row, col)
        self.save_heightmap(heightmap, output_path)

        if upscale_to:
            upscaled = cv2.resize(heightmap, (upscale_to, upscale_to), interpolation=cv2.INTER_CUBIC)
            upscale_path = output_path.replace(".png", f"_upscaled_{upscale_to}.png")
            self.save_heightmap(upscaled, upscale_path)

    def generate_and_save_area(self, x_start_m, y_start_m, width_m, res, output_path, apply_blur=False):
        heightmap = self.generate_area(x_start_m, y_start_m, width_m, res)
        if apply_blur:
            heightmap = cv2.GaussianBlur(heightmap, (3, 3), sigmaX=0.5)
        self.save_heightmap(heightmap, output_path)


# === Instantiate and Generate ===

generator = PerlinNoiseGenerator(
    seed=800,
    tile_res=1024,
    tile_size_m=5.0
)

# Generate small tile
generator.generate_and_save_tile(
    row=3,
    col=9,
    output_path="Terrain_generation/output/R3C9micro.png"
)

# Generate full terrain (500x500m @ 2048 resolution)
generator.generate_and_save_area(
    x_start_m=0.0,
    y_start_m=0.0,
    width_m=10.0,
    res=1024,
    output_path="Terrain_generation/output/10x10.png",
    apply_blur=True  # optional
)

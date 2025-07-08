from perlin_noise import PerlinNoise
import numpy as np
import os
import cv2
import time

class PerlinNoiseGenerator:
    def __init__(self, seed=0, tile_res=1024, tile_size_m=5.0):
        self.seed = seed
        self.tile_res = tile_res
        self.tile_size_m = tile_size_m
        self.m_per_pixel = tile_size_m / tile_res

        self.noise_macro = PerlinNoise(octaves=2, seed=self.seed)
        self.noise_meso = PerlinNoise(octaves=2, seed=self.seed + 10)

    def get_height(self, x_m, y_m):
        macro = self.noise_macro([x_m / 200.0, y_m / 200.0]) * 4.0
        meso = self.noise_meso([x_m / 60.0, y_m / 60.0]) * 2.5
        micro = self.noise_meso([x_m / .5, y_m / .5]) * 0.02
        height = (macro + meso) / 2.0
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
        np.save(path, heightmap)
        print(f"Saved raw float32 heightmap to: {path}.npy")
        print(f"Heightmap range: min={heightmap.min():.6f}, max={heightmap.max():.6f}")
        end = time.time()
        delta = end - self.start
        print(f"time taken: {delta}")

    def generate_and_save_tile(self, row, col, output_path, upscale_to=None):
        heightmap = self.generate_tile(row, col)
        self.save_heightmap(heightmap, output_path)

        if upscale_to:
            upscaled = cv2.resize(heightmap, (upscale_to, upscale_to), interpolation=cv2.INTER_CUBIC)
            upscale_path = output_path + f"_upscaled_{upscale_to}"
            self.save_heightmap(upscaled, upscale_path)

    def generate_and_save_area(self, x_start_m, y_start_m, width_m, res, output_path, apply_blur=False):
        self.start = time.time()
        heightmap = self.generate_area(x_start_m, y_start_m, width_m, res)
        if apply_blur:
            heightmap = cv2.GaussianBlur(heightmap, (3, 3), sigmaX=1)
        self.save_heightmap(heightmap, output_path)

# === Instantiate and Generate ===

generator = PerlinNoiseGenerator(
    seed=80,
    tile_res=707,
    tile_size_m=5.0
)

generator.generate_and_save_area(
    x_start_m=200, #in meters
    y_start_m=200,
    width_m=10.0,
    res=707,
    output_path="Terrain_generation/output/10x10macro1_nomicro"
)

generator.generate_and_save_area(
    x_start_m=0.0,
    y_start_m=0.0,
    width_m=500.0,
    res=707,
    output_path="Terrain_generation/output/full"
)

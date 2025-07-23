from perlin_noise import PerlinNoise
import numpy as np
import os
import time


'''
this class is in charge generating a .npy heightmap using PerlinNoise library
based on macro,meso parameters and terrain size.

'''
class PerlinNoiseGenerator:
    def __init__(self, seed=0, tile_res=1024, tile_size_m=5.0, macro_scale = 200, macro_height = 4, meso_scale = 60, meso_height = 2.5):
        self.seed = seed
        self.tile_res = tile_res
        self.tile_size_m = tile_size_m
        self.m_per_pixel = tile_size_m / tile_res
        self.macro_scale = macro_scale
        self.macro_height = macro_height
        self.meso_scale = meso_scale
        self.meso_height = meso_height
        self.noise_macro = PerlinNoise(octaves=2, seed=self.seed)
        self.noise_meso = PerlinNoise(octaves=2, seed=self.seed + 10)

    def get_height(self, x_m, y_m):
        macro = self.noise_macro([x_m / self.macro_scale, y_m / self.macro_scale]) * self.macro_height
        meso = self.noise_meso([x_m / self.meso_scale, y_m / self.meso_scale]) * self.meso_height
        return macro + meso 

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
        print(f"time taken: {time.time() - self.start:.2f} sec")

    def generate_and_save_area(self, x_start_m, y_start_m, width_m, res, output_path):
        print(f"starting noise generation...")
        self.start = time.time()
        heightmap = self.generate_area(x_start_m, y_start_m, width_m, res)
        self.save_heightmap(heightmap, output_path)





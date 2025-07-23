import numpy as np
import cv2
import os
from PIL import Image

'''
This class turns the .npy heightmap into a 3D .obj mesh using z_scale for height multiplier
'''
class TerrainMeshGenerator:
    def __init__(self, physical_size_m, hill_path_npy, obj_output_path, macro_z_scale, bump_path_npy, micro_z_scale, road_path, road_z_scale = 0):
        self.physical_size = physical_size_m
        self.hill_path = hill_path_npy
        self.obj_output_path = obj_output_path
        self.macro_z_scale = macro_z_scale
        self.bump_path = bump_path_npy
        self.micro_z_scale = micro_z_scale
        self.road_path = road_path
        self.road_z_scale = road_z_scale

    def process(self):
        self.generate_obj_from_heightmap(self.hill_path, self.obj_output_path, self.macro_z_scale)
        self.dirt = self.tile_texture(self.bump_path, (2048, 2048), self.physical_size/2)
        self.apply_micro_displacement(self.obj_output_path, self.obj_output_path, self.dirt, self.micro_z_scale)
       # self.apply_micro_displacement_png(self.obj_output_path, self.obj_output_path, self.road_path, self.road_z_scale)

    def generate_obj_from_heightmap(self, hill_path, obj_output_path, z_scale=1.0):
        height_data = np.load(hill_path)
        rows, cols = height_data.shape

        spacing_x = self.physical_size / (cols - 1)
        spacing_y = self.physical_size / (rows - 1)

        vertices = [(x * spacing_x, y * spacing_y, height_data[y, x] * z_scale)
                    for y in range(rows) for x in range(cols)]

        def idx(x, y): return y * cols + x + 1
        faces = [(idx(x, y), idx(x+1, y), idx(x, y+1))
                 for y in range(rows - 1) for x in range(cols - 1)]
        faces += [(idx(x+1, y), idx(x+1, y+1), idx(x, y+1))
                  for y in range(rows - 1) for x in range(cols - 1)]

        with open(obj_output_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print(f"[\u2713] Mesh saved to {obj_output_path}")

    def tile_texture(self, texture, target_shape, tile_multiplier=1.0):
        img = cv2.imread(texture, cv2.IMREAD_GRAYSCALE) / 255.0
        resized = cv2.resize(img, (int(img.shape[1] / tile_multiplier), int(img.shape[0] / tile_multiplier)))
        reps = (int(np.ceil(target_shape[0] / resized.shape[0])), int(np.ceil(target_shape[1] / resized.shape[1])))
        tiled = np.tile(resized, reps)
        return tiled[:target_shape[0], :target_shape[1]]

    def apply_micro_displacement(self, obj_path, output_path, heightmap, scale):
        h, w = heightmap.shape
        px = self.physical_size / (w - 1)
        py = self.physical_size / (h - 1)

        with open(obj_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("v "):
                x, y, z = map(float, line.strip().split()[1:])
                u = min(int(x / px), w - 1)
                v = min(int(y / py), h - 1)
                z += heightmap[v, u] * scale
                new_lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            else:
                new_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)

    def apply_micro_displacement_png(self, obj_path, output_path, img_path, scale):
        img = Image.open(img_path).convert("L")
        heightmap = np.array(img).astype(np.float32) / 255.0
        self.apply_micro_displacement(obj_path, output_path, heightmap, scale)
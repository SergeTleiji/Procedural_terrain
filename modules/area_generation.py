import numpy as np
import cv2
import os
from PIL import Image

class TerrainMeshGenerator:
    def __init__(self, physical_size_m, hill_path_npy, obj_output_path, macro_z_scale, bump_path_npy, micro_z_scale):
        self.physical_size = physical_size_m
        self.hill_path = hill_path_npy
        self.obj_output_path = obj_output_path
        self.macro_z_scale = macro_z_scale
        self.bump_path = bump_path_npy
        self.micro_z_scale = micro_z_scale

    def process(self):
        self.generate_obj_from_heightmap(self.hill_path, self.obj_output_path, self.macro_z_scale)
        self.dirt = self.tile_texture(self.bump_path, (2048, 2048), 10)
        self.apply_micro_displacement(self.obj_output_path, self.obj_output_path, self.dirt, self.micro_z_scale)

    def generate_obj_from_heightmap(self, hill_path, obj_output_path, z_scale=50.0):
        height_data = np.load(hill_path)  # expects float32 (0–1)
        rows, cols = height_data.shape

        spacing_x = self.physical_size / (cols - 1)
        spacing_y = self.physical_size / (rows - 1)

        vertices = []
        for y in range(rows):
            for x in range(cols):
                z = height_data[y, x] * z_scale
                vertices.append((x * spacing_x, y * spacing_y, z))

        faces = []
        def idx(x, y): return y * cols + x + 1
        for y in range(rows - 1):
            for x in range(cols - 1):
                a = idx(x, y)
                b = idx(x + 1, y)
                c = idx(x, y + 1)
                d = idx(x + 1, y + 1)
                faces.append((a, b, c))
                faces.append((b, d, c))

        with open(obj_output_path, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print(f"[✓] Mesh saved to {obj_output_path}")
        print(f"Spacing: {spacing_x:.4f} m — Final mesh size: {(cols - 1) * spacing_x:.2f} x {(rows - 1) * spacing_y:.2f} m")


    def tile_texture(self, texture, target_shape, tile_multiplier=1.0):
        # Load texture as grayscale
        self.img = cv2.imread(texture, cv2.IMREAD_GRAYSCALE) / 255.0

        # Resize texture to smaller tile
        new_h = int(self.img.shape[0] / tile_multiplier)
        new_w = int(self.img.shape[1] / tile_multiplier)
        resized = cv2.resize(self.img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute number of repetitions needed to reach the target size
        reps = (
            int(np.ceil(target_shape[0] / resized.shape[0])),
            int(np.ceil(target_shape[1] / resized.shape[1]))
        )

        # Tile the image (2D only)
        tiled = np.tile(resized, reps)
        cv2.imwrite("Terrain_generation/output/debug_tiled_micro.png", (tiled * 255).astype(np.uint8))

        # Crop to exact target shape
        return tiled[:target_shape[0], :target_shape[1]]

    def apply_micro_displacement(self, obj_input_path, obj_output_path, micro_heightmap_path, micro_z_scale=2.0):
        micro_data = np.array(self.dirt).astype(np.float32) / 255
        h, w = micro_data.shape

        # Micromap resolution vs real 5m world
        pixel_size_x = self.physical_size / (w -1)
        pixel_size_y = self.physical_size / (h -1)

        with open(obj_input_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith("v "):
                x, y, z = map(float, line.strip().split()[1:])

                u = min(int(x / pixel_size_x), w - 1)
                v = min(int(y / pixel_size_y), h - 1)

                z += micro_data[v, u] * micro_z_scale
                new_lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
            else:
                new_lines.append(line)

        with open(obj_output_path, 'w') as f:
            f.writelines(new_lines)

        print(f"[✓] Micro displacement applied and saved to: {obj_output_path}")
terrain_size = 10
macro_z_scale = 16.0
micro_z_scale = 10

hill_path_npy = "Terrain_generation/output/10x10macro1.npy"
bump_path_npy = "F:/IsaacSim/Terrain_generation/textures/dirt2/displacement.png"
mesh_output_path = "Terrain_generation/output/10x10test_nomicro.obj"

mesh = TerrainMeshGenerator(terrain_size, hill_path_npy, mesh_output_path, macro_z_scale, bump_path_npy, micro_z_scale)
mesh.process()

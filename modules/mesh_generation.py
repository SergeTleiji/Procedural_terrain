import numpy as np
from PIL import Image

class TerrainMeshGenerator:
    def __init__(self, physical_size_m, hill_path, obj_output_path, macro_z_scale, bump_path, micro_z_scale):
        self.physical_size = physical_size_m  # e.g. 5.0 for 5m
        self.hill_path = hill_path
        self.obj_output_path = obj_output_path
        self.macro_z_scale = macro_z_scale
        self.bump_path = bump_path
        self.micro_z_scale = micro_z_scale

    def process(self):
        self.generate_obj_from_heightmap(self.hill_path, self.obj_output_path, self.macro_z_scale)
        self.apply_micro_displacement(self.obj_output_path, self.obj_output_path, self.bump_path, self.micro_z_scale)

    def generate_obj_from_heightmap(self, hill_path, obj_output_path, z_scale=50.0):
        img = Image.open(hill_path).convert("L").resize((1024, 1024), Image.LANCZOS)
        height_data = np.array(img).astype(np.float32) / 255.0
        rows, cols = height_data.shape

        spacing_x = self.physical_size / (cols - 1)
        spacing_y = self.physical_size / (rows -1)

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



    def apply_micro_displacement(self, obj_input_path, obj_output_path, micro_heightmap_path, micro_z_scale=2.0):
        img = Image.open(micro_heightmap_path).convert("L").resize((1024, 1024), Image.LANCZOS)
        micro_data = np.array(img).astype(np.float32) / 255.0
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


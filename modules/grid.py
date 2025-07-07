import numpy as np
from PIL import Image

class TerrainMeshGenerator:
    def generate_obj_from_heightmap(self, hill_path, obj_output_path, z_scale=16.0):
        img = Image.open(hill_path).convert("L")#.resize((1024, 1024), Image.LANCZOS)
        height_data = np.array(img).astype(np.float32) / 255.0
        rows, cols = height_data.shape

        spacing_x = 500 / (cols - 1)  #the integer in spacing represents tile size (5 for 5x5m tile)
        spacing_y = 500 / (rows -1)

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

hill_path = "Terrain_generation/output/FULLTERRAIN.png"
obj_output_path = "Terrain_generation/output/FULLTERRAIN.obj"
generator = TerrainMeshGenerator()
generator.generate_obj_from_heightmap(hill_path, obj_output_path)

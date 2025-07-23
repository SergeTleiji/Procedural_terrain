import numpy as np
from scipy.ndimage import convolve
from PIL import Image

class SemanticMap:
    def __init__(self, probabilities, map_size=500, num_model_types=5, seed=42):
        self.map_size = map_size
        self.num_model_types = num_model_types
        self.seed = seed
        np.random.seed(self.seed)
        self.semantic_map = None
        self.model_id = None
        self.scale_class = None
        self.probabilities = probabilities


    def generate_value_noise(self, freq):
        size = self.map_size
        grid_size = size // freq + 2
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

                # Bilinear interpolation
                v00 = grid[y0, x0]
                v10 = grid[y0, x0 + 1]
                v01 = grid[y0 + 1, x0]
                v11 = grid[y0 + 1, x0 + 1]
                a = (1 - tx) * v00 + tx * v10
                b = (1 - tx) * v01 + tx * v11
                result[y, x] = (1 - ty) * a + ty * b

        return result

    def build(self):
        # === Generate and combine noise ===
        base_noise = self.generate_value_noise(freq=20)
        detail_noise = self.generate_value_noise(freq=5) * 0.4
        density_noise = base_noise + detail_noise
        density_noise -= density_noise.min()
        density_noise /= density_noise.max()

        # === Assign model IDs with custom probabilities ===
        bins = np.cumsum([0.0] + self.probabilities)
        self.model_id = np.digitize(density_noise, bins=bins) - 1


        # === Compute scale class based on local density ===
        self.scale_class = np.zeros_like(self.model_id)
        kernel = np.ones((5, 5), dtype=np.uint8)

        for i in range(self.num_model_types):
            mask = (self.model_id == i).astype(np.uint8)
            local_density = convolve(mask, kernel, mode='constant')
            self.scale_class[self.model_id == i] = np.digitize(local_density[self.model_id == i], bins=[2, 6, 10])  # Scale class: 0–3

        # === Assemble semantic map (channels: model_id, scale_class) ===
        self.semantic_map = np.zeros((self.map_size, self.map_size, 2), dtype=np.uint8)
        self.semantic_map[:, :, 0] = self.model_id
        self.semantic_map[:, :, 1] = self.scale_class

    def save(self, path_npy, path_png):
        if self.semantic_map is None:
            raise ValueError("Semantic map not generated. Call build() first.")

        np.save(path_npy, self.semantic_map)
        print(f"✅ Saved: {path_npy}")

        # Visual image: red = model_id, green = scale class
        visual_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        visual_img[:, :, 0] = (self.model_id * 50) % 255
        visual_img[:, :, 1] = (self.scale_class * 85) % 255
        Image.fromarray(visual_img).save(path_png)
        print(f"✅ Saved: {path_png}")

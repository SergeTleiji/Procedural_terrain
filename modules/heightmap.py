import numpy as np
import cv2

class HeightmapGenerator:
    def __init__(self, width, height, base_scale=64, octaves=4, persistence=0.5):
        self.width = width
        self.height = height
        self.base_scale = base_scale
        self.octaves = octaves
        self.persistence = persistence
        self.heightmap = None
        
    def Process(self, terrain_path):
        self.generate_fractal_noise()
        self.save_normalized(terrain_path)

    def generate_smooth_noise(self, scale):
        base = np.random.rand(self.height // scale + 1, self.width // scale + 1)
        return cv2.resize(base, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def generate_fractal_noise(self):
        heightmap = np.zeros((self.height, self.width), dtype=np.float32)
        amplitude = 1.0
        total_amplitude = 0.0

        for i in range(self.octaves):
            scale = max(1, int(self.base_scale / (2 ** i)))  # Prevent scale=0
            layer = self.generate_smooth_noise(scale)
            heightmap += layer * amplitude
            total_amplitude += amplitude
            amplitude *= self.persistence

        heightmap /= total_amplitude
        self.heightmap = heightmap

    def save_normalized(self, filepath, min_gray=30):
        if self.heightmap is None:
            raise ValueError("Heightmap not generated yet. Call generate_fractal_noise() first.")
        
        # Normalize to full 0-255 range first
        normalized = cv2.normalize(self.heightmap, None, 0, 255, cv2.NORM_MINMAX)

        # Apply minimum gray level threshold
        if min_gray > 0:
            normalized = normalized * ((255 - min_gray) / 255.0) + min_gray

        # Convert to 8-bit
        final = np.clip(normalized, 0, 255).astype(np.uint8)
        cv2.imwrite(filepath, final)



#How to Call the class correspondingly
#generator = HeightmapGenerator(width=1024, height=1024, base_scale=64, octaves=4, persistence=0.6)
#generator.generate_fractal_noise()
#generator.save_normalized(r"F:\work\heightmap_testing\heightmap_bw.png")

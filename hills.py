import numpy as np
import cv2

class HillGenerator:
    def __init__(self, width, height, base_scale=128, octaves=3, persistence=0.55):
        self.width = width
        self.height = height
        self.base_scale = base_scale
        self.octaves = octaves
        self.persistence = persistence
        self.heightmap = None
        self.seed = np.random.randint(0, 10000)
        
    def Process(self, terrain_path):
        self.generate_perlin_noise()
        self.save_normalized(terrain_path)

    def perlin(self, x, y, seed):
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()

        def fade(t): return 6*t**5 - 15*t**4 + 10*t**3
        def lerp(a, b, t): return a + t * (b - a)
        
        def grad(h, x, y):
            vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
            g = vectors[h % 4]  # (N, 2) shape where N = x.size
            return g[...,0]*x + g[...,1]*y

        xi = x.astype(int) % 256
        yi = y.astype(int) % 256
        xf = x - xi
        yf = y - yi
        u = fade(xf)
        v = fade(yf)

        # Hash coordinates to get gradient indices
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        n00 = grad(aa, xf, yf)
        n01 = grad(ab, xf, yf - 1)
        n10 = grad(ba, xf - 1, yf)
        n11 = grad(bb, xf - 1, yf - 1)

        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)
        return lerp(x1, x2, v)

    
    def generate_perlin_noise(self):
        lin = np.linspace(0, 5, self.width, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        self.heightmap = self.perlin(x, y, self.seed)
        self.heightmap = (self.heightmap - self.heightmap.min()) / (self.heightmap.max() - self.heightmap.min())


    def save_normalized(self, filepath):
        if self.heightmap is None:
            raise ValueError("Heightmap not generated yet. Call generate_fractal_noise() first.")
        normalized = cv2.normalize(self.heightmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(filepath, normalized)

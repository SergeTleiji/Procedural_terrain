import cv2
import numpy as np
import os

class TextureBlender:
    def __init__(self, mask_path, grass_path, dirt_path,
                 grass_height=None, dirt_height=None,
                 grass_normal=None, dirt_normal=None,
                 grass_metal=None, dirt_metal=None,
                 grass_rough=None, dirt_rough=None):
        
        self.mask_path = mask_path
        self.grass_path = grass_path
        self.dirt_path = dirt_path

        self.grass_height = grass_height
        self.dirt_height = dirt_height
        self.grass_normal = grass_normal
        self.dirt_normal = dirt_normal
        self.grass_metal = grass_metal
        self.dirt_metal = dirt_metal
        self.grass_rough = grass_rough
        self.dirt_rough = dirt_rough

    def Process(self, output_base_path):
        self.load_images()
        h, w = self.mask.shape
        tile_multiplier = 5  # Increase this to get smaller, more frequent tiles

        # Tile color
        self.grass = self.tile_texture(self.grass, (h, w), tile_multiplier)
        self.dirt = self.tile_texture(self.dirt, (h, w), tile_multiplier)

        # Tile optional maps
        if self.grass_height is not None:
            self.grass_height = self.tile_texture(self.grass_height, (h, w), tile_multiplier)
            self.dirt_height = self.tile_texture(self.dirt_height, (h, w), tile_multiplier)
        if self.grass_normal is not None:
            self.grass_normal = self.tile_texture(self.grass_normal, (h, w), tile_multiplier)
            self.dirt_normal = self.tile_texture(self.dirt_normal, (h, w), tile_multiplier)
        if self.grass_metal is not None:
            self.grass_metal = self.tile_texture(self.grass_metal, (h, w), tile_multiplier)
            self.dirt_metal = self.tile_texture(self.dirt_metal, (h, w), tile_multiplier)
        if self.grass_rough is not None:
            self.grass_rough = self.tile_texture(self.grass_rough, (h, w), tile_multiplier)
            self.dirt_rough = self.tile_texture(self.dirt_rough, (h, w), tile_multiplier)

        self.blend_all(output_base_path)

    def load_images(self):
        self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        threshold = 0.5
        softness = 0.2  # controls gradient softness
        self.mask = np.clip((self.mask - (threshold - softness)) / (2 * softness), 0, 1)

        self.grass = cv2.imread(self.grass_path).astype(np.float32)
        self.dirt = cv2.imread(self.dirt_path).astype(np.float32)

        # Load optional maps if paths are provided
        def read_optional(path):
            return cv2.imread(path).astype(np.float32) if path else None

        self.grass_height = read_optional(self.grass_height)
        self.dirt_height = read_optional(self.dirt_height)
        self.grass_normal = read_optional(self.grass_normal)
        self.dirt_normal = read_optional(self.dirt_normal)
        self.grass_metal = read_optional(self.grass_metal)
        self.dirt_metal = read_optional(self.dirt_metal)
        self.grass_rough = read_optional(self.grass_rough)
        self.dirt_rough = read_optional(self.dirt_rough)

    def tile_texture(self, texture, target_shape, tile_multiplier=1.0):
        # Increase tile_multiplier to repeat the texture more frequently
        new_h = int(texture.shape[0] / tile_multiplier)
        new_w = int(texture.shape[1] / tile_multiplier)
        resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        reps = (
            int(np.ceil(target_shape[0] / resized.shape[0])),
            int(np.ceil(target_shape[1] / resized.shape[1])),
        )
        tiled = np.tile(resized, (reps[0], reps[1], 1))
        return tiled[:target_shape[0], :target_shape[1]]


    def blend_map(self, tex1, tex2):
        return np.clip(tex1 * self.mask[..., None] + tex2 * (1 - self.mask[..., None]), 0, 255).astype(np.uint8)

    def blend_all(self, output_base_path):
        # Base color
        blended = self.blend_map(self.grass, self.dirt)
        cv2.imwrite(os.path.join(output_base_path, "blended_basecolor.png"), blended)

        # Optional PBR maps
        if self.grass_height is not None:
            blended_height = self.blend_map(self.grass_height, self.dirt_height)
            cv2.imwrite(os.path.join(output_base_path, "blended_height.png"), blended_height)
        if self.grass_normal is not None:
            blended_normal = self.blend_map(self.grass_normal, self.dirt_normal)
            cv2.imwrite(os.path.join(output_base_path, "blended_normal.png"), blended_normal)
        if self.grass_metal is not None:
            blended_metal = self.blend_map(self.grass_metal, self.dirt_metal)
            cv2.imwrite(os.path.join(output_base_path, "blended_metalness.png"), blended_metal)
        if self.grass_rough is not None:
            blended_rough = self.blend_map(self.grass_rough, self.dirt_rough)
            cv2.imwrite(os.path.join(output_base_path, "blended_roughness.png"), blended_rough)


mask_path = "Terrain_generation/output/FULLTERRAIN.png"
grass_path = "Terrain_generation/textures/grass/albedo.png"
grass_height = "Terrain_generation/textures/grass/height.png"
grass_metal = "Terrain_generation/textures/grass/metal.png"
grass_normal = "Terrain_generation/textures/grass/normal.png"
grass_rough = "Terrain_generation/textures/grass/rough.png"
dirt_path = "Terrain_generation/textures/dirt/albedo.png"
dirt_height = "Terrain_generation/textures/dirt/height.png"
dirt_metal = "Terrain_generation/textures/dirt/metal.png"
dirt_normal = "Terrain_generation/textures/dirt/normal.png"
dirt_rough = "Terrain_generation/textures/dirt/rough.png"
output_texture_path = "Terrain_generation/output"
# === Blend Terrain Texture ===
blender = TextureBlender(
    mask_path, grass_path, dirt_path, grass_height, dirt_height, grass_normal, dirt_normal, grass_metal, dirt_metal, grass_rough, dirt_rough
    )
blender.Process(output_texture_path)
print("[✔] Texture blending complete.")
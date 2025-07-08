from PIL import Image, ImageFilter
import numpy as np
import os

class RoadMerger:
    def __init__(self, road_path, terrain_path, blend_factor=0.4):
        self.road_path = road_path
        self.terrain_path = terrain_path
        self.blend_factor = blend_factor
        self.road_img = None
        self.terrain_img = None
        self.merged_array = None
        self.merged_img = None
        self.blurred_img = None
#        self.hill_img = None
#        self.hill_path = hill_path
#        self.merged_hill_array = None
#        self.merged_hill = None
#        self.blurred_hill = None

    def Process(self, radius, mask_path):
        self.load_images()
        self.apply_road_mask(mask_path)
        self.apply_blur(radius)
        self.save_blurred(mask_path)


    def load_images(self):
        self.road_img = Image.open(self.road_path).convert("LA")
        self.terrain_img = Image.open(self.terrain_path).convert("L")
#        self.hill_img = Image.open(self.hill_path).convert("L")

    def apply_road_mask(self, output_path):
        road_array = np.array(self.road_img)
        alpha_channel = road_array[:, :, 1]
        road_mask = alpha_channel > 0

        terrain_array = np.array(self.terrain_img)
#        hill_array = np.array(self.hill_img)
        self.merged_array = terrain_array.copy()
        self.merged_array[road_mask] = terrain_array[road_mask] * self.blend_factor
        self.merged_img = Image.fromarray(self.merged_array)
#        self.merged_img.save(output_path)
#        self.merged_hill_array = hill_array.copy()
#        self.merged_hill_array[road_mask] = hill_array[road_mask] * self.blend_factor
#        self.merged_hill = Image.fromarray(self.merged_hill_array)

    def apply_blur(self, radius):
        if self.merged_img is None:
            raise ValueError("No merged image to blur. Call apply_road_mask() first.")
        self.blurred_img = self.merged_img.filter(ImageFilter.GaussianBlur(radius=radius))
#        if self.merged_hill is None:
#            raise ValueError("No merged image to blur. Call apply_road_mask() first.")
#        self.blurred_hill = self.merged_hill.filter(ImageFilter.GaussianBlur(radius=radius))

    def save_blurred(self, output_path):
        if self.blurred_img is None:
            raise ValueError("No blurred image to save. Call apply_blur() first.")
        self.blurred_img.save(output_path)
        print(f"Saved blurred grass to: {output_path}")

#        if self.blurred_hill is None:
#            raise ValueError("No blurred image to save. Call apply_blur() first.")
#        self.blurred_hill.save(output_path)
#        print(f"Saved blurred grass to: {output_hill}")




#How to Call the class correspondingly
#merger = RoadMerger(road_path, terrain_path, blend_factor=0.4)
#merger.load_images()
#merger.apply_road_mask()
#merger.save_merged(r"F:\Work\heightmap_testing\merged.png")
#merger.apply_blur(radius=7.5)
#merger.save_blurred(r"F:\Work\heightmap_testing\merged_blurred.png")
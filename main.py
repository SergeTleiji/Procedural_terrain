from modules.heightmap import HeightmapGenerator
from modules.heightmap_merging import RoadMerger
from modules.Texture_generation import TextureBlending
from modules.hills import HillGenerator
from modules.mesh_generation import TerrainMeshGenerator
import os

def main():

    # === heightmap (macro) Configuration ===
    hill_path = "Terrain_generation/output/hill.png"
    hill_base_scale = 16
    hill_octaves = 4
    hill_persistence = 0.5
    
    # === heightmap (micro) Configuration ===
    width = 2048
    height = 2048
    base_scale = 4
    octaves = 5
    persistence = 0.45

    # === merging Configuration ===
    road_path = r"Terrain_generation/output/roadMap.png"
    bump_path = r"Terrain_generation/output/heightmap_bw.png"
    blend_factor = 0.4
    radius = 3

    # === Blending Configuration ===
    mask_path = "Terrain_generation/output/merged_blurred.png"
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

    # === Mesh Configuration ===
    mesh_output_path = "Terrain_generation/output/mesh.obj"
    terrain_size = 5.0
    macro_z_scale = 0.3
    micro_z_scale = 0.025
    # === generate heightmap ===
    generator = HeightmapGenerator(width, height, base_scale, octaves, persistence)
    generator.Process(bump_path)
    print("[✔] Heightmap generation complete.")

    # === generate hill ===
    hill = HillGenerator(width, height, hill_base_scale, hill_octaves, hill_persistence)
    hill.Process(hill_path)
    print("[✔] Heightmap generation complete.")
    

    # === Merge road and heightmap ===
    merger = RoadMerger(road_path, bump_path, blend_factor)
    merger.Process(radius, mask_path)
    print("[✔] Merging complete.")

    # === Blend Terrain Texture ===
    blender = TextureBlending(mask_path, grass_path, dirt_path, grass_height, dirt_height, grass_normal, dirt_normal, grass_metal, dirt_metal, grass_rough, dirt_rough)
    blender.Process(output_texture_path)
    print("[✔] Texture blending complete.")

        # === Mesh Generation ===
    mesh = TerrainMeshGenerator(terrain_size, hill_path, mesh_output_path, macro_z_scale, mask_path, micro_z_scale)
    mesh.process()
    print("[✔] Mesh generation complete.")


if __name__ == "__main__":
    main()

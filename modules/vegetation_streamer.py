"""
vegetation_streamer.py
----------------------
Manages runtime loading and unloading of vegetation tiles in Isaac Sim.

Purpose in Pipeline:
    - Step 11 in `main.py`: works alongside `terrain_streamer.py` to:
        1. Determine which vegetation instancers (grass + trees) need to be active.
        2. Spawn vegetation for newly loaded terrain tiles.
        3. Remove vegetation instancers for tiles outside the visible radius.

Key Differences from TerrainStreamer:
    - Handles *vegetation* only, not terrain meshes.
    - Spawns vegetation using the `Instancer` class.
    - Deletes PointInstancer prims when tiles are unloaded.

Workflow:
    1. Track robot position and determine which vegetation tiles should be active.
    2. When the robot enters a new tile:
        - Unload far vegetation instancers.
        - Spawn instancers for any new tiles in range.
    3. Vegetation spawning uses grass points (RandomScatterClass) and tree points (PoissonClass).

Inputs:
    - poisson_xy (dict): Tree placement points from PoissonClass.
    - points (dict): Grass placement points from RandomScatterClass.
    - grass_models (list): USD paths for grass model prototypes.
    - tile_size (float): Size of each tile in meters.
    - view_radius (int): Number of tiles around robot to keep vegetation loaded.
    - hdri_models (list): HDRI textures for lighting.
    - seed (int): Random seed for reproducibility.
    - num_instances (int): Max number of instances per tile.
    - npy_path (str): Path to heightmap .npy file.
    - usd_path (str): Path to base terrain USD file.
    - z_scale (float): Height scaling factor.
    - tree_models (list): USD paths for tree model prototypes.
    - tree_weights (list): Probabilities for tree model selection.
    - semantic_map_path (np.ndarray): Semantic vegetation map (model ID, scale class).
    - output_dir (str): Directory for runtime-generated files.
    - terrain_size (float): Global terrain size in meters.

Outputs:
    - Dynamically updates the USD stage with vegetation instancers.
    - Removes vegetation from tiles that are unloaded.

Dependencies:
    - isaacsim.core.utils.prims
    - modules.instancing.Instancer
    - Called by `main.py` → `vegetation_streamer.update()`.

Example:
    veg_streamer = VegetationStreamer(...config...)
    new_tiles = veg_streamer.update(robot_position)
"""

from math import floor
import os
import isaacsim.core.utils.prims as prims_utils
from modules.instancing import Instancer


class VegetationStreamer:
    """
    Streams vegetation instancers (grass + trees) into and out of the scene based on robot position.
    """

    def __init__(
        self,
        poisson_xy,
        points,
        grass_models,
        tile_size,
        view_radius,
        hdri_models,
        seed,
        num_instances,
        npy_path,
        usd_path,
        z_scale,
        tree_models,
        tree_weights,
        semantic_map_path,
        output_dir,
        terrain_size,
    ):
        self.poisson_xy = poisson_xy
        self.points = points
        self.grass_models = grass_models
        self.tile_size = tile_size
        self.view_radius = view_radius
        self.hdri_models = hdri_models
        self.seed = seed
        self.num_instances = num_instances
        self.npy_path = npy_path
        self.usd_path = usd_path
        self.z_scale = z_scale
        self.tree_models = tree_models
        self.tree_weights = tree_weights
        self.semantic_map_path = semantic_map_path
        self.output_dir = output_dir
        self.terrain_size = terrain_size

        self.loaded_tiles = set()
        self.prev_tile_coord = None

    def get_tile_coord(self, x, y):
        """
        Converts world coordinates to tile coordinates.
        """
        return (int(floor(x / self.tile_size)), int(floor(y / self.tile_size)))

    def get_required_tiles(self, center_coord):
        """
        Gets all vegetation tiles within the view radius.
        """
        cx, cy = center_coord
        return {
            (cx + dx, cy + dy)
            for dx in range(-self.view_radius, self.view_radius + 1)
            for dy in range(-self.view_radius, self.view_radius + 1)
        }

    def unload_far_tiles(self, required_tiles):
        """
        Removes vegetation instancers for tiles outside the required set.
        """
        tiles_to_remove = self.loaded_tiles - required_tiles
        for tile in tiles_to_remove:
            x, y = tile
            self.name_x = (
                f"N{abs(x * self.tile_size)}" if x < 0 else f"{x * self.tile_size}"
            )
            self.name_y = (
                f"N{abs(y * self.tile_size)}" if y < 0 else f"{y * self.tile_size}"
            )

            terrain_name = f"A{int(self.tile_size)}x{int(self.tile_size)}_{self.name_x}x{self.name_y}"
            self.obj_path = os.path.join(self.output_dir, terrain_name + ".obj")
            self.usd_path = os.path.join(self.output_dir, terrain_name + ".usd")
            self.prim_path = f"/World/{terrain_name}/mesh"

            self.remove_tile()
            self.loaded_tiles.remove(tile)

    def remove_tile(self):
        """
        Deletes the grass and tree PointInstancers for the current tile.
        """
        instancer_path = f"/World/Instancer_Tree_{self.name_x}x{self.name_y}"
        prims_utils.delete_prim(instancer_path)
        # For future expansion: handle more than two instancer types.

    def update(self, robot_pos):
        """
        Updates vegetation based on robot position.
        Spawns new vegetation instancers and unloads far ones.
        """
        robot_x, robot_y = robot_pos
        current_tile = self.get_tile_coord(robot_x, robot_y)

        if current_tile != self.prev_tile_coord:
            self.prev_tile_coord = current_tile
            required_tiles = self.get_required_tiles(current_tile)

            # Unload vegetation for far tiles
            self.unload_far_tiles(required_tiles)

            # Spawn vegetation for new tiles
            new_tiles = required_tiles - self.loaded_tiles
            self.loaded_tiles.update(new_tiles)
            for tile in new_tiles:
                x, y = tile
                self.name_x = (
                    f"N{abs(x * self.tile_size)}" if x < 0 else f"{x * self.tile_size}"
                )
                self.name_y = (
                    f"N{abs(y * self.tile_size)}" if y < 0 else f"{y * self.tile_size}"
                )
                terrain_name = f"A{int(self.tile_size)}x{int(self.tile_size)}_{self.name_x}x{self.name_y}"
                self.obj_path = os.path.join(self.output_dir, terrain_name + ".obj")
                self.usd_path = os.path.join(self.output_dir, terrain_name + ".usd")
                self.prim_path = f"/World/{terrain_name}/mesh"
                self.spawn_tile(*tile)

        return []

    def spawn_tile(self, tile_x, tile_y):
        """
        Creates an Instancer for vegetation in the given tile.
        """
        world_x = tile_x * self.tile_size
        world_y = tile_y * self.tile_size

        instancer = Instancer(
            hdri=self.hdri_models,
            seed=self.seed,
            world_x=world_x,
            world_y=world_y,
            N_grass=self.points,
            Poisson=self.poisson_xy,
            heightmap_path=self.npy_path,
            terrain_usd_path=self.usd_path,
            terrain_size_m=self.tile_size,
            z_scale=self.z_scale,
            grass_model_usd_paths=self.grass_models,
            tree_model_usd_paths=self.tree_models,
            num_instances=self.num_instances,
            tree_weights=self.tree_weights,
            semantic_map=self.semantic_map_path,
            global_size=self.terrain_size,
        )
        instancer.scatter_instances()

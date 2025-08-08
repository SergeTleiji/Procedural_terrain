"""
terrain_streamer.py
-------------------
Manages runtime loading and unloading of terrain tiles in Isaac Sim.

Purpose in Pipeline:
    - Step 10 in `main.py`: used during simulation runtime to:
        1. Determine which tiles need to be generated based on robot position.
        2. Unload tiles outside the visible radius to free resources.
        3. Switch vegetation LOD for tiles outside close view range.

Workflow:
    1. Track the robot's current tile coordinates.
    2. When the robot enters a new tile:
        - Compute the set of required tiles within `view_radius`.
        - Unload tiles no longer in range.
        - Return the list of new tiles that need to be generated.
    3. For tiles leaving the close view range:
        - Switch vegetation models to a lower LOD (LOD1 by default).
        - Delete the terrain prim from the stage.

Inputs:
    - grass_models (list): List of grass USD paths (used to switch LOD).
    - tile_size (float): Tile size in meters.
    - view_radius (int): Number of tiles around the robot to keep loaded.

Outputs:
    - list of (tile_x, tile_y) coordinates for tiles that need to be generated.

Dependencies:
    - omni.usd
    - isaacsim.core.utils.prims
    - Called by `main.py` → `terrain_streamer.update()`.

Example:
    terrain_streamer = TerrainStreamer(grass_models, tile_size=10, view_radius=1)
    new_tiles = terrain_streamer.update(robot_position)
    for tile in new_tiles:
        generate_tile(tile)
"""

from math import floor
import omni.usd
import isaacsim.core.utils.prims as prims_utils


class TerrainStreamer:
    """
    Handles dynamic loading/unloading of terrain tiles based on robot position.
    """

    def __init__(self, grass_models, tile_size=5.0, view_radius=1):
        self.grass_models = grass_models
        self.tile_size = tile_size
        self.view_radius = view_radius

        self.loaded_tiles = set()
        self.prev_tile_coord = None
        self.stage = omni.usd.get_context().get_stage()

    def get_tile_coord(self, x, y):
        """
        Converts world coordinates to tile coordinates.

        Args:
            x, y (float): World coordinates.

        Returns:
            tuple[int, int]: Tile coordinates (tile_x, tile_y).
        """
        return (int(floor(x / self.tile_size)), int(floor(y / self.tile_size)))

    def get_required_tiles(self, center_coord):
        """
        Gets all tile coordinates within view radius.

        Args:
            center_coord (tuple[int, int]): Center tile coordinates.

        Returns:
            set[tuple[int, int]]: Set of required tile coordinates.
        """
        cx, cy = center_coord
        return {
            (cx + dx, cy + dy)
            for dx in range(-self.view_radius, self.view_radius + 1)
            for dy in range(-self.view_radius, self.view_radius + 1)
        }

    def unload_far_tiles(self, required_tiles):
        """
        Removes tiles outside the required tile set.

        Args:
            required_tiles (set): Tiles that should remain loaded.
        """
        tiles_to_remove = self.loaded_tiles - required_tiles
        for tile in tiles_to_remove:
            self.remove_tile(tile)
            self.loaded_tiles.remove(tile)

    def remove_tile(self, tile_coord):
        """
        Removes a terrain tile from the USD stage and switches its vegetation LOD.

        Args:
            tile_coord (tuple[int, int]): Tile coordinates to remove.
        """
        x, y = tile_coord
        name_x = f"N{abs(x * self.tile_size)}" if x < 0 else f"{x * self.tile_size}"
        name_y = f"N{abs(y * self.tile_size)}" if y < 0 else f"{y * self.tile_size}"
        instancer_path = f"/World/Instancer_Tree_{name_x}x{name_y}"
        tile_path = f"/World/A{self.tile_size}x{self.tile_size}_{name_x}x{name_y}"

        # Switch grass models to LOD1
        for i, _ in enumerate(self.grass_models):
            proto_path = instancer_path + f"/Proto_{i}"
            prim = self.stage.GetPrimAtPath(f"{proto_path}/Grass_LOD")
            if prim.IsValid():
                variant_sets = prim.GetVariantSets()
                if "LODVariant" in variant_sets.GetNames():
                    variant_sets.GetVariantSet("LODVariant").SetVariantSelection("LOD1")

        # Delete the tile prim
        prims_utils.delete_prim(tile_path)

    def update(self, robot_pos):
        """
        Checks if new tiles need to be generated/unloaded based on robot movement.

        Args:
            robot_pos (tuple[float, float]): Robot's current (x, y) world coordinates.

        Returns:
            list[tuple[int, int]]: New tiles to generate.
        """
        robot_x, robot_y = robot_pos
        current_tile = self.get_tile_coord(robot_x, robot_y)

        # Only trigger updates when robot enters a new tile
        if current_tile != self.prev_tile_coord:
            self.prev_tile_coord = current_tile
            required_tiles = self.get_required_tiles(current_tile)

            # Unload tiles that are too far
            self.unload_far_tiles(required_tiles)

            # Identify new tiles to load
            new_tiles = required_tiles - self.loaded_tiles
            self.loaded_tiles.update(new_tiles)

            return list(new_tiles)

        return []

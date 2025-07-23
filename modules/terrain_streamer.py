import numpy as np
from math import floor
from pxr import Usd, Sdf
import omni.usd

class TerrainStreamer:
    def __init__(self, tile_size=5.0, view_radius=1):
        self.tile_size = tile_size
        self.view_radius = view_radius

        self.loaded_tiles = set()
        self.prev_tile_coord = None

    def get_tile_coord(self, x, y):
        return (int(floor(x / self.tile_size)), int(floor(y / self.tile_size)))

    def get_required_tiles(self, center_coord):
        cx, cy = center_coord
        return {
            (cx + dx, cy + dy)
            for dx in range(-self.view_radius, self.view_radius + 1)
            for dy in range(-self.view_radius, self.view_radius + 1)
        }
    def unload_far_tiles(self, required_tiles):
        tiles_to_remove = self.loaded_tiles - required_tiles
        for tile in tiles_to_remove:
            self.remove_tile(tile)
            self.loaded_tiles.remove(tile)
    

    def remove_tile(self, tile_coord):
        x, y = tile_coord
        name_x = f"N{abs(x*self.tile_size)}" if x < 0 else f"{x*self.tile_size}"
        name_y = f"N{abs(y*self.tile_size)}" if y < 0 else f"{y*self.tile_size}"
        
        # This time, remove the entire tile, not just the mesh
        tile_path = f"/World/Tiles/A{self.tile_size}x{self.tile_size}_{name_x}x{name_y}"

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(tile_path)
        if prim.IsValid():
            stage.RemovePrim(Sdf.Path(tile_path))



    def update(self, robot_pos):
        robot_x, robot_y = robot_pos
        current_tile = self.get_tile_coord(robot_x, robot_y)

        # Only trigger updates when tile changes
        if current_tile != self.prev_tile_coord:
            self.prev_tile_coord = current_tile
            required_tiles = self.get_required_tiles(current_tile)

            # Unload far tiles before adding new ones
            self.unload_far_tiles(required_tiles)

            # Determine new tiles to load
            new_tiles = required_tiles - self.loaded_tiles
            self.loaded_tiles.update(new_tiles)

            return list(new_tiles)

        return []


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom, Sdf, Gf, Usd
import isaacsim.core.utils.stage as stage_utils

import numpy as np
import random

# Open or create stage
stage = omni.usd.get_context().get_stage()
if not stage:
    stage = omni.usd.get_context().new_stage()
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

# PointInstancer setup
instancer_path = "/World/TreeInstancer"
instancer = UsdGeom.PointInstancer.Define(stage, Sdf.Path(instancer_path))

tree_usd_path = "F:/work/models/trees/tree.usd"
proto_prim_path = instancer_path + "/TreeProto_0"
proto_prim = stage.DefinePrim(proto_prim_path, "Xform")
proto_prim.GetReferences().AddReference(tree_usd_path)
instancer.CreatePrototypesRel().SetTargets([proto_prim.GetPath()])

# Add terrain mesh to stage
terrain_usd_path = "Terrain_generation/output/full.obj"
terrain_prim_path = "/World/Terrain"
stage_utils.add_reference_to_stage(usd_path=terrain_usd_path, prim_path=terrain_prim_path)

# === Terrain Sampler with Normals ===
class TerrainSampler:
    def __init__(self, heightmap_path, terrain_size_m, z_scale=1.0):
        self.heightmap = np.load(heightmap_path)
        self.size_m = terrain_size_m
        self.z_scale = z_scale
        self.res = self.heightmap.shape[0]
        self.m_per_pixel = self.size_m / self.res

    def get_height(self, x_m, y_m):
        x_m = np.clip(x_m, 0, self.size_m)
        y_m = np.clip(y_m, 0, self.size_m)
        i = int(y_m / self.m_per_pixel)
        j = int(x_m / self.m_per_pixel)
        i = np.clip(i, 0, self.res - 1)
        j = np.clip(j, 0, self.res - 1)
        return self.heightmap[i, j] * self.z_scale

    def get_normal(self, x_m, y_m):
        x_m = np.clip(x_m, 0, self.size_m)
        y_m = np.clip(y_m, 0, self.size_m)
        i = int(y_m / self.m_per_pixel)
        j = int(x_m / self.m_per_pixel)
        i = np.clip(i, 1, self.res - 2)
        j = np.clip(j, 1, self.res - 2)

        dzdx = (self.heightmap[i, j + 1] - self.heightmap[i, j - 5]) / (2 * self.m_per_pixel)
        dzdy = (self.heightmap[i + 1, j] - self.heightmap[i - 5, j]) / (2 * self.m_per_pixel)
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal /= np.linalg.norm(normal)
        return Gf.Vec3f(*normal)

    def get_orientation_quat(self, x_m, y_m):
        normal = self.get_normal(x_m, y_m)
        up = Gf.Vec3f(0, 0, 1)
        axis = up ^ normal  # Cross product
        dot = up * normal   # Dot product
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        angle = angle ** 0.7

        if angle > 0.01:
            print(f"angle: {np.degrees(angle):.2f}°, axis: ({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})")

        if axis.GetLength() < 1e-4:
            return Gf.Quath(1.0, Gf.Vec3h(0, 0, 0))  # No rotation needed
        axis.Normalize()
        return Gf.Quath(np.cos(angle / 2), Gf.Vec3h(axis[0], axis[1], axis[2]) * np.sin(angle / 2))

# === Scatter Trees ===
N = 1000
positions = []
orientations = []
scales = []

sampler = TerrainSampler("Terrain_generation/output/full.npy", terrain_size_m=500.0, z_scale=16.0)

for _ in range(N):
    x = random.uniform(0, 500)
    y = random.uniform(0, 500)
    scale = random.uniform(0.5, 1.5)
    z = sampler.get_height(x, y)
    z -= 0.1 #dig it further into ground 
    positions.append(Gf.Vec3f(x, y, z))

    normal = sampler.get_normal(x, y)
    quat = sampler.get_orientation_quat(x, y)

    # Add random twist around the trunk
    angle_z = random.uniform(0, 2 * np.pi)
    spin = Gf.Quath(
        float(np.cos(angle_z / 2)),
        Gf.Vec3h(normal[0], normal[1], normal[2]) * float(np.sin(angle_z / 2))
    )

    final_quat = spin * quat
    orientations.append(final_quat)

    scales.append(Gf.Vec3h(scale, scale, scale))


# Apply to PointInstancer
instancer.CreatePositionsAttr().Set(positions)
instancer.CreateOrientationsAttr().Set(orientations)
instancer.CreateScalesAttr().Set(scales)
instancer.CreateProtoIndicesAttr().Set([0] * N)

print(f"{N} tree instances scattered with terrain-aligned orientation.")

# Run simulation
while simulation_app.is_running():
    simulation_app.update()

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from pxr import Sdf, UsdLux, UsdGeom, Gf
import random  # type: ignore

# Updated imports based on deprecation warnings
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prims_utils

#grid drawing
from isaacsim.core.api.objects import VisualCuboid

#lidar
import omni.kit.commands
import omni.replicator.core as rep


# --- Terrain Import ---
terrain_usd_path = "Terrain_generation/output/full.obj"
terrain_prim_path = "/World/Terrain"
stage_utils.add_reference_to_stage(usd_path=terrain_usd_path, prim_path=terrain_prim_path)

terrain_usd_path2 = "Terrain_generation/output/10x10test.obj"
terrain_prim_path2 = "/World/Terrain"
stage_utils.add_reference_to_stage(usd_path=terrain_usd_path2, prim_path=terrain_prim_path2)



# Add Light Source
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

# Add Scattered Trees
tree_usd_path = "F:/work/models/trees/tree.usd"
for i in range(5):
    prim_path = f"/World/tree_{i}"
    stage_utils.add_reference_to_stage(usd_path=tree_usd_path, prim_path=prim_path)
    tree_prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(tree_prim)
    xform.ClearXformOpOrder()

    Tt = random.uniform(0, 50)
    xform.AddTranslateOp().Set(Gf.Vec3d(Tt, Tt, 0))
    Rt = random.uniform(0, 90)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, Rt))
    St = random.uniform(0.01, 0.1)
    xform.AddScaleOp().Set(Gf.Vec3f(St, St, St))

# 1. Create The Camera
_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/sensor",
    parent=None,
    config="OS2_REV6_32ch10hz2048res",
    translation=(0, 0, 1.0),
    orientation=Gf.Quatd(1,0,0,0),
)
# 2. Create and Attach a render product to the camera
render_product = rep.create.render_product(sensor.GetPath(), [1, 1])

# 3. Create Annotator to read the data from with annotator.get_data()
annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
annotator.attach(render_product)

# 4. Create a Replicator Writer that "writes" points into the scene for debug viewing
writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
writer.attach(render_product)




#lidar
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="Example_Rotary", help="Name of lidar config.")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

# Example for creating a RTX lidar sensor and publishing PCL data
import carb
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

# enable ROS bridge extension
enable_extension("isaacsim.util.debug_draw")

simulation_app.update()

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

lidar_config = args.config

# Create the lidar sensor that generates data into "RtxSensorCpu"
# Possible config options are Example_Rotary and Example_Solid_State
_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/sensor",
    parent=None,
    config=lidar_config,
    translation=(0, 0, 1.0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
)
hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

# Create the debug draw pipeline in the post process graph
writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud" + "Buffer")
writer.attach([hydra_texture])



















# Start a world to step simulator
my_world = World(stage_units_in_meters=1.0)

while simulation_app.is_running():
    simulation_app.update()

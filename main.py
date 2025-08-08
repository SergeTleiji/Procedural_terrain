"""
Main entry point for procedural terrain generation, vegetation instancing,
runtime streaming, and ROS2-enabled simulation in NVIDIA Isaac Sim.

Pipeline Overview:
    1. Generate heightmap using fractal Brownian motion noise (macro + meso scale).
    2. Convert heightmap to mesh (.obj) and then to USD format.
    3. Assign cube-mapped UVs and apply PBR material.
    4. Build semantic map for vegetation type distribution.
    5. Generate scatter points for grass (random) and trees (Poisson disk).
    6. Instance vegetation assets into the scene and place the robot with sensors.
    7. Start simulation with terrain and vegetation streaming based on robot position.

Runtime Modes:
    - Headless: pre-generates and simulates without rendering (for batch/data generation).
    - GUI: runs in Isaac Sim viewport with live streaming updates.

External Requirements:
    - Isaac Sim 4.5+ with ROS2 Bridge (Humble) installed.
    - Assets folder containing models, textures, HDRIs in expected paths.
    - ROS2 Humble environment configured (see `enable_extension` calls below).

Key References:
    - Argument Parser (below): list of all tunable parameters and their effects.
    - Constants & Paths section: configure asset/model/texture/HDRI paths.
    - `LOD.py`: convert high/low detail assets to IsaacSim LOD-supported USDs.
"""

import sys

early = [m for m in sys.modules if m.startswith(("omni.", "pxr", "carb"))]
if early:
    print("[WARN] Omni/USD modules imported before SimulationApp:", early)

import os
import time
import argparse
import numpy as np

# === STEP 0: Initialize Isaac Sim application ===
# Must be done before any pxr/Omni imports.
# Set HEADLESS=True for non-graphical mode (faster for batch runs).
try:
    from omni.isaac.kit import SimulationApp  # preferred when available
except ModuleNotFoundError:
    from isaacsim import SimulationApp  # fallback for pip/venv setups

HEADLESS = False
simulation_app = SimulationApp(
    {"headless": HEADLESS, "disable_viewport_updates": HEADLESS}
)

# === STEP 0.1: Import Isaac Sim, USD, and project modules ===
from pxr import UsdGeom, Usd, UsdPhysics, Sdf
import omni.usd
import omni.timeline
from isaacsim.core.api import World
import isaacsim.core.utils.prims as prims_utils
from omni.isaac.core.utils.extensions import enable_extension
import omni.kit.viewport.utility as vp_utils
import omni.kit.commands

# Project modules (procedural generation pipeline)
from modules.fbm import NoiseGenerator
from modules.warp_terrain_mesh import TerrainMeshGenerator
from modules.poisson import PoissonClass
from modules.instancing import Instancer
from modules.texture_bind import assign_cube_uvs
from modules.material_assign import assign_material_to_usd
from modules.asset_usd_converter import convert_single_file
from modules.random_generation import RandomScatterClass
from modules.generate_semantic_map import SemanticMap
from modules.terrain_streamer import TerrainStreamer
from modules.vegetation_streamer import VegetationStreamer

# === STEP 0.2: Setup ROS2 Bridge environment variables ===
# This allows publishing Isaac Sim sensor data over ROS2.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ["PATH"] += ";F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\lib"
os.environ["PATH"] += ";F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\bin"
os.environ["PYTHONPATH"] = (
    "F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\lib\\site-packages"
)
enable_extension("isaacsim.ros2.bridge")

# === STEP 1: Argument Parser ===
# Allows overriding simulation and generation settings via CLI arguments.
parser = argparse.ArgumentParser(description="Procedural Terrain Generation")
# Terrain & mesh parameters
parser.add_argument("--tile-size", type=int, default=10, help="tile size in m2")
parser.add_argument("--terrain-size", type=int, default=500, help="terrain size in m2")
parser.add_argument(
    "--tile-res", type=int, default=100, help="mesh resolution per tile"
)
parser.add_argument("--seed", type=int, default=36, help="RNG seed for reproducibility")
parser.add_argument(
    "--output-dir", type=str, default="output", help="where to store generated assets"
)
parser.add_argument(
    "--texture-dir",
    type=str,
    default="assets/textures",
    help="PBR texture folder for terrain",
)
parser.add_argument(
    "--roadmap",
    type=str,
    default="output/roadMap1024.png",
    help="reserved for road generation",
)
# Vegetation density & Poisson parameters
parser.add_argument("--density", type=int, default=20, help="grass density (grass/m2)")
parser.add_argument(
    "--min-poisson-radius", type=float, default=5, help="tree Poisson min spacing"
)
parser.add_argument(
    "--max-poisson-radius", type=float, default=7, help="tree Poisson max spacing"
)
# Noise macro/meso parameters
parser.add_argument(
    "--macro-scale", type=float, default=100.0, help="macro hill frequency scale (m)"
)
parser.add_argument(
    "--macro-height", type=float, default=15.0, help="macro hill max height (m)"
)
parser.add_argument(
    "--meso-scale", type=float, default=22.0, help="meso hill frequency scale (m)"
)
parser.add_argument(
    "--meso-height", type=float, default=6, help="meso hill max height (m)"
)
# Runtime simulation
parser.add_argument(
    "--sim-duration", type=float, default=30, help="simulation time (headless mode)"
)
parser.add_argument(
    "--tile-radius", type=int, default=1, help="tiles to load ahead of robot"
)
parser.add_argument(
    "--veg-radius", type=int, default=1, help="vegetation tiles to load ahead of robot"
)
args = parser.parse_args()

# === STEP 2: Asset paths & constants ===
# This section controls HDRIs, robot model, and vegetation assets.
hdri_models = ["hdri/sunny.exr", "hdri/sunset.exr", "hdri/night.exr"]
Robot_model_usd = "assets/robots/Husky.usd"
grass_model_usds = [
    "assets/models/grass/burdockLOD.usda",
    "assets/models/grass/dandelionLOD.usda",
    "assets/models/grass/meadowLOD.usda",
    "assets/models/grass/orchardLOD.usda",
    "assets/models/grass/perennialLOD.usda",
]
tree_model_usds = [
    "assets/models/trees/tree_small_02_1k.usdc",
    "assets/models/trees/pine_sapling_small_2k.usdc",
]
grass_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
tree_weights = [0.5, 0.5]

# === STEP 3: Heightmap generation ===
# Creates an .npy heightmap for the terrain using FBM noise.
noise = NoiseGenerator(
    macro_octaves=1,
    meso_octaves=2,
    macro_freq=1 / args.macro_scale,
    meso_freq=0.1 / args.meso_scale,
    frequency_multiplier=1,
    macro_amplitude=args.macro_height,
    meso_amplitude=args.meso_height,
    seed=args.seed,
)
npy_path = os.path.join(
    args.output_dir, f"A{args.terrain_size}x{args.terrain_size}_0x0.npy"
)
noise.generate_and_save_area(
    x_start_m=0, y_start_m=0, width_m=args.terrain_size, res=1024, output_path=npy_path
)

# === STEP 4: Mesh generation (OBJ) ===
# Converts heightmap to OBJ mesh, applies bump maps for micro detail.
usd_path = os.path.join(
    args.output_dir, f"A{args.terrain_size}x{args.terrain_size}_0x0.usd"
)
prim_path = f"/World/A{args.terrain_size}x{args.terrain_size}_0x0/mesh"
mesh_gen = TerrainMeshGenerator(
    physical_size_m=args.terrain_size,
    hill_path_npy=npy_path,
    obj_output_path=usd_path,
    prim_path=prim_path,
    macro_z_scale=1,
    bump_path_npy=os.path.join(args.texture_dir, "displacement.jpg"),
    micro_z_scale=0.05,
    road_path=args.roadmap,
)
mesh_gen.save_heightmap_usdc()
"""
# === STEP 5: Convert OBJ → USD ===
usd_path = os.path.join(
    args.output_dir, f"A{args.terrain_size}x{args.terrain_size}_0x0.usd"
)
convert_single_file(obj_path, usd_path, load_materials=True)

# === STEP 6: Apply UV mapping and material ===
prim_path = f"/World/A{args.terrain_size}x{args.terrain_size}_0x0/mesh"
assign_cube_uvs(usd_path, prim_path)
"""
assign_material_to_usd(usd_path, args.texture_dir, prim_path, args.terrain_size)

# === STEP 7: Semantic map generation ===
# Assigns vegetation IDs to terrain based on probability weights.
semantic = SemanticMap(grass_weights, map_size=1024, seed=args.seed)
semantic.build()
semantic.save(
    os.path.join(args.output_dir, "semantic_map.npy"),
    os.path.join(args.output_dir, "semantic_map_visual.png"),
)

# === STEP 8: Vegetation point scattering ===
# Grass: Random scatter | Trees: Poisson disk sampling
points = RandomScatterClass.generate_random_points(
    args.output_dir,
    args.terrain_size,
    args.density,
    tile_size=args.tile_size,
    seed=args.seed,
)
Poisson_xy = PoissonClass.generate_poisson_points(
    args.output_dir,
    args.terrain_size,
    r_min=args.min_poisson_radius,
    r_max=args.max_poisson_radius,
    k=1000,
    tile_size=args.tile_size,
    seed=args.seed,
)

# === STEP 9: Instancing models in the scene ===
# Loads terrain mesh, vegetation models, robot, and sensors.
instancer = Instancer(
    hdri=hdri_models,
    seed=args.seed,
    world_x=0,
    world_y=0,
    N_grass=points,
    Poisson=Poisson_xy,
    heightmap_path=npy_path,
    terrain_usd_path=usd_path,
    terrain_size_m=args.tile_size,
    z_scale=1,
    grass_model_usd_paths=grass_model_usds,
    tree_model_usd_paths=tree_model_usds,
    num_instances=args.density * (args.terrain_size**2),
    tree_weights=tree_weights,
    semantic_map=np.load(os.path.join(args.output_dir, "semantic_map.npy")),
    global_size=args.terrain_size,
    Robot_path=Robot_model_usd,
)
instancer.add_robot_and_sensors()
print("[✓] Instancing complete")

# === STEP 10: Setup streaming managers ===
terrain_streamer = TerrainStreamer(
    grass_model_usds, tile_size=args.tile_size, view_radius=args.tile_radius
)
vegetation_streamer = VegetationStreamer(
    poisson_xy=Poisson_xy,
    points=points,
    grass_models=grass_model_usds,
    tile_size=args.tile_size,
    view_radius=args.veg_radius,
    hdri_models=hdri_models,
    seed=args.seed,
    num_instances=args.density * (args.terrain_size**2),
    npy_path=npy_path,
    usd_path=usd_path,
    z_scale=1,
    tree_models=tree_model_usds,
    tree_weights=tree_weights,
    semantic_map_path=np.load(os.path.join(args.output_dir, "semantic_map.npy")),
    output_dir=args.output_dir,
    terrain_size=args.terrain_size,
)

# === STEP 11: Simulation loop ===
# In headless mode: fixed time steps until sim_duration.
# In GUI mode: continuous loop with live terrain streaming based on robot position.

stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=0.002, rendering_dt=0.005)
Initialize = True
# manually setting Husky's ThirdPersonCamera as active viewport
ThirdPersonCam = "/World/Husky/Base_link/thirdperson_view"
if stage.GetPrimAtPath(ThirdPersonCam):
    viewport_api = vp_utils.get_active_viewport()
    viewport_api.set_active_camera(ThirdPersonCam)

# post processing settings

# Global Volumetric Effects (RenderSettings/Common)

omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/raytracing/inscattering/enabled", value=True
)

# FFT Bloom settings (RenderSettings/PostProcessing)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/lensFlares/enabled", value=True
)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/lensFlares/flareScale", value=0.3
)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/lensFlares/physicalSettings", value=False
)

# Motion Blur (RenderSettings/PostProcessing)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/motionblur/enabled", value=True
)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/motionblur/maxBlurDiameterFraction", value=0.02
)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/motionblur/exposureFraction", value=1.0
)
omni.kit.commands.execute(
    "ChangeSetting", path="/rtx/post/motionblur/numSamples", value=8
)

# extracting robot prim from stage
robot_prim = stage.GetPrimAtPath("/World/Husky/Base_link")
xformable = UsdGeom.Xformable(robot_prim)

prims_utils.create_prim(
    prim_path=prim_path,
    prim_type="Xform",
    usd_path=usd_path,
    position=np.array([0, 0, -0.05]),
)


def track_robot_position():
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    robot_pos = (
        translation[0],
        translation[1],
    )
    # print(f"Robot Position: x={translation[0]:.2f}, y={translation[1]:.2f}, z={translation[2]:.2f}")

    new_tiles = terrain_streamer.update(robot_pos)
    if new_tiles:
        vegetation_streamer.update(robot_pos)
        print(f"found {len(new_tiles)} new tiles to create")
        for tile_x, tile_y in new_tiles:
            world_x = tile_x * args.tile_size
            world_y = tile_y * args.tile_size
            name_x = f"N{abs(world_x)}" if world_x < 0 else f"{world_x}"
            name_y = f"N{abs(world_y)}" if world_y < 0 else f"{world_y}"
            print(f"world_x = {world_x}, world_y = {world_y}")

            terrain_name = (
                f"A{int(args.tile_size)}x{int(args.tile_size)}_{name_x}x{name_y}"
            )
            npy_path = os.path.join(args.output_dir, terrain_name + ".npy")
            obj_path = os.path.join(args.output_dir, terrain_name + ".obj")
            usd_path = os.path.join(args.output_dir, terrain_name + ".usd")
            prim_path = f"/World/{terrain_name}/mesh"
            instancer_path = f"/World/Instancer_Tree_{name_x}x{name_y}"

            for i, _ in enumerate(grass_model_usds):
                proto_path = instancer_path + f"/Proto_{i}"
                print(proto_path)
                prim = stage.GetPrimAtPath(f"{proto_path}/Grass_LOD")
                if prim.IsValid():
                    variant_sets = prim.GetVariantSets()
                    if "LODVariant" in variant_sets.GetNames():
                        variant_sets.GetVariantSet("LODVariant").SetVariantSelection(
                            "LOD0"
                        )

            noise.generate_and_save_area(
                x_start_m=world_x,
                y_start_m=world_y,
                width_m=args.tile_size,
                res=args.tile_res,
                output_path=npy_path,
            )

            mesh_gen = TerrainMeshGenerator(
                physical_size_m=args.tile_size,
                hill_path_npy=npy_path,
                obj_output_path=usd_path,
                prim_path=prim_path,
                macro_z_scale=1,
                bump_path_npy=os.path.join(args.texture_dir, "displacement.jpg"),
                micro_z_scale=0.05,
                road_path=args.roadmap,
            )
            mesh_gen.save_heightmap_usdc()

            # convert_single_file(obj_path, usd_path, load_materials=True)
            # assign_cube_uvs(usd_path, prim_path)
            assign_material_to_usd(
                usd_path, args.texture_dir, prim_path, args.tile_size
            )

            prims_utils.create_prim(
                prim_path=prim_path,
                prim_type="Xform",
                usd_path=usd_path,
                position=np.array([world_x, world_y, 0.05]),
            )

            terrain_mesh = stage.GetPrimAtPath(f"{prim_path}/{terrain_name}/mesh")
            UsdPhysics.CollisionAPI.Apply(terrain_mesh)
            UsdPhysics.MeshCollisionAPI.Apply(terrain_mesh)

            collision_api = UsdPhysics.CollisionAPI(terrain_mesh)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(terrain_mesh)

            collision_api.CreateCollisionEnabledAttr(True)
            mesh_collision_api.CreateApproximationAttr()


args = parser.parse_args()
timeline = omni.timeline.get_timeline_interface()
SIM_FPS = 40.0
LIDAR_HZ = 10.0
LIDAR_INTERVAL_STEPS = int(SIM_FPS / LIDAR_HZ)

if HEADLESS:
    print(f"[INFO] Running headless for {args.sim_duration} simulated seconds...")

    for _ in range(30):
        simulation_app.update()

    timeline.play()

    total_sim_steps = int(args.sim_duration * SIM_FPS)
    lidar_tick = 0
    sim_start = time.time()

    for step in range(total_sim_steps):
        simulation_app.update()

        if step % int(SIM_FPS) == 0:
            sim_time = step / SIM_FPS
            wall_time = time.time() - sim_start
            print(f"[Sim] {sim_time:.2f}s | [Wall] {wall_time:.2f}s")

        if step % LIDAR_INTERVAL_STEPS == 0:
            lidar_tick += 1
            print(f"[LIDAR] Tick {lidar_tick} at sim step {step}")

    timeline.stop()
    simulation_app.close()

else:
    print("[INFO] Running in GUI mode...")
    while simulation_app.is_running():
        simulation_app.update()
        track_robot_position()
        if Initialize:
            instancer._add_lidar(Initialize=True)
            # instancer._add_camera(Initialize=True)
            print(f"added Lidar =====================")
            Initialize = False

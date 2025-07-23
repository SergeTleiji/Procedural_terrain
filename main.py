from isaacsim import SimulationApp
HEADLESS = False
simulation_app = SimulationApp({"headless": HEADLESS,
    "disable_viewport_updates": HEADLESS})



import argparse # type: ignore
import os
from pxr import UsdGeom, Gf, Usd

from modules.terrain_noise import PerlinNoiseGenerator
from modules.terrain_mesh import TerrainMeshGenerator
from modules.poisson import PoissonClass
from modules.instancing import Instancer
from modules.texture_bind import assign_cube_uvs
from modules.material_assign import assign_material_to_usd
from modules.asset_usd_converter import convert_single_file
from modules.random_generation import RandomScatterClass
from modules.generate_semantic_map import SemanticMap
from modules.terrain_streamer import TerrainStreamer


# === ENABLING ROS BRIDGE ===
from omni.isaac.core.utils.extensions import enable_extension

os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ["PATH"] += ";F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\lib"
os.environ["PATH"] += ";F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\bin"
os.environ["PYTHONPATH"] = "F:\\IsaacSim\\exts\\isaacsim.ros2.bridge\\humble\\lib\\site-packages"

enable_extension("isaacsim.ros2.bridge")
# === Argument Parser ===
parser = argparse.ArgumentParser(description="Procedural Terrain Generation")

parser.add_argument("--tile-size", type=int, default=10, help="Physical size of the terrain in meters")
parser.add_argument("--tile-res", type=int, default=707, help="Resolution of the terrain tile (e.g. 707x707)")
parser.add_argument("--seed", type=int, default=36, help="Random seed for terrain generation")
parser.add_argument("--output-dir", type=str, default="output/", help="Output directory for files")
parser.add_argument("--texture-dir", type=str, default="assets/textures/grass/", help="Directory for PBR texture maps")
parser.add_argument("--roadmap", type=str, default="output/roadMap1024.png", help="Path to road texture or displacement map")
parser.add_argument("--num-instances", type=int, default=1000, help="Number of instances (e.g. grass)")
parser.add_argument("--poisson-radius", type=float, default=0.5, help="Poisson disk sampling radius")
parser.add_argument("--z-scale", type=float, default=16.0, help="Height multiplier for terrain")
parser.add_argument("--macro-scale", type=float, default=200.0, help="x, y scale multiplier for Macro")
parser.add_argument("--macro-height", type=float, default=4.0, help="Height multiplier for Macro")
parser.add_argument("--meso-scale", type=float, default=60.0, help="x, y scale multiplier for Meso")
parser.add_argument("--meso-height", type=float, default=2.5, help="Height multiplier for Meso")
parser.add_argument("--sim-duration", type=float, default=30, help="duration of headless")
args = parser.parse_args()

# === MODELS SO FAR ===
grass_model_usds=[
    "assets/models/grass/picked/burdocklow.usd",
    "assets/models/grass/picked/meadowlow.usd",
    "assets/models/grass/picked/orchardlow.usd",
    "assets/models/grass/picked/Dandelionlow.usd",
    "assets/models/grass/picked/yorkshirelow.usd"
]
tree_model_usds=[
    "tree_small_02_1k.usdc"
]
'''
grass_model_usds=[
    "/home/serge/Downloads/rocks/models/F1.usdc",
    "/home/serge/Downloads/rocks/models/F2.usdc",
    "/home/serge/Downloads/rocks/models/F3.usdc",
    "/home/serge/Downloads/rocks/models/F4.usdc",
    "/home/serge/Downloads/rocks/models/F5.usdc",
    "/home/serge/Downloads/rocks/models/F6.usdc",
    "/home/serge/Downloads/rocks/models/F7.usdc",
    "/home/serge/Downloads/rocks/models/F8.usdc",
    "/home/serge/Downloads/rocks/models/F9.usdc",
]
'''
grass_weights=[ # === HAVE TO ADD UP TO 1 ===
    .05,
    .05,
    .35,
    .45,
    .1
]
tree_weights=[ # === HAVE TO ADD UP TO 1 ===
    .2,
    .3,
    .1,
    .2,
    .3
]

# === Dynamic file names ===
terrain_name = f"A{int(500)}x{int(500)}_{0}x{0}"
npy_path = os.path.join(args.output_dir, terrain_name + ".npy")
obj_path = os.path.join(args.output_dir, terrain_name + ".obj")
usd_path = os.path.join(args.output_dir, terrain_name + ".usd")
prim_path = f"/World/{terrain_name}/mesh"


# === Step 1: Heightmap ===
noise_gen = PerlinNoiseGenerator(seed=args.seed, tile_res=args.tile_res, tile_size_m=args.tile_size, macro_scale=args.macro_scale, macro_height=args.macro_height, meso_scale=args.meso_scale, meso_height=args.meso_height)
noise_gen.generate_and_save_area(
    x_start_m=0,
    y_start_m=0,
    width_m=500,
    res=1024,
    output_path=npy_path
)


# === Step 2: Mesh Generation ===
mesh_gen = TerrainMeshGenerator(
    physical_size_m=500,
    hill_path_npy=npy_path,
    obj_output_path=obj_path,
    macro_z_scale=args.z_scale,
    bump_path_npy=os.path.join(args.texture_dir, "displacement.jpg"),
    micro_z_scale=0.0125,
    road_path=args.roadmap,
    road_z_scale=1.0
)
mesh_gen.process()

# === Step 3: Convert OBJ to USD ===
convert_single_file(obj_path, usd_path, load_materials=True)

# === Step 4: Assign UVs ===
assign_cube_uvs(usd_path, prim_path)

# === Step 5: Assign Material ===
assign_material_to_usd(usd_path, args.texture_dir, prim_path, 500)

# === Step 6: model map ==
semantic = SemanticMap(grass_weights, map_size=500, num_model_types=len(grass_model_usds), seed=args.seed)
semantic.build()
semantic.save(
    f"{args.output_dir}semantic_map.npy",
    f"{args.output_dir}semantic_map_visual.png"
)

# == Step 7: scattering ==
points = RandomScatterClass.generate_random_points(args.tile_size, 0, 0, args.num_instances)
Poisson_xy = PoissonClass.generate_poisson_points(args.tile_size, 0, 0, r_min = 5, r_max = 7)

# === Step 8: Instancing ===
instancer = Instancer(
    world_x=0,
    world_y=0,
    N_grass=points,
    Poisson = Poisson_xy,
    heightmap_path=npy_path,
    terrain_usd_path=usd_path,
    terrain_size_m=args.tile_size,
    z_scale=args.z_scale,
    grass_model_usd_paths= grass_model_usds,
    tree_model= tree_model_usds,
    num_instances=args.num_instances,
    tree_weights= tree_weights,
    semantic_map=  f"{args.output_dir}semantic_map.npy"
)
instancer.Create_robot()
instancer.Create_Lidar()
instancer.Create_cam()
print("instancing complete")

# === BELOW ARE REQUIRED simulation_app RUNTIME FUNCTIONS ===
terrain_streamer = TerrainStreamer(tile_size=args.tile_size, view_radius=1)
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdGeom, UsdPhysics, Usd
import omni.usd
stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath("/World/Robot")
xformable = UsdGeom.Xformable(robot_prim)


# === COMMENT ME OUT IF U COMMENT OUT OG FUNCTIONS ABOVE ME (i generate 500x500) ===
stage_utils.add_reference_to_stage(usd_path, prim_path=prim_path)
terrain_mesh = stage.GetPrimAtPath(prim_path)
xform = UsdGeom.Xformable(terrain_mesh)


def track_robot_position():

    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    robot_pos = (translation[0], translation[1])
    print(f"🤖 Robot Position: x={translation[0]:.2f}, y={translation[1]:.2f}, z={translation[2]:.2f}")
    new_tiles = terrain_streamer.update(robot_pos)

    if new_tiles:
        print(f"found {len(new_tiles)} new tiles to create")
        for tile_x, tile_y in new_tiles:
            world_x = tile_x * args.tile_size
            world_y = tile_y * args.tile_size
            name_x = f"N{abs(world_x)}" if world_x < 0 else f"{world_x}"
            name_y = f"N{abs(world_y)}" if world_y < 0 else f"{world_y}"
            print(f"world_x = {world_x}, world_y = {world_y}")

            # === dynamics names ===
            terrain_name = f"A{int(args.tile_size)}x{int(args.tile_size)}_{name_x}x{name_y}"
            npy_path = os.path.join(args.output_dir, terrain_name + ".npy")
            obj_path = os.path.join(args.output_dir, terrain_name + ".obj")
            usd_path = os.path.join(args.output_dir, terrain_name + ".usd")
            prim_path = f"/World/{terrain_name}/mesh"

            # === Step 2: Generate heightmap ===
            noise_gen = PerlinNoiseGenerator(seed=args.seed, tile_res=args.tile_res, tile_size_m=args.tile_size, macro_scale=args.macro_scale, macro_height=args.macro_height, meso_scale=args.meso_scale, meso_height=args.meso_height)
            noise_gen.generate_and_save_area( x_start_m=world_x, y_start_m=world_y, width_m=args.tile_size, res=args.tile_res, output_path=npy_path)

            # === Step 3: Generate mesh ===
            mesh_gen = TerrainMeshGenerator(
            physical_size_m=args.tile_size,
            hill_path_npy=npy_path,
            obj_output_path=obj_path,
            macro_z_scale=args.z_scale,
            bump_path_npy=os.path.join(args.texture_dir, "displacement.jpg"),
            micro_z_scale=0.0125,
            road_path=args.roadmap,
            road_z_scale=1.0
            )
            mesh_gen.process()

            # === Step 4: Convert to USD ===
            convert_single_file(obj_path, usd_path, load_materials=True)

            # === Step 5: Apply UVs and material ===
            assign_cube_uvs(usd_path, prim_path)
            assign_material_to_usd(usd_path, args.texture_dir, prim_path, args.tile_size)

            # === TERRAIN SPAWN === /World/Terrain/A100x100/mesh.physics:approximation
            stage_utils.add_reference_to_stage(usd_path, prim_path="/World/Tiles")
            terrain_mesh = stage.GetPrimAtPath(f"/World/Tiles/{terrain_name}/mesh")
            xform = UsdGeom.Xformable(terrain_mesh)
            if xform.GetOrderedXformOps():
                translate_op = xform.GetOrderedXformOps()[0]
            else:
                translate_op = xform.AddTranslateOp()

            translate_op.Set(Gf.Vec3f(world_x, world_y, 0.0))

            # Apply both schemas
            UsdPhysics.CollisionAPI.Apply(terrain_mesh)
            UsdPhysics.MeshCollisionAPI.Apply(terrain_mesh)

            # Create each attribute from the correct API
            collision_api = UsdPhysics.CollisionAPI(terrain_mesh)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(terrain_mesh)

            collision_api.CreateCollisionEnabledAttr(True)
            mesh_collision_api.CreateApproximationAttr()#.Set("TriangleMesh")  # or "convexHull"

            points = RandomScatterClass.generate_random_points(args.tile_size, world_x, world_y, args.num_instances)
            Poisson_xy = PoissonClass.generate_poisson_points(args.tile_size, world_x, world_y, r_min = 5, r_max = 7)

            print(f"generated {args.num_instances} new points ")
            instancer = Instancer(world_x = world_x, world_y = world_y, N_grass=points, Poisson = Poisson_xy, heightmap_path=npy_path, terrain_usd_path=usd_path, terrain_size_m=args.tile_size, z_scale=args.z_scale, grass_model_usd_paths= grass_model_usds, tree_model= tree_model_usds, num_instances=args.num_instances, tree_weights= tree_weights, semantic_map=  f"{args.output_dir}semantic_map.npy")
            instancer.process(False)

        else:
            print("nothing to see here")



    
# === Step 8: Run Simulation ===
import omni.timeline
import time

# === Optional: Argument override for simulation duration ===
args = parser.parse_args()

timeline = omni.timeline.get_timeline_interface()

# === Fixed simulation step (Isaac default) ===
SIM_FPS = 60.0
LIDAR_HZ = 10.0
LIDAR_INTERVAL_STEPS = int(SIM_FPS / LIDAR_HZ)

if HEADLESS:
    print(f"[INFO] Running headless for {args.sim_duration} simulated seconds...")

    # Give assets a few frames to load
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

        # Optional: emulate LiDAR publishing every 6 steps (60/10 = 6)
        if step % LIDAR_INTERVAL_STEPS == 0:
            # LiDAR should auto-publish at 10Hz, but you could trigger callbacks here if needed
            lidar_tick += 1
            print(f"[LIDAR] Tick {lidar_tick} at sim step {step}")

    timeline.stop()
    simulation_app.close()


else:
    print("[INFO] Running in GUI mode...")
    while simulation_app.is_running():
        simulation_app.update()
        track_robot_position()
        


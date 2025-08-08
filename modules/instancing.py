"""
instancing.py
-------------
Handles placement and instancing of grass, trees, robot, and sensors
in Isaac Sim using USD PointInstancer primitives.

Also responsible for:
    - Applying vegetation orientation based on terrain normals.
    - Scaling vegetation based on semantic map classes.
    - Random variation for natural look.
    - HDRI dome lighting.
    - Robot placement and ROS2-enabled sensor setup (LiDAR + RGB camera).

Purpose in Pipeline:
    - Step 9 in `main.py`: after terrain, semantic map, and vegetation points
      have been generated, this module populates the scene.
    - Produces all visible environmental assets and robot sensors for the simulation.

Workflow:
    1. Initialize stage and prepare asset references (grass models, tree models, robot).
    2. Create two USD PointInstancers:
        - Grass instancer (LOD1 by default for performance).
        - Tree instancer (no LOD yet).
    3. Scatter vegetation in the current tile:
        - Grass points from RandomScatterClass output.
        - Tree points from PoissonClass output.
    4. Adjust vegetation:
        - Orient to terrain surface normals.
        - Scale based on semantic map scale class (grass).
        - Random scale within range (trees).
    5. Add HDRI dome light.
    6. Add robot at tile center and mount sensors:
        - RTX LiDAR with ROS2 PointCloud + LaserScan publishers.
        - RGB camera with ROS2 image publishing.

Inputs:
    - hdri (list): HDRI texture file paths for lighting.
    - seed (int): Random seed for reproducibility.
    - world_x, world_y (int): Tile position in world coordinates.
    - N_grass (dict): Grass points bucketed by tile.
    - Poisson (dict): Tree points bucketed by tile.
    - heightmap_path (str): Path to global heightmap `.npy`.
    - terrain_usd_path (str): USD path for the current tile mesh.
    - terrain_size_m (float): Size of each tile (meters).
    - z_scale (float): Height scale multiplier.
    - grass_model_usd_paths (list): USD paths for grass prototypes.
    - tree_model_usd_paths (list): USD paths for tree prototypes.
    - num_instances (int): Max instances per vegetation type.
    - tree_weights (list): Probabilities for tree model selection.
    - semantic_map (np.ndarray): Semantic vegetation map (model ID, scale class).
    - global_size (float): Size of entire terrain in meters.
    - Robot_path (str): USD path for robot model.

Outputs:
    - Updates the USD stage with vegetation, lighting, and sensors.

Dependencies:
    - Isaac Sim (omni.usd, omni.replicator, isaacsim.core.utils.stage, etc.)
    - numpy
    - pxr.UsdGeom, pxr.Gf, pxr.Sdf
    - Called by `main.py` → `Instancer.add_robot_and_sensors()` and `.scatter_instances()`.

Example:
    instancer = Instancer(
        hdri=["hdri/sunny.exr"],
        seed=42,
        world_x=0,
        world_y=0,
        N_grass=grass_points,
        Poisson=tree_points,
        heightmap_path="output/terrain.npy",
        terrain_usd_path="output/terrain.usd",
        terrain_size_m=10,
        z_scale=1,
        grass_model_usd_paths=[...],
        tree_model_usd_paths=[...],
        tree_weights=[0.5, 0.5],
        semantic_map=np.load("output/semantic_map.npy"),
        Robot_path="assets/robots/Husky.usd"
    )
    instancer.scatter_instances()
    instancer.add_robot_and_sensors()
"""

# === Standard Library ===
import random

# === Third-Party ===
import numpy as np
import hashlib

# === Isaac Sim Core ===
import omni.usd
import omni.syntheticdata
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera
import omni.syntheticdata._syntheticdata as sd

# === USD / Pixar ===
from pxr import UsdGeom, Sdf, Gf, UsdLux, Vt


class Instancer:
    """
    Manages vegetation instancing, robot placement, and sensor creation in Isaac Sim.
    """

    def __init__(
        self,
        hdri,
        seed,
        world_x,
        world_y,
        N_grass,
        Poisson,
        heightmap_path,
        terrain_usd_path,
        terrain_size_m,
        z_scale,
        grass_model_usd_paths,
        tree_model_usd_paths,
        num_instances=1000,
        tree_weights=None,
        semantic_map=None,
        global_size=500,
        Robot_path=None,
    ):

        # === Configuration ===
        self.seed = seed
        self.rng = random.Random(seed)
        self.hdri = hdri
        self.world_x = world_x
        self.world_y = world_y
        self.N_grass = N_grass
        self.Poisson = Poisson
        self.terrain_size_m = terrain_size_m
        self.z_scale = z_scale
        self.grass_models = grass_model_usd_paths
        self.tree_models = tree_model_usd_paths
        self.num_instances = num_instances
        self.tree_weights = tree_weights
        self.semantic_map = semantic_map
        self.global_size = global_size
        self.Robot_path = Robot_path

        # === Heightmap Data ===
        self.heightmap = np.load(heightmap_path)
        self.global_heightmap = self.heightmap
        self.global_res = self.heightmap.shape[0]
        self.global_size = global_size
        self.global_m_per_pixel = self.global_size / self.global_res

        # === Stage ===
        self.stage = self._prepare_stage()

        # === Naming for tile-based instancers ===
        self.name_x = f"N{abs(world_x)}" if world_x < 0 else f"{world_x}"
        self.name_y = f"N{abs(world_y)}" if world_y < 0 else f"{world_y}"
        self.instancer_path = f"/World/Instancer_Tree_{self.name_x}x{self.name_y}"
        self.tree_instancer_path = self.instancer_path + "2"

        # === Create PointInstancers ===
        self.total_models = self.grass_models + self.tree_models
        self.instancer = self._create_instancer(self.instancer_path, self.total_models)
        """
        self.tree_instancer = self._create_instancer(
            self.tree_instancer_path, self.tree_models
        )
        """

    # === Stage Preparation ===
    def _prepare_stage(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            stage = omni.usd.get_context().new_stage()
            stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        return stage

    def _setup_lighting(self):
        """Adds HDRI dome light if not already present."""
        if not self.stage.GetPrimAtPath("/World/Lights/DomeLight"):
            dome_light = UsdLux.DomeLight.Define(self.stage, "/World/Lights/DomeLight")
            dome_light.GetIntensityAttr().Set(1000)
            texture = self.hdri[self.rng.randint(0, len(self.hdri) - 1)]
            dome_light.GetTextureFileAttr().Set(texture)
            print("[✓] HDRI Dome Light set")

    # === Instancer Creation ===
    def _create_instancer(self, path, model_paths):
        """Creates a USD PointInstancer and assigns model prototypes."""
        instancer = UsdGeom.PointInstancer.Define(self.stage, Sdf.Path(path))
        proto_paths = []
        for i, model_path in enumerate(model_paths):
            proto_prim_path = f"{path}/Proto_{i}"
            proto_prim = self.stage.DefinePrim(proto_prim_path, "Xform")
            proto_prim.GetReferences().AddReference(model_path)
            proto_paths.append(proto_prim.GetPath())
            lod_prim = self.stage.GetPrimAtPath(f"{proto_prim_path}/Grass_LOD")
            if lod_prim.IsValid():
                variant_sets = lod_prim.GetVariantSets()
                if "LODVariant" in variant_sets.GetNames():
                    variant_sets.GetVariantSet("LODVariant").SetVariantSelection("LOD1")

        instancer.CreatePrototypesRel().SetTargets(proto_paths)
        return instancer

    # === Height Sampling ===
    def _get_height(self, x_world, y_world):
        """Gets terrain height at given world coordinates."""
        i = int(y_world / self.global_m_per_pixel)
        j = int(x_world / self.global_m_per_pixel)
        i = np.clip(i, 0, self.global_res - 1)
        j = np.clip(j, 0, self.global_res - 1)
        return self.global_heightmap[i, j]

    # === Vegetation Scattering ===
    def scatter_instances(self):
        """Scatters grass and tree instances for the current tile."""
        self.tile_x = self.world_x / self.terrain_size_m
        self.tile_y = self.world_y / self.terrain_size_m
        self._scatter_grass(self.N_grass)
        self._scatter_trees(self.Poisson)
        cleared = self.clear_unused_prototype_payloads(self.stage, self.instancer_path)
        if cleared:
            print(f"[PRUNE] Cleared payloads for {cleared} unused prototype slots")

    def _scatter_grass(self, points):
        """Populates grass instances based on semantic map scale classes."""
        points = points.get((self.tile_x, self.tile_y), [])
        positions, orientations, scales, proto_indices = [], [], [], []
        map_res = self.semantic_map.shape[0]
        scale_ranges = {0: (0.2, 0.4), 1: (0.4, 0.6), 2: (0.6, 0.8), 3: (0.8, 1.1)}

        for x, y in points:
            z = float(self._get_height(x, y))
            positions.append(Gf.Vec3f(x, y, z))
            normal = self.get_normal(x, y)
            orientations.append(self._get_orientation(x, y, normal))

            i_px = int(y / self.global_size * map_res)
            j_px = int(x / self.global_size * map_res)
            proto_index = int(self.semantic_map[i_px, j_px, 0])
            scale_class = self.semantic_map[i_px, j_px, 1]
            s = self.rng.uniform(*scale_ranges.get(scale_class, (0.4, 0.6)))
            scales.append(Gf.Vec3h(s, s, s))
            proto_indices.append(proto_index)

        self.instancer.CreatePositionsAttr().Set(positions)
        self.instancer.CreateOrientationsAttr().Set(orientations)
        self.instancer.CreateScalesAttr().Set(scales)
        self.instancer.CreateProtoIndicesAttr().Set(proto_indices)

    def _scatter_trees(self, points):
        points = points.get((self.tile_x, self.tile_y), [])
        positions, orientations, scales, proto_indices = [], [], [], []

        # CDF as you had it
        cdf = np.cumsum(self.tree_weights, dtype=np.float64)
        cdf /= cdf[-1]

        tree_offset = len(self.grass_models)  # <-- CRITICAL

        for x, y in points:
            z = float(self._get_height(x, y))
            positions.append(Gf.Vec3f(float(x), float(y), z))

            normal = self.get_normal(x, y)
            orientations.append(self._get_orientation(x, y, normal))

            s = self.rng.uniform(0.8, 1.2)
            scales.append(Gf.Vec3f(float(s), float(s), float(s)))

            # deterministic pick
            h = hashlib.blake2b(
                f"{int(x)}|{int(y)}|{self.seed}".encode(), digest_size=8
            ).digest()
            u = int.from_bytes(h, "little") / 2**64
            u = min(u, np.nextafter(1.0, 0.0))

            ID = int(np.searchsorted(cdf, u, side="right"))  # 0..len(tree_models)-1
            proto_indices.append(ID + tree_offset)  # <-- OFFSET APPLIED

        # Merge with existing grass arrays (keep types consistent)
        pi = self.instancer
        pos = list(pi.GetPositionsAttr().Get())
        pos.extend(positions)
        pi.GetPositionsAttr().Set(pos)
        ori = list(pi.GetOrientationsAttr().Get())
        ori.extend(orientations)
        pi.GetOrientationsAttr().Set(ori)
        scl = list(pi.GetScalesAttr().Get())
        scl.extend(scales)
        pi.GetScalesAttr().Set(scl)
        idx = list(pi.GetProtoIndicesAttr().Get())
        idx.extend(proto_indices)
        pi.GetProtoIndicesAttr().Set(idx)

    def clear_unused_prototype_payloads(self, stage, instancer_path):
        pi = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))
        idx = list(pi.GetProtoIndicesAttr().Get() or [])
        protos = list(pi.GetPrototypesRel().GetTargets() or [])
        used = sorted(set(i for i in idx if 0 <= i < len(protos)))
        if not used:  # no valid usage → delete instancer or bail
            return (len(protos), 0)

        remap = {old: new for new, old in enumerate(used)}
        pi.GetProtoIndicesAttr().Set(Vt.IntArray([remap[i] for i in idx]))
        pi.GetPrototypesRel().SetTargets([protos[i] for i in used])

        # now safe to delete old unused proto prims
        for i, p in enumerate(protos):
            if i not in used:
                stage.RemovePrim(p)
        return (len(protos), len(used))

    # === Orientation and Normals ===
    def _get_orientation(self, x, y, normal):
        """Computes final orientation quaternion for vegetation instances."""
        angle_z = self.world_x * self.world_y + self.seed
        spin = Gf.Quath(
            float(np.cos(angle_z / 2)),
            Gf.Vec3h(normal[0], normal[1], normal[2]) * float(np.sin(angle_z / 2)),
        )
        tilt = self.get_orientation_quat(x, y)
        return spin * tilt

    def get_normal(self, x, y):
        """Computes normalized surface normal vector from heightmap."""
        i = int(y / self.global_m_per_pixel)
        j = int(x / self.global_m_per_pixel)
        i = np.clip(i, 1, self.global_res - 2)
        j = np.clip(j, 1, self.global_res - 2)

        dzdx = (self.global_heightmap[i, j + 1] - self.global_heightmap[i, j - 1]) / (
            2 * self.global_m_per_pixel
        )
        dzdy = (self.global_heightmap[i + 1, j] - self.global_heightmap[i - 1, j]) / (
            2 * self.global_m_per_pixel
        )
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal /= np.linalg.norm(normal)
        return Gf.Vec3f(*normal)

    def get_orientation_quat(self, x, y):
        """Converts terrain normal into quaternion orientation."""
        normal = self.get_normal(x, y)
        up = Gf.Vec3f(0, 0, 1)
        axis = up ^ normal
        dot = up * normal
        angle = np.arccos(np.clip(dot, -1.0, 1.0)) ** 0.7

        if axis.GetLength() < 1e-4:
            return Gf.Quath(1.0, Gf.Vec3h(0, 0, 0))
        axis.Normalize()
        return Gf.Quath(
            np.cos(angle / 2), Gf.Vec3h(axis[0], axis[1], axis[2]) * np.sin(angle / 2)
        )

    # === Robot and Sensor Setup ===
    def add_robot_and_sensors(self):
        """Adds the robot, LiDAR, and camera to the scene."""
        self._setup_lighting()
        self._add_robot()
        self._add_lidar()
        # self._add_camera()

    def _add_robot(self):
        prim_path = "/World/Husky"
        stage_utils.add_reference_to_stage(
            usd_path=self.Robot_path, prim_path=prim_path
        )
        robot_prim = self.stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xform(robot_prim)
        # robot always spawns in center x, y + constant 5
        center = (self.global_size / 2) + 5
        center_height = self._get_height(center, center)
        xform.GetOrderedXformOps()[0].Set(Gf.Vec3f((center, center, center_height + 1)))
        print("[✓] Husky added")

    def _add_lidar(self, Initialize=False):

        if Initialize:

            # RTX sensors are cameras and must be assigned to their own render product
            hydra_texture = rep.create.render_product(
                "/World/Husky/Base_link/Lidar", [1, 1], name="Isaac"
            )

            # Create Point cloud publisher pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
            writer.initialize(topicName="point_cloud", frameId="Base_link")
            writer.attach([hydra_texture])

            # Create the debug draw pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
            writer.attach([hydra_texture])

            # Create LaserScan publisher pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "ROS2PublishLaserScan")
            writer.initialize(topicName="scan", frameId="Base_link")
            writer.attach([hydra_texture])
            print("[✓] Lidar Should be initialized ============")
            return

        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/World/Husky/Base_link/Lidar",
            config="OS1_REV7_128ch10hz2048res",
            translation=(0.22125, 0, 0.30296),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
        )

    def _add_camera(self, Initialize=False):
        camera_prim_path = "/World/Husky/Base_link/camera"

        if Initialize:
            # Create render product from prim path
            render_product = rep.create.render_product(camera_prim_path, [480, 320])

            # Get the renderVar string for RGB images
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.Rgb.name
            )

            # Initialize ROS2 image writer
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId="Base_link",
                nodeNamespace="",
                queueSize=1,
                topicName="/camera/image_raw",
            )
            writer.attach([render_product])
            print("[✓] Camera writer initialized ============")
            return

        # Create the camera prim
        width, height = 480, 320
        self.camera = Camera(
            prim_path=camera_prim_path,
            translation=(0, 0, 1),
            frequency=1,
            resolution=(width, height),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 90, 0]), degrees=True
            ),
        )
        self.camera.initialize()
        print("[✓] RGB camera added")

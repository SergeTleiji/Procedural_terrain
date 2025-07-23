import omni.syntheticdata
import omni.usd
from pxr import UsdGeom, Sdf, Gf, UsdPhysics, Usd, Vt, UsdLux
import isaacsim.core.utils.stage as stage_utils
import numpy as np
import random  # type: ignore

#for Lidar
from isaacsim.core.api import SimulationContext
import omni.replicator.core as rep 

#for Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw
'''
imports all final models, places terrain and instances while randomizing rotations
'''

class Instancer:
    def __init__(
        self,
        world_x: int,
        world_y:int,
        N_grass: list,
        Poisson: list,
        heightmap_path: str,
        terrain_usd_path: str,
        terrain_size_m: float,
        z_scale: float,
        grass_model_usd_paths: list,
        tree_model: list,
        num_instances: int = 1000,
        tree_weights: list = None,
        seed: int = 23,
        semantic_map: str = None  # <-- type it as a NumPy array
    ):
        self.world_x = world_x
        self.world_y = world_y
        self.N_grass = N_grass
        self.Poisson = Poisson
        self.heightmap = np.load(heightmap_path)
        self.res = self.heightmap.shape[0]
        self.size = terrain_size_m
        self.z_scale = z_scale
        self.grass_models = grass_model_usd_paths
        self.tree_model = tree_model
        self.num_instances = num_instances
        self.tree_weights = tree_weights
        self.rng = random.Random(seed)
        self.m_per_pixel = self.size / self.res
        self.semantic_map = np.load(semantic_map)

        self.stage = self._prepare_stage()
        self.instancer = self._create_instancer()
        '''
        # === TERRAIN SPAWN === /World/Terrain/A100x100/mesh.physics:approximation

        stage_utils.add_reference_to_stage(usd_path=terrain_usd_path, prim_path="/World/Terrain")
        terrain_mesh = self.stage.GetPrimAtPath(f"/World/Terrain/A{self.size}x{self.size}_0x0/mesh")
        # Apply both schemas
        UsdPhysics.CollisionAPI.Apply(terrain_mesh)
        UsdPhysics.MeshCollisionAPI.Apply(terrain_mesh)

        # Create each attribute from the correct API
        collision_api = UsdPhysics.CollisionAPI(terrain_mesh)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI(terrain_mesh)

        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision_api.CreateApproximationAttr()#.Set("TriangleMesh")  # or "convexHull"
        '''
        # === Default Light

        if not self.stage.GetPrimAtPath("/World/SunLight"):
            sun = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/SunLight"))
            sun.CreateIntensityAttr(2500.0)
            sun.CreateAngleAttr(3.0)  # softness
            print("[✓] Added directional sunlight")



    def process(self, Starting = True):
            self.scatter_instances(self.N_grass, True, self.tree_model)
            self.scatter_instances(self.Poisson, False, self.tree_model)
            if Starting:
                self.Create_robot()
                self.Create_Lidar()
                self.Create_cam()

    def _prepare_stage(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            stage = omni.usd.get_context().new_stage()
            stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        return stage

    def _create_instancer(self):
        self.name_x = f"N{abs(self.world_x)}" if self.world_x < 0 else f"{self.world_x}"
        self.name_y = f"N{abs(self.world_y)}" if self.world_y < 0 else f"{self.world_y}"
        instancer_path = f"/World/Instancer_Tree_{self.name_x}x{self.name_y}"
        instancer = UsdGeom.PointInstancer.Define(self.stage, Sdf.Path(instancer_path))

        proto_paths = []
        for i, path in enumerate(self.grass_models):
            proto_path = instancer_path + f"/Proto_{i}"
            prim = self.stage.DefinePrim(proto_path, "Xform")
            prim.GetReferences().AddReference(path)
            proto_paths.append(prim.GetPath())

        instancer.CreatePrototypesRel().SetTargets(proto_paths)
        return instancer
    

    def _get_height(self, x, y):
        i = min(int(y / self.m_per_pixel), self.res - 1)
        j = min(int(x / self.m_per_pixel), self.res - 1)
        return self.heightmap[i, j]

    def scatter_instances(self, scatter, Grass, Tree_model):
        if Grass:
            positions, orientations, scales, proto_indices = [], [], [], []

            map_res = self.semantic_map.shape[0]
            world_size = self.size  # Assuming 500.0 meters

            # Define scale range per class
            scale_ranges = {
                0: (0.2, 0.4),
                1: (0.4, 0.6),
                2: (0.6, 0.8),
                3: (0.8, 1.1)
            }

            for _ in range(len(scatter)):
                x, y = scatter[_]
                local_x = x - self.world_x
                local_y = y - self.world_y
                z = self._get_height(local_x, local_y) + 0.1
                positions.append(Gf.Vec3f(x, y, z))

                normal = self.get_normal(local_x, local_y)
                quat = self.get_orientation_quat(local_x, local_y)

                angle_z = random.uniform(0, 2 * np.pi)
                spin = Gf.Quath(
                    float(np.cos(angle_z / 2)),
                    Gf.Vec3h(normal[0], normal[1], normal[2]) * float(np.sin(angle_z / 2))
                )
                final_quat = spin * quat
                orientations.append(final_quat)

                # === Map global (x, y) to semantic map indices ===
                i_px = int(y / world_size * map_res)
                j_px = int(x / world_size * map_res)
                i_px = max(0, min(i_px, map_res - 1))
                j_px = max(0, min(j_px, map_res - 1))

                # === Read scale class & model ID from semantic map ===
                scale_class = self.semantic_map[i_px, j_px, 1]
                proto_index = self.semantic_map[i_px, j_px, 0]  # model ID

                # === Apply scale ===
                scale_range = scale_ranges.get(scale_class, (0.4, 0.6))
                s = self.rng.uniform(*scale_range)
                print(f"scale of {_}: {s}")
                scales.append(Gf.Vec3h(s, s, s))

                proto_indices.append(int(proto_index))
        
            self.instancer.CreatePositionsAttr().Set(positions)
            self.instancer.CreateOrientationsAttr().Set(orientations)
            self.instancer.CreateScalesAttr().Set(scales)
            self.instancer.CreateProtoIndicesAttr().Set(proto_indices)

            print(f"[✓] Instanced {self.num_instances} models from {len(self.grass_models)} options.")
        else:
            instancer_path = f"/World/Instancer_Tree_{self.name_x}x{self.name_y}2"
            instancer = UsdGeom.PointInstancer.Define(self.stage, Sdf.Path(instancer_path))
            proto_paths = []
            for i, path in enumerate(self.tree_model):
                proto_path = instancer_path + f"/Proto_{i}"
                prim = self.stage.DefinePrim(proto_path, "Xform")
                prim.GetReferences().AddReference(path)
                proto_paths.append(prim.GetPath())

            instancer.CreatePrototypesRel().SetTargets(proto_paths)
            orientations, positions, proto_indices = [], [], []
            for _ in range(len(scatter)):
                x, y = scatter[_]
                local_x = x - self.world_x
                local_y = y - self.world_y
                z = float(self._get_height(local_x, local_y))
                positions.append(Gf.Vec3f(x, y, z))

                normal = self.get_normal(local_x, local_y)
                quat = self.get_orientation_quat(local_x, local_y)

                angle_z = random.uniform(0, 2 * np.pi)
                spin = Gf.Quath(
                    float(np.cos(angle_z / 2)),
                    Gf.Vec3h(normal[0], normal[1], normal[2]) * float(np.sin(angle_z / 2))
                )
                final_quat = spin * quat
                orientations.append(final_quat)
                proto_indices.append(0)
            instancer.CreatePositionsAttr().Set(positions)
            instancer.CreateOrientationsAttr().Set(orientations)
            instancer.CreateProtoIndicesAttr().Set(proto_indices)
            print(f"[✓] Instanced {len(scatter)} trees")




    def get_normal(self, x_m, y_m):
        x_m = np.clip(x_m, 0, self.size)
        y_m = np.clip(y_m, 0, self.size)
        i = int(y_m / self.m_per_pixel)
        j = int(x_m / self.m_per_pixel)

        # Clamp to avoid indexing errors
        i = np.clip(i, 1, self.res - 2)
        j = np.clip(j, 1, self.res - 2)

        dzdx = (self.heightmap[i, j + 1] - self.heightmap[i, j - 1]) / (2 * self.m_per_pixel)
        dzdy = (self.heightmap[i + 1, j] - self.heightmap[i - 1, j]) / (2 * self.m_per_pixel)

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
    def Create_robot(self):
        stage_utils.add_reference_to_stage(usd_path="assets/robots/Husky.usd", prim_path="/World/Robot")
        xform = UsdGeom.Xform(self.stage.GetPrimAtPath("/World/Robot"))
        xform.GetOrderedXformOps()[0].Set(Gf.Vec3f(255, 255, self._get_height(self.size/2, self.size/2) + 0.2))
        print("Robot successfully added")
    def Create_Lidar(self):
        # Create the lidar sensor that generates data into "RtxSensorCpu"
        # Sensor needs to be rotated 90 degrees about X so that its Z up

        # Possible options are Example_Rotary and Example_Solid_State
        # drive sim applies 0.5,-0.5,-0.5,w(-0.5), we have to apply the reverse
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/World/Robot/base_link/Lidar",
            config="OS1_REV7_128ch10hz2048res",
            translation=(0.22125, 0, 0.30296),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
        )

        # RTX sensors are cameras and must be assigned to their own render product
        hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

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
        print("Lidar successfully added")
    def Create_cam(self):
        import omni.replicator.core as rep
        from isaacsim.sensors.camera import Camera
        import numpy as np
        import isaacsim.core.utils.numpy.rotations as rot_utils

        width, height = 480,320

        camera = Camera(
            prim_path="/World/Robot/base_link/camera",
            translation=(0, 0, 1),
            frequency=1,
            resolution=(width, height),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )
        camera.initialize()
        #Publishing to ros2
        import omni.syntheticdata._syntheticdata as sd
        import omni.graph.core as og
        render_product = camera.get_render_product_path()

        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId= "Base_link",
            nodeNamespace= "",
            queueSize= 1,
            topicName= "/camera/image_raw"
        )
        writer.attach([render_product])
        print("Camera successfully added")







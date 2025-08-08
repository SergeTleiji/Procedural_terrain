# terrain_from_heightmap.py
# Minimal: heightmap (.npy or in-memory) -> USD Mesh (.usdc)
# No points, no lights, no animation.

from __future__ import annotations
import os
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt


class TerrainMeshGenerator:
    def __init__(
        self,
        physical_size_m,
        hill_path_npy,
        obj_output_path,
        prim_path,
        macro_z_scale,
        bump_path_npy,
        micro_z_scale,
        road_path,
        road_z_scale=0,
    ):
        self.physical_size_m = physical_size_m
        self.hill_path = hill_path_npy
        self.obj_output_path = obj_output_path
        self.prim_path = prim_path
        self.macro_z_scale = macro_z_scale
        self.bump_path = bump_path_npy
        self.micro_z_scale = micro_z_scale
        self.road_path = road_path
        self.road_z_scale = road_z_scale

    def heightmap_to_mesh_arrays(
        self,
        heightmap: np.ndarray,  # (H,W) float32 (meters or unitless)
        meters_per_pixel: float = 1.0,  # grid spacing in X and Y (m)
        z_scale: float = 1.0,  # scalar for heights
        make_uvs: bool = True,
        make_normals: bool = True,
    ):
        """Return (positions(N,3), indices(M,), normals(N,3) or None, uvs(N,2) or None)."""
        hmap = np.asarray(heightmap, dtype=np.float32) * float(z_scale)
        H, W = hmap.shape
        sx = sy = float(meters_per_pixel)

        # Positions (grid)
        xs, ys = np.meshgrid(
            np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
        )
        xs *= sx
        ys *= sy
        zs = hmap
        positions = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)

        # Indices (two tris per quad)
        tris = []
        for i in range(H - 1):
            row = i * W
            next_row = (i + 1) * W
            for j in range(W - 1):
                a = row + j
                b = a + 1
                c = next_row + j + 1
                d = c - 1
                # tri1: a b c ; tri2: a c d
                tris.extend((a, b, c, a, c, d))
        indices = np.array(tris, dtype=np.int32)

        # UVs (planar [0,1])
        uvs = None
        if make_uvs:
            u = (xs / (sx * (W - 1))).astype(np.float32)
            v = (ys / (sy * (H - 1))).astype(np.float32)
            uvs = np.stack([u, v], axis=-1).reshape(-1, 2)

        # Normals (central-diff, per-vertex)
        normals = None
        if make_normals:
            dzdx = np.zeros_like(zs, dtype=np.float32)
            dzdy = np.zeros_like(zs, dtype=np.float32)
            dzdx[:, 1:-1] = (zs[:, 2:] - zs[:, :-2]) / (2.0 * sx)
            dzdy[1:-1, :] = (zs[2:, :] - zs[:-2, :]) / (2.0 * sy)
            n = np.stack([-dzdx, -dzdy, np.ones_like(zs)], axis=-1)
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-20
            normals = n.reshape(-1, 3).astype(np.float32)

        return positions, indices, normals, uvs

    def write_usdc_mesh(
        self,
        usd_path: str,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        prim_path: str = "/World/Terrain",
        meters_per_unit: float = 1.0,
        material_stage_path: str | None = None,  # optional external material USD
        material_prim_path: str | None = None,  # e.g. "/World/Looks/TerrainMtl"
    ):
        os.makedirs(os.path.dirname(usd_path), exist_ok=True)
        stage = Usd.Stage.CreateNew(usd_path)
        # Create an xform "/World" to be set as default prim
        default_prim: Usd.Prim = UsdGeom.Xform.Define(
            stage, Sdf.Path("/World")
        ).GetPrim()
        stage.SetDefaultPrim(default_prim)
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))

        # Points
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))

        # Topology
        counts = np.full(len(indices) // 3, 3, dtype=np.int32)
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(indices.astype(np.int32))
        )

        # Normals
        if normals is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # UVs
        if uvs is not None:
            pv = UsdGeom.PrimvarsAPI(mesh)
            st = pv.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            st.Set(Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32)))

        # Extent/bounds
        # Extent/bounds  (cast to Python floats to appease Boost)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)

        ext = Vt.Vec3fArray(2)
        ext[0] = Gf.Vec3f(float(mins[0]), float(mins[1]), float(mins[2]))
        ext[1] = Gf.Vec3f(float(maxs[0]), float(maxs[1]), float(maxs[2]))
        mesh.CreateExtentAttr(ext)

        # Optional material bind
        if material_stage_path and material_prim_path:
            rel = os.path.relpath(material_stage_path, os.path.dirname(usd_path))
            stage.GetRootLayer().subLayerPaths.append(rel)
            mtl = UsdShade.Material(stage.GetPrimAtPath(material_prim_path))
            if mtl:
                UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mtl)

        stage.Save()
        return usd_path

    def save_heightmap_usdc(
        self,
        z_scale: float = 1.0,
        meters_per_unit: float = 1.0,
        make_uvs: bool = True,
        make_normals: bool = True,
        material_stage_path: str | None = None,
        material_prim_path: str | None = None,
    ):
        hmap = np.load(self.hill_path).astype(np.float32)
        pos, idx, nrm, uv = self.heightmap_to_mesh_arrays(
            hmap,
            meters_per_pixel=self.physical_size_m / hmap.shape[0],
            z_scale=z_scale,
            make_uvs=make_uvs,
            make_normals=make_normals,
        )
        return self.write_usdc_mesh(
            self.obj_output_path,
            pos,
            idx,
            normals=nrm,
            uvs=uv,
            prim_path=self.prim_path,
            meters_per_unit=meters_per_unit,
            material_stage_path=material_stage_path,
            material_prim_path=material_prim_path,
        )

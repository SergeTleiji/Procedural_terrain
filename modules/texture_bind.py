"""
texture_bind.py
---------------
Generates and assigns UV coordinates to a terrain mesh in USD format,
allowing materials and textures to be correctly projected onto it.

Purpose in Pipeline:
    - Step 6 in `main.py`: after converting the OBJ mesh to USD,
      this step assigns UV mapping so textures align correctly in Isaac Sim.
    - Uses cube projection (X/Y planar projection) to generate UVs.

Workflow:
    1. Load the USD stage and access the specified mesh prim.
    2. Compute the bounding box of the mesh.
    3. For each vertex, compute normalized (u, v) coordinates relative to
       the bounding box dimensions.
    4. Store the UVs in a `Primvar` named "st".
    5. Save the updated USD stage.

Inputs:
    - usd_path: Path to USD file containing the mesh.
    - prim_path: Prim path of the mesh inside the USD stage.

Outputs:
    - Updates the USD file in place with UV coordinates.

Dependencies:
    - Pixar USD Python API (`pxr`).
    - Called by `main.py` → `assign_cube_uvs()`.

Limitations:
    - Uses simple planar projection (X/Y) — may cause stretching on steep slopes.
    - Designed for large-scale terrain; not ideal for objects requiring complex UVs.

Example:
    assign_cube_uvs(
        "output/terrain.usd",
        "/World/Terrain/mesh"
    )
"""

from pxr import Usd, UsdGeom, Gf, Vt, Sdf


def assign_cube_uvs(usd_path, prim_path):
    """
    Generates UV coordinates for a mesh using simple cube (planar) projection.

    Args:
        usd_path (str): Path to the USD file containing the mesh.
        prim_path (str): Prim path to the mesh within the USD stage.

    Raises:
        ValueError: If the mesh is missing or has no vertices.

    Modifies:
        - The USD file at `usd_path` (saves UVs directly to disk).
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise ValueError(f"[✘] Mesh not found at path: {prim_path}")

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()

    if not points or len(points) == 0:
        raise ValueError("[✘] Mesh has no vertices!")

    # Compute bounding box
    min_pt = Gf.Vec3f(points[0])
    max_pt = Gf.Vec3f(points[0])
    for pt in points:
        for i in range(3):
            min_pt[i] = min(min_pt[i], pt[i])
            max_pt[i] = max(max_pt[i], pt[i])
    size = max_pt - min_pt

    # Generate UVs (normalized X/Y coordinates)
    uvs = []
    for pt in points:
        local_pt = Gf.Vec3f(pt) - min_pt
        u = local_pt[0] / size[0] if size[0] > 0 else 0.0
        v = local_pt[1] / size[1] if size[1] > 0 else 0.0
        uvs.append((u, v))

    # Assign UVs as Primvar "st"
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    st.Set(Vt.Vec2fArray(uvs))

    # Save updated stage
    stage.GetRootLayer().Save()
    print(f"[✔] UVs assigned and saved to: {usd_path}")

from pxr import Usd, UsdGeom, Gf, Vt, Sdf

'''
creates the terrain UV map to allow materials to be projected on the mesh 
'''
def assign_cube_uvs(usd_path, prim_path):
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

    # Generate UVs via cube projection (X/Y face projection for simplicity)
    uvs = []
    for pt in points:
        local_pt = Gf.Vec3f(pt) - min_pt
        u = local_pt[0] / size[0] if size[0] > 0 else 0.0
        v = local_pt[1] / size[1] if size[1] > 0 else 0.0
        uvs.append((u, v))

    # Assign UVs to mesh
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)

    st.Set(Vt.Vec2fArray(uvs))

    stage.GetRootLayer().Save()
    print(f"[✔] UVs assigned and saved to: {usd_path}")
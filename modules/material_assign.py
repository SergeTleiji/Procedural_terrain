"""
material_assign.py
------------------
Creates and assigns a PBR (Physically Based Rendering) material to a terrain mesh
in USD format for use in Isaac Sim.

Purpose in Pipeline:
    - Step 6 (after `texture_bind.py`) in `main.py`: applies textures to the terrain mesh.
    - Uses UsdPreviewSurface shader with albedo, roughness, and normal maps.

Workflow:
    1. Open the USD stage containing the mesh.
    2. Create a new Material and Shader network (`UsdPreviewSurface`).
    3. Define texture readers for albedo, roughness, and normal maps.
    4. Set up a UV reader (`stReader`) and 2D transform for scaling and tiling.
    5. Connect textures to shader inputs.
    6. Bind the material to the mesh prim.
    7. Save the USD stage with the new material applied.

Inputs:
    - usd_path: Path to USD file containing the mesh.
    - texture_dir: Folder containing `albedo.jpg`, `rough.jpg`, `normal.jpg`.
    - mesh_path: Prim path to the mesh inside the USD stage.
    - TILE_SIZE_M: Scale factor for UV tiling.

Outputs:
    - Updates the USD file in place with material bindings.

Dependencies:
    - Pixar USD Python API (`pxr`).
    - Called by `main.py` → `assign_material_to_usd()`.

Example:
    assign_material_to_usd(
        usd_path="output/terrain.usd",
        texture_dir="assets/textures/grass2",
        mesh_path="/World/Terrain/mesh",
        TILE_SIZE_M=500
    )
"""

from pxr import Usd, UsdShade, Sdf


def assign_material_to_usd(
    usd_path: str, texture_dir: str, mesh_path: str, TILE_SIZE_M: int
):
    stage = Usd.Stage.Open(usd_path)
    mtl_path = Sdf.Path("/World/Looks/YourMaterial")
    material = UsdShade.Material.Define(stage, mtl_path)

    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.2)

    # === Diffuse Texture ===
    uv_tex = UsdShade.Shader.Define(stage, mtl_path.AppendPath("DiffuseColorTx"))
    uv_tex.CreateIdAttr("UsdUVTexture")
    uv_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        f"{texture_dir}/albedo.jpg"
    )
    uv_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    uv_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    uv_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    # === Roughness Texture ===
    roughness_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("RoughnessTx"))
    roughness_tx.CreateIdAttr("UsdUVTexture")
    roughness_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        f"{texture_dir}/rough.jpg"
    )
    roughness_tx.CreateOutput("r", Sdf.ValueTypeNames.Float)
    roughness_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    roughness_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    # === Normal Map Texture ===
    normal_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("NormalMapTx"))
    normal_tx.CreateIdAttr("UsdUVTexture")
    normal_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        f"{texture_dir}/normal.jpg"
    )
    normal_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    normal_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    normal_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    # === UV Reader and Transform ===
    uv_reader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("stReader"))
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    transform = UsdShade.Shader.Define(stage, mtl_path.AppendPath("UVTransform"))
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        uv_reader.ConnectableAPI(), "result"
    )
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
        (TILE_SIZE_M, TILE_SIZE_M)
    )
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((0.0, 0.0))

    # === Connect ST to Textures ===
    uv_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )
    roughness_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )
    normal_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        transform.ConnectableAPI(), "result"
    )

    # === Connect Texture Outputs to Shader ===
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        uv_tex.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness_tx.ConnectableAPI(), "r"
    )
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal_tx.ConnectableAPI(), "rgb"
    )

    # === Final Output ===
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # === Bind material to mesh ===
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"[✘] Mesh prim not found at: {mesh_path}")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    stage.GetRootLayer().Save()
    print(f"[✓] Material with normal map assigned and bound to: {usd_path}")

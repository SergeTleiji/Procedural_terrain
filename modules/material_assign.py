from pxr import Usd, UsdShade, Sdf

def assign_material_to_usd(usd_path: str, texture_dir: str, mesh_path: str, TILE_SIZE_M: int):
    stage = Usd.Stage.Open(usd_path)
    mtl_path = Sdf.Path("/World/Looks/YourMaterial")
    material = UsdShade.Material.Define(stage, mtl_path)

    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # === Diffuse Texture ===
    uv_tex = UsdShade.Shader.Define(stage, mtl_path.AppendPath("DiffuseColorTx"))
    uv_tex.CreateIdAttr("UsdUVTexture")
    uv_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"{texture_dir}/albedo.jpg")
    uv_tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    uv_tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    uv_tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    # === Roughness Texture ===
    roughness_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("RoughnessTx"))
    roughness_tx.CreateIdAttr("UsdUVTexture")
    roughness_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"{texture_dir}/rough.jpg")
    roughness_tx.CreateOutput("r", Sdf.ValueTypeNames.Float)
    roughness_tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    roughness_tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

    # === Normal Map Texture ===
    normal_tx = UsdShade.Shader.Define(stage, mtl_path.AppendPath("NormalMapTx"))
    normal_tx.CreateIdAttr("UsdUVTexture")
    normal_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"{texture_dir}/normal.jpg")
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
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((TILE_SIZE_M, TILE_SIZE_M))
    transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
    transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((0.0, 0.0))

    # === Connect ST to Textures ===
    uv_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
    roughness_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
    normal_tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")

    # === Connect Texture Outputs to Shader ===
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(uv_tex.ConnectableAPI(), "rgb")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(roughness_tx.ConnectableAPI(), "r")
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal_tx.ConnectableAPI(), "rgb")

    # === Final Output ===
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # === Bind material to mesh ===
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"[✘] Mesh prim not found at: {mesh_path}")
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

    stage.GetRootLayer().Save()
    print(f"[✓] Material with normal map assigned and bound to: {usd_path}")

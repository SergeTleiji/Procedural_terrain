"""
LOD.py
------

Standalone script to convert models into Isaac Sim-supported LOD variant models.

Purpose:
    Converts high- and low-detail model variants into a single USD file
    with proper LOD setup for use in Isaac Sim.

Requirements:
    Supported formats: .usd, .usdc, .usda
    In Blender (or any modeling software):
        - Include both model variants in the scene.
        - Set both model origins to the same base point.
        - Apply all transforms (scale, rotation, position).
        - Rename high-detail model (and its child vertex IDs) to: Grass_LOD0
        - Rename low-detail model (and its child vertex IDs) to: Grass_LOD1

Exporting from Blender 4.5.0:
    1. Delete any object that is not Grass_LOD0 or Grass_LOD1.
    2. Select both models.
    3. Export as USD:
        - Under General: Check "Selection Only", uncheck "Visible Only".
        - Under Object Types: Check "Meshes" only.
        - Click "Export USD".

Folder structure:
    - Place all models in the `{grass_dir}` directory.
    - Inside `{grass_dir}`, create a `textures` folder.
    - Place all textures in `textures/` with the following names (currently supports .png):
        For `model.usd`:
            - `modelAlbedo`   → color texture
            - `modelRough`    → roughness texture
            - `modelMetallic` → metallic texture
            - `modelNormal`   → normal map

"""

import os
from pxr import Usd
from pxr import Usd, UsdGeom, UsdShade, Sdf


def generate_lod_variants(grass_dir):
    for file in os.listdir(grass_dir):
        if not file.endswith((".usd", ".usdc", ".usda")):
            continue

        abs_path = os.path.abspath(os.path.join(grass_dir, file))
        try:
            src_stage = Usd.Stage.Open(abs_path)
        except Exception as e:
            print(f"[!] Failed to open {abs_path}: {e}")
            continue

        # Confirm required LOD prims exist
        lod_paths = [prim.GetPath().pathString for prim in src_stage.Traverse()]
        if "/root/Grass_LOD0" not in lod_paths or "/root/Grass_LOD1" not in lod_paths:
            print(f"[!] Skipping {file} — missing Grass_LOD0 or Grass_LOD1")
            continue

        # Create destination USD
        base_name = os.path.splitext(file)[0]
        dst_path = os.path.join(grass_dir, f"{base_name}LOD.usda")
        stage = Usd.Stage.CreateNew(dst_path)
        material = create_material(base_name, stage, grass_dir)
        UsdGeom.Xform.Define(stage, "/World")
        root = stage.DefinePrim("/World/Grass_LOD", "Xform")
        variant_set = root.GetVariantSets().AddVariantSet("LODVariant")

        for lod in ["LOD0", "LOD1"]:
            variant_set.AddVariant(lod)
            variant_set.SetVariantSelection(lod)
            with variant_set.GetVariantEditContext():
                lod_prim = stage.DefinePrim(f"/World/Grass_LOD/Grass_{lod}", "Xform")
                UsdShade.MaterialBindingAPI(lod_prim).Bind(material)
                lod_prim.GetReferences().AddReference(abs_path, f"/root/Grass_{lod}")

        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        stage.Save()
        print(f"✅ Generated: {dst_path}")


def create_material(base_name, stage, grass_dir):
    texture_dir = f"{grass_dir}/textures"
    mtl_path = Sdf.Path(f"/World/Looks/{base_name}Mat")
    material = UsdShade.Material.Define(stage, mtl_path)
    shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))

    shader.CreateIdAttr("UsdPreviewSurface")

    # Base PBR inputs
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # === Texture inputs ===
    texture_files = {
        "diffuseColor": f"{base_name}Albedo.png",
        "roughness": f"{base_name}Rough.png",
        "metallic": f"{base_name}Metallic.png",
        "normal": f"{base_name}Normal.png",
    }

    for input_name, tex_file in texture_files.items():
        tex_path = os.path.join(texture_dir, tex_file)
        if not os.path.exists(tex_path):
            print(f"Warning: {tex_path} not found")
            continue

        tex_shader = UsdShade.Shader.Define(
            stage, mtl_path.AppendPath(f"{input_name}Tx")
        )
        tex_shader.CreateIdAttr("UsdUVTexture")
        tex_shader = UsdShade.Shader.Define(
            stage, mtl_path.AppendPath(f"{input_name}Tx")
        )
        tex_shader.CreateIdAttr("UsdUVTexture")
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # === UV Reader block ===
        uv_reader = UsdShade.Shader.Define(
            stage, mtl_path.AppendPath(f"{input_name}_UVReader")
        )
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader.GetOutput("result")
        )

        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        if input_name == "normal":
            shader.CreateInput("normal", Sdf.ValueTypeNames.Float3).ConnectToSource(
                tex_shader.GetOutput("rgb")
            )
        else:
            shader.CreateInput(input_name, Sdf.ValueTypeNames.Float3).ConnectToSource(
                tex_shader.GetOutput("rgb")
            )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


# Use absolute or relative as you need
grass_dir = "assets/models/trees"
generate_lod_variants(grass_dir)

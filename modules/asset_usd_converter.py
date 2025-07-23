import asyncio
import omni
import os
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.kit.asset_converter")


'''
converts .obj to fully isaac sim supported .usd format
'''
async def convert(in_file, out_file, load_materials=False):
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)

    success = await task.wait_until_finished()
    return success

def convert_single_file(in_path: str, out_path: str, load_materials=True):
    if not os.path.exists(in_path):
        print(f"[✘] Input file does not exist: {in_path}")
        return False

    print(f"[•] Converting {in_path} to {out_path}")
    result = asyncio.get_event_loop().run_until_complete(
        convert(in_path, out_path, load_materials)
    )

    if result:
        print(f"[✓] Conversion complete: {out_path}")
    else:
        print(f"[✘] Conversion failed for: {in_path}")
    return result



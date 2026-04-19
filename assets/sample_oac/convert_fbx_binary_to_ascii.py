import fbx

def convert_fbx_binary_to_ascii(src_path: str, dst_path: str) -> None:
    mgr = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(mgr, fbx.IOSROOT)
    mgr.SetIOSettings(ios)

    # --- Import ---
    importer = fbx.FbxImporter.Create(mgr, "")
    if not importer.Initialize(src_path, -1, mgr.GetIOSettings()):
        raise RuntimeError("FBX import init failed: " + importer.GetStatus().GetErrorString())

    scene = fbx.FbxScene.Create(mgr, "scene")
    if not importer.Import(scene):
        raise RuntimeError("FBX import failed: " + importer.GetStatus().GetErrorString())
    importer.Destroy()

    # --- Find an ASCII writer ---
    reg = mgr.GetIOPluginRegistry()
    ascii_writer_id = reg.FindWriterIDByDescription("FBX ascii (*.fbx)")

    if ascii_writer_id == -1:
        # No ASCII writer available in this SDK build.
        raise RuntimeError(
            "This FBX SDK build does not include an 'FBX ascii (*.fbx)' writer. "
            "You need an SDK version/distribution that ships the ASCII writer, "
            "or export to another text format (e.g., glTF .gltf, OBJ, DAE)."
        )

    # --- Export ---
    exporter = fbx.FbxExporter.Create(mgr, "")
    if not exporter.Initialize(dst_path, ascii_writer_id, mgr.GetIOSettings()):
        raise RuntimeError("FBX export init failed: " + exporter.GetStatus().GetErrorString())

    # Optional: force an older FBX version (sometimes relevant for ASCII)
    # exporter.SetFileExportVersion("FBX201300", fbx.FbxSceneRenamer.eNone)

    if not exporter.Export(scene):
        raise RuntimeError("FBX export failed: " + exporter.GetStatus().GetErrorString())

    exporter.Destroy()
    mgr.Destroy()

# Example:
convert_fbx_binary_to_ascii("template_flame2020_binary_5k.fbx", "template_file_5k.fbx")

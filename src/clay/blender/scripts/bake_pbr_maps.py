"""Bake two maps for the existing delivery GLB without re-exporting its mesh."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from _common import import_mesh, join_meshes, payload, reset, run  # noqa: E402
from _normal_projection import bake_projected_normals  # noqa: E402


def bake_occlusion(node, path, size):
    image = bpy.data.images.new("AO", size, size, alpha=False, is_data=True)
    node.image = image
    bpy.ops.object.bake(type="EMIT")
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()


def write_tangents(mesh, path):
    mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
    positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    tangent = np.empty((len(mesh.loops), 3), dtype=np.float32)
    sign = np.empty(len(mesh.loops), dtype=np.float32)
    mesh.vertices.foreach_get("co", positions.ravel())
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    mesh.loops.foreach_get("tangent", tangent.ravel())
    mesh.loops.foreach_get("bitangent_sign", sign)
    used, first = np.unique(loop_vertices, return_index=True)
    if len(used) != len(positions):
        raise ValueError("Cannot assign a tangent to an unreferenced delivery vertex")
    tangent = tangent[first]
    # Inverse of Blender's glTF importer. The exporter uses the same rotation
    # for positions and tangents and preserves the MikkTSpace handedness.
    data = np.column_stack((positions[:, 0], positions[:, 2], -positions[:, 1],
                            tangent[:, 0], tangent[:, 2], -tangent[:, 1], sign[first]))
    data.astype("<f4", copy=False).tofile(path)


def rebuild_vertex_normals(low, path):
    mesh = low.data
    previous = np.empty((len(mesh.corner_normals), 3), dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", previous.ravel())
    bpy.context.view_layer.objects.active = low
    if mesh.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    mesh.update()
    current = np.empty_like(previous)
    mesh.corner_normals.foreach_get("vector", current.ravel())
    normals = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertex_normals.foreach_get("vector", normals.ravel())
    vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vertices)
    if not np.allclose(current, normals[vertices], atol=1e-5):
        raise ValueError("The delivery mesh needs split vertices for its shading normals")
    areas = np.empty(len(mesh.polygons), dtype=np.float32)
    mesh.polygons.foreach_get("area", areas)
    if not np.isfinite(areas).all() or not np.any(areas > 0):
        raise ValueError("The delivery mesh has no finite nondegenerate faces")
    normals[:, [1, 2]] = normals[:, [2, 1]] * (1, -1)
    normals.astype("<f4", copy=False).tofile(path)
    return {
        "degenerate_faces": int((areas <= 0).sum()),
        "reversed_shading_corners": int((np.sum(previous * current, axis=1) < 0).sum()),
    }


def main():
    args = payload()
    reset()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"

    import_mesh(args["high"])
    high = join_meshes()
    import_mesh(args["low"])
    low, = [obj for obj in bpy.context.selected_objects if obj.type == "MESH" and obj != high]
    if low.data.uv_layers.active is None:
        raise ValueError("The delivery mesh needs its final UV map before baking")
    scale = low.dimensions.length
    if scale <= 0:
        raise ValueError("The delivery mesh has zero extent")
    quality = rebuild_vertex_normals(low, args["vertex_normals"])
    bake = scene.render.bake
    bake.margin = 8
    bake_projected_normals(high, low, args["resolution"], bake.margin, args["normal_map"])
    if args.get("tangents"):
        write_tangents(low.data, args["tangents"])

    # Bake AO on the delivery surface itself. A second, nearly coincident mesh
    # in the AO scene would introduce false self-shadowing.
    bpy.data.objects.remove(high, do_unlink=True)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    # Only the scratch material changes. The delivery GLB is patched outside Blender.
    low.data.materials.clear()
    material = bpy.data.materials.new("Clay bake target")
    material.use_nodes = True
    low.data.materials.append(material)
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    material.node_tree.nodes.active = node

    bake.use_selected_to_active = False
    scene.cycles.samples = 8
    tree = material.node_tree
    occlusion = tree.nodes.new("ShaderNodeAmbientOcclusion")
    occlusion.only_local = True
    occlusion.samples = 24
    # Short contact rays avoid darkening broad exterior faces near thin shells.
    occlusion.inputs["Distance"].default_value = scale * 0.01
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    tree.links.new(geometry.outputs["True Normal"], occlusion.inputs["Normal"])
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(occlusion.outputs["AO"], emission.inputs["Color"])
    tree.links.new(emission.outputs[0], tree.nodes.get("Material Output").inputs["Surface"])
    tree.nodes.active = node
    # A bounded hemisphere bake preserves exposed surfaces and darkens nearby concavities.
    bake_occlusion(node, args["ao_map"], args["resolution"])
    return {"ok": True, "quality": quality}


run(main)

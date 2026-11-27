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
    return {"ok": True}


run(main)

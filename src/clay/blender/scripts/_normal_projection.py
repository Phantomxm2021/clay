"""Recover tangent-space normals from nearby points on the generated surface."""

import bpy
import numpy as np


def _array(collection, attribute, width, dtype=np.float32):
    values = np.empty((len(collection), width), dtype=dtype)
    collection.foreach_get(attribute, values.ravel())
    return values


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-20)


def _texels(uv, size):
    # Blender images use a bottom-left origin. Sample at pixel centers.
    triangle = uv * size - 0.5
    ab, ac = triangle[1:] - triangle[0]
    determinant = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(determinant) < 1e-8:
        return
    start = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), 0)
    end = np.minimum(np.floor(triangle.max(axis=0)).astype(int), size - 1)
    x = np.arange(start[0], end[0] + 1)[None, :] - triangle[0, 0]
    for first in range(start[1], end[1] + 1, 32):
        y = np.arange(first, min(first + 32, end[1] + 1))[:, None] - triangle[0, 1]
        u = (x * ac[1] - y * ac[0]) / determinant
        v = (ab[0] * y - ab[1] * x) / determinant
        w = 1 - u - v
        row, col = np.nonzero((u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5))
        if len(row):
            weights = np.column_stack((w[row, col], u[row, col], v[row, col])).astype(np.float32)
            yield row + first, col + start[0], weights


def _proximity_probe(high, low):
    # Use Blender's batched, native BVH query on the high mesh's faces.
    mesh = bpy.data.meshes.new("Normal bake probes")
    probe = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(probe)
    probe.matrix_world = low.matrix_world
    tree = bpy.data.node_groups.new("Project normal bake probes", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    input_node = tree.nodes.new("NodeGroupInput")
    output_node = tree.nodes.new("NodeGroupOutput")
    source = tree.nodes.new("GeometryNodeObjectInfo")
    source.inputs["Object"].default_value = high
    source.transform_space = "RELATIVE"
    proximity = tree.nodes.new("GeometryNodeProximity")
    proximity.target_element = "FACES"
    place = tree.nodes.new("GeometryNodeSetPosition")
    tree.links.new(source.outputs["Geometry"], proximity.inputs["Geometry"])
    tree.links.new(input_node.outputs["Geometry"], place.inputs["Geometry"])
    tree.links.new(proximity.outputs["Position"], place.inputs["Position"])
    tree.links.new(place.outputs["Geometry"], output_node.inputs["Geometry"])
    modifier = probe.modifiers.new(tree.name, "NODES")
    modifier.node_group = tree
    return probe


def _sample_normals(probe, positions, tangent, normal, radius):
    n = _unit(normal)
    t = _unit(tangent - n * (tangent * n).sum(axis=1, keepdims=True))
    b = np.cross(n, t)
    offsets = radius * np.stack((t, -t, b, -b), axis=1)
    queries = (positions[:, None, :] + offsets).reshape(-1, 3)
    mesh = probe.data
    mesh.clear_geometry()
    mesh.vertices.add(len(queries))
    mesh.vertices.foreach_set("co", queries.ravel())
    mesh.update()
    evaluated = probe.evaluated_get(bpy.context.evaluated_depsgraph_get())
    hits = _array(evaluated.data.vertices, "co", 3)
    distances = np.linalg.norm(hits - queries, axis=1).reshape(-1, 4)
    samples = hits.reshape(-1, 4, 3)
    du = samples[:, 0] - samples[:, 1]
    dv = samples[:, 2] - samples[:, 3]
    recovered = _unit(np.cross(du, dv))
    # The order of the offsets determines which side of the delivery shell
    # this normal faces. Source triangle winding does not enter the calculation.
    alignment = (n * recovered).sum(axis=1)
    valid = (
        (alignment > 0.30)
        & (distances.max(axis=1) <= 1.8 * radius)
        & (np.linalg.norm(du, axis=1) > 0.85 * radius)
        & (np.linalg.norm(dv, axis=1) > 0.85 * radius)
    )
    recovered[~valid] = n[~valid]
    return recovered


def _extend_margin(atlas, covered, margin):
    steps = ((np.s_[1:, :], np.s_[:-1, :]), (np.s_[:-1, :], np.s_[1:, :]),
             (np.s_[:, 1:], np.s_[:, :-1]), (np.s_[:, :-1], np.s_[:, 1:]))
    for _ in range(margin):
        grown = covered.copy()
        for dst, src in steps:
            fill = ~grown[dst] & covered[src]
            atlas[dst][fill] = atlas[src][fill]
            grown[dst] |= fill
        covered = grown


def _raster_batches(uv, triangles, corner_data, size):
    chunks = []
    count = 0
    for corners in triangles:
        for row, col, weights in _texels(uv[corners], size):
            chunks.append((row, col, weights @ corner_data[corners]))
            count += len(row)
            if count >= 131072:
                yield tuple(np.concatenate(values) for values in zip(*chunks))
                chunks.clear()
                count = 0
    if chunks:
        yield tuple(np.concatenate(values) for values in zip(*chunks))


def bake_projected_normals(high, low, size, margin, path):
    mesh = low.data
    mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
    mesh.calc_loop_triangles()
    uv = _array(mesh.uv_layers.active.uv, "vector", 2)
    triangles = _array(mesh.loop_triangles, "loops", 3, np.int32)
    vertices = _array(mesh.loops, "vertex_index", 1, np.int32)[:, 0]
    positions = _array(mesh.vertices, "co", 3)
    normals = _array(mesh.corner_normals, "vector", 3)
    tangents = _array(mesh.loops, "tangent", 3)
    signs = _array(mesh.loops, "bitangent_sign", 1)
    corner_data = np.column_stack((positions[vertices], normals, tangents, signs))
    # Use a model-space footprint to average out voxel-scale facets at any
    # atlas resolution. UV density varies substantially between charts.
    radius = float(np.linalg.norm(np.ptp(positions, axis=0))) * 0.006

    atlas = np.empty((size, size, 4), dtype=np.float32)
    atlas[:] = (0.5, 0.5, 1.0, 1.0)
    covered = np.zeros((size, size), dtype=bool)
    probe = _proximity_probe(high, low)
    for row, col, samples in _raster_batches(uv, triangles, corner_data, size):
        position, n, t = samples[:, :3], samples[:, 3:6], samples[:, 6:9]
        projected = _sample_normals(probe, position, t, n, radius)

        # Invert the interpolated tangent frame as in Blender's tangent
        # bake. The same triangle and handedness are used at runtime.
        sign = np.where(samples[:, 9:] < 0, -1, 1).astype(np.float32)
        b = np.cross(n, t) * sign
        inverse_rows = np.stack((np.cross(b, n), np.cross(n, t), np.cross(t, b)), axis=1)
        texture_normal = _unit((inverse_rows @ projected[..., None])[..., 0] * sign)
        invalid = ~np.isfinite(texture_normal).all(axis=1) | (texture_normal[:, 2] <= 0)
        texture_normal[invalid] = (0, 0, 1)
        atlas[row, col, :3] = texture_normal * 0.5 + 0.5
        covered[row, col] = True
    probe_mesh = probe.data
    probe_nodes = probe.modifiers[0].node_group
    bpy.data.objects.remove(probe, do_unlink=True)
    bpy.data.meshes.remove(probe_mesh)
    bpy.data.node_groups.remove(probe_nodes)

    if not covered.any():
        raise ValueError("The delivery UVs cover no normal-map texels")
    _extend_margin(atlas, covered, margin)
    image = bpy.data.images.new("NORMAL", size, size, alpha=False, is_data=True)
    image.pixels.foreach_set(atlas.ravel())
    image.file_format = "PNG"
    image.filepath_raw = path
    image.save()
    bpy.data.images.remove(image)

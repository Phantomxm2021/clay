"""Complete a TRELLIS.2 delivery GLB with CPU-baked tangent normals and AO."""

import argparse
import io
import json
import math
import struct
import tempfile
from pathlib import Path

from PIL import Image

from clay.blender.engine import run_script

PNG = "image/png"


def _read_glb(mesh):
    if len(mesh) < 28:
        raise ValueError("Incomplete GLB")
    magic, version, size, json_size, json_type = struct.unpack_from("<4sIII4s", mesh)
    binary_header = 20 + json_size
    if (magic, version, size, json_type) != (b"glTF", 2, len(mesh), b"JSON") or binary_header + 8 > size:
        raise ValueError("Invalid GLB 2.0 header")
    binary_size, binary_type = struct.unpack_from("<I4s", mesh, binary_header)
    binary_start = binary_header + 8
    if binary_type != b"BIN\x00" or binary_start + binary_size != size:
        raise ValueError("Invalid GLB binary chunk")
    doc = json.loads(mesh[20:binary_header])
    buffers = doc.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0] or not 0 < buffers[0]["byteLength"] <= binary_size:
        raise ValueError("Expected a single embedded GLB buffer")
    return doc, bytearray(mesh[binary_start:binary_start + buffers[0]["byteLength"]])


def _write_glb(doc, binary):
    doc["buffers"][0]["byteLength"] = len(binary)
    binary.extend(b"\0" * (-len(binary) % 4))
    metadata = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    header = struct.pack("<4sIII4s", b"glTF", 2, 28 + len(metadata) + len(binary), len(metadata), b"JSON")
    return b"".join((header, metadata, struct.pack("<I4s", len(binary), b"BIN\x00"), binary))


def _png(data, modes):
    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        if image.mode not in modes:
            raise ValueError(f"Unsupported PNG mode: {image.mode}")
        image.load()
        return image.copy()


def _png_bytes(image):
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


def _embedded_png(doc, binary, info):
    texture = doc["textures"][info["index"]]
    image = doc["images"][texture["source"]]
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    if image.get("mimeType") != PNG or view["buffer"] != 0 or not 0 <= start < end <= len(binary):
        raise ValueError("Expected an embedded PNG in the delivery GLB")
    return texture, image, bytes(binary[start:end])


def _append_view(doc, binary, data):
    binary.extend(b"\0" * (-len(binary) % 4))
    index = len(doc["bufferViews"])
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary.extend(data)
    return index


def _accessor_layout(doc, index, packed_size):
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    return (view.get("byteOffset", 0) + accessor.get("byteOffset", 0),
            view.get("byteStride", packed_size), accessor["count"])


def _unit(vector):
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 1e-12:
        return None
    if abs(length - 1.0) <= 1e-5:
        return vector
    return tuple(value / length for value in vector)


def _cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _tangent_frame(tangent, sign, normal):
    unit = _unit(tangent)
    if unit is None:
        axis = min(range(3), key=lambda index: abs(normal[index]))
        unit = _unit(_cross(normal, tuple(float(index == axis) for index in range(3))))
    return (*unit, -1.0 if sign < 0 else 1.0)


def _repair_vertex_frames(doc, binary, primitive):
    attributes = primitive["attributes"]
    normal_offset, normal_stride, count = _accessor_layout(doc, attributes["NORMAL"], 12)
    bad = {}
    changed = False
    for index in range(count):
        offset = normal_offset + index * normal_stride
        normal = struct.unpack_from("<3f", binary, offset)
        unit = _unit(normal)
        if unit is None:
            bad[index] = [0.0, 0.0, 0.0]
        elif unit != normal:
            struct.pack_into("<3f", binary, offset, *unit)
            changed = True
    if bad:
        # Recover only invalid vertices from incident faces. Isolated or
        # entirely degenerate faces have no useful shading direction.
        positions, stride, _ = _accessor_layout(doc, attributes["POSITION"], 12)
        if "indices" in primitive:
            idx = primitive["indices"]
            fmt = {5121: "<B", 5123: "<H", 5125: "<I"}[doc["accessors"][idx]["componentType"]]
            start, step, length = _accessor_layout(doc, idx, struct.calcsize(fmt))
            indices = (struct.unpack_from(fmt, binary, start + i * step)[0] for i in range(length))
        else:
            indices = iter(range(count))
        for triangle in zip(indices, indices, indices):
            invalid = bad.keys() & set(triangle)
            if not invalid:
                continue
            a, b, c = (struct.unpack_from("<3f", binary, positions + i * stride) for i in triangle)
            face = _cross(tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(a, c)))
            for i in invalid:
                for axis, value in enumerate(face):
                    bad[i][axis] += value
        for index, vector in bad.items():
            struct.pack_into("<3f", binary, normal_offset + index * normal_stride,
                             *(_unit(tuple(vector)) or (0.0, 1.0, 0.0)))
        changed = True
    if "TANGENT" in attributes:
        tangent_offset, tangent_stride, count = _accessor_layout(doc, attributes["TANGENT"], 16)
        for index in range(count):
            offset = tangent_offset + index * tangent_stride
            tangent = struct.unpack_from("<4f", binary, offset)
            normal = struct.unpack_from("<3f", binary, normal_offset + index * normal_stride)
            repaired = _tangent_frame(tangent[:3], tangent[3], normal)
            if repaired != tangent:
                struct.pack_into("<4f", binary, offset, *repaired)
                changed = True
    return changed


def _append_tangents(doc, binary, primitive, with_positions):
    attributes = primitive["attributes"]
    position = doc["accessors"][attributes["POSITION"]]
    count = position["count"]
    view = doc["bufferViews"][position["bufferView"]]
    offset = view.get("byteOffset", 0) + position.get("byteOffset", 0)
    stride = view.get("byteStride", 12)
    if len(with_positions) != count * 28 or position["componentType"] != 5126:
        raise ValueError("Cannot match baked tangents to the delivery GLB vertices")
    normal_offset, normal_stride, _ = _accessor_layout(doc, attributes["NORMAL"], 12)
    tangents = bytearray()
    for index, (x, y, z, tx, ty, tz, sign) in enumerate(struct.iter_unpack("<7f", with_positions)):
        original = struct.unpack_from("<3f", binary, offset + stride * index)
        if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(original, (x, y, z))):
            raise ValueError("Blender changed the delivery vertex order or frame")
        normal = struct.unpack_from("<3f", binary, normal_offset + index * normal_stride)
        tangents.extend(struct.pack("<4f", *_tangent_frame((tx, ty, tz), sign, normal)))
    target = _append_view(doc, binary, tangents)
    doc["bufferViews"][target]["target"] = 34962
    attributes["TANGENT"] = len(doc["accessors"])
    doc["accessors"].append({"bufferView": target, "componentType": 5126, "count": count, "type": "VEC4"})


def combine_maps(low_glb, normal_png, ao_png, tangents=None):
    """Patch only the material and texture data for the queue's single-mesh export."""
    doc, binary = _read_glb(low_glb)
    primitives = [primitive for mesh in doc["meshes"] for primitive in mesh["primitives"]]
    if len(primitives) != 1 or not {"NORMAL", "TEXCOORD_0"} <= primitives[0]["attributes"].keys():
        raise ValueError("Expected one TRELLIS.2 primitive with the final UVs and vertex normals")
    material = doc["materials"][primitives[0]["material"]]
    pbr = material["pbrMetallicRoughness"]
    base_info = pbr["baseColorTexture"]
    mr_info = pbr["metallicRoughnessTexture"]
    if any(info.get("texCoord", 0) != 0 or info.get("extensions") for info in (base_info, mr_info)):
        raise ValueError("Expected untransformed TEXCOORD_0 material textures")
    mr_texture, mr_image, mr_bytes = _embedded_png(doc, binary, mr_info)
    _, base_image, base_bytes = _embedded_png(doc, binary, base_info)
    if mr_image is base_image or "normalTexture" in material or "occlusionTexture" in material:
        raise ValueError("Unexpected texture bindings in the TRELLIS.2 output")
    base = _png(base_bytes, ("RGB", "RGBA"))
    mr = _png(mr_bytes, ("RGB", "RGBA"))
    normal = _png(normal_png, ("RGB", "RGBA")).convert("RGB")
    ao = _png(ao_png, ("L", "RGB", "RGBA")).getchannel(0)
    if len({base.size, mr.size, normal.size, ao.size}) != 1:
        raise ValueError("All five material channels need the same UV atlas dimensions")
    min_z, max_z = normal.getchannel("B").getextrema()
    if ao.getextrema()[1] == 0 or max_z <= 128:
        raise ValueError("Blender produced an empty AO or tangent normal map")
    if min_z < 128:
        raise ValueError("Blender produced a backwards tangent normal")

    orm = Image.merge("RGB", (ao, mr.getchannel("G"), mr.getchannel("B")))
    # Keep the old bytes and all of the existing binary offsets intact. Reuse
    # the image's view index when nothing else references that view.
    old_view = mr_image["bufferView"]
    new_view = _append_view(doc, binary, _png_bytes(orm))
    shared_view = any(other is not mr_image and other.get("bufferView") == old_view for other in doc["images"])
    used_by_geometry = any(accessor.get("bufferView") == old_view for accessor in doc["accessors"])
    if not shared_view and not used_by_geometry:
        doc["bufferViews"][old_view] = doc["bufferViews"].pop()
        new_view = old_view
    mr_image["bufferView"] = new_view
    normal_view = _append_view(doc, binary, _png_bytes(normal))
    normal_texture = {"source": len(doc["images"])}
    if "sampler" in mr_texture:
        normal_texture["sampler"] = mr_texture["sampler"]
    material["normalTexture"] = {"index": len(doc["textures"])}
    material["occlusionTexture"] = {"index": mr_info["index"]}
    doc["textures"].append(normal_texture)
    doc["images"].append({"bufferView": normal_view, "mimeType": PNG, "name": "Normal"})
    if tangents is not None and "TANGENT" not in primitives[0]["attributes"]:
        _append_tangents(doc, binary, primitives[0], tangents)
    return _write_glb(doc, binary)


def bake_pbr(high, low, output, *, blender=None):
    paths = tuple(Path(path).resolve() for path in (high, low, output))
    if len(set(paths)) != 3:
        raise ValueError("High mesh, delivery mesh and output must be different files")
    high, low, output = paths
    low_glb = low.read_bytes()
    doc, binary = _read_glb(low_glb)
    primitive = doc["meshes"][0]["primitives"][0]
    repaired = _repair_vertex_frames(doc, binary, primitive)
    material_index = primitive["material"]
    base_info = doc["materials"][material_index]["pbrMetallicRoughness"]["baseColorTexture"]
    _, _, base_bytes = _embedded_png(doc, binary, base_info)
    width, height = _png(base_bytes, ("RGB", "RGBA")).size
    if width != height or not 128 <= width <= 4096:
        raise ValueError("Expected a square 128–4096 pixel delivery UV atlas")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clay-bake-", dir=output.parent) as scratch:
        directory = Path(scratch)
        if repaired:
            low_glb = _write_glb(doc, binary)
            low = directory / "repaired-low.glb"
            low.write_bytes(low_glb)
        normal, ao = directory / "normal.png", directory / "ao.png"
        tangents = directory / "tangents.bin" if "TANGENT" not in primitive["attributes"] else None
        result = run_script("bake_pbr_maps.py", {
            "high": str(high), "low": str(low), "resolution": width,
            "normal_map": str(normal), "ao_map": str(ao),
            "tangents": str(tangents) if tangents else None,
        }, blender=blender, timeout=600)
        if not result.get("ok"):
            raise RuntimeError("Blender did not complete the PBR bake")
        staged = directory / "model.glb"
        staged.write_bytes(combine_maps(low_glb, normal.read_bytes(), ao.read_bytes(),
                                       tangents.read_bytes() if tangents else None))
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high", required=True, type=Path)
    parser.add_argument("--low", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender")
    args = parser.parse_args()
    bake_pbr(args.high, args.low, args.output, blender=args.blender)


if __name__ == "__main__":
    main()

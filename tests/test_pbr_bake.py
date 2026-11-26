from io import BytesIO
import struct

import pytest

Image = pytest.importorskip("PIL.Image")

from clay.pbr_bake import (  # noqa: E402
    _append_view, _embedded_png, _read_glb, _repair_vertex_frames, _write_glb, combine_maps,
)


def png(mode, color, size=(128, 128)):
    stream = BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


def delivery_glb(base_color, mr):
    geometry = b"\x00" * 12
    binary = bytearray(geometry)
    doc = {
        "asset": {"version": "2.0"}, "buffers": [{}],
        "meshes": [{"primitives": [{"attributes": {"NORMAL": 0, "TEXCOORD_0": 1}, "material": 0}]}],
        "accessors": [{"bufferView": 0}, {"bufferView": 0}],
        "bufferViews": [{"buffer": 0, "byteLength": len(geometry)}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                               "metallicRoughnessTexture": {"index": 1}}}],
        "textures": [{"source": 0}, {"source": 1}], "images": [],
    }
    for img in (base_color, mr):
        index = _append_view(doc, binary, img)
        doc["images"].append({"bufferView": index, "mimeType": "image/png"})
    return _write_glb(doc, binary)


def test_bake_preserves_material_channels_and_geometry():
    base_color = png("RGBA", (94, 31, 128, 255))
    mr_image = Image.new("RGB", (128, 128), (0, 200, 0))
    mr_image.paste((0, 42, 255), (64, 0, 128, 128))
    mr = BytesIO()
    mr_image.save(mr, "PNG")
    mesh = delivery_glb(base_color, mr.getvalue())
    normal = png("RGB", (126, 130, 255))
    ao = png("L", 201)
    original, original_data = _read_glb(mesh)
    final, data = _read_glb(combine_maps(mesh, normal, ao))

    for key in ("meshes", "accessors"):
        assert final[key] == original[key]
    assert final["bufferViews"][0] == original["bufferViews"][0]
    assert data.startswith(original_data)
    material = final["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert _embedded_png(final, data, pbr["baseColorTexture"])[2] == base_color
    assert material["occlusionTexture"] == pbr["metallicRoughnessTexture"]
    with Image.open(BytesIO(_embedded_png(final, data, material["occlusionTexture"])[2])) as orm:
        assert orm.getpixel((0, 0)) == (201, 200, 0)
        assert orm.getpixel((127, 0)) == (201, 42, 255)
    with Image.open(BytesIO(_embedded_png(final, data, material["normalTexture"])[2])) as baked_normal:
        assert baked_normal.getpixel((0, 0)) == (126, 130, 255)
    assert len(final["images"]) == 3


def test_bake_rejects_maps_with_incompatible_atlas_dimensions():
    mesh = delivery_glb(png("RGB", (200, 0, 0)), png("RGB", (0, 180, 12)))
    with pytest.raises(ValueError, match="same UV atlas dimensions"):
        combine_maps(mesh, png("RGB", (128, 128, 255), size=(64, 64)), png("L", 255))


@pytest.mark.parametrize("normal, ao", [((128, 128, 128), 200), ((128, 128, 255), 0)])
def test_bake_rejects_empty_maps(normal, ao):
    mesh = delivery_glb(png("RGB", (200, 0, 0)), png("RGB", (0, 180, 12)))
    with pytest.raises(ValueError, match="empty AO or tangent normal"):
        combine_maps(mesh, png("RGB", normal), png("L", ao))


def test_zero_normal_and_tangent_are_repaired_using_the_incident_face():
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    normals = struct.pack("<9f", 0, 0, 0, 0, 0, 1, 0, 0, 1)
    tangents = struct.pack("<12f", 0, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1)
    binary = bytearray()
    doc = {"accessors": [], "bufferViews": []}
    for data in (positions, normals, tangents):
        view = _append_view(doc, binary, data)
        doc["accessors"].append({"bufferView": view, "count": 3})
    primitive = {"attributes": {"POSITION": 0, "NORMAL": 1, "TANGENT": 2}}

    assert _repair_vertex_frames(doc, binary, primitive)
    assert struct.unpack_from("<3f", binary, len(positions)) == (0, 0, 1)
    assert struct.unpack_from("<4f", binary, len(positions) + len(normals)) == (0, 1, 0, 1)
    assert binary[:len(positions)] == positions
    assert binary[-32:] == tangents[-32:]
    assert not _repair_vertex_frames(doc, binary, primitive)

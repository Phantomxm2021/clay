from io import BytesIO

import pytest

Image = pytest.importorskip("PIL.Image")

from clay.pbr_bake import _append_view, _embedded_png, _read_glb, _write_glb, combine_maps  # noqa: E402


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

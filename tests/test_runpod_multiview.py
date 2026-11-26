import base64
import io
from types import SimpleNamespace

import pytest

Image = pytest.importorskip("PIL.Image")

import rp_handler
import rp_process
from clay.gpu_backend.multiview import condition_reference_views


class Sampler:
    def sample(self, model, noise, cond, **kwargs):
        return [self.sample_once(model, noise, step, step - 1, cond, **kwargs) for step in range(4, 0, -1)]

    def sample_once(self, model, noise, step, previous, cond, **kwargs):
        return cond, kwargs["neg_cond"]


def test_every_pass_uses_all_views_and_restores_native_conditioning():
    names = ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler")
    pipe = SimpleNamespace(**{name: Sampler() for name in names})
    pipe.preprocess_image = lambda image: "processed " + image

    def get_cond(images, resolution, include_neg_cond=True):
        result = {"cond": (images[0], resolution)}
        if include_neg_cond:
            result["neg_cond"] = "unconditional"
        return result

    pipe.get_cond = get_cond
    with pytest.raises(ValueError, match="simulated inference failure"):
        with condition_reference_views(pipe, ["side", "back"]):
            for name, resolution in ((names[0], 512), (names[1], 512), (names[1], 1024), (names[2], 1024)):
                cond = pipe.get_cond(["front"], resolution)
                samples = getattr(pipe, name).sample(None, None, **cond)
                assert samples == [((img, resolution), "unconditional")
                                   for img in ("front", "processed side", "processed back", "front")]
            raise ValueError("simulated inference failure")
    assert pipe.get_cond is get_cond
    for name in names:
        sampler = getattr(pipe, name)
        assert "sample" not in vars(sampler) and "sample_once" not in vars(sampler)
        assert sampler.sample(None, None, **pipe.get_cond(["next"], 512)) == [(('next', 512), 'unconditional')] * 4


@pytest.fixture
def photo():
    raw = io.BytesIO()
    Image.new("RGB", (8, 8), (68, 167, 91)).save(raw, format="PNG")
    return base64.b64encode(raw.getvalue()).decode()


def test_request_sends_named_views_and_bake_source_through_existing_export(monkeypatch, photo):
    monkeypatch.setenv("CLAY_IMAGE_BACKEND", "sprint3d")
    monkeypatch.setenv("CLAY_OUTPUT_BUCKET", "test-bucket")
    monkeypatch.setenv("CLAY_OUTPUT_ENDPOINT_URL", "https://example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    pytest.importorskip("botocore")

    def generate(backend, mode, *, image, prompt, target, seed, pbr_bake, views):
        assert (backend, mode, image, prompt, target, seed, pbr_bake) == (
            "trellis2", "image", None, None, 20000, 30, True)
        assert list(views) == ["front", "side", "back"]
        assert set(views.values()) == {photo}
        return b"glTF mesh", 2048, {"reference_views": 3}, b"glTF source"

    def upload(mesh, bucket, endpoint, job_id, source):
        assert source == b"glTF source"
        return {"pbr_bake": {"status": "staged"}}

    monkeypatch.setattr(rp_process, "generate", generate)
    monkeypatch.setattr(rp_handler, "_upload_glb", upload)
    result = rp_handler.handler({"input": {"views_b64": {"back": photo, "front": photo, "side": photo},
                                           "pbr_bake": True, "target_tris": 20000, "seed": 30}})
    assert result["triangles"] == 2048
    assert result["metrics"]["reference_views"] == 3
    assert result["pbr_bake"] == {"status": "staged"}


def test_rejects_incomplete_conflicting_and_oversize_views(monkeypatch, photo):
    monkeypatch.setenv("CLAY_IMAGE_BACKEND", "sprint3d")
    monkeypatch.delenv("CLAY_OUTPUT_BUCKET", raising=False)
    for views in ({"front": photo}, {"side": photo, "back": photo}, {"front": photo, "top": photo}):
        assert "front" in rp_handler.handler({"input": {"views_b64": views}})["error"]
    views = {"front": photo, "back": photo}
    assert "exactly one" in rp_handler.handler({"input": {"views_b64": views, "prompt": "object"}})["error"]
    monkeypatch.setattr(rp_handler, "MAX_INLINE_BYTES", len(base64.b64decode(photo)) + 1)
    assert "together" in rp_handler.handler({"input": {"views_b64": views}})["error"]


def test_single_image_and_text_keep_their_existing_dispatch(monkeypatch, photo):
    monkeypatch.setenv("CLAY_IMAGE_BACKEND", "sprint3d")
    monkeypatch.delenv("CLAY_OUTPUT_BUCKET", raising=False)
    calls = []

    def generate(backend, mode, **kwargs):
        calls.append((backend, mode, kwargs["image"], kwargs["prompt"], kwargs["views"]))
        return b"glTF result", 20, {}, None

    monkeypatch.setattr(rp_process, "generate", generate)
    for input in ({"image_b64": photo}, {"prompt": "a vase"}):
        assert rp_handler.handler({"input": input})["triangles"] == 20
    assert calls == [("trellis2", "image", photo, None, None), ("legacy", "text", None, "a vase", None)]

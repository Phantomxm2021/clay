import json
from types import SimpleNamespace

import pytest
import rp_inference as inference


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.mark.parametrize("pipeline_type", ["1024_cascade", "512"])
def test_skip_texture_512_only_for_cascade(tmp_path, pipeline_type):
    models = {
        key: f"ckpts/{key}"
        for key in (
            "shape_slat_flow_model_512",
            "shape_slat_flow_model_1024",
            "tex_slat_flow_model_512",
            "tex_slat_flow_model_1024",
        )
    }
    _write(
        tmp_path / "pipeline.json",
        json.dumps(
            {
                "args": {"models": models, "default_pipeline_type": pipeline_type},
            }
        ).encode(),
    )
    actual = inference._read_pipeline_config(tmp_path)["args"]["models"]
    if pipeline_type == "1024_cascade":
        models.pop("tex_slat_flow_model_512")
    assert actual == models


def test_fast_checkpoint_preserves_upstream_output_and_dtype(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    class Flow(torch.nn.Linear):
        def __init__(self, extra=False):
            super().__init__(2, 2, dtype=torch.float32)
            if extra:
                self.register_buffer("extra", torch.ones(2), persistent=False)

    path = tmp_path / "flow"
    _write(path.with_suffix(".json"), b'{"name":"SLatFlowModel","args":{}}')
    weights = {
        "weight": torch.tensor([[1.5, 2], [3, -4]], dtype=torch.float16),
        "bias": torch.tensor([0.5, 1], dtype=torch.float16),
    }
    safetensors.save_file(weights, path.with_suffix(".safetensors"))

    models = SimpleNamespace(SLatFlowModel=Flow)
    fast = inference._load_flow_checkpoint(models, path, {})
    stock = Flow()
    stock.load_state_dict(weights)
    inputs = torch.tensor([[2.0, -3.0]])
    torch.testing.assert_close(fast(inputs), stock(inputs), atol=0, rtol=0)
    assert fast.weight.dtype == stock.weight.dtype
    assert inference._load_flow_checkpoint(models, path, {"extra": True}) is None

    del weights["bias"]
    safetensors.save_file(weights, path.with_suffix(".safetensors"))
    assert inference._load_flow_checkpoint(models, path, {}) is None


@pytest.mark.parametrize("cache_state", ["complete", "partial", "other_revision"])
def test_runpod_cache_stages_only_supporting_weights(tmp_path, monkeypatch, cache_state):
    volume = tmp_path / "global-volume"
    hosted = tmp_path / "runpod-host-cache"
    weights = volume / "main"
    work = tmp_path / "scratch"
    work.mkdir()
    monkeypatch.setenv("CLAY_TRELLIS2_WEIGHTS", str(weights))
    monkeypatch.setenv("HF_HUB_CACHE", str(volume))
    monkeypatch.delenv("CLAY_DINOV3_PATH", raising=False)
    monkeypatch.setattr(inference, "RUNPOD_CACHED_MODELS", hosted)

    primary = "ckpts/main-flow"
    support = inference.SPARSE_CHECKPOINT
    config = {
        "args": {
            "models": {"main": primary, "sparse": f"{inference.SPARSE_REPO}/{support}"},
            "image_cond_model": {"args": {}},
            "rembg_model": {"args": {}},
        }
    }
    _write(weights / "pipeline.json", json.dumps(config).encode())
    source = inference._snapshot(volume, inference.SPARSE_REPO, inference.SPARSE_REVISION)
    revision = "old-revision" if cache_state == "other_revision" else inference.TRELLIS2_REVISION
    cache = inference._snapshot(hosted, inference.TRELLIS2_REPO, revision)
    primary_bytes = 0
    support_bytes = 0
    for extension in (".json", ".safetensors"):
        sparse_content = b"volume sparse data"
        _write(source / f"{support}{extension}", sparse_content)
        support_bytes += len(sparse_content)
        content = b"pinned model data" + extension.encode()
        _write(weights / f"{primary}{extension}", content)
        primary_bytes += len(content)
        if cache_state != "partial" or extension == ".json":
            _write(cache / f"{primary}{extension}", content)
    for repo, revision in (
        (inference.DINO_REPO, inference.DINO_REVISION),
        (inference.BIREFNET_REPO, inference.BIREFNET_REVISION),
    ):
        source = inference._snapshot(volume, repo, revision)
        for name in inference.SNAPSHOT_FILES[repo]:
            content = (
                b"class Config(PretrainedConfig):\n    def __init__(self) -> None:\n        pass\n"
                if name == "birefnet.py"
                else b"volume dependency data"
            )
            _write(source / name, content)
            support_bytes += len(content)

    metrics = inference._prepare_weights(work)

    main = work / f"{primary}.safetensors"
    assert main.read_bytes() == (weights / f"{primary}.safetensors").read_bytes()
    assert not (work / f"{support}.safetensors").is_symlink()
    assert not (work / "dinov3/model.safetensors").is_symlink()
    assert not (work / "birefnet/model.safetensors").is_symlink()
    assert "super().__init__()" in (work / "birefnet/birefnet.py").read_text()
    if cache_state == "complete":
        assert main.is_symlink() and main.resolve() == (cache / f"{primary}.safetensors").resolve()
        assert metrics["model_weights_source"] == "runpod_cached_model"
        assert metrics["model_direct_bytes"] == primary_bytes
        assert metrics["model_staged_bytes"] == support_bytes
    else:
        assert not main.is_symlink()
        assert metrics["model_weights_source"] == "volume_copy"
        assert metrics["model_direct_bytes"] == 0
        assert metrics["model_staged_bytes"] == primary_bytes + support_bytes

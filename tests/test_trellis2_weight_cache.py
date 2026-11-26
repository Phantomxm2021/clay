import json

import pytest
import rp_inference as inference


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


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

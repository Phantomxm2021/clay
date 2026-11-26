import io
import json
import os
import sys
from types import SimpleNamespace

import pytest
import rp_inference
import rp_process


def test_inference_process_reports_same_preloaded_model_on_later_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(rp_inference, "_preload_metrics", {})
    monkeypatch.setattr(rp_inference, "_load_trellis2", lambda: calls.append("load"))
    monkeypatch.setattr(
        rp_inference, "_trellis2_image", lambda *_: {"metrics": rp_inference._preload_metrics}
    )
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, encoding="utf-8") as file:
        monkeypatch.setattr(sys, "argv", ["rp_inference.py", "trellis2", str(write_fd)])
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"preload":true}\n{"output":"mesh.glb"}\n'))
        rp_inference.main()
        preload, job = map(json.loads, file)
    assert calls == ["load"]
    assert preload["model_preload_s"] >= 0
    assert len(preload["model_process_boot_id"]) == 32
    assert job["metrics"] == preload


@pytest.mark.parametrize("preload_fails", [False, True])
def test_preload_runs_before_queue_registration_and_can_fall_back(monkeypatch, preload_fails):
    pytest.importorskip("PIL")
    import rp_handler

    calls = []

    def preload():
        calls.append("preload")
        if preload_fails:
            raise TimeoutError()
        return {"model_preload_s": 1.0, "model_process_boot_id": "preloaded"}

    monkeypatch.setenv("CLAY_IMAGE_BACKEND", "sprint3d")
    monkeypatch.setenv("CLAY_TRELLIS2_PRELOAD", "1")
    monkeypatch.setattr(rp_process, "preload_trellis2", preload)
    monkeypatch.setitem(
        sys.modules,
        "runpod",
        SimpleNamespace(
            serverless=SimpleNamespace(
                start=lambda config: calls.append(config["handler"]),
            )
        ),
    )

    rp_handler.main()

    assert calls == ["preload", rp_handler.handler]

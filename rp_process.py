"""Keep the selected generation backend warm in an isolated Python process."""

import atexit
import json
import os
import select
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_active_process = None
_active_backend = None
_reply = None
_backend_scratch = None


def stop_backend():
    global _active_process, _active_backend, _reply, _backend_scratch

    if _active_process is None:
        return
    process, reply, scratch = _active_process, _reply, _backend_scratch
    _active_process = _active_backend = _reply = _backend_scratch = None
    with suppress(OSError):
        process.stdin.close()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        reply.close()
        if scratch is not None:
            scratch.cleanup()


atexit.register(stop_backend)


def _start_backend(backend):
    global _active_process, _active_backend, _reply, _backend_scratch

    stop_backend()
    python = "/opt/trellis2-venv/bin/python" if backend == "trellis2" else "/opt/clay-venv/bin/python"
    paths = [str(ROOT / "src")]
    if backend == "trellis2":
        paths.extend([
            "/opt/TRELLIS.2",
            "/opt/sprint3d/src",
            "/opt/sprint3d/experiments/baselines",
            "/opt/sprint3d/experiments/trellis2_mesh_path",
        ])
    else:
        paths.append("/opt/trellis")
    env = os.environ.copy()
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                 "CLAY_OUTPUT_ENDPOINT_URL", "RUNPOD_API_KEY"):
        env.pop(name, None)
    if backend == "legacy" and env.get("CLAY_TRELLIS2_WEIGHTS"):
        env.pop("HF_HUB_CACHE", None)
        env.pop("HF_MODULES_CACHE", None)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PATH"] = os.pathsep.join([str(Path(python).parent), env["PATH"]])
    scratch = None
    if backend == "trellis2":
        scratch = tempfile.TemporaryDirectory(prefix="clay-model-scratch-")
        env["CLAY_MODEL_SCRATCH"] = scratch.name
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w"):
        reply = os.fdopen(read_fd)
        try:
            _active_process = subprocess.Popen(
                [python, "-u", str(ROOT / "rp_inference.py"), backend, str(write_fd)],
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                pass_fds=(write_fd,),
            )
        except BaseException:
            reply.close()
            if scratch is not None:
                scratch.cleanup()
            raise
    _active_backend = backend
    _reply = reply
    _backend_scratch = scratch


def _exchange(backend, request, timeout):
    if _active_backend != backend or _active_process.poll() is not None:
        _start_backend(backend)
    try:
        _active_process.stdin.write(json.dumps(request) + "\n")
        _active_process.stdin.flush()
        if not select.select([_reply], [], [], timeout)[0]:
            raise TimeoutError("Clay inference exceeded the configured timeout")
        line = _reply.readline()
        if not line:
            raise RuntimeError("Clay's inference process exited. See worker logs.")
        response = json.loads(line)
        if error := response.get("error"):
            raise RuntimeError(error)
        return response
    except (OSError, ValueError, RuntimeError):
        stop_backend()
        raise


def preload_trellis2():
    timeout = int(os.environ.get("CLAY_TRELLIS2_PRELOAD_TIMEOUT_S", "300"))
    return _exchange("trellis2", {"preload": True}, timeout)


def generate(backend, mode, *, image, prompt, target, seed, pbr_bake=False, views=None):
    # The GLB stays on disk until the parent reads it. The pipe carries only
    # request parameters and completion metadata, separate from model logs.
    with tempfile.TemporaryDirectory(prefix="clay-inference-") as directory:
        output = Path(directory) / "model.glb"
        request = dict(mode=mode, image=image, prompt=prompt, target=target, seed=seed, output=str(output),
                       pbr_bake=pbr_bake)
        if views is not None:
            request["views"] = views
        timeout = int(os.environ.get("CLAY_INFERENCE_TIMEOUT_S", "720"))
        response = _exchange(backend, request, timeout)
        try:
            source = output.with_name("bake-source.glb").read_bytes() if pbr_bake else None
            return output.read_bytes(), response["triangles"], response.get("metrics", {}), source
        except OSError:
            stop_backend()
            raise

"""GPU inference process for the RunPod queue worker."""

import base64
import json
import os
import struct
import sys
import tempfile
import time
import traceback
from functools import lru_cache
from pathlib import Path

TRELLIS2_SOURCE = "/opt/TRELLIS.2"
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS2_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BIREFNET_REVISION = "66f7f80c1a64b3c4df327a07eeca4e87130b787b"


def require_pbr(mesh):
    """Verify that the GLB actually references embedded PBR textures and normals."""
    magic, version, size, length, kind = struct.unpack_from("<4sIII4s", mesh)
    if (magic, version, size, kind) != (b"glTF", 2, len(mesh), b"JSON"):
        raise RuntimeError("Invalid TRELLIS.2 GLB header")
    doc = json.loads(mesh[20:20 + length])
    for item in doc["meshes"]:
        for primitive in item["primitives"]:
            material = doc["materials"][primitive["material"]]["pbrMetallicRoughness"]
            if not {"NORMAL", "TEXCOORD_0"} <= primitive["attributes"].keys():
                raise RuntimeError("TRELLIS.2 exported a mesh without normals or UVs")
            for channel in ("baseColorTexture", "metallicRoughnessTexture"):
                texture = doc["textures"][material[channel]["index"]]
                if "bufferView" not in doc["images"][texture["source"]]:
                    raise RuntimeError(f"TRELLIS.2 did not embed {channel}")


def _prepare_weights(work_dir):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    # Check gated access before pulling the much larger 4B snapshot.
    dino_path = os.environ.get("CLAY_DINOV3_PATH")
    if not dino_path:
        try:
            dino_path = snapshot_download(DINO_REPO, allow_patterns=["*.json", "*.safetensors"])
        except GatedRepoError as error:
            raise RuntimeError("Hugging Face access to DINOv3 is required for TRELLIS.2.") from error

    cached = Path(f"/runpod-volume/huggingface-cache/hub/models--microsoft--TRELLIS.2-4B/snapshots/{TRELLIS2_REVISION}")
    weights = os.environ.get("CLAY_TRELLIS2_WEIGHTS")
    if not weights:
        weights = cached if (cached / "pipeline.json").is_file() else snapshot_download(
            TRELLIS2_REPO, revision=TRELLIS2_REVISION, allow_patterns=["pipeline.json", "ckpts/*"]
        )
    weights = Path(weights).resolve()
    config = json.loads((weights / "pipeline.json").read_text())
    args = config["args"]
    args["image_cond_model"]["args"]["model_name"] = str(dino_path)
    args["rembg_model"]["args"]["model_name"] = snapshot_download(
        "ZhengPeng7/BiRefNet", revision=BIREFNET_REVISION,
        allow_patterns=["*.json", "*.py", "*.safetensors"],
    )
    args["low_vram"] = True
    (work_dir / "pipeline.json").write_text(json.dumps(config))
    (work_dir / "ckpts").symlink_to(weights / "ckpts", target_is_directory=True)


@lru_cache(maxsize=1)
def _load_trellis2():
    from baseline_io import configure_process
    configure_process()
    from mesh_path import Trellis2MeshPath

    work_dir = tempfile.TemporaryDirectory(prefix="clay-trellis2-")
    try:
        _prepare_weights(Path(work_dir.name))

        class ClayMeshPath(Trellis2MeshPath):
            def _gpu_phase(self, mesh, resolution, divisor, timings):
                from sprint3d.mesh import bake

                return bake.geometry_gpu_phase(
                    mesh.vertices, mesh.faces, int(resolution), self.target_tris, self.texture_size,
                    mesh_module=self.mesh_module, bvh_cls=self.bvh_cls, remesh_divisor=divisor,
                    fast_simplify=True, gpu_charts=True, timings=timings,
                )

        method = ClayMeshPath(
            arm="mesh_path", disable=("mesh_prep_worker_thread",), source=TRELLIS2_SOURCE,
            weights=work_dir.name, record_latents=False, keep_models_on_gpu=False,
        )
        started = time.perf_counter()
        method.load()
        print(f"TRELLIS.2 models loaded in {time.perf_counter() - started:.2f}s", file=sys.stderr)
        return method, work_dir
    except BaseException:
        work_dir.cleanup()
        raise


def _trellis2_image(request, output):
    load_started = time.perf_counter()
    method, _ = _load_trellis2()
    load_s = time.perf_counter() - load_started

    import torch
    from baseline_io import Stages

    image = output.with_name("input-image")
    image.write_bytes(base64.b64decode(request["image"]))
    method.target_tris = request["target"]
    method.texture_size = int(os.environ.get("CLAY_TRELLIS2_TEXSIZE", "1024"))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stages = Stages()
    started = time.perf_counter()
    stats = method.run(str(image), output.parent, request["seed"], stages)
    torch.cuda.synchronize()
    metrics = {
        "model_load_s": round(load_s, 3),
        "image_to_glb_s": round(time.perf_counter() - started, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    require_pbr(output.read_bytes())
    return {"triangles": stats["output_faces"], "metrics": metrics}


def _legacy(request, output):
    from clay.gpu_backend import runtime

    mesh, triangles = runtime.generate(
        "trellis2", request["mode"], image_b64=request["image"], prompt=request["prompt"],
        fmt="glb", target_tris=request["target"], seed=request["seed"],
    )
    output.write_bytes(mesh)
    return {"triangles": triangles}


def main():
    backend, reply_fd = sys.argv[1:]
    run = _trellis2_image if backend == "trellis2" else _legacy
    with os.fdopen(int(reply_fd), "w", encoding="utf-8") as reply:
        os.set_inheritable(reply.fileno(), False)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = run(request, Path(request["output"]))
            except Exception:
                traceback.print_exc()
                reply.write('{"error":"Clay generation failed. See worker logs."}\n')
                reply.flush()
                return
            reply.write(json.dumps(result) + "\n")
            reply.flush()


if __name__ == "__main__":
    main()

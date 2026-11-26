"""GPU inference process for the RunPod queue worker."""

import base64
import faulthandler
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

TRELLIS2_SOURCE = "/opt/TRELLIS.2"
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS2_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
RUNPOD_CACHED_MODELS = Path("/runpod-volume/huggingface-cache/hub")
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "66f7f80c1a64b3c4df327a07eeca4e87130b787b"
SPARSE_REPO = "microsoft/TRELLIS-image-large"
SPARSE_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
SPARSE_CHECKPOINT = "ckpts/ss_dec_conv3d_16l8_fp16"
SNAPSHOT_FILES = {
    DINO_REPO: ("config.json", "model.safetensors", "preprocessor_config.json"),
    BIREFNET_REPO: ("config.json", "model.safetensors", "BiRefNet_config.py", "birefnet.py"),
}


@contextmanager
def _stage(name):
    started = time.perf_counter()
    print(f"Clay TRELLIS.2 {name} started", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        print(f"Clay TRELLIS.2 {name} failed after {time.perf_counter() - started:.2f}s",
              file=sys.stderr, flush=True)
        raise
    else:
        print(f"Clay TRELLIS.2 {name} completed in {time.perf_counter() - started:.2f}s",
              file=sys.stderr, flush=True)


def _snapshot(cache, repo, revision):
    return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def _patch_birefnet_config(path):
    source = path.read_text()
    config_init = "class Config(PretrainedConfig):\n    def __init__(self) -> None:\n"
    compatible_init = config_init + "        super().__init__()\n"
    if compatible_init in source:
        return
    if source.count(config_init) != 1:
        raise RuntimeError("Pinned BiRefNet config has an unexpected structure")
    # Match the upstream compatibility fix for Transformers 4.56 and later.
    path.write_text(source.replace(config_init, compatible_init, 1))


def _stage_cached_weights(work_dir, weights, hosted_weights):
    """Use RunPod's exact main snapshot and stage the remaining volume files locally."""
    cache = Path(os.environ["HF_HUB_CACHE"])
    dino = Path(os.environ.get("CLAY_DINOV3_PATH") or _snapshot(cache, DINO_REPO, DINO_REVISION))
    birefnet = _snapshot(cache, BIREFNET_REPO, BIREFNET_REVISION)
    sparse = _snapshot(cache, SPARSE_REPO, SPARSE_REVISION)
    config = json.loads((weights / "pipeline.json").read_text())
    args = config["args"]
    copies, main_files = [], []
    for key, checkpoint in args["models"].items():
        if checkpoint == f"{SPARSE_REPO}/{SPARSE_CHECKPOINT}":
            source = sparse / SPARSE_CHECKPOINT
            args["models"][key] = SPARSE_CHECKPOINT
            files = copies
        elif Path(checkpoint).parent == Path("ckpts"):
            source = weights / checkpoint
            files = main_files
        else:
            raise RuntimeError(f"Unexpected TRELLIS.2 checkpoint: {checkpoint}")
        files.extend((Path(f"{source}{ext}"), work_dir / f"{args['models'][key]}{ext}")
                      for ext in (".json", ".safetensors"))
    for repo, source, target in ((DINO_REPO, dino, "dinov3"), (BIREFNET_REPO, birefnet, "birefnet")):
        copies.extend((source / name, work_dir / target / name) for name in SNAPSHOT_FILES[repo])
    def size_checked(source):
        size = source.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Missing checkpoint contents: {source.name}")
        return size

    links = [(hosted_weights / destination.relative_to(work_dir), source, destination)
             for source, destination in main_files]
    direct = bool(links) and hosted_weights.resolve() != weights.resolve() and all(
        cached.is_file() and cached.stat().st_size == size_checked(source)
        for cached, source, _ in links
    )
    cached_bytes = 0
    if direct:
        for cached, _, destination in links:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(cached.resolve())
            cached_bytes += cached.stat().st_size
    else:
        copies.extend(main_files)

    sizes = {source: size_checked(source) for source, _ in copies}
    total = sum(sizes[source] for source, _ in copies)
    if shutil.disk_usage(work_dir).free < total + 2 * 1024**3:
        raise RuntimeError(f"Insufficient worker-local disk to stage {total / 1e9:.2f} GB of weights")

    def copy(entry):
        source, destination = entry
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _stage(f"copy {destination.relative_to(work_dir)} ({sizes[source] / 1e6:.1f} MB)"):
            shutil.copyfile(source, destination)
            if destination.stat().st_size != sizes[source]:
                raise RuntimeError(f"Incomplete local checkpoint copy: {destination.name}")

    # Two large sequential reads keep the network mount busy without opening every shard at once.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(copy, sorted(copies, key=lambda item: sizes[item[0]], reverse=True)))
    _patch_birefnet_config(work_dir / "birefnet" / "birefnet.py")
    args["image_cond_model"]["args"]["model_name"] = str(work_dir / "dinov3")
    args["rembg_model"]["args"]["model_name"] = str(work_dir / "birefnet")
    args["low_vram"] = True
    (work_dir / "pipeline.json").write_text(json.dumps(config))
    return {"model_weights_source": "runpod_cached_model" if direct else "volume_copy",
            "model_direct_bytes": cached_bytes, "model_staged_bytes": total}


def require_pbr(mesh):
    """Verify that the GLB actually references embedded PBR textures and normals."""
    from PIL import Image

    magic, version, size, length, kind = struct.unpack_from("<4sIII4s", mesh)
    if (magic, version, size, kind) != (b"glTF", 2, len(mesh), b"JSON"):
        raise RuntimeError("Invalid TRELLIS.2 GLB header")
    doc = json.loads(mesh[20:20 + length])
    binary_length, binary_tag = struct.unpack_from("<I4s", mesh, 20 + length)
    binary_start = 28 + length
    if binary_tag != b"BIN\x00" or binary_start + binary_length != len(mesh):
        raise RuntimeError("Invalid TRELLIS.2 GLB binary section")
    checked_images = set()
    for item in doc["meshes"]:
        for primitive in item["primitives"]:
            material = doc["materials"][primitive["material"]]["pbrMetallicRoughness"]
            if not {"NORMAL", "TEXCOORD_0"} <= primitive["attributes"].keys():
                raise RuntimeError("TRELLIS.2 exported a mesh without normals or UVs")
            for channel in ("baseColorTexture", "metallicRoughnessTexture"):
                texture = doc["textures"][material[channel]["index"]]
                source = texture["source"]
                image = doc["images"][source]
                if "bufferView" not in image or image.get("mimeType") != "image/png":
                    raise RuntimeError(f"TRELLIS.2 did not embed {channel}")
                if source in checked_images:
                    continue
                view = doc["bufferViews"][image["bufferView"]]
                offset = view.get("byteOffset", 0)
                end = offset + view["byteLength"]
                if view["buffer"] != 0 or not 0 <= offset < end <= binary_length:
                    raise RuntimeError(f"TRELLIS.2 exported an invalid {channel} image range")
                with Image.open(io.BytesIO(mesh[binary_start + offset:binary_start + end]), formats=["PNG"]) as png:
                    if png.mode not in ("RGB", "RGBA"):
                        raise RuntimeError(f"TRELLIS.2 exported an unsupported {channel} image mode")
                    png.verify()
                checked_images.add(source)


def _write_bake_source(path, vertices, faces):
    """Write Sprint3D's material-projection surface in the low GLB's frame."""
    import numpy as np

    vertices = np.asarray(vertices, dtype="<f4").copy(order="C")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected triangulated bake source geometry")
    if not vertices.size or not faces.size or faces.max() >= len(vertices):
        raise ValueError("Invalid bake source geometry")
    # Same position conversion as sprint3d.mesh.bake.bake_and_finalize.
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1]
    binary_length = vertices.nbytes + faces.nbytes
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": binary_length}],
        "bufferViews": [
            {"buffer": 0, "byteLength": vertices.nbytes, "target": 34962},
            {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": faces.nbytes, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3",
             "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
        ],
    }
    metadata = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    with path.open("wb") as file:
        file.write(struct.pack("<4sIII4s", b"glTF", 2, 28 + len(metadata) + binary_length, len(metadata), b"JSON"))
        file.write(metadata)
        file.write(struct.pack("<I4s", binary_length, b"BIN\x00"))
        file.write(memoryview(vertices).cast("B"))
        file.write(memoryview(faces).cast("B"))


def _prepare_weights(work_dir):
    cached = _snapshot(RUNPOD_CACHED_MODELS, TRELLIS2_REPO, TRELLIS2_REVISION)
    configured = os.environ.get("CLAY_TRELLIS2_WEIGHTS")
    if configured and os.environ.get("HF_HUB_CACHE"):
        return _stage_cached_weights(work_dir, Path(configured), cached)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    # Check gated access before pulling the much larger 4B snapshot.
    dino_path = os.environ.get("CLAY_DINOV3_PATH")
    if not dino_path:
        try:
            dino_path = snapshot_download(DINO_REPO, allow_patterns=["*.json", "*.safetensors"])
        except GatedRepoError as error:
            raise RuntimeError("Hugging Face access to DINOv3 is required for TRELLIS.2.") from error

    weights = configured
    if not configured:
        weights = cached if (cached / "pipeline.json").is_file() else snapshot_download(
            TRELLIS2_REPO, revision=TRELLIS2_REVISION, allow_patterns=["pipeline.json", "ckpts/*"]
        )
    weights = Path(weights).resolve()
    config = json.loads((weights / "pipeline.json").read_text())
    args = config["args"]
    args["image_cond_model"]["args"]["model_name"] = str(dino_path)
    args["rembg_model"]["args"]["model_name"] = snapshot_download(
        BIREFNET_REPO, revision=BIREFNET_REVISION,
        allow_patterns=["*.json", "*.py", "*.safetensors"],
    )
    args["low_vram"] = True
    (work_dir / "pipeline.json").write_text(json.dumps(config))
    (work_dir / "ckpts").symlink_to(weights / "ckpts", target_is_directory=True)
    source = "runpod_cached_model" if weights == cached.resolve() else "hugging_face"
    return {"model_weights_source": "configured_path" if configured else source}


@lru_cache(maxsize=1)
def _load_trellis2():
    from baseline_io import configure_process
    offline = bool(os.environ.get("CLAY_TRELLIS2_WEIGHTS") and os.environ.get("HF_HUB_CACHE"))
    configure_process(threads=min(8, len(os.sched_getaffinity(0))), offline=offline)
    imported = time.perf_counter()
    with _stage("import model runtime"):
        from mesh_path import Trellis2MeshPath
    imported = time.perf_counter() - imported

    work_dir = tempfile.TemporaryDirectory(prefix="clay-trellis2-", dir=os.environ.get("CLAY_MODEL_SCRATCH"))
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        staged = time.perf_counter()
        with _stage("prepare weights"):
            weights_metrics = _prepare_weights(Path(work_dir.name))
            print(f"Clay TRELLIS.2 weight preparation: {weights_metrics}", file=sys.stderr, flush=True)
        staged = time.perf_counter() - staged

        disable = ["mesh_prep_worker_thread"]
        if os.environ.get("CLAY_TRELLIS2_STOCK_NATIVE") == "1":
            disable.extend(("fast_gpu_simplify", "gpu_bvh_build"))

        class ClayMeshPath(Trellis2MeshPath):
            keep_pbr_source = False

            def _gpu_phase(self, mesh, resolution, divisor, timings):
                from sprint3d.mesh import bake

                return bake.geometry_gpu_phase(
                    mesh.vertices, mesh.faces, int(resolution), self.target_tris, self.texture_size,
                    mesh_module=self.mesh_module, bvh_cls=self.bvh_cls, remesh_divisor=divisor,
                    fast_simplify="fast_gpu_simplify" in self.levers, gpu_charts=True, timings=timings,
                )

            def _bake_and_write(self, stages, gpu_ctx, uv, attr_volume, coords, attr_layout, out_dir, stats):
                super()._bake_and_write(stages, gpu_ctx, uv, attr_volume, coords, attr_layout, out_dir, stats)
                if self.keep_pbr_source:
                    with _stage("write PBR bake source"):
                        _write_bake_source(out_dir / "bake-source.glb", gpu_ctx["vertices"].cpu().numpy(),
                                           gpu_ctx["faces"].cpu().numpy())

        method = ClayMeshPath(
            arm="mesh_path", disable=disable, source=TRELLIS2_SOURCE,
            weights=work_dir.name, record_latents=False, keep_models_on_gpu=False,
        )
        initialized = time.perf_counter()
        with _stage("initialize model and GPU extensions"):
            from trellis2 import models

            load_checkpoint = models.from_pretrained

            def load_timed(path, **kwargs):
                with _stage(f"load {Path(path).name}"):
                    return load_checkpoint(path, **kwargs)

            with patch.object(models, "from_pretrained", load_timed):
                method.load()
        return method, work_dir, {
            **weights_metrics,
            "model_import_s": round(imported, 3),
            "model_stage_s": round(staged, 3),
            "model_init_s": round(time.perf_counter() - initialized, 3),
        }
    except BaseException:
        work_dir.cleanup()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


def _reference_image(encoded):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        oriented = ImageOps.exif_transpose(image)
        return oriented.convert("RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB")


def _trellis2_image(request, output):
    model_cache_hit = _load_trellis2.cache_info().currsize > 0
    load_started = time.perf_counter()
    method, _, initialization = _load_trellis2()
    load_s = time.perf_counter() - load_started

    import torch
    from baseline_io import Stages
    from clay.gpu_backend.multiview import condition_reference_views

    image = output.with_name("input-image")
    views = request.get("views")
    if views:
        with _reference_image(views["front"]) as front:
            front.save(image, format="PNG")
        references = [_reference_image(views[name]) for name in ("side", "back") if name in views]
    else:
        image.write_bytes(base64.b64decode(request["image"]))
        references = []
    method.target_tris = request["target"]
    method.texture_size = int(os.environ.get("CLAY_TRELLIS2_TEXSIZE", "1024"))
    method.keep_pbr_source = request.get("pbr_bake", False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stages = Stages()
    started = time.perf_counter()
    conditioning = condition_reference_views(method.pipe, references) if references else nullcontext()
    with _stage("generate mesh and bake PBR maps"), torch.no_grad(), conditioning:
        stats = method.run(str(image), output.parent, request["seed"], stages)
    torch.cuda.synchronize()
    metrics = {
        "model_load_s": round(load_s, 3),
        "model_cache_hit": model_cache_hit,
        "image_to_glb_s": round(time.perf_counter() - started, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    if not model_cache_hit:
        metrics.update(initialization)
    if views:
        metrics["reference_views"] = len(views)
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

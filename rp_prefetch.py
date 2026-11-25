"""Stage verified TRELLIS.2 image-generation weights on a mounted RunPod volume."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

TRELLIS2 = "microsoft/TRELLIS.2-4B"
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BIREFNET = "ZhengPeng7/BiRefNet"
TRELLIS1 = "microsoft/TRELLIS-image-large"
REVISIONS = {
    TRELLIS2: "af44b45f2e35a493886929c6d786e563ec68364d",
    DINO: "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    BIREFNET: "66f7f80c1a64b3c4df327a07eeca4e87130b787b",
    TRELLIS1: "25e0d31ffbebe4b5a97464dd851910efc3002d96",
}
DINO_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
BIREFNET_FILES = ("config.json", "model.safetensors", "BiRefNet_config.py", "birefnet.py")
DECODER = f"{TRELLIS1}/ckpts/ss_dec_conv3d_16l8_fp16"
SUFFIXES = (".json", ".safetensors")
CHUNK = 8 * 1024 * 1024


def checkpoint_files(config):
    stems = config["args"]["models"].values()
    if DECODER not in stems:
        raise RuntimeError("Unexpected TRELLIS.2 sparse decoder in the pinned pipeline")
    local = ["pipeline.json"]
    for stem in stems:
        if stem == DECODER:
            continue
        path = PurePosixPath(stem)
        if path.parent != PurePosixPath("ckpts"):
            raise RuntimeError(f"Unexpected checkpoint path: {stem}")
        local.extend(stem + suffix for suffix in SUFFIXES)
    return {
        TRELLIS2: tuple(sorted(set(local))),
        DINO: DINO_FILES,
        BIREFNET: BIREFNET_FILES,
        TRELLIS1: tuple(DECODER.removeprefix(TRELLIS1 + "/") + suffix for suffix in SUFFIXES),
    }


def inspect_file(path, remote):
    if not path.is_file() or path.stat().st_size != remote.size:
        return None
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {remote.size}\0".encode())
    with path.open("rb") as chunked:
        for chunk in iter(lambda: chunked.read(CHUNK), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    if remote.lfs:
        if sha256.hexdigest() != remote.lfs.sha256:
            return None
    elif not remote.blob_id or git_blob.hexdigest() != remote.blob_id:
        return None
    return sha256.hexdigest()


def stage(root, scratch, dino_snapshot):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    config = hf_hub_download(
        TRELLIS2, "pipeline.json", revision=REVISIONS[TRELLIS2], cache_dir=scratch, token=False
    )
    planned = checkpoint_files(json.loads(Path(config).read_text()))
    metadata = {}
    # Fail before writing large weights if the approved DINOv3 token cannot read its files.
    for repo in (DINO, TRELLIS2, BIREFNET, TRELLIS1):
        info = api.model_info(repo, revision=REVISIONS[repo], files_metadata=True, token=False)
        files = {e.rfilename: e for e in info.siblings}
        if set(planned[repo]) - files.keys():
            raise RuntimeError(f"Required files are missing from the pinned revision of {repo}")
        for filename in planned[repo]:
            entry = files[filename]
            if not entry.size or not (entry.lfs or entry.blob_id):
                raise RuntimeError(f"Missing size or checksum for {repo}/{filename}")
            metadata[repo, filename] = entry

    # Check the authorized host cache before downloading several gigabytes.
    for filename in DINO_FILES:
        if not inspect_file(dino_snapshot / filename, metadata[DINO, filename]):
            raise RuntimeError(f"The authorized DINOv3 snapshot is unavailable or incomplete: {filename}")

    total = sum(remote.size for remote in metadata.values())
    print(f"Verifying and staging {len(metadata)} files ({total / 1e9:.2f} GB) in {root}", flush=True)
    results = []
    for repo, filenames in planned.items():
        revision = REVISIONS[repo]
        cache_repo = root / "hub" / ("models--" + repo.replace("/", "--"))
        snapshot = cache_repo / "snapshots" / revision
        for filename in filenames:
            remote = metadata[repo, filename]
            dest = snapshot / filename
            digest = inspect_file(dest, remote)
            if not digest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                downloaded = None
                if repo == DINO:
                    print(f"Copying authorized DINOv3 file {filename}", flush=True)
                    source = dino_snapshot / filename
                else:
                    print(f"Downloading {repo}/{filename} ({remote.size / 1e6:.1f} MB)", flush=True)
                    downloaded = Path(hf_hub_download(
                        repo, filename, revision=revision, cache_dir=scratch, token=False
                    ))
                    source = downloaded.resolve(strict=True)
                    if not inspect_file(source, remote):
                        raise RuntimeError(f"Upstream checksum mismatch for {repo}/{filename}")
                shutil.copyfile(source, dest)
                digest = inspect_file(dest, remote)
                if not digest:
                    raise RuntimeError(f"Mounted copy failed verification for {repo}/{filename}")
                if downloaded is not None:
                    if downloaded.is_symlink():
                        downloaded.unlink()
                    source.unlink()
            results.append({"repo": repo, "revision": revision, "file": filename,
                            "bytes": remote.size, "sha256": digest})
            print(f"Verified {len(results)}/{len(metadata)} {repo}/{filename}", flush=True)
        refs = cache_repo / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(revision)
    marker = root / "clay-trellis2-manifest.json"
    marker.write_text(json.dumps({"bytes": total, "files": results}, indent=2) + "\n")
    print(f"Completed. Verified {len(results)} files", flush=True)
    return {
        "status": "ready", "files": len(results), "model_bytes": total,
        "manifest": str(marker.relative_to(root.parent.parent)),
    }


def handler(job):
    if job.get("input") != {"operation": "prefetch_trellis2"}:
        return {"error": "Expected input.operation=prefetch_trellis2"}
    volume = Path("/runpod-volume")
    if not volume.is_mount():
        raise RuntimeError("The authorized Global Volume is not mounted at /runpod-volume")
    root = volume / "clay" / "trellis2-cache"
    root.mkdir(parents=True, exist_ok=True)
    dino_snapshot = volume / "huggingface-cache" / "hub" / ("models--" + DINO.replace("/", "--")) / "snapshots" / REVISIONS[DINO]
    with tempfile.TemporaryDirectory(prefix="clay-model-transfer-") as directory:
        return stage(root, Path(directory), dino_snapshot)


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})

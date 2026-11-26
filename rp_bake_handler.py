"""RunPod CPU queue for Clay's Normal and AO bake."""

import logging
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from clay.pbr_bake import bake_pbr

GLB_CONTENT_TYPE = "model/gltf-binary"
MAX_HIGH_BYTES = 512 * 1024**2
MAX_LOW_BYTES = 128 * 1024**2
LOGGER = logging.getLogger(__name__)
# httpx INFO includes request URLs and their signatures.
logging.getLogger("httpx").setLevel(logging.WARNING)


def _urls(inputs):
    urls = [inputs.get(key) for key in ("high_url", "low_url", "put_url")]
    if not all(isinstance(url, str) and len(url) <= 4096 for url in urls):
        raise ValueError("Expected three signed R2 URLs")
    high, low, output = [urlsplit(url) for url in urls]
    for url in (high, low, output):
        if (url.scheme != "https" or not re.fullmatch(r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", url.netloc)
                or url.netloc != high.netloc or url.fragment or "X-Amz-Signature" not in parse_qs(url.query)):
            raise ValueError("Expected signed URLs on the same R2 account")
    name = re.fullmatch(r"(/[a-z0-9.-]+/clay)/bake-inputs/([a-f0-9]{32})/high\.glb", high.path)
    if not name:
        raise ValueError("Invalid Clay bake input path")
    prefix, asset = name.groups()
    if low.path != f"{prefix}/bake-inputs/{asset}/low.glb" or output.path != f"{prefix}/{asset}.glb":
        raise ValueError("Bake URLs must address the same Clay asset")
    return urls


def _download(client, url, path, limit):
    with client.stream("GET", url) as response, path.open("wb") as file:
        response.raise_for_status()
        size = 0
        for chunk in response.iter_bytes(chunk_size=1024**2):
            size += len(chunk)
            if size > limit:
                raise ValueError("Clay bake input exceeds the configured size limit")
            file.write(chunk)


def handler(job):
    started = time.perf_counter()
    stage = "validate handoff"
    try:
        inputs = job.get("input")
        if not isinstance(inputs, dict):
            raise ValueError("Expected an input object")
        high_url, low_url, put_url = _urls(inputs)
        with tempfile.TemporaryDirectory(prefix="clay-bake-job-") as work, httpx.Client(timeout=90) as client:
            directory = Path(work)
            high, low, output = [directory / name for name in ("high.glb", "low.glb", "finished.glb")]
            stage = "download high mesh"
            _download(client, high_url, high, MAX_HIGH_BYTES)
            stage = "download low mesh"
            _download(client, low_url, low, MAX_LOW_BYTES)
            stage = "bake Normal and AO"
            bake_pbr(high, low, output)
            size = output.stat().st_size
            if size > MAX_LOW_BYTES:
                raise ValueError("Baked GLB exceeds the delivery size limit")
            stage = "publish GLB"
            with output.open("rb") as file:
                client.put(put_url, content=file, headers={"Content-Type": GLB_CONTENT_TYPE,
                                                          "Content-Length": str(size)}).raise_for_status()
        return {"format": "glb", "size_bytes": size, "bake_job_s": round(time.perf_counter()-started, 3)}
    except Exception as error:
        # HTTP errors can contain signed URLs. Keep only the exception class in logs.
        LOGGER.error("Clay PBR bake failed during %s: %s", stage, type(error).__name__)
        return {"error": "Clay PBR bake failed. Check the CPU worker logs."}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})

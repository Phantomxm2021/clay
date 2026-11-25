"""Runpod queue adapter for Clay's TRELLIS image and text pipelines."""

import base64
import binascii
import gc
import io

from PIL import Image, UnidentifiedImageError

PROVIDER = "trellis2"
MAX_INLINE_BYTES = 7 * 1024 * 1024
MAX_BASE64_CHARS = 4 * ((MAX_INLINE_BYTES + 2) // 3)
MAX_IMAGE_PIXELS = 16_000_000

_active_mode = None


def _image_payload(value):
    if not isinstance(value, str):
        raise ValueError("image_b64 must be a base64 string")
    if value.startswith("data:image/"):
        header, _, value = value.partition(",")
        if not header.endswith(";base64"):
            raise ValueError("Expected a base64 image data URL")
    if not value or len(value) > MAX_BASE64_CHARS:
        raise ValueError("Image payload is empty or exceeds the 7 MiB limit")
    try:
        raw = base64.b64decode(value, validate=True)
        if len(raw) > MAX_INLINE_BYTES:
            raise ValueError("Image exceeds the 7 MiB limit")
        with Image.open(io.BytesIO(raw), formats=("PNG", "JPEG", "WEBP")) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit")
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Expected a valid PNG, JPEG or WebP image encoded in base64") from exc
    return value


def handler(job):
    global _active_mode

    inputs = job.get("input")
    if not isinstance(inputs, dict):
        return {"error": "Expected an input object"}
    if inputs.get("provider", PROVIDER) != PROVIDER or inputs.get("format", "glb") != "glb":
        return {"error": "This worker supports provider trellis2 and format glb"}

    image = inputs.get("image_b64", inputs.get("image_base64"))
    prompt = inputs.get("prompt")
    target = inputs.get("target_tris", inputs.get("target_triangles", 30000))
    seed = inputs.get("seed", 42)
    try:
        if (image is None) == (prompt is None):
            raise ValueError("Provide either image_b64 or prompt")
        if type(target) is not int or not 1000 <= target <= 200000:
            raise ValueError("target_tris must be an integer from 1000 to 200000")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if image is not None:
            mode = "image"
            image = _image_payload(image)
        else:
            mode = "text"
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
                raise ValueError("prompt must be a nonempty string of up to 4096 characters")
    except ValueError as exc:
        return {"error": str(exc)}

    from clay.gpu_backend import runtime

    # Runpod's default queue invokes one job per worker. Keep only the model
    # used by the current mode resident so alternating jobs can fit on one GPU.
    if _active_mode is not None and _active_mode != mode:
        import torch

        runtime._load_trellis.cache_clear()
        runtime._load_trellis_text.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
    _active_mode = mode

    mesh, triangles = runtime.generate(
        PROVIDER,
        mode,
        image_b64=image,
        prompt=prompt,
        fmt="glb",
        target_tris=target,
        seed=seed,
    )
    if not mesh.startswith(b"glTF") or triangles <= 0:
        raise RuntimeError("Clay returned an invalid or empty GLB")
    if len(mesh) > MAX_INLINE_BYTES:
        return {"error": "GLB exceeds the 7 MiB inline limit. Retry with a smaller target_tris."}
    return {
        "mesh_b64": base64.b64encode(mesh).decode("ascii"),
        "format": "glb",
        "triangles": triangles,
        "textures": [],
        "size_bytes": len(mesh),
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})

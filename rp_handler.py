"""Runpod queue adapter for Clay's TRELLIS image and text pipelines."""

import base64
import binascii
import gc
import io
import logging
import os
import time
import uuid

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
PROVIDER = "trellis2"
GLB_CONTENT_TYPE = "model/gltf-binary"
DOWNLOAD_URL_TTL = 3600
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


def _upload_glb(mesh, bucket, endpoint_url, job_id):
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "auto",
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "total_max_attempts": 3},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    if isinstance(job_id, str) and job_id:
        asset_id = uuid.uuid5(uuid.NAMESPACE_URL, f"clay-runpod:{job_id}").hex
    else:
        asset_id = uuid.uuid4().hex
    object_key = f"clay/{asset_id}.glb"
    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=mesh,
        ContentType=GLB_CONTENT_TYPE,
    )
    glb_url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=DOWNLOAD_URL_TTL,
    )
    return {"asset_id": asset_id, "object_key": object_key, "glb_url": glb_url}


def handler(job):
    inputs = job.get("input")
    if not isinstance(inputs, dict):
        return {"error": "Expected an input object"}
    if inputs.get("provider", PROVIDER) != PROVIDER or inputs.get("format", "glb") != "glb":
        return {"error": "This worker supports provider trellis2 and format glb"}

    image = inputs.get("image_b64", inputs.get("image_base64"))
    prompt = inputs.get("prompt")
    target = inputs.get("target_tris", inputs.get("target_triangles", 30000))
    seed = inputs.get("seed", 42)
    output_format = inputs.get("output_format", "auto")
    bucket = os.environ.get("CLAY_OUTPUT_BUCKET")
    upload = output_format == "url" or (output_format == "auto" and bool(bucket))
    endpoint_url = os.environ.get("CLAY_OUTPUT_ENDPOINT_URL")
    try:
        if output_format not in ("auto", "url", "base64"):
            raise ValueError("output_format must be auto, url, or base64")
        if upload and not (
            bucket
            and endpoint_url
            and os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        ):
            raise ValueError("Object storage is missing bucket, endpoint, or credentials")
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

    if os.environ.get("CLAY_IMAGE_BACKEND") == "sprint3d":
        from rp_process import generate

        backend = "trellis2" if mode == "image" else "legacy"
        mesh, triangles, metrics = generate(
            backend, mode, image=image, prompt=prompt, target=target, seed=seed,
        )
    else:
        mesh, triangles = _generate_legacy(mode, image, prompt, target, seed)
        metrics = {}

    if not mesh.startswith(b"glTF") or triangles <= 0:
        raise RuntimeError("Clay returned an invalid or empty GLB")
    result = {
        "format": "glb",
        "triangles": triangles,
        "textures": [],
        "size_bytes": len(mesh),
    }
    if metrics:
        result["metrics"] = metrics
    if upload:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            started = time.perf_counter()
            result.update(_upload_glb(mesh, bucket, endpoint_url, job.get("id")))
            if metrics:
                metrics["r2_upload_s"] = round(time.perf_counter() - started, 3)
        except ClientError as exc:
            LOGGER.error("GLB upload failed with S3 error %s", exc.response.get("Error", {}).get("Code"))
            return {"error": "GLB upload to object storage failed. See worker logs."}
        except BotoCoreError as exc:
            LOGGER.error("GLB upload failed with %s", type(exc).__name__)
            return {"error": "GLB upload to object storage failed. See worker logs."}
        return result
    if len(mesh) > MAX_INLINE_BYTES:
        return {"error": "GLB exceeds the 7 MiB inline limit. Use URL output."}
    result["mesh_b64"] = base64.b64encode(mesh).decode("ascii")
    return result


def _generate_legacy(mode, image, prompt, target, seed):
    global _active_mode
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

    return runtime.generate(
        PROVIDER,
        mode,
        image_b64=image,
        prompt=prompt,
        fmt="glb",
        target_tris=target,
        seed=seed,
    )


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})

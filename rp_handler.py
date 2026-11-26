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
VIEW_NAMES = ("front", "side", "back")

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


def _view_payloads(value):
    if (not isinstance(value, dict) or "front" not in value or len(value) < 2
            or not value.keys() <= set(VIEW_NAMES)):
        raise ValueError("views_b64 needs front and at least one of side or back")
    views = {name: _image_payload(value[name]) for name in VIEW_NAMES if name in value}
    raw_size = sum(len(data) // 4 * 3 - (len(data) - len(data.rstrip("="))) for data in views.values())
    if raw_size > MAX_INLINE_BYTES:
        raise ValueError("Reference images together exceed the 7 MiB limit")
    return views


def _storage_client(endpoint_url):
    import boto3
    from botocore.config import Config

    return boto3.client(
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


def _upload_glb(mesh, bucket, endpoint_url, job_id, source=None):
    client = _storage_client(endpoint_url)
    asset_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"clay-runpod:{job_id}") if job_id else uuid.uuid4()
    object_key = f"clay/{asset_uuid.hex}.glb"

    def signed(operation, key, **headers):
        return client.generate_presigned_url(operation, Params={"Bucket": bucket, "Key": key, **headers},
                                             ExpiresIn=DOWNLOAD_URL_TTL)

    def put(key, body):
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=GLB_CONTENT_TYPE)

    result = {"asset_id": asset_uuid.hex, "object_key": object_key}
    if source is None:
        put(object_key, mesh)
        result["glb_url"] = signed("get_object", object_key)
        return result

    # Bake retries always read the original low mesh, including after a successful final PUT.
    staging = f"clay/bake-inputs/{asset_uuid.hex}"
    high_key, low_key = (f"{staging}/{name}.glb" for name in ("high", "low"))
    put(high_key, source)
    put(low_key, mesh)
    result["pbr_bake"] = {
        "high_url": signed("get_object", high_key), "low_url": signed("get_object", low_key),
        "put_url": signed("put_object", object_key, ContentType=GLB_CONTENT_TYPE),
        "glb_url": signed("get_object", object_key),
    }
    return result


def handler(job):
    inputs = job.get("input")
    if not isinstance(inputs, dict):
        return {"error": "Expected an input object"}
    if inputs.get("provider", PROVIDER) != PROVIDER or inputs.get("format", "glb") != "glb":
        return {"error": "This worker supports provider trellis2 and format glb"}

    image = inputs.get("image_b64", inputs.get("image_base64"))
    views = inputs.get("views_b64")
    prompt = inputs.get("prompt")
    target = inputs.get("target_tris", inputs.get("target_triangles", 30000))
    seed = inputs.get("seed", 42)
    output_format = inputs.get("output_format", "auto")
    pbr_bake = inputs.get("pbr_bake", False)
    bucket = os.environ.get("CLAY_OUTPUT_BUCKET")
    upload = output_format == "url" or (output_format == "auto" and bool(bucket))
    endpoint_url = os.environ.get("CLAY_OUTPUT_ENDPOINT_URL")
    sprint3d = os.environ.get("CLAY_IMAGE_BACKEND") == "sprint3d"
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
        if sum(value is not None for value in (image, views, prompt)) != 1:
            raise ValueError("Provide exactly one of image_b64, views_b64 or prompt")
        if views is not None and not sprint3d:
            raise ValueError("Multi-view requests need the TRELLIS.2 sprint3d backend")
        if type(pbr_bake) is not bool:
            raise ValueError("pbr_bake must be a boolean")
        if pbr_bake and (prompt is not None or not upload or not sprint3d):
            raise ValueError("PBR baking needs TRELLIS.2 image generation with R2 storage")
        if type(target) is not int or not 1000 <= target <= 200000:
            raise ValueError("target_tris must be an integer from 1000 to 200000")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if views is not None:
            mode = "image"
            views = _view_payloads(views)
        elif image is not None:
            mode = "image"
            image = _image_payload(image)
        else:
            mode = "text"
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
                raise ValueError("prompt must be a nonempty string of up to 4096 characters")
    except ValueError as exc:
        return {"error": str(exc)}

    if mode == "text" and os.environ.get("CLAY_TRELLIS2_IMAGE_ONLY") == "1":
        return {"error": "This Clay endpoint accepts image requests"}

    source = None
    if sprint3d:
        from rp_process import generate

        backend = "trellis2" if mode == "image" else "legacy"
        mesh, triangles, metrics, source = generate(
            backend, mode, image=image, prompt=prompt, target=target, seed=seed, pbr_bake=pbr_bake,
            views=views,
        )
        if pbr_bake and not source:
            raise RuntimeError("Clay did not produce a PBR bake source")
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
            result.update(_upload_glb(mesh, bucket, endpoint_url, job.get("id"), source=source))
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


def main():
    import runpod

    if os.environ.get("CLAY_IMAGE_BACKEND") == "sprint3d" and os.environ.get("CLAY_TRELLIS2_PRELOAD") == "1":
        from rp_process import preload_trellis2

        try:
            metrics = preload_trellis2()
            LOGGER.info("Clay TRELLIS.2 preloaded in %.3fs (process %s)",
                        metrics["model_preload_s"], metrics["model_process_boot_id"])
        except (OSError, ValueError, RuntimeError) as error:
            LOGGER.warning("Clay TRELLIS.2 preload failed (%s); continuing with lazy load", type(error).__name__)
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()

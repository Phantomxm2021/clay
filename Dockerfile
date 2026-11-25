FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS build

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git ninja-build \
    && rm -rf /var/lib/apt/lists/*

ARG TRELLIS_REF=442aa1e1afb9014e80681d3bf604e8d728a86ee7
ARG TORCH_CUDA_ARCH_LIST=8.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=4 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/clay-venv/bin:$PATH

# Match Clay's Modal GPU image. Clay is imported from source so the package
# metadata's Python and Torch minimums do not replace the validated CUDA stack.
RUN uv python install 3.11 && uv venv --python 3.11 --seed /opt/clay-venv \
    && python -m pip install --no-cache-dir 'setuptools<81' wheel packaging \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        'torch==2.4.0' 'torchvision==0.19.0' 'xformers==0.0.27.post2'

RUN python -m pip install --no-cache-dir \
    'numpy<2' 'opencv-python-headless<4.12' 'rembg==2.0.65' \
    'transformers==4.46.3' 'huggingface-hub>=0.28,<1' 'pymeshfix<0.18' \
    pillow imageio imageio-ffmpeg tqdm easydict scipy ninja onnxruntime \
    trimesh xatlas pyvista igraph safetensors einops open3d fastapi \
    fast-simplification pygltflib 'runpod==1.8.1' spconv-cu120 \
    'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8' \
    && python -m pip install --no-cache-dir 'kaolin==0.17.0' \
        --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

RUN git init /opt/trellis \
    && git -C /opt/trellis remote add origin https://github.com/microsoft/TRELLIS.git \
    && git -C /opt/trellis fetch --depth 1 origin "${TRELLIS_REF}" \
    && git -C /opt/trellis checkout --detach FETCH_HEAD \
    && git -C /opt/trellis submodule update --init --recursive --depth 1

RUN python -m pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0' \
    && git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast \
    && python -m pip install --no-cache-dir --no-build-isolation /tmp/diffoctreerast \
    && git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting \
    && python -m pip install --no-cache-dir --no-build-isolation \
        /tmp/mip-splatting/submodules/diff-gaussian-rasterization \
    && rm -rf /tmp/diffoctreerast /tmp/mip-splatting

FROM build AS trellis-export
COPY trellis-glb-material.patch /tmp/trellis-glb-material.patch
RUN git -C /opt/trellis apply /tmp/trellis-glb-material.patch

FROM nvidia/cuda:12.1.1-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    CUDA_MODULE_LOADING=LAZY \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    U2NET_HOME=/models/u2net \
    PATH=/opt/clay-venv/bin:$PATH \
    PYTHONPATH=/app/src:/opt/trellis \
    PYTHONUNBUFFERED=1

# The CUDA extensions are already compiled. Only the Python runtime, wheels,
# and TRELLIS source move into the worker image.
COPY --from=build /opt/ /opt/
COPY --from=trellis-export /opt/trellis/trellis/utils/postprocessing_utils.py /opt/trellis/trellis/utils/postprocessing_utils.py

WORKDIR /app
COPY src /app/src
COPY rp_handler.py /app/rp_handler.py
RUN python -m pip check \
    && python -c 'import torch, nvdiffrast.torch, diffoctreerast, diff_gaussian_rasterization; import trellis.pipelines, trellis.utils.postprocessing_utils, clay.gpu_backend.runtime, rp_handler; assert "error" in rp_handler.handler({"input": {}})'

CMD ["python", "-u", "rp_handler.py"]

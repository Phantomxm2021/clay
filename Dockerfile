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
COPY patches/trellis-glb-material.patch /tmp/trellis-glb-material.patch
RUN git -C /opt/trellis apply /tmp/trellis-glb-material.patch

FROM nvidia/cuda:12.1.1-base-ubuntu22.04 AS legacy

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

# TRELLIS.2's CUDA 12.8 stack is isolated from the first-generation text model.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS sprint3d-build

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/trellis2-python \
    PATH=/opt/trellis2-venv/bin:$PATH \
    MAX_JOBS=6
ARG SPRINT3D_REF=1053b33b4564c7d28a9afe31b711ba660c210204
ARG TRELLIS2_REF=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG TORCH_CUDA_ARCH_LIST=8.6+PTX

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git ninja-build libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.10 && uv venv --python 3.10 --seed /opt/trellis2-venv \
    && python -m pip install --no-cache-dir 'setuptools>=77,<81' wheel packaging \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128'

RUN git init -q /opt/sprint3d \
    && git -C /opt/sprint3d fetch --depth 1 https://github.com/AlayaLab/sprint3d.git "$SPRINT3D_REF" \
    && git -C /opt/sprint3d checkout --detach FETCH_HEAD \
    && git init -q /opt/TRELLIS.2 \
    && git -C /opt/TRELLIS.2 fetch --depth 1 https://github.com/microsoft/TRELLIS.2.git "$TRELLIS2_REF" \
    && git -C /opt/TRELLIS.2 checkout --detach FETCH_HEAD \
    && git -C /opt/TRELLIS.2 submodule update --init --recursive --depth 1 \
    && python -m pip install --no-cache-dir --no-build-isolation /opt/sprint3d \
        'omegaconf==2.3.0' \
        'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps \
      'git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e' \
      'git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f' \
      'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae' \
      'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8' \
      /opt/TRELLIS.2/o-voxel \
    && WORK_DIR=/tmp/sprint3d-build bash /opt/sprint3d/native/build_native.sh \
    && rm -rf /tmp/sprint3d-build

# Native packages declare these as Python dependencies even though Ninja is also installed by apt.
RUN python -m pip install --no-cache-dir \
        'kiui==0.3.5' 'ninja==1.13.0' 'pybind11==3.1.0' 'moderngl==5.12.0' \
    && python -m pip check \
    && python -c 'from importlib.util import find_spec; assert all(find_spec(m) for m in ("flash_attn", "flex_gemm", "cumesh", "o_voxel", "nvdiffrast", "sprint3d_cubvh", "sprint3d_cumesh"))'

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc libegl1 libgl1 libgles2 libglib2.0-0 libgomp1 libusb-1.0-0 \
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

COPY --from=legacy /opt/ /opt/
COPY --from=sprint3d-build /opt/trellis2-python/ /opt/trellis2-python/
COPY --from=sprint3d-build /opt/trellis2-venv/ /opt/trellis2-venv/
COPY --from=sprint3d-build /opt/TRELLIS.2/trellis2/ /opt/TRELLIS.2/trellis2/
COPY --from=sprint3d-build /opt/TRELLIS.2/LICENSE /opt/TRELLIS.2/LICENSE
COPY --from=sprint3d-build /opt/sprint3d/experiments/ /opt/sprint3d/experiments/

WORKDIR /app
COPY src /app/src
COPY rp_handler.py rp_process.py rp_inference.py /app/
RUN python -c 'import torch; import trellis.pipelines, trellis.utils.postprocessing_utils, rp_handler; assert "error" in rp_handler.handler({"input": {}})' \
    && /opt/trellis2-venv/bin/python -c 'import torch, flash_attn; assert torch.version.cuda == "12.8"'

CMD ["python", "-u", "rp_handler.py"]

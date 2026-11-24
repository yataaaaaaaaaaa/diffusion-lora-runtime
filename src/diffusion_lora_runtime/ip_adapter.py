"""IP-Adapter pieces that do not need a loaded pipeline: the CLIP ViT-H/14 image encoder configuration, image encoder
loading (ComfyUI single file or transformers folder), reference image opening and the pipeline family labels.

DiffusionRuntime.ensure_ip_adapter (diffusion_runtime.py) owns loading the adapter into a pipeline. This module is
torch-free at import time: transformers, safetensors.torch and accelerate are imported inside the functions that need
them (tests replace them with stubs).
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from PIL import Image

from .checkpoint_inspection import SAFETENSORS_SUFFIX
from .schemas import PipelineArchitectureName

# config.json of h94/IP-Adapter models/image_encoder (OpenCLIP ViT-H/14 laion2B-s32B-b79K, the image encoder of every
# "vit-h" IP-Adapter, SD 1.5 and SDXL alike) without its metadata keys (_name_or_path, architectures, torch_dtype,
# transformers_version). Single-file encoders (ComfyUI clip_vision files) carry no configuration: they are built with it.
VIT_H_14_IMAGE_ENCODER_CONFIG: Mapping[str, Any] = MappingProxyType(
    {
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "image_size": 224,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 5120,
        "layer_norm_eps": 1e-05,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 32,
        "patch_size": 14,
        "projection_dim": 1024,
    }
)

# A buffer some single files store (e.g. ComfyUI's CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors) but transformers
# rebuilds itself (non-persistent): dropped when the model does not expect it.
IMAGE_ENCODER_BUFFER_KEYS = frozenset({"vision_model.embeddings.position_ids"})

PIPELINE_ARCHITECTURE_LABELS: Mapping[PipelineArchitectureName, str] = MappingProxyType(
    {"sd15": "SD 1.5", "sdxl": "SDXL"}
)


class ReferenceImageError(ValueError):
    """A reference image of an IP-Adapter request is missing or cannot be read (or none was given)."""


class ImageEncoderLayoutError(ValueError):
    """A single-file image encoder is not a CLIP ViT-H/14 vision model in the transformers key layout."""


def architecture_mismatch_message(adapter: PipelineArchitectureName, pipeline: PipelineArchitectureName) -> str:
    """Why an IP-Adapter made for one pipeline family cannot be loaded into a pipeline of another one."""
    return (
        f"IP-Adapter made for {PIPELINE_ARCHITECTURE_LABELS[adapter]} cannot be used with an "
        f"{PIPELINE_ARCHITECTURE_LABELS[pipeline]} model."
    )


def open_reference_images(paths: Iterable[str | Path]) -> list[Image.Image]:
    """The reference images as RGB PIL images, in order (fully read: no file handle stays open).

    Raises ReferenceImageError (a ValueError) when no path is given, or a file is missing or not a readable image.
    """
    images: list[Image.Image] = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            raise ReferenceImageError(f"Reference image not found: {path}")
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ReferenceImageError(f"Reference image cannot be read: {path} ({exc})") from exc
    if not images:
        raise ReferenceImageError("IP-Adapter conditioning needs at least one reference image.")
    return images


def load_image_encoder(path: str | Path, *, torch_dtype: Any, device: str) -> Any:
    """The ``transformers.CLIPVisionModelWithProjection`` an IP-Adapter encodes its reference images with, in eval
    mode, converted to ``torch_dtype`` and placed on ``device``.

    - a ``.safetensors`` FILE in the transformers key layout (``vision_model.*``, ``visual_projection.weight``; the
      ComfyUI ``clip_vision`` files) is built with VIT_H_14_IMAGE_ENCODER_CONFIG and loaded strictly (buffers listed
      in IMAGE_ENCODER_BUFFER_KEYS are tolerated). The model is created without allocating weights (accelerate
      ``init_empty_weights``) and takes the file's tensors, so the 2.5 GB encoder is held in memory once;
    - a FOLDER is a transformers model folder (``config.json`` + weights): ``from_pretrained``.

    Raises FileNotFoundError for a missing path, ImageEncoderLayoutError for a single file of another layout and
    ValueError for a single file that is not ``.safetensors``.
    """
    encoder_path = Path(path)
    if encoder_path.is_dir():
        from transformers import CLIPVisionModelWithProjection

        encoder = CLIPVisionModelWithProjection.from_pretrained(str(encoder_path), torch_dtype=torch_dtype)
    elif encoder_path.is_file():
        if encoder_path.suffix.casefold() != SAFETENSORS_SUFFIX:
            raise ValueError(f"Single-file image encoders must be {SAFETENSORS_SUFFIX} files: {encoder_path}")
        encoder = _load_single_file_image_encoder(encoder_path)
    else:
        raise FileNotFoundError(f"Image encoder not found: {encoder_path}")
    encoder = encoder.to(device=device, dtype=torch_dtype)
    encoder.eval()
    return encoder


def _load_single_file_image_encoder(path: Path) -> Any:
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    with init_empty_weights():
        encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(**VIT_H_14_IMAGE_ENCODER_CONFIG))
    state_dict = load_file(str(path), device="cpu")
    expected = set(encoder.state_dict())
    for key in IMAGE_ENCODER_BUFFER_KEYS - expected:
        state_dict.pop(key, None)
    missing = sorted(expected - set(state_dict))
    unexpected = sorted(set(state_dict) - expected)
    if missing or unexpected:
        raise ImageEncoderLayoutError(
            f"{path} is not a CLIP ViT-H/14 image encoder in the transformers layout "
            f"({len(missing)} missing keys {missing[:3]}, {len(unexpected)} unexpected keys {unexpected[:3]})."
        )
    # assign=True: the parameters become the file's tensors (the empty model allocated none).
    encoder.load_state_dict(state_dict, strict=True, assign=True)
    return encoder


__all__ = [
    "IMAGE_ENCODER_BUFFER_KEYS",
    "PIPELINE_ARCHITECTURE_LABELS",
    "VIT_H_14_IMAGE_ENCODER_CONFIG",
    "ImageEncoderLayoutError",
    "ReferenceImageError",
    "architecture_mismatch_message",
    "load_image_encoder",
    "open_reference_images",
]

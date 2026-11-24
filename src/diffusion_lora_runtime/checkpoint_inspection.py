"""Single-file checkpoint inspection: which Stable Diffusion family a ``.safetensors`` checkpoint belongs to,
read from its header only (no tensor is loaded and torch is not imported).

A safetensors file starts with an unsigned 64-bit little-endian header length, followed by that many bytes of
JSON mapping every tensor name to its dtype/shape/offsets (plus an optional ``__metadata__`` entry). The tensor
names of the original Stability AI (ldm) layout identify the family:

- SDXL (base, Pony, Illustrious, NoobAI, ...): ``conditioner.embedders.1.*`` (its second, OpenCLIP bigG text
  encoder) or ``model.diffusion_model.label_emb.*`` (the UNet's size/crop conditioning);
- SD 1.x: ``cond_stage_model.transformer.*`` (a single CLIP ViT-L text encoder).

Anything else (SD 2.x, Flux, diffusers-layout files, ``.ckpt`` pickles, unreadable files) is ``None``.

``pipeline_architecture`` is the single owner of the rule "which pipeline family does the runtime load this model
as" (also used by consumers before any model is loaded, e.g. to pick an IP-Adapter made for that family).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Literal

SingleFileArchitecture = Literal["sd15", "sdxl"]

SAFETENSORS_SUFFIX = ".safetensors"
# Same bound as the safetensors library: a larger header is not a valid file.
MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
SAFETENSORS_METADATA_KEY = "__metadata__"

# Diffusers model folders: model_index.json "_class_name" prefixes of the families the runtime knows.
DIFFUSERS_MODEL_INDEX = "model_index.json"
SDXL_PIPELINE_CLASS_PREFIX = "StableDiffusionXL"
SD1_PIPELINE_CLASS_NAMES = frozenset({"StableDiffusionPipeline"})

SDXL_KEY_PREFIXES = ("conditioner.embedders.1.", "model.diffusion_model.label_emb.")
SD1_KEY_PREFIXES = ("cond_stage_model.transformer.",)


class SafetensorsHeaderError(ValueError):
    """The file is not a readable safetensors file."""


def read_safetensors_keys(path: str | Path) -> list[str]:
    """Tensor names of a safetensors file, in header order, read without loading any tensor.

    Raises SafetensorsHeaderError for a truncated or malformed header and OSError when the file cannot be read.
    """
    with open(path, "rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise SafetensorsHeaderError(f"{path}: file too short for a safetensors header.")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length > MAX_SAFETENSORS_HEADER_BYTES:
            raise SafetensorsHeaderError(f"{path}: safetensors header length {header_length} is too large.")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise SafetensorsHeaderError(f"{path}: truncated safetensors header.")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError(f"{path}: safetensors header is not JSON ({exc}).") from exc
    if not isinstance(header, dict):
        raise SafetensorsHeaderError(f"{path}: safetensors header is not a JSON object.")
    return [key for key in header if key != SAFETENSORS_METADATA_KEY]


def detect_single_file_architecture(path: str | Path) -> SingleFileArchitecture | None:
    """``"sdxl"`` or ``"sd15"`` for a single-file checkpoint of that family (see the module docstring), else None
    (not a ``.safetensors`` file, unreadable, or an unknown family)."""
    checkpoint = Path(path)
    if checkpoint.suffix.casefold() != SAFETENSORS_SUFFIX:
        return None
    try:
        keys = read_safetensors_keys(checkpoint)
    except (OSError, SafetensorsHeaderError):
        return None
    if any(key.startswith(SDXL_KEY_PREFIXES) for key in keys):
        return "sdxl"
    if any(key.startswith(SD1_KEY_PREFIXES) for key in keys):
        return "sd15"
    return None


def single_file_pipeline_architecture(path: str | Path) -> SingleFileArchitecture:
    """Family of the pipeline a single-file checkpoint loads as: ``"sd15"`` for SD 1.x files (StableDiffusionPipeline),
    ``"sdxl"`` for everything else (SDXL, Pony, Illustrious, NoobAI, and any file whose family cannot be detected keep
    loading with StableDiffusionXLPipeline)."""
    return "sd15" if detect_single_file_architecture(path) == "sd15" else "sdxl"


def pipeline_architecture(path: str | Path) -> SingleFileArchitecture | None:
    """Family of the pipeline the runtime loads ``path`` as, without loading anything (torch-free):

    - an existing file: ``single_file_pipeline_architecture``;
    - a Diffusers model folder: from the ``_class_name`` of its ``model_index.json`` (``StableDiffusionXL*`` ->
      ``"sdxl"``, ``StableDiffusionPipeline`` -> ``"sd15"``); None when the file is missing, unreadable or names
      another family (SD3, Flux, ...);
    - a missing path: None.
    """
    model = Path(path)
    if model.is_file():
        return single_file_pipeline_architecture(model)
    if not model.is_dir():
        return None
    try:
        index = json.loads((model / DIFFUSERS_MODEL_INDEX).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    class_name = index.get("_class_name") if isinstance(index, dict) else None
    if not isinstance(class_name, str):
        return None
    if class_name.startswith(SDXL_PIPELINE_CLASS_PREFIX):
        return "sdxl"
    if class_name in SD1_PIPELINE_CLASS_NAMES:
        return "sd15"
    return None

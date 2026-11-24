"""Single-file checkpoint inspection: which Stable Diffusion family a ``.safetensors`` checkpoint belongs to,
read from its header only (no tensor is loaded and torch is not imported).

A safetensors file starts with an unsigned 64-bit little-endian header length, followed by that many bytes of
JSON mapping every tensor name to its dtype/shape/offsets (plus an optional ``__metadata__`` entry). The tensor
names of the original Stability AI (ldm) layout identify the family:

- SDXL (base, Pony, Illustrious, NoobAI, ...): ``conditioner.embedders.1.*`` (its second, OpenCLIP bigG text
  encoder) or ``model.diffusion_model.label_emb.*`` (the UNet's size/crop conditioning);
- SD 1.x: ``cond_stage_model.transformer.*`` (a single CLIP ViT-L text encoder).

Anything else (SD 2.x, Flux, diffusers-layout files, ``.ckpt`` pickles, unreadable files) is ``None``.
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

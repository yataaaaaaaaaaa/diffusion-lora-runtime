"""Tiny handcrafted single-file checkpoints for the tests (valid safetensors files of a few bytes)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

# Real tensor names of the original Stability AI (ldm) checkpoint layouts.
SD15_KEYS = [
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "first_stage_model.decoder.conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
]
SDXL_KEYS = [
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.1.model.ln_final.weight",
    "first_stage_model.decoder.conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.label_emb.0.0.weight",
]
SD21_KEYS = [
    "cond_stage_model.model.ln_final.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
]
FLUX_KEYS = ["double_blocks.0.img_attn.norm.key_norm.scale", "img_in.weight"]


def write_safetensors(path: Path, keys: list[str], *, metadata: dict[str, str] | None = None) -> Path:
    """A tiny but valid safetensors file: one 1-element F16 tensor (2 zero bytes) per key."""
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    for index, key in enumerate(keys):
        header[key] = {"dtype": "F16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]}
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(2 * len(keys)))
    return path

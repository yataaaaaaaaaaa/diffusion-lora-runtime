from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from diffusion_lora_runtime import (
    SafetensorsHeaderError,
    detect_single_file_architecture,
    pipeline_architecture,
    read_safetensors_keys,
    single_file_pipeline_architecture,
)
from diffusion_lora_runtime.checkpoint_inspection import MAX_SAFETENSORS_HEADER_BYTES

from checkpoint_files import FLUX_KEYS, SD15_KEYS, SD21_KEYS, SDXL_KEYS, write_safetensors
from conftest import SRC_DIR


def write_raw(path: Path, header_length: int, header_bytes: bytes) -> Path:
    path.write_bytes(struct.pack("<Q", header_length) + header_bytes)
    return path


def test_read_safetensors_keys_in_header_order_without_metadata(tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "model.safetensors", SDXL_KEYS, metadata={"modelspec.title": "x"})

    assert read_safetensors_keys(path) == SDXL_KEYS
    assert read_safetensors_keys(str(path)) == SDXL_KEYS


def test_read_safetensors_keys_of_an_empty_file(tmp_path: Path) -> None:
    assert read_safetensors_keys(write_safetensors(tmp_path / "empty.safetensors", [])) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\x01\x02", "too short"),
        (struct.pack("<Q", 50) + b"{}", "truncated"),
        (struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES + 1), "too large"),
        (struct.pack("<Q", 5) + b"{nope", "not JSON"),
        (struct.pack("<Q", 2) + b"\xff\xfe", "not JSON"),
        (struct.pack("<Q", 2) + b"[]", "not a JSON object"),
    ],
)
def test_read_safetensors_keys_rejects_malformed_headers(tmp_path: Path, content: bytes, message: str) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(content)

    with pytest.raises(SafetensorsHeaderError, match=message):
        read_safetensors_keys(path)


def test_read_safetensors_keys_of_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_safetensors_keys(tmp_path / "missing.safetensors")


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (SD15_KEYS, "sd15"),
        (SDXL_KEYS, "sdxl"),
        # Either SDXL marker alone is enough (e.g. pruned files or UNet-only merges).
        (["conditioner.embedders.1.model.ln_final.weight"], "sdxl"),
        (["model.diffusion_model.label_emb.0.0.weight", "model.diffusion_model.out.2.weight"], "sdxl"),
        (SD21_KEYS, None),
        (FLUX_KEYS, None),
        ([], None),
    ],
)
def test_detect_single_file_architecture(tmp_path: Path, keys: list[str], expected: str | None) -> None:
    path = write_safetensors(tmp_path / "checkpoint.safetensors", keys)

    assert detect_single_file_architecture(path) == expected


def test_detection_ignores_the_suffix_case(tmp_path: Path) -> None:
    assert detect_single_file_architecture(write_safetensors(tmp_path / "MODEL.SafeTensors", SD15_KEYS)) == "sd15"


def test_non_safetensors_and_unreadable_files_are_unknown(tmp_path: Path) -> None:
    # A pickle checkpoint is never parsed, even when its bytes would look like a safetensors header.
    assert detect_single_file_architecture(write_safetensors(tmp_path / "model.ckpt", SD15_KEYS)) is None
    assert detect_single_file_architecture(tmp_path / "missing.safetensors") is None
    assert detect_single_file_architecture(write_raw(tmp_path / "bad.safetensors", 10, b"{")) is None
    folder = tmp_path / "diffusers-model.safetensors"
    folder.mkdir()
    assert detect_single_file_architecture(folder) is None


def test_inspection_module_does_not_import_torch() -> None:
    """The module stays importable (and cheap) without torch: it only parses the JSON header."""
    module_path = SRC_DIR / "diffusion_lora_runtime" / "checkpoint_inspection.py"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('checkpoint_inspection', {str(module_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print('torch' in sys.modules, 'safetensors' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False False"


# -- pipeline_architecture: the family the runtime loads a model as ------------------------------------------------


@pytest.mark.parametrize(
    ("keys", "expected"),
    [(SD15_KEYS, "sd15"), (SDXL_KEYS, "sdxl"), (SD21_KEYS, "sdxl"), (FLUX_KEYS, "sdxl")],
    ids=["sd15", "sdxl", "undetected-sd2", "undetected-flux"],
)
def test_single_files_load_as_sd15_or_else_sdxl(tmp_path: Path, keys: list[str], expected: str) -> None:
    path = write_safetensors(tmp_path / "model.safetensors", keys)

    assert single_file_pipeline_architecture(path) == expected
    assert pipeline_architecture(path) == expected


def test_pickle_checkpoints_load_as_sdxl(tmp_path: Path) -> None:
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(b"not inspected")

    assert pipeline_architecture(path) == "sdxl"


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("StableDiffusionXLPipeline", "sdxl"),
        ("StableDiffusionXLImg2ImgPipeline", "sdxl"),
        ("StableDiffusionPipeline", "sd15"),
        ("StableDiffusion3Pipeline", None),
        ("FluxPipeline", None),
    ],
)
def test_diffusers_folders_use_their_model_index(tmp_path: Path, class_name: str, expected: str | None) -> None:
    folder = tmp_path / "diffusers-model"
    folder.mkdir()
    (folder / "model_index.json").write_text(f'{{"_class_name": "{class_name}"}}', encoding="utf-8")

    assert pipeline_architecture(folder) == expected


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]", '{"_class_name": 3}'], ids=["missing", "invalid", "list", "number"])
def test_unreadable_diffusers_folders_and_missing_paths_are_unknown(tmp_path: Path, content: str | None) -> None:
    folder = tmp_path / "diffusers-model"
    folder.mkdir()
    if content is not None:
        (folder / "model_index.json").write_text(content, encoding="utf-8")

    assert pipeline_architecture(folder) is None
    assert pipeline_architecture(tmp_path / "missing.safetensors") is None

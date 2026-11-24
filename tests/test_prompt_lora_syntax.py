"""Inline LoRA prompt syntax helpers (torch-free): extract_prompt_lora_specs, strip_prompt_lora_tags, and the lazy
package import that lets consumers use them without loading torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from diffusion_lora_runtime.prompt_lora import extract_prompt_lora_specs, strip_prompt_lora_tags

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("1girl, <lora:add_detail:0.6>, smile", [("add_detail", 0.6)]),
        ("<lora:add_detail:.2>, <lora:other:-1>", [("add_detail", 0.2), ("other", -1.0)]),
        ("<lora:Dramatic Lighting Slider:1> <lora:no_weight>", [("Dramatic Lighting Slider", 1.0), ("no_weight", 1.0)]),
        ("< LoRA : spaced : 0.5 >", [("spaced", 0.5)]),
        ("no tags at all", []),
        ("<lora:two:weights:0.5:0.2>", []),
    ],
)
def test_extract_prompt_lora_specs(prompt: str, expected: list[tuple[str, float]]) -> None:
    assert extract_prompt_lora_specs(prompt) == expected


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("1girl, <lora:add_detail:0.6>, smile", "1girl, smile"),
        ("<lora:a:1>,<lora:b:1>, castle, sky", "castle, sky"),
        ("castle <lora:a:1> <lora:broken:x:y>", "castle"),
        ("high quality <lora:add_detail:1> <lora:realistic filter [IL]:1> Realistic", "high quality   Realistic"),
        ("no tags", "no tags"),
    ],
)
def test_strip_prompt_lora_tags(prompt: str, expected: str) -> None:
    assert strip_prompt_lora_tags(prompt) == expected


def test_prompt_lora_imports_without_torch() -> None:
    """The backend reuses prompt_lora outside the runtime: importing it (and the package) must not load torch."""
    code = (
        "import sys\n"
        "import diffusion_lora_runtime\n"
        "import diffusion_lora_runtime.prompt_lora\n"
        "heavy = sorted(name for name in ('torch', 'diffusers', 'transformers') if name in sys.modules)\n"
        "assert not heavy, heavy\n"
        "assert 'strip_prompt_lora_tags' in dir(diffusion_lora_runtime.prompt_lora)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_DIR), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr


def test_lazy_public_names_resolve() -> None:
    import diffusion_lora_runtime

    assert diffusion_lora_runtime.LoRASpec(id="a", name="a", path="a.safetensors").weight == 1.0
    assert "DiffusionGenerator" in dir(diffusion_lora_runtime)
    with pytest.raises(AttributeError):
        getattr(diffusion_lora_runtime, "NotAName")

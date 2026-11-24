"""Textual inversion on single text-encoder (SD 1.x) and dual text-encoder (SDXL) pipelines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from diffusion_lora_runtime import PromptEmbeddingResolutionError
from diffusion_lora_runtime import diffusion_runtime as runtime_module
from diffusion_lora_runtime.diffusion_runtime import DiffusionRuntime
from diffusion_lora_runtime.prompt_embedding import PromptEmbeddingReference

from conftest import SRC_DIR

CLIP_L_WIDTH = 768
CLIP_G_WIDTH = 1280


class FakeEncoder:
    def __init__(self, width: int) -> None:
        self.width = width


class Recorder:
    def __init__(self) -> None:
        self.resolved: list[tuple[int, int | None]] = []
        self.registered: list[tuple[Any, Any, str, Any]] = []


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Replaces the tensor work (file loading, vector resolution, token registration) by recording fakes."""
    recorder = Recorder()

    def resolve(state: Any, *, clip_l_dim: int, clip_g_dim: int | None) -> SimpleNamespace:
        recorder.resolved.append((clip_l_dim, clip_g_dim))
        return SimpleNamespace(clip_l="clip_l-vectors", clip_g="clip_g-vectors")

    def register(tokenizer: Any, text_encoder: Any, base_token: str, vectors: Any) -> list[str]:
        recorder.registered.append((tokenizer, text_encoder, base_token, vectors))
        return [base_token, f"{base_token}_1"]

    monkeypatch.setattr(runtime_module, "load_prompt_embedding_file", lambda path: {"path": path})
    monkeypatch.setattr(runtime_module, "resolve_prompt_embedding_vectors", resolve)
    monkeypatch.setattr(DiffusionRuntime, "_encoder_embedding_width", lambda self, encoder: encoder.width)
    monkeypatch.setattr(DiffusionRuntime, "_register_textual_inversion_tokens", lambda self, *args: register(*args))
    return recorder


@pytest.fixture()
def runtime() -> DiffusionRuntime:
    return DiffusionRuntime(device="cpu", allow_cpu=True)


@pytest.fixture()
def reference(tmp_path: Path) -> PromptEmbeddingReference:
    return PromptEmbeddingReference(name="EasyNegative", path=tmp_path / "easynegative.safetensors")


def test_single_text_encoder_pipelines_register_clip_l_tokens_only(
    runtime: DiffusionRuntime, recorder: Recorder, reference: PromptEmbeddingReference
) -> None:
    encoder = FakeEncoder(CLIP_L_WIDTH)
    runtime.pipe = SimpleNamespace(tokenizer="tokenizer", text_encoder=encoder)

    tokens = runtime.ensure_textual_inversions([reference])

    assert tokens == ["dlr_embed_easynegative"]
    assert recorder.resolved == [(CLIP_L_WIDTH, None)]
    assert recorder.registered == [("tokenizer", encoder, "dlr_embed_easynegative", "clip_l-vectors")]
    assert runtime.expand_prompt_embeddings("worst, dlr_embed_easynegative") == (
        "worst, dlr_embed_easynegative dlr_embed_easynegative_1"
    )

    runtime.ensure_textual_inversions([reference])
    assert len(recorder.registered) == 1  # loaded once


def test_dual_text_encoder_pipelines_register_both_token_sets(
    runtime: DiffusionRuntime, recorder: Recorder, reference: PromptEmbeddingReference
) -> None:
    clip_l, clip_g = FakeEncoder(CLIP_L_WIDTH), FakeEncoder(CLIP_G_WIDTH)
    runtime.pipe = SimpleNamespace(
        tokenizer="tokenizer", text_encoder=clip_l, tokenizer_2="tokenizer_2", text_encoder_2=clip_g
    )

    assert runtime.ensure_textual_inversions([reference]) == ["dlr_embed_easynegative"]

    assert recorder.resolved == [(CLIP_L_WIDTH, CLIP_G_WIDTH)]
    assert recorder.registered == [
        ("tokenizer", clip_l, "dlr_embed_easynegative", "clip_l-vectors"),
        ("tokenizer_2", clip_g, "dlr_embed_easynegative", "clip_g-vectors"),
    ]


def test_pipelines_without_a_text_encoder_are_refused(
    runtime: DiffusionRuntime, recorder: Recorder, reference: PromptEmbeddingReference
) -> None:
    runtime.pipe = SimpleNamespace(unet=object())

    with pytest.raises(PromptEmbeddingResolutionError, match="tokenizer/text encoder pair"):
        runtime.ensure_textual_inversions([reference])
    assert recorder.resolved == []


# resolve_prompt_embedding_vectors needs real tensors: it runs in a subprocess with the real torch (skipped when
# torch is not installed). The package __init__ is bypassed so Diffusers is not imported.
VECTOR_CHECKS = """
import importlib.util, sys, types
if importlib.util.find_spec("torch") is None:
    print("NO_TORCH")
    raise SystemExit(0)
package = types.ModuleType("diffusion_lora_runtime")
package.__path__ = [{package_dir!r}]
sys.modules["diffusion_lora_runtime"] = package
import torch
from diffusion_lora_runtime.prompt_embedding import PromptEmbeddingResolutionError, resolve_prompt_embedding_vectors

def shapes(vectors):
    return tuple(vectors.clip_l.shape), tuple(vectors.clip_g.shape)

def refused(payload, **dims):
    try:
        resolve_prompt_embedding_vectors(payload, **dims)
    except PromptEmbeddingResolutionError as exc:
        return str(exc)
    raise AssertionError("not refused")

single = dict(clip_l_dim=768, clip_g_dim=None)
dual = dict(clip_l_dim=768, clip_g_dim=1280)
# A1111 SD 1.x embedding (e.g. EasyNegative): string_to_param {{"*": (tokens, 768)}}.
a1111 = {{"string_to_param": {{"*": torch.ones(8, 768)}}}}
assert shapes(resolve_prompt_embedding_vectors(a1111, **single)) == ((8, 768), (8, 0))
assert shapes(resolve_prompt_embedding_vectors(a1111, **dual)) == ((8, 768), (8, 1280))
# SDXL embedding: only its clip_l part is usable by a single CLIP-L encoder.
sdxl = {{"clip_l": torch.ones(2, 768), "clip_g": torch.ones(3, 1280)}}
assert shapes(resolve_prompt_embedding_vectors(sdxl, **single)) == ((2, 768), (2, 0))
assert shapes(resolve_prompt_embedding_vectors(sdxl, **dual)) == ((3, 768), (3, 1280))
assert "only has SDXL clip_g" in refused({{"clip_g": torch.ones(2, 1280)}}, **single)
assert refused(torch.ones(2, 1024), **single).endswith("expected 768.")
assert refused(torch.ones(2, 1024), **dual).endswith("expected 768 or 1280.")
print("OK")
"""


def test_resolve_prompt_embedding_vectors_with_real_tensors() -> None:
    code = VECTOR_CHECKS.format(package_dir=str(SRC_DIR / "diffusion_lora_runtime"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)

    if result.stdout.strip() == "NO_TORCH":
        pytest.skip("torch is not installed in this environment")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"

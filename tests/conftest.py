"""Shared test setup: the tests never import real torch/diffusers so they run on machines without GPU or model
dependencies (real inference is integration-tested by calling Diffusers locally).

The stubs below are installed before any test module imports diffusion_lora_runtime. Their pipeline and
scheduler classes are placeholders: tests that exercise model loading monkeypatch recording fakes onto
diffusion_lora_runtime.diffusion_runtime.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
# Fixed tmp_path root (emptied at the start of every run), unless --basetemp is given.
DEFAULT_BASETEMP = Path(tempfile.gettempdir()) / "diffusion-lora-runtime-pytest"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    # pytest's default numbered temp dirs keep a "pytest-current" symlink whose cleanup fails with "Access is
    # denied" on some Windows machines, failing the whole session after every test passed.
    if config.option.basetemp is None:
        config.option.basetemp = str(DEFAULT_BASETEMP)


torch_stub = ModuleType("torch")
torch_stub.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
torch_stub.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
torch_stub.version = SimpleNamespace(cuda=None)
torch_stub.__version__ = "stub"
torch_stub.float16 = object()
torch_stub.bfloat16 = object()
torch_stub.float32 = object()


@contextmanager
def no_grad():
    yield


torch_stub.no_grad = no_grad
sys.modules.setdefault("torch", torch_stub)


def _placeholder(name: str) -> type:
    return type(name, (), {})


diffusers_stub = ModuleType("diffusers")
for _name in (
    "AutoPipelineForText2Image",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline",
):
    setattr(diffusers_stub, _name, _placeholder(_name))
sys.modules.setdefault("diffusers", diffusers_stub)

# Make the source tree importable without an editable install.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

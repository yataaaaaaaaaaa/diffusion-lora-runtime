"""Shared test setup: the tests never import real torch/diffusers so they run on machines without GPU or model
dependencies (real inference is integration-tested by calling Diffusers locally).

The stubs below are installed before any test module imports diffusion_lora_runtime. Their pipeline and
scheduler classes are placeholders: tests that exercise model loading use the ``fake_diffusers`` fixture, which
monkeypatches the recording fakes below onto diffusion_lora_runtime.diffusion_runtime.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

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


class StubGenerator:
    """torch.Generator: remembers its device and seed."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "StubGenerator":
        self.seed = seed
        return self


torch_stub.no_grad = no_grad
torch_stub.Generator = StubGenerator
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


# -- Recording Diffusers fakes (fixture ``fake_diffusers``) --------------------------------------------------------

CHECKPOINT_SCHEDULER_CONFIG = {"num_train_timesteps": 1000}


class FakeScheduler:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "FakeScheduler":
        return cls({**config, **overrides})


class FakeEulerAncestral(FakeScheduler):
    pass


class FakeEuler(FakeScheduler):
    pass


class FakePipeline:
    """A recording stand-in for a Diffusers pipeline.

    ``loads`` (per class) lists the pipelines created by ``from_single_file`` / ``from_pretrained``; ``calls`` lists,
    in order, every LoRA, IP-Adapter, offload and generation call made on the instance as ``(name, arguments)``.
    ``failures[name] = exc`` makes the next call of that method raise ``exc`` (after it is recorded).
    """

    loads: list["FakePipeline"]

    def __init__(self, source: str, kwargs: dict[str, Any], loader: str) -> None:
        self.source = source
        self.kwargs = kwargs
        self.loader = loader
        self.scheduler: FakeScheduler = FakeScheduler(dict(CHECKPOINT_SCHEDULER_CONFIG))
        self.device: str | None = None
        self.attention_slicing = False
        self.image_encoder: Any | None = None
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, Exception] = {}

    def _record(self, name: str, arguments: Any = None) -> None:
        self.calls.append((name, arguments))
        failure = self.failures.pop(name, None)
        if failure is not None:
            raise failure

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def calls_of(self, name: str) -> list[Any]:
        return [arguments for call, arguments in self.calls if call == name]

    @classmethod
    def from_single_file(cls, path: str, **kwargs: Any) -> "FakePipeline":
        pipe = cls(path, kwargs, "from_single_file")
        cls.loads.append(pipe)
        return pipe

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "FakePipeline":
        pipe = cls(path, kwargs, "from_pretrained")
        cls.loads.append(pipe)
        return pipe

    def to(self, device: str) -> "FakePipeline":
        self.device = device
        return self

    def enable_attention_slicing(self) -> None:
        self.attention_slicing = True

    def enable_model_cpu_offload(self) -> None:
        self._record("enable_model_cpu_offload")

    # LoRA (diffusers LoraLoaderMixin)
    def load_lora_weights(self, path: str, *, adapter_name: str, weight_name: str | None = None) -> None:
        self._record("load_lora_weights", {"path": path, "weight_name": weight_name, "adapter_name": adapter_name})

    def delete_adapters(self, adapter_names: list[str]) -> None:
        self._record("delete_adapters", list(adapter_names))

    def set_adapters(self, adapter_names: list[str], adapter_weights: list[float] | None = None) -> None:
        self._record("set_adapters", (list(adapter_names), list(adapter_weights or [])))

    def disable_lora(self) -> None:
        self._record("disable_lora")

    def enable_lora(self) -> None:
        self._record("enable_lora")

    def set_lora_device(self, adapter_names: list[str], device: str) -> None:
        self._record("set_lora_device", (list(adapter_names), device))

    # IP-Adapter (diffusers DiffusionPipeline.register_modules + IPAdapterMixin)
    def register_modules(self, **modules: Any) -> None:
        self._record("register_modules", dict(modules))
        for name, module in modules.items():
            setattr(self, name, module)

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: str,
        subfolder: str,
        weight_name: str,
        image_encoder_folder: str | None = "image_encoder",
    ) -> None:
        self._record(
            "load_ip_adapter",
            {
                "path": pretrained_model_name_or_path_or_dict,
                "subfolder": subfolder,
                "weight_name": weight_name,
                "image_encoder_folder": image_encoder_folder,
                "image_encoder": self.image_encoder,
            },
        )

    def set_ip_adapter_scale(self, scale: float) -> None:
        self._record("set_ip_adapter_scale", scale)

    def unload_ip_adapter(self) -> None:
        self._record("unload_ip_adapter")
        self.image_encoder = None

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self._record("__call__", kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))])


class FakeSD15Pipeline(FakePipeline):
    """One tokenizer/text encoder pair, like StableDiffusionPipeline."""

    loads = []
    tokenizer = "tokenizer"
    text_encoder = "text_encoder"


class FakeSDXLPipeline(FakePipeline):
    """Two tokenizer/text encoder pairs, like StableDiffusionXLPipeline."""

    loads = []
    tokenizer = "tokenizer"
    text_encoder = "text_encoder"
    tokenizer_2 = "tokenizer_2"
    text_encoder_2 = "text_encoder_2"


class FakeAutoPipeline(FakePipeline):
    loads = []


@pytest.fixture()
def fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Diffusers pipeline and scheduler classes used by the runtime with the recording fakes above."""
    from diffusion_lora_runtime import diffusion_runtime as runtime_module

    for pipeline_class in (FakeSD15Pipeline, FakeSDXLPipeline, FakeAutoPipeline):
        monkeypatch.setattr(pipeline_class, "loads", [])
    monkeypatch.setattr(runtime_module, "StableDiffusionPipeline", FakeSD15Pipeline)
    monkeypatch.setattr(runtime_module, "StableDiffusionXLPipeline", FakeSDXLPipeline)
    monkeypatch.setattr(runtime_module, "AutoPipelineForText2Image", FakeAutoPipeline)
    monkeypatch.setattr(runtime_module, "EulerAncestralDiscreteScheduler", FakeEulerAncestral)
    monkeypatch.setattr(runtime_module, "EulerDiscreteScheduler", FakeEuler)


def all_pipeline_loads() -> list[FakePipeline]:
    """Every pipeline the runtime created since the ``fake_diffusers`` fixture started (base model loads)."""
    return [*FakeSD15Pipeline.loads, *FakeSDXLPipeline.loads, *FakeAutoPipeline.loads]

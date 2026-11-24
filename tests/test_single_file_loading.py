"""DiffusionRuntime.ensure_model picks the Diffusers pipeline class from the checkpoint's family, and both
single-file loaders share one scheduler configuration. Diffusers is replaced by recording fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch  # the stub of tests/conftest.py

from diffusion_lora_runtime import ModelSpec
from diffusion_lora_runtime import diffusion_runtime as runtime_module
from diffusion_lora_runtime.diffusion_runtime import DiffusionRuntime

from checkpoint_files import SD15_KEYS, SD21_KEYS, SDXL_KEYS, write_safetensors

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
    """Records how it was loaded; behaves like a pipeline for ensure_model (scheduler, .to, attention slicing)."""

    loads: list["FakePipeline"]

    def __init__(self, source: str, kwargs: dict[str, Any], loader: str) -> None:
        self.source = source
        self.kwargs = kwargs
        self.loader = loader
        self.scheduler: FakeScheduler = FakeScheduler(dict(CHECKPOINT_SCHEDULER_CONFIG))
        self.device: str | None = None
        self.attention_slicing = False

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


class FakeSD15Pipeline(FakePipeline):
    loads = []


class FakeSDXLPipeline(FakePipeline):
    loads = []


class FakeAutoPipeline(FakePipeline):
    loads = []


@pytest.fixture(autouse=True)
def fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> None:
    for pipeline_class in (FakeSD15Pipeline, FakeSDXLPipeline, FakeAutoPipeline):
        monkeypatch.setattr(pipeline_class, "loads", [])
    monkeypatch.setattr(runtime_module, "StableDiffusionPipeline", FakeSD15Pipeline)
    monkeypatch.setattr(runtime_module, "StableDiffusionXLPipeline", FakeSDXLPipeline)
    monkeypatch.setattr(runtime_module, "AutoPipelineForText2Image", FakeAutoPipeline)
    monkeypatch.setattr(runtime_module, "EulerAncestralDiscreteScheduler", FakeEulerAncestral)
    monkeypatch.setattr(runtime_module, "EulerDiscreteScheduler", FakeEuler)


@pytest.fixture()
def runtime() -> DiffusionRuntime:
    # CPU explicitly allowed: the stubbed torch has no accelerator (the default guard is tested below).
    return DiffusionRuntime(device="cpu", allow_cpu=True)


def load(runtime: DiffusionRuntime, path: Path) -> FakePipeline:
    runtime.ensure_model(ModelSpec.from_path(path))
    assert isinstance(runtime.pipe, FakePipeline)
    return runtime.pipe


def test_sd15_checkpoints_load_as_stable_diffusion_pipeline(runtime: DiffusionRuntime, tmp_path: Path) -> None:
    path = write_safetensors(tmp_path / "realisticVision.safetensors", SD15_KEYS)

    pipe = load(runtime, path)

    assert FakeSD15Pipeline.loads == [pipe]
    assert FakeSDXLPipeline.loads == []
    assert (pipe.loader, pipe.source) == ("from_single_file", str(path))
    # The CPU device promotes float16 to float32.
    assert pipe.kwargs == {"use_safetensors": True, "torch_dtype": torch.float32}
    assert (pipe.device, pipe.attention_slicing) == ("cpu", True)


@pytest.mark.parametrize("keys", [SDXL_KEYS, SD21_KEYS], ids=["sdxl", "unknown-family"])
def test_sdxl_and_undetected_checkpoints_keep_loading_as_sdxl(
    runtime: DiffusionRuntime, tmp_path: Path, keys: list[str]
) -> None:
    pipe = load(runtime, write_safetensors(tmp_path / "illustrious.safetensors", keys))

    assert FakeSDXLPipeline.loads == [pipe]
    assert FakeSD15Pipeline.loads == []
    assert pipe.kwargs == {"use_safetensors": True, "torch_dtype": torch.float32}


def test_pickle_checkpoints_keep_loading_as_sdxl(runtime: DiffusionRuntime, tmp_path: Path) -> None:
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(b"not inspected")

    pipe = load(runtime, path)

    assert FakeSDXLPipeline.loads == [pipe]


def test_model_directories_load_with_auto_pipeline(runtime: DiffusionRuntime, tmp_path: Path) -> None:
    folder = tmp_path / "diffusers-model"
    folder.mkdir()

    pipe = load(runtime, folder)

    assert FakeAutoPipeline.loads == [pipe]
    assert (pipe.loader, pipe.kwargs) == ("from_pretrained", {"torch_dtype": torch.float32})
    assert type(pipe.scheduler) is FakeScheduler  # directory models keep their own scheduler


@pytest.mark.parametrize("keys", [SD15_KEYS, SDXL_KEYS], ids=["sd15", "sdxl"])
def test_both_loaders_share_the_scheduler_configuration(
    runtime: DiffusionRuntime, tmp_path: Path, keys: list[str]
) -> None:
    ancestral = load(runtime, write_safetensors(tmp_path / "a.safetensors", keys))
    assert type(ancestral.scheduler) is FakeEulerAncestral
    assert ancestral.scheduler.config == {**CHECKPOINT_SCHEDULER_CONFIG, "timestep_spacing": "leading"}

    runtime.configure_ksampler(sampler_name="euler")
    euler = load(runtime, write_safetensors(tmp_path / "b.safetensors", keys))
    assert type(euler.scheduler) is FakeEuler
    assert euler.scheduler.config == {**CHECKPOINT_SCHEDULER_CONFIG, "timestep_spacing": "leading"}

    # v-prediction is forced by file name, whatever the sampler.
    v_pred = load(runtime, write_safetensors(tmp_path / "noob-vpred-1.0.safetensors", keys))
    assert type(v_pred.scheduler) is FakeEulerAncestral
    assert v_pred.scheduler.config == {
        **CHECKPOINT_SCHEDULER_CONFIG,
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "leading",
    }


def test_the_same_model_is_not_reloaded(runtime: DiffusionRuntime, tmp_path: Path) -> None:
    model = ModelSpec.from_path(write_safetensors(tmp_path / "model.safetensors", SD15_KEYS))

    runtime.ensure_model(model)
    runtime.ensure_model(model)

    assert len(FakeSD15Pipeline.loads) == 1


def test_cpu_fallback_stays_refused_by_default() -> None:
    with pytest.raises(RuntimeError, match="No GPU/accelerator"):
        DiffusionRuntime()

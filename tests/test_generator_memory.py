"""DiffusionGenerator.generate orchestration over a FakeRuntime: only a base model change unloads the pipeline;
LoRA and IP-Adapter requests are passed on to the runtime, which updates them in place."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

# torch/diffusers are stubbed in tests/conftest.py.
from PIL import Image

from diffusion_lora_runtime import (
    DiffusionGenerator,
    IPAdapterSpec,
    KSampleConfig,
    LoRASpec,
    ReferenceImageError,
    StepProgress,
)

CONFIG = KSampleConfig(width=64, height=64, steps=1, cfg=1.0, seed=123)


def make_engine() -> DiffusionGenerator:
    """A generator whose runtime is replaced by FakeRuntime. CPU is allowed explicitly: the stubbed torch has no
    accelerator, and the default constructor (rightly) refuses to fall back to CPU."""
    return DiffusionGenerator(device="cpu", allow_cpu=True)


class FakeRuntime:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.unload_count = 0
        self.ensure_model_count = 0
        self.ensure_adapter_calls = []
        self.ensure_ip_adapter_calls: list[IPAdapterSpec | None] = []
        self.generate_calls: list[dict] = []
        self.configure_calls = []

    @contextmanager
    def serialized_operation(self):
        yield

    def configure_ksampler(self, *, sampler_name: str, scheduler_name: str) -> None:
        self.configure_calls.append((sampler_name, scheduler_name))

    def ensure_model(self, model) -> None:
        self.events.append("ensure_model")
        self.ensure_model_count += 1

    def ensure_adapters(self, loras, adapter_weights=None):
        self.events.append("ensure_adapters")
        self.ensure_adapter_calls.append((list(loras), list(adapter_weights or [])))

    def ensure_ip_adapter(self, spec) -> None:
        self.events.append("ensure_ip_adapter")
        self.ensure_ip_adapter_calls.append(spec)

    def ensure_textual_inversions(self, references):
        return []

    def generate(self, **kwargs):
        self.events.append("generate")
        self.generate_calls.append(kwargs)
        callback = kwargs.get("progress_callback")
        step_callback = kwargs.get("step_callback")
        if callback is not None:
            callback(0, kwargs["steps"])
            callback(kwargs["steps"], kwargs["steps"])
        if step_callback is not None:
            step_callback(StepProgress(step=0, total_steps=kwargs["steps"]))
            step_callback(StepProgress(step=kwargs["steps"], total_steps=kwargs["steps"]))
        return Image.new("RGB", (1, 1)), kwargs["seed"] or 42

    def cancel_current_generation(self) -> None:
        pass

    def unload(self) -> None:
        self.events.append("unload")
        self.unload_count += 1


@pytest.fixture()
def engine_and_runtime() -> tuple[DiffusionGenerator, FakeRuntime]:
    engine = make_engine()
    fake = FakeRuntime()
    engine.runtime = fake
    return engine, fake


def test_lora_changes_never_unload_the_model(engine_and_runtime: tuple[DiffusionGenerator, FakeRuntime]) -> None:
    engine, fake = engine_and_runtime
    lora = LoRASpec.from_path("/tmp/style.safetensors", weight=0.5)

    first = engine.generate(model_path="/tmp/model.safetensors", prompt="cat", loras=[lora], ksample=CONFIG)
    second = engine.generate(model_path="/tmp/model.safetensors", prompt="dog", loras=[lora], ksample=CONFIG)

    assert (first.seed, second.seed) == (123, 123)
    assert fake.unload_count == 0

    # Another LoRA, another weight, no LoRA at all: the runtime updates them in place.
    other = LoRASpec.from_path("/tmp/other.safetensors", weight=0.5)
    engine.generate(model_path="/tmp/model.safetensors", prompt="dog", loras=[other], ksample=CONFIG)
    engine.generate(
        model_path="/tmp/model.safetensors", prompt="dog", loras=[LoRASpec.from_path(other.path, weight=0.9)], ksample=CONFIG
    )
    engine.generate(model_path="/tmp/model.safetensors", prompt="dog", ksample=CONFIG)

    assert fake.unload_count == 0
    assert [weights for _, weights in fake.ensure_adapter_calls] == [[0.5], [0.5], [0.5], [0.9], []]
    assert [[spec.path for spec in specs] for specs, _ in fake.ensure_adapter_calls][2] == [other.path]


@pytest.mark.parametrize(
    "change",
    [
        {"model_path": "/tmp/other-model.safetensors"},
        {"torch_dtype": "bfloat16"},
        {"ksample": KSampleConfig(width=64, height=64, steps=1, cfg=1.0, seed=123, sampler_name="euler")},
    ],
    ids=["model-path", "dtype", "sampler"],
)
def test_a_base_model_change_unloads_first(
    engine_and_runtime: tuple[DiffusionGenerator, FakeRuntime], change: dict
) -> None:
    engine, fake = engine_and_runtime
    request = {"model_path": "/tmp/model.safetensors", "prompt": "cat", "ksample": CONFIG}

    engine.generate(**request)
    engine.generate(**{**request, **change})

    assert fake.unload_count == 1
    second_call = fake.events[4:]
    assert second_call == ["unload", "ensure_model", "ensure_adapters", "ensure_ip_adapter", "generate"]


def test_ip_adapter_requests_are_applied_after_the_loras(
    engine_and_runtime: tuple[DiffusionGenerator, FakeRuntime], tmp_path: Path
) -> None:
    engine, fake = engine_and_runtime
    face = tmp_path / "face.png"
    Image.new("RGB", (4, 2), "red").save(face)
    spec = IPAdapterSpec(
        adapter_path=str(tmp_path / "ip-adapter.safetensors"),
        image_encoder_path=str(tmp_path / "encoder.safetensors"),
        image_paths=(str(face),),
        scale=0.4,
        architecture="sdxl",
    )

    engine.generate(model_path="/tmp/model.safetensors", prompt="cat", ksample=CONFIG, ip_adapter=spec)
    engine.generate(model_path="/tmp/model.safetensors", prompt="cat", ksample=CONFIG)

    assert fake.events == [
        "ensure_model",
        "ensure_adapters",
        "ensure_ip_adapter",
        "generate",
        "ensure_model",
        "ensure_adapters",
        "ensure_ip_adapter",
        "generate",
    ]
    assert fake.ensure_ip_adapter_calls == [spec, None]
    with_reference, without = fake.generate_calls
    assert [image.size for image in with_reference["ip_adapter_images"]] == [(4, 2)]
    assert without["ip_adapter_images"] is None
    assert fake.unload_count == 0


def test_reference_images_are_checked_before_anything_is_loaded(
    engine_and_runtime: tuple[DiffusionGenerator, FakeRuntime], tmp_path: Path
) -> None:
    engine, fake = engine_and_runtime
    spec = IPAdapterSpec(
        adapter_path=str(tmp_path / "ip-adapter.safetensors"),
        image_encoder_path=str(tmp_path / "encoder.safetensors"),
        image_paths=(str(tmp_path / "missing.png"),),
    )

    with pytest.raises(ReferenceImageError, match="Reference image not found"):
        engine.generate(model_path="/tmp/model.safetensors", prompt="cat", ksample=CONFIG, ip_adapter=spec)
    with pytest.raises(ValueError, match="at least one reference image"):
        engine.generate(
            model_path="/tmp/model.safetensors",
            prompt="cat",
            ksample=CONFIG,
            ip_adapter=IPAdapterSpec(adapter_path=spec.adapter_path, image_encoder_path=spec.image_encoder_path, image_paths=()),
        )

    assert fake.events == []


def test_generate_emits_progress_callbacks(engine_and_runtime: tuple[DiffusionGenerator, FakeRuntime]) -> None:
    engine, _ = engine_and_runtime

    scalar_events: list[tuple[int, int]] = []
    step_events: list[StepProgress] = []
    engine.generate(
        model_path="/tmp/model.safetensors",
        prompt="cat",
        ksample=KSampleConfig(width=64, height=64, steps=2, cfg=1.0, seed=123),
        progress_callback=lambda step, total: scalar_events.append((step, total)),
        step_callback=step_events.append,
    )

    assert scalar_events == [(0, 2), (2, 2)]
    assert [event.step for event in step_events] == [0, 2]

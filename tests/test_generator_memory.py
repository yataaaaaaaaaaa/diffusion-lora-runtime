from __future__ import annotations

from contextlib import contextmanager

# torch/diffusers are stubbed in tests/conftest.py.
from PIL import Image

from diffusion_lora_runtime import DiffusionGenerator, KSampleConfig, LoRASpec, StepProgress


def make_engine() -> DiffusionGenerator:
    """A generator whose runtime is replaced by FakeRuntime. CPU is allowed explicitly: the stubbed torch has no
    accelerator, and the default constructor (rightly) refuses to fall back to CPU."""
    return DiffusionGenerator(device="cpu", allow_cpu=True)


class FakeRuntime:
    def __init__(self) -> None:
        self.unload_count = 0
        self.ensure_model_count = 0
        self.ensure_adapter_calls = []
        self.configure_calls = []

    @contextmanager
    def serialized_operation(self):
        yield

    def configure_ksampler(self, *, sampler_name: str, scheduler_name: str) -> None:
        self.configure_calls.append((sampler_name, scheduler_name))

    def ensure_model(self, model) -> None:
        self.ensure_model_count += 1

    def ensure_adapters(self, loras, adapter_weights=None):
        self.ensure_adapter_calls.append((list(loras), list(adapter_weights or [])))

    def ensure_textual_inversions(self, references):
        return []

    def generate(self, **kwargs):
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
        self.unload_count += 1


def test_generate_reuses_same_model_and_loras_until_signature_changes() -> None:
    engine = make_engine()
    fake = FakeRuntime()
    engine.runtime = fake

    lora = LoRASpec.from_path("/tmp/style.safetensors", weight=0.5)
    config = KSampleConfig(width=64, height=64, steps=1, cfg=1.0, seed=123)

    first = engine.generate(model_path="/tmp/model.safetensors", prompt="cat", loras=[lora], ksample=config)
    second = engine.generate(model_path="/tmp/model.safetensors", prompt="dog", loras=[lora], ksample=config)

    assert first.seed == 123
    assert second.seed == 123
    assert fake.unload_count == 0

    changed_lora = LoRASpec.from_path("/tmp/other.safetensors", weight=0.5)
    engine.generate(model_path="/tmp/model.safetensors", prompt="dog", loras=[changed_lora], ksample=config)

    assert fake.unload_count == 1


def test_generate_emits_progress_callbacks() -> None:
    engine = make_engine()
    fake = FakeRuntime()
    engine.runtime = fake

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


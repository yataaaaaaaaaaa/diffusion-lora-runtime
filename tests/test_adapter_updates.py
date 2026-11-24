"""LoRA sets are updated incrementally on the loaded pipeline: only a base model change reloads it.

DiffusionGenerator + DiffusionRuntime run for real over the recording Diffusers fakes of conftest.py (fixture
``fake_diffusers``); ``pipe.calls`` shows exactly which LoRA operations each generation made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from diffusion_lora_runtime import AdapterLoadError, DiffusionGenerator, KSampleConfig, LoRASpec
from diffusion_lora_runtime.diffusion_runtime import DiffusionRuntime, LoadedTextualInversion

from checkpoint_files import SD15_KEYS, write_safetensors
from conftest import FakePipeline, all_pipeline_loads

CONFIG = KSampleConfig(width=64, height=64, steps=1, cfg=1.0, seed=7)
LORA_NAMES = ("detail", "style", "pose")


@pytest.fixture()
def engine(fake_diffusers: None, monkeypatch: pytest.MonkeyPatch) -> DiffusionGenerator:
    # LoRA files are a few bytes: their state dict is never read (no LoCon folding needed).
    monkeypatch.setattr(DiffusionRuntime, "_read_adapter_state_dict", lambda self, path: {})
    monkeypatch.setattr(DiffusionRuntime, "_summarize_adapter_file", lambda self, path: "summary")
    return DiffusionGenerator(device="cpu", allow_cpu=True)


@pytest.fixture()
def model(tmp_path: Path) -> Path:
    return write_safetensors(tmp_path / "model.safetensors", SD15_KEYS)


@pytest.fixture()
def lora_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "loras"
    folder.mkdir()
    for name in LORA_NAMES:
        (folder / f"{name}.safetensors").write_bytes(b"lora")
    return folder


def lora(lora_dir: Path, name: str, weight: float = 1.0) -> LoRASpec:
    return LoRASpec.from_path(lora_dir / f"{name}.safetensors", weight=weight)


def generate(engine: DiffusionGenerator, model: Path, *loras: LoRASpec, prompt: str = "cat", **kwargs: Any) -> None:
    engine.generate(model_path=model, prompt=prompt, loras=list(loras), ksample=CONFIG, **kwargs)


def the_pipe() -> FakePipeline:
    """The single pipeline loaded so far (fails when the base model was loaded more than once)."""
    (pipe,) = all_pipeline_loads()
    return pipe


def adapter_calls(pipe: FakePipeline) -> list[tuple[str, Any]]:
    """The LoRA operations made since the last call, then forgets them (pipeline calls are left out)."""
    calls = [call for call in pipe.calls if call[0] != "__call__"]
    pipe.calls.clear()
    return calls


def load_call(lora_dir: Path, spec: LoRASpec) -> tuple[str, dict[str, str]]:
    return (
        "load_lora_weights",
        {"path": str(lora_dir), "weight_name": Path(spec.path).name, "adapter_name": spec.id},
    )


def test_the_first_request_loads_every_lora_and_applies_the_weights(
    engine: DiffusionGenerator, model: Path, lora_dir: Path
) -> None:
    detail, style = lora(lora_dir, "detail", 0.5), lora(lora_dir, "style")

    generate(engine, model, detail, style)

    assert adapter_calls(the_pipe()) == [
        load_call(lora_dir, detail),
        load_call(lora_dir, style),
        ("set_adapters", ([detail.id, style.id], [0.5, 1.0])),
    ]


def test_a_weight_only_change_only_sets_the_adapters(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    detail, style = lora(lora_dir, "detail", 0.5), lora(lora_dir, "style")
    generate(engine, model, detail, style)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, lora(lora_dir, "detail", 0.8), style)

    assert adapter_calls(pipe) == [("set_adapters", ([detail.id, style.id], [0.8, 1.0]))]
    assert the_pipe() is pipe  # no model reload


def test_an_unchanged_request_calls_nothing(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    detail = lora(lora_dir, "detail", 0.5)
    generate(engine, model, detail)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, detail, prompt="another prompt")

    assert adapter_calls(pipe) == []
    assert pipe.call_names() == []  # the generation itself was recorded and cleared above
    assert the_pipe() is pipe


def test_adding_a_lora_loads_exactly_that_one(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    detail, style = lora(lora_dir, "detail", 0.5), lora(lora_dir, "style", 0.7)
    generate(engine, model, detail)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, detail, style)

    assert adapter_calls(pipe) == [
        load_call(lora_dir, style),
        ("set_adapters", ([detail.id, style.id], [0.5, 0.7])),
    ]
    assert the_pipe() is pipe


def test_removing_a_lora_deletes_exactly_that_one(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    detail, style, pose = (lora(lora_dir, name) for name in LORA_NAMES)
    generate(engine, model, detail, style, pose)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, detail, pose)

    assert adapter_calls(pipe) == [
        ("delete_adapters", [style.id]),
        ("set_adapters", ([detail.id, pose.id], [1.0, 1.0])),
    ]
    assert engine.runtime.loaded_adapters.keys() == {detail.id, pose.id}
    assert the_pipe() is pipe


def test_replacing_a_lora_deletes_the_old_one_and_loads_the_new_one(
    engine: DiffusionGenerator, model: Path, lora_dir: Path
) -> None:
    detail, style = lora(lora_dir, "detail"), lora(lora_dir, "style")
    generate(engine, model, detail)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, style)

    assert adapter_calls(pipe) == [
        ("delete_adapters", [detail.id]),
        load_call(lora_dir, style),
        ("set_adapters", ([style.id], [1.0])),
    ]


def test_an_id_naming_another_file_is_reloaded(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    before = LoRASpec(id="character", name="character", path=str(lora_dir / "detail.safetensors"))
    after = LoRASpec(id="character", name="character", path=str(lora_dir / "style.safetensors"))
    generate(engine, model, before)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model, after)

    assert adapter_calls(pipe) == [
        ("delete_adapters", ["character"]),
        load_call(lora_dir, after),
        ("set_adapters", (["character"], [1.0])),
    ]


def test_an_empty_request_deletes_the_loras_and_disables_lora(
    engine: DiffusionGenerator, model: Path, lora_dir: Path
) -> None:
    detail, style = lora(lora_dir, "detail"), lora(lora_dir, "style")
    generate(engine, model, detail, style)
    pipe = the_pipe()
    adapter_calls(pipe)

    generate(engine, model)
    assert adapter_calls(pipe) == [("delete_adapters", [detail.id, style.id]), ("disable_lora", None)]
    assert engine.runtime.loaded_adapters == {}

    generate(engine, model)
    assert adapter_calls(pipe) == []

    # The next LoRA set enables LoRA layers again.
    generate(engine, model, detail)
    assert adapter_calls(pipe) == [
        load_call(lora_dir, detail),
        ("set_adapters", ([detail.id], [1.0])),
        ("enable_lora", None),
    ]
    assert the_pipe() is pipe


def test_a_first_request_without_loras_disables_lora(engine: DiffusionGenerator, model: Path) -> None:
    generate(engine, model)

    assert adapter_calls(the_pipe()) == [("disable_lora", None)]


def test_lora_changes_keep_textual_inversions_loaded(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    generate(engine, model, lora(lora_dir, "detail"))
    embedding = LoadedTextualInversion(token="dlr_embed_neg", path="neg.safetensors", tokens=["dlr_embed_neg"])
    engine.runtime.loaded_textual_inversions["neg"] = embedding

    generate(engine, model, lora(lora_dir, "style", 0.4))
    generate(engine, model)

    assert engine.runtime.loaded_textual_inversions == {"neg": embedding}


def test_a_model_change_reloads_the_pipeline_and_its_loras(
    engine: DiffusionGenerator, model: Path, lora_dir: Path, tmp_path: Path
) -> None:
    detail = lora(lora_dir, "detail", 0.5)
    generate(engine, model, detail)
    first = the_pipe()
    adapter_calls(first)

    other_model = write_safetensors(tmp_path / "other.safetensors", SD15_KEYS)
    generate(engine, other_model, detail)

    first_load, second = all_pipeline_loads()
    assert first_load is first and engine.runtime.pipe is second
    assert adapter_calls(first) == []  # the old pipeline is dropped, not updated
    assert adapter_calls(second) == [load_call(lora_dir, detail), ("set_adapters", ([detail.id], [0.5]))]


def test_inline_prompt_loras_follow_the_same_rules(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    def inline(prompt: str) -> None:
        generate(engine, model, prompt=prompt, prompt_lora_dir=lora_dir)

    inline("cat <lora:detail:0.5>")
    pipe = the_pipe()
    (load,) = pipe.calls_of("load_lora_weights")
    detail_id = load["adapter_name"]
    assert load["weight_name"] == "detail.safetensors"
    assert pipe.calls_of("__call__")[0]["prompt"] == "cat"
    adapter_calls(pipe)

    inline("cat <lora:detail:0.8>")
    assert adapter_calls(pipe) == [("set_adapters", ([detail_id], [0.8]))]

    inline("dog <lora:detail:0.8>")
    assert adapter_calls(pipe) == []

    inline("dog <lora:detail:0.8> <lora:style:1>")
    (added,) = pipe.calls_of("load_lora_weights")
    assert adapter_calls(pipe) == [
        ("load_lora_weights", added),
        ("set_adapters", ([detail_id, added["adapter_name"]], [0.8, 1.0])),
    ]

    inline("dog <lora:style:1>")
    assert adapter_calls(pipe) == [
        ("delete_adapters", [detail_id]),
        ("set_adapters", ([added["adapter_name"]], [1.0])),
    ]
    assert the_pipe() is pipe


def test_an_inline_tag_overrides_the_same_explicit_lora(engine: DiffusionGenerator, model: Path, lora_dir: Path) -> None:
    generate(engine, model, lora(lora_dir, "detail", 0.5), prompt="cat <lora:detail:0.9>", prompt_lora_dir=lora_dir)

    pipe = the_pipe()
    (load,) = pipe.calls_of("load_lora_weights")  # one adapter per file
    assert load["weight_name"] == "detail.safetensors"
    assert adapter_calls(pipe) == [("load_lora_weights", load), ("set_adapters", ([load["adapter_name"]], [0.9]))]


def test_a_failed_delete_is_reported_and_retried_by_the_next_request(
    engine: DiffusionGenerator, model: Path, lora_dir: Path
) -> None:
    detail, style = lora(lora_dir, "detail"), lora(lora_dir, "style")
    generate(engine, model, detail, style)
    pipe = the_pipe()
    adapter_calls(pipe)
    pipe.failures["delete_adapters"] = RuntimeError("adapter layers busy")

    with pytest.raises(AdapterLoadError, match="Failed to remove LoRA style: adapter layers busy"):
        generate(engine, model, detail)
    assert engine.runtime.loaded_adapters.keys() == {detail.id, style.id}
    adapter_calls(pipe)

    generate(engine, model, detail)
    assert adapter_calls(pipe) == [("delete_adapters", [style.id]), ("set_adapters", ([detail.id], [1.0]))]


def test_a_failed_load_is_discarded_and_retried_by_the_next_request(
    engine: DiffusionGenerator, model: Path, lora_dir: Path
) -> None:
    detail = lora(lora_dir, "detail")
    generate(engine, model)
    pipe = the_pipe()
    adapter_calls(pipe)
    pipe.failures["load_lora_weights"] = RuntimeError("keys have not been correctly renamed")

    with pytest.raises(AdapterLoadError, match="Failed to load LoRA 'detail'"):
        generate(engine, model, detail)
    assert adapter_calls(pipe) == [load_call(lora_dir, detail), ("delete_adapters", [detail.id])]
    assert engine.runtime.loaded_adapters == {}

    generate(engine, model, detail)
    assert adapter_calls(pipe) == [
        load_call(lora_dir, detail),
        ("set_adapters", ([detail.id], [1.0])),
        ("enable_lora", None),
    ]
    assert the_pipe() is pipe

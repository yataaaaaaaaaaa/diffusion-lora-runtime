"""IP-Adapter support: DiffusionRuntime.ensure_ip_adapter over the recording Diffusers fakes of conftest.py (with a
monkeypatched image encoder loader), the reference images given to the pipeline, and ip_adapter.py (configuration,
image encoder loading with stubbed transformers / safetensors.torch / accelerate, and once with the real libraries
in a subprocess on a tiny model)."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch  # the stub of tests/conftest.py
from PIL import Image

from diffusion_lora_runtime import (
    AdapterLoadError,
    DiffusionGenerator,
    IPAdapterSpec,
    KSampleConfig,
    ModelSpec,
    ReferenceImageError,
)
from diffusion_lora_runtime import diffusion_runtime as runtime_module
from diffusion_lora_runtime.diffusion_runtime import DiffusionRuntime
from diffusion_lora_runtime.ip_adapter import (
    ImageEncoderLayoutError,
    VIT_H_14_IMAGE_ENCODER_CONFIG,
    load_image_encoder,
    open_reference_images,
)

from checkpoint_files import SD15_KEYS, SDXL_KEYS, write_safetensors
from conftest import SRC_DIR, FakePipeline, all_pipeline_loads

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_METADATA_KEYS = {"_name_or_path", "architectures", "torch_dtype", "transformers_version"}
SINGLE_FILE_KEYS = ["vision_model.embeddings.class_embedding", "visual_projection.weight"]
POSITION_IDS = "vision_model.embeddings.position_ids"


@dataclass
class FakeEncoder:
    path: str
    torch_dtype: Any
    device: str


@pytest.fixture()
def encoders(monkeypatch: pytest.MonkeyPatch) -> list[FakeEncoder]:
    """Every image encoder the runtime loads (load_image_encoder is replaced by a recording fake)."""
    loaded: list[FakeEncoder] = []

    def fake_load(path: str, *, torch_dtype: Any, device: str) -> FakeEncoder:
        encoder = FakeEncoder(str(path), torch_dtype, device)
        loaded.append(encoder)
        return encoder

    monkeypatch.setattr(runtime_module, "load_image_encoder", fake_load)
    return loaded


@pytest.fixture()
def files(tmp_path: Path) -> SimpleNamespace:
    adapters = tmp_path / "ipadapter"
    adapters.mkdir()
    encoders = tmp_path / "clip_vision"
    encoders.mkdir()
    names = {
        "sdxl_face": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "sdxl_plus": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "sd15_face": "ip-adapter-plus-face_sd15.safetensors",
    }
    for name in names.values():
        (adapters / name).write_bytes(b"adapter")
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(red)
    Image.new("RGBA", (2, 4), (0, 0, 255, 128)).save(blue)  # converted to RGB
    return SimpleNamespace(
        adapters=adapters,
        **{key: adapters / name for key, name in names.items()},
        encoder=encoders / "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        other_encoder=encoders / "image_encoder",
        red=red,
        blue=blue,
        sdxl_model=write_safetensors(tmp_path / "sdxl.safetensors", SDXL_KEYS),
        sd15_model=write_safetensors(tmp_path / "sd15.safetensors", SD15_KEYS),
    )


def make_spec(files: SimpleNamespace, adapter: Path | None = None, **changes: Any) -> IPAdapterSpec:
    spec = IPAdapterSpec(
        adapter_path=str(adapter or files.sdxl_face),
        image_encoder_path=str(files.encoder),
        image_paths=(str(files.red),),
        scale=0.6,
        architecture="sdxl",
    )
    return replace(spec, **changes)


@pytest.fixture()
def runtime(fake_diffusers: None, encoders: list[FakeEncoder]) -> DiffusionRuntime:
    return DiffusionRuntime(device="cpu", allow_cpu=True)


def load_model(runtime: DiffusionRuntime, path: Path) -> FakePipeline:
    runtime.ensure_model(ModelSpec.from_path(path))
    assert isinstance(runtime.pipe, FakePipeline)
    return runtime.pipe


def calls(pipe: FakePipeline) -> list[tuple[str, Any]]:
    """The calls made since the last check, then forgets them."""
    recorded = list(pipe.calls)
    pipe.calls.clear()
    return recorded


def load_calls(files: SimpleNamespace, adapter: Path, encoder: Any, scale: float = 0.6) -> list[tuple[str, Any]]:
    return [
        ("register_modules", {"image_encoder": encoder}),
        (
            "load_ip_adapter",
            {
                "path": str(files.adapters),
                "subfolder": "",
                "weight_name": adapter.name,
                "image_encoder_folder": None,
                "image_encoder": encoder,  # registered before the adapter is loaded
            },
        ),
        ("set_ip_adapter_scale", scale),
    ]


# -- DiffusionRuntime.ensure_ip_adapter ------------------------------------------------------------------------------


def test_the_first_load_registers_the_encoder_before_loading_the_adapter(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)

    runtime.ensure_ip_adapter(make_spec(files))

    # The CPU device promotes float16 to float32: the encoder gets the pipeline dtype.
    assert encoders == [FakeEncoder(str(files.encoder), torch.float32, "cpu")]
    assert calls(pipe) == load_calls(files, files.sdxl_face, encoders[0])
    assert all_pipeline_loads() == [pipe]


def test_the_scale_changes_alone_and_the_same_spec_loads_once(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)
    runtime.ensure_ip_adapter(make_spec(files))
    calls(pipe)

    runtime.ensure_ip_adapter(make_spec(files))
    assert calls(pipe) == []

    runtime.ensure_ip_adapter(make_spec(files, scale=0.35))
    assert calls(pipe) == [("set_ip_adapter_scale", 0.35)]
    assert len(encoders) == 1
    assert all_pipeline_loads() == [pipe]


def test_another_adapter_is_swapped_in_and_reuses_the_same_encoder(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)
    runtime.ensure_ip_adapter(make_spec(files))
    calls(pipe)

    runtime.ensure_ip_adapter(make_spec(files, files.sdxl_plus))

    assert calls(pipe) == [("unload_ip_adapter", None), *load_calls(files, files.sdxl_plus, encoders[0])]
    assert len(encoders) == 1
    assert all_pipeline_loads() == [pipe]


def test_another_image_encoder_is_loaded(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)
    runtime.ensure_ip_adapter(make_spec(files))
    calls(pipe)

    runtime.ensure_ip_adapter(make_spec(files, image_encoder_path=str(files.other_encoder)))

    assert [encoder.path for encoder in encoders] == [str(files.encoder), str(files.other_encoder)]
    assert calls(pipe) == [("unload_ip_adapter", None), *load_calls(files, files.sdxl_face, encoders[1])]


def test_none_unloads_the_adapter_and_keeps_the_base_model(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)
    runtime.ensure_ip_adapter(None)
    assert calls(pipe) == []  # nothing loaded: nothing to do

    runtime.ensure_ip_adapter(make_spec(files))
    calls(pipe)

    runtime.ensure_ip_adapter(None)
    assert calls(pipe) == [("unload_ip_adapter", None)]
    assert runtime.loaded_ip_adapter is None
    runtime.ensure_ip_adapter(None)
    assert calls(pipe) == []

    # The encoder was released with the adapter: loading again reads it again.
    runtime.ensure_ip_adapter(make_spec(files))
    assert calls(pipe) == load_calls(files, files.sdxl_face, encoders[1])
    assert len(encoders) == 2
    assert all_pipeline_loads() == [pipe]


@pytest.mark.parametrize(
    ("model", "adapter", "architecture", "message"),
    [
        ("sd15_model", "sdxl_face", "sdxl", "IP-Adapter made for SDXL cannot be used with an SD 1.5 model."),
        ("sdxl_model", "sd15_face", "sd15", "IP-Adapter made for SD 1.5 cannot be used with an SDXL model."),
    ],
)
def test_an_architecture_mismatch_is_refused(
    runtime: DiffusionRuntime,
    files: SimpleNamespace,
    encoders: list[FakeEncoder],
    model: str,
    adapter: str,
    architecture: str,
    message: str,
) -> None:
    pipe = load_model(runtime, getattr(files, model))

    with pytest.raises(AdapterLoadError) as raised:
        runtime.ensure_ip_adapter(make_spec(files, getattr(files, adapter), architecture=architecture))

    assert str(raised.value) == message
    assert calls(pipe) == [] and encoders == []
    assert runtime.loaded_ip_adapter is None


def test_an_unknown_architecture_is_not_checked(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sd15_model)

    runtime.ensure_ip_adapter(make_spec(files, files.sd15_face, architecture=None))

    assert calls(pipe) == load_calls(files, files.sd15_face, encoders[0])


def test_model_cpu_offload_hooks_the_encoder(
    fake_diffusers: None, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    runtime = DiffusionRuntime(device="cuda")
    pipe = load_model(runtime, files.sdxl_model)
    assert calls(pipe) == [("enable_model_cpu_offload", None)]

    runtime.ensure_ip_adapter(make_spec(files))
    # Loaded on the CPU (the hooks move it), then the hook chain is rebuilt with image_encoder in it.
    assert encoders == [FakeEncoder(str(files.encoder), torch.float16, "cpu")]
    register, load, scale = load_calls(files, files.sdxl_face, encoders[0])
    assert calls(pipe) == [register, load, ("enable_model_cpu_offload", None), scale]

    runtime.ensure_ip_adapter(None)
    assert calls(pipe) == [("unload_ip_adapter", None), ("enable_model_cpu_offload", None)]
    assert all_pipeline_loads() == [pipe]


def test_without_offload_the_encoder_is_loaded_on_the_device(
    fake_diffusers: None, files: SimpleNamespace, encoders: list[FakeEncoder], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "MODEL_CPU_OFFLOAD", False)
    runtime = DiffusionRuntime(device="cuda")
    pipe = load_model(runtime, files.sdxl_model)

    runtime.ensure_ip_adapter(make_spec(files))

    assert pipe.device == "cuda"
    assert encoders == [FakeEncoder(str(files.encoder), torch.float16, "cuda")]
    assert calls(pipe) == load_calls(files, files.sdxl_face, encoders[0])


def test_a_model_change_forgets_the_adapter_and_the_encoder(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder], tmp_path: Path
) -> None:
    first = load_model(runtime, files.sdxl_model)
    runtime.ensure_ip_adapter(make_spec(files))

    second = load_model(runtime, write_safetensors(tmp_path / "other-sdxl.safetensors", SDXL_KEYS))
    assert runtime.loaded_ip_adapter is None

    runtime.ensure_ip_adapter(make_spec(files))
    assert calls(second) == load_calls(files, files.sdxl_face, encoders[1])
    assert all_pipeline_loads() == [first, second]


def test_a_failed_load_is_cleaned_up_and_reported(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)
    pipe.failures["load_ip_adapter"] = ValueError("Required keys are (`image_proj` and `ip_adapter`) missing")

    with pytest.raises(AdapterLoadError) as raised:
        runtime.ensure_ip_adapter(make_spec(files))

    assert str(raised.value) == (
        "Failed to load IP-Adapter 'ip-adapter-plus-face_sdxl_vit-h.safetensors' with image encoder "
        "'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors': Required keys are (`image_proj` and `ip_adapter`) missing"
    )
    register, load, _ = load_calls(files, files.sdxl_face, encoders[0])
    assert calls(pipe) == [register, load, ("unload_ip_adapter", None)]
    assert runtime.loaded_ip_adapter is None

    runtime.ensure_ip_adapter(make_spec(files))  # retried from scratch
    assert calls(pipe) == load_calls(files, files.sdxl_face, encoders[1])


def test_missing_adapter_weights_are_reported_before_loading_the_encoder(
    runtime: DiffusionRuntime, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    pipe = load_model(runtime, files.sdxl_model)

    with pytest.raises(AdapterLoadError, match="IP-Adapter weights not found"):
        runtime.ensure_ip_adapter(make_spec(files, files.adapters / "missing.safetensors"))

    assert calls(pipe) == [] and encoders == []


def test_image_encoder_failures_are_reported(
    runtime: DiffusionRuntime, files: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = load_model(runtime, files.sdxl_model)

    def broken(path: str, **_: Any) -> None:
        raise ImageEncoderLayoutError(f"{path} is not a CLIP ViT-H/14 image encoder")

    monkeypatch.setattr(runtime_module, "load_image_encoder", broken)

    with pytest.raises(AdapterLoadError, match="is not a CLIP ViT-H/14 image encoder"):
        runtime.ensure_ip_adapter(make_spec(files))
    assert calls(pipe) == []


def test_generate_requires_reference_images_exactly_when_an_adapter_is_loaded(
    runtime: DiffusionRuntime, files: SimpleNamespace
) -> None:
    load_model(runtime, files.sd15_model)
    request = dict(prompt="cat", negative_prompt="", width=8, height=8, steps=1, guidance_scale=1.0, seed=1)

    with pytest.raises(ValueError, match="no IP-Adapter is loaded"):
        runtime.generate(**request, ip_adapter_images=[Image.new("RGB", (1, 1))])

    runtime.ensure_ip_adapter(make_spec(files, files.sd15_face, architecture="sd15"))
    with pytest.raises(ValueError, match="An IP-Adapter is loaded"):
        runtime.generate(**request)
    with pytest.raises(ValueError, match="at least one image"):
        runtime.generate(**request, ip_adapter_images=[])


# -- DiffusionGenerator.generate(ip_adapter=...) ---------------------------------------------------------------------


@pytest.fixture()
def engine(fake_diffusers: None, encoders: list[FakeEncoder]) -> DiffusionGenerator:
    return DiffusionGenerator(device="cpu", allow_cpu=True)


def test_the_pipeline_gets_the_reference_images_in_order(
    engine: DiffusionGenerator, files: SimpleNamespace, encoders: list[FakeEncoder]
) -> None:
    spec = make_spec(files, files.sd15_face, architecture="sd15", image_paths=(str(files.red), str(files.blue)))
    ksample = KSampleConfig(width=8, height=8, steps=1, cfg=1.0, seed=3)

    engine.generate(model_path=files.sd15_model, prompt="portrait", ksample=ksample, ip_adapter=spec)

    (pipe,) = all_pipeline_loads()
    assert pipe.call_names() == [
        "disable_lora",
        "register_modules",
        "load_ip_adapter",
        "set_ip_adapter_scale",
        "__call__",
    ]
    (call,) = pipe.calls_of("__call__")
    (images,) = call["ip_adapter_image"]  # one list per adapter
    assert [(image.mode, image.size, image.getpixel((0, 0))) for image in images] == [
        ("RGB", (4, 2), (255, 0, 0)),
        ("RGB", (2, 4), (0, 0, 255)),
    ]
    pipe.calls.clear()

    # Without a reference: the adapter is unloaded, the base model and the encoder file are not read again.
    engine.generate(model_path=files.sd15_model, prompt="portrait", ksample=ksample)
    assert pipe.call_names() == ["unload_ip_adapter", "__call__"]
    assert "ip_adapter_image" not in pipe.calls_of("__call__")[0]
    assert all_pipeline_loads() == [pipe]
    assert len(encoders) == 1


def test_a_strength_change_between_generations_only_rescales(
    engine: DiffusionGenerator, files: SimpleNamespace
) -> None:
    spec = make_spec(files, files.sd15_face, architecture="sd15")
    ksample = KSampleConfig(width=8, height=8, steps=1, cfg=1.0, seed=3)
    engine.generate(model_path=files.sd15_model, prompt="portrait", ksample=ksample, ip_adapter=spec)
    (pipe,) = all_pipeline_loads()
    pipe.calls.clear()

    engine.generate(model_path=files.sd15_model, prompt="portrait", ksample=ksample, ip_adapter=replace(spec, scale=0.9))

    assert pipe.call_names() == ["set_ip_adapter_scale", "__call__"]
    assert all_pipeline_loads() == [pipe]


@pytest.mark.parametrize("broken", ["missing", "unreadable"])
def test_a_bad_reference_image_fails_before_any_model_load(
    engine: DiffusionGenerator, files: SimpleNamespace, tmp_path: Path, broken: str
) -> None:
    image = tmp_path / f"{broken}.png"
    if broken == "unreadable":
        image.write_bytes(b"not an image")
    spec = make_spec(files, image_paths=(str(files.red), str(image)))

    with pytest.raises(ReferenceImageError) as raised:
        engine.generate(model_path=files.sdxl_model, prompt="portrait", ip_adapter=spec)

    assert isinstance(raised.value, ValueError)
    assert str(image) in str(raised.value)
    assert all_pipeline_loads() == []


# -- ip_adapter.py ---------------------------------------------------------------------------------------------------


def test_the_vit_h_config_is_the_h94_image_encoder_config_without_metadata() -> None:
    fixture = json.loads((FIXTURES / "clip_vit_h_14_image_encoder_config.json").read_text(encoding="utf-8"))

    assert dict(VIT_H_14_IMAGE_ENCODER_CONFIG) == {
        key: value for key, value in fixture.items() if key not in CONFIG_METADATA_KEYS
    }
    assert fixture.keys() >= CONFIG_METADATA_KEYS


def test_open_reference_images_needs_at_least_one_image() -> None:
    with pytest.raises(ReferenceImageError, match="at least one reference image"):
        open_reference_images([])


class StubVisionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class StubVisionModel:
    """transformers.CLIPVisionModelWithProjection: records how it was built, loaded, converted and placed."""

    expected_keys: list[str] = SINGLE_FILE_KEYS

    def __init__(self, config: StubVisionConfig | None = None) -> None:
        self.config = config
        self.built_empty = False
        self.source: tuple[Any, ...] = ("config",)
        self.loaded: tuple[dict[str, Any], bool, bool] | None = None
        self.moves: list[tuple[str, Any]] = []
        self.evaluated = False

    @classmethod
    def from_pretrained(cls, path: str, *, torch_dtype: Any) -> "StubVisionModel":
        model = cls()
        model.source = ("from_pretrained", path, torch_dtype)
        return model

    def state_dict(self) -> dict[str, None]:
        return {key: None for key in self.expected_keys}

    def load_state_dict(self, state_dict: dict[str, Any], *, strict: bool, assign: bool) -> None:
        self.loaded = (dict(state_dict), strict, assign)

    def to(self, *, device: str, dtype: Any) -> "StubVisionModel":
        self.moves.append((device, dtype))
        return self

    def eval(self) -> "StubVisionModel":
        self.evaluated = True
        return self


@pytest.fixture()
def stub_libraries(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """transformers, safetensors.torch and accelerate replaced by stubs (the real ones would import torch)."""
    state = SimpleNamespace(file_keys=[*SINGLE_FILE_KEYS, POSITION_IDS], loaded_files=[], empty_init=[])

    transformers = ModuleType("transformers")
    transformers.CLIPVisionConfig = StubVisionConfig
    transformers.CLIPVisionModelWithProjection = StubVisionModel

    safetensors_torch = ModuleType("safetensors.torch")

    def load_file(path: str, device: str) -> dict[str, str]:
        state.loaded_files.append((path, device))
        return {key: f"tensor:{key}" for key in state.file_keys}

    safetensors_torch.load_file = load_file

    accelerate = ModuleType("accelerate")

    @contextmanager
    def init_empty_weights():
        state.empty_init.append("enter")
        yield

    accelerate.init_empty_weights = init_empty_weights

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    return state


def test_a_single_file_encoder_is_built_with_the_vit_h_config(stub_libraries: SimpleNamespace, tmp_path: Path) -> None:
    path = tmp_path / "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    path.write_bytes(b"weights")

    encoder = load_image_encoder(path, torch_dtype=torch.float16, device="cuda")

    assert isinstance(encoder, StubVisionModel)
    assert encoder.config.kwargs == dict(VIT_H_14_IMAGE_ENCODER_CONFIG)
    assert stub_libraries.empty_init == ["enter"]
    assert stub_libraries.loaded_files == [(str(path), "cpu")]
    # The position_ids buffer of the file is tolerated (dropped: the model does not expect it); strict otherwise.
    assert encoder.loaded == ({key: f"tensor:{key}" for key in SINGLE_FILE_KEYS}, True, True)
    assert encoder.moves == [("cuda", torch.float16)]
    assert encoder.evaluated


def test_a_buffer_the_model_expects_is_kept(
    stub_libraries: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(StubVisionModel, "expected_keys", [*SINGLE_FILE_KEYS, POSITION_IDS])
    path = tmp_path / "encoder.safetensors"
    path.write_bytes(b"weights")

    encoder = load_image_encoder(path, torch_dtype=torch.float32, device="cpu")

    assert encoder.loaded is not None and POSITION_IDS in encoder.loaded[0]


def test_a_single_file_of_another_layout_is_refused(stub_libraries: SimpleNamespace, tmp_path: Path) -> None:
    stub_libraries.file_keys = ["visual.proj", "visual.transformer.resblocks.0.attn.in_proj_weight"]
    path = tmp_path / "open_clip.safetensors"
    path.write_bytes(b"weights")

    with pytest.raises(ImageEncoderLayoutError, match="not a CLIP ViT-H/14 image encoder in the transformers layout"):
        load_image_encoder(path, torch_dtype=torch.float16, device="cpu")


def test_a_folder_encoder_uses_from_pretrained(stub_libraries: SimpleNamespace, tmp_path: Path) -> None:
    folder = tmp_path / "image_encoder"
    folder.mkdir()

    encoder = load_image_encoder(folder, torch_dtype=torch.float16, device="cuda")

    assert encoder.source == ("from_pretrained", str(folder), torch.float16)
    assert stub_libraries.loaded_files == [] and stub_libraries.empty_init == []
    assert encoder.moves == [("cuda", torch.float16)]
    assert encoder.evaluated


def test_missing_and_non_safetensors_encoders_are_refused(stub_libraries: SimpleNamespace, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Image encoder not found"):
        load_image_encoder(tmp_path / "missing.safetensors", torch_dtype=torch.float16, device="cpu")

    pickle = tmp_path / "encoder.bin"
    pickle.write_bytes(b"weights")
    with pytest.raises(ValueError, match=r"must be \.safetensors files"):
        load_image_encoder(pickle, torch_dtype=torch.float16, device="cpu")


def test_ip_adapter_module_imports_without_torch() -> None:
    code = (
        "import sys\n"
        "from diffusion_lora_runtime import ip_adapter, ReferenceImageError\n"
        "heavy = sorted(name for name in ('torch', 'diffusers', 'transformers', 'accelerate') if name in sys.modules)\n"
        "assert not heavy, heavy\n"
        "assert issubclass(ReferenceImageError, ValueError)\n"
    )
    env_path = str(SRC_DIR)
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {env_path!r})\n{code}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


# The runtime with the real torch / transformers / accelerate / safetensors / diffusers, in a subprocess (no model file,
# no network, no GPU): load_image_encoder on a tiny CLIP vision model saved as a ComfyUI-like single file and as a
# transformers folder (same code path as the 2.5 GB ViT-H/14 file, with a patched configuration), then
# DiffusionRuntime.ensure_ip_adapter / generate over a tiny random SD 1.x pipeline and h94-layout adapters: what the
# recording fakes above assume of the Diffusers API. Skipped when a library is not installed.
REAL_LIBRARY_CHECKS = """
import importlib.util, json, sys
from pathlib import Path
LIBRARIES = ("torch", "transformers", "accelerate", "safetensors", "diffusers")
if any(importlib.util.find_spec(name) is None for name in LIBRARIES):
    print("MISSING")
    raise SystemExit(0)
sys.path.insert(0, {src_dir!r})
import numpy as np
import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, CLIPVisionConfig, CLIPVisionModelWithProjection

from diffusion_lora_runtime import IPAdapterSpec, ip_adapter
from diffusion_lora_runtime.diffusion_runtime import DiffusionRuntime

tmp = Path({tmp_dir!r})
torch.manual_seed(0)

# -- load_image_encoder: a ComfyUI-like single file (transformers keys + position_ids) and a transformers folder ----
tiny = dict(ip_adapter.VIT_H_14_IMAGE_ENCODER_CONFIG, hidden_size=32, intermediate_size=37, num_attention_heads=4,
            num_hidden_layers=2, projection_dim=16, image_size=32, patch_size=8)
ip_adapter.VIT_H_14_IMAGE_ENCODER_CONFIG = tiny  # same code path as the 2.5 GB ViT-H/14 file
reference = CLIPVisionModelWithProjection(CLIPVisionConfig(**tiny)).eval()
state = {{key: value.contiguous() for key, value in reference.state_dict().items()}}
state["vision_model.embeddings.position_ids"] = reference.vision_model.embeddings.position_ids.clone()
single = tmp / "clip_vision.safetensors"
save_file(state, str(single))
reference.save_pretrained(str(tmp / "image_encoder"))
pixels = torch.rand(2, 3, 32, 32)
with torch.no_grad():
    expected = reference(pixels, output_hidden_states=True)
for source in (single, tmp / "image_encoder"):
    encoder = ip_adapter.load_image_encoder(source, torch_dtype=torch.float32, device="cpu")
    assert not encoder.training, source
    with torch.no_grad():
        actual = encoder(pixels, output_hidden_states=True)
    assert torch.allclose(actual.image_embeds, expected.image_embeds), source
    assert torch.allclose(actual.hidden_states[-2], expected.hidden_states[-2]), source
half = ip_adapter.load_image_encoder(single, torch_dtype=torch.float16, device="cpu")
assert {{p.dtype for p in half.parameters()}} == {{torch.float16}}
assert half.vision_model.embeddings.position_ids.device.type == "cpu"

# -- DiffusionRuntime.ensure_ip_adapter / generate over a tiny random SD 1.x pipeline -------------------------------
unet = UNet2DConditionModel(
    block_out_channels=(4, 8), layers_per_block=1, sample_size=8, in_channels=4, out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32, norm_num_groups=2,
)
vae = AutoencoderKL(
    block_out_channels=[4, 8], in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=2,
    down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2,
)
(tmp / "vocab.json").write_text(json.dumps({{"<|startoftext|>": 0, "<|endoftext|>": 1}}), encoding="utf-8")
(tmp / "merges.txt").write_text("#version: 0.2\\n", encoding="utf-8")
tokenizer = CLIPTokenizer(str(tmp / "vocab.json"), str(tmp / "merges.txt"), model_max_length=77)
text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=2, hidden_size=32, intermediate_size=37, num_attention_heads=4,
                                            num_hidden_layers=2, max_position_embeddings=77))
pipe = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    scheduler=EulerDiscreteScheduler(steps_offset=1), safety_checker=None, feature_extractor=None,
    image_encoder=None, requires_safety_checker=False,
)
pipe.set_progress_bar_config(disable=True)


def write_adapter(name, seed):
    # A base IP-Adapter ("proj" image projection) sized for this UNet, in the h94 safetensors layout.
    generator = torch.Generator().manual_seed(seed)
    weights = {{
        "image_proj.proj.weight": torch.randn(4 * 32, 16, generator=generator),
        "image_proj.proj.bias": torch.zeros(4 * 32),
        "image_proj.norm.weight": torch.ones(32),
        "image_proj.norm.bias": torch.zeros(32),
    }}
    key_id = 1
    for processor in unet.attn_processors:
        if processor.endswith("attn1.processor"):
            continue
        shape = unet.get_submodule(processor[: -len(".processor")]).to_k.weight.shape
        weights[f"ip_adapter.{{key_id}}.to_k_ip.weight"] = torch.randn(*shape, generator=generator)
        weights[f"ip_adapter.{{key_id}}.to_v_ip.weight"] = torch.randn(*shape, generator=generator)
        key_id += 2
    path = tmp / "ipadapter" / name
    path.parent.mkdir(exist_ok=True)
    save_file(weights, str(path))
    return path


first, second = write_adapter("ip-adapter_first.safetensors", 1), write_adapter("ip-adapter_second.safetensors", 2)
red, green = tmp / "red.png", tmp / "green.png"
Image.new("RGB", (40, 24), (255, 0, 0)).save(red)
Image.new("RGB", (24, 40), (0, 255, 0)).save(green)

runtime = DiffusionRuntime(device="cpu", allow_cpu=True)
runtime.pipe = pipe
runtime.current_torch_dtype = torch.float32
encoded_batches = []


def generate(images=None):
    image, _ = runtime.generate(prompt="", negative_prompt="", width=32, height=32, steps=2, guidance_scale=2.0, seed=5,
                                ip_adapter_images=images)
    return np.asarray(image, dtype=np.int16)


base = generate()
spec = IPAdapterSpec(adapter_path=str(first), image_encoder_path=str(single), image_paths=(str(red), str(green)),
                     scale=0.6, architecture="sd15")
references = ip_adapter.open_reference_images(spec.image_paths)
runtime.ensure_ip_adapter(spec)
assert pipe.image_encoder is runtime._image_encoder.module
assert pipe.feature_extractor is not None  # created by diffusers
pipe.image_encoder.register_forward_hook(lambda module, args, output: encoded_batches.append(args[0].shape[0]))
processors = [p for p in unet.attn_processors.values() if type(p).__name__.startswith("IPAdapter")]
assert processors and all(p.scale == [0.6] for p in processors)
with_references = generate(references)
assert encoded_batches == [2], encoded_batches  # both reference images, one batch for the single adapter
assert np.abs(with_references - base).max() > 0, "the IP-Adapter had no effect"

runtime.ensure_ip_adapter(IPAdapterSpec(**{{**spec.__dict__, "scale": 0.0}}))
assert all(p.scale == [0.0] for p in processors)
assert np.array_equal(generate(references), base)  # scale 0: no influence

encoder = pipe.image_encoder
runtime.ensure_ip_adapter(IPAdapterSpec(**{{**spec.__dict__, "adapter_path": str(second)}}))
assert pipe.image_encoder is encoder  # swapped adapter, same encoder module
generate(references[:1])

runtime.ensure_ip_adapter(None)
assert pipe.image_encoder is None
assert not any(type(p).__name__.startswith("IPAdapter") for p in unet.attn_processors.values())
assert np.array_equal(generate(), base)  # unloading restores the base pipeline exactly
print("OK")
"""


def test_ip_adapter_with_the_real_libraries(tmp_path: Path) -> None:
    code = REAL_LIBRARY_CHECKS.format(src_dir=str(SRC_DIR), tmp_dir=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)

    if result.stdout.strip() == "MISSING":
        pytest.skip("torch, transformers, accelerate, safetensors or diffusers is not installed in this environment")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")

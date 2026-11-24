from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable, Literal

from PIL import Image

TorchDTypeName = Literal["float16", "bfloat16", "float32"]
# Pipeline families (same values as checkpoint_inspection.SingleFileArchitecture, which owns the detection rule).
PipelineArchitectureName = Literal["sd15", "sdxl"]
DeviceName = str

DEFAULT_SINGLE_FILE_CONFIG = "stabilityai/stable-diffusion-xl-base-1.0"


@dataclass(frozen=True)
class LoRASpec:
    """A LoRA adapter to load into the diffusion pipeline."""

    id: str
    name: str
    path: str
    weight: float = 1.0

    @classmethod
    def from_path(cls, path: str | Path, *, weight: float = 1.0, id: str | None = None, name: str | None = None) -> "LoRASpec":
        resolved_path = Path(path)
        adapter_name = name or resolved_path.stem
        adapter_hash = hashlib.sha1(resolved_path.resolve().as_posix().casefold().encode("utf-8")).hexdigest()[:12]
        return cls(
            id=id or f"lora::{adapter_name}::{adapter_hash}",
            name=adapter_name,
            path=str(resolved_path),
            weight=float(weight),
        )


@dataclass(frozen=True)
class IPAdapterSpec:
    """IP-Adapter conditioning of one generation: reference images that guide the image through an IP-Adapter.

    - ``adapter_path``: IP-Adapter weights (h94/IP-Adapter ``.safetensors`` / ``.bin``, e.g.
      ``ip-adapter-plus-face_sdxl_vit-h.safetensors``);
    - ``image_encoder_path``: the CLIP vision encoder the adapter was trained with: a single ``.safetensors`` file in
      the transformers key layout (ComfyUI ``clip_vision`` files such as ``CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors``,
      loaded with the ViT-H/14 configuration) or a transformers folder with its ``config.json``;
    - ``image_paths``: reference images (at least one), in order;
    - ``scale``: adapter strength (diffusers ``set_ip_adapter_scale``);
    - ``architecture``: the pipeline family the adapter was made for, checked against the loaded pipeline (None: not
      checked).
    """

    adapter_path: str
    image_encoder_path: str
    image_paths: tuple[str, ...]
    scale: float = 0.6
    architecture: PipelineArchitectureName | None = None


@dataclass(frozen=True)
class ModelSpec:
    """A Diffusers-compatible model directory or a single-file checkpoint."""

    id: str
    name: str
    path: str
    default_width: int = 1024
    default_height: int = 1024
    torch_dtype: TorchDTypeName = "float16"
    single_file_config: str | None = DEFAULT_SINGLE_FILE_CONFIG

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        id: str | None = None,
        name: str | None = None,
        default_width: int = 1024,
        default_height: int = 1024,
        torch_dtype: TorchDTypeName = "float16",
        single_file_config: str | None = DEFAULT_SINGLE_FILE_CONFIG,
    ) -> "ModelSpec":
        resolved_path = Path(path)
        return cls(
            id=id or f"model::{resolved_path.resolve().as_posix().casefold()}::{torch_dtype}",
            name=name or resolved_path.stem,
            path=str(resolved_path),
            default_width=default_width,
            default_height=default_height,
            torch_dtype=torch_dtype,
            single_file_config=single_file_config,
        )


@dataclass(frozen=True)
class KSampleConfig:
    """The ComfyUI-style sampler settings this package maps onto a Diffusers call."""

    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg: float = 7.0
    seed: int | None = None
    sampler_name: str = "euler_ancestral"
    scheduler: str = "simple"

    @property
    def guidance_scale(self) -> float:
        return self.cfg

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class PromptAssetPaths:
    """Optional folders used to resolve inline prompt assets."""

    prompt_lora_dir: str | Path | None = None
    prompt_embedding_dir: str | Path | None = None


@dataclass(frozen=True)
class ResolvedPrompt:
    """Prompt after inline LoRA/embedding tags have been resolved."""

    prompt: str
    negative_prompt: str
    loras: list[LoRASpec] = field(default_factory=list)
    lora_weights: list[float] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result returned by DiffusionGenerator.generate()."""

    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str
    model_path: str
    loras: list[LoRASpec]
    ksample: KSampleConfig
    output_path: str | None = None


@dataclass(frozen=True)
class StepProgress:
    """Progress event emitted after each denoising/sampling step.

    ``step`` is zero at generation start and then 1..``total_steps`` during
    sampling. ``timestep`` is the raw Diffusers scheduler timestep when the
    pipeline provides it.
    """

    step: int
    total_steps: int
    timestep: int | float | None = None

    @property
    def fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, self.step / self.total_steps))


ProgressCallback = Callable[[int, int], None]
StepCallback = Callable[[StepProgress], None]

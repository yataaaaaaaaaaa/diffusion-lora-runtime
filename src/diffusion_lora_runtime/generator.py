from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from PIL import Image

from .diffusion_runtime import AdapterLoadError, DiffusionRuntime, GenerationCancelled
from .prompt_embedding import (
    PromptEmbeddingResolutionError,
    extract_prompt_embedding_names,
    resolve_prompt_embeddings,
)
from .prompt_lora import (
    PromptLoRAResolutionError,
    extract_prompt_lora_specs,
    resolve_prompt_loras,
)
from .schemas import (
    GenerateResult,
    KSampleConfig,
    LoRASpec,
    ModelSpec,
    ProgressCallback,
    StepCallback,
    PromptAssetPaths,
    ResolvedPrompt,
    TorchDTypeName,
    DeviceName,
)

logger = logging.getLogger(__name__)


class DiffusionGenerator:
    """Small reusable runtime for Diffusers model, LoRA, and prompt-resolution generation.

    The public surface is intentionally minimal: call :meth:`generate` and call
    :meth:`unload` when you want to release memory.
    """

    def __init__(self, *, device: DeviceName = "auto", allow_cpu: bool = False) -> None:
        """Create a reusable generator.

        Args:
            device: ``"auto"`` prefers CUDA, then MPS, then CPU. You may also
                pass values such as ``"cuda"``, ``"cuda:0"`` or ``"cpu"``.
            allow_cpu: CPU inference is disabled by default because SDXL/FP16
                CPU fallback commonly appears as a hang. Set this to ``True``
                only for explicit CPU debugging; FP16/BF16 is promoted to FP32
                on CPU.
        """
        self.runtime = DiffusionRuntime(device=device, allow_cpu=allow_cpu)
        self._loaded_signature: tuple[str, tuple[tuple[str, float], ...]] | None = None

    def generate(
        self,
        *,
        prompt: str,
        model_path: str | Path,
        loras: Iterable[str | Path | LoRASpec] | None = None,
        negative_prompt: str = "",
        ksample: KSampleConfig | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        torch_dtype: TorchDTypeName = "float16",
        model_id: str | None = None,
        model_name: str | None = None,
        prompt_lora_dir: str | Path | None = None,
        prompt_embedding_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
        step_callback: StepCallback | None = None,
    ) -> GenerateResult:
        """Generate one image and keep the current model/LoRA set in memory.

        The currently loaded model and LoRA signature is reused across calls.
        If a later call asks for a different model path, dtype, LoRA path, or
        LoRA weight, the previous pipeline is unloaded before the new one is
        prepared. Inline prompt LoRA tags count as part of that LoRA signature.
        """

        config = self._resolve_ksampler(ksample, width=width, height=height, steps=steps, cfg=cfg, seed=seed)
        model = ModelSpec.from_path(
            model_path,
            id=model_id,
            name=model_name,
            default_width=config.width,
            default_height=config.height,
            torch_dtype=torch_dtype,
        )
        explicit_loras = self._normalize_loras(loras or [])
        resolved_prompt = self._resolve_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            explicit_loras=explicit_loras,
            asset_paths=PromptAssetPaths(
                prompt_lora_dir=prompt_lora_dir,
                prompt_embedding_dir=prompt_embedding_dir,
            ),
        )

        signature = self._signature(model, resolved_prompt.loras, resolved_prompt.lora_weights, config)
        with self.runtime.serialized_operation():
            if self._loaded_signature is not None and self._loaded_signature != signature:
                self.runtime.unload()
                self._loaded_signature = None

            self.runtime.configure_ksampler(sampler_name=config.sampler_name, scheduler_name=config.scheduler)
            self.runtime.ensure_model(model)
            self.runtime.ensure_adapters(resolved_prompt.loras, adapter_weights=resolved_prompt.lora_weights)

            embedding_references = []
            if prompt_embedding_dir is not None:
                prompt_resolution = resolve_prompt_embeddings(resolved_prompt.prompt, Path(prompt_embedding_dir))
                negative_resolution = resolve_prompt_embeddings(resolved_prompt.negative_prompt, Path(prompt_embedding_dir))
                embedding_references = list(
                    {
                        str(reference.path.resolve()).casefold(): reference
                        for reference in [*prompt_resolution.references, *negative_resolution.references]
                    }.values()
                )
                if embedding_references:
                    self.runtime.ensure_textual_inversions(embedding_references)
                generation_prompt = prompt_resolution.prompt
                generation_negative_prompt = negative_resolution.prompt
            else:
                generation_prompt = resolved_prompt.prompt
                generation_negative_prompt = resolved_prompt.negative_prompt

            image, actual_seed = self.runtime.generate(
                prompt=generation_prompt,
                negative_prompt=generation_negative_prompt,
                width=config.width,
                height=config.height,
                steps=config.steps,
                guidance_scale=config.guidance_scale,
                seed=config.seed,
                progress_callback=progress_callback,
                step_callback=step_callback,
            )
            self._loaded_signature = signature

        saved_path: str | None = None
        if output_path is not None:
            saved_path = str(output_path)
            self._save_image(image, Path(output_path))

        return GenerateResult(
            image=image,
            seed=actual_seed,
            prompt=generation_prompt,
            negative_prompt=generation_negative_prompt,
            model_path=str(Path(model.path)),
            loras=resolved_prompt.loras,
            ksample=KSampleConfig(
                width=config.width,
                height=config.height,
                steps=config.steps,
                cfg=config.cfg,
                seed=actual_seed,
                sampler_name=config.sampler_name,
                scheduler=config.scheduler,
            ),
            output_path=saved_path,
        )

    def unload(self) -> None:
        """Release any loaded model/LoRA/textual-inversion state and CUDA cache."""
        with self.runtime.serialized_operation():
            self.runtime.cancel_current_generation()
            self.runtime.unload()
            self._loaded_signature = None

    def _resolve_ksampler(
        self,
        ksample: KSampleConfig | None,
        *,
        width: int | None,
        height: int | None,
        steps: int | None,
        cfg: float | None,
        seed: int | None,
    ) -> KSampleConfig:
        config = ksample or KSampleConfig()
        return KSampleConfig(
            width=width if width is not None else config.width,
            height=height if height is not None else config.height,
            steps=steps if steps is not None else config.steps,
            cfg=cfg if cfg is not None else config.cfg,
            seed=seed if seed is not None else config.seed,
            sampler_name=config.sampler_name,
            scheduler=config.scheduler,
        )

    def _normalize_loras(self, loras: Iterable[str | Path | LoRASpec]) -> list[LoRASpec]:
        normalized: list[LoRASpec] = []
        for item in loras:
            if isinstance(item, LoRASpec):
                normalized.append(item)
            else:
                normalized.append(LoRASpec.from_path(item))
        return normalized

    def _resolve_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        explicit_loras: list[LoRASpec],
        asset_paths: PromptAssetPaths,
    ) -> ResolvedPrompt:
        inline_prompt_loras = extract_prompt_lora_specs(prompt)
        prompt_embedding_names = extract_prompt_embedding_names(prompt)
        negative_embedding_names = extract_prompt_embedding_names(negative_prompt)
        logger.info(
            "Generation preflight prompt=%r negative_prompt=%r explicit_loras=%s inline_prompt_loras=%s prompt_embeddings=%s negative_embeddings=%s",
            prompt,
            negative_prompt,
            [spec.id for spec in explicit_loras],
            inline_prompt_loras,
            prompt_embedding_names,
            negative_embedding_names,
        )

        resolved_prompt = prompt
        inline_loras: list[LoRASpec] = []
        inline_weights: list[float] = []
        if asset_paths.prompt_lora_dir is not None:
            prompt_lora_resolution = resolve_prompt_loras(prompt, Path(asset_paths.prompt_lora_dir))
            resolved_prompt = prompt_lora_resolution.prompt
            inline_loras = [reference.to_spec() for reference in prompt_lora_resolution.references]
            inline_weights = [reference.weight for reference in prompt_lora_resolution.references]
        elif inline_prompt_loras:
            raise PromptLoRAResolutionError(
                "Prompt contains inline <lora:name:weight> tags, but prompt_lora_dir was not provided."
            )

        if asset_paths.prompt_embedding_dir is None and (prompt_embedding_names or negative_embedding_names):
            raise PromptEmbeddingResolutionError(
                "Prompt contains inline embedding:name tags, but prompt_embedding_dir was not provided."
            )

        combined_by_path: dict[str, tuple[LoRASpec, float]] = {}
        combined_order: list[str] = []

        def register_adapter(spec: LoRASpec, weight: float) -> None:
            spec_path = str(Path(spec.path).resolve()).casefold()
            if spec_path not in combined_by_path:
                combined_order.append(spec_path)
            combined_by_path[spec_path] = (spec, float(weight))

        for spec in explicit_loras:
            register_adapter(spec, spec.weight)
        for spec, weight in zip(inline_loras, inline_weights, strict=True):
            register_adapter(spec, weight)

        combined_specs = [combined_by_path[item][0] for item in combined_order]
        combined_weights = [combined_by_path[item][1] for item in combined_order]

        return ResolvedPrompt(
            prompt=resolved_prompt,
            negative_prompt=negative_prompt,
            loras=combined_specs,
            lora_weights=combined_weights,
        )

    def _signature(self, model: ModelSpec, loras: list[LoRASpec], weights: list[float], ksample: KSampleConfig) -> tuple[str, tuple[tuple[str, float], ...]]:
        model_key = f"{Path(model.path).resolve().as_posix().casefold()}::{model.torch_dtype}::{ksample.sampler_name}::{ksample.scheduler}"
        lora_key = tuple(
            (Path(spec.path).resolve().as_posix().casefold(), float(weight))
            for spec, weight in zip(loras, weights, strict=True)
        )
        return (model_key, lora_key)

    def _save_image(self, image: Image.Image, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


__all__ = [
    "AdapterLoadError",
    "DiffusionGenerator",
    "GenerationCancelled",
    "PromptEmbeddingResolutionError",
    "PromptLoRAResolutionError",
]

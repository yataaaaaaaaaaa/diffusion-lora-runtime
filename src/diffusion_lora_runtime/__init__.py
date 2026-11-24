"""diffusion-lora-runtime public API.

Public names are resolved lazily (PEP 562): importing the package, or one of its torch-free modules
(``prompt_lora``, ``normalization``, ``prompt_assets``, ``schemas``, ``checkpoint_inspection``), never imports
torch or diffusers. They are only loaded when a name that needs them (``DiffusionGenerator``,
``AdapterLoadError``, ``GenerationCancelled``, ``PromptEmbeddingResolutionError``) is first accessed. Consumers
can therefore reuse the prompt syntax helpers (``diffusion_lora_runtime.prompt_lora``) without loading torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Public name -> submodule defining it.
_EXPORTS: dict[str, str] = {
    "AdapterLoadError": "diffusion_runtime",
    "GenerationCancelled": "diffusion_runtime",
    "DiffusionGenerator": "generator",
    "PromptEmbeddingResolutionError": "prompt_embedding",
    "PromptLoRAResolutionError": "prompt_lora",
    "GenerateResult": "schemas",
    "KSampleConfig": "schemas",
    "LoRASpec": "schemas",
    "ModelSpec": "schemas",
    "PromptAssetPaths": "schemas",
    "StepProgress": "schemas",
    "SafetensorsHeaderError": "checkpoint_inspection",
    "SingleFileArchitecture": "checkpoint_inspection",
    "detect_single_file_architecture": "checkpoint_inspection",
    "read_safetensors_keys": "checkpoint_inspection",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # resolved once
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # static analysers see eager imports
    from .checkpoint_inspection import (
        SafetensorsHeaderError,
        SingleFileArchitecture,
        detect_single_file_architecture,
        read_safetensors_keys,
    )
    from .diffusion_runtime import AdapterLoadError, GenerationCancelled
    from .generator import DiffusionGenerator
    from .prompt_embedding import PromptEmbeddingResolutionError
    from .prompt_lora import PromptLoRAResolutionError
    from .schemas import GenerateResult, KSampleConfig, LoRASpec, ModelSpec, PromptAssetPaths, StepProgress

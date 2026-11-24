from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .normalization import normalize_token
from .prompt_assets import PromptAssetResolutionError, catalog_prompt_asset_files, resolve_prompt_asset_path

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:  # pragma: no cover - optional dependency at import time
    load_safetensors_file = None


PROMPT_EMBEDDING_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_])embedding\s*:\s*([^,\n\r()<>]+?)(?=(?:\s*,|\s*$|\n|\r|\)))",
    re.IGNORECASE,
)


class PromptEmbeddingResolutionError(RuntimeError):
    pass


def extract_prompt_embedding_names(prompt: str) -> list[str]:
    return [match.group(1).strip() for match in PROMPT_EMBEDDING_PATTERN.finditer(prompt)]


@dataclass(frozen=True)
class PromptEmbeddingReference:
    name: str
    path: Path

    @property
    def token(self) -> str:
        normalized_name = normalize_token(self.path.stem) or "prompt_embedding"
        return f"dlr_embed_{normalized_name}"


@dataclass(frozen=True)
class PromptEmbeddingResolution:
    prompt: str
    references: list[PromptEmbeddingReference]


@dataclass(frozen=True)
class PromptEmbeddingVectors:
    clip_l: torch.Tensor
    clip_g: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.clip_l.shape[0])


def _normalize_embedding_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    direct = _normalize_embedding_tensor(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        for nested in value.values():
            tensor = _first_tensor(nested)
            if tensor is not None:
                return tensor
    return None


def load_prompt_embedding_file(path: Path) -> Any:
    suffix = path.suffix.casefold()
    if suffix == ".safetensors":
        if load_safetensors_file is None:
            raise PromptEmbeddingResolutionError(
                "safetensors support is unavailable while loading prompt embeddings."
            )
        return load_safetensors_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _split_generic_embedding(
    generic: torch.Tensor,
    *,
    clip_l_dim: int,
    clip_g_dim: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(clip_l, clip_g) for an embedding stored as one unnamed tensor, matched by its width."""
    width = int(generic.shape[-1])
    if width == clip_l_dim:
        return generic, None
    if clip_g_dim is not None and width == clip_g_dim:
        return None, generic
    expected = " or ".join(str(dim) for dim in (clip_l_dim, clip_g_dim) if dim is not None)
    raise PromptEmbeddingResolutionError(f"Unsupported prompt embedding width {width}; expected {expected}.")


def resolve_prompt_embedding_vectors(
    payload: Any,
    *,
    clip_l_dim: int,
    clip_g_dim: int | None,
) -> PromptEmbeddingVectors:
    """Embedding vectors for the pipeline's text encoder(s).

    ``clip_g_dim`` is the width of the second (OpenCLIP bigG) text encoder of SDXL pipelines, or ``None`` for
    single text-encoder pipelines (SD 1.x): then only CLIP-L vectors are usable and ``clip_g`` is an empty
    ``(tokens, 0)`` tensor.
    """
    clip_l = None
    clip_g = None

    if isinstance(payload, dict):
        clip_l = _normalize_embedding_tensor(payload.get("clip_l"))
        clip_g = _normalize_embedding_tensor(payload.get("clip_g"))

        if clip_l is None and clip_g is None:
            generic = None
            if "string_to_param" in payload:
                generic = _first_tensor(payload.get("string_to_param"))
            if generic is None and "emb_params" in payload:
                generic = _first_tensor(payload.get("emb_params"))
            if generic is None:
                tensor_entries = [
                    tensor
                    for tensor in (_normalize_embedding_tensor(item) for item in payload.values())
                    if tensor is not None
                ]
                if len(tensor_entries) == 1:
                    generic = tensor_entries[0]
            if generic is not None:
                clip_l, clip_g = _split_generic_embedding(generic, clip_l_dim=clip_l_dim, clip_g_dim=clip_g_dim)
    else:
        generic = _normalize_embedding_tensor(payload)
        if generic is not None:
            clip_l, clip_g = _split_generic_embedding(generic, clip_l_dim=clip_l_dim, clip_g_dim=clip_g_dim)

    if clip_l is None and clip_g is None:
        raise PromptEmbeddingResolutionError(
            "Unsupported prompt embedding file format. Expected ComfyUI/SDXL clip_l/clip_g tensors or a single embedding tensor."
        )
    if clip_g_dim is None:
        if clip_l is None:
            raise PromptEmbeddingResolutionError(
                "This prompt embedding only has SDXL clip_g vectors; the loaded model has a single CLIP-L text encoder."
            )
        clip_g = None

    token_count = max(
        int(clip_l.shape[0]) if clip_l is not None else 0,
        int(clip_g.shape[0]) if clip_g is not None else 0,
        1,
    )

    def aligned(tensor: torch.Tensor | None, width: int) -> torch.Tensor:
        if tensor is None:
            return torch.zeros((token_count, width), dtype=torch.float32)
        if tensor.shape[0] == token_count:
            return tensor.to(dtype=torch.float32)
        if tensor.shape[0] > token_count:
            return tensor[:token_count].to(dtype=torch.float32)
        pad = tensor[-1:, :].repeat(token_count - tensor.shape[0], 1)
        return torch.cat([tensor, pad], dim=0).to(dtype=torch.float32)

    return PromptEmbeddingVectors(
        clip_l=aligned(clip_l, clip_l_dim),
        clip_g=aligned(clip_g, clip_g_dim or 0),
    )


def resolve_prompt_embeddings(prompt: str, search_root: Path) -> PromptEmbeddingResolution:
    matches = list(PROMPT_EMBEDDING_PATTERN.finditer(prompt))
    if not matches:
        return PromptEmbeddingResolution(prompt=prompt, references=[])

    try:
        catalog = catalog_prompt_asset_files(
            search_root,
            asset_label="Prompt embedding",
            ignored_suffixes={".txt"},
        )
    except PromptAssetResolutionError as exc:
        raise PromptEmbeddingResolutionError(str(exc)) from exc

    references_by_path: dict[Path, PromptEmbeddingReference] = {}

    def replacer(match: re.Match[str]) -> str:
        raw_name = match.group(1).strip()
        try:
            path = resolve_prompt_asset_path(
                raw_name,
                catalog,
                asset_label="Prompt embedding",
                search_root=search_root,
            )
        except PromptAssetResolutionError as exc:
            raise PromptEmbeddingResolutionError(str(exc)) from exc

        reference = PromptEmbeddingReference(name=raw_name, path=path)
        references_by_path[path] = reference
        return reference.token

    rewritten_prompt = PROMPT_EMBEDDING_PATTERN.sub(replacer, prompt)
    return PromptEmbeddingResolution(prompt=rewritten_prompt, references=list(references_by_path.values()))

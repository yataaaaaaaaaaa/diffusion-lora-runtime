from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .normalization import normalize_prompt_segment, normalize_token
from .prompt_assets import PromptAssetResolutionError, catalog_prompt_asset_files, resolve_prompt_asset_path
from .schemas import LoRASpec


PROMPT_LORA_PATTERN = re.compile(
    r"<\s*lora\s*:\s*([^:>]+?)\s*(?::\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)))?\s*>",
    re.IGNORECASE,
)
PROMPT_LORA_ANY_TAG_PATTERN = re.compile(r"<\s*lora\s*:[^>]*>", re.IGNORECASE)
TXT_SUFFIX = ".txt"


class PromptLoRAResolutionError(RuntimeError):
    pass


def extract_prompt_lora_specs(prompt: str) -> list[tuple[str, float]]:
    references: list[tuple[str, float]] = []
    for match in PROMPT_LORA_PATTERN.finditer(prompt):
        raw_name = match.group(1).strip()
        raw_weight = (match.group(2) or "1.0").strip()
        try:
            weight = float(raw_weight)
        except ValueError:
            weight = 1.0
        references.append((raw_name, weight))
    return references


@dataclass(frozen=True)
class PromptLoRAReference:
    name: str
    weight: float
    path: Path
    replacement_text: str

    @property
    def adapter_id(self) -> str:
        normalized_name = normalize_token(self.path.stem) or "prompt_lora"
        return f"prompt-lora::{normalized_name}"

    def to_spec(self) -> LoRASpec:
        return LoRASpec(id=self.adapter_id, name=self.path.stem, path=str(self.path))


@dataclass(frozen=True)
class PromptLoRAResolution:
    prompt: str
    references: list[PromptLoRAReference]


def _read_optional_sidecar(path: Path) -> str:
    txt_path = path.with_suffix(TXT_SUFFIX)
    if not txt_path.is_file():
        return ""
    return txt_path.read_text(encoding="utf-8").strip()


def _cleanup_prompt_text(text: str) -> str:
    cleaned = re.sub(r",\s*,+", ",", text)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",[ \t]+", ", ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return normalize_prompt_segment(cleaned)


def resolve_prompt_loras(prompt: str, search_root: Path) -> PromptLoRAResolution:
    matches = list(PROMPT_LORA_PATTERN.finditer(prompt))
    if not matches:
        if PROMPT_LORA_ANY_TAG_PATTERN.search(prompt):
            raise PromptLoRAResolutionError(
                "Invalid A1111 LoRA syntax. Expected '<lora:name:weight>' or '<lora:name>'."
            )
        return PromptLoRAResolution(prompt=prompt, references=[])

    try:
        catalog = catalog_prompt_asset_files(
            search_root,
            asset_label="Prompt LoRA",
            ignored_suffixes={TXT_SUFFIX},
        )
    except PromptAssetResolutionError as exc:
        raise PromptLoRAResolutionError(str(exc)) from exc
    references_by_path: dict[Path, PromptLoRAReference] = {}

    def replacer(match: re.Match[str]) -> str:
        raw_name = match.group(1).strip()
        raw_weight = (match.group(2) or "1.0").strip()
        try:
            path = resolve_prompt_asset_path(
                raw_name,
                catalog,
                asset_label="Prompt LoRA",
                search_root=search_root,
            )
        except PromptAssetResolutionError as exc:
            raise PromptLoRAResolutionError(str(exc)) from exc
        reference = PromptLoRAReference(
            name=raw_name,
            weight=float(raw_weight),
            path=path,
            replacement_text=_read_optional_sidecar(path),
        )
        references_by_path[path] = reference
        return reference.replacement_text

    try:
        rewritten_prompt = PROMPT_LORA_PATTERN.sub(replacer, prompt)
    except ValueError as exc:
        raise PromptLoRAResolutionError(str(exc)) from exc

    return PromptLoRAResolution(
        prompt=_cleanup_prompt_text(rewritten_prompt),
        references=list(references_by_path.values()),
    )

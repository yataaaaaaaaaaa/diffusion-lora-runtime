from __future__ import annotations

from pathlib import Path


class PromptAssetResolutionError(RuntimeError):
    pass


def catalog_prompt_asset_files(search_root: Path, *, asset_label: str, ignored_suffixes: set[str] | None = None) -> dict[str, list[Path]]:
    ignored = {suffix.casefold() for suffix in (ignored_suffixes or set())}
    if not search_root.exists() or not search_root.is_dir():
        raise PromptAssetResolutionError(f"{asset_label} folder '{search_root}' does not exist.")

    candidates = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.casefold() not in ignored
    ]
    if not candidates:
        raise PromptAssetResolutionError(f"{asset_label} folder '{search_root}' is empty.")

    catalog: dict[str, list[Path]] = {}
    for candidate in candidates:
        catalog.setdefault(candidate.stem.casefold(), []).append(candidate)
    return catalog


def resolve_prompt_asset_path(name: str, catalog: dict[str, list[Path]], *, asset_label: str, search_root: Path) -> Path:
    candidates = catalog.get(name.casefold(), [])
    if not candidates:
        raise PromptAssetResolutionError(f"{asset_label} '{name}' was not found in '{search_root}'.")
    if len(candidates) > 1:
        resolved = ", ".join(str(candidate) for candidate in sorted(candidates))
        raise PromptAssetResolutionError(f"{asset_label} '{name}' is ambiguous in '{search_root}': {resolved}")
    return candidates[0]

from __future__ import annotations

import re


def normalize_token(token: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", token).strip("_").lower()
    return re.sub(r"_+", "_", normalized)


def prompt_to_key(category_name: str, text: str) -> str:
    category_part = normalize_token(category_name) or "category"
    prompt_parts = sorted(filter(None, (normalize_token(token) for token in text.split(","))))
    if not prompt_parts:
        return category_part
    return f"{category_part}, {', '.join(prompt_parts)}"


def normalize_text_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_contains(haystack: str, needle: str) -> bool:
    normalized_needle = normalize_text_for_match(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalize_text_for_match(haystack)


def split_activation_terms(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def priority_to_sort_key(priority: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in priority.split("."):
        raw_part = raw_part.strip()
        if not raw_part:
            continue
        if raw_part.isdigit():
            parts.append(int(raw_part))
            continue
        digits = "".join(character for character in raw_part if character.isdigit())
        parts.append(int(digits) if digits else 10_000)
    return tuple(parts or [10_000])


def normalize_prompt_segment(text: str) -> str:
    collapsed = re.sub(r",\s*,+", ",", text)
    return collapsed.strip(" \t\r\n,")


def join_prompt_segments(parts: list[str]) -> str:
    normalized = [normalize_prompt_segment(part) for part in parts]
    return ",\n".join(part for part in normalized if part)

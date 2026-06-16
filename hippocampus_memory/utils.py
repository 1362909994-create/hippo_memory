from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def iso_after_days(days: int | None) -> str | None:
    if days is None:
        return None
    return (datetime.now(UTC) + timedelta(days=days)).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def content_hash(content: str) -> str:
    normalized = normalize_text(content).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex}"


def dumps_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def estimate_tokens(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    if model:
        tiktoken_count = _estimate_tiktoken(text, model)
        if tiktoken_count is not None:
            return tiktoken_count
    ascii_words = re.findall(r"[A-Za-z0-9_]+", text)
    non_ascii = re.findall(r"[^\x00-\x7f]", text)
    punctuation = re.findall(r"[,.!?;:，。！？；：]", text)
    return max(1, len(ascii_words) + len(non_ascii) + len(punctuation) // 2)


def token_counter_name(model: str | None = None) -> str:
    if not model:
        return "heuristic"
    if _tiktoken_available(model):
        return f"tiktoken:{model}"
    return f"heuristic:fallback-for-{model}"


def _estimate_tiktoken(text: str, model: str) -> int | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _tiktoken_available(model: str) -> bool:
    return _estimate_tiktoken("", model) is not None


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def tokenize(text: str) -> list[str]:
    lowered = text.casefold()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", lowered)
    return _dedupe_preserving_order([*ascii_tokens, *cjk_search_terms(lowered), *cjk_tokens])


def cjk_search_terms(text: str) -> list[str]:
    sequences = re.findall(r"[\u4e00-\u9fff]+", text.casefold())
    terms: list[str] = []
    for sequence in sequences:
        terms.append(sequence)
        terms.extend(_jieba_terms(sequence))
        terms.extend(_cjk_ngrams(sequence))
    return _dedupe_preserving_order(terms)


def text_similarity(left: str, right: str) -> float:
    """Return a 0..1 near-duplicate similarity score.

    rapidfuzz is used when installed; otherwise this falls back to token-set
    Jaccard/containment so the default install stays light and offline.
    """

    normalized_left = normalize_text(left).casefold()
    normalized_right = normalize_text(right).casefold()
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    try:
        from rapidfuzz import fuzz
    except ImportError:
        left_tokens = set(tokenize(normalized_left))
        right_tokens = set(tokenize(normalized_right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        jaccard = overlap / len(left_tokens | right_tokens)
        containment = overlap / min(len(left_tokens), len(right_tokens))
        return max(jaccard, containment)
    return max(
        fuzz.ratio(normalized_left, normalized_right),
        fuzz.token_set_ratio(normalized_left, normalized_right),
    ) / 100.0


def _jieba_terms(text: str) -> list[str]:
    try:
        import jieba
    except ImportError:
        return []
    return [term for term in jieba.cut(text) if len(term) > 1]


def _cjk_ngrams(text: str) -> list[str]:
    grams: list[str] = []
    for width in (2, 3):
        if len(text) < width:
            continue
        grams.extend(text[index : index + width] for index in range(0, len(text) - width + 1))
    return grams


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output

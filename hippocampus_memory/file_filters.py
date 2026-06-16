from __future__ import annotations

from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".hippo.toml",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".cache",
    ".hippo",
    ".next",
    ".idea",
    ".vscode",
    ".reasonix",
}

GENERATED_DIR_SUFFIXES = (
    ".egg-info",
    ".dist-info",
)

INDEXABLE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".cs",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".onnx",
    ".bin",
    ".pt",
    ".pth",
    ".zip",
    ".7z",
    ".exe",
    ".dll",
}


def should_ignore_path(path: Path) -> bool:
    return any(
        part in IGNORED_DIRS or part.endswith(GENERATED_DIR_SUFFIXES)
        for part in path.parts
    )


def is_indexable_file(path: Path, max_size_bytes: int = 1_000_000) -> bool:
    if should_ignore_path(path):
        return False
    if path.suffix.casefold() in BINARY_EXTENSIONS:
        return False
    if path.suffix.casefold() not in INDEXABLE_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > max_size_bytes:
            return False
    except OSError:
        return False
    return True

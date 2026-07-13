"""Repository inspection utilities for the SPPIG simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


RELEVANT_NAME_HINTS = (
    "env",
    "dynamics",
    "policy",
    "controller",
    "simulation",
    "train",
    "experiment",
    "teacher",
    "student",
    "psm",
)
IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".tox",
    "artifacts",
}
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".md",
    ".ipynb",
    ".sh",
}


@dataclass
class RepoCandidate:
    path: str
    suffix: str
    size_bytes: int
    score: int
    hints: List[str]


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def scan_repo(repo_root: Path, max_files: int = 400) -> Dict[str, object]:
    """Build a compact manifest of relevant source/config/readme files."""

    repo_root = repo_root.resolve()
    candidates: List[RepoCandidate] = []
    skipped_large = 0
    total_seen = 0
    by_suffix: Dict[str, int] = {}

    for path in repo_root.rglob("*"):
        rel = path.relative_to(repo_root)
        if _is_ignored(rel) or not path.is_file():
            continue
        total_seen += 1
        suffix = path.suffix.lower()
        by_suffix[suffix or "<none>"] = by_suffix.get(suffix or "<none>", 0) + 1
        if suffix not in TEXT_SUFFIXES and not path.name.lower().startswith("readme"):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 250_000:
            skipped_large += 1
            continue
        lowered = str(rel).lower()
        hints = [hint for hint in RELEVANT_NAME_HINTS if hint in lowered]
        score = 10 * len(hints)
        if suffix == ".py":
            score += 5
        if path.name.lower().startswith("readme"):
            score += 3
        if hints or suffix in TEXT_SUFFIXES or path.name.lower().startswith("readme"):
            candidates.append(
                RepoCandidate(str(rel), suffix or "<none>", int(size), score, hints)
            )

    candidates.sort(key=lambda item: (-item.score, item.path))
    selected = candidates[:max_files]
    return {
        "repo_root": str(repo_root),
        "total_files_seen": total_seen,
        "skipped_large_text_files": skipped_large,
        "selected_count": len(selected),
        "file_type_counts": dict(sorted(by_suffix.items())),
        "preferred_name_hints": list(RELEVANT_NAME_HINTS),
        "files": [candidate.__dict__ for candidate in selected],
    }

import os
from pathlib import Path


def _resolve_against_anchor_suffixes(path: Path, anchor_dir: Path) -> Path | None:
    for suffix_length in range(1, len(path.parts) + 1):
        candidate = (anchor_dir / Path(*path.parts[-suffix_length:])).resolve(strict=False)
        if candidate.exists():
            return candidate
    return None


def resolve_stored_path(path_value, anchor_dir=None, cwd=None) -> Path:
    """Resolve a stored path against cwd first, then an anchor directory."""
    path = Path(path_value).expanduser()
    if anchor_dir is not None:
        anchor_match = _resolve_against_anchor_suffixes(path, Path(anchor_dir))
        if anchor_match is not None:
            return anchor_match

    if path.is_absolute():
        return path.resolve(strict=False)

    search_roots = [Path.cwd() if cwd is None else Path(cwd)]
    if anchor_dir is not None:
        anchor_path = Path(anchor_dir)
        if anchor_path not in search_roots:
            search_roots.append(anchor_path)

    for root in search_roots:
        candidate = (root / path).resolve(strict=False)
        if candidate.exists():
            return candidate

    return (search_roots[0] / path).resolve(strict=False)


def to_cwd_relative_path(path_value, cwd=None) -> str:
    """Serialize a path relative to the current working directory."""
    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    path = Path(path_value).expanduser()
    resolved_path = path.resolve(strict=False) if path.is_absolute() else (cwd_path / path).resolve(strict=False)
    return os.path.relpath(resolved_path, cwd_path)
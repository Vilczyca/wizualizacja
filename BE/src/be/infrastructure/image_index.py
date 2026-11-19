import os
from pathlib import Path


def build_file_index(images_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Buduje szybki indeks ścieżek obrazów za pomocą os.scandir (bez nadmiarowych obiektów Path)."""
    suffix_index: dict[str, Path] = {}
    basename_index: dict[str, Path] = {}

    if not images_root.exists():
        return suffix_index, basename_index

    root_str = os.fspath(images_root)
    stack: list[tuple[str, str]] = [(root_str, "")]

    while stack:
        current_dir, rel_prefix = stack.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        next_prefix = f"{rel_prefix}/{name}" if rel_prefix else name
                        stack.append((entry.path, next_prefix))
                    elif entry.is_file(follow_symlinks=False):
                        full_path = Path(entry.path)
                        name_lower = name.lower()
                        rel_posix = (
                            f"{rel_prefix}/{name}".lower() if rel_prefix else name_lower
                        )
                        suffix_index[rel_posix] = full_path
                        if rel_prefix:
                            parent_lower = rel_prefix.rsplit("/", 1)[-1].lower()
                            suffix_index.setdefault(
                                f"{parent_lower}/{name_lower}", full_path
                            )
                        basename_index.setdefault(name_lower, full_path)
        except OSError:
            continue

    return suffix_index, basename_index


def resolve_image_path(
    rel_path: str,
    suffix_index: dict[str, Path],
    basename_index: dict[str, Path],
) -> Path | None:
    """Dopasowuje względną ścieżkę obrazu do fizycznego pliku w indeksie."""
    norm = rel_path.replace("\\", "/").lstrip("/").lower()
    if norm in suffix_index:
        return suffix_index[norm]

    parts = norm.split("/")
    if len(parts) >= 2:
        tail_two = f"{parts[-2]}/{parts[-1]}"
        if tail_two in suffix_index:
            return suffix_index[tail_two]

    return basename_index.get(parts[-1])

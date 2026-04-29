"""Test-dataset analyzer + server-side folder browser.

Two layouts supported:
  - Hugging Face dataset (`datasets.load_from_disk` directory). This is what
    `fl_app.data.build_loader` consumes.
  - ImageFolder (`<dataset>/<class_name>/*.jpg|png`). Universal — anyone with
    a labelled image dataset can drop it in.

Returns a structure safe to expose publicly: format, num_samples,
num_classes, class_names, image_size, image_mode.

Browser is sandboxed to REPO_ROOT — admin cannot escape outside the project tree.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "data"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cap for mean/std estimation: sampling is fine, the test set is large enough
# that a 1k-image estimate is within ~1e-3 of the population value.
_NORM_SAMPLE_CAP = 1000


def _norm_stats(images: Iterable[Image.Image]) -> tuple[list[float], list[float]] | tuple[None, None]:
    """Pixel mean/std on [0,1] scale, per channel. Skips broken images."""
    chan_sum = np.zeros(3, dtype=np.float64)
    chan_sqsum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for img in images:
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            chan_sum += arr.reshape(-1, 3).sum(axis=0)
            chan_sqsum += (arr.reshape(-1, 3) ** 2).sum(axis=0)
            n_pixels += arr.shape[0] * arr.shape[1]
        except Exception:
            continue
    if n_pixels == 0:
        return None, None
    mean = chan_sum / n_pixels
    var = np.maximum(chan_sqsum / n_pixels - mean ** 2, 0.0)
    std = np.sqrt(var)
    return [round(float(x), 4) for x in mean], [round(float(x), 4) for x in std]


class DatasetAnalysisError(ValueError):
    pass


def browse_directory(rel_path: str) -> dict[str, Any]:
    """List subdirectories of REPO_ROOT/rel_path. Sandboxed to DATASETS_ROOT (data/).

    Empty rel_path → contents of `data/`. Returned paths stay repo-relative so
    they can be passed verbatim to `analyze_dataset` afterwards.
    Raises DatasetAnalysisError if path is outside `data/` or not a directory.
    """
    repo = REPO_ROOT.resolve()
    sandbox = DATASETS_ROOT.resolve()
    candidate = sandbox if not rel_path else (repo / rel_path).resolve()
    try:
        candidate.relative_to(sandbox)
    except ValueError as e:
        raise DatasetAnalysisError(
            "Path is outside the data/ folder; admin browsing is sandboxed there."
        ) from e
    if not candidate.exists():
        raise DatasetAnalysisError(f"Path does not exist: {candidate}")
    if not candidate.is_dir():
        raise DatasetAnalysisError(f"Path is not a directory: {candidate}")

    rel = candidate.relative_to(repo)
    subdirs = sorted(
        p.name
        for p in candidate.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    )
    parent_rel: str | None = None
    if candidate != sandbox:
        parent = candidate.parent
        parent_rel = str(parent.relative_to(repo))
    return {
        "path": str(rel),
        "parent": parent_rel,
        "subdirs": subdirs,
    }


def analyze_dataset(path_str: str) -> dict[str, Any]:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise DatasetAnalysisError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise DatasetAnalysisError(f"Path is not a directory: {path}")

    # Try HuggingFace `load_from_disk` first.
    if (path / "dataset_info.json").exists() or (path / "state.json").exists():
        try:
            info = _analyze_hf(path)
        except Exception as e:
            raise DatasetAnalysisError(f"HF dataset detected but failed to load: {e}") from e
    else:
        info = _analyze_imagefolder(path)

    info["name"] = _derive_name(path)
    return info


_SPLIT_NAMES = {"test", "train", "val", "validation", "eval"}


def _derive_name(path: Path) -> str:
    last = path.name
    if last.lower() in _SPLIT_NAMES and path.parent.name:
        return path.parent.name
    return last or str(path)


def _analyze_hf(path: Path) -> dict[str, Any]:
    from datasets import load_from_disk  # heavy import, defer

    ds = load_from_disk(str(path))
    # Resolve label feature: first ClassLabel-typed column.
    label_feature = None
    label_col = None
    for col, feat in ds.features.items():
        if hasattr(feat, "names"):
            label_feature = feat
            label_col = col
            break
    class_names: list[str] = list(getattr(label_feature, "names", []) or [])

    # Image column: first PIL.Image-like feature, fallback to "image"/"img".
    image_col = None
    for col, feat in ds.features.items():
        if type(feat).__name__ == "Image":
            image_col = col
            break
    if image_col is None:
        for cand in ("image", "img"):
            if cand in ds.features:
                image_col = cand
                break

    image_size: list[int] | None = None
    image_mode: str | None = None
    if image_col is not None and len(ds) > 0:
        sample = ds[0][image_col]
        if isinstance(sample, Image.Image):
            image_size = [sample.height, sample.width]
            image_mode = sample.mode

    mean: list[float] | None = None
    std: list[float] | None = None
    if image_col is not None and len(ds) > 0:
        rng = random.Random(0)
        n = min(len(ds), _NORM_SAMPLE_CAP)
        idxs = rng.sample(range(len(ds)), n) if len(ds) > n else range(len(ds))
        mean, std = _norm_stats(ds[i][image_col] for i in idxs)

    return {
        "format": "huggingface",
        "num_samples": len(ds),
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_column": label_col,
        "image_column": image_col,
        "image_size": image_size,
        "image_mode": image_mode,
        "mean": mean,
        "std": std,
    }


def _analyze_imagefolder(path: Path) -> dict[str, Any]:
    class_dirs = sorted(p for p in path.iterdir() if p.is_dir())
    if not class_dirs:
        raise DatasetAnalysisError(
            f"No class subdirectories found in {path}. Expected ImageFolder layout: "
            "<root>/<class_name>/<image_files>."
        )

    class_names: list[str] = []
    num_samples = 0
    sample_image: Path | None = None
    for cd in class_dirs:
        images = [p for p in cd.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue
        class_names.append(cd.name)
        num_samples += len(images)
        if sample_image is None:
            sample_image = images[0]

    if not class_names:
        raise DatasetAnalysisError(
            f"No image files found under any class subdirectory of {path}."
        )

    image_size: list[int] | None = None
    image_mode: str | None = None
    if sample_image is not None:
        try:
            with Image.open(sample_image) as im:
                image_size = [im.height, im.width]
                image_mode = im.mode
        except Exception:
            pass

    all_images: list[Path] = []
    for cd in class_dirs:
        all_images.extend(
            p for p in cd.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    rng = random.Random(0)
    n = min(len(all_images), _NORM_SAMPLE_CAP)
    sample_paths = rng.sample(all_images, n) if len(all_images) > n else all_images

    def _open_iter() -> Iterable[Image.Image]:
        for p in sample_paths:
            try:
                with Image.open(p) as im:
                    yield im.copy()
            except Exception:
                continue

    mean, std = _norm_stats(_open_iter())

    return {
        "format": "imagefolder",
        "num_samples": num_samples,
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_column": None,
        "image_column": None,
        "image_size": image_size,
        "image_mode": image_mode,
        "mean": mean,
        "std": std,
    }

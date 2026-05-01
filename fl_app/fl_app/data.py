"""Contract-driven data loader.

Layout-agnostic: works on HuggingFace `load_from_disk` directories (server's
test set, sim partitions) and ImageFolder layouts (real client `/data`).
The canonical class→index mapping is fixed by the project contract — same
mapping used by every server and every client, no per-host divergence.

Contract shape (delivered via `_fl_contract.json`):
    {
      "class_names": [...],   # canonical order; index in this list IS the label
      "image_size":  [H, W],
      "image_mode":  "RGB",
      "mean":        [r, g, b],
      "std":         [r, g, b],
    }
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CONTRACT_FILENAME = "_fl_contract.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_contract(path: str | Path) -> dict[str, Any]:
    """Read contract JSON. Accepts a path to the file or a directory holding it."""
    p = Path(path)
    if p.is_dir():
        p = p / CONTRACT_FILENAME
    return json.loads(p.read_text())


def _is_hf_layout(d: Path) -> bool:
    return (d / "dataset_info.json").exists() or (d / "state.json").exists()


def _build_transforms(*, image_size, mean, std, train: bool):
    # Train aug branches by image size: ≤64px → CIFAR recipe (RandomCrop+pad),
    # >64px → ImageNet recipe (RandomResizedCrop). Eval is always plain Resize.
    h, w = int(image_size[0]), int(image_size[1])
    ops: list = []
    if train and h > 64:
        ops.append(transforms.RandomResizedCrop((h, w), scale=(0.8, 1.0)))
    else:
        ops.append(transforms.Resize((h, w)))
    if train:
        if h <= 64:
            ops.append(transforms.RandomCrop(h, padding=4))
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(tuple(mean), tuple(std))]
    return transforms.Compose(ops)


class _HFDataset(Dataset):
    def __init__(self, hf_ds, tf, img_col: str, label_col: str):
        self.ds = hf_ds
        self.tf = tf
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.img_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), int(item[self.label_col])


class _ImageFolderDataset(Dataset):
    """ImageFolder root → samples list with canonical class indices.

    Subdirs whose name is NOT in contract.class_names are silently dropped
    (caller already validated the layout via the bootstrap step). Subdirs
    matching a class name contribute all their image files.
    """

    def __init__(self, root: Path, class_names: list[str], tf):
        self.tf = tf
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        self.skipped_extras: list[str] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            cidx = name_to_idx.get(sub.name)
            if cidx is None:
                self.skipped_extras.append(sub.name)
                continue
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((p, cidx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        p, label = self.samples[i]
        with Image.open(p) as im:
            img = im.convert("RGB") if im.mode != "RGB" else im.copy()
        return self.tf(img), label

    def labels(self) -> list[int]:
        return [s[1] for s in self.samples]


def _stratified_indices(labels, fraction: float, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for _, idxs in by_class.items():
        k = max(1, int(round(len(idxs) * fraction)))
        rng.shuffle(idxs)
        picked.extend(idxs[:k])
    picked.sort()
    return picked


def build_loader(
    data_dir: str | Path,
    *,
    batch_size: int,
    train: bool,
    contract: dict[str, Any],
    chunk_fraction: float = 1.0,
    chunk_seed: int = 0,
) -> DataLoader:
    data_dir = Path(data_dir)
    class_names = contract["class_names"]
    tf = _build_transforms(
        image_size=contract["image_size"],
        mean=contract["mean"],
        std=contract["std"],
        train=train,
    )

    if _is_hf_layout(data_dir):
        from datasets import load_from_disk

        hf_ds = load_from_disk(str(data_dir))
        img_col = _resolve_image_column(hf_ds)
        label_col = _resolve_label_column(hf_ds)
        if train and chunk_fraction < 1.0:
            idx = _stratified_indices(hf_ds[label_col], chunk_fraction, chunk_seed)
            hf_ds = hf_ds.select(idx)
        ds: Dataset = _HFDataset(hf_ds, tf, img_col, label_col)
    else:
        full = _ImageFolderDataset(data_dir, class_names, tf)
        if train and chunk_fraction < 1.0:
            idx = _stratified_indices(full.labels(), chunk_fraction, chunk_seed)
            ds = torch.utils.data.Subset(full, idx)
        else:
            ds = full

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def count_labels(data_dir: str | Path, *, contract: dict[str, Any]) -> dict[int, int]:
    """Class-index histogram of the data at `data_dir`. Used by data-profiling."""
    data_dir = Path(data_dir)
    if _is_hf_layout(data_dir):
        from datasets import load_from_disk

        hf_ds = load_from_disk(str(data_dir))
        label_col = _resolve_label_column(hf_ds)
        counts: dict[int, int] = {}
        for y in hf_ds[label_col]:
            counts[int(y)] = counts.get(int(y), 0) + 1
        return counts
    name_to_idx = {n: i for i, n in enumerate(contract["class_names"])}
    counts = {}
    for sub in data_dir.iterdir():
        if not sub.is_dir():
            continue
        cidx = name_to_idx.get(sub.name)
        if cidx is None:
            continue
        n = sum(1 for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if n > 0:
            counts[cidx] = n
    return counts


def _resolve_image_column(hf_ds) -> str:
    for col, feat in hf_ds.features.items():
        if type(feat).__name__ == "Image":
            return col
    for cand in ("image", "img"):
        if cand in hf_ds.features:
            return cand
    raise ValueError(f"No image column in HF dataset. Features: {list(hf_ds.features.keys())}")


def _resolve_label_column(hf_ds) -> str:
    for col, feat in hf_ds.features.items():
        if hasattr(feat, "names"):
            return col
    for cand in ("label", "labels", "fine_label"):
        if cand in hf_ds.features:
            return cand
    raise ValueError(f"No label column in HF dataset. Features: {list(hf_ds.features.keys())}")

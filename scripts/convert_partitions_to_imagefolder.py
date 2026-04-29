"""Convert HuggingFace partitions to ImageFolder layout.

Each client's partition (HF `load_from_disk` directory) becomes
`<dst_root>/client_<N>/<class_name>/<idx>.png`. Class names come from the
HF ClassLabel feature; PNG is lossless and matches the project contract
size, so the converted partition mirrors what a real client would have on
their machine.

Server's `test/` partition is intentionally NOT converted — it stays HF.
fl_app/data.py reads both formats; the server side keeps HF.

Usage:
    python -m scripts.convert_partitions_to_imagefolder \\
        data/partitions/cifar100__iid__n10__s42 \\
        data/partitions/cifar100__iid__n10__s42__imagefolder
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_from_disk


def _resolve_columns(ds) -> tuple[str, str, list[str]]:
    img_col = next(
        (col for col, feat in ds.features.items() if type(feat).__name__ == "Image"),
        None,
    )
    if img_col is None:
        for cand in ("image", "img"):
            if cand in ds.features:
                img_col = cand
                break
    if img_col is None:
        raise ValueError(f"No image column. Features: {list(ds.features.keys())}")

    label_col = next(
        (col for col, feat in ds.features.items() if hasattr(feat, "names")), None
    )
    if label_col is None:
        raise ValueError(f"No ClassLabel column. Features: {list(ds.features.keys())}")
    class_names = list(ds.features[label_col].names)
    return img_col, label_col, class_names


def convert_partition(src: Path, dst: Path) -> int:
    ds = load_from_disk(str(src))
    img_col, label_col, class_names = _resolve_columns(ds)
    dst.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {n: 0 for n in class_names}
    for i, item in enumerate(ds):
        cls_id = int(item[label_col])
        cls_name = class_names[cls_id]
        cls_dir = dst / cls_name
        if counts[cls_name] == 0:
            cls_dir.mkdir(parents=True, exist_ok=True)
        out = cls_dir / f"{i:06d}.png"
        img = item[img_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out, format="PNG")
        counts[cls_name] += 1
    return len(ds)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python -m scripts.convert_partitions_to_imagefolder "
            "<src_partitions_dir> <dst_imagefolder_dir>",
            file=sys.stderr,
        )
        return 2
    src_root = Path(sys.argv[1]).expanduser().resolve()
    dst_root = Path(sys.argv[2]).expanduser().resolve()

    if not src_root.is_dir():
        print(f"src not a directory: {src_root}", file=sys.stderr)
        return 1
    dst_root.mkdir(parents=True, exist_ok=True)

    client_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("client_"))
    if not client_dirs:
        print(f"no client_* subdirs in {src_root}", file=sys.stderr)
        return 1

    total = 0
    for cd in client_dirs:
        out_dir = dst_root / cd.name
        n = convert_partition(cd, out_dir)
        total += n
        print(f"  {cd.name}: {n} samples → {out_dir}")
    print(f"done: {total} samples across {len(client_dirs)} partitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

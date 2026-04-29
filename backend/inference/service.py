"""Lazy-loaded model cache + predict pipeline."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
from torch import nn

from backend.inference.datasets import get_class_names, get_eval_transform, open_rgb


log = logging.getLogger(__name__)

# Repo root — used to resolve relative weights paths from the DB.
REPO_ROOT = Path(__file__).resolve().parents[2]


class _ModelCache:
    """Caches loaded torch models keyed by (model_name, weights_path)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], nn.Module] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, weights_path: str) -> nn.Module:
        key = (model_name, weights_path)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        # Build outside the lock — torch.load is slow.
        from fl_app.models import build_model
        model = build_model(model_name)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        log.info("loaded model %s from %s", model_name, weights_path)
        with self._lock:
            # Race-safe — another thread might have populated the cache.
            self._cache.setdefault(key, model)
            return self._cache[key]

    def evict(self, model_name: str, weights_path: str) -> None:
        with self._lock:
            self._cache.pop((model_name, weights_path), None)


_cache = _ModelCache()


def resolve_weights_path(weights_path: str) -> Path:
    """Allow either absolute paths or paths relative to the repo root."""
    p = Path(weights_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def predict(
    *,
    model_name: str,
    dataset: str,
    weights_path: str,
    image_bytes: bytes,
    top_k: int = 5,
) -> list[dict]:
    """Returns a list of {class_id, class_name, confidence} sorted by confidence desc."""
    resolved = resolve_weights_path(weights_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"weights file not found: {resolved}")

    model = _cache.get(model_name, str(resolved))
    transform = get_eval_transform(dataset)
    class_names = get_class_names(dataset)

    img = open_rgb(image_bytes)
    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top = torch.topk(probs, k=min(top_k, probs.shape[0]))

    return [
        {
            "class_id": int(idx),
            "class_name": class_names[int(idx)] if int(idx) < len(class_names) else str(int(idx)),
            "confidence": float(probs[int(idx)]),
        }
        for idx in top.indices.tolist()
    ]

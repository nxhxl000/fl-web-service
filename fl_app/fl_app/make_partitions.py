"""make_partitions.py — создание партиций для FL-экспериментов.

Единственная точка входа для создания клиентских партиций в data/partitions/.

Методы (каждый — запись в реестре METHODS):
    iid                — равномерное, стратифицированное по классам
    dirichlet          — Dirichlet(α) per class; params: alpha, min_per_class
    quantity           — volume skew (IID-метки, Dir(β) размеры); params: beta
    dirichlet_quantity — label+quantity skew (Dir(α) × Dir(β)); params: alpha, beta
    meta2-disjoint     — CIFAR-100: 2 суперкласса на клиента (disjoint)
    meta1-iid          — CIFAR-100: 1 эксклюзивный суперкласс + 10 IID
    meta1-dirichlet    — CIFAR-100: 1 эксклюзивный суперкласс + 10 Dir(α); params: alpha
    structural-mono    — PlantVillage: культуры без пересечений
    structural-family  — PlantVillage: 20% overlap внутри семьи
    structural-mix     — PlantVillage: 40% overlap внутри группы

Добавление нового метода: написать build-функцию, зарегистрировать через
register(Method(...)) в секции 6. Никаких правок в существующих методах/пайплайне.

Запуск:
    python make_partitions.py cifar100
    python make_partitions.py plantvillage
    python make_partitions.py cifar100 --only quantity
    python make_partitions.py cifar100 --only dirichlet --alphas 0.05 0.2
    python make_partitions.py --list
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    label_col: str
    img_col: str
    has_test: bool
    test_fraction: float = 0.2


DATASET_CONFIG: dict[str, DatasetConfig] = {
    "cifar100": DatasetConfig(
        label_col="fine_label", img_col="img", has_test=True,
    ),
    "plantvillage": DatasetConfig(
        label_col="label", img_col="image", has_test=False,
        test_fraction=0.15,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.strip().lower()
    if key not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_CONFIG)}."
        )
    return DATASET_CONFIG[key]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN/TEST + SERVER-DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _validate_label_col(dataset: Dataset, label_col: str) -> None:
    if label_col not in dataset.features:
        raise KeyError(
            f"Column '{label_col}' not found. "
            f"Available: {sorted(dataset.features.keys())}."
        )


def _stratified_train_test_split(
    dataset: Dataset, test_fraction: float, label_col: str, seed: int,
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    labels = np.array(dataset[label_col])
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(np.ceil(len(cls_idx) * test_fraction)))
        n_test = min(n_test, len(cls_idx) - 1)
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    return dataset.select(sorted(train_idx)), dataset.select(sorted(test_idx))


def prepare_splits(
    ds: DatasetDict | Dataset, dataset_name: str, seed: int = 42,
) -> tuple[Dataset, Dataset]:
    cfg = get_dataset_config(dataset_name)
    if isinstance(ds, DatasetDict):
        if cfg.has_test and "test" in ds:
            return ds["train"], ds["test"]
        full = ds["train"]
        _validate_label_col(full, cfg.label_col)
        train, test = _stratified_train_test_split(
            full, cfg.test_fraction, cfg.label_col, seed,
        )
        print(
            f"[split] {dataset_name}: train={len(train):,}, test={len(test):,} "
            f"(strat. {1-cfg.test_fraction:.0%}/{cfg.test_fraction:.0%})"
        )
        return train, test
    _validate_label_col(ds, cfg.label_col)
    return _stratified_train_test_split(ds, cfg.test_fraction, cfg.label_col, seed)


def extract_server_dataset(
    dataset: Dataset, server_size: int, *, seed: int, label_col: str,
) -> tuple[Dataset, Dataset]:
    _validate_label_col(dataset, label_col)
    rng = np.random.default_rng(seed)
    labels = np.array(dataset[label_col])
    classes = sorted(set(labels.tolist()))
    per_class = server_size // len(classes)
    if per_class == 0:
        raise ValueError(
            f"server_size={server_size} too small for {len(classes)} classes"
        )
    server_idx, rest_idx = [], []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        if len(cls_idx) < per_class:
            raise ValueError(
                f"class {cls}: {len(cls_idx)} samples < required {per_class}"
            )
        server_idx.extend(cls_idx[:per_class].tolist())
        rest_idx.extend(cls_idx[per_class:].tolist())
    return dataset.select(sorted(server_idx)), dataset.select(sorted(rest_idx))


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOW-LEVEL SPLITS (per-class, работают с np.ndarray индексов)
# ─────────────────────────────────────────────────────────────────────────────

def _iid_per_class(indices: np.ndarray, num_clients: int, rng) -> list[np.ndarray]:
    rng.shuffle(indices)
    return list(np.array_split(indices, num_clients))


def _dirichlet_per_class(
    indices: np.ndarray, num_clients: int, alpha: float, rng,
) -> list[np.ndarray]:
    if len(indices) == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]
    rng.shuffle(indices)
    proportions = rng.dirichlet([alpha] * num_clients)
    counts = (proportions * len(indices)).astype(int)
    diff = len(indices) - counts.sum()
    frac = proportions * len(indices) - counts
    for i in np.argsort(-frac)[:diff]:
        counts[i] += 1
    splits, start = [], 0
    for c in counts:
        splits.append(indices[start:start + c])
        start += c
    return splits


def _floor_per_class(
    indices: np.ndarray, num_clients: int, min_per_client: int, rng,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Раздаём min_per_client каждому клиенту, возвращаем остаток."""
    rng.shuffle(indices)
    actual = min(min_per_client, len(indices) // num_clients)
    floor_total = actual * num_clients
    floor = [indices[i*actual:(i+1)*actual] for i in range(num_clients)]
    return floor, indices[floor_total:]


def _fixed_proportions_per_class(
    indices: np.ndarray, num_clients: int, proportions: np.ndarray, rng,
) -> list[np.ndarray]:
    """Режем индексы согласно ФИКСИРОВАННЫМ пропорциям (shared across classes)."""
    if len(indices) == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]
    rng.shuffle(indices)
    counts = (proportions * len(indices)).astype(int)
    diff = len(indices) - counts.sum()
    frac = proportions * len(indices) - counts
    for i in np.argsort(-frac)[:diff]:
        counts[i] += 1
    splits, start = [], 0
    for c in counts:
        splits.append(indices[start:start + c])
        start += c
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILDERS (работают с Dataset, вызываются из Method.build)
# ─────────────────────────────────────────────────────────────────────────────

def _build_iid(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    rng = np.random.default_rng(seed)
    labels = np.array(train_ds[label_col])
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        # двойной shuffle для соответствия старому partition_dataset(min_per_class=0)
        floor, rest = _floor_per_class(cls_idx.copy(), num_clients, 0, rng)
        splits = _iid_per_class(rest, num_clients, rng)
        for i in range(num_clients):
            client_idx[i].extend(splits[i].tolist())
    return [train_ds.select(sorted(idx)) for idx in client_idx]


def _build_dirichlet(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    alpha = float(params["alpha"])
    min_per_class = int(params.get("min_per_class", 0))
    rng = np.random.default_rng(seed)
    labels = np.array(train_ds[label_col])
    if min_per_class > 0:
        for cls in sorted(set(labels.tolist())):
            n_cls = int((labels == cls).sum())
            needed = min_per_class * num_clients
            if n_cls < needed:
                print(
                    f"[warn] class {cls}: {n_cls} samples, min_per_class={min_per_class}×{num_clients}={needed}"
                )
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        floor, rest = _floor_per_class(cls_idx.copy(), num_clients, min_per_class, rng)
        splits = _dirichlet_per_class(rest, num_clients, alpha, rng)
        for i in range(num_clients):
            client_idx[i].extend(floor[i].tolist())
            client_idx[i].extend(splits[i].tolist())
    return [train_ds.select(sorted(idx)) for idx in client_idx]


def _build_dirichlet_quantity(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    """Комбинированный label + quantity skew (NIID-Bench стиль).

    Для каждого класса k: w_ik ∝ q_i · p_ik, где
      p_k ~ Dir(α) — per-class label skew (как в обычном Dirichlet),
      q   ~ Dir(β) — per-client volume multiplier (одна выборка на всю партицию).
    Дегенеративные случаи: β→∞ эквивалентен обычному Dirichlet(α);
    α→∞ эквивалентен quantity(β)."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    rng = np.random.default_rng(seed)
    q = rng.dirichlet([beta] * num_clients)
    labels = np.array(train_ds[label_col])
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        p = rng.dirichlet([alpha] * num_clients)
        w = q * p
        total = w.sum()
        w = w / total if total > 0 else np.full(num_clients, 1.0 / num_clients)
        splits = _fixed_proportions_per_class(cls_idx.copy(), num_clients, w, rng)
        for i in range(num_clients):
            client_idx[i].extend(splits[i].tolist())
    sizes = [len(idx) for idx in client_idx]
    if min(sizes) == 0:
        zero = [i for i, s in enumerate(sizes) if s == 0]
        raise ValueError(
            f"dirichlet_quantity α={alpha} β={beta}: clients {zero} got 0 samples. "
            f"Try larger α or β."
        )
    return [train_ds.select(sorted(idx)) for idx in client_idx]


def _build_quantity(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    """Volume skew: proportions ~ Dir(β) сэмплируются один раз, применяются
    стратифицированно к каждому классу → P_i ≈ uniform, n_i перекошены."""
    beta = float(params["beta"])
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet([beta] * num_clients)
    labels = np.array(train_ds[label_col])
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        splits = _fixed_proportions_per_class(cls_idx.copy(), num_clients, proportions, rng)
        for i in range(num_clients):
            client_idx[i].extend(splits[i].tolist())
    sizes = [len(idx) for idx in client_idx]
    if min(sizes) == 0:
        zero_clients = [i for i, s in enumerate(sizes) if s == 0]
        raise ValueError(
            f"quantity β={beta}: clients {zero_clients} got 0 samples. "
            f"Proportions={np.round(proportions, 4).tolist()}. "
            f"Try larger β (≥0.5) or different seed."
        )
    return [train_ds.select(sorted(idx)) for idx in client_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRUCTURAL CONFIGS И BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

# ─── CIFAR-100 (meta-based, coarse_label) ──────────────────────────────

CIFAR_COARSE_COL = "coarse_label"
CIFAR_STRUCTURAL_NUM_CLIENTS = 10  # hard-coded: 10 × 2 = 20 суперклассов


def _cifar_check_clients(num_clients: int, method: str) -> None:
    if num_clients != CIFAR_STRUCTURAL_NUM_CLIENTS:
        raise ValueError(
            f"{method} requires num_clients={CIFAR_STRUCTURAL_NUM_CLIENTS}, got {num_clients}"
        )


def _build_cifar_meta2_disjoint(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    _cifar_check_clients(num_clients, "meta2-disjoint")
    coarse = np.array(train_ds[CIFAR_COARSE_COL])
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]
    for cid in range(num_clients):
        mask = np.isin(coarse, [2*cid, 2*cid + 1])
        client_idx[cid] = np.where(mask)[0].tolist()
    return [train_ds.select(sorted(idx)) for idx in client_idx]


def _cifar_meta1_shared(
    train_ds: Dataset, num_clients: int, seed: int, label_col: str,
    shared_scheme: str, shared_alpha: float,
) -> list[Dataset]:
    """1 эксклюзивный суперкласс (cid → coarse=cid) + суперклассы [N..19] делятся."""
    coarse = np.array(train_ds[CIFAR_COARSE_COL])
    fine = np.array(train_ds[label_col])
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]

    # exclusive: каждый клиент забирает ВСЁ по своему суперклассу
    for cid in range(num_clients):
        mask = coarse == cid
        client_idx[cid].extend(np.where(mask)[0].tolist())

    # shared: суперклассы [num_clients..19] делятся поклассно
    rng = np.random.default_rng(seed + 1)
    shared_metas = list(range(num_clients, 20))
    mask_shared = np.isin(coarse, shared_metas)
    shared_indices = np.where(mask_shared)[0]
    shared_labels = fine[shared_indices]
    for cls in sorted(set(shared_labels.tolist())):
        cls_idx = shared_indices[shared_labels == cls].copy()
        if shared_scheme == "iid":
            splits = _iid_per_class(cls_idx, num_clients, rng)
        else:
            splits = _dirichlet_per_class(cls_idx, num_clients, shared_alpha, rng)
        for cid, s in enumerate(splits):
            client_idx[cid].extend(s.tolist())
    return [train_ds.select(sorted(idx)) for idx in client_idx]


def _build_cifar_meta1_iid(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    _cifar_check_clients(num_clients, "meta1-iid")
    return _cifar_meta1_shared(train_ds, num_clients, seed, label_col, "iid", 0.0)


def _build_cifar_meta1_dirichlet(
    train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
    params: dict[str, Any],
) -> list[Dataset]:
    _cifar_check_clients(num_clients, "meta1-dirichlet")
    alpha = float(params["alpha"])
    return _cifar_meta1_shared(train_ds, num_clients, seed, label_col, "dirichlet", alpha)


# ─── PlantVillage (explicit class assignments) ─────────────────────────

PV_ASSIGN_MONO: dict[int, list[int]] = {
    0: [28, 29, 30],
    1: [31, 32, 33],
    2: [34, 35, 36, 37],
    3: [0, 1, 2, 3],
    4: [11, 12, 13, 14],
    5: [7, 8, 9, 10],
    6: [15],
    7: [24],
    8: [18, 19, 20, 21, 22],
    9: [4, 5, 6, 16, 17, 23, 25, 26, 27],
}
PV_ASSIGN_FAMILY: dict[int, list[int]] = {
    0: [28, 29, 30],
    1: [31, 32, 33],
    2: [34, 35, 36, 37],
    3: [18, 19, 20, 21, 22],
    4: [0, 1, 2, 3],
    5: [5, 6, 16, 17, 26, 27],
    6: [4, 11, 12, 13, 14],
    7: [7, 8, 9, 10],
    8: [15, 23, 25],
    9: [24],
}
PV_ASSIGN_MIX: dict[int, list[int]] = {
    0: [28, 29, 30, 31, 32],
    1: [33, 34, 35, 36, 37],
    2: [0, 1, 2, 3],
    3: [7, 8, 9, 10],
    4: [15],
    5: [24],
    6: [4, 11, 12, 13, 14],
    7: [5, 6, 16, 17, 26, 27],
    8: [18, 19, 20, 21, 22],
    9: [23, 25],
}
PV_GROUPS_MONO: list[list[int]] = [[i] for i in range(10)]
PV_GROUPS_FAMILY: list[list[int]] = [[0, 1, 2, 3], [4, 5], [6], [7], [8], [9]]
PV_GROUPS_MIX: list[list[int]] = [[0, 1, 8], [2, 7], [6, 9], [3], [4], [5]]

PV_STRUCTURAL: dict[str, dict] = {
    "structural-mono":   {"assign": PV_ASSIGN_MONO,   "groups": PV_GROUPS_MONO,   "overlap": 0.0},
    "structural-family": {"assign": PV_ASSIGN_FAMILY, "groups": PV_GROUPS_FAMILY, "overlap": 0.2},
    "structural-mix":    {"assign": PV_ASSIGN_MIX,    "groups": PV_GROUPS_MIX,    "overlap": 0.4},
}


def _pv_validate_variant(
    assign: dict[int, list[int]], groups: list[list[int]],
    num_clients: int, num_classes: int,
) -> None:
    all_cls = set()
    for cid in range(num_clients):
        if cid not in assign:
            raise ValueError(f"client {cid} missing in assignment")
        for c in assign[cid]:
            if c in all_cls:
                raise ValueError(f"class {c} assigned to multiple clients")
            all_cls.add(c)
    missing = set(range(num_classes)) - all_cls
    if missing:
        raise ValueError(f"classes not assigned: {sorted(missing)}")
    in_groups: set[int] = set()
    for g in groups:
        for c in g:
            if c in in_groups:
                raise ValueError(f"client {c} in multiple groups")
            in_groups.add(c)
    if in_groups != set(range(num_clients)):
        raise ValueError(f"groups don't cover all {num_clients} clients")


def _pv_structural_builder(variant: str) -> Callable[..., list[Dataset]]:
    spec = PV_STRUCTURAL[variant]
    assign = spec["assign"]
    groups = spec["groups"]
    overlap = spec["overlap"]

    def _builder(
        train_ds: Dataset, *, num_clients: int, seed: int, label_col: str,
        params: dict[str, Any],
    ) -> list[Dataset]:
        if num_clients != 10:
            raise ValueError(
                f"{variant} requires num_clients=10, got {num_clients}"
            )
        labels = np.array(train_ds[label_col])
        num_classes = int(labels.max()) + 1
        _pv_validate_variant(assign, groups, num_clients, num_classes)

        class_owner = {c: cid for cid, cls_list in assign.items() for c in cls_list}
        client_group = {c: g for g in groups for c in g}

        rng = np.random.default_rng(seed)
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            primary = class_owner[cls]
            group = client_group[primary]
            secondaries = [c for c in group if c != primary]
            idx = np.where(labels == cls)[0].copy()
            rng.shuffle(idx)
            if not secondaries or overlap == 0.0:
                client_idx[primary].extend(idx.tolist())
                continue
            n_primary = int(round(len(idx) * (1.0 - overlap)))
            client_idx[primary].extend(idx[:n_primary].tolist())
            remaining = idx[n_primary:]
            per_sec = np.array_split(remaining, len(secondaries))
            for sec, chunk in zip(secondaries, per_sec):
                client_idx[sec].extend(chunk.tolist())
        return [train_ds.select(sorted(idx)) for idx in client_idx]

    return _builder


# ─────────────────────────────────────────────────────────────────────────────
# 6. METHOD REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

BuildFn = Callable[..., list[Dataset]]


@dataclass(frozen=True)
class Method:
    """Партиционирующий метод. Добавление нового = ещё одна register(Method(...))."""
    name: str
    build: BuildFn
    datasets: tuple[str, ...] = ("cifar100", "plantvillage")
    validate: Callable[[dict], None] = field(default=lambda p: None)
    dir_suffix: Callable[[dict], str] = field(default=lambda p: "")
    manifest_extra: Callable[[dict], dict] = field(default=lambda p: {})
    legacy_fields: Callable[[dict], dict] = field(default=lambda p: {})


METHODS: dict[str, Method] = {}


def register(m: Method) -> Method:
    if m.name in METHODS:
        raise ValueError(f"Method {m.name!r} already registered")
    METHODS[m.name] = m
    return m


# ─── iid ───────────────────────────────────────────────────────────────

def _iid_validate(p: dict) -> None:
    if p:
        raise ValueError(f"iid takes no params, got {p}")


register(Method(
    name="iid",
    build=_build_iid,
    validate=_iid_validate,
    legacy_fields=lambda p: {"scheme": "iid", "alpha": None, "min_per_class": 0},
))


# ─── dirichlet ─────────────────────────────────────────────────────────

def _dirichlet_validate(p: dict) -> None:
    if "alpha" not in p:
        raise ValueError("dirichlet requires 'alpha'")
    if float(p["alpha"]) <= 0:
        raise ValueError(f"alpha must be > 0, got {p['alpha']}")


def _dirichlet_suffix(p: dict) -> str:
    m = int(p.get("min_per_class", 0))
    return f"a{p['alpha']}__m{m}"


register(Method(
    name="dirichlet",
    build=_build_dirichlet,
    validate=_dirichlet_validate,
    dir_suffix=_dirichlet_suffix,
    manifest_extra=lambda p: {
        "alpha": float(p["alpha"]),
        "min_per_class": int(p.get("min_per_class", 0)),
    },
    legacy_fields=lambda p: {
        "scheme": "dirichlet",
        "alpha": float(p["alpha"]),
        "min_per_class": int(p.get("min_per_class", 0)),
    },
))


# ─── quantity ──────────────────────────────────────────────────────────

def _quantity_validate(p: dict) -> None:
    if "beta" not in p:
        raise ValueError("quantity requires 'beta'")
    if float(p["beta"]) <= 0:
        raise ValueError(f"beta must be > 0, got {p['beta']}")


register(Method(
    name="quantity",
    build=_build_quantity,
    validate=_quantity_validate,
    dir_suffix=lambda p: f"b{p['beta']}",
    manifest_extra=lambda p: {"beta": float(p["beta"])},
    legacy_fields=lambda p: {
        "scheme": "quantity",
        "alpha": None,
        "beta": float(p["beta"]),
        "min_per_class": 0,
    },
))


# ─── dirichlet_quantity (label + quantity skew) ────────────────────────

def _dir_quant_validate(p: dict) -> None:
    for k in ("alpha", "beta"):
        if k not in p:
            raise ValueError(f"dirichlet_quantity requires {k!r}")
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0, got {p[k]}")


register(Method(
    name="dirichlet_quantity",
    build=_build_dirichlet_quantity,
    validate=_dir_quant_validate,
    dir_suffix=lambda p: f"a{p['alpha']}__b{p['beta']}",
    manifest_extra=lambda p: {
        "alpha": float(p["alpha"]),
        "beta":  float(p["beta"]),
    },
    legacy_fields=lambda p: {
        "scheme":        "dirichlet_quantity",
        "alpha":         float(p["alpha"]),
        "beta":          float(p["beta"]),
        "min_per_class": 0,
    },
))


# ─── CIFAR-100 structural ──────────────────────────────────────────────

register(Method(
    name="meta2-disjoint",
    datasets=("cifar100",),
    build=_build_cifar_meta2_disjoint,
    legacy_fields=lambda p: {"scheme": "meta2-disjoint", "alpha": None, "min_per_class": 0},
))

register(Method(
    name="meta1-iid",
    datasets=("cifar100",),
    build=_build_cifar_meta1_iid,
    legacy_fields=lambda p: {"scheme": "meta1-iid", "alpha": None, "min_per_class": 0},
))


def _meta1_dir_validate(p: dict) -> None:
    if "alpha" not in p:
        raise ValueError("meta1-dirichlet requires 'alpha'")


register(Method(
    name="meta1-dirichlet",
    datasets=("cifar100",),
    build=_build_cifar_meta1_dirichlet,
    validate=_meta1_dir_validate,
    dir_suffix=lambda p: f"a{p['alpha']}",
    manifest_extra=lambda p: {"alpha": float(p["alpha"])},
    # historical bug-compat: alpha хранится в dir_suffix, в legacy-поле было None
    legacy_fields=lambda p: {"scheme": "meta1-dirichlet", "alpha": None, "min_per_class": 0},
))


# ─── PlantVillage structural ───────────────────────────────────────────

for _v in ("structural-mono", "structural-family", "structural-mix"):
    register(Method(
        name=_v,
        datasets=("plantvillage",),
        build=_pv_structural_builder(_v),
        # исторически save_partitions писал alpha=None для любого не-dirichlet
        legacy_fields=lambda p, _s=_v: {"scheme": _s, "alpha": None, "min_per_class": 0},
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 7. MANIFEST & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def _write_partition_to_disk(
    out_dir: Path, *,
    dataset: str, method: Method, params: dict,
    num_clients: int, seed: int,
    train_ds: Dataset, test_ds: Dataset,
    partitions: list[Dataset], server_ds: Dataset | None,
    label_col: str,
) -> None:
    out_dir.mkdir(parents=True)

    if server_ds is not None:
        server_ds.save_to_disk(str(out_dir / "server"))
        print(f"  server:   {len(server_ds):,} samples")
    for i, part in enumerate(partitions):
        part.save_to_disk(str(out_dir / f"client_{i}"))
        print(f"  client_{i}: {len(part):,} samples")
    test_ds.save_to_disk(str(out_dir / "test"))
    print(f"  test:     {len(test_ds):,} samples")

    # class names
    label_feat = partitions[0].features.get(label_col) if partitions else None
    if label_feat and hasattr(label_feat, "names"):
        class_names = list(label_feat.names)
    else:
        all_cls: set[int] = set()
        for p in partitions:
            all_cls.update(p[label_col])
        all_cls.update(test_ds[label_col])
        class_names = [str(c) for c in sorted(all_cls)]
    num_classes = len(class_names)

    # per-client class counts
    client_stats = []
    for part in partitions:
        ctr = collections.Counter(part[label_col])
        classes_count = {str(i): int(ctr.get(i, 0)) for i in range(num_classes)}
        client_stats.append({"total": len(part), "classes": classes_count})

    manifest: dict[str, Any] = {
        "dataset":       dataset,
        "method":        method.name,
        "method_params": params,
        **method.legacy_fields(params),
        "seed":          seed,
        "num_clients":   len(partitions),
        "num_classes":   num_classes,
        "class_names":   class_names,
        "label_col":     label_col,
        "clients":       client_stats,
        "test_size":     len(test_ds),
        "server_size":   len(server_ds) if server_ds is not None else None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def make_partition(
    *,
    dataset: str,
    method_name: str,
    num_clients: int,
    seed: int,
    params: dict[str, Any] | None = None,
    server_size: int = 0,
    data_dir: Path = Path("data"),
    out_root: Path = Path("data/partitions"),
    force: bool = False,
) -> Path:
    params = dict(params or {})

    if method_name not in METHODS:
        raise ValueError(
            f"Unknown method {method_name!r}. Registered: {sorted(METHODS)}"
        )
    method = METHODS[method_name]

    if dataset not in method.datasets:
        raise ValueError(
            f"Method {method_name!r} does not support dataset {dataset!r}. "
            f"Supported: {method.datasets}"
        )
    method.validate(params)

    cfg = get_dataset_config(dataset)

    # Имя: {dataset}__{method}[__{suffix}]__n{N}__s{S}[__srv{K}]
    parts = [dataset, method.name]
    suffix = method.dir_suffix(params)
    if suffix:
        parts.append(suffix)
    parts.extend([f"n{num_clients}", f"s{seed}"])
    if server_size > 0:
        parts.append(f"srv{server_size}")
    dir_name = "__".join(parts)
    out_dir = out_root / dir_name

    if out_dir.exists():
        if force:
            shutil.rmtree(out_dir)
        else:
            print(f"[skip] Already exists: {out_dir}")
            return out_dir

    header = f"── {method.name}" + (f" {params}" if params else "") + " ──"
    print(f"\n{header}")
    ds = load_from_disk(str(data_dir / dataset))
    train_ds, test_ds = prepare_splits(ds, dataset, seed=seed)

    server_ds = None
    if server_size > 0:
        server_ds, train_ds = extract_server_dataset(
            train_ds, server_size, seed=seed, label_col=cfg.label_col,
        )

    partitions = method.build(
        train_ds,
        num_clients=num_clients,
        seed=seed,
        label_col=cfg.label_col,
        params=params,
    )

    sizes = [len(p) for p in partitions]
    print(
        f"[partition] {method.name}: samples/client "
        f"min={min(sizes):,} max={max(sizes):,} avg={sum(sizes)//len(sizes):,}"
    )

    _write_partition_to_disk(
        out_dir,
        dataset=dataset, method=method, params=params,
        num_clients=num_clients, seed=seed,
        train_ds=train_ds, test_ds=test_ds,
        partitions=partitions, server_ds=server_ds,
        label_col=cfg.label_col,
    )
    print(f"[saved] {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 9. RECIPES
# ─────────────────────────────────────────────────────────────────────────────

RECIPES: dict[str, dict] = {
    "cifar100": {
        "num_clients": 10,
        "seed": 42,
        "items": [
            {"method": "iid"},
            *[{"method": "dirichlet", "params": {"alpha": a, "min_per_class": 0}}
              for a in [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05]],
            *[{"method": "quantity", "params": {"beta": b}}
              for b in [10.0, 1.0, 0.5]],
            *[{"method": "dirichlet_quantity", "params": {"alpha": a, "beta": b}}
              for a, b in [(0.5, 1.0), (0.5, 0.5), (0.2, 1.0), (0.2, 0.5)]],
            {"method": "meta2-disjoint"},
            {"method": "meta1-iid"},
            {"method": "meta1-dirichlet", "params": {"alpha": 0.3}},
        ],
    },
    "plantvillage": {
        "num_clients": 10,
        "seed": 42,
        "items": [
            {"method": "iid"},
            *[{"method": "dirichlet", "params": {"alpha": a, "min_per_class": 0}}
              for a in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]],
            *[{"method": "quantity", "params": {"beta": b}}
              for b in [10.0, 1.0, 0.5]],
            {"method": "structural-mono"},
            {"method": "structural-family"},
            {"method": "structural-mix"},
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _list_methods() -> None:
    print("Зарегистрированные методы:\n")
    for name, m in METHODS.items():
        print(f"  {name:<20} datasets={m.datasets}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Создание партиций для FL-экспериментов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset", nargs="?", choices=sorted(RECIPES),
                    help="датасет из RECIPES")
    ap.add_argument("--only", action="append", default=None, metavar="METHOD",
                    help="оставить только указанные методы (можно повторять)")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="переопределить список α для dirichlet/meta1-dirichlet")
    ap.add_argument("--betas", nargs="+", type=float, default=None,
                    help="переопределить список β для quantity")
    ap.add_argument("--force", action="store_true",
                    help="перезаписывать существующие директории")
    ap.add_argument("--list", action="store_true",
                    help="показать зарегистрированные методы и выйти")
    return ap.parse_args()


def _filter_items(items: list[dict], args: argparse.Namespace) -> list[dict]:
    # --only: фильтр по методам
    if args.only:
        only = set(args.only)
        items = [it for it in items if it["method"] in only]

    # --alphas: переопределить список α (влияет на dirichlet и meta1-dirichlet)
    if args.alphas is not None:
        alpha_methods = {"dirichlet", "meta1-dirichlet"}
        non_alpha = [it for it in items if it["method"] not in alpha_methods]
        alpha_items: list[dict] = []
        for m in alpha_methods:
            template = next((it for it in items if it["method"] == m), None)
            if template is None:
                continue
            base = {k: v for k, v in template.get("params", {}).items() if k != "alpha"}
            for a in args.alphas:
                alpha_items.append({"method": m, "params": {**base, "alpha": a}})
        items = non_alpha + alpha_items

    # --betas: переопределить список β (quantity)
    if args.betas is not None:
        non_beta = [it for it in items if it["method"] != "quantity"]
        items = non_beta + [
            {"method": "quantity", "params": {"beta": b}} for b in args.betas
        ]

    return items


def main() -> None:
    args = _parse_args()
    if args.list:
        _list_methods()
        return
    if not args.dataset:
        print("Укажи датасет: cifar100 | plantvillage (или --list)")
        sys.exit(1)

    recipe = RECIPES[args.dataset]
    items = _filter_items(recipe["items"], args)

    if not items:
        print(f"После фильтров не осталось методов для {args.dataset}")
        return

    print(f"[recipe] {args.dataset}: {len(items)} методов")
    for it in items:
        make_partition(
            dataset=args.dataset,
            method_name=it["method"],
            num_clients=recipe["num_clients"],
            seed=recipe["seed"],
            params=it.get("params"),
            force=args.force,
        )


if __name__ == "__main__":
    main()

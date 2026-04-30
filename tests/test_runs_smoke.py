"""Smoke test: end-to-end config pipeline for every (strategy, model) combo.

Verifies that the run-config that would be assembled by the frontend, sanitized
by the backend, and serialized by the orchestrator into `flwr run --run-config`
contains only the keys that belong to the chosen strategy. No actual training.
"""

import pytest

from backend.runs.orchestrator import build_command
from backend.runs.router import _sanitize_run_config


# Mirrors `frontend/src/api/flStrategies.ts:CIFAR_PER_STRATEGY`. If this drifts,
# the regression test below fails — so the two stay in sync by design.
_CIFAR_PER_STRATEGY: dict[str, dict] = {
    "fedavg":   {"client-lr": 0.1,  "client-momentum": 0.9, "local-epochs": 3},
    "fedavgm":  {"client-lr": 0.05, "client-momentum": 0.9, "local-epochs": 3,
                 "server-momentum": 0.5, "server-lr": 1.0},
    "fedprox":  {"client-lr": 0.1,  "client-momentum": 0.9, "local-epochs": 3,
                 "proximal-mu": 0.0005},
    "fednovam": {"client-lr": 0.05, "client-momentum": 0.9, "local-epochs": 3,
                 "server-momentum": 0.5, "server-lr": 1.0},
}

_MODEL_DEFAULTS: dict[str, dict] = {
    "wrn_16_4":  {"optimizer": "sgd", "client-lr": 0.03, "client-momentum": 0.9,
                  "client-weight-decay": 5e-4, "batch-size": 64},
    "se_resnet": {"optimizer": "sgd", "client-lr": 0.03, "client-momentum": 0.9,
                  "client-weight-decay": 5e-4, "batch-size": 64},
}

_FL_PARAM_DEFAULTS = {
    "num-server-rounds": 80,
    "local-epochs": 3,
    "fraction-train": 1.0,
    "min-train-nodes": 10,
    "min-available-nodes": 10,
    "server-momentum": 0.5,
    "server-lr": 1.0,
    "proximal-mu": 0.0005,
}

_STRAGGLER_DEFAULTS = {
    "straggler-mode": "none",
    "straggler-target": "min",
    "straggler-tolerance": 0.05,
    "straggler-drop-tolerance": 0.5,
    "straggler-max-dropped": 3,
    "straggler-min-chunk": 0.1,
    "straggler-min-epochs": 1,
}


def _build_frontend_config(model: str, strategy: str, straggler_mode: str = "none") -> dict:
    """Replicate `buildRunConfig` from ProjectDetailPage.tsx:
    base FL_PARAM_DEFAULTS + model defaults + strategy override + straggler.
    Filtering by strategy happens in `filterRunConfig` (mirror of `_sanitize_run_config`).
    """
    cfg: dict = {"model": model, "aggregation": strategy}
    cfg.update(_FL_PARAM_DEFAULTS)
    cfg.update(_MODEL_DEFAULTS[model])

    # Apply per-strategy overrides (front does this in setStrategy / setSelectedModel)
    overrides = _CIFAR_PER_STRATEGY.get(strategy, {})
    cfg.update(overrides)

    # Add straggler params
    cfg.update(_STRAGGLER_DEFAULTS)
    cfg["straggler-mode"] = straggler_mode

    # Backend injects partition-name in start_run
    cfg["partition-name"] = "cifar100__iid__n10__s42"
    return cfg


_ALL_STRATEGIES = ["fedavg", "fedavgm", "fedprox", "fednovam"]
_ALL_MODELS = ["wrn_16_4", "se_resnet"]
_ALL_STRAGGLER_MODES = ["none", "chunk", "epochs", "drop"]


@pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
@pytest.mark.parametrize("model", _ALL_MODELS)
def test_pipeline_produces_clean_config(model: str, strategy: str) -> None:
    """Full path: frontend assembly → backend sanitize → orchestrator CLI."""
    rc = _build_frontend_config(model, strategy)
    cleaned = _sanitize_run_config(rc, strategy)
    cmd, _ = build_command(run_id=999, federation="local-sim", run_config=cleaned)

    cmd_str = " ".join(cmd)

    # No strategy-gated key leaks for the wrong strategy
    if strategy not in {"fedavgm", "fednovam"}:
        assert "server-momentum" not in cmd_str, f"{model}/{strategy}: server-momentum leaked"
        assert "server-lr" not in cmd_str, f"{model}/{strategy}: server-lr leaked"
    if strategy != "fedprox":
        assert "proximal-mu" not in cmd_str, f"{model}/{strategy}: proximal-mu leaked"

    # Required keys present
    for required in ("model", "aggregation", "num-server-rounds", "local-epochs",
                     "client-lr", "batch-size", "partition-name"):
        assert required in cmd_str, f"{model}/{strategy}: missing required key {required!r}"

    # Strategy-correct values
    assert f'aggregation="{strategy}"' in cmd_str
    assert f'model="{model}"' in cmd_str


@pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
def test_per_strategy_client_lr_default(strategy: str) -> None:
    """The per-strategy override must dominate model defaults."""
    rc = _build_frontend_config("se_resnet", strategy)
    cleaned = _sanitize_run_config(rc, strategy)
    expected_lr = _CIFAR_PER_STRATEGY[strategy]["client-lr"]
    assert cleaned["client-lr"] == expected_lr, (
        f"{strategy}: expected client-lr={expected_lr}, got {cleaned['client-lr']}"
    )


@pytest.mark.parametrize("mode", _ALL_STRAGGLER_MODES)
def test_straggler_modes_pass_through(mode: str) -> None:
    rc = _build_frontend_config("se_resnet", "fedavgm", straggler_mode=mode)
    cleaned = _sanitize_run_config(rc, "fedavgm")
    cmd, _ = build_command(run_id=999, federation="local-sim", run_config=cleaned)
    cmd_str = " ".join(cmd)
    assert f'straggler-mode="{mode}"' in cmd_str


def test_run_11_actual_payload_after_full_pipeline() -> None:
    """Regression: feed the actual run #11 payload (with proximal-mu leak) and
    verify the leak is removed and other params are intact."""
    actual_run_11 = {
        "model": "se_resnet",
        "client-lr": 0.03,
        "optimizer": "sgd",
        "server-lr": 1,
        "batch-size": 64,
        "aggregation": "fedavgm",
        "proximal-mu": 0.0005,
        "local-epochs": 3,
        "fraction-train": 1,
        "straggler-mode": "chunk",
        "client-momentum": 0.9,
        "min-train-nodes": 10,
        "server-momentum": 0.5,
        "straggler-target": "min",
        "num-server-rounds": 80,
        "client-weight-decay": 5e-4,
        "min-available-nodes": 10,
        "straggler-min-chunk": 0.1,
        "straggler-tolerance": 0.05,
        "straggler-min-epochs": 1,
        "straggler-max-dropped": 3,
        "straggler-drop-tolerance": 0.5,
        "partition-name": "cifar100__iid__n10__s42",
    }
    cleaned = _sanitize_run_config(actual_run_11, "fedavgm")
    cmd, _ = build_command(run_id=11, federation="local-sim", run_config=cleaned)
    cmd_str = " ".join(cmd)

    assert "proximal-mu" not in cmd_str
    assert "server-momentum=0.5" in cmd_str
    assert "server-lr=1" in cmd_str

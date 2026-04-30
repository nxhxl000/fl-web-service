"""Unit tests for `_sanitize_run_config` — strategy-gated key filtering."""

import pytest

from backend.runs.router import _sanitize_run_config


def _full_rc(extra: dict | None = None) -> dict:
    """A run-config with every strategy-gated key present, plus common keys."""
    base = {
        "model": "se_resnet",
        "aggregation": "fedavgm",
        "num-server-rounds": 80,
        "local-epochs": 3,
        "fraction-train": 1.0,
        "min-train-nodes": 10,
        "min-available-nodes": 10,
        "client-lr": 0.05,
        "client-momentum": 0.9,
        "client-weight-decay": 5e-4,
        "batch-size": 64,
        "optimizer": "sgd",
        "server-momentum": 0.5,
        "server-lr": 1.0,
        "proximal-mu": 0.0005,
        "straggler-mode": "chunk",
        "straggler-target": "min",
        "straggler-tolerance": 0.05,
        "straggler-drop-tolerance": 0.5,
        "straggler-max-dropped": 3,
        "straggler-min-chunk": 0.1,
        "straggler-min-epochs": 1,
        "partition-name": "cifar100__iid__n10__s42",
    }
    if extra:
        base.update(extra)
    return base


@pytest.mark.parametrize(
    "aggregation, kept_strategy_keys, dropped_strategy_keys",
    [
        ("fedavg",   set(),                         {"server-momentum", "server-lr", "proximal-mu"}),
        ("fedavgm",  {"server-momentum", "server-lr"}, {"proximal-mu"}),
        ("fedprox",  {"proximal-mu"},               {"server-momentum", "server-lr"}),
        ("fednovam", {"server-momentum", "server-lr"}, {"proximal-mu"}),
    ],
)
def test_sanitize_keeps_only_strategy_gated_keys_for_chosen_aggregation(
    aggregation: str, kept_strategy_keys: set[str], dropped_strategy_keys: set[str]
) -> None:
    rc = _full_rc()
    result = _sanitize_run_config(rc, aggregation)

    for k in kept_strategy_keys:
        assert k in result, f"{aggregation}: expected {k!r} kept"
    for k in dropped_strategy_keys:
        assert k not in result, f"{aggregation}: expected {k!r} dropped, got {result.get(k)!r}"


def test_sanitize_preserves_non_gated_keys() -> None:
    rc = _full_rc()
    result = _sanitize_run_config(rc, "fedavg")
    # Non-gated keys must survive regardless of strategy
    expected = {
        "model", "aggregation", "num-server-rounds", "local-epochs",
        "fraction-train", "min-train-nodes", "min-available-nodes",
        "client-lr", "client-momentum", "client-weight-decay",
        "batch-size", "optimizer",
        "straggler-mode", "straggler-target", "straggler-tolerance",
        "straggler-drop-tolerance", "straggler-max-dropped",
        "straggler-min-chunk", "straggler-min-epochs",
        "partition-name",
    }
    for k in expected:
        assert k in result, f"non-gated key {k!r} dropped"


def test_sanitize_unknown_strategy_drops_all_gated_keys() -> None:
    rc = _full_rc()
    result = _sanitize_run_config(rc, "scaffold")  # not in our 4-strategy set
    for k in ("server-momentum", "server-lr", "proximal-mu"):
        assert k not in result


def test_sanitize_empty_aggregation_is_safe() -> None:
    rc = _full_rc()
    result = _sanitize_run_config(rc, "")
    for k in ("server-momentum", "server-lr", "proximal-mu"):
        assert k not in result


def test_sanitize_does_not_mutate_input() -> None:
    rc = _full_rc()
    snapshot = dict(rc)
    _sanitize_run_config(rc, "fedavg")
    assert rc == snapshot


def test_sanitize_is_case_insensitive_for_aggregation() -> None:
    rc = _full_rc()
    result = _sanitize_run_config(rc, "FedAvgM")
    assert "server-momentum" in result
    assert "proximal-mu" not in result


def test_sanitize_run_11_real_payload_drops_proximal_mu() -> None:
    """Regression: run #11 was started with `proximal-mu=0.0005` despite
    aggregation=fedavgm. The leak came from `flParams` spread without
    strategy filtering on the frontend; the backend sanitize must catch it."""
    actual_run_11 = {
        "model": "se_resnet",
        "client-lr": 0.03,
        "optimizer": "sgd",
        "server-lr": 1,
        "batch-size": 64,
        "aggregation": "fedavgm",
        "proximal-mu": 0.0005,    # ← the leak
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
    assert "proximal-mu" not in cleaned
    assert cleaned["server-momentum"] == 0.5
    assert cleaned["server-lr"] == 1

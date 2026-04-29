"""Flower ServerApp — central evaluate + JSONL event stream + best-model checkpoint.

Single source of truth for run output: events.jsonl (append-only).
Web orchestrator tails it for live updates; offline analysis reads the same file.

Output dir resolution:
  - run_config["output-dir"]   → use directly (orchestrator-driven)
  - else                       → {experiments-dir}/{dataset}/{model}/{agg}/{tail}__r{N}__{ts}/

Event types in events.jsonl:
  - run_started        : ts, config
  - data_heterogeneity : MPJS, Gini, num_classes (after round 1)
  - schedule           : straggler schedule (after round 1)
  - round              : round-level metrics + clients[]
  - run_done           : best_acc/round, rounds_completed, model_best_path, per_class_accuracy
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fl_app.data import build_loader, load_contract
from fl_app.models import build_model, get_hparams
from fl_app.profiling import _gini_sizes, _mean_pairwise_js
from fl_app.scheduler import Schedule, compute_schedule
from fl_app.strategies import build_strategy, with_cosine_lr_decay
from fl_app.training import evaluate, get_device

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    model_name = rc["model"]
    agg_name = str(rc["aggregation"]).lower()
    num_rounds = int(rc.get("num-server-rounds", 10))
    model_hp = get_hparams(model_name, agg_name)
    local_epochs = int(rc.get("local-epochs", model_hp["local-epochs"]))
    partition = rc["partition-name"]
    data_dir = rc.get("data-dir", "data/")
    exp_root = rc.get("experiments-dir", "simulation")
    dataset_name = partition.partition("__")[0]

    # Output directory
    output_dir_override = rc.get("output-dir")
    if output_dir_override:
        exp_dir = Path(output_dir_override)
    else:
        _, _, partition_tail = partition.partition("__")
        partition_tail = partition_tail.replace("__", "_") or "default"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = (
            Path(exp_root) / dataset_name / model_name / agg_name
            / f"{partition_tail}__r{num_rounds}__{timestamp}"
        )
    exp_dir.mkdir(parents=True, exist_ok=True)
    events_path = exp_dir / "events.jsonl"

    def _emit(event: dict) -> None:
        line = json.dumps(event, default=str)
        with events_path.open("a") as f:
            f.write(line + "\n")
        # Mirror a one-line summary to stdout so live tail of stdout.log makes
        # progress visible. Combined with PYTHONUNBUFFERED=1 in the orchestrator
        # this gives realtime debug of where each round is.
        t = event.get("type")
        if t == "run_started":
            print(f"[event] run_started", flush=True)
        elif t == "round":
            r = event.get("round")
            ta = event.get("test_acc", 0)
            tl = event.get("test_loss", 0)
            tcm = event.get("t_compute_mean", 0)
            print(
                f"[event] round={r} test_acc={ta:.4f} test_loss={tl:.4f} "
                f"t_compute_mean={tcm:.1f}s",
                flush=True,
            )
        elif t == "run_done":
            ba = event.get("best_acc")
            br = event.get("best_round")
            rc_done = event.get("rounds_completed")
            print(f"[event] run_done best_acc={ba} @r{br} rounds={rc_done}", flush=True)
        elif t == "schedule":
            print(
                f"[event] schedule mode={event.get('mode')} "
                f"T_target={event.get('T_target', 0):.1f}s T_upper={event.get('T_upper', 0):.1f}s",
                flush=True,
            )
        elif t == "data_heterogeneity":
            print(
                f"[event] data_het MPJS={event.get('mpjs', 0):.3f} "
                f"Gini={event.get('gini_quantity', 0):.3f}",
                flush=True,
            )

    _emit({"type": "run_started", "ts": time.time(), "config": dict(rc)})

    # Project contract written by the orchestrator before launch.
    contract = load_contract(exp_dir)
    test_dataset_path = contract.get("test_dataset_path") or str(
        Path(data_dir) / "partitions" / partition / "test"
    )

    # Model + initial weights
    model = build_model(model_name)
    initial_arrays = ArrayRecord(model.state_dict())
    comm_mb = sum(a.numpy().nbytes for a in initial_arrays.values()) / (1024 ** 2)

    # Centralized test loader
    test_loader = build_loader(
        test_dataset_path,
        batch_size=256, train=False, contract=contract,
    )
    device = get_device()

    # Captured state across rounds
    per_client_rows: list[dict] = []
    round_counter = [0]
    system_het_per_round: dict[int, dict] = {}
    current_schedule: list[Schedule | None] = [None]

    def with_per_client_timing_capture(strategy):
        original = strategy.aggregate_train

        def wrapped(server_round, replies):
            t_aggr_start = time.time()
            replies = list(replies)
            round_counter[0] = server_round
            class_counts_by_pid: dict[int, dict[int, int]] = {}
            n_examples_by_pid: dict[int, float] = {}
            t_compute_by_pid: dict[int, float] = {}
            for reply in replies:
                if not reply.has_content():
                    continue
                m = reply.content["metrics"]
                t_compute = float(m.get("t-compute", 0))
                t_serialize = float(m.get("t-serialize", 0))
                created_at = float(reply.metadata.created_at)
                t_lifecycle = t_aggr_start - created_at
                drift = float(m.get("w-drift", 0))
                pid = int(m.get("partition-id", -1))
                num_ex = float(m.get("num-examples", 0))
                t_compute_by_pid[pid] = t_compute
                n_examples_by_pid[pid] = num_ex
                chunk_frac = float(m.get("chunk-fraction", 1.0))
                local_eps = float(m.get("local-epochs", local_epochs))
                w_client = num_ex * chunk_frac * local_eps
                # Round 1: extract data_cls_{N} class distribution; strip data_* keys
                # so weighted aggregate doesn't choke on non-numeric semantics.
                if server_round == 1:
                    counts: dict[int, int] = {}
                    keys_to_drop = [k for k in m.keys() if k.startswith("data_")]
                    for k in keys_to_drop:
                        if k.startswith("data_cls_"):
                            cls_id = int(k.removeprefix("data_cls_"))
                            counts[cls_id] = int(m[k])
                        del m[k]
                    if counts:
                        class_counts_by_pid[pid] = counts
                # node-name comes via a separate ConfigRecord (MetricRecord is
                # float-only). Falls back to f"pid {pid}" for sim or legacy clients.
                node_name = ""
                node_info = reply.content.get("node-info")
                if node_info is not None:
                    node_name = str(node_info.get("node-name", "") or "")
                if not node_name:
                    node_name = f"pid {pid}"
                per_client_rows.append({
                    "round":            server_round,
                    "partition_id":     pid,
                    "node_name":        node_name,
                    "num_examples":     num_ex,
                    "chunk_fraction":   chunk_frac,
                    "local_epochs":     local_eps,
                    "w_client":         w_client,
                    "train_loss_first": float(m.get("train-loss-first", 0)),
                    "train_loss_last":  float(m.get("train-loss-last", 0)),
                    "t_compute":        t_compute,
                    "t_serialize":      t_serialize,
                    "t_local":          t_compute + t_serialize,
                    "t_lifecycle":      t_lifecycle,
                    "w_drift":          drift,
                    "update_norm_rel":  float(m.get("update-norm-rel", 0)),
                    "grad_norm_last":   float(m.get("grad-norm-last", 0)),
                })

            # System heterogeneity (every round): SR, IF, I_s — formulas 26-31.
            # Active clients only (drop-mode excluded carry t_compute=0).
            ws = [r["w_client"] for r in per_client_rows if r["round"] == server_round]
            n_dropped = sum(1 for w in ws if w == 0)
            w_total = sum(ws)
            w_mean = w_total / len(ws) if ws else 0.0
            w_std = (sum((w - w_mean) ** 2 for w in ws) / len(ws)) ** 0.5 if ws else 0.0
            w_imbal = w_std / w_mean if w_mean > 0 else 0.0

            sr = idle_frac = i_s_val = T_min = T_max = 0.0
            active_pids = [p for p, t in t_compute_by_pid.items() if t > 0]
            if active_pids:
                times = [t_compute_by_pid[p] for p in active_pids]
                T_max, T_min = max(times), min(times)
                sr = T_max / T_min if T_min > 0 else 0.0
                idle_frac = sum((T_max - t) / T_max for t in times) / len(times) if T_max > 0 else 0.0
                ss = [n_examples_by_pid[p] * local_epochs / t_compute_by_pid[p] for p in active_pids]
                N = len(ss)
                sum_abs = sum(abs(ss[i] - ss[j]) for i in range(N) for j in range(N))
                i_s_val = sum_abs / (2 * N * sum(ss)) if sum(ss) > 0 else 0.0
                print(f"  [r{server_round}] SYS HET: SR={sr:.3f}  IF={idle_frac:.3f}  I_s={i_s_val:.4f}  T_min={T_min:.1f}s  T_max={T_max:.1f}s  (active={len(active_pids)}/{len(t_compute_by_pid)})", flush=True)
                print(f"  [r{server_round}] WORK:    W_total={w_total:.0f}  W_imbal={w_imbal:.3f}  n_dropped={n_dropped}", flush=True)

            system_het_per_round[server_round] = {
                "SR": sr, "IF": idle_frac, "I_s": i_s_val,
                "T_min": T_min, "T_max": T_max,
                "W_total": w_total, "W_imbalance": w_imbal,
                "n_dropped": n_dropped,
            }

            # Data heterogeneity (round 1 only)
            if server_round == 1 and class_counts_by_pid:
                num_classes_d = len(contract.get("class_names") or [])
                dists = [class_counts_by_pid[p] for p in sorted(class_counts_by_pid.keys())]
                mpjs = _mean_pairwise_js(dists, num_classes_d)
                gini_q = _gini_sizes(dists)
                print(f"  [r1] DATA HET: MPJS={mpjs:.4f}  Gini_quantity={gini_q:.4f}  num_classes={num_classes_d}", flush=True)
                _emit({
                    "type": "data_heterogeneity",
                    "mpjs": mpjs,
                    "gini_quantity": gini_q,
                    "num_classes": num_classes_d,
                })

            # Schedule (round 1 only)
            if server_round == 1 and t_compute_by_pid:
                straggler_mode     = str(rc.get("straggler-mode", "none")).lower()
                straggler_target   = str(rc.get("straggler-target", "min")).lower()
                straggler_tol      = float(rc.get("straggler-tolerance", 0.05))
                straggler_drop_tol = float(rc.get("straggler-drop-tolerance", 0.5))
                straggler_max_drop = int(rc.get("straggler-max-dropped", 3))
                straggler_min_chk  = float(rc.get("straggler-min-chunk", 0.1))
                straggler_min_ep   = int(rc.get("straggler-min-epochs", 1))
                sched = compute_schedule(
                    t_compute_by_pid,
                    mode=straggler_mode,
                    base_epochs=local_epochs,
                    target=straggler_target,
                    tolerance=straggler_tol,
                    drop_tolerance=straggler_drop_tol,
                    max_dropped=straggler_max_drop,
                    min_chunk=straggler_min_chk,
                    min_epochs=straggler_min_ep,
                )
                current_schedule[0] = sched
                _emit({"type": "schedule", **sched.to_dict()})
                if sched.mode == "drop":
                    print(f"  [r1] SCHEDULE (drop, target={sched.target}→T_target={sched.T_target:.1f}s, T_drop={sched.T_drop:.1f}s):", flush=True)
                    print(f"    → {len(sched.excluded)}/{len(t_compute_by_pid)} dropped: {sched.excluded}", flush=True)
                elif sched.mode in ("chunk", "epochs"):
                    print(f"  [r1] SCHEDULE ({sched.mode}, target={sched.target}→T_target={sched.T_target:.1f}s, T_upper={sched.T_upper:.1f}s):", flush=True)
                else:
                    print(f"  [r1] SCHEDULE: mode=none", flush=True)

            return original(server_round, replies)

        strategy.aggregate_train = wrapped
        return strategy

    def with_dynamic_schedule(strategy):
        original = strategy.configure_train

        def wrapped(server_round, arrays, config, grid):
            sched = current_schedule[0]
            if sched is not None and sched.mode != "none":
                if sched.mode == "drop":
                    config["excluded-clients"] = sched.excluded_str()
                else:
                    config["per-client-chunks"] = sched.chunks_str()
                    config["per-client-epochs"] = sched.epochs_str()
            return original(server_round, arrays, config, grid)

        strategy.configure_train = wrapped
        return strategy

    train_metrics_per_round: dict[int, dict] = {}

    def train_aggr(reply_contents, weighted_by_key):
        agg = aggregate_metricrecords(reply_contents, weighted_by_key)
        if round_counter[0] > 0:
            train_metrics_per_round[round_counter[0]] = {
                k: float(v) for k, v in dict(agg).items() if isinstance(v, (int, float))
            }
        return agg

    strategy = build_strategy(agg_name, cfg=rc)
    strategy = with_cosine_lr_decay(strategy, num_rounds)
    strategy = with_per_client_timing_capture(strategy)
    strategy = with_dynamic_schedule(strategy)
    strategy.train_metrics_aggr_fn = train_aggr

    best = {"acc": -1.0, "f1": 0.0, "loss": 0.0, "round": 0, "state_dict": None}

    def eval_fn(server_round: int, arrays: ArrayRecord):
        m = build_model(model_name)
        m.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        r = evaluate(m, test_loader, device)
        if r["acc"] > best["acc"]:
            best["acc"] = r["acc"]
            best["f1"] = r["f1_macro"]
            best["loss"] = r["loss"]
            best["round"] = server_round
            # Snapshot weights as plain detached torch tensors. Holding the
            # ArrayRecord here would pin the underlying flwr gRPC stream
            # buffer for the rest of the run — that's the deployment-mode
            # leak that didn't show in sim (Ray passes by reference).
            best["state_dict"] = {k: v.detach().clone() for k, v in m.state_dict().items()}

        # Initial eval (round 0) and post-final eval — don't emit a "round" event
        if server_round == 0 or server_round > num_rounds:
            return MetricRecord({
                "test-loss": r["loss"],
                "test-acc":  r["acc"],
                "test-f1":   r["f1_macro"],
            })

        tm = train_metrics_per_round.get(server_round, {})
        sh = system_het_per_round.get(server_round, {})
        diag_dict = {d["round"]: d for d in getattr(strategy, "diagnostics", [])}
        dg = diag_dict.get(server_round, {})
        clients_this_round = [
            {k: v for k, v in row.items() if k != "round"}
            for row in per_client_rows
            if row["round"] == server_round
        ]
        _emit({
            "type": "round",
            "round": server_round,
            "ts": time.time(),
            "test_loss": float(r["loss"]),
            "test_acc":  float(r["acc"]),
            "test_f1":   float(r["f1_macro"]),
            "train_loss_first_mean": float(tm.get("train-loss-first", 0)),
            "train_loss_last_mean":  float(tm.get("train-loss-last", 0)),
            "t_compute_mean":        float(tm.get("t-compute", 0)),
            "drift_mean":            float(tm.get("w-drift", 0)),
            "update_norm_rel_mean":  float(tm.get("update-norm-rel", 0)),
            "grad_norm_last_mean":   float(tm.get("grad-norm-last", 0)),
            "comm_mb": comm_mb,
            "system_het": {
                "SR":          float(sh.get("SR", 0)),
                "IF":          float(sh.get("IF", 0)),
                "I_s":         float(sh.get("I_s", 0)),
                "T_min":       float(sh.get("T_min", 0)),
                "T_max":       float(sh.get("T_max", 0)),
                "W_total":     float(sh.get("W_total", 0)),
                "W_imbalance": float(sh.get("W_imbalance", 0)),
                "n_dropped":   int(sh.get("n_dropped", 0)),
            },
            "strategy": {
                "delta_norm":    float(dg.get("delta_norm", 0)),
                "momentum_norm": float(dg.get("momentum_norm", 0)),
                "c_server_norm": float(dg.get("c_server_norm", 0)),
            },
            "clients": clients_this_round,
        })

        return MetricRecord({
            "test-loss": r["loss"],
            "test-acc":  r["acc"],
            "test-f1":   r["f1_macro"],
        })

    # FL loop
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": local_epochs}),
        evaluate_fn=eval_fn,
    )

    # Save best-model checkpoint + per-class accuracy
    model_best_path = exp_dir / "model_best.pt"
    per_class: list[dict] = []
    if best["state_dict"] is not None:
        bm = build_model(model_name)
        bm.load_state_dict(best["state_dict"], strict=True)
        torch.save(bm.state_dict(), model_best_path)
        br = evaluate(bm, test_loader, device)
        contract_names = contract.get("class_names") or []
        for cid, acc in enumerate(br["per_class"]):
            name = contract_names[cid] if cid < len(contract_names) else str(cid)
            per_class.append({"class_id": cid, "name": name, "accuracy": float(acc)})
    else:
        # Fallback: no eval ever succeeded — save final aggregated weights
        final = build_model(model_name)
        final.load_state_dict(result.arrays.to_torch_state_dict(), strict=True)
        torch.save(final.state_dict(), model_best_path)

    _emit({
        "type": "run_done",
        "ts": time.time(),
        "best_acc":          float(best["acc"]) if best["acc"] >= 0 else None,
        "best_f1":           float(best["f1"])  if best["acc"] >= 0 else None,
        "best_loss":         float(best["loss"]) if best["acc"] >= 0 else None,
        "best_round":        best["round"] if best["acc"] >= 0 else None,
        "rounds_completed":  round_counter[0],
        "num_rounds":        num_rounds,
        "dataset":           dataset_name,
        "model":             model_name,
        "aggregation":       agg_name,
        "model_best_path":   str(model_best_path.resolve()),
        "per_class_accuracy": per_class,
    })

    print(f"[server] Done. Artifacts: {exp_dir}")

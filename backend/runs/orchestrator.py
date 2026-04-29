"""Subprocess manager for FL training runs.

Spawns `flwr run fl_app/ <federation> --federation-config ... --run-config ... --stream`
with `output-dir` forced to `runs_data/run_<id>/`. The training process writes
`events.jsonl` + `model_best.pt` there; `finalize_run` reads them and auto-registers
a TrainedModel for the project.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any

from backend.db import SessionLocal
from backend.runs.service import finalize_run

log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
FL_APP_DIR = REPO_ROOT / "fl_app"
RUNS_DATA_DIR = REPO_ROOT / "runs_data"


# Local-sim federation defaults — `flwr federation simulation-config` doesn't
# survive ephemeral local-sim SuperLink restarts, so we always pass them explicitly.
LOCAL_SIM_FEDERATION_CONFIG: dict[str, Any] = {
    "num-supernodes": 10,
    "client-resources-num-cpus": 2,
    "client-resources-num-gpus": 0.2,
}


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f'"{v}"'


def _to_args_str(d: dict[str, Any]) -> str:
    return " ".join(f"{k}={_format_value(v)}" for k, v in d.items())


def _flwr_bin() -> str:
    return str(Path(sys.executable).parent / "flwr")


def build_command(
    run_id: int, federation: str, run_config: dict[str, Any]
) -> tuple[list[str], Path]:
    output_dir = RUNS_DATA_DIR / f"run_{run_id}"
    rc = dict(run_config)
    rc["output-dir"] = str(output_dir)

    cmd = [_flwr_bin(), "run", str(FL_APP_DIR), federation]
    if federation == "local-sim":
        cmd += ["--federation-config", _to_args_str(LOCAL_SIM_FEDERATION_CONFIG)]
    cmd += ["--run-config", _to_args_str(rc), "--stream"]
    return cmd, output_dir


_FLWR_RUN_ID_RE = re.compile(r"Successfully started run (\d+)")


def _parse_flwr_run_id(stdout_log: Path) -> int | None:
    """Find SuperLink-side run id from `flwr run --stream` output.

    The CLI prints "🎊 Successfully started run <id>" on its first stdout line
    once the SuperLink has accepted the run. We need this id to call `flwr stop`
    and tell SuperLink to abort, which also clears it from state.db.
    """
    if not stdout_log.exists():
        return None
    try:
        with stdout_log.open(errors="replace") as f:
            for line in f:
                m = _FLWR_RUN_ID_RE.search(line)
                if m:
                    return int(m.group(1))
    except OSError:
        return None
    return None


class RunOrchestrator:
    def __init__(self, runs_data_dir: Path) -> None:
        self._runs_data_dir = runs_data_dir
        self._runs_data_dir.mkdir(parents=True, exist_ok=True)
        # Map our internal run id → (process, federation, stdout_log path)
        self._processes: dict[int, tuple[subprocess.Popen[bytes], str, Path]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        run_id: int,
        federation: str,
        run_config: dict[str, Any],
        contract: dict[str, Any],
    ) -> tuple[int, Path, Path]:
        cmd, output_dir = build_command(run_id, federation, run_config)
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_blob = json.dumps(contract, indent=2)
        contract_path = output_dir / "_fl_contract.json"
        contract_path.write_text(contract_blob)
        # Sim clients (Ray-spawned) read the contract from `data_dir.parent`,
        # which resolves to data/partitions/<partition>/. Drop a copy there
        # so the sim path works without mutating the partition tree itself.
        partition_name = run_config.get("partition-name")
        if partition_name:
            sim_root = REPO_ROOT / "data" / "partitions" / str(partition_name)
            if sim_root.is_dir():
                (sim_root / "_fl_contract.json").write_text(contract_blob)
        log_path = output_dir / "stdout.log"
        log_file = log_path.open("ab")
        log.info("starting run %s: %s", run_id, " ".join(cmd))
        # PYTHONUNBUFFERED forces line-flush — without it, Python pipes its
        # stdout in 4KB chunks when redirected to a file and we see almost
        # nothing in stdout.log for the entire run. Critical for debugging
        # mid-run hangs (e.g., serverapp waiting on a stuck client).
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
            env=env,
        )
        with self._lock:
            self._processes[run_id] = (process, federation, log_path)
        threading.Thread(
            target=self._wait_and_finalize,
            args=(run_id, process, log_file),
            daemon=True,
        ).start()
        log.info("started run %s pid=%s output=%s", run_id, process.pid, output_dir)
        return process.pid, log_path, output_dir

    def _wait_and_finalize(
        self, run_id: int, process: subprocess.Popen[bytes], log_file: IO[bytes]
    ) -> None:
        exit_code = process.wait()
        log_file.close()
        with self._lock:
            self._processes.pop(run_id, None)
        log.info("run %s finished with exit_code=%s", run_id, exit_code)
        with SessionLocal() as db:
            finalize_run(db, run_id, exit_code)

    def cancel(self, run_id: int) -> bool:
        """Cancel a running flwr submission.

        Two-step: (1) ask SuperLink to abort via `flwr stop` so the run is
        removed from state.db (otherwise SuperLink resumes it on next restart);
        (2) SIGTERM the local `flwr run --stream` wrapper.
        """
        with self._lock:
            entry = self._processes.get(run_id)
        if entry is None:
            return False
        process, federation, stdout_log = entry

        flwr_run_id = _parse_flwr_run_id(stdout_log)
        if flwr_run_id is not None:
            try:
                subprocess.run(
                    [_flwr_bin(), "stop", str(flwr_run_id), federation],
                    cwd=REPO_ROOT,
                    timeout=10,
                    capture_output=True,
                )
                log.info("flwr stop %s on federation %s sent", flwr_run_id, federation)
            except (subprocess.TimeoutExpired, OSError) as e:
                log.warning("flwr stop failed for %s: %s", flwr_run_id, e)
        else:
            log.warning("could not parse flwr_run_id for %s; SuperLink state may persist", run_id)

        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def is_alive(self, run_id: int) -> bool:
        with self._lock:
            entry = self._processes.get(run_id)
        if entry is None:
            return False
        process, _, _ = entry
        return process.poll() is None


_orchestrator: RunOrchestrator | None = None


def get_orchestrator() -> RunOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RunOrchestrator(RUNS_DATA_DIR)
    return _orchestrator

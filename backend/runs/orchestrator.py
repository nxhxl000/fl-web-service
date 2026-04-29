"""Subprocess manager for FL training runs.

Spawns `flwr run fl_app/ <federation> --federation-config ... --run-config ... --stream`
with `output-dir` forced to `runs_data/run_<id>/`. The training process writes
`events.jsonl` + `model_best.pt` there; `finalize_run` reads them and auto-registers
a TrainedModel for the project.
"""

from __future__ import annotations

import json
import logging
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


class RunOrchestrator:
    def __init__(self, runs_data_dir: Path) -> None:
        self._runs_data_dir = runs_data_dir
        self._runs_data_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
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
        contract_path = output_dir / "_fl_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2))
        log_path = output_dir / "stdout.log"
        log_file = log_path.open("ab")
        log.info("starting run %s: %s", run_id, " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
        with self._lock:
            self._processes[run_id] = process
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
        with self._lock:
            process = self._processes.get(run_id)
        if process is None:
            return False
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return False
        return True

    def is_alive(self, run_id: int) -> bool:
        with self._lock:
            process = self._processes.get(run_id)
        return process is not None and process.poll() is None


_orchestrator: RunOrchestrator | None = None


def get_orchestrator() -> RunOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RunOrchestrator(RUNS_DATA_DIR)
    return _orchestrator

"""Subprocess manager for FL training runs.

Day 5: dummy command only — proves the lifecycle (spawn, wait, finalize, cancel)
without touching the user's parallel ML experiments.

Day 6+ will replace DUMMY_COMMAND with `flwr run fl_app/ <federation> --run-config "..."`.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import threading
from pathlib import Path
from typing import IO

from backend.db import SessionLocal
from backend.runs.service import finalize_run

log = logging.getLogger(__name__)


# Dummy command: prints "round N/5" each 5s, total ~25s. Easy to test the lifecycle.
DUMMY_COMMAND: list[str] = [
    "python",
    "-c",
    (
        "import time, sys\n"
        "for i in range(1, 6):\n"
        "    print(f'round {i}/5', flush=True)\n"
        "    time.sleep(5)\n"
        "print('done', flush=True)\n"
    ),
]


class RunOrchestrator:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def start(self, run_id: int, command: list[str] | None = None) -> tuple[int, Path]:
        cmd = command or DUMMY_COMMAND
        log_path = self._log_dir / f"run_{run_id}.log"
        log_file = log_path.open("ab")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._lock:
            self._processes[run_id] = process
        threading.Thread(
            target=self._wait_and_finalize,
            args=(run_id, process, log_file),
            daemon=True,
        ).start()
        log.info("started run %s pid=%s log=%s", run_id, process.pid, log_path)
        return process.pid, log_path

    def _wait_and_finalize(
        self,
        run_id: int,
        process: subprocess.Popen[bytes],
        log_file: IO[bytes],
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
        log_dir = Path(__file__).resolve().parents[2] / "runs_data" / "logs"
        _orchestrator = RunOrchestrator(log_dir)
    return _orchestrator

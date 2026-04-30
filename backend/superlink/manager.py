"""Auto-spawn `flower-superlink` if no one is listening yet.

Idempotent: probe the Fleet+Control ports, only spawn when both are dead.
Detached: the SuperLink process survives backend restarts (new session,
no shared file descriptors with uvicorn). We never kill it from here —
backend just makes sure one is around when it boots.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)
ops = logging.getLogger("ops")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


_HOME = Path.home()

SUPERLINK_DB = _env_path("SUPERLINK_DB", _HOME / "flwr-superlink" / "state.db")
SUPERLINK_FFS = _env_path("SUPERLINK_FFS", _HOME / "flwr-superlink" / "ffs")
SUPERLINK_LOG = _env_path("SUPERLINK_LOG", _HOME / "flwr-superlink" / "superlink.log")
SUPERLINK_BIN = os.environ.get(
    "SUPERLINK_BIN", str(Path(sys.executable).parent / "flower-superlink")
)

FLEET_PORT = int(os.environ.get("SUPERLINK_FLEET_PORT", "9092"))
CONTROL_PORT = int(os.environ.get("SUPERLINK_CONTROL_PORT", "9093"))
HEALTH_TIMEOUT_S = float(os.environ.get("SUPERLINK_HEALTH_TIMEOUT_S", "20"))


def _tcp_alive(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def is_alive() -> bool:
    """Both Fleet and Control APIs reachable."""
    return _tcp_alive(FLEET_PORT) and _tcp_alive(CONTROL_PORT)


def ensure_running() -> None:
    """Probe SuperLink; spawn detached if dead. No-op if already alive."""
    if is_alive():
        ops.info(
            "superlink ensure: already alive on :%d/:%d — no-op",
            FLEET_PORT, CONTROL_PORT,
        )
        return

    ops.info("superlink ensure: not running, spawning")

    SUPERLINK_DB.parent.mkdir(parents=True, exist_ok=True)
    SUPERLINK_FFS.mkdir(parents=True, exist_ok=True)
    SUPERLINK_LOG.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        SUPERLINK_BIN,
        "--insecure",
        "--database", str(SUPERLINK_DB),
        "--storage-dir", str(SUPERLINK_FFS),
    ]
    # SuperLink spawns `flower-superexec` as a child by bare name (relies on
    # PATH). When uvicorn is launched as `.venv/bin/uvicorn ...` without venv
    # activation, the venv's bin/ is NOT in PATH and the spawn fails with
    # `FileNotFoundError: 'flower-superexec'`. Prepend it explicitly.
    bin_dir = str(Path(SUPERLINK_BIN).parent)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    child_env["PATH"] = bin_dir + os.pathsep + child_env.get("PATH", "")

    log_fp = open(SUPERLINK_LOG, "ab")  # noqa: SIM115 — owned by detached child
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detached: survives backend restart / SIGTERM
        env=child_env,
    )

    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if is_alive():
            ops.info(
                "superlink spawned pid=%d ports=%d,%d db=%s ffs=%s log=%s",
                proc.pid, FLEET_PORT, CONTROL_PORT,
                SUPERLINK_DB, SUPERLINK_FFS, SUPERLINK_LOG,
            )
            return
        if proc.poll() is not None:
            ops.error(
                "superlink spawn FAILED: exited with code %d before ports came up; "
                "see %s",
                proc.returncode, SUPERLINK_LOG,
            )
            raise RuntimeError(
                f"flower-superlink exited prematurely (code {proc.returncode}); "
                f"check {SUPERLINK_LOG}"
            )
        time.sleep(0.3)

    ops.error(
        "superlink spawn FAILED: ports not ready in %.0fs; pid=%d still running, see %s",
        HEALTH_TIMEOUT_S, proc.pid, SUPERLINK_LOG,
    )
    raise RuntimeError(
        f"flower-superlink ports not responding after {HEALTH_TIMEOUT_S}s; "
        f"pid={proc.pid}, check {SUPERLINK_LOG}"
    )

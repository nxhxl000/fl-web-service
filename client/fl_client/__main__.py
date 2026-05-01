"""fl-client bootstrap + heartbeat + flwr-supernode launcher.

Container entrypoint: validates the local data folder against the project
contract from the backend, writes `_fl_contract.json` next to the data,
starts a heartbeat thread, then exec's `flwr-supernode` with the data dir
pushed via `--node-config`.

Env vars:
  FL_TOKEN             (required) opaque project token
  FL_SERVER_URL        (required) backend base URL, e.g. https://api.example.com
  FL_SUPERLINK         (required) Flower SuperLink fleet addr, host:port
  FL_DATA_DIR          (default /data) ImageFolder root with class subdirs
  FL_HEARTBEAT_INTERVAL (default 30s)
  FL_INSECURE          (default 1) pass --insecure to flwr-supernode
"""

import email.utils
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_INTERVAL_SECONDS = 30
CONTRACT_FILENAME = "_fl_contract.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("fl-client")

_stop = threading.Event()


def _handle_signal(signum: int, _frame: object) -> None:
    log.info("received signal %s, shutting down", signum)
    _stop.set()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        log.error("%s is not set", name)
        sys.exit(2)
    return value


def _wait_for_clock_sync(
    server_url: str, max_wait_s: int = 300, threshold_s: float = 5.0
) -> None:
    """Block until our clock is within `threshold_s` of the backend's HTTP Date header.

    Cloud VMs can boot the container before NTP has finished syncing. Flower 1.28
    rejects gRPC handshakes with `Invalid timestamp` if drift exceeds its internal
    tolerance, and the bad handshake state poisons the heartbeat thread for the rest
    of the run. Probing our own backend's `Date` header sidesteps that race without
    needing UDP/123 to NTP servers.
    """
    deadline = time.time() + max_wait_s
    url = f"{server_url}/health"
    last_drift: float | None = None
    while True:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=5) as resp:
                date_hdr = resp.headers.get("Date")
            if date_hdr:
                server_time = email.utils.parsedate_to_datetime(date_hdr).timestamp()
                drift = abs(time.time() - server_time)
                last_drift = drift
                if drift <= threshold_s:
                    log.info("clock sync ok: drift=%.1fs", drift)
                    return
                log.info("clock sync: drift=%.1fs > %.1fs, waiting...", drift, threshold_s)
        except Exception as exc:  # noqa: BLE001
            log.warning("clock sync probe failed: %s", exc)
        if time.time() > deadline:
            log.error(
                "clock sync: giving up after %ds (last drift=%.1fs); supernode will likely fail",
                max_wait_s, last_drift if last_drift is not None else -1,
            )
            return
        time.sleep(3)


def _fetch_contract(server_url: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{server_url}/client/dataset-manifest",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _validate_local_data(data_dir: Path, class_names: list[str]) -> None:
    """Reject if local layout violates the contract.

    Rules:
      - each subdir must be in class_names (no foreign classes)
      - at least one subdir must intersect class_names with ≥1 image
    """
    if not data_dir.exists() or not data_dir.is_dir():
        log.error("FL_DATA_DIR=%s does not exist or is not a directory", data_dir)
        sys.exit(3)

    canonical = set(class_names)
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    extra = sorted(p.name for p in subdirs if p.name not in canonical)
    if extra:
        log.error(
            "Local data has classes not in the project contract: %s. "
            "Remove or rename them. Allowed: %s.",
            extra,
            sorted(canonical),
        )
        sys.exit(3)

    matched_with_images = []
    for p in subdirs:
        if p.name not in canonical:
            continue
        if any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
            matched_with_images.append(p.name)

    if not matched_with_images:
        log.error(
            "No class subdirectory under %s contains images matching the project contract.",
            data_dir,
        )
        sys.exit(3)

    log.info(
        "data ok: %d/%d project classes present locally (%s)",
        len(matched_with_images),
        len(canonical),
        ", ".join(sorted(matched_with_images)[:5])
        + ("…" if len(matched_with_images) > 5 else ""),
    )


def _write_contract(data_dir: Path, manifest: dict) -> None:
    contract = {
        "class_names": manifest["class_names"],
        "image_size": manifest.get("image_size"),
        "image_mode": manifest.get("image_mode"),
        "mean": manifest.get("mean"),
        "std": manifest.get("std"),
    }
    (data_dir / CONTRACT_FILENAME).write_text(json.dumps(contract, indent=2))


def _heartbeat_loop(server_url: str, token: str, interval: int) -> None:
    url = f"{server_url}/client/heartbeat"
    req_template = urllib.request.Request(
        url, method="POST", headers={"Authorization": f"Bearer {token}"}
    )
    while not _stop.is_set():
        try:
            with urllib.request.urlopen(req_template, timeout=10) as resp:
                log.debug("heartbeat ok (%s)", resp.status)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                log.error("server rejected token (401)")
                _stop.set()
                return
            log.warning("heartbeat failed: HTTP %s", exc.code)
        except Exception as exc:  # noqa: BLE001
            log.warning("heartbeat failed: %s", exc)
        _stop.wait(interval)


def _print_welcome_banner(*, project_name: str, node_name: str, num_classes: int) -> None:
    """Friendly status block shown right after the contract is verified.

    Goes to stderr/stdout via plain `print()` (not the logger) so the box
    stays intact regardless of log format. Helps non-technical participants
    confirm they actually joined.
    """
    line = "=" * 78
    print()
    print(line, flush=True)
    print(f"  ✓ Successfully joined project '{project_name}'", flush=True)
    print(f"    You are connected as '{node_name}'.", flush=True)
    print(f"    {num_classes} classes from your dataset are recognized.", flush=True)
    print()
    print("    • Keep this container running.", flush=True)
    print("    • Do NOT stop it during training — that drops you from the round.", flush=True)
    print("    • Your data never leaves this machine; only model updates are shared.", flush=True)
    print()
    print("    Thank you for contributing your compute!", flush=True)
    print(line, flush=True)
    print(flush=True)


def _start_supernode(
    superlink: str, data_dir: Path, node_name: str | None, insecure: bool
) -> subprocess.Popen:
    """Launch supernode as a subprocess so the heartbeat thread can keep running.

    `os.execvp` replaces the whole Python process and would kill the daemon
    heartbeat thread. With Popen, this process stays alive: heartbeat in the
    background, supernode in the foreground (we wait on it).
    """
    args = ["flower-supernode"]
    if insecure:
        args.append("--insecure")
    args += ["--superlink", superlink]
    nc = f'data-dir="{data_dir}"'
    if node_name:
        nc += f' node-name="{node_name}"'
    args += ["--node-config", nc]
    log.info("spawn: %s", " ".join(args))
    return subprocess.Popen(args)


def main() -> int:
    token = _require_env("FL_TOKEN")
    server_url = _require_env("FL_SERVER_URL").rstrip("/")
    superlink = _require_env("FL_SUPERLINK")
    data_dir = Path(os.environ.get("FL_DATA_DIR", "/data"))
    interval = int(os.environ.get("FL_HEARTBEAT_INTERVAL", DEFAULT_INTERVAL_SECONDS))
    insecure = os.environ.get("FL_INSECURE", "1") not in ("0", "false", "False")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _wait_for_clock_sync(server_url)

    log.info("fetching project contract from %s", server_url)
    try:
        manifest = _fetch_contract(server_url, token)
    except urllib.error.HTTPError as exc:
        log.error("fetch contract failed: HTTP %s — %s", exc.code, exc.reason)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("fetch contract failed: %s", exc)
        return 1

    log.info(
        "contract: project=%r num_classes=%s image=%sx%s",
        manifest.get("project_name"),
        manifest.get("num_classes"),
        *(manifest.get("image_size") or [None, None]),
    )

    class_names = manifest.get("class_names") or []
    if not class_names:
        log.error("contract has no class_names — admin must run Analyze on the project")
        return 1

    _validate_local_data(data_dir, class_names)
    _write_contract(data_dir, manifest)
    log.info("wrote %s", data_dir / CONTRACT_FILENAME)

    _print_welcome_banner(
        project_name=str(manifest.get("project_name") or "FL project"),
        node_name=str(manifest.get("node_name") or "client"),
        num_classes=int(manifest.get("num_classes") or 0),
    )

    threading.Thread(
        target=_heartbeat_loop,
        args=(server_url, token, interval),
        daemon=True,
    ).start()

    # Supernode supervision loop: when the SuperLink restarts (or any other
    # transient gRPC failure), flower-supernode exits. We restart it instead of
    # letting the container die — keeps the federation stable across server
    # restarts without depending on Docker --restart.
    current_proc: list[subprocess.Popen | None] = [None]

    def _forward(sig: int, _frame: object) -> None:
        log.info("forwarding signal %s, shutting down supervisor", sig)
        _stop.set()
        p = current_proc[0]
        if p is not None:
            try:
                p.send_signal(sig)
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    backoff_s = 5
    while not _stop.is_set():
        proc = _start_supernode(superlink, data_dir, manifest.get("node_name"), insecure)
        current_proc[0] = proc
        rc = proc.wait()
        current_proc[0] = None
        if _stop.is_set():
            return rc
        log.warning(
            "supernode exited (code=%s); restarting in %ss — likely SuperLink restart or transient gRPC error",
            rc, backoff_s,
        )
        _stop.wait(backoff_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

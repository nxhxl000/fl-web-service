"""fl-client heartbeat loop.

Reads FL_TOKEN and FL_SERVER_URL from the environment and POSTs a heartbeat
to {FL_SERVER_URL}/client/heartbeat every FL_HEARTBEAT_INTERVAL seconds.

Day 4 milestone: prove the Docker container can authenticate with an opaque
token and have its last_seen_at updated in the backend. No Flower yet.
"""

import logging
import os
import signal
import sys
import threading
import urllib.error
import urllib.request

DEFAULT_INTERVAL_SECONDS = 30

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


def _send_heartbeat(url: str, token: str) -> int:
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


def main() -> int:
    token = _require_env("FL_TOKEN")
    server_url = _require_env("FL_SERVER_URL").rstrip("/")
    interval = int(os.environ.get("FL_HEARTBEAT_INTERVAL", DEFAULT_INTERVAL_SECONDS))

    heartbeat_url = f"{server_url}/client/heartbeat"
    log.info("starting fl-client heartbeat loop -> %s (every %ss)", heartbeat_url, interval)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not _stop.is_set():
        try:
            status = _send_heartbeat(heartbeat_url, token)
            log.info("heartbeat ok (%s)", status)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                log.error("server rejected token (401), exiting")
                return 1
            log.warning("heartbeat failed: HTTP %s", exc.code)
        except urllib.error.URLError as exc:
            log.warning("heartbeat failed: %s", exc.reason)
        except Exception as exc:  # noqa: BLE001
            log.warning("heartbeat failed: %s", exc)

        _stop.wait(interval)

    log.info("fl-client stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

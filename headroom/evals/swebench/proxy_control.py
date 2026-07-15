"""Start / stop the Headroom proxy for one eval arm and read its /stats.

Each arm runs its own proxy process so we can flip compression on/off:
``baseline`` starts with ``--no-optimize`` (byte-exact passthrough), the treatment
arm starts with optimization on. Launch is via ``python -m headroom.proxy.server``
(the same entrypoint headroom/evals/suite_runner.py uses) which supports
``--no-optimize``, ``--anthropic-api-url``/``--openai-api-url``, ``--no-rate-limit``
and the ``HEADROOM_MODE`` env var.

HTTP is done with urllib (stdlib) so this module has no third-party deps.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import IO


def _port_available(port: int) -> bool:
    """True if we can bind 127.0.0.1:port right now (i.e. nothing is listening)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


class ProxyServer:
    """Context-manager wrapper around a single ``headroom.proxy.server`` process."""

    def __init__(
        self,
        *,
        port: int,
        optimize: bool,
        mode: str = "cache",
        upstream_anthropic_url: str = "",
        upstream_openai_url: str = "",
        skip_upstream_check: bool = False,
        ledger_path: str = "",
        log_path: str = "",
        ready_timeout_s: int = 180,
    ) -> None:
        self.port = port
        self.optimize = optimize
        self.mode = mode
        self.upstream_anthropic_url = upstream_anthropic_url
        self.upstream_openai_url = upstream_openai_url
        self.skip_upstream_check = skip_upstream_check
        self.ledger_path = ledger_path
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self._proc: subprocess.Popen | None = None
        self._log_fh: IO[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        # Refuse to start on an occupied port. Otherwise a foreign/leaked proxy
        # already bound here would answer /readyz while our child dies on
        # EADDRINUSE, and the arm would silently measure the WRONG proxy —
        # invalidating the whole A/B.
        if not _port_available(self.port):
            raise RuntimeError(
                f"port {self.port} is already in use — refusing to start a proxy there. "
                "An eval arm would otherwise measure a foreign proxy. Stop the process "
                "on that port, or pass a different --port."
            )

        cmd = [
            sys.executable,
            "-m",
            "headroom.proxy.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--no-rate-limit",
        ]
        if not self.optimize:
            cmd.append("--no-optimize")
        if self.upstream_anthropic_url:
            cmd += ["--anthropic-api-url", self.upstream_anthropic_url]
        if self.upstream_openai_url:
            cmd += ["--openai-api-url", self.upstream_openai_url]

        env = dict(os.environ)
        env["HEADROOM_MODE"] = self.mode
        # Telemetry defaults off; make sure the run is stateless-friendly and isolated.
        env.setdefault("HEADROOM_TELEMETRY", "off")
        if self.skip_upstream_check:
            env["HEADROOM_SKIP_UPSTREAM_CHECK"] = "1"
        if self.ledger_path:
            # Isolate this arm's durable savings ledger to a per-run file.
            env["HEADROOM_SAVINGS_EVENTS_PATH"] = self.ledger_path

        stdout: int | IO[bytes] = subprocess.DEVNULL
        if self.log_path:
            self._log_fh = open(self.log_path, "wb")  # noqa: SIM115 (closed in stop)
            stdout = self._log_fh
        try:
            self._proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env)
        except OSError:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None
            raise

    def wait_ready(self) -> bool:
        """Poll ``/readyz`` until 200 or timeout. Returns True on success."""
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False  # process died during startup
            try:
                with urllib.request.urlopen(f"{self.url}/readyz", timeout=3) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1.0)
        return False

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            finally:
                self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> ProxyServer:
        self.start()
        if not self.wait_ready():
            self.stop()
            raise RuntimeError(
                f"Headroom proxy did not become ready on port {self.port} "
                f"within {self.ready_timeout_s}s"
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# -- /stats helpers (also usable against an already-running proxy) ---------
def reset_stats(proxy_url: str, timeout: float = 5.0) -> bool:
    """POST /stats/reset (loopback-only) to zero in-memory metrics + cost tracker."""
    req = urllib.request.Request(f"{proxy_url.rstrip('/')}/stats/reset", method="POST", data=b"")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError):
        return False


def get_stats(proxy_url: str, timeout: float = 10.0) -> dict:
    """GET /stats and return the parsed JSON payload (empty dict on failure)."""
    try:
        with urllib.request.urlopen(f"{proxy_url.rstrip('/')}/stats", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {}


def is_up(proxy_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{proxy_url.rstrip('/')}/livez", timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, OSError):
        return False

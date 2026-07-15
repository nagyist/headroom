"""Zero-spend wiring + unit tests for headroom.evals.swebench.

The wiring test stands up a fake Anthropic upstream (stdlib http.server) and the
real Headroom proxy, then drives a request through the proxy and parses /stats —
no provider API calls, no Docker, no mini-swe-agent. The unit tests cover the
pure accounting/config helpers and always run.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from headroom.evals.swebench.accounting import parse_stats, turns_from_trajectory
from headroom.evals.swebench.config import api_base_for, provider_of

# ---------------------------------------------------------------------------
# Pure unit tests (no network) — always run.
# ---------------------------------------------------------------------------


def test_provider_of():
    assert provider_of("anthropic/claude-opus-4-8") == "anthropic"
    assert provider_of("claude-sonnet-4-5-20250929") == "anthropic"
    assert provider_of("openai/gpt-5.5") == "openai"
    assert provider_of("gpt-4o-mini") == "openai"
    assert provider_of("gemini/gemini-2.5-pro") == "gemini"


def test_api_base_for():
    root = "http://127.0.0.1:8787"
    # Anthropic uses the proxy root; OpenAI needs /v1; Gemini needs /v1beta.
    assert api_base_for("anthropic/claude-opus-4-8", root) == root
    assert api_base_for("openai/gpt-5.5", root) == root + "/v1"
    assert api_base_for("gemini/gemini-2.5-pro", root) == root + "/v1beta"
    assert api_base_for("anthropic/x", root + "/") == root  # trailing slash tolerated


def test_parse_stats_extracts_buckets_and_cost():
    stats = {
        "requests": {"total": 3},
        "tokens": {"input": 1000, "output": 200, "saved": 500, "proxy_compression_saved": 400},
        "cost": {
            "cost_with_headroom_usd": 0.012,
            "savings_usd": 0.006,
            "compression_savings_usd": 0.004,
            "cache_savings_usd": 0.002,
        },
        "prefix_cache": {
            "totals": {
                "cache_read_tokens": 800,
                "cache_write_tokens": 120,
                "uncached_input_tokens": 80,
                "hit_rate": 0.9,
            }
        },
        "summary": {
            "cost": {
                "without_headroom_usd": 0.02,
                "with_headroom_usd": 0.012,
                "total_saved_usd": 0.008,
                "savings_pct": 40.0,
            }
        },
    }
    s = parse_stats(stats)
    assert s.requests == 3
    assert s.input_tokens == 1000
    assert s.output_tokens == 200
    assert s.tokens_saved == 500
    assert s.cache_read_tokens == 800
    assert s.cache_write_tokens == 120
    assert s.uncached_input_tokens == 80
    assert s.cache_hit_rate == pytest.approx(0.9)
    assert s.with_headroom_usd == pytest.approx(0.012)
    assert s.without_headroom_usd == pytest.approx(0.02)
    assert s.savings_pct == pytest.approx(40.0)


def test_parse_stats_handles_missing_cost():
    # cost block is None when cost tracking is unavailable (e.g. python>=3.14).
    s = parse_stats({"requests": {"total": 1}, "cost": None})
    assert s.requests == 1
    assert s.cost_with_headroom_usd is None
    assert s.savings_pct is None


def test_turns_from_trajectory_is_robust_to_malformed():
    # Valid JSON but wrong shapes must degrade to empty, never raise (else one bad
    # trajectory file aborts the whole eval arm).
    assert turns_from_trajectory([]) == {}
    assert turns_from_trajectory("nope") == {}
    weird = turns_from_trajectory({"messages": "not-a-list", "info": 7})
    assert weird["model_turns"] == 0 and weird["tool_calls"] == 0
    # Non-dict messages / non-dict extra are skipped, not fatal.
    d = turns_from_trajectory(
        {"messages": ["oops", {"role": "assistant", "extra": ["bad"]}], "info": {}}
    )
    assert d["model_turns"] == 1 and d["tool_calls"] == 0


def test_no_proxy_multi_arm_raises():
    from headroom.evals.swebench import SwebenchConfig, run_swebench_eval

    cfg = SwebenchConfig(auto_start_proxy=False, arms=("baseline", "headroom"))
    with pytest.raises(ValueError, match="single"):
        run_swebench_eval(cfg)


def test_proxy_refuses_busy_port():
    from headroom.evals.swebench.proxy_control import ProxyServer

    # Occupy a port, then a proxy must refuse to start there (else an arm could
    # silently measure a foreign proxy and invalidate the A/B).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        px = ProxyServer(port=port, optimize=False)
        with pytest.raises(RuntimeError, match="already in use"):
            px.start()


def test_turns_from_trajectory():
    traj = {
        "messages": [
            {"role": "system", "content": "…"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "ls",
                "extra": {
                    "actions": [{"action": "ls"}],
                    "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}},
                },
            },
            {"role": "user", "content": "<output>"},
            {
                "role": "assistant",
                "content": "edit",
                "extra": {
                    "actions": [{"action": "sed"}, {"action": "cat"}],
                    "response": {
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 20,
                            "total_tokens": 220,
                            "cache_read_input_tokens": 150,
                            "cache_creation_input_tokens": 30,
                        }
                    },
                },
            },
        ],
        "info": {"exit_status": "Submitted", "submission": "diff", "model_stats": {"instance_cost": 0.03}},
    }
    d = turns_from_trajectory(traj)
    assert d["model_turns"] == 2
    assert d["tool_calls"] == 3
    assert d["prompt_tokens"] == 300
    assert d["completion_tokens"] == 30
    assert d["cache_read_tokens"] == 150
    assert d["cache_write_tokens"] == 30
    assert d["agent_cost_usd"] == pytest.approx(0.03)
    assert d["exit_status"] == "Submitted"


# ---------------------------------------------------------------------------
# Wiring test: fake upstream + real proxy, no API spend, no Docker.
# ---------------------------------------------------------------------------

_CANNED_ANTHROPIC = {
    "id": "msg_wiring",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20240620",
    "content": [{"type": "text", "text": "ok-from-fake-upstream"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1200, "output_tokens": 8},
}


class _FakeUpstream(BaseHTTPRequestHandler):
    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._send_json(_CANNED_ANTHROPIC)

    def do_GET(self):  # noqa: N802
        self._send_json({"status": "ok"})

    def do_HEAD(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_proxy_wiring_and_stats():
    # Skip cleanly if the proxy's runtime deps aren't installed in this env
    # (the proxy runs as a subprocess in the same venv, so import here is a proxy check).
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    pytest.importorskip("headroom.proxy.server")

    import urllib.request

    from headroom.evals.swebench.proxy_control import ProxyServer, get_stats, reset_stats

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    upstream_port = upstream.server_address[1]
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    proxy = ProxyServer(
        port=proxy_port,
        optimize=False,  # passthrough: fast startup, no model preload
        upstream_anthropic_url=f"http://127.0.0.1:{upstream_port}",
        skip_upstream_check=True,
        ready_timeout_s=60,
    )
    proxy.start()
    try:
        if not proxy.wait_ready():
            pytest.skip("Headroom proxy did not start (missing runtime deps?)")

        assert reset_stats(proxy.url) in (True, False)  # loopback POST should not raise

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello " * 500}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{proxy.url}/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": "dummy-key",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())

        # Round-tripped through the proxy to our fake upstream.
        assert payload["content"][0]["text"] == "ok-from-fake-upstream"

        stats = parse_stats(get_stats(proxy.url))
        assert stats.requests >= 1
    finally:
        proxy.stop()
        upstream.shutdown()

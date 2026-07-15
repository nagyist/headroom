"""Turn a proxy /stats payload and agent trajectories into eval numbers.

Two independent views of cost/tokens:

* **Proxy /stats** (authoritative for the A/B headline): the proxy's savings
  ledger already prices tokens and splits them into the four buckets that make
  cost != token-count (uncached input / output / cache-read / cache-write). We
  parse the fields documented at headroom/proxy/server.py:_build_stats_payload.
* **Trajectory** (per-instance detail): mini-SWE-agent records, per assistant
  message, ``extra.response.usage`` and ``extra.cost`` plus ``info.model_stats``.
  We use it for turn counts and a per-instance cost/usage cross-check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from headroom.evals.swebench.config import ArmStats, InstanceResult


def _d(x: object) -> dict:
    """Coerce to a dict (empty if not one) — keeps malformed input and mypy happy."""
    return x if isinstance(x, dict) else {}


def _num(d: dict, *keys: str, default: Any = 0) -> Any:
    """Nested get returning the first present numeric value."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else default


def parse_stats(stats: dict) -> ArmStats:
    """Extract the A/B headline numbers from a /stats payload."""
    tokens = stats.get("tokens") or {}
    cost = stats.get("cost") or {}
    prefix = (stats.get("prefix_cache") or {}).get("totals") or {}
    summary_cost = ((stats.get("summary") or {}).get("cost")) or {}
    requests = stats.get("requests") or {}

    def _opt(d: dict, key: str) -> float | None:
        v = d.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    return ArmStats(
        requests=int(_num(requests, "total")),
        input_tokens=int(_num(tokens, "input")),
        output_tokens=int(_num(tokens, "output")),
        tokens_saved=int(_num(tokens, "saved")),
        proxy_compression_saved=int(_num(tokens, "proxy_compression_saved")),
        total_before_compression=int(_num(tokens, "total_before_compression")),
        cache_read_tokens=int(_num(prefix, "cache_read_tokens")),
        cache_write_tokens=int(_num(prefix, "cache_write_tokens")),
        uncached_input_tokens=int(_num(prefix, "uncached_input_tokens")),
        cache_hit_rate=float(_num(prefix, "hit_rate", default=0.0)),
        cost_with_headroom_usd=_opt(cost, "cost_with_headroom_usd"),
        savings_usd=_opt(cost, "savings_usd"),
        compression_savings_usd=_opt(cost, "compression_savings_usd"),
        cache_savings_usd=_opt(cost, "cache_savings_usd"),
        without_headroom_usd=_opt(summary_cost, "without_headroom_usd"),
        with_headroom_usd=_opt(summary_cost, "with_headroom_usd"),
        total_saved_usd=_opt(summary_cost, "total_saved_usd"),
        savings_pct=_opt(summary_cost, "savings_pct"),
        raw={"tokens": tokens, "cost": cost, "prefix_cache_totals": prefix, "summary_cost": summary_cost},
    )


def _usage_cache_read(usage: dict) -> int:
    for k in ("cache_read_input_tokens", "cached_tokens"):
        v = usage.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), (int, float)):
        return int(details["cached_tokens"])
    return 0


def _usage_cache_write(usage: dict) -> int:
    v = usage.get("cache_creation_input_tokens")
    return int(v) if isinstance(v, (int, float)) else 0


def turns_from_trajectory(traj: dict) -> dict:
    """Count turns/tool-calls and sum per-message usage from a mini-SWE-agent traj.

    Returns a plain dict of the fields InstanceResult needs (turns, tool_calls,
    exit_status, per-message token totals, instance cost).
    """
    if not isinstance(traj, dict):
        return {}
    messages_raw = traj.get("messages")
    messages: list = messages_raw if isinstance(messages_raw, list) else []
    info = _d(traj.get("info"))
    model_stats = _d(info.get("model_stats"))

    model_turns = 0
    tool_calls = 0
    prompt = completion = total = cread = cwrite = 0

    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        model_turns += 1
        extra = _d(m.get("extra"))
        actions = extra.get("actions")
        if isinstance(actions, list):
            tool_calls += len(actions)
        elif isinstance(m.get("tool_calls"), list):
            tool_calls += len(m["tool_calls"])
        resp = _d(extra.get("response"))
        usage = _d(resp.get("usage"))
        prompt += int(_num(usage, "prompt_tokens"))
        completion += int(_num(usage, "completion_tokens"))
        total += int(_num(usage, "total_tokens"))
        cread += _usage_cache_read(usage)
        cwrite += _usage_cache_write(usage)

    instance_cost = model_stats.get("instance_cost")
    return {
        "model_turns": model_turns,
        "tool_calls": tool_calls,
        "exit_status": str(info.get("exit_status") or ""),
        "submission": str(info.get("submission") or ""),
        "agent_cost_usd": float(instance_cost) if isinstance(instance_cost, (int, float)) else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_read_tokens": cread,
        "cache_write_tokens": cwrite,
    }


def instance_result_from_files(
    *, instance_id: str, arm: str, traj_path: Path | None, model_patch: str
) -> InstanceResult:
    """Build an InstanceResult from a trajectory file + its prediction patch."""
    data: dict = {}
    if traj_path is not None and traj_path.exists():
        try:
            traj = json.loads(traj_path.read_text())
            data = turns_from_trajectory(traj)
        except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
            data = {}
    return InstanceResult(
        instance_id=instance_id,
        arm=arm,
        model_turns=int(data.get("model_turns", 0)),
        tool_calls=int(data.get("tool_calls", 0)),
        exit_status=str(data.get("exit_status", "")),
        empty_patch=not (model_patch or "").strip(),
        agent_cost_usd=data.get("agent_cost_usd"),
        prompt_tokens=int(data.get("prompt_tokens", 0)),
        completion_tokens=int(data.get("completion_tokens", 0)),
        total_tokens=int(data.get("total_tokens", 0)),
        cache_read_tokens=int(data.get("cache_read_tokens", 0)),
        cache_write_tokens=int(data.get("cache_write_tokens", 0)),
    )

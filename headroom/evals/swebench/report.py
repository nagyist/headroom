"""Write eval outputs: results.csv (per instance), summary.json, report.md/html.

All headline savings numbers come from the proxy /stats; correctness from the
grader; turns from trajectories. The report deliberately surfaces the four token
buckets separately, because compression can lower total tokens yet raise cost
(cache-write is ~12.5x cache-read for Anthropic).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from headroom.evals.swebench.config import BASELINE_ARM, ArmResult, ArmStats, EvalResult


def billed_usd(s: ArmStats) -> float | None:
    """Actual billed input cost for an arm.

    ``cost.cost_with_headroom_usd`` is populated in BOTH arms (it's the real
    litellm-priced billed cost); ``summary.cost.with_headroom_usd`` is only
    meaningful once optimization has run (0.0 in a passthrough baseline), so it's
    a fallback only.
    """
    return s.cost_with_headroom_usd if s.cost_with_headroom_usd is not None else s.with_headroom_usd


def saved_usd(s: ArmStats) -> float | None:
    return s.total_saved_usd if s.total_saved_usd is not None else s.savings_usd

RESULTS_COLUMNS = [
    "run_id",
    "arm",
    "instance_id",
    "model",
    "resolved",
    "model_turns",
    "tool_calls",
    "exit_status",
    "empty_patch",
    "agent_cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
]


def write_results_csv(result: EvalResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RESULTS_COLUMNS)
        for arm in result.arms:
            for inst in arm.instances:
                w.writerow(
                    [
                        result.run_id,
                        arm.arm,
                        inst.instance_id,
                        result.config.model,
                        "" if inst.resolved is None else inst.resolved,
                        inst.model_turns,
                        inst.tool_calls,
                        inst.exit_status,
                        inst.empty_patch,
                        "" if inst.agent_cost_usd is None else round(inst.agent_cost_usd, 6),
                        inst.prompt_tokens,
                        inst.completion_tokens,
                        inst.cache_read_tokens,
                        inst.cache_write_tokens,
                        inst.total_tokens,
                    ]
                )


def _arm_summary(arm: ArmResult) -> dict:
    s = arm.stats
    return {
        "arm": arm.arm,
        "optimize": arm.optimize,
        "instances": arm.n_instances,
        "resolved": arm.n_resolved,
        "resolve_rate": arm.resolve_rate,
        "median_turns": arm.median_turns,
        "proxy_requests": s.requests,
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "tokens_saved": s.tokens_saved,
        "cache_read_tokens": s.cache_read_tokens,
        "cache_write_tokens": s.cache_write_tokens,
        "uncached_input_tokens": s.uncached_input_tokens,
        "cache_hit_rate": s.cache_hit_rate,
        "cost_with_headroom_usd": billed_usd(s),
        "cost_without_headroom_usd": s.without_headroom_usd,
        "total_saved_usd": saved_usd(s),
        "savings_pct": s.savings_pct,
    }


def build_summary(result: EvalResult) -> dict:
    arms = {a.arm: _arm_summary(a) for a in result.arms}
    summary = {
        "run_id": result.run_id,
        "model": result.config.model,
        "dataset": result.dataset_name,
        "subset": result.config.subset,
        "split": result.config.split,
        "arms": arms,
    }
    base = result.arm(BASELINE_ARM)
    treat = next((a for a in result.arms if a.arm != BASELINE_ARM), None)
    if base is not None and treat is not None:
        b, t = _arm_summary(base), _arm_summary(treat)

        def _delta(key: str) -> float | None:
            bv, tv = b.get(key), t.get(key)
            return (tv - bv) if isinstance(bv, (int, float)) and isinstance(tv, (int, float)) else None

        def _pct(key: str) -> float | None:
            bv, tv = b.get(key), t.get(key)
            if isinstance(bv, (int, float)) and bv and isinstance(tv, (int, float)):
                return round((tv - bv) / bv * 100, 1)
            return None

        summary["delta_headroom_vs_baseline"] = {
            "resolve_rate": _delta("resolve_rate"),
            "median_turns": _delta("median_turns"),
            "input_tokens_pct": _pct("input_tokens"),
            "total_saved_usd": t.get("total_saved_usd"),
            "cost_with_headroom_usd_pct": _pct("cost_with_headroom_usd"),
        }
    return summary


def write_summary_json(result: EvalResult, path: str) -> None:
    Path(path).write_text(json.dumps(build_summary(result), indent=2, default=str))


def _fmt(v: object, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(result: EvalResult) -> str:
    summary = build_summary(result)
    lines = [
        f"# SWE-bench A/B eval — `{result.config.model}`",
        "",
        f"- **run_id**: `{result.run_id}`",
        f"- **dataset**: `{result.dataset_name}` (subset `{result.config.subset}`, split `{result.config.split}`)",
        "",
        "| metric | " + " | ".join(a.arm for a in result.arms) + " |",
        "|---|" + "---|" * len(result.arms),
    ]

    def row(label: str, fn: Callable[[ArmResult], str]) -> str:
        return "| " + label + " | " + " | ".join(fn(a) for a in result.arms) + " |"

    lines.append(row("instances", lambda a: str(a.n_instances)))
    lines.append(row("resolved", lambda a: f"{a.n_resolved} ({_fmt(float(a.resolve_rate) * 100, 1)}%)" if a.resolve_rate is not None else "—"))
    lines.append(row("median turns", lambda a: _fmt(a.median_turns, 1)))
    lines.append(row("proxy requests", lambda a: str(a.stats.requests)))
    lines.append(row("input tokens", lambda a: f"{a.stats.input_tokens:,}"))
    lines.append(row("output tokens", lambda a: f"{a.stats.output_tokens:,}"))
    lines.append(row("tokens saved", lambda a: f"{a.stats.tokens_saved:,}"))
    lines.append(row("cache read tok", lambda a: f"{a.stats.cache_read_tokens:,}"))
    lines.append(row("cache write tok", lambda a: f"{a.stats.cache_write_tokens:,}"))
    lines.append(row("cache hit rate", lambda a: _fmt(a.stats.cache_hit_rate, 2)))
    lines.append(row("cost w/ headroom $", lambda a: _fmt(billed_usd(a.stats), 4)))
    lines.append(row("saved $", lambda a: _fmt(saved_usd(a.stats), 4)))
    lines.append(row("savings %", lambda a: _fmt(a.stats.savings_pct, 1)))

    delta = summary.get("delta_headroom_vs_baseline")
    if delta:
        lines += [
            "",
            "## Headroom vs baseline",
            "",
            f"- resolve-rate Δ: **{_fmt(float(delta['resolve_rate']) * 100, 1) if delta['resolve_rate'] is not None else '—'} pts**",
            f"- input-token Δ: **{_fmt(delta['input_tokens_pct'], 1)}%**",
            f"- cost Δ: **{_fmt(delta['cost_with_headroom_usd_pct'], 1)}%**  (saved ${_fmt(delta['total_saved_usd'], 4)})",
            f"- median-turns Δ: **{_fmt(delta['median_turns'], 1)}**",
            "",
            "> Token count ≠ cost: cache-write is ~12.5× cache-read for Anthropic, so watch "
            "the cache buckets, not just totals.",
        ]
    return "\n".join(lines) + "\n"


def render_html(result: EvalResult) -> str:
    arms = result.arms

    def th(a: ArmResult) -> str:
        return f"<th>{a.arm}</th>"

    def tr(label: str, fn: Callable[[ArmResult], object]) -> str:
        cells = "".join(f"<td>{fn(a)}</td>" for a in arms)
        return f"<tr><th class='rowlabel'>{label}</th>{cells}</tr>"

    rows = [
        tr("instances", lambda a: a.n_instances),
        tr("resolved", lambda a: f"{a.n_resolved} ({_fmt(float(a.resolve_rate) * 100, 1)}%)" if a.resolve_rate is not None else "—"),
        tr("median turns", lambda a: _fmt(a.median_turns, 1)),
        tr("proxy requests", lambda a: a.stats.requests),
        tr("input tokens", lambda a: f"{a.stats.input_tokens:,}"),
        tr("output tokens", lambda a: f"{a.stats.output_tokens:,}"),
        tr("tokens saved", lambda a: f"{a.stats.tokens_saved:,}"),
        tr("cache read tok", lambda a: f"{a.stats.cache_read_tokens:,}"),
        tr("cache write tok", lambda a: f"{a.stats.cache_write_tokens:,}"),
        tr("cache hit rate", lambda a: _fmt(a.stats.cache_hit_rate, 2)),
        tr("cost w/ headroom $", lambda a: _fmt(billed_usd(a.stats), 4)),
        tr("saved $", lambda a: _fmt(saved_usd(a.stats), 4)),
        tr("savings %", lambda a: _fmt(a.stats.savings_pct, 1)),
    ]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SWE-bench A/B — {result.config.model}</title>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .meta {{ color: #555; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; min-width: 480px; }}
  th, td {{ padding: 6px 14px; text-align: right; border-bottom: 1px solid #e5e5e5; }}
  th.rowlabel {{ text-align: left; color: #444; font-weight: 500; }}
  thead th {{ text-align: right; border-bottom: 2px solid #ccc; }}
  code {{ background: #f2f2f2; padding: 1px 5px; border-radius: 4px; }}
  .note {{ color: #666; font-size: 0.9rem; margin-top: 1rem; max-width: 640px; }}
</style></head><body>
<h1>SWE-bench A/B eval — <code>{result.config.model}</code></h1>
<div class="meta">run <code>{result.run_id}</code> · dataset <code>{result.dataset_name}</code>
  · subset <code>{result.config.subset}</code> · split <code>{result.config.split}</code></div>
<table><thead><tr><th class="rowlabel">metric</th>{"".join(th(a) for a in arms)}</tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="note">All token/cost/savings numbers come from the Headroom proxy's
<code>/stats</code>; correctness from the official SWE-bench grader; turns from agent
trajectories. Token count ≠ cost — cache-write is ~12.5× cache-read for Anthropic,
so the cache buckets matter as much as the totals.</p>
</body></html>
"""


def write_outputs(result: EvalResult, base_dir: str) -> dict[str, str]:
    """Write results.csv, summary.json, report.md, report.html. Returns paths."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": str(base / "results.csv"),
        "summary_json": str(base / "summary.json"),
        "report_md": str(base / "report.md"),
        "report_html": str(base / "report.html"),
    }
    write_results_csv(result, paths["results_csv"])
    write_summary_json(result, paths["summary_json"])
    Path(paths["report_md"]).write_text(render_markdown(result))
    Path(paths["report_html"]).write_text(render_html(result))
    return paths


# Kept for callers that want the raw dataclasses as dicts.
def result_as_dict(result: EvalResult) -> dict:
    return {
        "run_id": result.run_id,
        "dataset": result.dataset_name,
        "config": asdict(result.config),
        "arms": [
            {
                "arm": a.arm,
                "optimize": a.optimize,
                "stats": asdict(a.stats),
                "resolved_ids": a.resolved_ids,
                "grade_counts": a.grade_counts,
                "instances": [asdict(i) for i in a.instances],
            }
            for a in result.arms
        ],
    }

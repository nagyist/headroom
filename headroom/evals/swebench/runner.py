"""Orchestrate the SWE-bench A/B run: for each arm, start the proxy in the right
mode, run the agent through it, snapshot /stats, grade, and collect per-instance
turns. Returns an :class:`EvalResult`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from headroom.evals.swebench import agent_runner
from headroom.evals.swebench import grade as grading
from headroom.evals.swebench.accounting import instance_result_from_files, parse_stats
from headroom.evals.swebench.config import (
    BASELINE_ARM,
    ArmResult,
    ArmStats,
    EvalResult,
    SwebenchConfig,
)
from headroom.evals.swebench.proxy_control import ProxyServer, get_stats, is_up, reset_stats


def _log(msg: str) -> None:
    print(f"[swebench-eval] {msg}", flush=True)


def _run_arm(cfg: SwebenchConfig, arm: str, base_dir: str, run_id: str) -> ArmResult:
    optimize = arm != BASELINE_ARM
    arm_dir = os.path.join(base_dir, arm)
    os.makedirs(arm_dir, exist_ok=True)
    proxy_root = f"http://127.0.0.1:{cfg.port}"

    _log(f"arm '{arm}': optimize={optimize} mode={cfg.mode if optimize else 'passthrough'}")

    proxy: ProxyServer | None = None
    if cfg.auto_start_proxy:
        proxy = ProxyServer(
            port=cfg.port,
            optimize=optimize,
            mode=cfg.mode,
            upstream_anthropic_url=cfg.upstream_anthropic_url,
            upstream_openai_url=cfg.upstream_openai_url,
            skip_upstream_check=cfg.skip_upstream_check,
            ledger_path=os.path.join(arm_dir, "savings_events.jsonl"),
            log_path=os.path.join(arm_dir, "proxy.log"),
            ready_timeout_s=cfg.ready_timeout_s,
        )
        _log(f"arm '{arm}': starting proxy on :{cfg.port} …")
        proxy.start()
        if not proxy.wait_ready():
            proxy.stop()
            raise RuntimeError(
                f"proxy failed to become ready for arm '{arm}' (see {arm_dir}/proxy.log)"
            )
    elif not is_up(proxy_root):
        raise RuntimeError(
            f"--no-proxy set but no proxy is listening on {proxy_root}. "
            "Start one, or run a single arm."
        )

    stats = ArmStats()
    preds_path = str(Path(arm_dir) / "preds.json")
    try:
        if not reset_stats(proxy_root):
            _log(
                f"arm '{arm}': WARNING /stats/reset failed — token/cost numbers may "
                "include counters from a prior arm"
            )
        _log(f"arm '{arm}': running agent over subset='{cfg.subset}' split='{cfg.split}' …")
        preds_path = agent_runner.run_agent_batch(cfg, proxy_root, arm_dir)
        stats = parse_stats(get_stats(proxy_root))
    finally:
        if proxy is not None:
            proxy.stop()

    # -- per-instance turns/usage from trajectories + predictions --
    preds = agent_runner.load_predictions(preds_path)
    instances = [
        instance_result_from_files(
            instance_id=iid,
            arm=arm,
            traj_path=agent_runner.trajectory_path(arm_dir, iid),
            model_patch=str(pred.get("model_patch", "")),
        )
        for iid, pred in preds.items()
    ]
    _log(f"arm '{arm}': {len(instances)} instances ran, {stats.requests} proxy requests")

    result = ArmResult(
        arm=arm,
        optimize=optimize,
        stats=stats,
        instances=instances,
        preds_path=preds_path,
        out_dir=arm_dir,
    )

    # -- grade --
    if cfg.grade and instances:
        _log(f"arm '{arm}': grading with the official SWE-bench harness …")
        g = grading.grade_predictions(
            preds_path=preds_path,
            dataset_name=cfg.grader_dataset_name(),
            split=cfg.split,
            run_id=f"{run_id}-{arm}",
            out_dir=arm_dir,
            max_workers=cfg.grade_workers,
        )
        result.resolved_ids = g.resolved_ids
        result.grade_counts = g.counts
        if g.ok:
            resolved = set(g.resolved_ids)
            for inst in instances:
                inst.resolved = inst.instance_id in resolved
            _log(f"arm '{arm}': resolved {len(resolved)}/{len(instances)}")
        else:
            _log(f"arm '{arm}': grading unavailable ({g.error}); resolved left unknown")

    return result


def run_swebench_eval(cfg: SwebenchConfig) -> EvalResult:
    """Run every arm and return the combined result."""
    if not cfg.auto_start_proxy and len(cfg.arms) > 1:
        raise ValueError(
            "--no-proxy runs every arm against one long-lived proxy whose optimize mode "
            "cannot be flipped between arms, so the arms would be identical. Run a single "
            "arm (e.g. --arms headroom)."
        )
    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_dir = cfg.output_dir or os.path.join(os.getcwd(), f"swebench-eval-{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    _log(f"run_id={run_id} model={cfg.model} arms={list(cfg.arms)} out={base_dir}")

    result = EvalResult(
        config=cfg, dataset_name=cfg.grader_dataset_name(), run_id=run_id, base_dir=base_dir
    )
    for arm in cfg.arms:
        result.arms.append(_run_arm(cfg, arm, base_dir, run_id))
    return result

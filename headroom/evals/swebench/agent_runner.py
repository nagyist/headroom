"""Drive mini-SWE-agent over a SWE-bench subset, routed through the proxy.

We subprocess mini-SWE-agent's batch runner
(``python -m minisweagent.run.benchmarks.swebench``) rather than importing it, so
we lean on its dataset loading, per-instance Docker environment, parallelism and
``preds.json`` writing. The only thing we inject is
``model.model_kwargs.api_base`` pointing at the Headroom proxy (litellm forwards
``api_base`` verbatim).

Requires the optional deps: ``pip install headroom-ai[swebench]`` and a running
Docker daemon (SWE-bench instance images are x86_64).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from headroom.evals.swebench.config import SwebenchConfig, api_base_for


def build_agent_command(cfg: SwebenchConfig, proxy_root: str, out_dir: str) -> list[str]:
    """Construct the mini-SWE-agent batch-runner command for one arm."""
    api_base = api_base_for(cfg.model, proxy_root)
    cmd = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        cfg.subset,
        "--split",
        cfg.split,
        "--output",
        out_dir,
        "--workers",
        str(cfg.workers),
        "--model",
        cfg.model,
        # NOTE: the FIRST -c REPLACES the builtin default, so we re-add the base
        # config explicitly before layering key=value overrides on top.
        "--config",
        cfg.config_spec,
        "--config",
        f"model.model_kwargs.api_base={api_base}",
        "--config",
        f"agent.step_limit={cfg.step_limit}",
        "--config",
        f"agent.cost_limit={cfg.cost_limit}",
    ]
    if cfg.reasoning_effort:
        cmd += ["--config", f"model.model_kwargs.reasoning_effort={cfg.reasoning_effort}"]
    if cfg.slice:
        cmd += ["--slice", cfg.slice]
    if cfg.instances:
        cmd += ["--filter", cfg.instances]
    return cmd


def run_agent_batch(cfg: SwebenchConfig, proxy_root: str, out_dir: str) -> str:
    """Run the agent over the subset. Returns the path to ``preds.json``."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_agent_command(cfg, proxy_root, out_dir)

    env = dict(os.environ)
    # The proxy may forward a model name litellm can't price locally; don't let
    # mini-swe-agent abort the whole run on a pricing miss.
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    # Keep tool output deterministic/uncoloured inside the agent's shell.
    env.setdefault("TQDM_DISABLE", "1")

    subprocess.run(cmd, env=env, check=False)
    return str(Path(out_dir) / "preds.json")


def load_predictions(preds_path: str) -> dict[str, dict]:
    """Load mini-SWE-agent preds.json (dict keyed by instance_id)."""
    p = Path(preds_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, list):  # tolerate .jsonl-style list
        return {d.get("instance_id", str(i)): d for i, d in enumerate(data)}
    return data if isinstance(data, dict) else {}


def trajectory_path(out_dir: str, instance_id: str) -> Path | None:
    """Locate ``<out>/<iid>/<iid>.traj.json`` (with a glob fallback)."""
    direct = Path(out_dir) / instance_id / f"{instance_id}.traj.json"
    if direct.exists():
        return direct
    matches = list(Path(out_dir).glob(f"{instance_id}/*.traj.json"))
    return matches[0] if matches else None

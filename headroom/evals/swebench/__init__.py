"""SWE-bench A/B agentic eval for Headroom.

Runs mini-SWE-agent over public SWE-bench tasks twice — proxy passthrough
("baseline") vs compression on ("headroom") — and compares resolved rate, tokens,
cost and turns. See ``README.md`` in this package.

Public API::

    from headroom.evals.swebench import SwebenchConfig, run_swebench_eval, write_outputs

    result = run_swebench_eval(SwebenchConfig(model="anthropic/claude-opus-4-8", slice="0:5"))
    paths = write_outputs(result, "out/")

Requires ``pip install headroom-ai[swebench]`` and a running Docker daemon for the
agent's per-instance environment and the official grader.
"""

from __future__ import annotations

from headroom.evals.swebench.config import (
    ArmResult,
    ArmStats,
    EvalResult,
    InstanceResult,
    SwebenchConfig,
)
from headroom.evals.swebench.report import build_summary, render_markdown, write_outputs
from headroom.evals.swebench.runner import run_swebench_eval

__all__ = [
    "SwebenchConfig",
    "EvalResult",
    "ArmResult",
    "ArmStats",
    "InstanceResult",
    "run_swebench_eval",
    "write_outputs",
    "build_summary",
    "render_markdown",
]

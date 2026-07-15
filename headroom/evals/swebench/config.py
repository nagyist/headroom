"""Types and defaults for the SWE-bench A/B agentic eval.

The eval runs a real coding agent (mini-SWE-agent) over public SWE-bench tasks
twice — once with the Headroom proxy in **passthrough** mode ("baseline") and once
with compression **on** ("headroom") — then compares resolved rate, tokens, cost,
and turns. All headline token/cost/savings numbers come from the proxy's own
``/stats`` endpoint (the savings ledger already prices tokens); per-instance turn
counts come from the agent trajectories; correctness comes from the official
SWE-bench grader.

This module is intentionally stdlib-only so ``import headroom.evals.swebench`` and
``python -m headroom.evals swebench --help`` work even when the optional
``headroom-ai[swebench]`` dependencies (mini-swe-agent, swebench, datasets) are
not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# mini-SWE-agent subset -> HuggingFace dataset used by the OFFICIAL grader.
# (mini-swe-agent's own DATASET_MAPPING uses a capital "B"; both resolve on HF.)
GRADER_DATASET = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
    "multimodal": "princeton-nlp/SWE-bench_Multimodal",
}

# The two arms we compare. "baseline" == proxy passthrough (optimize off);
# any other arm name runs with compression on.
BASELINE_ARM = "baseline"


def provider_of(model: str) -> str:
    """Return the litellm provider family for a model string.

    Used to decide the proxy ``api_base`` suffix: Anthropic clients post to
    ``{api_base}/v1/messages`` (so api_base is the proxy root), OpenAI clients
    post to ``{api_base}/chat/completions`` (so api_base needs a ``/v1`` suffix).
    """
    m = model.lower()
    head = m.split("/", 1)[0]
    if head in ("openai", "azure", "openrouter") or m.startswith(("gpt-", "gpt4", "o1", "o3", "o4")):
        return "openai"
    if head in ("gemini", "vertex_ai", "vertex") or m.startswith("gemini"):
        return "gemini"
    # Anthropic is the default (matches mini-swe-agent's default model).
    return "anthropic"


def api_base_for(model: str, proxy_root: str) -> str:
    """Proxy ``api_base`` a litellm client should use for this model.

    Anthropic -> proxy root (litellm appends ``/v1/messages``).
    OpenAI    -> proxy root + ``/v1`` (litellm appends ``/chat/completions``).
    Gemini    -> proxy root + ``/v1beta`` (litellm's custom-base gemini client builds
                 ``{api_base}/models/{model}:generateContent``, which must land on the
                 proxy's ``/v1beta/models/...`` handler).

    NB: ``vertex_ai/*`` is bucketed as "gemini" by :func:`provider_of` but litellm
    builds vertex URLs differently, so only Anthropic / OpenAI / Gemini-AI-Studio
    families are wired end-to-end today.
    """
    proxy_root = proxy_root.rstrip("/")
    provider = provider_of(model)
    if provider == "openai":
        return f"{proxy_root}/v1"
    if provider == "gemini":
        return f"{proxy_root}/v1beta"
    return proxy_root


@dataclass
class SwebenchConfig:
    """Everything a SWE-bench A/B run needs."""

    # --- agent / dataset ---
    model: str = "anthropic/claude-sonnet-4-5-20250929"
    subset: str = "verified"          # verified | lite | full | <hf dataset path>
    split: str = "test"               # SWE-bench Verified/Lite scoring uses "test"
    slice: str = ""                   # e.g. "0:5" — run only the first 5 instances
    instances: str = ""               # regex filter on instance_id
    workers: int = 1                  # agent-run parallelism (default 1)
    config_spec: str = "swebench.yaml"  # mini-swe-agent config; *_backticks.yaml for text models
    reasoning_effort: str = ""        # "", low, medium, high, xhigh
    step_limit: int = 250
    cost_limit: float = 3.0

    # --- arms ---
    arms: tuple[str, ...] = (BASELINE_ARM, "headroom")
    mode: str = "cache"               # Headroom optimize mode for the treatment arm

    # --- proxy ---
    port: int = 8787
    auto_start_proxy: bool = True
    ready_timeout_s: int = 180
    # Point the proxy's UPSTREAM somewhere other than the real provider
    # (a local simulator / self-hosted model). Empty = real provider.
    upstream_anthropic_url: str = ""
    upstream_openai_url: str = ""
    skip_upstream_check: bool = False  # set when upstream is a mock (avoids /readyz 503)

    # --- grading ---
    grade: bool = True                # run the official SWE-bench Docker grader
    grade_workers: int = 4

    # --- output ---
    output_dir: str = ""              # base dir; resolved to a real path in the runner

    def grader_dataset_name(self) -> str:
        return GRADER_DATASET.get(self.subset, self.subset)


@dataclass
class ArmStats:
    """Headline numbers parsed from the proxy ``/stats`` payload for one arm."""

    requests: int = 0
    input_tokens: int = 0             # forwarded (post-compression) input total
    output_tokens: int = 0
    tokens_saved: int = 0             # all layers
    proxy_compression_saved: int = 0
    total_before_compression: int = 0
    # prefix-cache buckets (this is why token-count != cost)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_hit_rate: float = 0.0
    # dollars (None when cost tracking is unavailable, e.g. python>=3.14)
    cost_with_headroom_usd: float | None = None
    savings_usd: float | None = None
    compression_savings_usd: float | None = None
    cache_savings_usd: float | None = None
    without_headroom_usd: float | None = None
    with_headroom_usd: float | None = None
    total_saved_usd: float | None = None
    savings_pct: float | None = None
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class InstanceResult:
    """Per-instance outcome for one arm."""

    instance_id: str
    arm: str
    resolved: bool | None = None      # None = not graded
    model_turns: int = 0              # assistant messages
    tool_calls: int = 0              # bash tool calls across the trajectory
    exit_status: str = ""
    empty_patch: bool = True
    agent_cost_usd: float | None = None  # from trajectory model_stats.instance_cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ArmResult:
    arm: str
    optimize: bool
    stats: ArmStats
    instances: list[InstanceResult] = field(default_factory=list)
    resolved_ids: list[str] = field(default_factory=list)
    grade_counts: dict = field(default_factory=dict)
    preds_path: str = ""
    out_dir: str = ""

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def n_resolved(self) -> int:
        return sum(1 for i in self.instances if i.resolved is True)

    @property
    def resolve_rate(self) -> float | None:
        if not self.instances or all(i.resolved is None for i in self.instances):
            return None
        graded = [i for i in self.instances if i.resolved is not None]
        return (sum(1 for i in graded if i.resolved) / len(graded)) if graded else None

    @property
    def median_turns(self) -> float:
        vals = sorted(i.model_turns for i in self.instances)
        if not vals:
            return 0.0
        mid = len(vals) // 2
        return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


@dataclass
class EvalResult:
    config: SwebenchConfig
    dataset_name: str
    run_id: str
    base_dir: str = ""
    arms: list[ArmResult] = field(default_factory=list)

    def arm(self, name: str) -> ArmResult | None:
        return next((a for a in self.arms if a.arm == name), None)

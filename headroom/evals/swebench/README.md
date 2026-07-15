# SWE-bench A/B eval

Run a real coding agent (**mini-SWE-agent**) over **public SWE-bench** tasks
twice — once with the Headroom proxy in **passthrough** mode (`baseline`) and once
with compression **on** (`headroom`) — and compare **resolved rate, tokens, cost,
and turns**. This is the bed for answering: *does a compression change help a
coding agent without breaking it?*

It is deliberately the mirror image of the internal `ramp-swebench` harness:
public tasks, no Ramp infra, laptop-runnable, results land in local files.

## Install

```bash
pip install -e ".[swebench]"     # mini-swe-agent + swebench + litellm + datasets
# also required: a running Docker daemon (x86_64) for the agent env + grader
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY for an openai/ model
```

## Run

```bash
# 5-task smoke, no grading (no Docker needed), on-vs-off:
python -m headroom.evals swebench --slice 0:5 --no-grade

# Full SWE-bench Verified A/B with the official grader:
python -m headroom.evals swebench --subset verified --model anthropic/claude-opus-4-8

# Single arm against your own already-running proxy:
python -m headroom.evals swebench --no-proxy --arms headroom --slice 0:1
```

Outputs land in `swebench-eval-<timestamp>/`:
`results.csv` (per instance), `summary.json`, `report.md`, `report.html`, plus each
arm's `preds.json`, agent trajectories, and `proxy.log`.

## How it works

```
SWE-bench subset → mini-SWE-agent (bash) → litellm → Headroom proxy → provider
                                            api_base ↘  baseline: --no-optimize
                                                         headroom: compression on
   per-instance trajectory ─→ turns/usage        proxy /stats ─→ tokens, cost, savings
   preds.json ─→ official grader ─→ resolved_ids
                     ↓
        results.csv + summary.json + report.{md,html}
```

- **Correctness** comes from the official `swebench.harness.run_evaluation`
  (`resolved_ids`), not a reimplementation.
- **Tokens / cost / savings** come from the proxy's own `/stats` (the savings
  ledger already prices tokens). Both arms run *through* the proxy so one
  accounting path measures both; only the compression toggle differs.
- **Turns** come from the agent trajectory: `model_turns` (assistant messages)
  and `tool_calls` (bash calls).

## Why the four token buckets matter

Token count ≠ cost. Anthropic prompt-cache writes cost ~1.25× base and reads
~0.1× — a **12.5× asymmetry**. A change can lower *total* tokens yet *raise* cost.
The report always surfaces uncached-input / output / cache-read / cache-write
separately for exactly this reason.

## Not yet (roadmap)

- **A/A/B rigor** (Phase 2): a second baseline arm to measure the trajectory noise
  floor, paired stats (mean-diff + 95% CI + MDE), McNemar on resolved, and a
  publication gate. Today's on-vs-off diff is descriptive, not yet significance-tested.
- **Modal backend** (Phase 3) for 500-instance runs; optional LangFuse exporter.

## Zero-spend wiring test

`tests/test_swebench_wiring.py` starts a fake upstream + the real proxy and asserts
routing + `/stats` parsing, with no API calls and no Docker. Run:

```bash
python -m pytest tests/test_swebench_wiring.py -v
```

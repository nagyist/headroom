"""Grade predictions with the OFFICIAL SWE-bench evaluation harness.

We subprocess ``python -m swebench.harness.run_evaluation`` (Docker-based) and read
the run report it writes to CWD (``<model with '/'->'__'>.<run_id>.json``), whose
``resolved_ids`` list is the source of truth for correctness. Requires the optional
``swebench`` dep and a running Docker daemon.

Grading is best-effort: if the harness or Docker is unavailable the run still
reports tokens/cost/turns, with ``resolved`` left unknown.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GradeResult:
    resolved_ids: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    ok: bool = False
    error: str = ""


def _report_path(out_dir: Path, model: str, run_id: str) -> Path | None:
    expected = out_dir / f"{model.replace('/', '__')}.{run_id}.json"
    if expected.exists():
        return expected
    matches = sorted(out_dir.glob(f"*.{run_id}.json"))
    return matches[0] if matches else None


def grade_predictions(
    *,
    preds_path: str,
    dataset_name: str,
    split: str,
    run_id: str,
    out_dir: str,
    max_workers: int = 4,
) -> GradeResult:
    """Run the official grader in ``out_dir`` and parse resolved ids."""
    if not Path(preds_path).exists():
        return GradeResult(error=f"predictions not found: {preds_path}")

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        preds_path,
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    try:
        proc = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, check=False)
    except (OSError, ValueError) as exc:  # e.g. swebench not installed
        return GradeResult(error=f"grader failed to launch: {exc}")

    # Figure out the model name from the first prediction (that's what the harness
    # uses to name the report file).
    model = ""
    try:
        data = json.loads(Path(preds_path).read_text())
        first = next(iter(data.values())) if isinstance(data, dict) else data[0]
        model = str(first.get("model_name_or_path", ""))
    except (json.JSONDecodeError, OSError, StopIteration, IndexError, KeyError):
        pass

    report = _report_path(Path(out_dir), model, run_id)
    if report is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        return GradeResult(error="report not written; grader output tail: " + " | ".join(tail))

    try:
        payload = json.loads(report.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return GradeResult(error=f"could not read report {report}: {exc}")

    resolved = [str(x) for x in (payload.get("resolved_ids") or [])]
    counts = {
        k: payload.get(k)
        for k in (
            "total_instances",
            "submitted_instances",
            "completed_instances",
            "resolved_instances",
            "unresolved_instances",
            "empty_patch_instances",
            "error_instances",
        )
    }
    return GradeResult(resolved_ids=resolved, counts=counts, ok=True)

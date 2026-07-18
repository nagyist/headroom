"""Byte-identical differential tests for registry-resolved built-in dispatch.

The content router now dispatches the SIMPLE built-in strategies (CONFIG, LOG,
SEARCH, TABULAR) through the compressor registry instead of a hardcoded direct
``self._get_*().compress(...)`` call in ``_apply_strategy_to_content``. Each
built-in adapter delegates to the SAME ``_get_*`` getter+method with the SAME
arguments, so registry-resolved dispatch must be byte-identical to the historical
direct dispatch: same compressed content, same token count, same single-entry
``strategy_chain``.

Each FLIPPED strategy has a differential test comparing the router's dispatch
output to the built-in's direct output obtained via its ``_get_*`` getter — i.e.
"registry dispatch == old dispatch". DEFERRED strategies (SMART_CRUSHER, KOMPRESS)
are asserted unchanged: they still route through their bespoke paths (fallback
chain / the ``_try_ml_compressor`` ML boundary), not the registry.

Offline guardrails:
  * No real ML/ONNX/HF inference — the deferred KOMPRESS path is mocked.
  * The flipped strategies shrink their representative content, so no zero-savings
    Kompress fallback fires (that would touch the ML boundary and append KOMPRESS
    to the chain).
  * ``lossless_then_lossy`` and ``relevance_split`` are off so the if/elif branch
    is the terminal path; STAGE 0 (``_lossless_first``) is neutralized so search/
    log folds don't return before the branch. Both are shared, unchanged code.
  * The broad ``content_router``/``compression`` -k selection is NOT exercised
    (it hangs on HF-Hub/ONNX).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def _router() -> ContentRouter:
    """Router whose if/elif branch is the terminal dispatch path.

    ``relevance_split`` off (no LOG/SEARCH relevance split) and
    ``lossless_then_lossy`` off (no lossy layer on top of a strategy result) so a
    successful strategy result returns directly. ``ccr_inject_marker`` off makes
    the compressed output deterministic and marker-free; it is applied identically
    to the direct reference and the dispatch router, so the differential holds
    regardless of its value.
    """
    return ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
        )
    )


def _isolate_branch(monkeypatch: pytest.MonkeyPatch, router: ContentRouter) -> None:
    """Neutralize STAGE 0 so the if/elif branch under test is exercised.

    ``_lossless_first`` runs unconditionally and can fold search/log content,
    returning before the if/elif. It is shared, unchanged code (the flip only
    touches the branch bodies), so forcing it to a no-op isolates what the flip
    actually changed without altering the branch semantics.
    """
    monkeypatch.setattr(router, "_lossless_first", lambda content, strategy: (content, None))


# Representative content per type. The flipped strategies must SHRINK this so the
# fallback-eligible strategies (TABULAR/CONFIG) don't trip the zero-savings
# Kompress fallback (which would append KOMPRESS to the chain).
_SEARCH = "\n".join(f"src/file{i}.py:{i}: def func{i}(): return {i}" for i in range(30))
_LOG = (
    "\n".join(f"2024-01-01 12:00:{i:02d} INFO task {i}" for i in range(30))
    + "\n"
    + "\n".join("identical repeated line" for _ in range(25))
)
# A markdown table the tabular compressor actually shrinks (schema-fold), and a
# repetitive INI the config compressor actually shrinks (block-fold) — so these
# fallback-eligible strategies produce a real token saving and the shared
# zero-savings Kompress fallback does NOT fire (chain stays single-entry).
_TABULAR = "| id | name | status | score |\n|----|------|--------|-------|\n" + "\n".join(
    f"| {i} | row{i} | ok | {i * 3} |" for i in range(60)
)
_CONFIG = "\n".join(
    f"[section_{i}]\nname = svc{i}\ntimeout = 30\nretries = 3\nverbose = false\nregion = us-east-1"
    for i in range(40)
)


# ───────────────────────── flipped (registry dispatch) ────────────────────────


def test_search_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "func", 1.0
    # OLD dispatch reference: same getter + method the branch used before the flip.
    direct = (
        _router()._get_search_compressor().compress(_SEARCH, context=context, bias=bias).compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _SEARCH, CompressionStrategy.SEARCH, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.SEARCH.value]


def test_log_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    bias = 1.0
    direct = _router()._get_log_compressor().compress(_LOG, bias=bias).compressed
    out, tokens, chain = router._apply_strategy_to_content(
        _LOG, CompressionStrategy.LOG, "", bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.LOG.value]


def test_tabular_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = (
        _router()
        ._get_tabular_compressor()
        .compress(_TABULAR, context=context, bias=bias)
        .compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _TABULAR, CompressionStrategy.TABULAR, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    # Fallback-eligible, but a real shrink means no zero-savings Kompress fallback.
    assert chain == [CompressionStrategy.TABULAR.value]
    assert len(out) < len(_TABULAR)


def test_config_router_dispatch_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = (
        _router()._get_config_compressor().compress(_CONFIG, context=context, bias=bias).compressed
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _CONFIG, CompressionStrategy.CONFIG, context, bias=bias
    )
    assert out == direct
    # CONFIG's historical metric is len(text.split()), NOT _estimate_tokens; the
    # flip must preserve that exact metric.
    assert tokens == len(direct.split())
    assert chain == [CompressionStrategy.CONFIG.value]
    assert len(out) < len(_CONFIG)


# ─────────────────────────── deferred (unchanged) ────────────────────────────


def test_smart_crusher_deferred_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER is DEFERRED (its branch feeds a Kompress→Log fallback chain in
    # the shared post-strategy block), so it still dispatches via the direct
    # crusher, not the registry. A JSON array shrinks, so the chain stays single.
    router = _router()
    _isolate_branch(monkeypatch, router)
    content = json.dumps(
        [{"id": i, "status": "ok", "level": "INFO", "value": i * 2} for i in range(40)]
    )
    direct = _router()._get_smart_crusher().crush(content, query="q", bias=1.0).compressed
    out, _tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.SMART_CRUSHER, "q", bias=1.0
    )
    assert out == direct
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]


def test_kompress_deferred_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # KOMPRESS is DEFERRED (it is the ML boundary — dispatched through
    # _try_ml_compressor, not a built-in adapter). Mock the underlying model so no
    # real ONNX/HF inference runs, and assert the router still routes through
    # _try_ml_compressor rather than the registry.
    router = _router()
    _isolate_branch(monkeypatch, router)
    fake = SimpleNamespace(
        is_ready=lambda: True,
        ensure_background_load=lambda: None,
        compress=lambda text, **kwargs: SimpleNamespace(
            compressed="KOMPRESSED::" + text, compressed_tokens=7
        ),
    )
    monkeypatch.setattr(router, "_get_kompress", lambda: fake)
    content = "some plain text that the ML model would compress. " * 4
    out, _tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "", bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.KOMPRESS.value]
    # Still the bespoke ML path (unchanged), not a registry round-trip.
    assert out == router._try_ml_compressor(content, "", None)[0]

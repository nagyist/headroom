"""Byte-lossless search-fold for EXCLUDED navigation-tool output (#6).

Excluded tools (Read/Grep/Glob/Write/Edit) are protected from *lossy*
compression for accuracy. This feature still applies the reversible
search-heading fold to grep-shaped output of excluded tools: real savings
(~36% on code-grep) with zero information loss — ``search_unheading``
reproduces the original byte-for-byte. Read/source/glob are left untouched.
Off by default (``compact_excluded_search``).
"""

from __future__ import annotations

import pytest

from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.lossless_compaction import search_unheading

# grep output grouped per file (consecutive same-path rows), like real ripgrep.
GREP = "".join(
    f"src/module_{f}.py:{ln * 3}:matched occurrence with some real content here\n"
    for f in range(6)
    for ln in range(15)
)
# a Read of source code (excluded 'read' tool) — no path:line: rows.
CODE = "def foo(x):\n    return x + 1\n\nclass Bar:\n    value = 42\n" * 30
# a Glob file-path list — not search-shaped.
GLOB = "\n".join(f"src/module_{i}.py" for i in range(60)) + "\n"


@pytest.fixture
def tokenizer():
    provider = OpenAIProvider()
    return Tokenizer(provider.get_token_counter("gpt-4o"), "gpt-4o")


def _fold(content: str, on: bool):
    router = ContentRouter(ContentRouterConfig(compact_excluded_search=on))
    return router._lossless_fold_excluded(content)


def test_helper_folds_grep_losslessly():
    folded = _fold(GREP, True)
    assert folded is not None
    assert len(folded) < len(GREP)
    assert search_unheading(folded) == GREP  # byte-exact → zero accuracy loss


def test_helper_noop_on_source_and_glob():
    assert _fold(CODE, True) is None  # Read/source is not search-shaped
    assert _fold(GLOB, True) is None  # glob path-list is not search-shaped


def test_helper_off_by_default():
    assert _fold(GREP, False) is None


def _run_tool(content: str, tool: str, on: bool, tokenizer):
    router = ContentRouter(ContentRouterConfig(compact_excluded_search=on))
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": tool, "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
    ]
    result = router.apply(messages, tokenizer, compress_user_messages=True)
    return result.messages[1]["content"], result.transforms_applied


def test_pipeline_default_is_verbatim(tokenizer):
    out, transforms = _run_tool(GREP, "grep", False, tokenizer)
    assert out == GREP
    assert "router:excluded:lossless_search" not in transforms


def test_pipeline_folds_excluded_grep_losslessly(tokenizer):
    out, transforms = _run_tool(GREP, "grep", True, tokenizer)
    assert len(out) < len(GREP)
    assert "router:excluded:lossless_search" in transforms
    assert search_unheading(out) == GREP  # recover the original exactly


def test_pipeline_leaves_excluded_read_untouched(tokenizer):
    # A Read of source code must stay byte-identical (Edit needs exact bytes).
    out, _ = _run_tool(CODE, "read", True, tokenizer)
    assert out == CODE

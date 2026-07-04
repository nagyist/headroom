"""Cross-turn dedup: cache-safety (prefix-monotonicity) + accuracy (info-preserving)."""
import re

from headroom.transforms.cross_turn_dedup import (
    DedupBlock,
    dedup_blocks,
    is_prefix_monotonic,
)

_PTR_RE = re.compile(r"identical to output shown earlier \(turn (\d+), lines (\d+)-(\d+)\)")


def _blk(text, turn, protected=False):
    return DedupBlock(text=text, turn=turn, protected=protected)


def _code(prefix, n):
    # A realistic, non-trivial multi-line source span.
    return "\n".join(
        f"{prefix}    result_{i} = compute_overdraft(business_id={i}, amount={i * 100})"
        for i in range(n)
    )


def _reconstruct(orig_blocks, out_blocks):
    """Replace each pointer with the referenced turn's original lines and assert
    it reproduces the original block — proves references are faithful & in-context."""
    by_turn = {b.turn: b.text.split("\n") for b in orig_blocks}
    for orig, out in zip(orig_blocks, out_blocks):
        if orig.protected:
            assert out.text == orig.text
            continue
        rebuilt = []
        for line in out.text.split("\n"):
            m = _PTR_RE.search(line)
            if m and line.startswith("[headroom:"):
                t, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
                assert t < orig.turn, "reference must point to an EARLIER turn"
                rebuilt.extend(by_turn[t][a : b + 1])
            else:
                rebuilt.append(line)
        assert "\n".join(rebuilt) == orig.text, f"turn {orig.turn} not faithfully reconstructable"


def test_verbatim_reread_is_folded_keep_earliest():
    span = _code("", 8)
    blocks = [_blk(f"cat merge.py\n{span}\ntail", 1), _blk(f"sed run\n{span}\nmore", 5)]
    out, stats = dedup_blocks(blocks)
    assert out[0].text == blocks[0].text  # earliest untouched
    assert "[headroom:" in out[1].text  # later occurrence folded
    assert stats["spans_folded"] == 1
    _reconstruct(blocks, out)


def test_cache_safety_prefix_monotonic():
    span = _code("x", 10)
    blocks = [
        _blk("intro line one\nintro line two\n" + span, 1),
        _blk("unrelated diff output\n@@ -1 +1 @@\n-a\n+b", 2),
        _blk("here again:\n" + span, 3),
        _blk("and once more\n" + span + "\ntrailer", 4),
    ]
    assert is_prefix_monotonic(blocks) is True


def test_below_min_lines_not_folded():
    span = _code("", 3)  # < 6 lines
    blocks = [_blk(span, 1), _blk(span, 2)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0
    assert out[1].text == blocks[1].text


def test_trivial_repeated_lines_not_folded():
    junk = "\n".join(["}"] * 20)  # trivial lines only
    blocks = [_blk(junk, 1), _blk(junk, 2)]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 0


def test_deterministic():
    span = _code("z", 9)
    blocks = [_blk(span, 1), _blk("mid\n" + span, 2), _blk(span, 3)]
    a, _ = dedup_blocks(blocks)
    b, _ = dedup_blocks(blocks)
    assert [x.text for x in a] == [x.text for x in b]


def test_protected_block_not_rewritten_but_is_reference_target():
    span = _code("", 8)
    blocks = [
        _blk(span, 1, protected=True),  # cache_control block — never rewritten
        _blk("later:\n" + span, 2),  # should still fold against the protected one
    ]
    out, stats = dedup_blocks(blocks)
    assert out[0].text == blocks[0].text
    assert "[headroom:" in out[1].text
    _reconstruct(blocks, out)


def test_info_preserving_reconstruction_multiref():
    s1 = _code("a", 7)
    s2 = _code("b", 8)
    blocks = [
        _blk("h1\n" + s1, 1),
        _blk("h2\n" + s2, 2),
        _blk("mix\n" + s1 + "\n---\n" + s2, 3),  # two folds in one block
    ]
    out, stats = dedup_blocks(blocks)
    assert stats["spans_folded"] == 2
    _reconstruct(blocks, out)

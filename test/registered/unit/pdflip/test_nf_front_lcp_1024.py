"""FRONT-LCP-1024 -- the arrival's SpanLRU scan grew with the boot, not the prompt.

MEASURED, NF boot ``..._cebd0da264_1023_234711`` (front log, 376 arrivals):
the gap between ``PDFLIP SESSION`` (arrival) and the start of the exact count
(X-EXACT-PRICE ``wait_ms`` subtracted from its timestamp) grew from 12-45 ms on
the first requests to 220-359 ms in the last third, while the front
tokenizer's own ``count_ms`` stayed 8-90 ms (segment-incremental since 26.09.).
SESSION -> TITO p50 per sixth of the boot: 56, 94, 133, 193, 240, 261 ms at a
flat prompt size (p50 54k-75k tokens). The step between the two is
``price_remainder`` -> ``SpanLRU.uncached_tokens``: ``common_prefix_len``
against each of up to 512 served texts (100-600k chars), whose binary search
copied ``a[:mid]``/``b[:mid]`` on every step.

DANGER DIRECTION (named): a wrong prefix length at a segment edge changes the
credit of an entry, so a request could be priced kurz/SHORT while D must
prefill more than X (SOLL R7 'kann D einen Request unter X nicht nehmen, ist
das ein Defekt') or LONG for nothing. Therefore the new function must give the
SAME integer as the old one for every pair -- pinned at the gallop edges, for
non-ASCII and astral text, both argument orders, and through SpanLRU itself;
the mutants below each break one edge and must be caught by the edge check.
"""

import random
import time

import pytest

from flliper.srt.pdflip import front as F
from flliper.srt.pdflip.front import SpanLRU, common_prefix_len


def _reference(a: str, b: str) -> int:
    """The pre-1024 implementation, verbatim."""
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


_ALPHABET = list("abcdefghij kl<>|_\n") + ["ä", "—", "→", "\U0001F600", "Ü"]


def _text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(n))


def _edges():
    g = F._LCP_GALLOP0
    out = {0, 1, 2, g - 1, g, g + 1}
    acc = 0
    step = g
    for _ in range(5):  # the cumulative gallop boundaries: g, 3g, 7g, 15g, 31g
        acc += step
        out.update({acc - 1, acc, acc + 1})
        step *= 2
    return sorted(out)


def _edge_pairs(rng: random.Random):
    """Pairs that diverge exactly at, or end exactly at, every gallop edge."""
    base = _text(rng, 31 * F._LCP_GALLOP0 + 50)
    for k in _edges():
        a = base[: k + 20]
        # diverge AT k (the char at k differs)
        repl = "Z" if a[k:k + 1] != "Z" else "Y"
        yield a, a[:k] + repl + a[k + 1:]
        # one string ends at k (proper prefix)
        yield a, a[:k]
        # equal strings of length k
        yield a[:k], a[:k]
        # diverge at k with a wide char (string kinds differ from the segment on)
        yield a, a[:k] + "\U0001F600" + a[k + 1:]


def test_edges_equal_reference():
    rng = random.Random(1024)
    for a, b in _edge_pairs(rng):
        assert common_prefix_len(a, b) == _reference(a, b), (len(a), len(b))
        assert common_prefix_len(b, a) == _reference(b, a), (len(a), len(b))


def test_random_pairs_equal_reference():
    rng = random.Random(7)
    pool = _text(rng, 60000)
    for _ in range(600):
        st = rng.randint(0, 20000)
        s = pool[st:st + rng.randint(0, 40000)]
        k = rng.randint(0, len(s))
        t = s[:k] + rng.choice(["", "x", "—", "\U0001F600", "ä"]) + _text(rng, rng.randint(0, 300))
        assert common_prefix_len(s, t) == _reference(s, t)
        assert common_prefix_len(t, s) == _reference(t, s)
    for a, b in [("", ""), ("a", ""), ("", "a"), ("abc", "abd"), ("y", "x"), ("ä", "a")]:
        assert common_prefix_len(a, b) == _reference(a, b)


def _agent_lru(rng: random.Random, sessions: int = 6, turns: int = 30):
    """A SpanLRU shaped like the boot's: sessions sharing a tools/system head,
    each turn a prefix extension of the previous one."""
    head = _text(rng, 30000)
    sess = [head + _text(rng, 5000) for _ in range(sessions)]
    lru = SpanLRU(agent_span=True)
    texts = []
    for t in range(turns):
        for s in range(sessions):
            sess[s] += _text(rng, rng.randint(800, 3000))
            texts.append(sess[s])
            ct = int(len(sess[s]) / 3.1) - rng.randint(0, 200)
            lru.record_presence(sess[s], ct, prompt_tokens=int(len(sess[s]) / 3.1), held_epoch=t % 3)
            if rng.random() < 0.4:
                # #59: a positive resumable depth caps the entry -> known_prefix_depth reads it
                lru.record_presence(sess[s] + "x", ct, prompt_tokens=ct + 10,
                                    resumable_depth=max(1, ct - rng.randint(0, 5000)))
    return lru, sess, head


def test_spanlru_prices_unchanged(monkeypatch):
    rng = random.Random(42)
    lru, sess, head = _agent_lru(rng)
    probes = [s + _text(rng, 1500) for s in sess]
    probes += [head[:12345] + "Q" + _text(rng, 999), _text(rng, 5000), sess[0][:F._LCP_GALLOP0 * 3]]
    assert lru.depth_caps, "the pin must reach known_prefix_depth's scan (review 1024 finding 1)"
    legacy = SpanLRU(agent_span=False)
    for k in list(lru.entries)[:60]:
        etext, ct, _pt, _h = lru.entries[k]
        legacy.record_presence(etext, ct)

    def prices():
        return ([(lru.uncached_tokens(p, int(len(p) / 3) + 1, e), lru.last_src, lru.known_prefix_depth(p))
                 for p in probes for e in (None, 0, 1, 2)]
                + [(legacy.span_tokens(p), legacy.uncached_tokens(p, int(len(p) / 3) + 1)) for p in probes])

    new = prices()
    assert any(d > 0 for (_u, _s, d) in new[: 4 * len(probes)]), "known_prefix_depth never credited"
    monkeypatch.setattr(F, "common_prefix_len", _reference)
    old = prices()
    assert new == old


def test_scan_cost_does_not_grow_with_log_n():
    """The point of the fix: the per-arrival scan at the boot's late LRU size.
    Generous bound (the desk measured ~8x: 380 entries x 171k chars, 61 -> 8 ms)."""
    rng = random.Random(3)
    head = "".join(rng.choice("abcdefgh ä—") for _ in range(60000))
    sess = [head + "".join(rng.choice("abcdefgh ä—") for _ in range(20000)) for _ in range(8)]
    entries = []
    for _ in range(30):
        for s in range(8):
            sess[s] += "".join(rng.choice("abcdefgh ä—") for _ in range(4000))
            entries.append(sess[s])
    probe = sess[3] + "tail"

    def run(fn):
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            for e in entries:
                fn(e, probe)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best

    t_old, t_new = run(_reference), run(common_prefix_len)
    assert t_new < 0.5 * t_old, (t_old, t_new)


# --- mutants of the edge logic: each must be caught by the edge check -------

_SRC = '''
def m(a, b):
    n = min(len(a), len(b))
    if n == 0 or a[0] != b[0]:
        return 0
    lo, step = 0, max(1, int(G))
    while True:
        hi = min(n, lo + step)
        if a[lo:hi] == b[lo:hi]:
            lo = hi
            if lo == n:
                return n
            step *= 2
        else:
            break
    hi -= 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo
'''

MUTANTS = {
    "M1_bracket_cut_one_too_far": ("    hi -= 1\n", "    hi -= 2\n"),
    "M2_mismatch_returns_segment_start": ("        else:\n            break\n", "        else:\n            return lo\n"),
    "M3_bsearch_wrong_base": ("        if a[lo:mid] == b[lo:mid]:", "        if a[:mid] == b[lo:mid]:"),
    "M4_first_char_skipped_wrong": ("    if n == 0 or a[0] != b[0]:\n        return 0\n", "    if n == 0:\n        return 0\n    if a[0] != b[0]:\n        return 1\n"),
    "M5_full_match_off_by_one": ("            if lo == n:\n                return n\n", "            if lo == n:\n                return n - 1\n"),
}


def _build(src):
    ns = {"G": F._LCP_GALLOP0}
    exec(src, ns)  # noqa: S102 -- test-local source
    return ns["m"]


def _edge_check(fn) -> bool:
    rng = random.Random(1024)
    for a, b in _edge_pairs(rng):
        if fn(a, b) != _reference(a, b) or fn(b, a) != _reference(b, a):
            return False
    return True


def test_unmutated_copy_passes_the_edge_check():
    import inspect
    live = inspect.getsource(F.common_prefix_len)
    # the copy above IS the live body (minus docstring/comments): keep them in step
    for line in ("hi -= 1", "if a[lo:mid] == b[lo:mid]:", "if a[lo:hi] == b[lo:hi]:", "if n == 0 or a[0] != b[0]:", "max(1, int(_LCP_GALLOP0))"):
        assert line in live
    assert _edge_check(_build(_SRC))


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_killed(name):
    old, new = MUTANTS[name]
    assert old in _SRC, name
    assert not _edge_check(_build(_SRC.replace(old, new, 1))), f"{name} survived the edge check"

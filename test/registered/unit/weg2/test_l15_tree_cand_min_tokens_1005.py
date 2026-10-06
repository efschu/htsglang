# SPDX-License-Identifier: Apache-2.0
"""L15-TREE-CAND-MIN-TOKENS (desk 2025, kurz4 image l15f = A + B + C on, boot ..._41aba62120_1005_235733).

Specimen (D log, TP0): ``no_kv_host`` 0, ``anc_miss`` 0, ``tip_miss`` 0, ``no_mamba_host`` 4 of 81 -- the held tip
now IS offered (s0's k=1 turn weg2-14-24: ``LOAD-DEVICE tokens=13691 matched=0`` = only the 13.7k delta
loaded, 19712 tokens consumed from the held tip). But from the 4th sleep on ``local=10-12 agreed=8``: the vote
is capped at SGLANG_WEG2_L15_TREE_CAND_N (8) most recent tips, and what fills it is junk: the 3999-token warm-up tip,
two old 769-token background tips and 3-4 fresh background tips per cycle (L15-L2-SHADOW-ADOPT lists the 8 held tips
per sleep). Session tips (19.7k) held per sleep: 2, 2, 2, 1, 0, 0 -- s1's tip was cut at 00:02:28, s2's at 00:04:02,
s3's never held; s1/s2/s3 k=1 turns loaded matched=57/44/44.

Fix (default 0 = off): ``SGLANG_WEG2_L15_TREE_CAND_MIN_TOKENS`` -- tips with fewer chain tokens are not offered to the
vote (``local_candidates(min_n=...)``, fed from ``build``). Chain length is the same on every rank and the env is the
group's: every rank filters alike, no collective is added.

Hermetic, no CUDA. Run:
  cd /spinning/wt-27b-l15-econsume3-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/weg2/test_l15_tree_cand_min_tokens_1005.py
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402

from sglang.srt.weg2 import l15_tree_cand as tc  # noqa: E402

ROOT_SRT = pathlib.Path(tc.__file__).resolve().parents[1]
ENVIRON = (ROOT_SRT / "environ.py").read_text()
KEY = "SGLANG_WEG2_L15_TREE_CAND_MIN_TOKENS"


class _Root:
    pass


def _tip(root, seed, n, la):
    """One device tip: a single node of ``n`` tokens under the root, last used at ``la``."""
    return SimpleNamespace(parent=root, last_access_time=la, id=seed,
                           key=SimpleNamespace(token_ids=list(range(seed * 100000, seed * 100000 + n)), is_bigram=False))


def _tree(sizes_by_age):
    """sizes_by_age: [(n_tokens, last_access)] -> (tree, tips). Older = smaller last_access."""
    root = _Root()
    tips = [_tip(root, i + 1, n, la) for i, (n, la) in enumerate(sizes_by_age)]
    return SimpleNamespace(root_node=root), tips


def _kurz4_shape():
    """4 session tips (19.7k, idle: old) + 8 background/warm tips (769 / 3999, fresher) = local 12 > cap 8."""
    return [(19713, 10.0 + i) for i in range(4)] + [(769, 100.0 + i) for i in range(7)] + [(3999, 120.0)]


def _vote(tree, tips, env, ranks=1):
    lines = []
    with patch.object(tc, "tips_of", lambda t: list(tips)):
        out = tc.build(tree, lambda local: [list(local)] * ranks, 0, env=env, log=lines.append)
    return out, lines


def _sizes(reqs):
    return sorted(c_n for c_n in (int(r.l15_tree_n) if hasattr(r, "l15_tree_n") else -1 for r in reqs))


def _agreed_sizes(lines):
    line = [x for x in lines if x.startswith("L15-TREE-CAND local=")][0]
    rids = line.split("rids=")[1]
    return [int(x.rsplit("(", 1)[1].rstrip(")")) for x in rids.split(",") if x]


def test_red_without_a_floor_the_cap_is_filled_with_background_tips():
    """The specimen: 12 tips, cap 8, the 19.7k session tips are the oldest -> none agreed."""
    tree, tips = _tree(_kurz4_shape())
    out, lines = _vote(tree, tips, env={})
    assert "local=12 agreed=8" in lines[0]
    assert all(n < 10000 for n in _agreed_sizes(lines))
    assert len(out) == 8


def test_green_with_a_floor_the_session_tips_are_agreed():
    tree, tips = _tree(_kurz4_shape())
    out, lines = _vote(tree, tips, env={KEY: "4096"})
    assert "local=4 agreed=4" in lines[0]
    assert _agreed_sizes(lines) == [19713] * 4
    assert len(out) == 4
    floor = [x for x in lines if "floor min_tokens=4096" in x]
    assert floor and "offered=4 agreed=4" in floor[0]


def test_no_floor_line_without_the_switch_and_two_ranks_agree():
    tree, tips = _tree(_kurz4_shape())
    _out, lines = _vote(tree, tips, env={})
    assert not [x for x in lines if "floor" in x]
    # two ranks with the same tree and the same env filter alike: the agreed list is the same 4
    out2, lines2 = _vote(tree, tips, env={KEY: "4096"}, ranks=2)
    assert "local=4 agreed=4" in lines2[0] and len(out2) == 4


@pytest.mark.parametrize("val,want", [(None, 0), ("", 0), ("0", 0), ("4096", 4096), (" 4096 ", 4096),
                                      ("abc", 0), ("-5", 0), ("1.5", 0)])
def test_min_tokens_parsing(val, want):
    env = {} if val is None else {KEY: val}
    assert tc.min_tokens(env) == want


def test_env_declared_default_off():
    assert "%s = EnvInt(0)" % KEY in ENVIRON


@pytest.mark.parametrize("lazy", [True, False])
def test_floor_boundary_both_walks(lazy):
    tree, tips = _tree([(4095, 1.0), (4096, 2.0), (4097, 3.0)])
    with patch.object(tc, "tips_of", lambda t: list(tips)):
        got = tc.local_candidates(tree, None, lazy_tokens=lazy, min_n=4096)
        none = tc.local_candidates(tree, None, lazy_tokens=lazy)
    assert sorted(c.n_tokens for c in got) == [4096, 4097]
    assert len(none) == 3  # min_n default 0: unchanged


@pytest.mark.parametrize("lazy", [True, False])
def test_floor_applies_before_the_limit(lazy):
    tree, tips = _tree([(19713, 1.0), (769, 5.0), (769, 6.0)])
    with patch.object(tc, "tips_of", lambda t: list(tips)):
        got = tc.local_candidates(tree, 1, lazy_tokens=lazy, min_n=4096)
    assert [c.n_tokens for c in got] == [19713]  # not [769]: the limit counts only offered tips


def _exec(fn, old, new):
    src = inspect.getsource(fn)
    assert old in src, old
    ns = dict(vars(tc))
    exec(src.replace(old, new), ns)
    return ns[fn.__name__]


@pytest.mark.parametrize("old,new", [
    ("if min_n > 0 and got[1] < min_n:", "if False:"),
    ("if min_n > 0 and len(toks) < min_n:", "if False:"),
    ("if min_n > 0 and got[1] < min_n:", "if min_n > 0 and got[1] <= min_n:"),
])
def test_local_candidates_mutants_die(old, new):
    mut = _exec(tc.local_candidates, old, new)
    tree, tips = _tree([(4095, 1.0), (4096, 2.0), (4097, 3.0)])
    bad = False
    for lazy in (True, False):
        with patch.object(tc, "tips_of", lambda t: list(tips)), patch.dict(mut.__globals__, {"tips_of": lambda t: list(tips)}):
            got = mut(tree, None, lazy_tokens=lazy, min_n=4096)
        bad = bad or sorted(c.n_tokens for c in got) != [4096, 4097]
    assert bad, (old, new)


def test_min_tokens_mutant_junk_raises_dies():
    mut = _exec(tc.min_tokens, "except ValueError:\n        return 0", "except KeyError:\n        return 0")
    with pytest.raises(ValueError):
        mut({KEY: "abc"})


def test_build_mutants_die():
    for old, new in (("min_n=min_tokens(env))", "min_n=0)"),):
        mut = _exec(tc.build, old, new)
        tree, tips = _tree(_kurz4_shape())
        lines = []
        with patch.object(tc, "tips_of", lambda t: list(tips)), patch.dict(mut.__globals__, {"tips_of": lambda t: list(tips)}):
            out = mut(tree, lambda local: [list(local)], 0, env={KEY: "4096"}, log=lines.append)
        assert len(out) == 8, "mutant: the floor is not fed from build -> the cap is full of background tips again"

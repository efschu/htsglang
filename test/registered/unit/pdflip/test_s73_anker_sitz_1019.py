"""S73 (NF, 19.10., rc16 bf9ca46f45 ..._1019_121629): ANKER-VERFEHLT-P.

Three pieces, each pinned to the metal numbers of the rc16 boot:

1. P INTAKE READ MIN (fix). pdflip-194-759: the store held 43392 (D's
   PARK-END-ANCHOR 14:23:58), P PP0 refused the 128-token read at intake
   (``#915 PREFETCH REFUSED reason=too_short need=128 threshold=256``) and P
   prefilled [43264, 43447) = 183 tokens instead of 55. rc16 P PP0: 23 such
   refusals, 3456 tokens. A group-P store read is now read whatever its length
   (``handback_claim.p_intake_min_tokens``), folded into ``_prefetch_kvcache``
   with the other min-1 laws.

2. PARK-DEMOTE-INCOMPLETE (instrument). A kept KV span with absent pages and
   nothing to copy printed no line (190-747, 30528 kept, P read 25280): now
   named once per (rid, absent, missing, busy) change.

3. PUBLISH-SWEEP refused_by (instrument; the real sweep in
   mem_cache/test_s73_sweep_refused_by_1019.py). The sweep's BACKUP-REFUSED lines are
   sampled per reason (rc16 TP1 14:13:15: 39 refusals between n=2225 and
   n=2265, none printed), so a worker's persistent ``refused=6`` had no reason
   in the log. The sweep line now carries its own refusal census.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import inspect
import logging
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import handback_claim as hc
from flliper.srt.pdflip import handoff_pending as hp
from flliper.srt.pdflip import park_demote

# ---------------------------------------------------------------- 1. P intake


@pytest.mark.parametrize(
    "env,want",
    [
        ({"FLLIPER_PDFLIP_GROUP": "P"}, 1),                       # default ON
        ({"FLLIPER_PDFLIP_GROUP": "p"}, 1),
        ({"FLLIPER_PDFLIP_GROUP": "P", hc.P_INTAKE_READ_MIN_ENV: "1"}, 1),
        ({"FLLIPER_PDFLIP_GROUP": "P", hc.P_INTAKE_READ_MIN_ENV: "0"}, None),
        ({"FLLIPER_PDFLIP_GROUP": "P", hc.P_INTAKE_READ_MIN_ENV: "off"}, None),
        ({"FLLIPER_PDFLIP_GROUP": "D"}, None),                    # D keeps #915
        ({}, None),                                               # no flip group
    ],
)
def test_p_intake_min_tokens(env, want):
    assert hc.p_intake_min_tokens(env=env) == want


def _gate(prefetch_length: int, min_tokens):
    """The #915 too_short term of ``UnifiedRadixCache.prefetch_from_storage``
    (``_min_len = prefetch_threshold if min_tokens is None else max(1, ..)``)."""
    threshold = 256
    min_len = threshold if min_tokens is None else max(1, int(min_tokens))
    return "too_short" if prefetch_length < min_len else None


def test_194_759_metal_numbers():
    """rc16 P PP0 14:24:04: need=128 refused; with the S73 minimum it is read."""
    store, start, prompt = 43392, 43264, 43447
    need = store - start
    assert need == 128
    assert _gate(need, None) == "too_short"
    assert _gate(need, hc.p_intake_min_tokens(env={"FLLIPER_PDFLIP_GROUP": "P"})) is None
    assert prompt - start == 183 and prompt - store == 55


def test_gate_source_still_the_one_the_fold_feeds():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.prefetch_from_storage)
    assert "_min_len = self.prefetch_threshold if min_tokens is None else max(1, int(min_tokens))" in src
    assert 'reason = "too_short"' in src


def test_prefetch_wiring_folds_p_intake_min():
    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._prefetch_kvcache)
    assert "p_intake_min_tokens()" in src
    # folded after the other laws, before the kwargs are built, by min()
    assert src.index("handback_min_tokens(") < src.index("p_intake_min_tokens()") < src.index(
        '_tail_kw = {"min_tokens"')
    seg = src[src.index("p_intake_min_tokens()"):src.index('_tail_kw = {"min_tokens"')]
    assert "min(int(_tail_min), int(_pi_min))" in seg
    # both prefetch_from_storage calls take the folded kwargs
    assert src.count("**_tail_kw") >= 2


# ---------------------------------------------------------------- 2. PARK-DEMOTE-INCOMPLETE


class _Backend:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def arena_copy_to_disk(self, arena, stems):
        self.calls += 1
        return dict(self.answers.pop(0))


def _pool(role):
    p = types.SimpleNamespace(arena=object())
    setattr(p, hp.ROLE_ATTR, role)
    return p


def _incomplete(caplog):
    return [r.getMessage() for r in caplog.records if "PARK-DEMOTE-INCOMPLETE" in r.getMessage()]


def test_kv_span_busy_without_copy_is_named_once(monkeypatch, caplog):
    """rc16 174-648 second pass shape: 512 pages, 478 in L3, 34 claimed and
    never completed (busy), nothing left to copy."""
    stems = ["s%d" % i for i in range(512)]
    monkeypatch.setattr(hp, "rid_spans", lambda pool: [(hp.ROLE_PARK, "pdflip-174-648", stems)])
    park_demote._INCOMPLETE_SEEN.clear()
    pool = _pool("kv")
    ans = {"on_disk": 478, "absent": 34, "missing": 0, "busy": 34}
    be = _Backend([ans, ans, {"on_disk": 478, "absent": 34, "missing": 34, "busy": 0},
                   {"on_disk": 512}])
    state = {}
    caplog.set_level(logging.WARNING, logger=park_demote.__name__)
    park_demote.demote_once(be, [pool], state=state, batch=1024)
    lines = _incomplete(caplog)
    assert len(lines) == 1
    assert "rid=pdflip-174-648" in lines[0] and "pages=512 l3_pages=478 absent=34 (missing=0 busy=34)" in lines[0]
    assert not any(state.values())          # still open: the next pass asks again
    park_demote.demote_once(be, [pool], state=state, batch=1024)
    assert len(_incomplete(caplog)) == 1    # same signature: no second line
    park_demote.demote_once(be, [pool], state=state, batch=1024)
    lines = _incomplete(caplog)
    assert len(lines) == 2 and "(missing=34 busy=0)" in lines[1]   # the slots went: named again
    park_demote.demote_once(be, [pool], state=state, batch=1024)
    assert any(state.values()) and len(_incomplete(caplog)) == 2   # complete: done, no line
    # the span left the order: its memo goes with it
    monkeypatch.setattr(hp, "rid_spans", lambda pool: [])
    park_demote.demote_once(be, [pool], state=state, batch=1024)
    assert not park_demote._INCOMPLETE_SEEN


def test_written_pass_keeps_its_own_line(monkeypatch, caplog):
    """A pass that copied something keeps the #248 PARK-DEMOTE line (it names
    absent/busy already) and adds no INCOMPLETE line."""
    stems = ["s%d" % i for i in range(4)]
    monkeypatch.setattr(hp, "rid_spans", lambda pool: [(hp.ROLE_PARK, "r", stems)])
    park_demote._INCOMPLETE_SEEN.clear()
    be = _Backend([{"written": 1, "on_disk": 1, "absent": 2, "busy": 2}])
    caplog.set_level(logging.INFO, logger=park_demote.__name__)
    recs = park_demote.demote_once(be, [_pool("kv")], state={}, batch=1024)
    assert recs and recs[0]["busy"] == 2
    assert not _incomplete(caplog)
    assert any("#248 PARK-DEMOTE rid=r" in r.getMessage() for r in caplog.records)


def test_anchor_pool_missing_candidate_is_not_named(monkeypatch, caplog):
    """The anchor pool's second candidate has no slot by design (0929)."""
    monkeypatch.setattr(hp, "rid_spans", lambda pool: [(hp.ROLE_PARK, "r", ["a", "b"])])
    park_demote._INCOMPLETE_SEEN.clear()
    be = _Backend([{"on_disk": 0, "absent": 2, "missing": 1, "busy": 1}])
    caplog.set_level(logging.WARNING, logger=park_demote.__name__)
    park_demote.demote_once(be, [_pool("anchor")], state={}, batch=1024)
    assert not _incomplete(caplog)


# ---------------------------------------------------------------- 3. sweep refused_by


def test_refusal_outside_a_sweep_is_not_collected():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    ns = types.SimpleNamespace(root_node=None, _s73_sweep_why=None,
                               pdflip_node_depth=lambda n: 0, _mamba_pins_held=lambda: 0,
                               _mamba_pin_budget=0)
    node = types.SimpleNamespace(id=1, key=[1], parent=None, evicted=False, backuped=False)
    UnifiedRadixCache._1421_refused(ns, "mamba_pin", node)
    assert ns._s73_sweep_why is None
    ns._s73_sweep_why = {}
    UnifiedRadixCache._1421_refused(ns, "mamba_pin", node)
    UnifiedRadixCache._1421_refused(ns, "write_none:x", node)
    assert ns._s73_sweep_why == {"mamba_pin": 1, "write_none": 1}

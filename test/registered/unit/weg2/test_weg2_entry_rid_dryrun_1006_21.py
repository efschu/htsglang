# SPDX-License-Identifier: Apache-2.0
"""NF-NEXT-1006-21: two observation markers for the Held-credit question, NO behaviour change.

Basis 369f31e2e2 (cand4c final), NF k2 05.10. 20:29-20:30Z (nf-next-1006-19 report):

  weg2-16-80 (9140 tokens) finished on D with ct=0 (held epoch, depth 9216 -> K2
  clamp 9088); weg2-16-87 (9948 tokens) was priced ``credit=9088 src=d_served_epoch``
  off that very entry and D read nothing (``X-EXACT-ERR ... d_uncached=9948
  err=-9088 match=1``): a whole loss. The price line did not say WHICH entry the
  credit came from.

(1) ``X-EXACT-PRICE ... entry_rid=<rid of the entry> entry_src=<kind>`` (additive,
    at the end of the line; ``TokenSpans.pending`` only REPORTS the winner in the
    side field ``last_winner``, its return value and signature are unchanged);
(2) ``PRESENCE-CONTRADICTED-DRYRUN`` in ``_x_exact_record``: ``direct``, ``match=1``,
    credited >= 1024 and ct <= one page -- logs what a retract would be about and
    writes nothing.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import os
import re

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import Count, TokenSpans  # noqa: E402

X = 4855


@pytest.fixture(autouse=True)
def _no_early_flip(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_EARLY_FLIP", "0")


def _ids(n, seed=0):
    rng = np.random.default_rng(seed)
    out = rng.integers(1, 240000, size=n, dtype=np.int64).astype(np.int32)
    out[0] = 248045
    return out


class _Tok:
    state = "ready"
    why = ""

    def __init__(self):
        self.ids = None
        self.m = {}
        self.executor = None

    def ids_for(self, text):
        return self.m.get(text)

    def remember(self, text, ids):
        self.m[text] = ids

    def count(self, path, payload):
        return Count(n=int(self.ids.size), ids=self.ids, ms=1.0, reused=0,
                     encoded=int(self.ids.size))


def _front(epoch=0, awake="D"):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = epoch
    f.awake = awake
    f.state = "serving"
    f.tp_prefill_max_tokens = X
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f.store_probe = None
    f._x_exact_reprice_queue = lambda why: 0
    return f


def _price(f, rid, ids):
    f.ftok.ids = ids
    return asyncio.run(f._x_exact_price(rid, "/v1/messages", {"m": 1}, rid, int(ids.size), int(ids.size)))


def _finish(f, rid, ids, pt, ct, depth=None, epoch=0, pending=None):
    f.ftok.m[rid] = ids
    f._x_exact_record(rid, rid, pt, ct, pending, epoch, resumable_depth=depth)


def _lines(caplog, marker):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("WEG2 " + marker)]


# the k2 chain: weg2-16-80 (9140 tokens, ct 0) -> weg2-16-87 (9948 tokens = the same 9140 + a tail)
A = _ids(9140)
B = np.concatenate([A, _ids(808, seed=7)])


def _chain(f):
    """weg2-16-80 priced + finished (ct=0, held epoch, depth 9216), then weg2-16-87 priced."""
    _price(f, "weg2-16-80", A)
    _finish(f, "weg2-16-80", A, pt=9140, ct=0, depth=9216)
    return _price(f, "weg2-16-87", B)


# ---- (1) entry_rid in X-EXACT-PRICE ------------------------------------------------------

def test_price_line_names_the_entry_the_credit_came_from(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    xx = _chain(f)
    assert (xx.pending, xx.credit, xx.src) == (860, 9088, "d_served_epoch")  # the base's price
    line = _lines(caplog, "X-EXACT-PRICE rid=weg2-16-87")[0]
    assert " entry_rid=weg2-16-80 entry_src=served_epoch" in line
    assert line.endswith("entry_rid=weg2-16-80 entry_src=served_epoch"), "additive, at the end"
    # a request with no entry to credit names none
    first = _lines(caplog, "X-EXACT-PRICE rid=weg2-16-80")[0]
    assert first.endswith("entry_rid=- entry_src=-")


def test_the_follow_up_in_the_same_epoch_is_named_held_or_leg2_cached(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    f.tspans.record_presence(A, 4096, prompt_tokens=9140, held_epoch=None, rid="weg2-1-1")
    _price(f, "weg2-1-2", B)
    assert _lines(caplog, "X-EXACT-PRICE rid=weg2-1-2")[0].endswith(
        "entry_rid=weg2-1-1 entry_src=leg2_cached")


def test_every_winner_kind(caplog):
    ts = TokenSpans(agent_span=True)
    # held, same epoch, no credit beyond ct -> held; credit past ct -> served_epoch
    ts.record_presence(A, 9140, prompt_tokens=9140, held_epoch=5, rid="r-held")
    ts.pending(B, epoch=5)
    assert ts.last_winner[1:] == ("r-held", "held")
    ts2 = TokenSpans(agent_span=True)
    ts2.record_presence(A, 0, prompt_tokens=9140, held_epoch=5, resumable_depth=9216, rid="r-ep")
    ts2.pending(B, epoch=5)
    assert ts2.last_winner[1:] == ("r-ep", "served_epoch")
    # another epoch, cap named -> served_anchor
    ts2.pending(B, epoch=6)
    assert ts2.last_winner[1:] == ("r-ep", "served_anchor")
    # plain measured reading, no epoch
    ts3 = TokenSpans(agent_span=True)
    ts3.record_presence(A, 4096, prompt_tokens=9140, held_epoch=None, rid="r-m")
    ts3.pending(B, epoch=None)
    assert ts3.last_winner[1:] == ("r-m", "leg2_cached")
    # in-flight: the rid of the leg, not of a reading
    ts4 = TokenSpans(agent_span=True)
    ts4.record_d_inflight("r-fl", A, 9140)
    ts4.pending(B, epoch=None)
    assert ts4.last_winner[1:] == ("r-fl", "inflight")
    # store anchor: no rid
    ts5 = TokenSpans(agent_span=True)
    ts5.record_store_depth(A, 4096, source="l3_index")
    ts5.pending(B, epoch=None)
    assert ts5.last_winner[1:] == ("-", "store")
    # nothing wins: last_winner is None again
    ts5.pending(_ids(500, seed=99), epoch=None)
    assert ts5.last_winner is None


def test_the_rid_follows_the_entry_not_the_text():
    ts = TokenSpans(agent_span=True)
    ts.record_presence(A, 4096, prompt_tokens=9140, rid="first")
    ts.record_presence(A, 2048, prompt_tokens=9140, rid="second")  # a finish replaces the entry
    ts.pending(B)
    assert ts.last_winner[1] == "second"
    ts.record_presence(A, 0, prompt_tokens=9140)  # ct=0, no epoch: no entry, no stale rid
    assert ts.entry_rid == {}
    ts.pending(B)
    assert ts.last_winner is None


def test_the_rid_side_table_is_bounded():
    ts = TokenSpans(agent_span=True, cap=4)
    for i in range(40):
        ts.record_presence(_ids(300, seed=i), 128, prompt_tokens=300, rid="r%d" % i)
    assert len(ts.entries) == 4 and len(ts.entry_rid) <= 8


def test_pending_signature_and_return_are_the_bases():
    sig = inspect.signature(TokenSpans.pending)
    assert list(sig.parameters) == ["self", "ids", "epoch", "since_seq"]
    ts = TokenSpans(agent_span=True)
    ts.record_presence(A, 0, prompt_tokens=9140, held_epoch=5, resumable_depth=9216, rid="x")
    got = ts.pending(B, epoch=5)
    assert isinstance(got, tuple) and got == (860, 9088, True, "d_served_epoch")


# BASE GOLDEN: this scenario, run on 369f31e2e2 (/tmp/golden_1006_21.py against .wt-cand4c-1006), returned exactly these
GOLDEN = [
    (860, 9088, True, "d_served_epoch"),
    (860, 9088, True, "d_served_anchor"),
    (4948, 5000, True, "d_leg2_cached"),
    (860, 9088, True, "d_inflight"),
    (1900, 4000, True, "l3_index"),
    (1900, 4000, True, "l3_index"),
    (5900, 0, False, "none"),
    (9948, 0, True, "none"),
]


def _scenario(with_rid):
    r = (lambda s: {"rid": s}) if with_rid else (lambda s: {})
    out = []
    ts = TokenSpans(agent_span=True)
    ts.record_presence(A, 0, prompt_tokens=9140, held_epoch=5, resumable_depth=9216, **r("a"))
    out.append(ts.pending(B, epoch=5))      # held, K2-clamped
    out.append(ts.pending(B, epoch=6))      # other epoch: served anchor
    ts2 = TokenSpans(agent_span=True)
    ts2.record_presence(_ids(5000, seed=3), 5000, prompt_tokens=5000, **r("b"))
    out.append(ts2.pending(np.concatenate([_ids(5000, seed=3), _ids(4948, seed=4)])))
    ts3 = TokenSpans(agent_span=True)
    ts3.record_d_inflight("fl", A, 9140)
    ts3.record_presence(_ids(300, seed=1), 64, prompt_tokens=300, **r("c"))
    out.append(ts3.pending(B))             # in flight: the page floor of the prompt
    ts4 = TokenSpans(agent_span=True)
    ts4.record_store_depth(_ids(4000, seed=6), 4000, source="l3_index")
    tail = np.concatenate([_ids(4000, seed=6), _ids(1900, seed=8)])
    out.append(ts4.pending(tail))          # a store fact
    out.append(ts4.pending(tail, since_seq=0))
    out.append(ts4.pending(tail, since_seq=10))                    # only FRESH evidence: none
    out.append(ts4.pending(_ids(9948, seed=77)))                   # nothing shared
    return out


def test_return_values_are_byte_equal_to_the_base_with_and_without_rids():
    assert _scenario(False) == GOLDEN
    assert _scenario(True) == GOLDEN


# ---- (2) PRESENCE-CONTRADICTED-DRYRUN ------------------------------------------------------

def test_a_whole_loss_logs_what_a_retract_would_be_about(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    _finish(f, "weg2-16-87", B, pt=9948, ct=0, depth=9216)
    got = _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")
    assert len(got) == 1
    line = got[0]
    assert line.startswith("WEG2 PRESENCE-CONTRADICTED-DRYRUN rid=weg2-16-87 via=d_direct src=d_served_epoch ")
    for field in ("credited=9088", "realised_cached=0", "d_uncached=9948", "tokens_front=9948",
                  "page=64", "entry_rid=weg2-16-80", "entry_src=served_epoch", "own_entry=1"):
        assert field in line, field
    assert re.search(r"source_key=[0-9a-f]{12} ", line)
    assert "nothing was changed" in line
    # the existing lines of the same request: X-EXACT-ERR unchanged in front of it
    assert any(m.startswith("WEG2 X-EXACT-ERR rid=weg2-16-87 via=d_direct pending_priced=860 "
                            "d_uncached=9948 err=-9088 ") for m in caplog.messages)


def test_the_dryrun_writes_nothing(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    snap = lambda ts: ({k: (v[1], v[2], v[3]) for k, v in ts.entries.items()},  # noqa: E731
                       dict(ts.depth_caps), dict(ts.entry_seq), dict(ts.raw_depths), dict(ts.entry_rid))
    # the same two readings on a bare TokenSpans: what the base state is after the finish of 87
    ref = TokenSpans(agent_span=True)
    ref.record_presence(A, 0, prompt_tokens=9140, held_epoch=0, resumable_depth=9216, rid="weg2-16-80")
    ref.record_presence(B, 0, prompt_tokens=9948, held_epoch=0, resumable_depth=9216, rid="weg2-16-87")
    counters = collections.Counter(f.counters)
    _finish(f, "weg2-16-87", B, pt=9948, ct=0, depth=9216)
    assert _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")
    assert snap(f.tspans) == snap(ref)
    after = f.counters - counters
    assert set(after) == {"x_exact_err_n", "x_exact_err_abs_sum"}, "no counter of its own"
    assert not any("contradict" in k for k in f.counters)


@pytest.mark.parametrize("ct,expect", [(0, True), (64, True), (65, False), (4096, False), (4480, False)])
def test_only_a_whole_loss_logs(caplog, ct, expect):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    _finish(f, "weg2-16-87", B, pt=9948, ct=ct, depth=9216)
    assert bool(_lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")) is expect


def test_a_partial_loss_of_the_k2_class_does_not_log(caplog):
    # cand4c 031341 weg2-14-117 class: D continued at a DEEPER-than-credit anchor (cached 9984 of 12864)
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    _finish(f, "weg2-16-87", B, pt=9948, ct=9984 - 9088 + 64)  # D read 960 of the 9088 credited
    assert not _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")


def test_a_small_credit_does_not_log(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    f.tspans.record_presence(A, 640, prompt_tokens=9140, held_epoch=None, rid="e")  # credit 640 < 1024
    _price(f, "weg2-3-3", B)
    _finish(f, "weg2-3-3", B, pt=9948, ct=0, epoch=None)
    assert not _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")
    assert _lines(caplog, "X-EXACT-ERR rid=weg2-3-3")


def test_after_p_and_token_mismatch_do_not_log(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    # after_p: D read P's prefill back, no comparison
    _finish(f, "weg2-16-87", B, pt=9948, ct=0, pending=object())
    assert not _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")
    # match=0: D tokenised another length
    f2 = _front()
    _chain(f2)
    _finish(f2, "weg2-16-87", B, pt=9949, ct=0)
    assert not _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")


def test_a_request_without_a_price_record_logs_nothing(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _finish(f, "weg2-9-9", B, pt=9948, ct=0)
    assert not _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")


def test_a_three_tuple_x_exact_record_from_older_callers_still_works(caplog):
    # the tests (and any caller) that set ``_x_exact_rid[rid] = (pending, n, src)`` directly
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    f._x_exact_rid["weg2-4-4"] = (860, 9948, "d_served_epoch")
    _finish(f, "weg2-4-4", B, pt=9948, ct=0)
    line = _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")[0]
    assert "entry_rid=- entry_src=- source_key=-" in line


# ---- the scripts that parse the front log keep parsing ----------------------------------------

PRICE_RX = r"X-EXACT-PRICE rid=(\S+) pending=(\d+) tokens=(\d+) credit=(\d+) src=(\S+)"
ERR_RX = (r"X-EXACT-ERR rid=(\S+) via=(\S+) pending_priced=(\d+) d_uncached=(\d+) err=(\S+) "
          r".*?match=(\d) src=(\S+)")
ERR_FULL_RX = (r"X-EXACT-ERR rid=(\S+) via=(\S+) pending_priced=(\d+) d_uncached=(\d+) err=(\S+) "
               r"tokens_front=(\d+) tokens_d=(\d+) match=(\d) src=(\S+)")
PRICE_Q530_RX = r"X-EXACT-PRICE rid=(\S+) .*credit=(\d+) src=(\S+)"


def test_the_parsers_of_the_desk_scripts_read_the_new_lines_as_before(caplog):
    caplog.set_level(logging.INFO, logger="weg2.front")
    f = _front()
    _chain(f)
    _finish(f, "weg2-16-87", B, pt=9948, ct=0, depth=9216)
    price = _lines(caplog, "X-EXACT-PRICE rid=weg2-16-87")[0]
    err = _lines(caplog, "X-EXACT-ERR rid=weg2-16-87")[0]
    assert re.search(PRICE_RX, price).groups() == ("weg2-16-87", "860", "9948", "9088", "d_served_epoch")
    assert re.search(PRICE_Q530_RX, price).groups() == ("weg2-16-87", "9088", "d_served_epoch")
    assert re.search(ERR_RX, err).groups() == ("weg2-16-87", "d_direct", "860", "9948", "-9088", "1",
                                              "d_served_epoch")
    assert re.search(ERR_FULL_RX, err).group(8) == "1"
    # a pattern that picks the first rid= of a line (the front log counters) still gets the request's
    for m in (price, err, _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")[0]):
        assert re.search(r"rid=(\S+)", m).group(1) == "weg2-16-87"
        assert re.search(r"rid=([^\s:,;)]+)", m).group(1) == "weg2-16-87"
    # the guards of the boot watchers (waechter_1006*.awk) key on none of the new words
    new = [price, _lines(caplog, "PRESENCE-CONTRADICTED-DRYRUN")[0]]
    guards = (r"RANK-DEATH|Scheduler hit an exception|Prefill out of memory|EVICTION UNDER-DELIVERED|"
              r"DEBUG-HOLD|OutOfMemoryError|CUDA out of memory|WEG2 STOP W[0-9]+|"
              r"failed to pass monitoredBarrier|RPC-STALL-WATCHDOG|WEG2-FLIP STALL|^Traceback|HAENGT|"
              r"DEADMAN|HW-BORROWED|HW-DERIVE|HW-MISMATCH|FORCED BOOT|FORCED-PAST|BAR1-WINDOW:|"
              r"STORE READ INCOMPLETE|ARENA-CLAIM REFUSED|ADMISSION-WEDGE|PRESENCE-ANCHOR-LOST|"
              r"WEG2-ANCHOR-LOST|ANCHOR-LOST|WEG2 X-GATE rid=|#988 LOADBACK rid=|CAP-BLIND-ADMIT|"
              r"D-MEM-SCHED CAP-LIFT")
    for m in new:
        assert not re.search(guards, m), m

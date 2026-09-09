# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1290 (D leg 2): a leg-2 arrival's store read must REGISTER (trimmed to the
group's room) instead of being refused whole, so the X gate has something
pending to defer on and prices the post-prefetch remainder.

THE MEASUREMENT THIS FILE IS BUILT FROM -- boot weg2sb5g, 2026-09-09, rid
``a977bd8d``.  A LONG request completed leg 1 on P (write_through put the
prefix in the store) and arrived on D for leg 2.  D's intake DID issue the
prefetch (scheduler.py ``_add_request_to_queue`` -> ``_prefetch_kvcache``),
but ``prefetch_from_storage``'s symmetric participation vote refused it
WHOLE::

    #915 PREFETCH REFUSED reason=vote_negative rid=a977bd8d need=24655
      available=20103 threshold=256 occupied=0 limit=27466   (all 3 ranks)

Nothing registered, so nothing was pending: ``WEG2 X-DEFER`` = 0 all boot
(the #1238 fix-7 completion predicate abstains when ``pending_ms is None``),
the witness read ``state=unprobed``, and the X gate priced the WHOLE prompt::

    W50 Weg2TpPrefillExceeded rid=a977bd8d... uncached=24657 X=8742

as if P had never run.  Served fraction on the LONG arm: 3 of 53.

THE FIX, one mechanism: the symmetric vote MIN-reduces the allocated LENGTH
instead of a boolean.  Every rank allocates what it can BEFORE the vote and
trims to the group minimum AFTER it (release-only, so all-or-none survives).
The registered read makes fix-7's EXISTING predicate defer the X gate, and
the post-load group match prices the remainder: 24657 - 20103 = 4554 <= 8742
admits.

RED at the parent ``1f837b8e17``: the boolean vote refuses the whole read and
``test_r1`` fails on the missing registration -- the parent commit IS the
"probe skipped" mutant.  The other danger directions are in-file mutants.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU, no collective --
the reduce is simulated by MIN-ing the ranks' captured votes, which is the
arithmetic ``_all_reduce_attn_groups`` performs.
"""

import logging
import os
import time
from types import MethodType, SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

import sglang.srt.managers.scheduler as sched_mod
import sglang.srt.managers.tp_head_congruence as thc
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import match_refusal_census as census_mod
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# ------------------------------------------------------------------ fixture
# EVERY number below is read off boot weg2sb5g's own lines (module docstring),
# never picked: the #915 refusal line carries NEED / AVAILABLE / THRESHOLD /
# LIMIT, the W50 line carries PROMPT (uncached) and X, and the launcher argv
# carries PAGE (--page-size 1 on this form).
NEED = 24655
AVAILABLE = 20103
THRESHOLD = 256
PAGE = 1
PROMPT = 24657
X = 8742
#: What the group trim must register with the sb5g pools: the page-floored
#: room, which at PAGE=1 is the room itself.
GROUP_LEN = min(NEED, AVAILABLE - (AVAILABLE % PAGE))
#: The remainder D must price after the trimmed read lands.
REMAINDER = PROMPT - GROUP_LEN


class _FakeHostPool:
    """The two calls ``prefetch_from_storage`` makes, over a free counter."""

    def __init__(self, free):
        self.free = int(free)
        self.allocs = []
        self.released = 0

    def available_size(self):
        return self.free

    def alloc(self, need_size):
        assert need_size % PAGE == 0
        if need_size > self.free:
            return None
        self.free -= need_size
        self.allocs.append(need_size)
        return torch.arange(need_size, dtype=torch.int64)


def _cache_stub(available, peer_votes, page=PAGE, threshold=THRESHOLD):
    """A UnifiedRadixCache stand-in with the REAL ``prefetch_from_storage``
    bound and a reduce stub that MINs this rank's vote against ``peer_votes``
    -- the arithmetic the real MIN all-reduce performs, with the tag head
    forwarded unchanged (every simulated peer is in the same collective).

    The captured local vote is kept on ``stub.votes`` so a rank-uniformity
    test can rerun the MIN across ranks explicitly.
    """
    pool = _FakeHostPool(available)
    registered = {}

    def _reduce(t, op, label):
        assert label == "prefetch_participation_vote"
        stub.votes.append(int(t[2].item()))
        t[2] = min([int(t[2].item())] + [int(v) for v in peer_votes])
        return t

    def _prefetch(req_id, host_indices, prefetch_key, last_hash, prefix_keys,
                  extra_pools=None):
        registered[req_id] = SimpleNamespace(
            host_indices=host_indices,
            key_len=len(prefetch_key),
            extra_pools=extra_pools,
        )
        return SimpleNamespace()

    controller = SimpleNamespace(
        mem_pool_host=pool,
        prefetch_rate_limited=lambda: False,
        prefetch_tokens_occupied=0,
        append_host_mem_release=lambda host_indices=None, extra_pools=None: (
            setattr(pool, "released", pool.released + (
                0 if host_indices is None else len(host_indices)))
        ),
        prefetch=_prefetch,
    )
    stub = SimpleNamespace(
        enable_storage=True,
        cache_controller=controller,
        page_size=page,
        prefetch_threshold=threshold,
        is_eagle=False,
        _components_tuple=(),
        sidecar_pool_specs=(),
        ongoing_prefetch={},
        votes=[],
        registered=registered,
        pool=pool,
        _hicache_prefetch_symmetric=lambda: True,
        _all_reduce_attn_groups=_reduce,
        evict_host=lambda *a, **k: None,
        inc_host_lock_ref=lambda node: SimpleNamespace(to_dec_params=lambda: "l"),
        dec_host_lock_ref=lambda node, params: None,
        _retire_ongoing_prefetch=lambda rid: None,
    )
    for name in (
        "prefetch_from_storage",
        "_build_sidecar_transfers",
        "_log_prefetch_refused",
        "_log_prefetch_truncated",
        "_prefetch_line_terms",
    ):
        setattr(stub, name, MethodType(getattr(UnifiedRadixCache, name), stub))
    return stub


def _node():
    return SimpleNamespace(key=None)


def _issue(stub, rid="a977bd8d", tokens=NEED):
    before = dict(census_mod.PREFETCH_GATE_COUNTS)
    stub.prefetch_from_storage(rid, _node(), list(range(tokens)))
    delta = {
        k: census_mod.PREFETCH_GATE_COUNTS.get(k, 0) - before.get(k, 0)
        for k in set(census_mod.PREFETCH_GATE_COUNTS) | set(before)
    }
    return {k: v for k, v in delta.items() if v}


# ================================ r1: the read REGISTERS instead of refusing
def test_r1_the_leg2_read_registers_trimmed_to_the_groups_room(caplog):
    """RED AT THE PARENT 1f837b8e17 -- the parent is the 'probe skipped'
    mutant: the boolean vote refuses the whole read (vote_negative, nothing
    in ``ongoing_prefetch``... via the controller: nothing registered), the
    witness stays ``unprobed`` and W50 prices the full prompt.

    With the length vote, the sb5g request's read registers at the group's
    page-floored room, spoken on the existing grep-able L2 line."""
    stub = _cache_stub(AVAILABLE, peer_votes=[GROUP_LEN, GROUP_LEN])
    with caplog.at_level(logging.WARNING):
        delta = _issue(stub)
    assert "a977bd8d" in stub.registered, (
        "the leg-2 store read was REFUSED whole -- this is boot weg2sb5g's "
        "vote_negative exactly (need > available on every rank)"
    )
    got = stub.registered["a977bd8d"]
    assert got.key_len == GROUP_LEN, "registered at the group minimum"
    assert len(got.host_indices) == GROUP_LEN
    assert delta.get("host_pool_truncated") == 1, (
        "counted under the existing truncation key, which stands beside "
        "issued and keeps the #915 intake partition summable"
    )
    assert delta.get("host_pool_truncated_tokens") == NEED - GROUP_LEN, (
        "and the token companion carries the trimmed span"
    )
    assert delta.get("vote_negative") is None, "not a refusal any more"
    assert "#915 PREFETCH TRUNCATED" in caplog.text, "the grep-able line"
    assert f"got={GROUP_LEN}" in caplog.text
    assert stub.pool.released == 0, (
        "this rank allocated exactly the group length; nothing to release"
    )


def test_r1b_a_rank_above_the_min_releases_its_tail_only():
    """The trim is RELEASE-ONLY after consensus: a rank that allocated its
    full span but was bounded by a poorer peer hands back exactly the tail
    and registers the group length -- no post-vote alloc exists to fail."""
    poorer = GROUP_LEN - PAGE * 7
    stub = _cache_stub(NEED, peer_votes=[poorer, NEED])
    _issue(stub)
    got = stub.registered["a977bd8d"]
    assert got.key_len == poorer
    assert len(got.host_indices) == poorer
    assert stub.pool.released == NEED - poorer, "the tail, nothing else"
    assert stub.pool.allocs == [NEED], "one alloc, before the vote"


def test_r2_three_ranks_register_one_identical_length():
    """RANK-UNIFORM (the six-rank/#968 rule): the verdict is the reduce's,
    followers never diverge -- a rank-local span decision is a bug.  Three
    ranks with DIVERGENT pool room register the SAME group length, each
    releasing its own surplus.

    The can-fail half is the mutant below (``test_m1``): wired to its own
    vote instead of the reduced one, the lengths genuinely diverge."""
    rooms = [AVAILABLE, AVAILABLE - PAGE * 3719, AVAILABLE + PAGE * 4897]
    votes = [min(NEED, r - (r % PAGE)) for r in rooms]
    group = min(votes)
    stubs = []
    for me, room in enumerate(rooms):
        peers = [votes[j] for j in range(3) if j != me]
        stub = _cache_stub(room, peer_votes=peers)
        _issue(stub)
        stubs.append(stub)
    lengths = {s.registered["a977bd8d"].key_len for s in stubs}
    assert lengths == {group}, (
        f"the ranks registered {lengths}: a split registration set is the "
        "#580 desync -- mismatched completion collectives"
    )
    for s, vote in zip(stubs, votes):
        assert s.votes == [vote], "each rank voted its own allocated length"
        assert s.pool.released == vote - group, "and released its own surplus"


def test_r3_a_group_below_threshold_still_refuses_uniformly(caplog):
    """The old negative consensus is intact: a peer with less than one
    threshold of room zeroes the vote, every rank refuses by the same name,
    releases everything, and registers nothing.  W50 then fires on a
    genuinely unprobeable read -- the honest case, where the front's #1291
    W53 terminal is the right answer."""
    stub = _cache_stub(AVAILABLE, peer_votes=[0, GROUP_LEN])
    with caplog.at_level(logging.WARNING):
        delta = _issue(stub)
    assert stub.registered == {}, "nothing registers on a negative consensus"
    assert delta.get("vote_negative") == 1
    assert "reason=vote_negative" in caplog.text
    assert stub.pool.released == GROUP_LEN, "the local alloc is handed back"


# =========================== r4: the X gate joins -- defer, then remainder
def _defer_stub(reqs, ongoing_rids=(), x=X):
    """The fix-7 harness shape (test_weg2_sched_fix7_0908) with this file's
    numbers: the REAL predicate bodies bound over a fake tree whose
    ``ongoing_prefetch`` holds the trimmed registration."""
    tree = SimpleNamespace(
        ongoing_prefetch={r: object() for r in ongoing_rids},
        cache_controller=SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=10 ** 9)
        ),
        prefetch_timeout_base=1.0,
        prefetch_timeout_per_page=0.01,
        page_size=64,
    )
    stub = SimpleNamespace(
        waiting_queue=list(reqs),
        tree_cache=tree,
        ps=SimpleNamespace(tp_size=3),
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
    )
    for name in (
        "_weg2_store_read_is_pending",
        "_weg2_local_store_read_pending_ages",
        "_weg2_local_store_read_pending_ms",
        "_weg2_x_store_read_bound_s",
        "_weg2_x_defers",
        "_deferred_prefetch_bound_s",
        "_weg2_x_refuses",
        "_weg2_host_carry_tokens",
        "weg2_uncached_extent",
    ):
        setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub


def _req(rid="a977bd8d", prompt_tokens=PROMPT):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(prompt_tokens)),
        origin_input_ids=list(range(prompt_tokens)),
        prefix_indices=[],
        host_hit_length=0,
        prefetch_deferred=None,
    )


def _head(rid, group_ms, match=0):
    canonical = thc.canonical_head_rids([rid])
    ages = {} if group_ms is None else {rid: int(group_ms)}
    return thc.build_uniform_head_inputs(
        canonical,
        thc.build_head_order_payload(canonical, {rid: match}),
        None,
        True,
        thc.build_x_pending_payload(canonical, ages),
    )


def test_r4_the_registered_read_makes_fix7_defer_and_the_remainder_admits(caplog):
    """THE JOIN, with the sb5g numbers end to end.  While the trimmed read is
    in flight the EXISTING fix-7 predicate defers (X-DEFER speaks -- it was 0
    all boot because nothing ever registered); once it lands, the priced
    extent is the post-prefetch remainder, and the remainder admits."""
    req = _req()
    stub = _defer_stub([req], ongoing_rids=[req.rid])
    stub._weg2_x_defer_since = {req.rid: time.monotonic() - 0.5}
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert stub._weg2_x_defers(req, _head(req.rid, 500)) is True
    assert "WEG2 X-DEFER" in caplog.text and "verdict=defer" in caplog.text
    # ... the read lands: the record is popped, the group match is the loaded
    # span, and the extent is the remainder -- never the whole prompt.
    landed = _defer_stub([req], ongoing_rids=[])
    head = _head(req.rid, None, match=GROUP_LEN)
    assert landed._weg2_x_defers(req, head) is False, "nothing pending: price"
    assert landed.weg2_uncached_extent(req, head) == REMAINDER
    assert REMAINDER <= X, "the sb5g arithmetic itself"
    assert landed._weg2_x_refuses(req, head) is False, (
        "the remainder admits: this is the request weg2sb5g answered W50"
    )


def test_r5_an_unregistered_read_still_prices_and_w50_stays_reachable():
    """W50 remains reachable EXACTLY where the read provably could not be
    issued (group refusal, ``test_r3``): nothing pending, the gate prices the
    full extent, the front's #1291 W53 terminal takes it from there.  The
    defer must never manufacture a wait for a read that is not coming."""
    req = _req()
    stub = _defer_stub([req], ongoing_rids=[])
    head = _head(req.rid, None, match=0)
    assert stub._weg2_store_read_is_pending(req) is False
    assert stub._weg2_x_defers(req, head) is False
    assert stub.weg2_uncached_extent(req, head) == PROMPT
    assert stub._weg2_x_refuses(req, head) is True, "W50, honestly"


# ============================================== the danger-direction mutants
def test_m1_a_rank_reading_its_own_vote_splits_the_registration_set():
    """MUTANT (rank-local probe decision): wire each rank to its OWN vote
    instead of the reduced one.  The registered lengths genuinely diverge --
    which is what ``test_r2`` refuses -- proving r2 discriminates."""
    rooms = [AVAILABLE, AVAILABLE - PAGE * 3719, AVAILABLE + PAGE * 4897]
    votes = [min(NEED, r - (r % PAGE)) for r in rooms]
    stubs = []
    for me, room in enumerate(rooms):
        stub = _cache_stub(room, peer_votes=[votes[me]])  # own echo: no MIN
        _issue(stub)
        stubs.append(stub)
    lengths = {s.registered["a977bd8d"].key_len for s in stubs}
    assert len(lengths) == 3, (
        "the mutant must produce the divergence r2 exists to catch; if it "
        "does not, r2 is not measuring rank uniformity"
    )


def test_m2_pricing_the_pre_prefetch_extent_would_refuse_the_served_request():
    """MUTANT (X gate reads pre-prefetch uncached): price with the match the
    gate saw BEFORE the read landed (0).  The extent is the whole prompt and
    the gate refuses -- boot weg2sb5g's W50 verbatim -- proving r4's
    remainder assertion discriminates."""
    req = _req()
    stub = _defer_stub([req], ongoing_rids=[])
    pre = _head(req.rid, None, match=0)
    assert stub.weg2_uncached_extent(req, pre) == PROMPT
    assert stub._weg2_x_refuses(req, pre) is True, (
        "the pre-prefetch price refuses the request the remainder admits"
    )


def test_m3_a_defer_that_never_expires_would_hold_forever():
    """MUTANT (defer never resolves = hang): the bound is what prevents it.
    Past the span's own length-priced bound the verdict is BOUND_EXPIRED and
    the request is priced; a verdict wired to defer-while-pending would hold
    the queue for ever.  Asserted on the verdict function itself, the same
    term ``_weg2_x_defers`` consumes."""
    req = _req()
    stub = _defer_stub([req], ongoing_rids=[req.rid])
    bound = stub._weg2_x_store_read_bound_s(req)
    assert bound > 0.0
    late_ms = int((bound + 5.0) * 1000)
    assert thc.x_completion_verdict(late_ms, bound) == thc.X_BOUND_EXPIRED
    stub._weg2_x_defer_since = {req.rid: time.monotonic() - (bound + 5.0)}
    assert stub._weg2_x_defers(req, _head(req.rid, late_ms)) is False, (
        "past the bound the gate prices instead of deferring"
    )
    # The mutant: a verdict that never expires. It answers defer on the same
    # reading -- the hang r5/m3 exist to keep impossible.
    mutant = lambda pending_ms, bound_s: (  # noqa: E731
        thc.X_PRICE if pending_ms is None else thc.X_DEFER
    )
    assert mutant(late_ms, bound) == thc.X_DEFER, (
        "the mutant genuinely hangs this reading; only the bound term "
        "separates the fix from a livelock in the other costume"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

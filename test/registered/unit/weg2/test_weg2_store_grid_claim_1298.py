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
"""#1298: a store read that LANDED still handed D nothing.

BOOT weg2sb5h, 2026-09-09, tip ``57fef0ce6e``.  P wrote every long prompt
whole, under D's own key, 7.1-43.4 s before D first admitted it -- proven by
round-trip and not by a counter: 15 rids came back at
``cached_tokens == prompt_tokens - 2`` (e.g. 22,406 of 22,408).  D then
refused 24 of those requests at the X gate with ``uncached`` equal to the
WHOLE prompt.  Two terms in series, both on D's read path, cut the handback to
zero:

* **T1** the D-phase host staging pool is 30,518 rows / limit 27,466 tokens and
  fits exactly ONE ~22k read.  The first leg-2 offer of an epoch takes it; the
  siblings get the residual, granted UN-QUANTIZED because the existing floor is
  ``available % page_size`` and ``page_size`` is 1 on this form.
* **T2** the residual misses the store's 4,096-token block boundary by 42-315
  tokens, the probe credits ``4094`` instead of ``8190``, and
  ``resolve_draft_claim``'s ``trim`` branch collapses that to **0**.

MEASURED POPULATION, this file's whole fixture (instruments: ``#915 PREFETCH
TRUNCATED`` and ``WEG2 DRAFT-PRESENCE`` in the sb5h D log; denominators stated
per constant below, joined per rid):

* 111 truncation lines = 37 rids x 3 ranks, rank-unanimous, all
  ``over_bound=true``, all ``chunk=4096``.
* 123 DRAFT-PRESENCE lines.  **69 took ``trim`` and every one of the 69
  returned claim=0.**  The 54 that took ``cold`` each claimed their full ``k``.
* Joined per rid, 25 of the 37 truncated rids also carry a presence line: 22
  distinct ``(need, got, k, d)`` rows ended in claim 0, 2 in claim 8190.  The
  only thing separating them is the 8,192 boundary
  (``got >= 8192 -> k=8190``, ``got < 8192 -> k=4094``, 75/75 joined lines,
  0 mismatches).

RED AT THE PARENT ``57fef0ce6e``: ``test_b1`` returns claim 0 for the boot's
own arguments, ``test_a1`` grants 8,150 un-quantized rows, and ``test_c1`` --
the real chain, a real ``HiCacheFile`` on disk written through the WRITER's key
funnel and probed through the READER's -- ends at claim 0.  ``test_f3`` is RED
on the sb5h log itself and stays red until part (C) lands; it is the next
boot's acceptance, not a claim about this commit.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU, no collective.  The
MIN all-reduce is simulated by MIN-ing the ranks' captured votes, which is the
arithmetic ``_all_reduce_attn_groups`` performs (same harness shape as
``test_weg2_leg2_store_probe_1290``).
"""

import logging
import os
import re
import tempfile
from types import MethodType, SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.managers.cache_controller import resolve_draft_claim
from sglang.srt.mem_cache import match_refusal_census as census_mod
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _store_grid_floor,
)

# ---------------------------------------------------------------- constants
# EVERY number below is read off boot weg2sb5h's own lines, never picked.
#: `chunk=` on all 111 `#915 PREFETCH TRUNCATED` lines.
CHUNK = 4096
#: `--page-size 1` on this form; the `available % page_size` floor is a no-op.
PAGE = 1
#: `#915 PREFETCH LIMIT now=27466 (fraction=0.9 x host size 30518) role=staging`
POOL_ROWS = 30518
POOL_LIMIT = 27466
THRESHOLD = 256
#: The 22 distinct `(need, got, kv_pages, draft_pages)` rows that ended in
#: claim=0, joined per rid across the two instruments.  Not a sample: this is
#: every truncated rid of the boot that also carries a presence line and went
#: to `trim`.
TRIM_ROWS = (
    (15982, 7991, 4094, 48), (15982, 8074, 4094, 48), (16198, 8043, 4094, 48),
    (16198, 8100, 4094, 48), (16215, 7991, 4094, 48), (16215, 8043, 4094, 48),
    (16520, 7891, 4094, 48), (16826, 7877, 4094, 47), (18187, 7991, 4094, 48),
    (18360, 8128, 4094, 48), (18360, 8150, 4094, 48), (18493, 7991, 4094, 47),
    (18493, 8016, 4094, 47), (18557, 7877, 4094, 48), (22190, 8043, 4094, 47),
    (22331, 8016, 4094, 47), (22331, 8074, 4094, 45), (22352, 7891, 4094, 45),
    (22362, 8128, 4094, 45), (22364, 8150, 4094, 45), (22461, 7991, 4094, 47),
    (22493, 8100, 4094, 46),
)
#: The two rows that survived, and the only difference: got >= 8192.
COLD_ROWS = ((16520, 8247, 8190, 48), (18187, 8247, 8190, 48))
#: The one trim row off the 4,096 grid -- proof the zero is not a property of
#: 4094 but of the branch (`WEG2 DRAFT-PRESENCE ... kv_pages=2727
#: draft_pages=2671 claim=0 mode=trim`, 3 lines / 1 rid).
OFF_GRID_TRIM = (2727, 2671)
#: The sb5h D log, the F3 subject.  Absent on the remote desk by design (the
#: evidence tree is not shipped over the link), so F3 skips there.
SB5H_D_LOG = (
    "/spinning/evidence-665-f1/"
    "boot_weg2_weg2sb5h_57fef0ce6e_0909_103119.D.log"
)


# ------------------------------------------------------- (A) the pool grant
class _FakeHostPool:
    """The two calls ``prefetch_from_storage`` makes, over a free counter."""

    def __init__(self, free, refuse_first=0):
        self.free = int(free)
        self.allocs = []
        self.released = 0
        #: A pool that REPORTS room and refuses the alloc anyway -- the
        #: fragmentation case the tree already names (`host_alloc_failed`,
        #: "the room raced away between the read and the alloc").  It is the
        #: only way the symmetric branch is reached with the WHOLE span still
        #: affordable, so it is the only way the full-grant guard is
        #: reachable at all.  Measured: without it, `test_a2` passed against a
        #: mutant that had the guard removed.
        self.refuse_first = int(refuse_first)

    def available_size(self):
        return self.free

    def alloc(self, need_size):
        if self.refuse_first > 0:
            self.refuse_first -= 1
            return None
        if need_size > self.free:
            return None
        self.free -= need_size
        self.allocs.append(need_size)
        return torch.arange(need_size, dtype=torch.int64)


def _cache_stub(available, peer_votes, page=PAGE, threshold=THRESHOLD, chunk=CHUNK,
                refuse_first=0):
    """A ``UnifiedRadixCache`` stand-in driving the REAL
    ``prefetch_from_storage`` (and through it the REAL ``_store_grid_floor``),
    with a reduce stub that MINs this rank's vote against ``peer_votes``."""
    pool = _FakeHostPool(available, refuse_first=refuse_first)
    registered = {}

    def _reduce(t, op, label):
        assert label == "prefetch_participation_vote"
        stub.votes.append(int(t[2].item()))
        t[2] = min([int(t[2].item())] + [int(v) for v in peer_votes])
        return t

    def _prefetch(req_id, host_indices, prefetch_key, last_hash, prefix_keys,
                  extra_pools=None):
        registered[req_id] = SimpleNamespace(
            host_indices=host_indices, key_len=len(prefetch_key)
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
        _prefetch_chunk_tokens=chunk,
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


def _issue(stub, rid, tokens):
    before = dict(census_mod.PREFETCH_GATE_COUNTS)
    stub.prefetch_from_storage(rid, SimpleNamespace(key=None), list(range(tokens)))
    delta = {
        k: census_mod.PREFETCH_GATE_COUNTS.get(k, 0) - before.get(k, 0)
        for k in set(census_mod.PREFETCH_GATE_COUNTS) | set(before)
    }
    return {k: v for k, v in delta.items() if v}


def test_a1_the_residual_grant_is_quantized_to_the_store_grid():
    """RED AT THE PARENT: the residual is floored to ``page_size`` (=1), so
    the rank takes 8,150 rows and the probe credits what 4,096 rows would
    have bought.  Every one of the boot's ten distinct ``got`` values is
    driven, not one specimen."""
    got_values = sorted({row[1] for row in TRIM_ROWS} | {row[1] for row in COLD_ROWS})
    assert len(got_values) == 10, "the boot's ten distinct granted spans"
    for got in got_values:
        need = 22331  # a `need` from the boot, larger than every residual
        stub = _cache_stub(got, peer_votes=[got, got])
        _issue(stub, "grid", need)
        registered = stub.registered.get("grid")
        assert registered is not None, f"got={got}: the read must still register"
        assert registered.key_len % CHUNK == 0, (
            f"got={got}: granted {registered.key_len} rows, which is "
            f"{registered.key_len % CHUNK} rows past a {CHUNK}-token block "
            "boundary -- rows the store credits nothing for and the siblings "
            "cannot use"
        )
        assert registered.key_len == got - (got % CHUNK)
        assert stub.pool.free == got - registered.key_len, (
            "the rows above the boundary stay in the pool for the siblings"
        )


def test_a2_a_full_span_grant_is_never_floored():
    """THE DANGER DIRECTION of (A), through the ONLY path that reaches it.

    15 round trips of this boot came back at ``prompt_tokens - 2`` (22,406 of
    22,408) -- NOT ``floor_4096(22408)``.  A grant that covers the whole span
    is not a residual and must pass through untouched, or the fix breaks the
    path that works.

    REACHABILITY, and it is the point of this test rather than a detail: the
    symmetric branch runs ONLY after ``alloc(prefetch_length)`` has already
    failed twice, so a pool with plain room never enters the helper and an
    assertion written that way passes no matter what the helper does.  It has
    to be a pool that REPORTS the room and refuses the alloc -- the
    fragmentation case the tree names itself.  Measured: the naive version of
    this test passed against a mutant with the full-grant guard deleted.
    """
    for need in (22408, 22331, 18187, 8192, CHUNK + 1, THRESHOLD):
        stub = _cache_stub(
            POOL_LIMIT + need, peer_votes=[need, need], refuse_first=2
        )
        _issue(stub, "full", need)
        registered = stub.registered.get("full")
        assert registered is not None and registered.key_len == need, (
            f"need={need}: a full grant was floored to "
            f"{None if registered is None else registered.key_len}"
        )
        assert stub.pool.allocs == [need], (
            f"need={need}: the third alloc must ask for the WHOLE span"
        )


def test_a2b_the_grid_floor_itself_leaves_a_full_grant_alone():
    """The same guard, stated on the helper directly, so a future call site
    inherits the property rather than re-deriving it."""
    for span in (22408, 22331, 18187, 8192, CHUNK, CHUNK + 1, 1):
        assert _store_grid_floor(span, span, CHUNK) == span
        assert _store_grid_floor(span + 1, span, CHUNK) == span + 1, (
            "a grant ABOVE the span is not a residual either"
        )
    # and a genuine residual is floored, on the same helper
    assert _store_grid_floor(8150, 22331, CHUNK) == CHUNK
    assert _store_grid_floor(8247, 22331, CHUNK) == 2 * CHUNK
    assert _store_grid_floor(8150, 22331, 0) == 8150, "no grid, no floor"
    assert _store_grid_floor(8150, 22331, -1) == 8150


def test_a3_a_residual_below_one_block_takes_the_existing_exit(caplog):
    """No new refusal reason is invented.  A residual that the grid floor
    drops under ``prefetch_threshold`` leaves ``host_indices`` None, votes 0,
    and the group refuses under the name it already uses."""
    stub = _cache_stub(CHUNK - 1, peer_votes=[CHUNK - 1, CHUNK - 1])
    with caplog.at_level(logging.WARNING):
        delta = _issue(stub, "short", 22331)
    assert stub.registered == {}, "nothing registers on a sub-block residual"
    assert delta.get("vote_negative") == 1
    assert "reason=vote_negative" in caplog.text
    assert stub.pool.allocs == [], "and no rows were taken to be thrown away"


def test_a4_the_grid_floor_is_rank_uniform():
    """Ranks never disagree.  Three ranks with DIVERGENT pool room register
    ONE identical length -- the group MIN of three floored votes -- and each
    releases only its own surplus."""
    rooms = [8150, 8247, 12290]
    votes = [min(22331, r) - (min(22331, r) % CHUNK) for r in rooms]
    group = min(votes)
    stubs = []
    for me, room in enumerate(rooms):
        peers = [votes[j] for j in range(3) if j != me]
        stub = _cache_stub(room, peer_votes=peers)
        _issue(stub, "uniform", 22331)
        stubs.append(stub)
    assert {s.registered["uniform"].key_len for s in stubs} == {group}
    for s, vote in zip(stubs, votes):
        assert s.votes == [vote], "each rank voted its own floored length"
        assert s.pool.released == vote - group


def test_a5_a_tree_without_the_chunk_term_is_left_alone():
    """``over_bound=unknown`` is a stand-in, never a verdict against an
    unmeasured bound -- so a tree built without ``chunked_prefill_size`` must
    not be floored against a guessed grid."""
    stub = _cache_stub(8150, peer_votes=[8150, 8150], chunk=-1)
    _issue(stub, "nogrid", 22331)
    assert stub.registered["nogrid"].key_len == 8150


# ------------------------------------------------------------ (B) the claim
def _cold_claim(k, d):
    """What the ``cold`` contract yields for the same input: the whole KV
    prefix, with ``[d, k)`` named for the #993 zero fill."""
    return k


def test_b1_the_trim_zero_is_gone_on_the_boots_own_arguments():
    """RED AT THE PARENT -- F1.  Arguments read verbatim off ``D L77876`` /
    ``D L105408``: ``resolve_draft_claim(4094, 47, 4096, reprobe->0)`` returned
    ``(0, 0, 'trim', None)``, discarding 4,047 valid KV pages.  Driven over all
    22 measured rows, not one specimen."""
    for need, got, k, d in TRIM_ROWS:
        claim, draft_claim, mode, span = resolve_draft_claim(k, d, CHUNK, lambda _d: 0)
        assert claim >= _cold_claim(k, d), (
            f"need={need} got={got} k={k} d={d}: claim {claim} is below the "
            f"{_cold_claim(k, d)} the cold branch yields for the same input"
        )
        assert (claim, draft_claim, mode, span) == (k, d, "cold", (d, k))


def test_b2_the_cold_guard_arm_is_unchanged():
    """The arm that already worked stays byte-identical: ``got=8247`` credited
    8,190 and served."""
    for need, got, k, d in COLD_ROWS:
        assert resolve_draft_claim(k, d, CHUNK, lambda _d: 0) == (
            k, d, "cold", (d, k)
        ), f"need={need} got={got}"
    assert resolve_draft_claim(10, 10, CHUNK, lambda d: d) == (10, 10, "full", None)
    assert resolve_draft_claim(10, 12, CHUNK, lambda d: d) == (10, 10, "full", None)


def test_b3_no_reprobe_answer_can_lower_the_claim():
    """The cap ``max(0, min(reprobe(d), d))`` was the mechanism: whatever the
    store answered, the branch could not return more than ``d``.  Sweep the
    whole answer space -- generous, exact, short, zero, negative, absurd -- on
    the boot's own ``(k, d)`` and on the one row that is NOT on the 4,096
    grid."""
    pairs = [(k, d) for _n, _g, k, d in TRIM_ROWS] + [OFF_GRID_TRIM]
    for k, d in pairs:
        for answer in (0, -5, 1, d - 1, d, d + 1, k, k * 4):
            claim, _dc, mode, _s = resolve_draft_claim(
                k, d, CHUNK, lambda _d, a=answer: a
            )
            assert claim == k and mode == "cold", (
                f"k={k} d={d} reprobe->{answer}: claim {claim} mode {mode}"
            )


def test_b4_the_reprobe_is_no_longer_called_at_all():
    """The deleted branch was the only caller.  A store round trip that can
    only lower the claim is not merely unused, it is a cost: one
    ``batch_exists_v2`` per presence probe."""
    calls = []
    resolve_draft_claim(4094, 47, CHUNK, lambda d: calls.append(d) or 0)
    assert calls == [], f"the store was re-probed {len(calls)} times for nothing"


# --------------------------------------------- (C) the log-invariant, F3
@pytest.mark.skipif(
    not os.path.exists(SB5H_D_LOG),
    reason="the sb5h evidence tree is not shipped to the remote desk",
)
def test_f3_every_truncated_read_reaches_a_terminal_line():
    """F3 -- THE ACCEPTANCE FOR PART (C), AND IT IS EXPECTED RED HERE.

    A read the pool cut must end somewhere a reader can see: a completion, a
    reap, a refusal, or a defer.  On sb5h, 34 of 37 truncated rids have NONE
    of the four -- issued, cut, and gone without a word ~1 s later, which is
    why ``WEG2 X-DEFER`` is 0 in 109,471 lines while 24 requests were priced
    at their whole prompt.  Parts (A) and (B) of #1298 do not close this;
    part (C) does, and this test is how the next boot says so.
    """
    trunc = {}
    terminal = {}
    rt = re.compile(r"#915 PREFETCH TRUNCATED rid=(\S+) need=(\d+) got=(\d+)")
    markers = (
        ("HiCache prefetch success req=", "success"),
        ("#905 PREFETCH-COMPLETE", "complete"),
        ("#1157 PREFETCH REAPED", "reaped"),
        ("#915 PREFETCH REFUSED", "refused"),
        ("WEG2 X-DEFER", "defer"),
    )
    with open(SB5H_D_LOG, errors="replace") as fh:
        for line in fh:
            hit = rt.search(line)
            if hit:
                trunc.setdefault(hit.group(1)[:8], (hit.group(2), hit.group(3)))
            for literal, name in markers:
                if literal in line:
                    rid = re.search(r"(?:req|rid)=([0-9a-f]{6,})", line)
                    if rid:
                        terminal.setdefault(rid.group(1)[:8], set()).add(name)
    assert trunc, "the truncation instrument itself is missing from this log"
    orphans = sorted(r for r in trunc if not terminal.get(r))
    assert not orphans, (
        f"{len(orphans)} of {len(trunc)} truncated reads have no completion, "
        f"no reap, no refusal and no defer line: {orphans[:6]}... -- the X "
        "gate sees neither a pending read nor a refused one, so it prices the "
        "whole prompt (#1298 part C)"
    )


# ------------------------------------- the REAL chain, on a real disk store
IDENTITY = "0123456789abcdef"


def _store(root):
    return HiCacheFile(
        HiCacheStorageConfig(
            tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=True,
            model_name="Qwen3.8-27B", model_identity_hash=IDENTITY,
        ),
        file_path=root,
    )


def _draft_controller(store, page_size=PAGE, chunk=CHUNK):
    """The REAL ``_apply_draft_claim`` and ``_draft_chunk_pages`` bound over a
    bare controller -- the D-side consumer of the store's probe."""
    stub = SimpleNamespace(
        storage_backend=store,
        page_size=page_size,
        tp_rank=0,
        _draft_trim_requests=0,
        _draft_cold_requests=0,
        _draft_presence_n=0,
        draft_cold_spans={},
        _chunk=chunk,
    )
    stub._draft_chunk_pages = lambda: max(1, chunk // max(1, page_size))
    stub._apply_draft_claim = MethodType(
        HybridCacheController._apply_draft_claim, stub
    )
    return stub


def test_c1_the_real_chain_p_writes_d_reads_and_the_claim_is_not_zero():
    """RED AT THE PARENT, and it is the whole ticket in one function.

    A REAL ``HiCacheFile`` on disk.  P writes the published prefix through the
    WRITER's key funnel (``set(_log_key(pool, key))``, what ``_write_page``
    calls); D probes through the READER's funnel
    (``batch_exists_v2`` -> ``_get_component_key``) -- the two funnels #1295
    proved share the canonical suffix -- and D's real ``_apply_draft_claim``
    resolves the claim.  The sequence is the log-proven one: ``need=22331``,
    the pool grants ``got=8016``, the store holds 4,094 published KV pages and
    47 draft pages, and the handback must not be zero.
    """
    need, got, k, d = 22331, 8016, 4094, 47
    with tempfile.TemporaryDirectory() as root:
        store = _store(root)
        if os.path.realpath(store.file_path) != os.path.realpath(root):
            pytest.skip(
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR overrides the store dir"
            )
        keys = [f"weg2-1298-page{i:07d}" for i in range(got)]
        blob = torch.arange(8, dtype=torch.uint8)
        # --- P's leg: publish the KV prefix, then the shorter draft prefix.
        for key in keys[:k]:
            assert store.set(store._log_key(PoolName.KV, key), blob)
        for key in keys[:d]:
            assert store.set(store._log_key(PoolName.DRAFT, key), blob)
        # --- D's leg: the real presence probe over the granted span.
        draft_probe = PoolTransfer(
            name=PoolName.DRAFT,
            keys=keys[:got],
            hit_policy=PoolHitPolicy.ALL_PAGES,
            caps_claim=False,
        )
        hit = store.batch_exists_v2(keys, [draft_probe], None)
        assert hit.kv_hit_pages == k, (
            f"the store credited {hit.kv_hit_pages} of {k} published pages -- "
            "this test's premise (P's write landed) is broken, not its claim"
        )
        assert int(hit.extra_pool_hit_pages.get(str(PoolName.DRAFT), 0)) == d
        # --- D's claim, through the real resolver.
        ctrl = _draft_controller(store)
        out = ctrl._apply_draft_claim(
            SimpleNamespace(request_id="weg2-1298", draft_claim_pages=None,
                            draft_cold_span=None),
            keys,
            [draft_probe],
            draft_probe,
            hit,
            None,
        )
        assert out.kv_hit_pages == k, (
            f"need={need} got={got}: the store handed D {k} pages and D "
            f"claimed {out.kv_hit_pages}. Zero here is boot weg2sb5h's "
            "`WEG2 DRAFT-PRESENCE ... claim=0 mode=trim`, 69 lines out of 69 "
            "trims, and it is what priced 24 served prompts at their whole "
            "extent."
        )
        assert ctrl.draft_cold_spans["weg2-1298"] == (d * PAGE, k * PAGE), (
            "and the uncovered draft rows are NAMED for the #993 zero fill, "
            "never silently claimed"
        )

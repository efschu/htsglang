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
WHOLE prompt.

WHAT ACTUALLY CUT THE HANDBACK -- and it is NOT what fix 1 of this branch
claimed.  Fix 1 asserted the store "does not credit odd numbers: its presence
probe answers on whole ``chunked_prefill_size`` blocks", and floored the pool
grant to 4,096.  The boot's own instrument refutes that, on every line it
emitted (instrument ``#1028B FETCH CAP``, D log, denominator: all 36
component-capped lines of the boot = 12 rids x 3 ranks, rank-unanimous)::

    kv=8150 claimed=4094 lost=4056 caps={MAMBA: 4094, draft-...: 48} keys=8150
    kv=8247 claimed=8190 lost=57   caps={MAMBA: 8190, draft-...: 48} keys=8247

``kv == keys`` on **36/36**: for every truncated read the store credited every
key it was asked about.  The store quantized nothing.  There is no block grid
on that path at all -- the KV prefix is a per-page contiguous scan
(``hicache_storage.py`` ``batch_exists_v2``) -- and the cut came from
``final_pages = min(final_pages, boundary)``, the MAMBA component boundary.
The same lines carry the discriminator (``#1035b``)::

    kv=8247 claimed=8190  mamba anchors_in_range(count, deepest_idx)=(2, 8189)
    kv<8192 claimed=4094  mamba anchors_in_range(count, deepest_idx)=(1, 4093)

so the binding term is the MAMBA ANCHOR STRIDE, and the anchors sit at 0-based
index ``4096k - 3`` -> boundary ``4096k - 2`` (4094, 8190) -- the same ``-2``
as the byte-exact round trips (22,406 of 22,408).  That the stride is also
4,096 is an unstated coupling to P's write-back cadence which
``hicache_storage.py`` explicitly plans to break ("a GRANULARITY problem, fixed
by publishing anchors more often").

So this file pins the mechanism (M) rather than a constant, and GUARDS (G) the
pool grant against being quantized against a guessed one -- fix 1's floor
dropped a whole block for any residual in ``[4096k-2, 4096k)`` and could push
the realised ``lost`` outside the one-chunk bound (#939) that the same line
reports.  See the record block, SECTION 1az.

WHAT REMAINS FIXED HERE (T2, unrefuted): ``resolve_draft_claim``'s ``trim``
branch capped its own claim at ``max(0, min(reprobe(d), d))``, so it could
never return more than ``d`` pages while the ``cold`` branch beside it claims
the whole ``k`` for the same input -- unconditionally, for every input, not
only on this boot's distribution.  MEASURED (instrument ``WEG2
DRAFT-PRESENCE``; denominator: all 123 presence lines of the boot, 3 ranks per
request, rank-unanimous): 69 lines took ``trim`` and **every one of the 69
returned claim=0** -- 66 at ``k=4094 d=47`` and 3 at ``k=2727 d=2671``, the
latter OFF the 4,096 grid, which is how we know the zero is the branch and not
the value.  The 54 ``cold`` lines each claimed their full ``k``.

RED AT THE PARENT ``57fef0ce6e``: the whole (B) family, and ``test_c1``.
``test_g1``/``test_g2`` are green at the parent BY CONSTRUCTION -- they are
regression guards for the reverted floor, not claims about this commit, and
they go red against it.  ``test_f3`` is RED on the sb5h log itself and stays
red until part (C) lands; it is the next boot's acceptance.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU, no collective.  The
MIN all-reduce is simulated by MIN-ing the ranks' captured votes, which is the
arithmetic ``_all_reduce_attn_groups`` performs (same harness shape as
``test_weg2_leg2_store_probe_1290``).
"""

import logging
import os
import re
import subprocess
import tempfile
from types import MethodType, SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from sglang.srt.managers.cache_controller import resolve_draft_claim
from sglang.srt.mem_cache import match_refusal_census as census_mod
from sglang.srt.mem_cache.hicache_collective import HiCacheCollectiveDesyncError
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
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

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
#: The MEASURED mamba anchor boundaries and the 0-based deepest index that
#: produced each, off the 36 `#1028B FETCH CAP` lines that carry a MAMBA cap:
#: `(kv_pages_present, anchor_count_in_range, deepest_idx, claimed)`.
#: 9 lines at the first row, 27 across the rest -- every capped line of the
#: boot, not a sample.
ANCHOR_ROWS = (
    (8247, 2, 8189, 8190),
    (8150, 1, 4093, 4094),
    (8128, 1, 4093, 4094),
    (8100, 1, 4093, 4094),
    (8074, 1, 4093, 4094),
    (8043, 1, 4093, 4094),
    (8016, 1, 4093, 4094),
    (7991, 1, 4093, 4094),
    (7891, 1, 4093, 4094),
)
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
#: The two rows that survived, and the only difference: a SECOND anchor was in
#: range (`anchors_in_range=(2, 8189)`), not that `got` cleared a block edge.
COLD_ROWS = ((16520, 8247, 8190, 48), (18187, 8247, 8190, 48))
#: The one trim row off the 4,096 grid -- proof the zero is not a property of
#: 4094 but of the branch (`WEG2 DRAFT-PRESENCE ... kv_pages=2727
#: draft_pages=2671 claim=0 mode=trim`, 3 lines / 1 rid).
OFF_GRID_TRIM = (2727, 2671)
#: The sb5h D log, the F3 subject.  Absent on the remote desk by design (the
#: evidence tree is not shipped over the link).
SB5H_D_LOG = (
    "/spinning/evidence-665-f1/"
    "boot_weg2_weg2sb5h_57fef0ce6e_0909_103119.D.log"
)
#: THE SAME SUBJECT, SHIPPED.  Part C round 1 -- F3 was written by fix 2 and
#: NEVER EXECUTED as a pytest node: it skipped on the remote (no evidence tree)
#: and no local pytest is allowed while a boot window runs, so its assertion
#: had only ever been computed by hand off the log.  An acceptance test that
#: has never run is the `desk-written-never-executed` class, one level in --
#: so the lines F3 reads are cut into the tree and the test runs EVERYWHERE.
#:
#: THE CUT IS PROVABLY FAITHFUL, not merely small: F3's own walk looks at
#: exactly two kinds of line (the truncation instrument, and the five terminal
#: markers), so the fixture keeps every truncation line plus every terminal
#: line whose rid is one of the truncated ones, and drops nothing the walk
#: could have read.  `test_f3_fixture_is_faithful` re-runs the walk over the
#: FULL log wherever the evidence tree exists and asserts both populations are
#: identical, so the fixture cannot drift into a friendlier subject.
#:
#: THE EXTENSION IS LOAD-BEARING -- ``.txt``, never ``.log``. Measured on the
#: first remote run of this commit: ``.gitignore:62`` is a blanket ``*.log``,
#: so ``git add -A`` skipped the fixture WITHOUT A WORD, the commit shipped a
#: test whose subject was not in the tree, and F3 came back red on the remote
#: with ``FileNotFoundError`` -- one red in the tally, indistinguishable from
#: the red this test is SUPPOSED to produce. It would have been reported as
#: "F3 executes and is red as designed" and it was nothing of the kind.
#: `desk-written-never-executed`, one level in: the test ran and reached the
#: wrong thing. `test_f3_the_fixture_is_actually_in_the_tree` below is the
#: guard, because a comment cannot fail.
SB5H_D_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "sb5h_D_truncation_terminals_1298.txt",
)
#: The five ways a store read can END where a reader can see it.  `X-DEFER`
#: counts as terminal on purpose: a deferred read is one the X gate has
#: accounted for, which is the whole of part (C).
F3_TERMINAL_MARKERS = (
    ("HiCache prefetch success req=", "success"),
    ("#905 PREFETCH-COMPLETE", "complete"),
    ("#1157 PREFETCH REAPED", "reaped"),
    ("#915 PREFETCH REFUSED", "refused"),
    ("WEG2 X-DEFER", "defer"),
)
#: The population floor.  Asserted so a green can never come from an empty
#: walk (the ratchet rule): sb5h carries 37 truncated rids, all `over_bound`.
F3_MIN_TRUNCATED_RIDS = 30


def _f3_walk(path):
    """(truncated rid -> (need, got), rid -> set(terminal names)) over one log.

    ONE walk, used by F3 and by its fidelity check, so the fixture and the
    full log can never be compared through two different readings.

    ``over_bound=true`` ONLY: the invariant part (C) must close is about reads
    the pool cut by more than one chunk (#939), which is the population the X
    gate then prices at the whole prompt.  A cut inside the one-chunk law is
    not a #1298 event and is not counted here -- on sb5h the filter removes
    nothing (111/111 lines are `true`), and saying so is the point: the
    denominator is stated rather than assumed.
    """
    trunc, terminal = {}, {}
    rt = re.compile(
        r"#915 PREFETCH TRUNCATED rid=(\S+) need=(\d+) got=(\d+) "
        r"lost=\d+ chunk=\d+ over_bound=(\S+)"
    )
    rr = re.compile(r"(?:req|rid)=([0-9a-f]{6,})")
    with open(path, errors="replace") as fh:
        for line in fh:
            hit = rt.search(line)
            if hit and hit.group(4) == "true":
                trunc.setdefault(hit.group(1)[:8], (hit.group(2), hit.group(3)))
            for literal, name in F3_TERMINAL_MARKERS:
                if literal in line:
                    rid = rr.search(line)
                    if rid:
                        terminal.setdefault(rid.group(1)[:8], set()).add(name)
    return trunc, terminal
IDENTITY = "0123456789abcdef"
TREE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


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


def _skip_if_store_dir_overridden(store, root):
    if os.path.realpath(store.file_path) != os.path.realpath(root):
        pytest.skip(
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR overrides the store dir"
        )


# ------------------------------------------ (M) the mechanism, on the store
def test_m1_the_store_credits_every_key_it_holds_and_the_anchor_does_the_cutting():
    """THE COUNTER-MEASUREMENT, PINNED SO IT CANNOT BE RE-INVENTED.

    Fix 1 of this branch shipped the claim that the store "does not credit odd
    numbers: its presence probe answers on whole ``chunked_prefill_size``
    blocks", and floored the pool grant to that assumed grid.  The boot said
    otherwise on all 36 of its capped lines (``kv == keys``, 36/36), and so
    does the real store here.

    DRIVEN ROWS, and why these three of the nine: the rows differ in exactly
    two ways, the anchor COUNT in range (1 or 2) and the span.  ``kv=8247``
    is the whole count=2 arm (9 of the 36 lines); ``kv=7891`` and ``kv=8150``
    are the extremes of the count=1 arm (the other 27 lines all sit between
    them with the identical ``deepest_idx=4093``).  The remaining six rows add
    no arm, only file-writes -- their arithmetic is asserted over ALL nine in
    ``test_m2``.  Each driven row: publish ALL ``kv`` KV pages, publish the
    MAMBA anchor at exactly the measured ``deepest_idx``, probe the whole span
    with the real ``TRAILING_PAGES`` transfer the mamba component builds
    (``keys=[node.hash_value[-1]]`` -> trailing 1), and read the two numbers
    the ``#1028B`` line prints:

    * ``kv_uncapped`` -- the KV prefix BEFORE any component cap -- equals the
      number of keys asked, for an odd, non-block span.  No grid, anywhere.
    * ``kv_hit_pages`` -- the cross-pool MIN -- equals the measured ``claimed``,
      and the pool that produced it is MAMBA.

    Anything that floors a read against a store-side block grid is fixing a
    mechanism this test says does not exist.
    """
    driven = [row for row in ANCHOR_ROWS if row[0] in (8247, 8150, 7891)]
    assert len(driven) == 3 and {r[1] for r in driven} == {1, 2}, driven
    for kv, count, deepest, claimed in driven:
        with tempfile.TemporaryDirectory() as root:
            store = _store(root)
            _skip_if_store_dir_overridden(store, root)
            keys = [f"w2-1298-m1-{kv}-{i:07d}" for i in range(kv)]
            blob = torch.arange(8, dtype=torch.uint8)
            for key in keys:
                assert store.set(store._log_key(PoolName.KV, key), blob)
            # the anchors the boot's `#1035b` probe found in range
            anchors = [deepest - j * CHUNK for j in range(count)]
            assert anchors[-1] >= 0 and max(anchors) == deepest
            for idx in anchors:
                assert store.set(store._log_key(PoolName.MAMBA, keys[idx]), blob)
            mamba = PoolTransfer(
                name=PoolName.MAMBA,
                keys=[keys[-1]],  # trailing 1, as hi_mamba_radix_cache builds it
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
            hit = store.batch_exists_v2(keys, [mamba], None)
            assert hit.keys_asked == kv
            assert hit.kv_uncapped == kv, (
                f"kv={kv}: the store credited {hit.kv_uncapped} of {kv} keys "
                "it holds. THE STORE HAS NO BLOCK GRID -- if this ever fails, "
                "the probe changed, not the assumption"
            )
            assert kv % CHUNK != 0, "the boot's spans are all off the grid"
            assert hit.kv_hit_pages == claimed, (
                f"kv={kv}: cross-pool MIN {hit.kv_hit_pages}, measured {claimed}"
            )
            assert (
                int(hit.extra_pool_hit_pages.get(str(PoolName.MAMBA), 0)) == claimed
            ), "and MAMBA is the pool that produced it"


def test_m2_the_anchor_boundary_is_not_the_block_boundary():
    """WHY A ``% 4096`` FLOOR IS THE WRONG QUANTITY, in one arithmetic.

    The measured boundaries are ``4096k - 2`` (4094, 8190), from anchors at
    0-based ``4096k - 3``.  So for a granted span in ``[4096k-2, 4096k)`` the
    store credits block ``k`` while a ``granted - granted % 4096`` floor drops
    the grant to block ``k-1`` -- at ``k=1`` to ZERO, under
    ``prefetch_threshold``, refusing a read the parent performed and credited
    4,094 pages for.  Stated on the numbers rather than argued, and asserted
    in the direction that keeps the boundary honest if either side moves.
    """
    for kv, _count, deepest, claimed in ANCHOR_ROWS:
        assert claimed == deepest + 1, "boundary is the prefix length"
        assert claimed % CHUNK == CHUNK - 2, (
            f"claimed={claimed}: the anchor grid is 4096k-2, not 4096k"
        )
        assert claimed - (claimed % CHUNK) != claimed, (
            "so flooring the credited boundary itself to a block loses it"
        )
    # the interval a block floor destroys, on the boot's own stride
    for k in (1, 2, 3):
        edge = k * CHUNK - 2  # the credited boundary
        for granted in (edge, edge + 1):
            assert granted - (granted % CHUNK) == (k - 1) * CHUNK, (
                f"granted={granted}: a block floor drops to block {k - 1} "
                f"while the store credits {edge}"
            )
    assert (1 * CHUNK - 2) - ((1 * CHUNK - 2) % CHUNK) == 0, (
        "and at the first block it floors to zero, i.e. a refusal"
    )


# ---------------------------------- (G) the grant is NOT quantized: guards
class _FakeHostPool:
    """The two calls ``prefetch_from_storage`` makes, over a free counter."""

    def __init__(self, free):
        self.free = int(free)
        self.allocs = []
        self.released = 0

    def available_size(self):
        return self.free

    def alloc(self, need_size):
        if need_size > self.free:
            return None
        self.free -= need_size
        self.allocs.append(need_size)
        return torch.arange(need_size, dtype=torch.int64)


def _cache_stub(available, peer_votes, page=PAGE, threshold=THRESHOLD, chunk=CHUNK,
                symmetric=True, peer_spans=None):
    """A ``UnifiedRadixCache`` stand-in driving the REAL
    ``prefetch_from_storage``, with a reduce stub that MINs this rank's vote
    against ``peer_votes``.

    #1298 (S1): the stub now reduces EVERY slot the real payload carries, not
    just the length at index 2.  ``peer_spans`` is what the OTHER ranks entered
    the vote with; ``None`` means they entered with the same span as this rank,
    which is the uniform case and the behaviour every pre-#1298 caller relied
    on.  Reducing only index 2 would have left the span slots holding this
    rank's own numbers, i.e. silently modelled a one-rank group on exactly the
    arm whose whole subject is disagreement between ranks.
    """
    pool = _FakeHostPool(available)
    registered = {}

    def _reduce(t, op, label):
        assert label == "prefetch_participation_vote"
        stub.votes.append(int(t[2].item()))
        t[2] = min([int(t[2].item())] + [int(v) for v in peer_votes])
        if t.numel() > 4:
            mine = int(t[3].item())
            spans = [mine] + [
                int(s) for s in (peer_spans if peer_spans is not None else [])
            ]
            t[3] = min(spans)
            t[4] = min(-s for s in spans)
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
        _hicache_prefetch_symmetric=lambda: symmetric,
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


def test_g1_a_residual_is_granted_whole_never_floored_to_a_block():
    """REGRESSION GUARD for the reverted fix-1 floor (F2/F3).

    The residual the pool can carry is granted at exactly ``available``
    (page-floored, and ``page_size`` is 1 here).  The dangerous inputs are the
    ones sitting just below a block edge -- ``4096k - 1`` and ``4096k - 2``,
    the second being the boundary the store actually credits (``test_m2``).  A
    block floor turns those into block ``k-1``, and at ``k=1`` into a refusal
    of a read the parent performs.  Driven over the boot's ten granted spans
    plus the three edges, on the symmetric (#1290) site.
    """
    # why a residual exists at all on this form: the D-phase staging pool is
    # 30,518 rows / limit 27,466 tokens and fits exactly ONE ~22k read, so
    # every sibling of an epoch is served from what the first one left.
    assert POOL_LIMIT < 2 * 22331 <= POOL_ROWS + 22331
    edges = [k * CHUNK - off for k in (1, 2) for off in (1, 2)]
    spans = sorted({row[1] for row in TRIM_ROWS} | {row[1] for row in COLD_ROWS})
    assert len(spans) == 10, "the boot's ten distinct granted spans"
    # EDGES FIRST, deliberately: they are the inputs F2 is about, so a guard
    # that goes red names one of them rather than an ordinary span.
    for got in edges + spans:
        stub = _cache_stub(got, peer_votes=[got, got])
        _issue(stub, "resid", 22331)
        registered = stub.registered.get("resid")
        assert registered is not None, (
            f"got={got}: the read must still register -- a block floor at "
            f"{got - (got % CHUNK)} would drop it under the threshold"
        )
        assert registered.key_len == got, (
            f"got={got}: granted {registered.key_len}. The residual is granted "
            "WHOLE: the store credits every key it holds (test_m1), so rows "
            "above a block edge are not dead weight, and the term that does "
            "cut the claim is the mamba anchor, which this site cannot see."
        )
        assert stub.pool.allocs == [got]


def test_g2_the_realised_loss_is_the_room_that_was_missing_and_nothing_more():
    """REGRESSION GUARD (F4): the #939 one-chunk bound is reported on the
    REALISED loss, so a quantizer that discards rows the pool did hold would
    push a truncation that honoured the bound outside it -- under a counter
    (``host_pool_truncated_tokens``) whose name says the pool had no room.

    On the non-symmetric site, which is the one that speaks the line: for a
    need one row past a whole number of blocks against room one row short of
    it, ``lost`` must be exactly 2 and ``over_bound`` must read false.
    """
    for m in (2, 3):
        need, available = m * CHUNK + 1, m * CHUNK - 1
        stub = _cache_stub(available, peer_votes=[need, need], symmetric=False)
        with _CaptureLines(logging.getLogger(TREE_LOGGER)) as cap:
            delta = _issue(stub, "bound", need)
        assert delta.get("host_pool_truncated") == 1
        assert delta.get("host_pool_truncated_tokens") == need - available == 2, (
            f"need={need} available={available}: the counter must carry the "
            "room that was missing, not rows a quantizer threw away"
        )
        line = [ln for ln in cap.lines if "#915 PREFETCH TRUNCATED" in ln]
        assert len(line) == 1, cap.lines
        assert f"got={available}" in line[0] and "lost=2" in line[0], line[0]
        assert "over_bound=false" in line[0], line[0]


class _CaptureLines(logging.Handler):
    """A handler, not ``caplog``: the census harnesses in this tree bind their
    own root config and ``caplog`` propagation is not reliable under them."""

    def __init__(self, logger):
        super().__init__(level=logging.WARNING)
        self._logger = logger
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        self._prev = self._logger.level
        self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(self)
        return self

    def __exit__(self, *a):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev)
        return False


def test_g3_ranks_still_agree_on_the_unquantized_grant():
    """Ranks never disagree.  Three ranks with DIVERGENT pool room register ONE
    identical length -- the group MIN of three votes -- and each releases only
    its own surplus.  Unchanged by the revert, and stated here because it is
    the property any future quantizer must not break."""
    rooms = [8150, 8247, 12290]
    votes = [min(22331, r) for r in rooms]
    group = min(votes)
    stubs = []
    for me, room in enumerate(rooms):
        peers = [votes[j] for j in range(3) if j != me]
        stub = _cache_stub(room, peer_votes=peers)
        _issue(stub, "uniform", 22331)
        stubs.append(stub)
    assert {s.registered["uniform"].key_len for s in stubs} == {group}
    for s, vote in zip(stubs, votes):
        assert s.votes == [vote], "each rank voted its own allocated length"
        assert s.pool.released == vote - group


# ------------------------------------------------------------ (B) the claim
def _cold_claim(k, d):
    """What the ``cold`` contract yields for the same input: the whole KV
    prefix, with ``[d, k)`` named for the #993 zero fill."""
    return k


def test_b1_the_trim_zero_is_gone_on_the_boots_own_arguments():
    """RED AT THE PARENT.  Arguments read verbatim off ``D L77876`` /
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
    """The arm that already worked stays byte-identical: ``got=8247`` reached a
    second anchor, credited 8,190 and served."""
    for need, got, k, d in COLD_ROWS:
        assert resolve_draft_claim(k, d, CHUNK, lambda _d: 0) == (
            k, d, "cold", (d, k)
        ), f"need={need} got={got}"
    assert resolve_draft_claim(10, 10, CHUNK, lambda d: d) == (10, 10, "full", None)
    assert resolve_draft_claim(10, 12, CHUNK, lambda d: d) == (10, 10, "full", None)


def test_b3_no_reprobe_answer_can_lower_the_claim():
    """The cap ``max(0, min(reprobe(d), d))`` was the mechanism: whatever the
    store answered, the branch could not return more than ``d``, and ``d < k``
    is the branch's own precondition -- so it lost to ``cold`` for EVERY input,
    not only this boot's.  Sweep the whole answer space -- generous, exact,
    short, zero, negative, absurd -- on the boot's own ``(k, d)`` and on the
    one row that is NOT on the 4,096 grid."""
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
def test_f3_the_fixture_is_actually_in_the_tree():
    """F3's subject must be SHIPPED, not merely present on the author's box.

    RED-FIRST AND OBSERVED RED, on the real trap rather than a constructed
    one: with the fixture named ``.log`` it was swallowed by the blanket
    ``*.log`` in ``.gitignore:62``, so the remote had no subject and F3
    returned ``FileNotFoundError`` -- a red that counts the same in a tally as
    the red F3 is designed to produce, and would have been reported as it.
    This test tells the two apart by name.

    Tracked-ness is asserted where git can answer and existence everywhere:
    on the remote desk the worktree is materialised from the PUSHED sha, so
    existence there IS tracked-ness, which is exactly where the trap fired.
    """
    assert os.path.exists(SB5H_D_FIXTURE), (
        f"{SB5H_D_FIXTURE} is missing. If it exists on your box but not here, "
        "it was never committed -- check .gitignore (a blanket *.log ate this "
        "fixture once already) and confirm with `git ls-files`."
    )
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", SB5H_D_FIXTURE],
            cwd=os.path.dirname(SB5H_D_FIXTURE),
            capture_output=True,
        ).returncode
    except (OSError, ValueError):  # no git here: existence above is the check
        tracked = 0
    assert tracked == 0, (
        f"{SB5H_D_FIXTURE} exists but git does not track it -- it will not "
        "reach any other checkout, and F3 will fail there on a missing file "
        "rather than on its invariant (.gitignore:62 is a blanket *.log)"
    )


def test_f3_the_walk_reproduces_the_recorded_defect_exactly():
    """F3, THE CHARACTERISATION HALF -- and why it is not the acceptance.

    THE FIXTURE IS A RECORDING OF A BOOT THAT RAN WITHOUT THE FIX.  No code
    change on this branch can make it greener, and a test that can only ever
    be red is not an acceptance -- it is a permanent red that people learn to
    scroll past.  So the frozen subject gets the assertion it can actually
    carry: the walk reproduces the recorded defect EXACTLY, 34 orphans out of
    37 ``over_bound=true`` truncated rids.

    That pins two things that do matter here: the walk itself works (it is the
    same function the acceptance runs), and the pre-fix number is on the
    record in executable form, so nobody has to trust the prose.

    THE ACCEPTANCE IS
    :func:`test_f3_invariant_holds_on_a_post_fix_log`, which runs the SAME
    walk against a NEW boot's D log.  That is gate 4 of the #1298 boot ticket
    and it is the only place the invariant can honestly go green.
    """
    trunc, terminal = _f3_walk(SB5H_D_FIXTURE)
    assert len(trunc) == 37, (
        f"{len(trunc)} truncated rids in the recording, expected 37 -- the "
        "fixture or the instrument's line format moved"
    )
    orphans = sorted(r for r in trunc if not terminal.get(r))
    assert len(orphans) == 34, (
        f"the recorded defect is 34 orphans of 37; this walk found "
        f"{len(orphans)}. The recording cannot change, so the walk did."
    )


def test_f3_invariant_holds_on_a_post_fix_log():
    """F3, THE ACCEPTANCE -- gate 4 of the #1298 boot ticket.

    Every ``over_bound=true`` truncated rid must reach exactly one terminal
    line: a completion, a reap, a refusal, or a defer.  ``WEG2 X-DEFER`` counts
    as terminal, because a deferred read is one the X gate has accounted for,
    which is the whole of part (C).

    On boot weg2sb5h, 34 of 37 had none of the five -- issued, cut, and gone
    without a word about a second later, which is why ``WEG2 X-DEFER`` is 0 in
    109,471 lines while 24 requests were priced at their whole prompt.

    Point it at the next boot's D log::

        WEG2_F3_LOG=/spinning/evidence-665-f1/boot_..._D.log \
          pytest test/registered/unit/weg2/test_weg2_store_grid_claim_1298.py \
          -k test_f3_invariant

    It SKIPS without that variable rather than silently passing on an empty
    walk, and it asserts its own population floor for the same reason.
    """
    path = os.environ.get("WEG2_F3_LOG")
    if not path:
        pytest.skip(
            "set WEG2_F3_LOG to a post-fix boot's D log -- the shipped fixture "
            "is a PRE-fix recording and is pinned by "
            "test_f3_the_walk_reproduces_the_recorded_defect_exactly instead"
        )
    trunc, terminal = _f3_walk(path)
    assert len(trunc) >= F3_MIN_TRUNCATED_RIDS, (
        f"only {len(trunc)} truncated rids in {path} -- below the "
        f"{F3_MIN_TRUNCATED_RIDS} floor. A green from an empty walk is not a "
        "green: either the boot never filled the staging pool (in which case "
        "this gate has no population and must not be reported as passed) or "
        "the instrument's line format moved."
    )
    orphans = sorted(r for r in trunc if not terminal.get(r))
    assert not orphans, (
        f"{len(orphans)} of {len(trunc)} truncated reads have no completion, "
        f"no reap, no refusal and no defer line: {orphans[:6]}... -- the X "
        "gate sees neither a pending read nor a refused one, so it prices the "
        "whole prompt (#1298 part C)"
    )


@pytest.mark.skipif(
    not os.path.exists(SB5H_D_LOG),
    reason="the sb5h evidence tree is not shipped to the remote desk",
)
def test_f3_fixture_is_faithful():
    """The shipped cut answers F3 exactly as the 23 MB original does.

    Without this, shrinking the subject to make it shippable is also a way to
    make it friendlier, and nobody would see it.  Both populations are
    compared -- the truncated set AND the orphan set -- because a fixture that
    dropped a terminal line would make F3 MORE red and one that dropped a
    truncation line would make it LESS, and only checking both catches both.
    """
    full_t, full_term = _f3_walk(SB5H_D_LOG)
    fix_t, fix_term = _f3_walk(SB5H_D_FIXTURE)
    assert set(fix_t) == set(full_t), (
        "the fixture's truncated rid set differs from the full log's: "
        f"missing={sorted(set(full_t) - set(fix_t))[:5]} "
        f"extra={sorted(set(fix_t) - set(full_t))[:5]}"
    )
    full_orphans = sorted(r for r in full_t if not full_term.get(r))
    fix_orphans = sorted(r for r in fix_t if not fix_term.get(r))
    assert fix_orphans == full_orphans, (
        f"the fixture answers F3 with {len(fix_orphans)} orphans where the "
        f"full log answers {len(full_orphans)}: a terminal line was dropped "
        "or invented by the cut"
    )


# --------------------------- the real D-side chain, on a real disk store
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


def test_c1_the_real_d_side_chain_from_a_published_prefix_is_not_zero():
    """RED AT THE PARENT: the D-side resolver chain, on a real store.

    A REAL ``HiCacheFile`` on disk.  The KV prefix is published through the
    WRITER's key funnel (``set(_log_key(pool, key))``, what ``_write_page``
    calls) and probed through the READER's (``batch_exists_v2`` ->
    ``_get_component_key``) -- the two funnels #1295 proved share the canonical
    suffix -- and D's real ``_apply_draft_claim`` resolves the claim.

    SCOPE, STATED HONESTLY: this drives the chain from a KV prefix of ``k``
    pages to the claim.  It reproduces ``k=4094`` by publishing 4,094 pages,
    i.e. by WRITE PROGRESS, and passes NO mamba transfer -- so the anchor cap,
    the term that actually produced 4,094 on the boot (``test_m1``), does not
    fire here.  It does not need to: the resolver is a function of ``(k, d)``
    and does not care where ``k`` came from, and the claim it returned for
    this ``(k, d)`` was zero.  ``test_m1`` covers the cap; this covers what D
    then does with it.
    """
    need, got, k, d = 22331, 8016, 4094, 47
    with tempfile.TemporaryDirectory() as root:
        store = _store(root)
        _skip_if_store_dir_overridden(store, root)
        keys = [f"weg2-1298-page{i:07d}" for i in range(got)]
        blob = torch.arange(8, dtype=torch.uint8)
        # --- publish the KV prefix, then the shorter draft prefix.
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
            "this test's premise (the write landed) is broken, not its claim"
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


# ================================================================= #1298 (S1)
# The group-agreed truncation fact: the pre-vote SPAN rides the participation
# vote, so `group_len < span` is computed from two REDUCED values on every
# rank instead of from one reduced and one rank-local one.
#
# RED ON `fe867c601a`: the payload is three slots wide there, so slots 3/4 do
# not exist and the trim condition still reads this rank's own
# `len(prefetch_key)`.
def test_s1_the_payload_carries_the_span_both_ways_beside_the_length():
    """The contract as a shape: MIN yields the low end directly and the high
    end through the negation, exactly as the tag already does in slots 0/1."""
    seen = {}

    def _capture(t, op, label):
        assert label == "prefetch_participation_vote"
        seen["numel"] = int(t.numel())
        seen["span"] = int(t[3].item())
        seen["neg_span"] = int(t[4].item())
        return t

    tree = _cache_stub(available=100000, peer_votes=[])
    tree._all_reduce_attn_groups = _capture
    _issue(tree, "rid-shape", 5000)
    assert seen["numel"] == 5, f"payload is {seen['numel']} slots, expected 5"
    assert seen["span"] == 5000
    assert seen["neg_span"] == -5000, (
        "slot 4 must be the NEGATED span so one MIN yields both ends"
    )


def test_s1_one_short_rank_cuts_the_group_and_the_rank_with_room_agrees():
    """THE PHYSICS, and why a shortfall BIT would have been wrong arithmetic.

    ``group_len`` is a MIN, so ONE short rank drags every rank's registered
    span down: the read IS cut, for the group.  The group fact is therefore
    the OR over ranks ("someone was short"), which is exactly what
    ``group_len < span`` computes -- not the AND that a MIN over a raw
    shortfall bit would have produced.

    THIS rank has room for all 8,000 and is not short; a peer voted 4,096.
    The rank with room must still register the cut span and must count the
    GROUP truncation, or the fact would be rank-local and the mark could
    diverge -- which is the hazard the whole arm exists to avoid.
    """
    tree = _cache_stub(available=100000, peer_votes=[4096])
    delta = _issue(tree, "rid-one-short", 8000)
    assert tree.registered["rid-one-short"].key_len == 4096
    assert delta.get("host_pool_truncated_group") == 1, delta
    assert delta.get("host_pool_truncated") == 1, (
        "the group key must stand BESIDE the existing one, not replace it"
    )


def test_s1_no_shortfall_anywhere_is_not_a_truncation():
    """The negative arm, so the counter above is not simply always bumped."""
    tree = _cache_stub(available=100000, peer_votes=[8000])
    delta = _issue(tree, "rid-whole", 8000)
    assert tree.registered["rid-whole"].key_len == 8000
    assert "host_pool_truncated_group" not in delta, delta
    assert "host_pool_truncated" not in delta, delta


def test_s1_a_span_split_between_ranks_is_a_named_stop():
    """THE CAN-FAIL AGAINST THE HAZARD, and why the mark is safe to hang.

    Two ranks agree a LENGTH and entered with different SPANS: each would trim
    its own key to ``group_len`` and register a different token range under one
    rid.  Lengths agree, contents do not, and every length-based reduce
    downstream would report agreement.  The group stops by name rather than
    reconciling a span nobody voted for.
    """
    tree = _cache_stub(available=100000, peer_votes=[4096], peer_spans=[7000])
    with pytest.raises(HiCacheCollectiveDesyncError) as exc:
        _issue(tree, "rid-split", 8000)
    msg = str(exc.value)
    assert "W55 Weg2PrefetchSpanSplit" in msg, msg
    assert "min=7000" in msg and "max=8000" in msg, msg
    assert "rid-split" not in tree.registered, (
        "a request must never register on a split span"
    )


def test_s1_a_declining_peer_still_takes_the_existing_vote_negative_exit():
    """WHY THE STOP SITS BELOW THE THRESHOLD RETURN.

    An INELIGIBLE rank enters this vote carrying nothing (``scheduler.py``:
    "enter the vote carrying nothing"), so its span is 0.  A span check placed
    ABOVE the threshold return would fire on that -- the most ordinary
    condition in the system -- instead of on a real split.  Below the return,
    ``group_len >= threshold > 0`` proves every rank was eligible AND
    allocated, so only real spans are ever compared.

    Here the peer declines (votes 0, span 0).  The exit must be the EXISTING
    ``vote_negative``, never the new STOP.
    """
    tree = _cache_stub(available=100000, peer_votes=[0], peer_spans=[0])
    delta = _issue(tree, "rid-inelig", 8000)
    assert "rid-inelig" not in tree.registered
    assert delta.get("vote_negative") == 1, delta
    assert "host_pool_truncated_group" not in delta, delta

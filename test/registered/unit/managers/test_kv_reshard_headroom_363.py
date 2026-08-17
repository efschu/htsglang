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
"""#363 defect 2: the reshard cutover must not allocate blind.

THE DEFECT, as measured on metal in the ACT window (2026-08-14).

``kv_reshard.py`` ``_exchange`` allocated its receive buffers with
``torch.empty`` and no headroom check. It died twice, deterministically:

    616 MiB requested, 550 MiB free   (under load)
    758 MiB requested, 256 MiB free   (DRAINED -- worse, not better)

then ``SIGQUIT`` and the whole TP group went down. In BOTH runs the corridor
law broke SEVEN SECONDS BEFORE the traceback (gpu2 to 229 MiB, then 463), which
is the direction that matters: the allocation drove free memory under the
floor, free memory did not drift into the allocation.

``fit_check`` existed and PASSED both times. It answers a different question --
does the TARGET VECTOR fit the backed pool -- and nothing priced the TRANSIENT
the move allocates on top of the resident pool.

THE FIX, and why it is where it is.

The guard is folded into ``on_round``'s group-wide MIN reduction alongside
``fit_ok``, NOT into the allocation site. That placement is the whole design:
``_exchange`` runs on every rank in the same round, so a rank that aborted
locally while its peers proceeded into the exchange would leave them posting
sends to a rank that will never receive -- "the #94/#194/#259 hang, or worse, a
silent mixed-ownership pool", in the module's own words. Every rank prices the
move, the group takes the MIN, and either all of them move or none of them
allocates a byte.

The refusal is CLEAN: the request stays on the incumbent layout, the move is
disarmed, and the arithmetic is named in the log. Disarm rather than hold,
because the drained run needed MORE memory than the loaded one -- this is not a
transient that a silent retry loop grows out of.
"""

from __future__ import annotations

import threading
import unittest

import torch

from sglang.srt.layers.dcp.reshard_plan import (
    owner_of,
    reshard_ceiling_rows,
    rows_of,
)
from sglang.srt.managers.kv_reshard import (
    CORRIDOR_FLOOR_MIB,
    KvPoolView,
    KvReshardRuntime,
)

MIB = 1024 * 1024
FLOOR = CORRIDOR_FLOOR_MIB * MIB

VECTORS = [(7, 3, 3), (2, 11, 10)]


# ---------------------------------------------------------------------------
# A simulated three-rank fleet, one thread per rank.
#
# Inherited from test/registered/scheduler/test_kv_reshard.py: the same
# barrier-MIN consensus channel and mailbox exchange, so a rank that never
# arrives raises in every waiting thread instead of hanging. Extended with two
# things this ticket needs: per-rank free-memory injection, and an exchange
# that RECORDS whether it was ever entered.
# ---------------------------------------------------------------------------


class _BarrierMinChannel:
    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._slots = [None] * n
        self._result = None

    def channel_for(self, rank):
        def _reduce(vals):
            self._slots[rank] = list(vals)
            if self._barrier.wait() == 0:
                self._result = [min(col) for col in zip(*self._slots)]
            self._barrier.wait()
            return list(self._result)

        return _reduce


class _SpyExchange:
    """The mailbox channel, plus a record of every entry and every byte.

    ``entries`` is the assertion target for "none allocates": the receive
    buffers are allocated INSIDE ``_exchange``, so a refusal that works has
    an empty ``entries`` list on every rank.
    """

    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._mail = {}
        self._lock = threading.Lock()
        self.entries = []

    def exchange_for(self, rank):
        def _exchange(outgoing, incoming_nbytes):
            with self._lock:
                self.entries.append(
                    {"rank": rank, "recv_bytes": sum(incoming_nbytes.values())}
                )
            for peer, payload in outgoing.items():
                self._mail[(rank, peer)] = payload.clone()
            self._barrier.wait()
            received = {}
            for peer in incoming_nbytes:
                payload = self._mail.get((peer, rank))
                received[peer] = (
                    payload
                    if payload is not None
                    else torch.empty(0, dtype=torch.uint8)
                )
            self._barrier.wait()
            return received

        return _exchange


def _make_pools(n_ranks, vectors, num_slots, layers=2, heads=2, dim=4, seed=3):
    torch.manual_seed(seed)
    old = vectors[0]
    rows_needed = [reshard_ceiling_rows(num_slots, vectors, r) for r in range(n_ranks)]
    slots = torch.arange(num_slots, dtype=torch.int64)
    live = slots[torch.randperm(num_slots)[: int(num_slots * 0.8)]].sort().values
    gk = [
        torch.randn(num_slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)
    ]
    gv = [
        torch.randn(num_slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)
    ]
    pools, views = [], []
    owners = owner_of(live, old)
    for r in range(n_ranks):
        kb = [
            torch.zeros(rows_needed[r], heads, dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ]
        vb = [
            torch.zeros(rows_needed[r], heads, dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ]
        mine = live[owners == r]
        rows = rows_of(mine, old, r)
        for layer in range(layers):
            kb[layer][rows] = gk[layer][mine]
            vb[layer][rows] = gv[layer][mine]
        pools.append((kb, vb))
        views.append(KvPoolView(kb, vb))
    return views, pools, (gk, gv), live


def _snapshot(pools):
    return [([k.clone() for k in kb], [v.clone() for v in vb]) for kb, vb in pools]


def _pools_unchanged(pools, snap):
    for (kb, vb), (skb, svb) in zip(pools, snap):
        for a, b in zip(kb, skb):
            if not torch.equal(a, b):
                return False
        for a, b in zip(vb, svb):
            if not torch.equal(a, b):
                return False
    return True


class _Fleet:
    """Three ranks, one thread each, with injected per-rank free memory."""

    def __init__(self, free_bytes, num_slots=600, rounds=8, interval=2):
        self.n = len(free_bytes)
        self.views, self.pools, self.ref, self.live = _make_pools(
            self.n, VECTORS, num_slots
        )
        self.channel = _BarrierMinChannel(self.n)
        self.spy = _SpyExchange(self.n)
        self.cutovers = [[] for _ in range(self.n)]
        self.runtimes = []
        for r in range(self.n):
            fb = free_bytes[r]
            self.runtimes.append(
                KvReshardRuntime(
                    dcp_size=self.n,
                    dcp_rank=r,
                    allowed_vectors=VECTORS,
                    current_vector=VECTORS[0],
                    consensus_interval=interval,
                    collective_min=self.channel.channel_for(r),
                    exchange=self.spy.exchange_for(r),
                    pool_view=self.views[r],
                    live_slots_fn=lambda live=self.live: live,
                    ready_fn=lambda: True,
                    cutover_fn=lambda vec, r=r: self.cutovers[r].append(tuple(vec)),
                    free_bytes_fn=(
                        None
                        if fb is None
                        else (lambda fb=fb: fb())
                        if callable(fb)
                        else (lambda fb=fb: fb)
                    ),
                )
            )
        self.rounds = rounds
        self.exceptions = [None] * self.n

    def run(self, target=VECTORS[1]):
        snap = _snapshot(self.pools)

        def _worker(r):
            try:
                self.runtimes[r].arm(target, source=f"test-rank{r}")
                for _ in range(self.rounds):
                    self.runtimes[r].on_round()
            except BaseException as e:  # noqa: BLE001 -- the assertion target
                self.exceptions[r] = e

        threads = [threading.Thread(target=_worker, args=(r,)) for r in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        alive = [t for t in threads if t.is_alive()]
        if alive:
            raise AssertionError(
                f"{len(alive)} rank threads still alive after 30 s -- a reshard "
                f"hang, which is exactly what a per-rank abort would produce"
            )
        self.snap = snap
        return self


def _plenty():
    """Free memory that comfortably clears the floor and any transient here."""
    return 8 * 1024 * MIB


def _short():
    """A card sitting EXACTLY on the corridor floor.

    Nothing above 1024 MiB is spendable, so any transient at all -- and this
    fleet's is only kilobytes -- puts the margin negative. Deliberately not
    "below the floor": the interesting case is the card that is legal right
    now and would be made illegal by the move, which is what happened on
    metal seven seconds before the OOM."""
    return FLOOR


# ---------------------------------------------------------------------------
# 1. The guard can ADMIT. Without this arm every refusal below is vacuous.
# ---------------------------------------------------------------------------


class TestTheGuardCanAdmit(unittest.TestCase):
    def test_a_move_with_headroom_still_completes(self):
        f = _Fleet([_plenty()] * 3).run()
        self.assertEqual([e for e in f.exceptions if e], [])
        for r in range(3):
            self.assertEqual(f.cutovers[r], [VECTORS[1]], f"rank {r} did not cut over")
            self.assertEqual(f.runtimes[r].completed, 1)
            self.assertEqual(f.runtimes[r].refused_headroom, 0)
        self.assertTrue(f.spy.entries, "the exchange was never entered")

    def test_admitting_leaves_the_pool_correct(self):
        """The guard must not disturb the move it admits: byte identity under
        the NEW owner rule still holds."""
        f = _Fleet([_plenty()] * 3).run()
        gk, gv = f.ref
        owners = owner_of(f.live, VECTORS[1])
        for r, (kb, vb) in enumerate(f.pools):
            mine = f.live[owners == r]
            rows = rows_of(mine, VECTORS[1], r)
            for layer in range(len(kb)):
                self.assertTrue(torch.equal(kb[layer][rows], gk[layer][mine]))
                self.assertTrue(torch.equal(vb[layer][rows], gv[layer][mine]))

    def test_an_unwired_guard_is_inert(self):
        """free_bytes_fn=None is the simulated-fleet path and must not refuse."""
        f = _Fleet([None] * 3).run()
        self.assertEqual([e for e in f.exceptions if e], [])
        self.assertEqual(f.runtimes[0].completed, 1)


# ---------------------------------------------------------------------------
# 2. THE DESYNC FALSIFIER -- the reason the guard is in the reduction
# ---------------------------------------------------------------------------


class TestOneShortRankRefusesTheWholeGroup(unittest.TestCase):
    """One rank short -> ALL ranks refuse, NONE allocates, none diverges.

    This is the arm that distinguishes the shipped design from the obvious
    one. A guard at the allocation site would let ranks 0 and 2 enter
    ``_exchange`` and post sends while rank 1 aborted: `spy.entries` would be
    non-empty and the barrier would break. Here the veto travels through the
    consensus reduction, so it arrives BEFORE any rank allocates."""

    def _one_short(self):
        return _Fleet([_plenty(), _short(), _plenty()]).run()

    def test_no_rank_allocates(self):
        f = self._one_short()
        self.assertEqual(
            f.spy.entries,
            [],
            "the exchange was entered despite the group refusing -- some rank "
            "allocated its receive buffers, which is the OOM this guard exists "
            "to prevent",
        )

    def test_every_rank_refuses_not_just_the_short_one(self):
        f = self._one_short()
        for r in range(3):
            self.assertEqual(
                f.runtimes[r].refused_headroom,
                1,
                f"rank {r} did not refuse; a partial refusal IS the desync",
            )

    def test_no_rank_cuts_over(self):
        f = self._one_short()
        for r in range(3):
            self.assertEqual(f.cutovers[r], [], f"rank {r} cut over unilaterally")

    def test_no_rank_raises_and_no_rank_hangs(self):
        """A refusal is an outcome, not an error. The thread-join timeout in
        _Fleet.run is the hang detector."""
        f = self._one_short()
        self.assertEqual([repr(e) for e in f.exceptions if e], [])

    def test_the_pool_is_byte_identical_afterwards(self):
        f = self._one_short()
        self.assertTrue(
            _pools_unchanged(f.pools, f.snap),
            "a refused move still touched the pool",
        )

    def test_the_group_stays_on_the_incumbent_vector(self):
        f = self._one_short()
        for r in range(3):
            self.assertEqual(f.runtimes[r]._current, VECTORS[0])
            self.assertEqual(f.runtimes[r].completed, 0)

    def test_the_refusal_is_terminal_not_a_retry_loop(self):
        """8 rounds at interval 2 = 4 boundaries. Exactly ONE refusal: the
        move is disarmed, not re-refused every boundary. The drained metal run
        needed MORE memory than the loaded one, so retrying is not a strategy."""
        f = self._one_short()
        for r in range(3):
            self.assertEqual(f.runtimes[r].refused_headroom, 1)
            self.assertIsNone(f.runtimes[r]._pending)

    def test_it_does_not_matter_which_rank_is_short(self):
        for short in range(3):
            free = [_plenty()] * 3
            free[short] = _short()
            f = _Fleet(free).run()
            self.assertEqual(f.spy.entries, [], f"short rank {short} let others in")
            self.assertEqual(
                [f.runtimes[r].refused_headroom for r in range(3)], [1, 1, 1]
            )


# ---------------------------------------------------------------------------
# 3. The arithmetic: the corridor floor is subtracted BEFORE the move is priced
# ---------------------------------------------------------------------------


class TestTheCorridorFloorIsNotSpendable(unittest.TestCase):
    def _peak_of(self, fleet, rank=0):
        """What the guard priced on that rank, in bytes."""
        return fleet.runtimes[rank].last_headroom["peak_bytes"]

    def test_enough_for_the_move_but_not_enough_to_keep_1024_is_a_refusal(self):
        """The metal shape exactly: the allocation succeeds and the corridor
        breaks. Free = floor + peak - 1 MiB, i.e. one MiB short of legal."""
        probe = _Fleet([_plenty()] * 3).run()
        peak = self._peak_of(probe)
        self.assertGreater(peak, 0, "nothing was priced")
        f = _Fleet([FLOOR + peak - MIB] * 3).run()
        self.assertEqual([f.runtimes[r].refused_headroom for r in range(3)], [1, 1, 1])
        self.assertEqual(f.spy.entries, [])

    def test_exactly_enough_is_admitted(self):
        """The guard refuses what cannot fit, not what is merely tight. A
        margin of exactly zero is legal -- the corridor floor is preserved."""
        probe = _Fleet([_plenty()] * 3).run()
        peak = self._peak_of(probe)
        f = _Fleet([FLOOR + peak] * 3).run()
        self.assertEqual([f.runtimes[r].refused_headroom for r in range(3)], [0, 0, 0])
        self.assertEqual(f.runtimes[0].completed, 1)

    def test_the_verdict_records_its_terms(self):
        f = _Fleet([_plenty()] * 3).run()
        d = f.runtimes[0].last_headroom
        self.assertEqual(d["corridor_floor_bytes"], FLOOR)
        self.assertEqual(
            d["margin_bytes"],
            d["free_bytes"] - d["corridor_floor_bytes"] - d["peak_bytes"],
        )
        for term in ("staged", "packed", "largest_peer_pack", "recv", "peak"):
            self.assertIn(term, d["terms"])

    def test_the_priced_recv_matches_what_the_exchange_is_asked_for(self):
        """A guard that prices a different move than the one it admits is not
        a guard. The `recv` term must equal the bytes _exchange then receives."""
        f = _Fleet([_plenty()] * 3).run()
        by_rank = {e["rank"]: e["recv_bytes"] for e in f.spy.entries}
        for r in range(3):
            self.assertEqual(
                f.runtimes[r].last_headroom["terms"]["recv"],
                by_rank[r],
                f"rank {r}: priced recv != requested recv",
            )

    def test_the_peak_covers_the_receive_buffers_with_room_for_the_pack(self):
        """`recv` alone was the allocation that raised on metal; the pack
        phase holds its own copies at the same instant, so the peak must be
        strictly larger than recv whenever anything moves."""
        f = _Fleet([_plenty()] * 3).run()
        for r in range(3):
            t = f.runtimes[r].last_headroom["terms"]
            self.assertGreaterEqual(t["peak"], t["recv"] + t["staged"])


# ---------------------------------------------------------------------------
# 4. Fail-closed, and the refusal says what happened
# ---------------------------------------------------------------------------


class TestFailClosedAndNamed(unittest.TestCase):
    def test_a_rank_that_cannot_read_free_memory_refuses_for_the_group(self):
        """Refusing costs a flip. Guessing cost the server."""

        def _blows_up():
            raise RuntimeError("NVML is not talking to us")

        f = _Fleet([_plenty(), _blows_up, _plenty()]).run()
        self.assertEqual(f.spy.entries, [])
        self.assertEqual([f.runtimes[r].refused_headroom for r in range(3)], [1, 1, 1])
        self.assertEqual([e for e in f.exceptions if e], [])
        self.assertIn("NVML is not talking to us", f.runtimes[1].last_headroom["error"])

    def test_the_refusal_names_the_numbers(self):
        with self.assertLogs("sglang.srt.managers.kv_reshard", level="WARNING") as cm:
            _Fleet([_plenty(), _short(), _plenty()]).run()
        refusals = [m for m in cm.output if "REFUSED for headroom" in m]
        self.assertTrue(refusals, "the refusal was not logged")
        short = [m for m in refusals if "margin" in m]
        self.assertTrue(short, "no rank showed its arithmetic")
        text = "\n".join(refusals)
        for token in ("corridor floor", "incumbent vector", "NO rank allocates"):
            self.assertIn(token, text)

    def test_the_metal_numbers_reproduce_as_arithmetic(self):
        """The two recorded failures, replayed through the verdict function.

        616 MiB needed / 550 MiB free (under load) and 758 / 256 (drained).
        Both must refuse -- and both would have been ADMITTED by a check that
        only asked whether the allocation itself fits, which is the check that
        did not exist."""
        for need_mib, free_mib in ((616, 550), (758, 256)):
            rt = _Fleet([_plenty()] * 3).runtimes[0]
            plan = type("P", (), {"terms": {"peak": need_mib * MIB}})()
            rt._free_bytes_fn = lambda free_mib=free_mib: free_mib * MIB
            ok, d = rt._headroom_check(plan)
            self.assertEqual(ok, 0, f"{need_mib} MiB into {free_mib} MiB was admitted")
            self.assertEqual(
                d["margin_bytes"], (free_mib - CORRIDOR_FLOOR_MIB - need_mib) * MIB
            )

    def test_a_move_that_fits_but_eats_the_reserve_is_still_refused(self):
        """The distinction the metal runs could not show, because on metal
        there was no check of any kind.

        A guard that only asked "does the allocation fit in free memory"
        would ADMIT 616 MiB into 700 MiB free and leave the card at 84 MiB --
        legal as an allocation, a corridor breach as an outcome. Subtracting
        the floor FIRST is what makes the two different."""
        need_mib, free_mib = 616, 700
        rt = _Fleet([_plenty()] * 3).runtimes[0]
        plan = type("P", (), {"terms": {"peak": need_mib * MIB}})()
        rt._free_bytes_fn = lambda: free_mib * MIB
        self.assertGreater(free_mib, need_mib, "the allocation itself fits")
        ok, d = rt._headroom_check(plan)
        self.assertEqual(ok, 0, "a move that fits but breaks the corridor was admitted")
        self.assertEqual(d["margin_bytes"], (700 - 1024 - 616) * MIB)


# ---------------------------------------------------------------------------
# 5. The production path is wired
# ---------------------------------------------------------------------------


class TestTheProductionWiringSuppliesTheGuard(unittest.TestCase):
    def test_build_kv_reshard_runtime_passes_free_bytes_fn(self):
        """A wiring regression must be a red test, not a silent unguarded
        boot: `free_bytes_fn=None` is inert by design."""
        import inspect

        from sglang.srt.managers import kv_reshard

        src = inspect.getsource(kv_reshard.build_kv_reshard_runtime)
        self.assertIn("free_bytes_fn=", src)

    def test_the_free_reader_resolves_by_uuid_not_by_index(self):
        """A narrowed CUDA_VISIBLE_DEVICES renumbers indices; UUIDs survive."""
        import inspect

        from sglang.srt.managers import kv_reshard

        src = inspect.getsource(kv_reshard._free_bytes_fn_for)
        self.assertIn("nvmlDeviceGetUUID", src)
        self.assertIn("nvmlDeviceGetMemoryInfo", src)
        self.assertNotIn("nvmlDeviceGetHandleByIndex(device)", src)


if __name__ == "__main__":
    unittest.main()

"""The #757 disarm drain is a SNAPSHOT, not a barrier, and #787 is the gap
this leaves open (a live wire, not a mock -- see WHY THIS TEST IS THREE REAL
PROCESSES OVER REAL GLOO below).

THE METAL FACT THIS ENCODES. A boot died 60 s after health with, on PP1:

    RuntimeError: #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=0
    seq=117 rows=512 arrived while this rank is on mb_id=2. It belongs to
    a pass this rank did not run -- ...

raised from the receive-side guard at scheduler_pp_mixin.py:2817-2835,
reached via `_pp_recv_proxy_tensors` (:2740-2835) -> `_pp_recv_typed_dict`
(:2681-2721). PP0/PP2 then closed their gloo connections downstream.

WHAT IT IS NOT. The #757 guard is CORRECT and stays as the backstop -- this
file does not touch it, and does not argue it should fire less. #757 fixed
the case where the leftover is already on the wire (or already counted as
posted) at the moment the disarm-time drain runs: `pp_flip_drain_leftover_
dicts` (scheduler_pp_mixin.py:1597-1688) receives it, sees its stamp does
not match the pass this rank resumes on, and drops it -- the guard never
fires. `test_pp_flip_leftover_proxy_757.py` covers exactly that, and its
`test_leftover_is_drained_at_disarm` stays green under this bug too, because
that is not what #787 is.

WHAT IT IS. The drain decides "finished" from a SNAPSHOT of a cross-process
counter, taken exactly once:

    posted = counters.sent(CHAN_DICT, upstream)
    if counters.local_consumed(CHAN_DICT) >= posted:
        break                                              # :1648-1650

and that sweep fires EXACTLY ONCE, at the falling edge of
`_pp_flip_pass_tick` (:1244-1247). The upstream publishes its sent-counter
only AFTER posting the isend (`phase_flip_counters.py:187-198`
`bump_sent`), and each rank disarms on its OWN clock via
`_abandon_no_quorum` and `_abandon_unjoined_flip`
(`phase_flip_runtime.py`) -- neither used to do a collective or a channel
re-check. So: if the upstream posts its stale proxy STRICTLY AFTER the
downstream's one-shot sweep already ran and returned (because at that
instant `posted` legitimately read zero -- nothing had been sent yet), the
message was never drained. It would reach the ordinary blocking receive
later and only the #631 guard would catch it, exactly as the specimen
shows. `seq` (`stamp[1]`, built in `_pp_proxy_stamp` at :2723-2738) is a
per-rank monotone counter, never reset and never compared by any drain
logic -- only `mb_id` (`stamp[0]`) is compared, at :345, :348, :1668,
:2818 -- so it cannot be the discriminator of a fix; it is carried here
only because the specimen carries it.

THE FIX, IN TWO HALVES, NEITHER SUFFICIENT ALONE.
  1. RECEIVER-SIDE: `pp_flip_drain_leftover_dicts` no longer treats
     `local_consumed >= posted` as an instant break. It opens a BOUNDED
     settle window (`DRAIN_SETTLE_BUDGET_S` / `DRAIN_SETTLE_STEP_S`,
     scheduler_pp_mixin.py, named constants with a not-a-planner-quantity
     rationale comment) and re-polls the SAME local SHM counter a few more
     times before actually giving up -- NEVER a timing-out gloo
     `Work.wait()`, which would destroy the sender/receiver pairing on the
     wire. This closes the gap for a message that IS sent within the
     window but was not yet counted at the moment of the one-shot read.
  2. SENDER-SIDE: `_abandon_no_quorum` and `_abandon_unjoined_flip`
     (phase_flip_runtime.py) now call a `flush_pending_sends_fn` hook
     (`pp_flip_flush_pending_dict_sends` in scheduler_pp_mixin.py)
     BEFORE clearing local flip state, reaping/counting anything this
     rank already posted on CHAN_DICT so a peer's settle window is not
     racing against bookkeeping this rank could have closed out itself.
  Half 1 alone cannot see a send that has not been issued yet -- only one
  that has been issued but not yet counted. That is why half 2 exists,
  and why the can-fail case below neuters half 2's effect (by letting a
  stale send land beyond the settle budget, the harness-level analogue of
  an absent sender-side ordering guarantee) and confirms the #631 guard
  still fires with the real specimen text in that case.

WHY THIS TEST IS THREE REAL PROCESSES OVER REAL GLOO. #787, like #757
before it, is an ordering fact about when a real isend lands relative to a
real drain call -- a mocked group has no wire for a message to arrive on
"too late". The transport below is the same thin real-gloo adapter
`test_pp_flip_leftover_proxy_757.py` uses; every line of logic under test
(`_pp_recv_typed_dict`, the #631 guard, `pp_flip_drain_leftover_dicts`) is
the SHIPPED code, bound to a holder the same way (the 630 pattern). The one
addition here is a barrier FILE between the two peer processes: the victim
stamps it the instant BEFORE it enters its single (now settle-aware) drain
call, and the upstream waits on it before deciding when to send -- the
whole point of #787 is a specific relative order between "victim's drain
call has started polling" and "upstream's stale send lands", a fact this
test must force, not hope for from scheduler jitter. The upstream's counter
file (`sent_counter`, written synchronously right after each real isend,
mirroring `bump_sent`'s synchronicity with the send call in
`_pp_send_dict_to_next_stage`) is what gives the settle window under test
something live to observe -- a fixed, unchanging stub cannot exercise a
loop whose entire job is to notice a value changing while it waits.

COLOUR CONVENTION, AND WHY IT CHANGED. An earlier revision of this file made
its primary case assert the BUGGY behaviour (the guard firing) as the thing
under test -- green today, and it would have gone RED the day #787 is fixed,
which is backwards for a red-first falsifier: a falsifier must fail while the
defect is live and pass once it is gone, not the other way around. The
primary case below asserts the INVARIANT the fix establishes -- no proxy
with mb_id < current survives cutover, i.e. the ordinary receive that
follows a cutover must not raise, and must deliver the message that is
actually owed. What used to be preserved as a separate "documents the known
defect" case is retired now that the invariant holds; in its place is a
can-fail case that keeps the buggy failure mode alive under a condition the
receiver-side half cannot cover by itself, proving the fix needs both halves.

THE CASES:

  test_no_stale_proxy_survives_cutover          THE PRIMARY INVARIANT.
      Reproduces the specimen's own ordering -- mb_id=0, seq=117, rows=512,
      victim live on mb_id=2 -- with the stale send landing a short,
      IN-BUDGET delay after the victim's drain call has started polling.
      Asserts the ordinary receive must not raise, and must return the
      proxy that is actually owed to this pass (stamped `OWED_STAMP`, sent
      right after the stale one). Before the #787 fix this failed with the
      #631 guard's RuntimeError, because nothing drained the stale message
      before the ordinary receive reached it; the receiver-side settle
      window now catches it inside the same drain call that used to give
      up instantly.

  test_settle_window_alone_is_insufficient_beyond_budget   THE CAN-FAIL
      PROOF. Identical ordering and stamps, but the upstream's send is
      delayed BEYOND `DRAIN_SETTLE_BUDGET_S` before it lands -- the
      harness-level analogue of an absent or violated sender-side ordering
      guarantee (in production, the whole point of flushing pending sends
      in `_abandon_no_quorum` / `_abandon_unjoined_flip` before disarm is
      to keep a peer's stale send from landing arbitrarily late relative to
      this window). The victim's settle window correctly gives up once its
      bounded budget elapses -- it cannot wait forever, that is the entire
      point of it being bounded -- and the ordinary receive that follows
      then hits the real #631 guard, with the specimen's exact numbers.
      This is the proof that the receiver-side settle window is a bounded
      engineering tolerance, not a substitute for the sender-side ordering
      guarantee: widen the budget arbitrarily and this case still exists,
      just further out.

  test_leftover_before_drain_is_correctly_dropped   GREEN today, and stays
      green. The CONTRAST case: the identical stale message, same stamp,
      same live mb_id -- but posted BEFORE the drain call reads the
      counter, exactly as #757 intended. The shipped drain drops it and the
      guard never fires. Together with the two cases above this pins down
      that the defect was a TIMING gap in when the one-shot sweep was
      allowed to observe the message, not a logic error in the stamp
      comparison itself.

  test_an_owed_output_and_proxy_survive_the_drain   CORPSE S guard, GREEN
      today and MUST STAY GREEN under any future #787 change. The
      2026-08-09 attempt at the #757 fix ate an output and stranded a rank
      for ever; the #787 settle window must not regress that guarantee.
      This case sends an output owed from before the arm together with a
      proxy that legitimately belongs to the pass the victim resumes on,
      drives the shipped drain over both, and asserts neither is swallowed.
      If this ever fails, the #787 fix has become the corpse.
"""

import json
import os
import pickle
import tempfile
import time
import types
import unittest
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2

#: The specimen's own numbers. Red-first means red on THESE, not on a
#: convenient pair: the victim resumes on mb_id=2, and the leftover that
#: kills it is stamped mb_id=0 seq=117 rows=512.
LIVE_MB = 2
LEFTOVER_STAMP = (0, 117, 512)
#: A proxy that legitimately belongs to the pass the victim resumes on.
#: In the CORPSE case this is one of two messages the shipped drain must
#: not eat. In the cutover case (below) it is what a correctly-completed
#: cutover has left for the ordinary receive to find, once the stale
#: message ahead of it has actually been dealt with.
OWED_STAMP = (LIVE_MB, 118, 512)

#: How long a peer will wait on the barrier file before giving up and
#: failing loudly. The victim writes it almost immediately; this bound only
#: exists so a genuinely stuck rank fails fast instead of hanging the join.
BARRIER_TIMEOUT_S = 30.0


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted here -- pickle to bytes, bytes over gloo.
    The demultiplexing, the stamp check and the drain are the shipped
    functions. Mirrors the `pp_group` surface the code under test touches.

    `rank_in_group` / `world_size` are carried because `pp_typed_channel.
    resolve_src` (used by `stash_typed` / `take_typed` on the #757/#787
    demultiplex path) now derives the peer identity from them rather than
    from a bare `src` -- this wire's ring is a straight line
    UPSTREAM(0) -> VICTIM(1) -> DOWNSTREAM(2), the same numbering as the
    real global rank here, so `rank_in_group` is just `rank`.
    """

    def __init__(self, rank: int, src: int, dst: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, d, all_gather_group=None):
        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=self.dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst)

    def send_tensor_dict_nowait(self, d, keepalive: list):
        """Post a message without blocking on the peer ever receiving it.

        Used ONLY for a message that may legitimately go unconsumed on the
        wire within this test's lifetime -- e.g. the owed successor in the
        cutover case, when TODAY's guard raises on the message ahead of it
        and the victim never issues the matching recv. A blocking
        `dist.send` there would hang this rank forever waiting for a recv
        that (today) never comes; `isend` posts and returns immediately,
        exactly as production's own `async_send` path does. `keepalive`
        must outlive this call (the caller holds it until process exit) so
        the source tensors are not garbage-collected while gloo may still
        be reading them.

        Returns the two ``Work`` handles so a caller that DOES know its
        message will be matched (e.g. the in-budget cutover case) can wait
        on them before tearing the process group down -- an isend's Work
        completing means "posted", not "matched", and destroying the
        process group before a peer's matching recv catches up closes the
        socket out from under it.
        """
        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        payload = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
        w_size = dist.isend(size, dst=self.dst)
        w_payload = dist.isend(payload, dst=self.dst)
        keepalive.extend([size, payload, w_size, w_payload])
        return w_size, w_payload

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        # `src` (the shipped caller sometimes passes it positionally, as
        # `None`) is accepted and ignored: this wire already has exactly one
        # fixed peer per direction, set at construction.
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))


def _victim(rank, wire):
    """The shipped mixin methods, bound to a holder (the 630 pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,  # drain is counter-driven; see _drain_n below
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_upstream = lambda: UPSTREAM
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "pp_flip_drain_leftover_dicts",
        # #789 HARNESS REPAIR (interface drift, no assertion touched, same
        # category as the resolve_src repair documented in
        # test_pp_flip_leftover_proxy_757.py): _pp_recv_proxy_tensors now
        # calls self._pp_wait_for_proxy_readiness(mb_id) before its
        # existing receive. With pp_flip_counters=None above, the bound
        # method's own "if counters is None: return" fast path makes this
        # a true no-op -- restoring, not changing, this file's behaviour.
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _drain_n(h, live_mb_id, n):
    """Drive the SHIPPED drain body against a counter that says "n posted",
    FIXED for the whole call -- used by the two cases (`before_drain`,
    `corpse`) that are not about the #787 timing gap itself and do not need
    a live counter. This is the mechanism #787 originally exploited, before
    the receiver-side settle window existed: the drain used to trust a
    single read of `counters.sent(...)` and never re-read it once that read
    satisfied the break condition. The shipped drain now settle-polls this
    same stub a few more times before giving up -- since the value here
    never changes, that only costs a bounded wait (`DRAIN_SETTLE_BUDGET_S`),
    not a behaviour change: a fixed stub still cannot ever reveal a message
    that was not counted at the first read, which is exactly why the
    cutover cases below use `_drain_dynamic` instead.
    """
    state = {"consumed": 0}
    h.pp_flip_counters = types.SimpleNamespace(
        sent=lambda chan, rank: n,
        # #789 HARNESS REPAIR (interface drift, no assertion touched):
        # the readiness gate now also reads an 'entered the send'
        # count, which distinguishes a RENDEZVOUS sender from an idle
        # one. Reading the same source as `sent` keeps this stub's
        # meaning exactly as it was: whatever it says was posted, it
        # had necessarily been entered first.
        attempted=lambda chan, rank: n,
        local_consumed=lambda chan: state["consumed"],
    )
    h._pp_flip_bump_consumed = lambda chan: state.__setitem__(
        "consumed", state["consumed"] + 1
    )
    return h.pp_flip_drain_leftover_dicts(live_mb_id)


def _drain_dynamic(h, live_mb_id, counter_path):
    """Drive the SHIPPED drain body against a LIVE, file-backed counter.

    Unlike `_drain_n` above, this counter can change WHILE the drain call
    is running -- exactly what a real cross-process /dev/shm counter does,
    and exactly what the #787 receiver-side settle window needs in order
    to have anything to observe. `counter_path` is written by the UPSTREAM
    process via `_bump_sent_counter` below, synchronously with each real
    isend it posts (mirroring `bump_sent`'s real synchronicity with the
    send call in `_pp_send_dict_to_next_stage`), and is read here fresh on
    every poll rather than captured once.
    """
    state = {"consumed": 0}

    def _read_sent(_chan, _rank):
        try:
            with open(counter_path) as f:
                raw = f.read().strip()
        except FileNotFoundError:
            return 0
        return int(raw) if raw else 0

    h.pp_flip_counters = types.SimpleNamespace(
        sent=_read_sent,
        # #789 HARNESS REPAIR (interface drift, no assertion touched):
        # the readiness gate now also reads an 'entered the send'
        # count, which distinguishes a RENDEZVOUS sender from an idle
        # one. Reading the same source as `sent` keeps this stub's
        # meaning exactly as it was: whatever it says was posted, it
        # had necessarily been entered first.
        attempted=_read_sent,
        local_consumed=lambda chan: state["consumed"],
    )
    h._pp_flip_bump_consumed = lambda chan: state.__setitem__(
        "consumed", state["consumed"] + 1
    )
    return h.pp_flip_drain_leftover_dicts(live_mb_id)


def _bump_sent_counter(counter_path, n):
    """Publish `n` as the new sent-count, atomically (write + rename).

    Called by the UPSTREAM process immediately after posting each real
    isend -- mirroring production's `bump_sent`, which fires synchronously,
    in the same call, right after `send_tensor_dict`/`isend` returns (see
    `_pp_send_dict_to_next_stage`). The write-then-rename avoids a torn
    read on the victim's side; local-filesystem writes of a few bytes are
    already effectively atomic, but the rename makes it a hard guarantee
    rather than an assumption.
    """
    tmp = counter_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(n))
    os.replace(tmp, counter_path)


def _wait_for_barrier(path, timeout_s=BARRIER_TIMEOUT_S):
    """Poll for a barrier file. Fails LOUDLY on timeout instead of hanging.

    This is what turns "the stale send lands after the victim's drain call
    has started polling" from a hope into a forced fact: the upstream side
    blocks here until the victim has stamped that fact to disk, then --
    and only then -- starts its (in-budget or beyond-budget) delay before
    sending.
    """
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"barrier file {path} never appeared within {timeout_s}s -- "
                f"a peer is stuck; failing loudly instead of hanging the join"
            )
        time.sleep(0.02)


def _worker(rank, init_file, out_dir, case):
    res = {
        "rank": rank,
        "ok": False,
        "error": None,
        "note": None,
        "drained": None,
        "delivered_h": None,
    }
    barrier_path = os.path.join(out_dir, "victim_drain_starting")
    counter_path = os.path.join(out_dir, "sent_counter")
    keepalive: list = []
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            if case in ("cutover", "cutover_late"):
                # Wait until the victim has PROVABLY already entered its
                # single (now settle-aware) drain call -- the exact
                # ordering #787 needs, forced by the barrier file, not
                # scheduling luck. Then delay before sending: IN-BUDGET for
                # the primary case (the receiver-side settle window must
                # catch it), BEYOND BUDGET for the can-fail case (the
                # harness-level analogue of an absent sender-side ordering
                # guarantee -- nothing then bounds how late this send could
                # land). Both messages are posted non-blocking (`isend`),
                # with the sent-counter bumped synchronously right after
                # each post, mirroring `bump_sent`'s real synchronicity
                # with the send call in `_pp_send_dict_to_next_stage`.
                from sglang.srt.managers.scheduler_pp_mixin import (
                    DRAIN_SETTLE_BUDGET_S,
                )

                _wait_for_barrier(barrier_path)
                delay = (
                    DRAIN_SETTLE_BUDGET_S * 0.3
                    if case == "cutover"
                    else DRAIN_SETTLE_BUDGET_S + 0.3
                )
                time.sleep(delay)
                w1 = wire.send_tensor_dict_nowait(
                    {"__msg_type__": "proxy", "__stamp__": LEFTOVER_STAMP, "h": 1},
                    keepalive,
                )
                _bump_sent_counter(counter_path, 1)
                w2 = wire.send_tensor_dict_nowait(
                    {"__msg_type__": "proxy", "__stamp__": OWED_STAMP, "h": 2},
                    keepalive,
                )
                _bump_sent_counter(counter_path, 2)
                # Wait for what the victim will ACTUALLY consume before
                # this process is free to tear its process group down.
                # The leftover is always matched (either drained here, or
                # received by the ordinary receive that then guards on
                # it), but the owed successor is matched only in the
                # in-budget "cutover" case -- in "cutover_late" the victim
                # raises on the leftover and never issues a second recv,
                # so waiting on w2 there would hang this rank forever.
                for w in w1:
                    w.wait()
                if case == "cutover":
                    for w in w2:
                        w.wait()
            elif case == "before_drain":
                # CONTRAST: posted before the drain call reads the counter,
                # exactly the case #757 already covers.
                wire.send_tensor_dict(
                    {"__msg_type__": "proxy", "__stamp__": LEFTOVER_STAMP, "h": 1}
                )
            else:  # "corpse"
                wire.send_tensor_dict({"__msg_type__": "output", "tok": 7})
                wire.send_tensor_dict(
                    {"__msg_type__": "proxy", "__stamp__": OWED_STAMP, "h": 2}
                )
        elif rank == VICTIM:
            wire = _GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM)
            h = _victim(rank, wire)
            if case in ("cutover", "cutover_late"):
                # Signal "about to enter the drain call" BEFORE calling it,
                # not after: the settle-aware drain can now legitimately
                # block for up to DRAIN_SETTLE_BUDGET_S polling this same
                # counter, so the upstream must be free to run its
                # delay-then-send sequence CONCURRENTLY with this call, not
                # only after it returns.
                with open(barrier_path, "w") as f:
                    f.write("drain starting")
                dropped = _drain_dynamic(h, LIVE_MB, counter_path)
                res["drained"] = dropped
                # THE INVARIANT UNDER TEST ("cutover"): this must not
                # raise, and must deliver the proxy actually owed to this
                # pass. THE CAN-FAIL PROOF ("cutover_late"): this IS
                # expected to raise the real #631 guard, because the stale
                # send lands after the settle window already gave up --
                # see the module docstring's COLOUR CONVENTION note.
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                res["delivered_h"] = proxy["h"]
                res["note"] = "cutover delivered the owed proxy without raising"
            elif case == "before_drain":
                dropped = _drain_n(h, LIVE_MB, 1)
                res["drained"] = dropped
                assert dropped == 1, f"expected exactly 1 leftover dropped, got {dropped}"
                # The wire owes nothing further; the guard must never fire.
                res["note"] = f"dropped={dropped}, guard never reached"
            else:  # "corpse"
                dropped = _drain_n(h, LIVE_MB, 2)
                res["drained"] = dropped
                assert dropped == 0, (
                    f"CORPSE S REGRESSION: the drain discarded {dropped} "
                    f"message(s) that were owed, not leftover"
                )
                got = h._pp_recv_typed_dict(expected_kind="output")
                assert got.get("tok") == 7, f"output was eaten or altered: {got}"
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                assert proxy["h"] == 2, f"wrong proxy delivered: {proxy}"
                res["note"] = "output and owed proxy both survived the drain"
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _run(case):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(_worker, args=(init_file, tmp, case), nprocs=WORLD, join=True)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = json.load(f)
        return out


class DrainCompleteness787(unittest.TestCase):
    def test_no_stale_proxy_survives_cutover(self):
        """THE PRIMARY INVARIANT the #787 fix must give.

        The upstream's stale proxy lands a short, IN-BUDGET delay after the
        victim's single (settle-aware) drain call has started polling --
        forced by the barrier file, not scheduling luck. A legitimate
        successor for the live pass follows it. The ordinary receive that
        comes next must not raise, and must deliver that successor, not the
        stale message ahead of it. See
        `test_settle_window_alone_is_insufficient_beyond_budget` for the
        can-fail proof that the receiver-side half alone is not enough.
        """
        res = _run("cutover")
        v = res.get(VICTIM, {})
        self.assertIsNone(
            v.get("error"),
            f"a proxy with mb_id={LEFTOVER_STAMP[0]} (< current mb_id="
            f"{LIVE_MB}) survived cutover and reached the ordinary receive: "
            f"{v.get('error')}",
        )
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertEqual(
            v.get("delivered_h"),
            2,
            f"the ordinary receive must deliver the proxy owed to this "
            f"pass (h=2), not the stale one (h=1): {v}",
        )

    def test_settle_window_alone_is_insufficient_beyond_budget(self):
        """THE CAN-FAIL PROOF: the receiver-side settle window is bounded,
        on purpose, and is not a substitute for the sender-side ordering
        guarantee.

        Identical ordering and stamps to the primary case, but the
        upstream's send is delayed BEYOND `DRAIN_SETTLE_BUDGET_S` before it
        lands -- the harness-level analogue of an absent or violated
        sender-side ordering guarantee (in production, flushing pending
        sends in `_abandon_no_quorum` / `_abandon_unjoined_flip` before
        disarm is precisely what keeps a peer's stale send from landing
        arbitrarily late relative to this window). The victim's settle
        window correctly gives up once its bounded budget elapses -- it
        must not wait forever, that is the entire point of a BOUNDED
        tolerance -- and the ordinary receive that follows then hits the
        real #631 guard, with the specimen's exact numbers.
        """
        res = _run("cutover_late")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(
            v.get("error"),
            f"expected the #631 guard to fire once the stale send lands "
            f"beyond the settle budget, but the victim reported no error: {v}",
        )
        self.assertIn("#631 PROXY LEFTOVER REFUSED", v["error"])
        self.assertIn(f"mb_id={LEFTOVER_STAMP[0]}", v["error"])
        self.assertIn(f"seq={LEFTOVER_STAMP[1]}", v["error"])
        self.assertIn(f"rows={LEFTOVER_STAMP[2]}", v["error"])
        self.assertIn(f"this rank is on mb_id={LIVE_MB}", v["error"])

    def test_leftover_before_drain_is_correctly_dropped(self):
        """GREEN today, and must stay green: the CONTRAST case.

        Identical stale message and stamp as the cutover cases above, but
        posted before the drain reads the counter -- the #757 case. Proves
        the defect is a TIMING gap in the single snapshot, not a fault in
        the stamp comparison the drain performs.
        """
        res = _run("before_drain")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertEqual(v.get("drained"), 1, f"expected the leftover to be dropped: {v}")

    def test_an_owed_output_and_proxy_survive_the_drain(self):
        """CORPSE S GUARD. Must stay green under any future #787 fix.

        Whatever closes the #787 gap must not regress the #757 guarantee:
        an output owed from before the arm, and a proxy that legitimately
        belongs to the pass the victim resumes on, must both survive a
        drain call that observes them.
        """
        res = _run("corpse")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertEqual(
            v.get("drained"), 0, f"an owed message must never count as discarded: {v}"
        )


if __name__ == "__main__":
    unittest.main()

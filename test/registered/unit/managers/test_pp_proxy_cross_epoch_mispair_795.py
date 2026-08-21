"""#795: a proxy from BEFORE a cutover, paired with a batch from after it.

THE SPECIMEN (boot instr15, 2026-08-21, commit c4a29d38de,
/spinning/evidence-665-f1/SPECIMEN_795_proxy_batch_mismatch.txt). The longest
run of the series -- health 06:03:53Z, 10 drive rounds, 10/10 bursts served
8/8, 12 full flip cycles, OOM=0, RESIDENCY CAP=0. Then, in the same second as
a cutover completed:

    06:10:42 PP0/PP1/PP2  PHASE-FLIP DONE pp_to_tp
    06:10:44 PP0          PHASE-POLICY arming tp_to_pp: decode bundle complete
    06:10:48 PP0/PP1/PP2  PHASE-FLIP DONE tp_to_pp
    06:10:48 PP1          ValueError: #631 PP proxy/batch mismatch: received
                          hidden_states with 119 row(s) for a batch of 27
                          token(s) (bs=1)

TWO FACTS MAKE IT SHARP, and this file exists because of the first.

1. THE STAMP GUARD DID NOT CATCH IT. ``PROXY LEFTOVER REFUSED`` is 0 on the
   entire boot, as are the two drains' own log lines -- nothing recognised
   this message as a leftover at any of the three sites that look at stamps.
   The only thing that stopped it was the WIDTH check in
   ``model_runner.forward``, 30 layers into compute, which the guard's own
   docstring names as the check it exists to complement.

2. WHY THE GUARD AGREED. ``mb_id`` is an index into the microbatch slot ring,
   so it lives in ``range(pp_loop_size)``. The cutover calls
   ``init_pp_loop_state`` (phase_flip_runtime.py:1580), which REBUILDS that
   ring for the new topology and restarts the numbering from zero. So the
   stamp's namespace restarts at every cutover, the same small slot numbers
   are handed out again to unrelated passes, and a proxy stranded on the wire
   across the cutover matches ``int(stamp[0]) == int(mb_id)`` one time in
   ``pp_loop_size`` instead of never. Every probe the cutover does take is a
   structure INSIDE the process (``pp_outputs``, ``last_rank_comm_queue``,
   ``send_output_work``, the tensor-dict inbox -- phase_flip_runtime.py:1562),
   so a message on the WIRE is invisible to all of them: instr15 logged
   "output path empty at cutover" 72 times while this was happening.

THE FIX UNDER TEST. ``_pp_proxy_stamp`` appends the phase-flip epoch --
``PhaseFlipRuntime._epoch``, which increments exactly once per completed
cutover (phase_flip_runtime.py:7398), i.e. once per rebuild of the very ring
``mb_id`` indexes -- and ``pp_proxy_stamp_names_pass`` compares (epoch, slot)
instead of slot alone at all three stamp-reading sites. It is not a smaller
window: the epoch is monotone and unbounded, so a message from another
generation of the ring can never coincide, where a slot number always could.

WHY THIS IS THREE REAL PROCESSES OVER REAL GLOO. The bug is an ordering fact
about a real wire -- a message that outlives the state it was addressed to.
The transport below is a thin real-gloo adapter (copied from
test_pp_flip_leftover_proxy_757.py, which reproduced the sibling defect the
same way); every line of logic under test -- ``_pp_recv_typed_dict``, the
#631 receive guard, ``pp_flip_drain_leftover_dicts``,
``pp_proxy_stamp_names_pass`` -- is the SHIPPED code, bound to a holder.

THE WIRE SUPPLIES A HAZARD, NOT A GUARANTEE. The upstream sends a real
pre-cutover proxy whose slot number COINCIDES with the slot the victim
resumes on, exactly as the ring's arithmetic produces one time in three, and
then the owed post-cutover proxy behind it. Nothing in the harness tells the
victim which is which; the shipped code has to work that out from what
crossed the wire. If the discrimination is removed, the victim computes on
the wrong one -- which is the red arm below.

THE FOUR CASES:

  test_cross_epoch_proxy_is_mispaired_without_the_epoch  RED + CAN-FAIL. Only
      the epoch COMPARISON is neutered in the child; the stamp still carries
      the epoch and every API is intact. The victim then accepts the
      pre-cutover 119-row proxy for its 27-token batch -- the specimen.
  test_cross_epoch_proxy_is_refused_by_the_receive_guard  GREEN at the
      backstop: the shipped guard refuses it and names both epochs.
  test_cross_epoch_proxy_is_dropped_at_disarm            GREEN at the
      prevention half: the disarm drain discards it and delivers the owed one.
  test_an_output_is_never_eaten_across_a_cutover         CORPSE S REGRESSION.
      The epoch test may never reach a non-proxy message.
"""

import json
import os
import pickle
import tempfile
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

#: The specimen's own numbers. The victim resumes on slot 1 of the ring the
#: cutover just rebuilt; the leftover names slot 1 of the ring before it. The
#: slot numbers COINCIDE -- that is the whole hazard, and red-first means red
#: on a coinciding pair, not on a conveniently different one.
LIVE_MB = 1
EPOCH_BEFORE_CUTOVER = 11
EPOCH_AFTER_CUTOVER = 12

#: 119 rows for a 27-token batch: the widths the ValueError reported.
LEFTOVER_ROWS = 119
OWED_ROWS = 27

LEFTOVER_STAMP = (LIVE_MB, 4181, LEFTOVER_ROWS, EPOCH_BEFORE_CUTOVER)
OWED_STAMP = (LIVE_MB, 4182, OWED_ROWS, EPOCH_AFTER_CUTOVER)


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted -- pickle to bytes, bytes over gloo. Copied
    from test_pp_flip_leftover_proxy_757.py, including its `resolve_src`
    repair: this ring is a straight line UPSTREAM(0) -> VICTIM(1) ->
    DOWNSTREAM(2), so `rank_in_group` is just `rank`.
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

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))


def _proxy(stamp, rows):
    """A proxy message with a REAL hidden-states tensor of the stamped width.

    The rows matter: the specimen's downstream tripwire is the width check, so
    the payload has to carry the width the stamp claims or the red arm would
    be asserting on a label rather than on the thing that corrupts memory.
    """
    return {
        "__msg_type__": "proxy",
        "__stamp__": stamp,
        "hidden_states": torch.zeros(rows, 4),
    }


def _victim(rank, wire, epoch):
    """The SHIPPED mixin methods, bound to a holder (the 630/757 pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,  # drain is counter-driven; see _drain_n below
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
        # The one authority `_pp_flip_epoch` reads. A SimpleNamespace with an
        # `epoch` attribute is the whole surface it touches, so the holder can
        # stand in for the runtime without building one.
        phase_flip_runtime=types.SimpleNamespace(epoch=epoch),
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_upstream = lambda: UPSTREAM
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_flip_epoch",
        "pp_flip_drain_leftover_dicts",
        # #789 interface drift: no-op here because pp_flip_counters is None.
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _drain_n(h, live_mb_id, n):
    """Drive the SHIPPED drain body for exactly n known-posted messages.

    Copied from test_pp_flip_leftover_proxy_757.py: production gates the drain
    on PhaseFlipCounters in /dev/shm, and here the number in flight is known
    by construction, so a counter stub drives the same loop without a mount.
    """
    state = {"consumed": 0}
    h.pp_flip_counters = types.SimpleNamespace(
        sent=lambda chan, rank: n,
        local_consumed=lambda chan: state["consumed"],
    )
    h._pp_flip_bump_consumed = lambda chan: state.__setitem__(
        "consumed", state["consumed"] + 1
    )
    return h.pp_flip_drain_leftover_dicts(live_mb_id)


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "note": None, "rows": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            if case == "output_first":
                # An OUTPUT for work launched BEFORE the cutover, still owed.
                wire.send_tensor_dict({"__msg_type__": "output", "tok": 7})
            # The hazard: a pre-cutover proxy whose SLOT COINCIDES with the
            # one the victim resumes on, then the proxy that is genuinely owed.
            wire.send_tensor_dict(_proxy(LEFTOVER_STAMP, LEFTOVER_ROWS))
            wire.send_tensor_dict(_proxy(OWED_STAMP, OWED_ROWS))
        elif rank == VICTIM:
            wire = _GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM)
            h = _victim(rank, wire, EPOCH_AFTER_CUTOVER)
            if case == "no_drain":
                # Straight to the shipped receive, as production does when
                # neither drain saw the message -- instr15's own situation.
                got = h._pp_recv_proxy_tensors(LIVE_MB)
                res["rows"] = int(got["hidden_states"].shape[0])
                res["note"] = "guard did NOT fire -- the mispair was accepted"
            elif case == "output_first":
                _drain_n(h, LIVE_MB, 3)
                out = h._pp_recv_typed_dict(expected_kind="output")
                assert out.get("tok") == 7, f"output was eaten or altered: {out}"
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                rows = int(proxy["hidden_states"].shape[0])
                assert rows == OWED_ROWS, f"wrong proxy delivered: {rows} rows"
                res["rows"] = rows
                res["note"] = "output survived and the owed proxy arrived"
            else:  # "drain"
                dropped = _drain_n(h, LIVE_MB, 2)
                assert dropped == 1, f"expected exactly 1 leftover dropped: {dropped}"
                proxy = h._pp_recv_proxy_tensors(LIVE_MB)
                rows = int(proxy["hidden_states"].shape[0])
                assert rows == OWED_ROWS, f"wrong proxy delivered: {rows} rows"
                res["rows"] = rows
                res["note"] = f"dropped={dropped}, owed proxy delivered"
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _blind_worker(rank, init_file, out_dir, case):
    """The SAME driver with ONLY the epoch COMPARISON neutered, IN THE CHILD.

    THE SPAWN TRAP, restated because #796 paid for it: `_run` uses the "spawn"
    start method, so a patch applied inside a test METHOD reaches no child.
    The neutering therefore happens here.

    WHAT IS AND IS NOT NEUTERED. `pp_proxy_stamp_epoch` returning None is
    exactly the pre-#795 reader: it says "this stamp names no epoch", which
    every consumer maps to the slot-only comparison that shipped before. The
    function still EXISTS with its signature intact, `_pp_proxy_stamp` still
    appends the epoch, `_pp_flip_epoch` still answers, and
    `pp_proxy_stamp_names_pass` still runs its own body -- so nothing here can
    produce an AttributeError, and a green result would mean the harness never
    depended on the fix rather than that the fix is present. A wholesale
    revert would prove neither.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.pp_proxy_stamp_epoch = lambda stamp: None
    return _worker(rank, init_file, out_dir, case)


def _run(target, case):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(target, args=(init_file, tmp, case), nprocs=WORLD, join=True)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = json.load(f)
        return out


class PPProxyCrossEpochMispair795(unittest.TestCase):
    def test_cross_epoch_proxy_is_mispaired_without_the_epoch(self):
        """RED, and the can-fail proof for every green below.

        Neuter ONLY the epoch comparison and the coinciding slot number is
        believed again: the victim accepts the 119-row pre-cutover proxy for
        the batch it means to run 27 tokens of. That is boot instr15's own
        pairing, and the reason its guard counted zero refusals.
        """
        res = _run(_blind_worker, "no_drain")
        v = res.get(VICTIM, {})
        self.assertIsNone(
            v.get("error"),
            f"the blinded receive was supposed to ACCEPT the mispair: {v}",
        )
        self.assertEqual(
            v.get("rows"),
            LEFTOVER_ROWS,
            f"the blinded receive did not deliver the leftover: {v}",
        )

    def test_cross_epoch_proxy_is_refused_by_the_receive_guard(self):
        """GREEN at the backstop, on the shipped code, same wire, same pair."""
        res = _run(_worker, "no_drain")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(v.get("error"), f"guard did not fire: {v}")
        self.assertIn("#631 PROXY LEFTOVER REFUSED", v["error"])
        self.assertIn(f"epoch={EPOCH_BEFORE_CUTOVER}", v["error"])
        self.assertIn(f"flip epoch {EPOCH_AFTER_CUTOVER}", v["error"])
        self.assertIn(f"mb_id={LIVE_MB}", v["error"])

    def test_cross_epoch_proxy_is_dropped_at_disarm(self):
        """GREEN at the prevention half: the drain discards the pre-cutover
        proxy whose slot coincides, and the owed one is delivered intact."""
        res = _run(_worker, "drain")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertEqual(v.get("rows"), OWED_ROWS, f"wrong proxy delivered: {v}")

    def test_an_output_is_never_eaten_across_a_cutover(self):
        """CORPSE S REGRESSION. The epoch test may never reach a non-proxy
        message: an output is owed to a real consumer whatever epoch it is
        from, and the 2026-08-09 drain that ate one blocked PP1 for ever."""
        res = _run(_worker, "output_first")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")


if __name__ == "__main__":
    unittest.main()

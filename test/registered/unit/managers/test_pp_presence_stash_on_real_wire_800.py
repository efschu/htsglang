"""#800 over a REAL wire: the drain stashes, and the gate still lets go.

WHY THIS EXISTS AND THE UNIT SUITE DOES NOT SUFFICE. Every case in
test_pp_presence_withholding_deadlock_800.py builds the inbox BY HAND -- it
appends to `(0, "admission_decision")` and then asks the shipped probe about it.
That proves the probe's arithmetic and proves nothing about the KEY: if the
shipped drain wrote its stash under a different `(src, kind)` than the census
reads, or classified a real wire message as some other kind, every one of those
cases would stay green while metal wedged exactly as before.

AND METAL WILL NOT CLOSE THAT GAP ON ITS OWN. The 2026-08-22 17:03 boot carried
this fix through a full pp_to_tp + tp_to_pp cycle with SIX flips and logged
ZERO "STASHED it: kind=" lines: the triggering condition -- an admission
decision in flight at the instant of the arm -- simply never arose in that run.
A run that never triggers the defect cannot confirm the fix, only fail to
falsify it. So the trigger is produced deliberately here, on a real gloo wire,
with the shipped classifier, the shipped stash and the shipped probe.

THE THREE THINGS EACH CASE PROVES, on the real wire:
  1. the shipped drain takes a real `admission_decision` off the wire and
     stashes it under the key the shipped census actually reads;
  2. the shipped hygiene probe then reports the channels EMPTY, so the rank
     announces instead of withholding -- the wedge, not reproduced;
  3. the message is still there afterwards and is delivered to its real
     consumer. This is the half a discard would break: an abandon resets
     nothing, the resumed loop pops exactly one decision per pass, so a
     dropped one puts every later receive off by one for ever.

The CONTRAST case sends an `output` down the same wire through the same drain
and requires the probe to WITHHOLD. Same code path, opposite verdict -- which
is what makes the exemption a decision rather than a hole.

Transport only is adapted (pickle over gloo), exactly as
test_pp_flip_leftover_proxy_757 does; the classifier, the demultiplexer, the
stash, the census and the probe are all shipped code bound to a holder.
"""

import json
import os
import pickle
import tempfile
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo (transport only)."""

    def __init__(self, rank: int, src: int, dst: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, d, dst=None, all_gather_group=None, async_send=False):
        buf = pickle.dumps(d)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=self.dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))


def _victim_holder(wire, drained_state):
    """The SHIPPED methods bound to a holder carrying only what they read."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_chain_receiver=None,
        send_req_work=None,
        send_output_work=None,
        send_proxy_work=None,
        last_rank_comm_queue=None,
        pp_outputs=None,
        _pp_proxy_drops=0,
    )
    # The production drain is counter-driven off /dev/shm. Here the number in
    # flight is known by construction, so a stub counter drives the SAME loop.
    h.pp_flip_counters = types.SimpleNamespace(
        sent=lambda chan, rank: drained_state["posted"],
        local_consumed=lambda chan: drained_state["consumed"],
    )
    h._pp_flip_bump_consumed = lambda chan: drained_state.__setitem__(
        "consumed", drained_state["consumed"] + 1
    )
    h._pp_flip_upstream = lambda: UPSTREAM
    h._pp_ran_mb_ids = lambda: set()
    h._pp_boundary_stats = lambda: None
    # THE REAL STORE, not a second one. On the shipped Scheduler this name is a
    # property returning `typed_inbox(self.pp_group)`; a plain holder cannot
    # resolve a class property, so the same group-owned dict is bound here by
    # hand. It is the identical object `stash_typed(self.pp_group, ...)` writes
    # into, which is the whole point of this file: the census must read the key
    # the drain actually wrote, not one the test invented.
    from sglang.srt.distributed.pp_typed_channel import typed_inbox

    h._pp_tensor_dict_inbox = typed_inbox(wire)
    for name in (
        "pp_flip_drain_tensor_dicts",
        "pp_flip_channels_empty",
        "pp_flip_retire_pp_loop_stash",
        "_pp_recv_typed_dict",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "note": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            kind = "output" if case == "output" else ADMISSION_DECISION_KIND
            # Shaped like the real thing: the kind rides in __msg_type__ and
            # there is no __stamp__, which is exactly what the specimen logged
            # ("kind=admission_decision stamp=None").
            wire.send_tensor_dict({"__msg_type__": kind, "payload": 42})
        elif rank == VICTIM:
            from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

            wire = _GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM)
            state = {"posted": 1, "consumed": 0}
            h = _victim_holder(wire, state)

            took = h.pp_flip_drain_tensor_dicts()
            assert took == 1, f"the shipped drain took {took} messages, expected 1"

            inbox = dict(h._pp_tensor_dict_inbox)
            keys = [k for k, q in inbox.items() if q]
            res["note"] = f"drain stashed under keys={keys}"
            assert keys, "the shipped drain took the message but stashed nothing"

            why = h.pp_flip_channels_empty()
            if case == "output":
                assert why is not None, (
                    "an owed OUTPUT stashed by the real drain did not withhold "
                    "presence; a token would cross the cutover"
                )
                assert "output" in why, f"the reason does not name the kind: {why}"
                res["note"] += f" | withheld as required: {why[:90]}"
            else:
                assert why is None, (
                    "the real drain stashed a real admission_decision and the "
                    "shipped probe STILL reports a non-empty channel: "
                    f"{why!r}. That is the 2026-08-22 wedge, on a real wire"
                )
                # And it must still be deliverable: an abandon resets nothing.
                got = h._pp_recv_typed_dict(expected_kind=ADMISSION_DECISION_KIND)
                assert got.get("payload") == 42, f"message lost or altered: {got}"
                res["note"] += " | probe clean AND message still delivered"
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
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


class PresenceStashOnRealWire(unittest.TestCase):
    def test_a_real_admission_decision_does_not_withhold_presence(self):
        """THE WEDGE, on a real wire, not reproduced.

        The trigger metal declined to produce in six flips is produced here on
        purpose: a real admission decision in flight at the instant of the arm.
        """
        res = _run("admission_decision")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertIn("still delivered", v.get("note") or "")

    def test_can_fail_a_real_output_still_withholds_presence(self):
        """Same wire, same drain, opposite verdict.

        Without this arm the exemption above could be a hole rather than a
        decision -- a probe that returned None for everything would pass it.
        """
        res = _run("output")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("ok"), f"victim did not finish: {v}")
        self.assertIn("withheld as required", v.get("note") or "")


if __name__ == "__main__":
    unittest.main()

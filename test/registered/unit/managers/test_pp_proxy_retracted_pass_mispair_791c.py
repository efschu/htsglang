"""#791c: a proxy whose IDENTITY is perfect and whose BATCH is not.

THE SPECIMEN (boot instr17, 2026-08-21, commit da71b5af6c,
/spinning/evidence-665-f1/boot_instr17.log:62997-63056). 18m44s clean, 33 full
flip cycles, then, on the FIRST admission after the tp_to_pp cutover:

    07:12:49 PP0  #788 PP-ADMISSION verdict=ADMIT n_reqs=2
                  rids=51a294650b...,5e744c29f8... prefix_lens=0,16896
    07:12:49 PP1  #791 PP-ADMISSION unhonourable prefix on rank 1:
                  rid=5e744c29f8... told=16896 local=0
    07:12:49 PP1  #788 PP-ADMISSION verdict=ADMIT n_reqs=1
                  rids=51a294650b... prefix_lens=0
    07:12:49 PP1  ValueError: #631 PP proxy/batch mismatch: received
                  hidden_states with 126 row(s) for a 1 batch of 22 token(s)

126 = 22 + 104. PP0's batch is PP1's batch PLUS the one request PP1 retracted.

WHAT THIS KILLS, AND IT IS THE POINT OF THE FILE. Three boots died of this
raise (instr15 6m55s/12 cycles, instr16 17m53s/27 cycles, instr17
18m44s/33 cycles) and all three were read as a STRANDED message -- a leftover
that outlived the state it was addressed to. #795 closed the cross-cutover
half of that reading. instr17 refutes the rest of it outright:

  * ``PROXY LEFTOVER REFUSED`` is 0 across the whole boot, and so are both
    drains' lines -- nothing anywhere recognised a leftover;
  * the #795 wire probe printed "output path empty at cutover, and no
    unconsumed tensor dict on the wire" on all three ranks at 07:12:49, so
    nothing was in flight across the cutover either;
  * the message's slot, sequence and flip epoch were all CORRECT.

The message was this pass's. No sharper pass identity -- a per-slot
generation, a receiver-derived sequence, a rolling pass id on the admission
decision -- could have discriminated it, because there was nothing stale to
discriminate. What diverged was MEMBERSHIP: `reconcile_pp_admission_decision`
(pp_admission_congruence.py) drops a rid whose `told` prefix this rank cannot
honour, scheduler.py's admission loop then omits it from THIS rank's batch,
and the upstream had already built and launched its own batch from the
decision as it stood before that retraction. A batch in flight cannot be
amended.

THE DISCRIMINATOR UNDER TEST, and why it is one the receiver can predict.
`pp_proxy_pass_retraction_reason` asks the receiver about ITS OWN
retraction -- performed by this rank, at the top of this same pass, strictly
before the proxy receive, and already recorded per slot by #791b's
`_pp_admission_amended_by_slot`. Nothing new crosses the wire, and the
receiver is not asked to predict anything a sender wrote.

IT BEATS THE SHIPPED TRIPWIRE ON THE CASE THAT MATTERS. model_runner.forward
compares ROW COUNTS; two different request sets with the same token count
pass it silently (chunked prefill caps every chunk at the same size, so equal
widths are ordinary). That is silent wrong output, not a shape error -- the
same hazard the #631 stamp names for leftovers. `test_a_same_width_divergence
_is_still_refused` pins it.

WHY THREE REAL PROCESSES OVER REAL GLOO. The claim is about a message that
crosses a stage boundary and the state the receiving rank holds when it
arrives. The transport below is the thin real-gloo adapter from
test_pp_proxy_cross_epoch_mispair_795.py; every line of logic under test --
`pp_admission_decision_to_wire`/`_from_wire`, `reconcile_pp_admission_
decision`, `_pp_note_output_expectation`, `_pp_pass_retraction_reason`,
`_pp_recv_typed_dict` and the #631/#795/#791c receive guard -- is the SHIPPED
code, bound to a holder.

THE FIVE CASES:

  test_a_retracted_pass_is_mispaired_without_the_retraction_test  RED +
      CAN-FAIL. Only `entries_retracted_by_rank`'s RETURN VALUE is neutered in
      the child; the function, `pp_proxy_pass_retraction_reason`,
      `_pp_pass_retraction_reason` and the guard all still run their own
      bodies. The victim then accepts the 126-row proxy for its 22-token
      batch -- instr17's own pairing.
  test_a_retracted_pass_is_refused_by_the_receive_guard  GREEN: the shipped
      guard refuses it and names the retracted rid, its told and its local.
  test_an_unretracted_pass_is_delivered_unchanged  DEFAULT PATH. A decision
      every rank can honour retracts nothing and the proxy is delivered.
  test_a_same_width_divergence_is_still_refused  THE BLIND SPOT. Rows equal
      the victim's own token count, so model_runner's check would pass; the
      membership test still refuses.
  test_a_retraction_by_another_rank_does_not_refuse  REGRESSION. An entry an
      EARLIER rank retracted is passed through unchanged; it did not narrow
      THIS rank's batch relative to ITS upstream, and refusing on it would
      break every pass downstream of any retraction.

CPU-only.
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

register_cpu_ci(est_time=90)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2

#: The specimen's own slot, epoch, rids and widths.
LIVE_MB = 1
FLIP_EPOCH = 66
RID_HONOURABLE = "51a294650b8b464495eda568e42530d7"
RID_UNHONOURABLE = "5e744c29f8de41fe96cb2c673b8582e5"
TOLD_PREFIX = 16896
HONOURABLE_EXTEND = 22
UNHONOURABLE_EXTEND = 104

#: 126 = 22 + 104: the upstream's batch is the victim's plus the retracted
#: request. Both numbers are read off the ValueError instr17 died on.
UPSTREAM_ROWS = HONOURABLE_EXTEND + UNHONOURABLE_EXTEND
VICTIM_TOKENS = HONOURABLE_EXTEND

PROXY_SEQ = 4181


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted -- pickle to bytes, bytes over gloo. Copied
    from test_pp_proxy_cross_epoch_mispair_795.py (itself from
    test_pp_flip_leftover_proxy_757.py), including its `resolve_src` repair:
    this ring is a straight line UPSTREAM(0) -> VICTIM(1) -> DOWNSTREAM(2), so
    `rank_in_group` is just `rank`.
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


def _pp0_decision():
    """PP0's verdict, exactly as instr17 logged it: two requests, one of them
    resting on a 16896-token prefix PP0 has and the victim does not."""
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )

    return PPAdmissionDecision(
        mb_id=LIVE_MB,
        entries=(
            PPAdmissionEntry(
                rid=RID_HONOURABLE, prefix_len=0, extend_len=HONOURABLE_EXTEND
            ),
            PPAdmissionEntry(
                rid=RID_UNHONOURABLE,
                prefix_len=TOLD_PREFIX,
                extend_len=UNHONOURABLE_EXTEND,
            ),
        ),
    )


def _proxy(rows):
    """The upstream's proxy, with a REAL hidden-states tensor of ITS batch's
    width and a stamp that is CORRECT in every element: this pass's slot,
    this rank's next sequence number, the true row count, this flip epoch.
    The hazard is that all of that is true and the message is still not the
    victim's to compute on."""
    return {
        "__msg_type__": "proxy",
        "__stamp__": (LIVE_MB, PROXY_SEQ, rows, FLIP_EPOCH),
        "hidden_states": torch.zeros(rows, 4),
    }


def _victim(wire, pp_rank=VICTIM):
    """The SHIPPED mixin methods, bound to a holder (the 630/757/795 pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
        pp_loop_size=3,
        # The one authority `_pp_flip_epoch` reads, and the one field
        # `_pp_pass_retraction_reason` reads for the rank number.
        phase_flip_runtime=types.SimpleNamespace(epoch=FLIP_EPOCH),
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=WORLD),
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_upstream = lambda: UPSTREAM
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_flip_epoch",
        "_pp_note_output_expectation",
        "_pp_pass_retraction_reason",
        # #789 interface drift: no-op here because pp_flip_counters is None.
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _reconcile_and_note(h, decision, local_match_lens, pp_rank=VICTIM):
    """The victim's top-of-pass #791 block, on the SHIPPED functions.

    This is `_event_loop_pp_body`'s own sequence, minus the parts that need a
    real waiting_queue and tree_cache: reconcile the received decision against
    this rank's local match lengths, then record the amendment for the slot --
    which is what `_pp_pass_retraction_reason` reads back below. The one thing
    stubbed out is the tree-cache lookup that PRODUCES `local_match_lens`; the
    verdict it feeds, and the recording of that verdict, are shipped code.
    """
    from sglang.srt.managers.pp_admission_congruence import (
        reconcile_pp_admission_decision,
    )

    effective, amended = reconcile_pp_admission_decision(
        decision, local_match_lens, rank=pp_rank, pp_size=WORLD
    )
    h._pp_note_output_expectation(LIVE_MB, False, amended)
    return effective, amended


def _local_matches(case):
    """This rank's radix-cache match per rid -- the only stubbed input."""
    if case == "honourable":
        # The victim's cache holds the whole prefix PP0 named: nothing is
        # retracted and the pass is congruent.
        return {RID_HONOURABLE: 0, RID_UNHONOURABLE: TOLD_PREFIX}
    # instr17: PP1's cache had nothing for the rid PP0 offered at 16896.
    return {RID_HONOURABLE: 0, RID_UNHONOURABLE: 0}


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None, "note": None, "rows": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            from sglang.srt.managers.scheduler_pp_mixin import (
                pp_admission_decision_to_wire,
            )

            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            # THE DECISION FIRST, THEN THE PROXY -- the shipped order.
            # `_pp_send_admission_decision` runs right after
            # `get_next_batch_to_run`, the proxy send at the end of the same
            # pass, and both ride the one FIFO tensor-dict wire.
            decision_msg = pp_admission_decision_to_wire(_pp0_decision())
            decision_msg["__msg_type__"] = "admission_decision"
            wire.send_tensor_dict(decision_msg)
            rows = VICTIM_TOKENS if case == "same_width" else UPSTREAM_ROWS
            wire.send_tensor_dict(_proxy(rows))
        elif rank == VICTIM:
            from sglang.srt.managers.scheduler_pp_mixin import (
                pp_admission_decision_from_wire,
            )

            wire = _GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM)
            h = _victim(wire)
            raw = h._pp_recv_typed_dict(expected_kind="admission_decision")
            decision = pp_admission_decision_from_wire(raw)
            assert decision.mb_id == LIVE_MB, f"wrong slot on the wire: {decision}"
            if case == "other_rank":
                # An entry the rank BEFORE this one already retracted: passed
                # through unchanged by `reconcile_pp_admission_decision`, and
                # therefore NOT this rank's own narrowing.
                from dataclasses import replace

                from sglang.srt.managers.pp_admission_congruence import (
                    PPAdmissionDecision,
                )

                entries = tuple(
                    replace(
                        e,
                        admitted=False,
                        retracted=True,
                        retracted_by_rank=DOWNSTREAM,
                        observed_local=0,
                    )
                    if e.rid == RID_UNHONOURABLE
                    else e
                    for e in decision.entries
                )
                decision = PPAdmissionDecision(mb_id=LIVE_MB, entries=entries)
            effective, amended = _reconcile_and_note(
                h, decision, _local_matches("honourable" if case == "clean" else case)
            )
            res["note"] = f"effective={sorted(effective)}"
            got = h._pp_recv_proxy_tensors(LIVE_MB)
            res["rows"] = int(got["hidden_states"].shape[0])
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:900]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _blind_worker(rank, init_file, out_dir, case):
    """The SAME driver with ONLY the retraction LOOKUP neutered, IN THE CHILD.

    THE SPAWN TRAP, restated because #796 paid for it: `_run` uses the "spawn"
    start method, so a patch applied inside a test METHOD reaches no child.
    The neutering therefore happens here.

    WHAT IS AND IS NOT NEUTERED. `entries_retracted_by_rank` returning an empty
    tuple is exactly the pre-#791c reader: "this rank narrowed nothing", which
    `pp_proxy_pass_retraction_reason` maps to None and the guard reads as
    "nothing known against this pass" -- the behaviour that shipped before.
    The function still EXISTS with its signature intact and is still LOOKED UP
    through `scheduler_pp_mixin`'s own module globals at call time;
    `pp_proxy_pass_retraction_reason`, `_pp_pass_retraction_reason` and the
    receive guard all still run their own bodies, the reconciliation still
    retracts, and `_pp_note_output_expectation` still records the amendment.
    So nothing here can produce an AttributeError, and a green result would
    mean the harness never depended on the fix rather than that the fix is
    present. A wholesale revert would prove neither.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.entries_retracted_by_rank = lambda decision, rank: ()
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


class PPProxyRetractedPassMispair791c(unittest.TestCase):
    def test_a_retracted_pass_is_mispaired_without_the_retraction_test(self):
        """RED, and the can-fail proof for every green below.

        Neuter ONLY the retraction lookup and the upstream's 126 rows are
        believed again for a batch the victim built with 22 tokens in it.
        That is boot instr17's own pairing, and the reason its stamp guard --
        which had the right slot, sequence and epoch to compare -- counted
        zero refusals.
        """
        res = _run(_blind_worker, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNone(
            v.get("error"),
            f"the blinded receive was supposed to ACCEPT the mispair: {v}",
        )
        self.assertEqual(
            v.get("rows"),
            UPSTREAM_ROWS,
            f"the blinded receive did not deliver the wider proxy: {v}",
        )
        # The retraction really did happen in the blinded child: only the
        # LOOKUP was neutered, so the victim's own batch is still the narrow
        # one -- which is what makes the accepted pair a mispair.
        self.assertEqual(v.get("note"), f"effective={[RID_HONOURABLE]}")

    def test_a_retracted_pass_is_refused_by_the_receive_guard(self):
        """GREEN on the shipped code, same wire, same pair, and it names the
        cause rather than the symptom."""
        res = _run(_worker, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(v.get("error"), f"guard did not fire: {v}")
        self.assertIn("#791c PROXY BATCH DIVERGED", v["error"])
        self.assertIn(RID_UNHONOURABLE, v["error"])
        self.assertIn(f"told={TOLD_PREFIX}", v["error"])
        self.assertIn("local=0", v["error"])
        # The identity it is NOT complaining about, spelled out so a later
        # reader cannot mistake this for another leftover.
        self.assertIn(f"mb_id={LIVE_MB}", v["error"])
        self.assertIn(f"epoch {FLIP_EPOCH}", v["error"])
        self.assertNotIn("PROXY LEFTOVER REFUSED", v["error"])

    def test_an_unretracted_pass_is_delivered_unchanged(self):
        """DEFAULT PATH. A decision this rank can honour retracts nothing, and
        the proxy goes through untouched -- the guard adds no new refusal to a
        congruent pass."""
        res = _run(_worker, "clean")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertEqual(v.get("rows"), UPSTREAM_ROWS, f"proxy not delivered: {v}")
        self.assertEqual(
            v.get("note"),
            f"effective={sorted([RID_HONOURABLE, RID_UNHONOURABLE])}",
            f"the clean case must retract nothing: {v}",
        )

    def test_a_same_width_divergence_is_still_refused(self):
        """THE BLIND SPOT THE SHIPPED TRIPWIRE HAS.

        The upstream's proxy carries exactly as many rows as the victim's own
        batch has tokens, so `model_runner.forward`'s `_hs.shape[0] != _want`
        would pass it into compute without a word -- one request set's hidden
        states on another's metadata, silent wrong output rather than a shape
        error. The harness sets the widths equal deliberately: the property
        under test is that #791c does not consult them at all.
        """
        res = _run(_worker, "same_width")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(
            v.get("error"), f"a same-width divergence was accepted: {v}"
        )
        self.assertIn("#791c PROXY BATCH DIVERGED", v["error"])
        self.assertIn(RID_UNHONOURABLE, v["error"])

    def test_a_retraction_by_another_rank_does_not_refuse(self):
        """REGRESSION, and the reason `retracted_by_rank` is compared rather
        than `retracted`.

        An entry an earlier rank retracted arrives ALREADY retracted and takes
        `reconcile_pp_admission_decision`'s pass-through branch: this rank
        narrows nothing, and its upstream built its batch from the same
        already-narrowed decision. Refusing here would break every pass
        downstream of any retraction anywhere in the ring.
        """
        res = _run(_worker, "other_rank")
        v = res.get(VICTIM, {})
        self.assertIsNone(
            v.get("error"), f"refused another rank's retraction: {v.get('error')}"
        )
        self.assertEqual(v.get("rows"), UPSTREAM_ROWS, f"proxy not delivered: {v}")


if __name__ == "__main__":
    unittest.main()

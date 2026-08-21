"""#791 CORE: a downstream PP rank EXECUTES the forwarded pass geometry.

THE SPECIMEN THIS FILE EXISTS TO KILL is boot instr20, PP1, 2026-08-21
09:40:30 (/spinning/evidence-665-f1/boot_instr20.log:5171,5181-5183, crash at
:5187):

    PP0  #788 verdict=ADMIT rid=6cbe2733 prefix_lens=0   chunked=1
    PP1  #788 verdict=ADMIT rid=6cbe2733 prefix_lens=512 chunked=0
    PP2  #788 verdict=ADMIT rid=6cbe2733 prefix_lens=512 chunked=0
    PP1  ValueError: #631 PP proxy/batch mismatch: received hidden_states with
         512 row(s) for a 1 batch of 333 token(s)

ONE request, 845 fill tokens, TWO different splits of it taken in the same
second off the SAME forwarded decision. PP0 split it 0 + 512 (a chunk); PP1
and PP2 split it 512 + 333 (a remainder). Nothing was lost on the wire and no
identity was stale: `extend_len=512` was in the decision PP1 received, and
PP1 recomputed 333 for itself anyway, because nothing ever read that number.

THE MECHANISM, exactly, and it is a mid-pass change of LOCAL state:

  1. PP1 retracted this rid at told=512/local=0 and #797 voided the pass.
  2. `HiCache prefetch success req=6cbe2733 completed_local=512` landed on
     PP1 and PP2 -- and NOT on PP0, which already held it.
  3. PP0 re-offered at prefix_len=0 (its learned floor) with extend_len=512.
  4. PP1's admission loop clamped `prefix_indices` to the offered 0
     (scheduler.py) -- correct, and immediately undone: `add_one_req`'s
     `req.needs_host_load_back()` was now true, `init_load_back` concatenated
     the 512 freshly-resident prefix indices back on
     (schedule_policy.py:1539-1549), and 845 - 512 = 333 then fitted
     `rem_chunk_tokens` whole, so the NON-chunked branch fired and set
     `extend_range(512, 845)`.
     `MAMBA-HOST-RESUME ... this match triggers load_back` is in the log on
     PP1 and PP2 and absent on PP0. That asymmetry IS the bug.

So the two numbers that decide the cross-stage tensor's row count were
re-derived by the consumer from state that moved under it between the
decision and the batch. This is not a race that can be closed by ordering:
the prefetch is legitimate, the load-back is legitimate, and PP0 cannot know
about either before it commits. The only fix is to stop asking.

WHAT IS UNDER TEST. `PrefillAdder`, given a forwarded
`scheduled_extents` mapping, reads `(prefix_len, extend_len)` off it and
performs no local derivation at all -- no `rem_chunk_tokens` truncation, no
page/alignment rounding, no host load-back, no budget veto. The budget is
still charged; only the local OPINION is gone. A geometry that is genuinely
impossible becomes a loud `PPScheduleRefused` that quotes the upstream's
numbers, never a smaller batch.

RED-FIRST, OVER THREE LIVE SPAWNED PROCESSES, REAL GLOO, SHIPPED FUNCTIONS.
The upstream really sends the shipped admission decision and a real 512-row
proxy over gloo; the victim really receives both through
`_pp_recv_admission_decision` / `_pp_recv_proxy_tensors`, really reconciles
through `reconcile_pp_admission_decision`, really takes its schedule through
the shipped `_pp_forwarded_schedule_from`, and really builds its extend range
through the shipped `PrefillAdder.add_one_req`. The row count it ends up with
is compared against the delivered proxy using `model_runner.forward`'s own
expression (`_hs.shape[0] != _want`), so a red arm reproduces the instr20
ValueError arithmetic rather than an assertion invented here.

THE CAN-FAIL IS A SINGLE RETURN-VALUE REBIND IN THE CHILD.
`scheduler_pp_mixin.forwarded_schedule` is rebound to a function of the same
signature returning `{}` -- the answer that shipped before this change, i.e.
"no forwarded geometry". Every other function still exists and still runs its
own body: `_pp_forwarded_schedule_from`, `scheduled_extent_for`,
`_add_scheduled_req`, `schedule_refusal_reason`, the reconcile, the void, the
proxy receive. Nothing can raise an AttributeError, so a green result under
the neuter would mean the harness never depended on the fix -- which is the
whole reason a wholesale revert proves nothing.
"""

import json
import os
import pickle
import tempfile
import types
import unittest
from collections import defaultdict, deque
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionDecision,
    PPAdmissionEntry,
    PPScheduleRefused,
    forwarded_schedule,
    schedule_refusal_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2

LIVE_MB = 1
FLIP_EPOCH = 8
PROXY_SEQ = 4181

#: instr20's own numbers.
RID = "6cbe27334f2a4574978be55f5a600b96"
PROMPT_TOKENS = 845
#: What PP0 committed: no prefix reuse, one 512-token chunk.
TOLD_PREFIX = 0
TOLD_EXTEND = 512
#: What the HiCache prefetch put into the victim's cache mid-pass, and what
#: `init_load_back` would concatenate back onto a clamped `prefix_indices`.
PREFETCHED_PREFIX = 512
#: What the victim recomputed for itself on instr20: 845 - 512.
LOCAL_REMAINDER = PROMPT_TOKENS - PREFETCHED_PREFIX

SENTINEL_TOKEN = 7777


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo (the #791c/#797
    pattern): only the TRANSPORT is adapted, pickle to bytes and bytes over
    gloo. Every message below is built and read by shipped code."""

    def __init__(self, rank: int, src: int, dst: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, tensor_dict=None, all_gather_group=None, **kwargs):
        buf = pickle.dumps(tensor_dict)
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


def _pp0_decision() -> PPAdmissionDecision:
    """PP0's committed pass, in instr20's shape: prefix 0, one 512-chunk."""
    return PPAdmissionDecision(
        mb_id=LIVE_MB,
        entries=(
            PPAdmissionEntry(rid=RID, prefix_len=TOLD_PREFIX, extend_len=TOLD_EXTEND),
        ),
    )


def _proxy():
    """PP0's real hidden states for that pass: 512 rows, and a stamp correct
    in every element -- this slot, this sequence, this width, this epoch. The
    hazard is that all of that is true and the victim still computes a
    different batch under it."""
    hs = torch.zeros(TOLD_EXTEND, 4)
    return {
        "__msg_type__": "proxy",
        "__stamp__": (LIVE_MB, PROXY_SEQ, int(hs.shape[0]), FLIP_EPOCH),
        "hidden_states": hs,
    }


def _sentinel():
    return {
        "__msg_type__": "output",
        "next_token_ids": torch.tensor([SENTINEL_TOKEN], dtype=torch.long),
    }


def _holder(wire, pp_rank):
    """The SHIPPED mixin methods bound to a holder (the #630/#757/#795/#797
    pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
        _pp_admission_send_work=[],
        _pp_gapped_wire=False,
        pp_loop_size=3,
        phase_flip_runtime=types.SimpleNamespace(epoch=FLIP_EPOCH),
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=WORLD),
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_upstream = lambda: pp_rank - 1
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_send_dict_to_next_stage",
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_forwarded_schedule_from",
        "_pp_void_retracted_pass",
        "_pp_drain_voided_proxy",
        "_pp_flip_epoch",
        "_pp_note_output_expectation",
        "_pp_pass_retraction_reason",
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


class _Req:
    """The one request of instr20's pass, with the real `Req` semantics the
    adder touches. Not a MagicMock: `set_extend_range` and `extend_range` are
    the quantities under test, and a mock would record the call instead of
    answering the question."""

    def __init__(self, prefix_len: int, host_resident: int):
        from sglang.srt.utils.common import Range

        self.rid = RID
        # A TENSOR of KV-pool slot pointers, as the real `Req` carries -- the
        # load-back path `torch.cat`s onto it, and #796 is the standing
        # reminder that this object's type is load-bearing.
        self.prefix_indices = torch.arange(prefix_len)
        self.full_untruncated_fill_ids = list(range(PROMPT_TOKENS))
        self.output_ids = []
        self.origin_input_ids = self.full_untruncated_fill_ids
        self.extend_range = Range(0, 0)
        self.retracted_stain = False
        self.last_node = "node"
        self.best_match_node = "node"
        self.mamba_pool_idx = None
        self.session = None
        self.born_spilled = False
        self.born_spilled_deep = False
        self.swa_host_hit_length = 0
        self.cache_protected_len = prefix_len
        self.sampling_params = types.SimpleNamespace(
            max_new_tokens=16, ignore_eos=False
        )
        #: THE MID-PASS STATE CHANGE. The HiCache prefetch landed after the
        #: decision was built, so this rank's host tier now holds a prefix the
        #: upstream's decision was taken without.
        self.host_hit_length = host_resident
        self._host_resident = host_resident

    def needs_host_load_back(self) -> bool:
        return self._host_resident > len(self.prefix_indices)

    def set_extend_range(self, start: int, end: int) -> None:
        from sglang.srt.utils.common import Range

        self.extend_range = Range(start, end)

    def finished(self) -> bool:
        return False


def _adder(scheduled_extents):
    """A real `PrefillAdder`, with instr20's chunk width and a cache that
    performs the load-back the victim's own state now permits."""
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.base_prefix_cache import (
        DecLockRefResult,
        IncLockRefResult,
    )
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    tree_cache = MagicMock()
    tree_cache.disable = False
    tree_cache.evictable_size.return_value = 1 << 20
    tree_cache.full_evictable_size.return_value = 1 << 20
    tree_cache.swa_evictable_size.return_value = 0
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    tree_cache.is_tree_cache.return_value = False
    # THE LOAD-BACK, as instr20 performed it: the freshly host-resident prefix
    # indices, concatenated onto whatever `prefix_indices` currently holds.
    tree_cache.init_load_back.return_value = (
        torch.arange(PREFETCHED_PREFIX),
        "node",
    )

    allocator = MagicMock()
    allocator.available_size.return_value = 1 << 20
    allocator.full_available_size.return_value = 1 << 20
    allocator.swa_available_size.return_value = 0

    running_batch = MagicMock()
    running_batch.reqs = []

    return PrefillAdder(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=1 << 20,
        # instr20's chunk width: 845 does not fit, 333 does. That is what makes
        # the local derivation pick a different split from the upstream's.
        rem_chunk_tokens=TOLD_EXTEND,
        scheduled_extents=scheduled_extents,
    )


def _victim_pass(h, decision, res):
    """The victim's pass, on shipped functions throughout."""
    from sglang.srt.managers.pp_admission_congruence import (
        reconcile_pp_admission_decision,
    )
    from sglang.srt.managers.schedule_policy import AddReqResult

    # This rank's own radix match for the rid, AFTER the prefetch landed.
    # `local >= told` is the SAFE branch, so nothing is retracted and nothing
    # is voided: the pass on instr20 was perfectly congruent in MEMBERSHIP
    # and in PREFIX. Only the split diverged.
    effective, amended = reconcile_pp_admission_decision(
        decision, {RID: PREFETCHED_PREFIX}, rank=VICTIM, pp_size=WORLD
    )
    effective, amended = h._pp_void_retracted_pass(effective, amended)
    res["voided"] = bool(h._pp_admission_pass_voided)
    res["effective"] = {k: int(v) for k, v in effective.items()}

    # THE SEAM UNDER TEST, taken through the shipped mixin method so the
    # child-side neuter reaches it.
    schedule = h._pp_forwarded_schedule_from(amended)
    res["schedule"] = {k: list(v) for k, v in schedule.items()}

    # The admission loop's clamp to the offered prefix (scheduler.py), which
    # instr20 also performed correctly -- and which the load-back then undid.
    req = _Req(prefix_len=PREFETCHED_PREFIX, host_resident=PREFETCHED_PREFIX)
    told = effective.get(RID)
    if told is not None and len(req.prefix_indices) > told:
        req.prefix_indices = req.prefix_indices[:told]
    res["prefix_after_clamp"] = len(req.prefix_indices)

    adder = _adder(schedule or None)
    try:
        outcome = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )
        res["add_result"] = str(outcome)
        res["refused"] = None
    except PPScheduleRefused as exc:
        res["add_result"] = None
        res["refused"] = str(exc)[:400]
        outcome = None

    if outcome == AddReqResult.CONTINUE:
        res["batch_prefix"] = int(req.extend_range.start)
        res["batch_tokens"] = int(req.extend_range.length)
        res["load_back_calls"] = int(adder.tree_cache.init_load_back.call_count)
    else:
        res["batch_prefix"] = None
        res["batch_tokens"] = None
        res["load_back_calls"] = int(adder.tree_cache.init_load_back.call_count)

    # The upstream's hidden states, really received.
    got = h._pp_recv_proxy_tensors(LIVE_MB)
    rows = int(got["hidden_states"].shape[0])
    res["rows"] = rows
    # model_runner.forward's OWN expression, verbatim: `_hs.shape[0] != _want`.
    # Reproducing the arithmetic here is what makes a red arm the instr20
    # ValueError rather than a claim about it.
    res["mismatch"] = (
        None if res["batch_tokens"] is None else rows != res["batch_tokens"]
    )

    tail = h._pp_recv_typed_dict(expected_kind="output")
    res["sentinel"] = int(tail["next_token_ids"][0].item())


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            from sglang.srt.managers.scheduler_pp_mixin import (
                pp_admission_decision_to_wire,
            )

            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            msg = pp_admission_decision_to_wire(_pp0_decision())
            msg["__msg_type__"] = "admission_decision"
            msg["__pp_output_expected__"] = True
            msg["__pp_pass_voided__"] = False
            msg["__pp_upstream_launched__"] = True
            wire.send_tensor_dict(tensor_dict=msg)
            wire.send_tensor_dict(tensor_dict=_proxy())
            wire.send_tensor_dict(tensor_dict=_sentinel())
        elif rank == VICTIM:
            h = _holder(_GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM), VICTIM)
            decision = h._pp_recv_admission_decision()
            assert decision.mb_id == LIVE_MB, f"wrong slot on the wire: {decision}"
            _victim_pass(h, decision, res)
            h._pp_send_admission_decision(
                decision,
                expects_output=False,
                pass_voided=bool(h._pp_admission_pass_voided),
                launched=True,
            )
        else:
            h = _holder(_GlooWire(rank, src=VICTIM, dst=UPSTREAM), DOWNSTREAM)
            incoming = h._pp_recv_admission_decision()
            res["schedule"] = {
                k: list(v) for k, v in h._pp_forwarded_schedule_from(incoming).items()
            }
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


def _blind_forwarded_schedule(rank, init_file, out_dir, case):
    """THE CAN-FAIL: the forwarded geometry, and NOTHING else, IN THE CHILD.

    `mp.spawn` uses the "spawn" start method, so a patch applied inside a test
    METHOD reaches no child -- the trap #796 paid for. Rebound here, in the
    child's own entry point, through `scheduler_pp_mixin`'s module globals,
    which is the namespace `_pp_forwarded_schedule_from` resolves it in.

    `{}` is precisely the answer that shipped before this change: no forwarded
    geometry, so `PrefillAdder.scheduled_extents` is None and `add_one_req`
    takes the local-derivation path it always took. Every function involved
    still exists with its signature intact and still runs its own body, so an
    AttributeError is impossible and a green result here would mean the
    harness never depended on the fix.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.forwarded_schedule = lambda decision: {}
    return _worker(rank, init_file, out_dir, case)


def _run(target, case="live"):
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


class PPForwardedSchedule791(unittest.TestCase):
    def test_red_without_the_forwarded_geometry_instr20_reappears(self):
        """RED. Blind the forwarded geometry and the victim re-derives
        instr20's split: 512 prefix + 333 extend against the upstream's 512
        rows -- the exact ValueError, arithmetic and all."""
        out = _run(_blind_forwarded_schedule)
        v = out[VICTIM]
        self.assertIsNone(v["error"], v["error"])
        self.assertFalse(v["voided"], "the pass was congruent in membership")
        self.assertEqual(v["schedule"], {}, "the neuter must be the only change")
        # The admission loop clamped to the offered prefix...
        self.assertEqual(v["prefix_after_clamp"], TOLD_PREFIX)
        # ...and the load-back put it straight back.
        self.assertEqual(v["load_back_calls"], 1)
        self.assertEqual(v["batch_prefix"], PREFETCHED_PREFIX)
        self.assertEqual(v["batch_tokens"], LOCAL_REMAINDER)
        self.assertEqual(v["rows"], TOLD_EXTEND)
        self.assertTrue(
            v["mismatch"],
            f"#631 would have raised: {v['rows']} row(s) for a batch of "
            f"{v['batch_tokens']} token(s)",
        )

    def test_green_the_forwarded_geometry_survives_the_mid_pass_prefetch(self):
        """GREEN, and it is the whole claim: the victim's LOCAL state changed
        mid-pass exactly as it did on instr20 -- the prefix is host-resident,
        `needs_host_load_back()` is true, `rem_chunk_tokens` would have picked
        a different split -- and the batch it builds is still the upstream's,
        to the token."""
        out = _run(_worker)
        v = out[VICTIM]
        self.assertIsNone(v["error"], v["error"])
        self.assertFalse(v["voided"])
        self.assertEqual(v["schedule"], {RID: [TOLD_PREFIX, TOLD_EXTEND]})
        self.assertEqual(v["prefix_after_clamp"], TOLD_PREFIX)
        # THE LOAD-BACK NEVER RAN. It is the instr20 line, and it is a
        # rank-local improvement to a quantity this rank no longer owns.
        self.assertEqual(v["load_back_calls"], 0)
        self.assertEqual(v["batch_prefix"], TOLD_PREFIX)
        self.assertEqual(v["batch_tokens"], TOLD_EXTEND)
        self.assertEqual(v["rows"], TOLD_EXTEND)
        self.assertFalse(v["mismatch"], "#631 is structurally unreachable here")
        self.assertEqual(v["sentinel"], SENTINEL_TOKEN)

    def test_the_geometry_reaches_the_next_rank_unamended(self):
        """The third rank reads the SAME two numbers off the decision the
        victim forwarded, through the shipped receive -- so 'PP0 decides once'
        holds for the whole chain and not just the first hop."""
        out = _run(_worker)
        self.assertIsNone(out[DOWNSTREAM]["error"], out[DOWNSTREAM]["error"])
        self.assertEqual(out[DOWNSTREAM]["schedule"], {RID: [TOLD_PREFIX, TOLD_EXTEND]})
        self.assertEqual(out[DOWNSTREAM]["schedule"], out[VICTIM]["schedule"])


class PPForwardedSchedulePure791(unittest.TestCase):
    """The pure half: no processes, no wire."""

    def _decision(self, **kw):
        base = dict(rid=RID, prefix_len=TOLD_PREFIX, extend_len=TOLD_EXTEND)
        base.update(kw)
        return PPAdmissionDecision(mb_id=LIVE_MB, entries=(PPAdmissionEntry(**base),))

    def test_a_voided_decision_schedules_nothing(self):
        """#797's void marks every survivor `admitted=False`, and the schedule
        must empty in the same breath as `effective` -- the two naming
        different rid sets is a batch nobody agreed on."""
        self.assertEqual(forwarded_schedule(self._decision(admitted=False)), {})

    def test_a_retracted_entry_schedules_nothing(self):
        self.assertEqual(
            forwarded_schedule(
                self._decision(admitted=False, retracted=True, retracted_by_rank=1)
            ),
            {},
        )

    def test_no_decision_is_no_schedule(self):
        """A rank with nothing recorded reads as 'no forwarded geometry',
        which is byte-identically the pre-change behaviour."""
        self.assertEqual(forwarded_schedule(None), {})

    def test_the_refusal_quotes_the_forwarded_decision(self):
        reason = schedule_refusal_reason(
            rid=RID,
            scheduled_prefix_len=TOLD_PREFIX,
            scheduled_extend_len=TOLD_EXTEND,
            local_prefix_len=PREFETCHED_PREFIX,
            local_fill_len=PROMPT_TOKENS,
        )
        self.assertIsNotNone(reason)
        self.assertIn(f"prefix_len={TOLD_PREFIX}", reason)
        self.assertIn(str(PREFETCHED_PREFIX), reason)
        self.assertIn(RID, reason)

    def test_an_executable_geometry_has_no_reason(self):
        self.assertIsNone(
            schedule_refusal_reason(
                rid=RID,
                scheduled_prefix_len=TOLD_PREFIX,
                scheduled_extend_len=TOLD_EXTEND,
                local_prefix_len=TOLD_PREFIX,
                local_fill_len=PROMPT_TOKENS,
            )
        )

    def test_a_geometry_past_the_request_is_refused(self):
        reason = schedule_refusal_reason(
            rid=RID,
            scheduled_prefix_len=TOLD_PREFIX,
            scheduled_extend_len=PROMPT_TOKENS + 1,
            local_prefix_len=TOLD_PREFIX,
            local_fill_len=PROMPT_TOKENS,
        )
        self.assertIsNotNone(reason)
        self.assertIn(str(PROMPT_TOKENS), reason)

    def test_an_empty_extend_is_refused_not_run(self):
        """A pass that computes nothing for a named request is not a smaller
        pass, it is a different one."""
        self.assertIsNotNone(
            schedule_refusal_reason(
                rid=RID,
                scheduled_prefix_len=TOLD_PREFIX,
                scheduled_extend_len=0,
                local_prefix_len=TOLD_PREFIX,
                local_fill_len=PROMPT_TOKENS,
            )
        )

    def test_no_mapping_is_the_untouched_default_path(self):
        """`scheduled_extent_for` must answer None for every rank that owns
        its own admission truth, so `add_one_req` never enters the executor."""
        adder = _adder(None)
        self.assertIsNone(adder.scheduled_extent_for(_Req(0, 0)))
        adder = _adder({})
        self.assertIsNone(adder.scheduled_extent_for(_Req(0, 0)))
        adder = _adder({"someone-else": (0, 4)})
        self.assertIsNone(adder.scheduled_extent_for(_Req(0, 0)))

    def test_the_carried_chunk_is_executed_not_re_chunked(self):
        """`add_chunked_req` is the consumer with NO forwarded-decision gate
        at all: it is entered before the admission loop and decided its chunk
        length from `rem_chunk_tokens` alone. On instr20 PP0 carried this rid
        as `chunked=1` while PP1 and PP2 carried it as `chunked=0`."""
        adder = _adder({RID: (TOLD_PREFIX, TOLD_EXTEND)})
        req = _Req(prefix_len=TOLD_PREFIX, host_resident=PREFETCHED_PREFIX)
        still_chunked = adder.add_chunked_req(req)
        self.assertIs(still_chunked, req, "512 of 845 is not the last chunk")
        self.assertEqual(req.extend_range.start, TOLD_PREFIX)
        self.assertEqual(req.extend_range.length, TOLD_EXTEND)
        # A carried chunk is already the scheduler's `chunked_req`; announcing
        # it as a NEW one trips `assert self.chunked_req is None`.
        self.assertIsNone(adder.new_chunked_req)

    def test_a_carried_chunk_that_finishes_is_released(self):
        adder = _adder({RID: (0, PROMPT_TOKENS)})
        req = _Req(prefix_len=0, host_resident=0)
        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual(req.extend_range.length, PROMPT_TOKENS)

    def test_the_executor_charges_the_budget_it_spends(self):
        """The local VETO is gone; the bookkeeping is not. A pass that spent
        tokens without charging them would over-commit the next request in the
        same round."""
        adder = _adder({RID: (TOLD_PREFIX, TOLD_EXTEND)})
        before = adder.rem_chunk_tokens
        adder.add_one_req(
            _Req(prefix_len=TOLD_PREFIX, host_resident=0),
            has_chunked_req=False,
            truncation_align_size=None,
        )
        self.assertEqual(adder.rem_chunk_tokens, before - TOLD_EXTEND)
        self.assertEqual(adder.log_input_tokens, TOLD_EXTEND)

    def test_a_non_last_chunk_is_announced_as_the_new_chunked_request(self):
        adder = _adder({RID: (TOLD_PREFIX, TOLD_EXTEND)})
        req = _Req(prefix_len=TOLD_PREFIX, host_resident=0)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIs(adder.new_chunked_req, req)

    def test_a_last_chunk_is_not(self):
        adder = _adder({RID: (0, PROMPT_TOKENS)})
        req = _Req(prefix_len=0, host_resident=0)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIsNone(adder.new_chunked_req)

    def test_an_impossible_geometry_raises_rather_than_narrowing(self):
        adder = _adder({RID: (TOLD_PREFIX, TOLD_EXTEND)})
        req = _Req(prefix_len=PREFETCHED_PREFIX, host_resident=PREFETCHED_PREFIX)
        with self.assertRaises(PPScheduleRefused) as ctx:
            adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(RID, str(ctx.exception))
        self.assertEqual(adder.can_run_list, [], "no partial batch may survive")


if __name__ == "__main__":
    unittest.main()

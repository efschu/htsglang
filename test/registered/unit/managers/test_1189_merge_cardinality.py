"""#1189, behavioural half -- ``ScheduleBatch.merge_batch`` re-admits requests
it is already holding, and every list and tensor index-parallel to ``reqs``
grows with them.

WHAT THE SLOT DEFECT HANDS TO THIS FILE. `test_1189_slot_publish_lap.py`
pins the control-flow root: a PP slot that runs nothing keeps its previous
``last_mbs[slot]``, and that entry is an EXTEND batch. ``scheduler.py``'s
``get_next_batch_to_run`` then takes ``last_batch.forward_mode.is_extend()``
and reaches ``running_batch.merge_batch(last_batch)`` on EVERY later visit to
that slot. The object is DISTINCT from ``running_batch`` -- so the identity
guard at ``schedule_batch.py:4691`` (`if other is self`) does not fire -- and
the duplication lands INSIDE one batch's ``reqs`` -- so ``harvest_resident_
batches``'s dedupe by ``id(batch)`` does not see it either. Boot 8
(`/spinning/evidence-665-f1/boot_855_weg1b8_e9d1a719ac_0904_064622.log`):
``running_bs`` 7768 against ``max_running_requests=8``, at most 3 distinct
rids, growing ~+400 every 5 s; 183 ADMIT against 27127 DECLINE.

THE HARD CONSTRAINT THIS FILE EXISTS TO ENFORCE (fact 2 of the briefing, and
the reason "same pass or not at all"): ``merge_batch`` maintains state that
is INDEX-PARALLEL to ``reqs``, across THREE logprob arms, and the third
REBUILDS both lists from ``len(self.reqs)`` before the extend:

    schedule_batch.py:4718      self.encoder_lens_cpu.extend(...)
    schedule_batch.py:4747/4748 arm 1 -- both sides return_logprob
    schedule_batch.py:4750/4751 arm 2 -- SELF only   (the `elif` arms that
                                          no earlier report named)
    schedule_batch.py:4753/4754 arm 3 -- OTHER only  (a REBUILD from
                                          len(self.reqs), before :4755)
    schedule_batch.py:4755      self.reqs.extend(other.reqs)
    schedule_batch.py:4757      self.multimodal_inputs.extend(...)
    sampling/sampling_batch_info.py:405  self.custom_params.extend(...)

A dedup on ``reqs`` alone -- or on only two of the three arms -- leaves those
lists LONGER than ``reqs`` and mis-indexed, which is worse than the defect it
replaces: every per-request lookup after the merge then reads another
request's row. Each test below therefore asserts BOTH halves at once, the
cardinality AND the parallelism, so a partial fix cannot go green.

A MEASURED CORRECTION TO "SIX SIBLINGS". The list masks are six, but the
req-parallel population of this method is larger: ``req_pool_indices``,
``req_pool_indices_cpu``, ``seq_lens``, ``orig_seq_lens`` and ``seq_lens_cpu``
are ``torch.cat``-ed one entry per request in the same call, and
``SamplingBatchInfo.merge_batch`` cats ``temperatures``/``top_ps``/``top_ks``/
``min_ps``/``sampling_seed`` and the ``custom_logit_processor`` masks the same
way. ``test_req_parallel_member_census`` pins the set this suite checks so the
next reader is not told "six" again.

FIXTURE POLICY. The method under test is the SHIPPING
``ScheduleBatch.merge_batch``, taken unbound and driven against a minimal
stand-in ``self``/``other``. ``_assert_stub_is_faithful`` AST-scans the
shipping source for every ``self.<attr>`` and ``other.<attr>`` it touches and
fails if the stand-in does not carry it -- the #630 lesson that an unfaithful
stub does not merely miss a defect, it encodes the defect's assumption and
then certifies it. ``SamplingBatchInfo`` is the real class, built through
``__new__`` so ``__len__`` reads a real ``temperatures`` tensor.

CPU only, no CUDA, no distributed. Plain ``unittest.TestCase``:
``CustomTestCase`` retries three times and then replaces the reason with
"retry() exceed maximum number of retries." in the summary line, which is
exactly what a red-first suite may not do.
"""

import ast
import pathlib
import types
import unittest

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Faithfulness guard (#630): the stand-in must carry every attribute the
# shipping method reads, or this suite is testing a fiction.
# --------------------------------------------------------------------------


SCHEDULE_BATCH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "schedule_batch.py"
)


def _attrs_touched():
    """Read the SOURCE FILE, not ``inspect.getsource`` on the bound method.

    A satisfiability harness that replaces ``ScheduleBatch.merge_batch`` to
    prove these assertions are reachable makes ``inspect.getsource`` raise
    ``OSError: could not get source code`` -- measured while writing this
    suite. The file on disk is the authority anyway.
    """
    fn = None
    tree = ast.parse(SCHEDULE_BATCH.read_text())
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "ScheduleBatch":
            for n in cls.body:
                if isinstance(n, ast.FunctionDef) and n.name == "merge_batch":
                    fn = n
    assert fn is not None, "ScheduleBatch.merge_batch not found in the tree"
    tree = fn
    on_self, on_other = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if n.value.id == "self":
                on_self.add(n.attr)
            elif n.value.id == "other":
                on_other.add(n.attr)
    return on_self, on_other


class _Orchestrator:
    """Penalizers are not index-parallel state; merging them is a no-op here."""

    def merge(self, other):
        pass


class _Req:
    """The minimum a Req is to merge_batch: an rid, and object identity."""

    def __init__(self, rid):
        self.rid = rid

    def __repr__(self):
        return f"<Req {self.rid}>"


class _Batch:
    """A ScheduleBatch-shaped stand-in; the METHOD under test is the real one."""

    def batch_size(self):
        return len(self.reqs)


def _sampling_info(n, *, custom=True):
    s = SamplingBatchInfo.__new__(SamplingBatchInfo)
    s.penalizer_orchestrator = _Orchestrator()
    s.has_custom_logit_processor = custom
    s.custom_logit_processor = None
    s.custom_params = [{"slot": i} for i in range(n)]
    s.logit_bias = None
    s.device = "cpu"
    # __len__ reads temperatures, so it must be genuinely per-request.
    s.temperatures = torch.ones(n, 1)
    s.top_ps = torch.ones(n)
    s.top_ks = torch.ones(n, dtype=torch.int32)
    s.min_ps = torch.zeros(n)
    s.sampling_seed = None
    s.is_all_greedy = True
    s.is_any_greedy = False
    s.need_top_p_sampling = False
    s.need_top_k_sampling = False
    s.need_min_p_sampling = False
    return s


def _batch(
    reqs,
    *,
    return_logprob=True,
    encoder_decoder=False,
    multimodal=True,
    custom_params=True,
    tag="",
):
    b = _Batch()
    n = len(reqs)
    b.reqs = list(reqs)
    b.sampling_info = _sampling_info(n, custom=custom_params)
    b.model_config = types.SimpleNamespace(is_encoder_decoder=encoder_decoder)
    b.encoder_lens = torch.arange(n) if encoder_decoder else None
    b.encoder_lens_cpu = [f"{tag}enc{i}" for i in range(n)]
    b.req_pool_indices = torch.arange(n)
    b.req_pool_indices_cpu = torch.arange(n)
    b.seq_lens = torch.arange(n)
    b.orig_seq_lens = torch.arange(n)
    b.seq_lens_cpu = torch.arange(n)
    b.out_cache_loc = None
    b.seq_lens_sum = None
    b.input_ids = None
    b.mamba_track_indices = None
    b.mamba_track_mask = None
    b.mamba_track_seqlens = None
    b.return_logprob = return_logprob
    b.top_logprobs_nums = [f"{tag}top{i}" for i in range(n)]
    b.token_ids_logprobs = [f"{tag}tid{i}" for i in range(n)]
    b.multimodal_inputs = [f"{tag}mm{i}" for i in range(n)] if multimodal else None
    b.has_grammar = False
    b.return_hidden_states = False
    b.is_prefill_only = True
    b.spec_info = None
    b.spec_algorithm = types.SimpleNamespace(name="none")
    return b


# The members this suite holds to index-parallelism with ``reqs``. Every one
# is grown by exactly one entry per merged request by the shipping method.
LIST_MEMBERS = (
    ("top_logprobs_nums", "schedule_batch.py:4747/:4750/:4753"),
    ("token_ids_logprobs", "schedule_batch.py:4748/:4751/:4754"),
    ("multimodal_inputs", "schedule_batch.py:4757"),
    ("encoder_lens_cpu", "schedule_batch.py:4718"),
)
TENSOR_MEMBERS = (
    ("req_pool_indices", "schedule_batch.py:4719-4721"),
    ("req_pool_indices_cpu", "schedule_batch.py:4722-4724"),
    ("seq_lens", "schedule_batch.py:4725"),
    ("orig_seq_lens", "schedule_batch.py:4726"),
    ("seq_lens_cpu", "schedule_batch.py:4741"),
)
SAMPLING_LIST_MEMBERS = (("custom_params", "sampling_batch_info.py:405"),)
SAMPLING_TENSOR_MEMBERS = (
    ("temperatures", "sampling_batch_info.py:419-429"),
    ("top_ps", "sampling_batch_info.py:419-429"),
    ("top_ks", "sampling_batch_info.py:419-429"),
    ("min_ps", "sampling_batch_info.py:419-429"),
)


ENCODER_ONLY = ("encoder_lens", "encoder_lens_cpu")


def _widths(batch):
    """Every req-parallel member's width, keyed by name, for one batch.

    ``encoder_lens``/``encoder_lens_cpu`` are merged only behind
    ``model_config.is_encoder_decoder`` (schedule_batch.py:4716-4718), so on a
    non-encoder-decoder batch they are present but NOT req-parallel through
    this method. Counting them there would report a breach the method never
    caused -- an instrument measuring something other than what it claims.
    """
    enc = bool(batch.model_config.is_encoder_decoder)
    out = {"reqs": len(batch.reqs)}
    for name, _ in LIST_MEMBERS:
        if name in ENCODER_ONLY and not enc:
            continue
        v = getattr(batch, name, None)
        if v is not None:
            out[name] = len(v)
    for name, _ in TENSOR_MEMBERS:
        if name in ENCODER_ONLY and not enc:
            continue
        v = getattr(batch, name, None)
        if v is not None:
            out[name] = int(v.shape[0])
    for name, _ in SAMPLING_LIST_MEMBERS:
        v = getattr(batch.sampling_info, name, None)
        if v is not None:
            out[f"sampling_info.{name}"] = len(v)
    for name, _ in SAMPLING_TENSOR_MEMBERS:
        v = getattr(batch.sampling_info, name, None)
        if v is not None:
            out[f"sampling_info.{name}"] = int(v.shape[0])
    return out


def _anchor(member):
    for name, where in (
        LIST_MEMBERS + TENSOR_MEMBERS + SAMPLING_LIST_MEMBERS + SAMPLING_TENSOR_MEMBERS
    ):
        if member.endswith(name):
            return where
    return "schedule_batch.py:4755"


class MergeCardinality(unittest.TestCase):
    """A merge may never re-admit a request this batch already holds."""

    def setUp(self):
        self.r = [_Req(f"rid{i}") for i in range(5)]

    # ---- the #630 guard --------------------------------------------------

    def test_stub_is_faithful_to_the_shipping_method(self):
        on_self, on_other = _attrs_touched()
        lhs = _batch(self.r[:2], encoder_decoder=True)
        rhs = _batch(self.r[2:4], encoder_decoder=True)
        missing_self = sorted(a for a in on_self if not hasattr(lhs, a))
        missing_other = sorted(a for a in on_other if not hasattr(rhs, a))
        self.assertEqual(
            (missing_self, missing_other),
            ([], []),
            "the stand-in does not carry every attribute the shipping "
            "merge_batch reads; a stub in this shape certifies its own "
            "assumptions (#630). merge_batch has drifted -- extend the "
            "fixture, do not relax this check.",
        )

    def test_req_parallel_member_census(self):
        """What "index-parallel to reqs" actually covers in this method.

        The briefing and the sweep both say SIX siblings. Measured here on a
        disjoint merge of a 2-request batch into a 2-request batch: every
        member below grows from 2 to 4 in lockstep with ``reqs``. Six of them
        are lists; the rest are ``torch.cat`` on the same axis. A fix that
        de-duplicates ``reqs`` must repair ALL of them, not the six.
        """
        lhs = _batch(self.r[:2], encoder_decoder=True, tag="L")
        rhs = _batch(self.r[2:4], encoder_decoder=True, tag="R")
        before = _widths(lhs)
        ScheduleBatch.merge_batch(lhs, rhs)
        after = _widths(lhs)
        grew = sorted(k for k in after if after[k] == before[k] + 2)
        self.assertEqual(
            after["reqs"], 4, "the disjoint control merge must append both reqs"
        )
        self.assertEqual(
            grew,
            sorted(after),
            f"every member this suite tracks must be req-parallel; "
            f"before={before} after={after}",
        )
        self.assertGreaterEqual(
            len(grew),
            10,
            f"the req-parallel population of merge_batch is larger than the "
            f"six list masks named in every earlier report; measured here: "
            f"{grew}",
        )

    # ---- the defect ------------------------------------------------------

    def test_full_overlap_merge_is_a_no_op(self):
        """The boot-8 shape: a DISTINCT batch object holding resident reqs.

        This is what ``last_mbs[slot]`` becomes once #1189's missing publish
        preserves it: a different ``ScheduleBatch`` whose ``reqs`` are the
        same ``Req`` objects the running batch already holds. Merging it may
        add nothing at all.
        """
        running = _batch(self.r[:3], tag="run")
        stale = _batch(self.r[:3], tag="stale")
        self.assertIsNot(running, stale, "the stale entry is a DISTINCT object")
        before = _widths(running)
        ScheduleBatch.merge_batch(running, stale)
        after = _widths(running)
        distinct = len({id(x) for x in running.reqs})
        self.assertEqual(
            (len(running.reqs), distinct),
            (3, 3),
            f"#1189: merging a batch whose requests are ALREADY resident grew "
            f"reqs to {len(running.reqs)} entries over {distinct} distinct "
            f"Req objects. schedule_batch.py:4755 `self.reqs.extend("
            f"other.reqs)` has no membership test and :4691 `if other is "
            f"self` cannot see a distinct object. On boot 8 this reached "
            f"running_bs=7768 against max_running_requests=8. BOTH numbers "
            f"are asserted on purpose: a dedup that collapses the list to "
            f"fewer than 3 distinct rids is not a fix either.",
        )
        self.assertEqual(
            after,
            before,
            f"a no-op merge must leave every req-parallel member untouched; "
            f"before={before} after={after}",
        )

    def test_repeated_revisit_does_not_ramp(self):
        """One stale entry, visited five times -- the boot-8 ramp in miniature.

        #1189 does not fire once. The slot is revisited every PP cycle and
        ``merge_batch`` runs again each time, so the growth is per-visit and
        unbounded: boot 8 climbed 1027 -> 6648 -> 7768 across three minutes
        while at most three distinct rids existed.
        """
        running = _batch(self.r[:3], tag="run")
        stale = _batch(self.r[:3], tag="stale")
        widths = []
        for _ in range(5):
            ScheduleBatch.merge_batch(running, stale)
            widths.append(len(running.reqs))
        self.assertEqual(
            widths,
            [3, 3, 3, 3, 3],
            f"#1189 ramp: five revisits of one stale slot entry took reqs to "
            f"{widths} over 3 distinct Req objects. The consumer "
            f"`scheduler.py` computes `res = limit - max(0, running_bs - "
            f"parked)`, so this number closes admission outright.",
        )

    def test_partial_overlap_admits_only_the_new_request(self):
        """The case a reqs-only dedup gets WRONG, and the reason for one pass.

        ``other`` holds two already-resident requests and one new one. The
        correct merge appends exactly ONE req and exactly ONE entry to every
        req-parallel member -- and that entry must be the NEW request's, not
        the first of ``other``'s. A dedup applied to ``reqs`` alone appends
        one req and three mask entries, which silently mis-indexes every
        per-request lookup after the merge. That is strictly worse than
        #1189, which is why fact 2 says "same pass or not at all".
        """
        running = _batch(self.r[:3], encoder_decoder=True, tag="run")
        incoming = _batch(
            [self.r[1], self.r[2], self.r[3]], encoder_decoder=True, tag="inc"
        )
        ScheduleBatch.merge_batch(running, incoming)
        self.assertEqual(
            [r.rid for r in running.reqs],
            ["rid0", "rid1", "rid2", "rid3"],
            f"#1189 partial overlap: merging a batch holding rid1/rid2 "
            f"(already resident) plus rid3 (new) produced "
            f"{[r.rid for r in running.reqs]}. Only rid3 may be admitted. "
            f"schedule_batch.py:4755 extends unconditionally.",
        )
        w = _widths(running)
        off = {k: v for k, v in w.items() if v != 4}
        self.assertEqual(
            off,
            {},
            f"index-parallel breach after a partial-overlap merge: "
            f"{ {k: (v, _anchor(k)) for k, v in off.items()} } against "
            f"len(reqs)={len(running.reqs)}. Every member must carry exactly "
            f"one entry per resident request.",
        )
        self.assertEqual(
            running.top_logprobs_nums,
            ["runtop0", "runtop1", "runtop2", "inctop2"],
            "the appended mask entry must come from the NEW request's "
            "position in `other` (index 2), not from other's index 0",
        )
        self.assertEqual(
            running.token_ids_logprobs,
            ["runtid0", "runtid1", "runtid2", "inctid2"],
        )
        self.assertEqual(
            running.multimodal_inputs,
            ["runmm0", "runmm1", "runmm2", "incmm2"],
        )
        self.assertEqual(
            running.encoder_lens_cpu,
            ["runenc0", "runenc1", "runenc2", "incenc2"],
            "schedule_batch.py:4718 must take the NEW request's row too",
        )
        self.assertEqual(
            [p["slot"] for p in running.sampling_info.custom_params],
            [0, 1, 2, 2],
            "sampling_batch_info.py:405 custom_params must take the new "
            "request's row, not other's first row",
        )

    # ---- the three logprob arms, one test each ---------------------------

    def _overlap_arm(self, self_lp, other_lp):
        running = _batch(self.r[:3], return_logprob=self_lp, tag="run")
        stale = _batch(self.r[:3], return_logprob=other_lp, tag="stale")
        before_top = list(running.top_logprobs_nums)
        before_tid = list(running.token_ids_logprobs)
        ScheduleBatch.merge_batch(running, stale)
        return running, before_top, before_tid

    def _assert_arm(self, arm, anchor, self_lp, other_lp):
        running, before_top, before_tid = self._overlap_arm(self_lp, other_lp)
        n = len(running.reqs)
        self.assertEqual(
            (len(running.top_logprobs_nums), len(running.token_ids_logprobs), n),
            (3, 3, 3),
            f"#1189 arm {arm} ({anchor}): a full-overlap merge grew the "
            f"index-parallel logprob masks to "
            f"{len(running.top_logprobs_nums)}/"
            f"{len(running.token_ids_logprobs)} against reqs={n} over 3 "
            f"distinct Req objects. This arm must be repaired in the SAME "
            f"pass as schedule_batch.py:4755 -- a dedup on reqs alone leaves "
            f"these lists longer than reqs and mis-indexed.",
        )
        self.assertEqual(running.top_logprobs_nums, before_top)
        self.assertEqual(running.token_ids_logprobs, before_tid)

    def test_arm1_both_sides_return_logprob(self):
        self._assert_arm("1", "schedule_batch.py:4747/:4748", True, True)

    def test_arm2_self_only_the_unnamed_elif(self):
        """``elif self.return_logprob:`` at :4749.

        The arms at :4750/:4751 pad with ``[0] * len(other.reqs)`` and
        ``[None] * len(other.reqs)`` -- they are index-parallel by
        construction and no earlier report named them. If a fix filters
        ``other.reqs`` without changing the multiplier here, the padding is
        computed from the UNFILTERED count.
        """
        self._assert_arm("2", "schedule_batch.py:4750/:4751", True, False)

    def test_arm3_other_only_rebuilds_from_len_self_reqs(self):
        """``elif other.return_logprob:`` at :4752 -- the REBUILD arm.

        :4753/:4754 do not extend, they REPLACE both lists with
        ``[0] * len(self.reqs) + other.top_logprobs_nums``, computed BEFORE
        :4755 extends ``reqs``. A fix that filters at :4755 without touching
        this arm produces lists sized to the unfiltered ``other``.
        """
        self._assert_arm("3", "schedule_batch.py:4753/:4754", False, True)

    def test_encoder_lens_cpu_stays_parallel(self):
        """schedule_batch.py:4718, behind ``model_config.is_encoder_decoder``."""
        running = _batch(self.r[:3], encoder_decoder=True, tag="run")
        stale = _batch(self.r[:3], encoder_decoder=True, tag="stale")
        before = list(running.encoder_lens_cpu)
        ScheduleBatch.merge_batch(running, stale)
        # ORDER MATTERS. Today this list and ``reqs`` BOTH grow to 6, so a
        # length-only assertion is GREEN on the defect -- the exact "worthless
        # as written" shape. The value assertion is therefore first and is
        # what turns this test red today; the length assertion below is the
        # trap for a fix that de-duplicates ``reqs`` alone.
        self.assertEqual(
            running.encoder_lens_cpu,
            before,
            f"schedule_batch.py:4718 encoder_lens_cpu is index-parallel to "
            f"reqs and took {len(running.encoder_lens_cpu) - len(before)} new "
            f"entries on a merge that admitted no new request "
            f"({running.encoder_lens_cpu}). It sits behind "
            f"model_config.is_encoder_decoder, so it is invisible to any test "
            f"that never sets that flag.",
        )
        self.assertEqual(
            len(running.encoder_lens_cpu),
            len(running.reqs),
            "index-parallel breach: a fix that filtered reqs at :4755 without "
            "filtering :4718 leaves encoder_lens_cpu longer than reqs",
        )

    def test_custom_params_stays_parallel(self):
        """sampling_batch_info.py:405, behind ``has_custom_logit_processor``.

        The delegation runs at schedule_batch.py:4713, BEFORE reqs is
        extended, so a fix placed at :4755 does not reach it at all.
        """
        running = _batch(self.r[:3], custom_params=True, tag="run")
        stale = _batch(self.r[:3], custom_params=True, tag="stale")
        before = [p["slot"] for p in running.sampling_info.custom_params]
        ScheduleBatch.merge_batch(running, stale)
        got = [p["slot"] for p in running.sampling_info.custom_params]
        # Same ordering rule as encoder_lens_cpu: today custom_params and reqs
        # grow together, so only the VALUE assertion can be red on the defect.
        self.assertEqual(
            got,
            before,
            f"sampling_batch_info.py:405 custom_params took "
            f"{len(got) - len(before)} new rows ({got}) on a merge that "
            f"admitted no new request. ScheduleBatch.merge_batch delegates to "
            f"SamplingBatchInfo.merge_batch at schedule_batch.py:4713 -- "
            f"BEFORE :4755 -- so a filter applied only at :4755 never reaches "
            f"this list at all.",
        )
        self.assertEqual(
            len(got),
            len(running.reqs),
            "index-parallel breach: custom_params is longer than reqs",
        )

    # ---- controls that are GREEN today and must stay green ---------------

    def test_disjoint_merge_still_appends_everything(self):
        """The ordinary merge must be untouched by any #1189 fix.

        This is the over-reach guard: a fix that suppresses duplicates must
        not suppress a legitimate admission. Green today, green after.
        """
        running = _batch(self.r[:2], encoder_decoder=True, tag="run")
        incoming = _batch(self.r[2:4], encoder_decoder=True, tag="inc")
        ScheduleBatch.merge_batch(running, incoming)
        self.assertEqual(
            [r.rid for r in running.reqs], ["rid0", "rid1", "rid2", "rid3"]
        )
        w = _widths(running)
        self.assertEqual(
            {k: v for k, v in w.items() if v != 4},
            {},
            f"a disjoint merge must leave every member at 4: {w}",
        )
        self.assertEqual(
            running.top_logprobs_nums, ["runtop0", "runtop1", "inctop0", "inctop1"]
        )

    def test_self_merge_guard_is_preserved(self):
        """schedule_batch.py:4691 stays; it is ADDED TO, never replaced.

        The identity guard addresses the 2026-08-09 self-merge death (a slot
        rebound running_batch and last_batch to ONE object; the resident
        count walked 2**23 -> 2**25 in three seconds). #1189 is the distinct-
        object case. Both refusals must survive the fix.
        """
        running = _batch(self.r[:3], tag="run")
        before = _widths(running)
        ScheduleBatch.merge_batch(running, running)
        self.assertEqual(
            _widths(running),
            before,
            "merge_batch(b, b) must remain a refusal, not a doubling",
        )


if __name__ == "__main__":
    unittest.main()

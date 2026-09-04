"""#1176 round 6 -- THE STORE WITNESS MEASURES AND NEVER STOPS THE GROUP.

WHY THIS FILE EXISTS. The store witness was reformulated SIX times --
4b277fff25, be3ec1760b, 1634bc3d28, 8e73b2a9cc, c6fccf75f0, ac4b1d4bf8 -- and
every round was returned FIX_REQUIRED by adversarial reviewers who EXECUTED
breaking inputs. Each round failed in one of exactly two fatal directions:

  * FALSE STOP: it raised on a request whose stamped prefix was genuinely
    present. A raise is a PP0 group STOP, so a false one kills the boot
    (rounds 1, 3, 4, 5 all reproduced it; boot weg1b6 died at 16:08:22 on
    rid 1e95e023 with stamped=6008, matched=3456, loaded=0 -- a shortfall of
    2552 well INSIDE the 4096 one-chunk allowance).
  * LICENSING: it answered "hit" where an ancestor had refused, permitting a
    re-prefill beyond the one-chunk #939 allowance (rounds 2, 4, 5).

The root cause is established, not suspected: the witness's ENTIRE read set is
rank-local with lifecycle holes (host_hit_length written at two sites and never
cleared by reset_for_retract; _prefetch_registered_prefix_len one writer no
clearer; cached_prompt_tokens_at_retract one writer no clearer; matched/loaded
span-relative and MIN-reduced only under tp_world_size > 1, while the shipping
form is --tp-size 1 --pp-size 3). A correct rank-local arithmetic predicate
over that read set was attempted six times and falsified six times.

THE DECISION (operator, deliberate): the witness stops being a control-flow
actor and becomes a LOUD, MEASURED OBSERVATION; #939 enforcement moves to the
acceptance instrument (/spinning/gpu-arb/accept_weg1_1068.py check A14), which
fails the acceptance on a MEASURED breach. That removes both fatal directions
by construction -- an observation cannot false-STOP a boot, and an observation
cannot split the ranks.

WHAT THIS FILE PINS:
  A  INERTNESS   no raise on any reachable witness path; the control-flow
                 apparatus (StoreWitnessContradiction, witness_stop_authority,
                 assert_store_witness_at_admission, may_stop) is GONE from the
                 tree, and the admission sites bind no verdict.
  B  IDENTITY    the prefill/admission gate decides identically with the
                 observation present and with it removed entirely.
  C  MEASUREMENT one line per witnessed rid with every term the six rounds
                 argued about and ALL FOUR candidate readings named, no one of
                 them called "the" presence, plus the named rate-limit.
"""

import ast
import inspect
import subprocess
import types
import unittest

import torch

from sglang.srt.managers import phase_purity
from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    observe_store_witness,
    seam_transport_premise_holds,
    store_witness,
    witness_readings,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

CHUNK = 4_096
#: The weg1b6 16:08:22 record, verbatim: PP0 was REAPED at the 7.87 s budget
#: with completed=3456 matched=3456 loaded=0 against stamp 6008.
B6_STAMP = 6_008
B6_MATCHED = 3_456

#: The names round 6 DELETES. Every one of them was part of the apparatus that
#: turned a rank-local reading into a group STOP.
DELETED = (
    "StoreWitnessContradiction",
    "witness_stop_authority",
    "assert_store_witness_at_admission",
)

WITNESS_FUNCS = ("_witness_from_outcome", "witness_readings", "observe_store_witness")


def _witness_reachable(module, entries=WITNESS_FUNCS):
    """Every module-level function TRANSITIVELY reachable from the witness
    entry points, by name, out of the module AST.

    WHY A CLOSURE AND NOT A LIST (review round 7, non-blocking finding): the
    round-6 guard named exactly three function bodies, and a reviewer's mutant
    that inserted a `raise` into `_store_witness_allowance` -- a transitive
    callee of `observe_store_witness` -- walked straight through it. A callee
    added tomorrow is covered here without editing the test.
    """
    tree = ast.parse(inspect.getsource(module))
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    seen, stack = set(), list(entries)
    while stack:
        name = stack.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                stack.append(node.func.id)
    return seen


def _req(rid="1e95e023", *, stamp=B6_STAMP, tokens=8192, resident=0, host_hit=0,
         registered=None, seam_epoch=3):
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(tokens)),
        # #1176 round 7 (review B4): the production request carries a
        # torch.Tensor here (schedule_batch.py:1460), and the round-6
        # suite built a Python list -- so no test in this file could see
        # the reader raise on the real type. Every scenario below now
        # runs on the production type.
        prefix_indices=torch.arange(resident, dtype=torch.int64),
        host_hit_length=host_hit,
        storage_hit_length=0,
    )
    if registered is not None:
        r._prefetch_registered_prefix_len = registered
    setattr(r, SEAM_READMIT_ATTR, seam_epoch)
    setattr(r, SEAM_GRANT_CONSUMED_ATTR, False)
    return r


def _sched(reqs, *, pending=(), outcomes=None, phase="tp", pp_rank=0):
    pool = types.SimpleNamespace(size=100)
    pool.available_size = lambda: 50
    tree = types.SimpleNamespace(
        root_node=types.SimpleNamespace(children={}),
        cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        enable_storage=True,
        ongoing_prefetch={rid: object() for rid in pending},
        prefetch_loaded_tokens_by_reqid=dict(outcomes or {}),
        prefetch_threshold=256,
        _prefetch_chunk_tokens=CHUNK,
    )
    return types.SimpleNamespace(
        tree_cache=tree,
        waiting_queue=list(reqs),
        phase_flip_runtime=types.SimpleNamespace(phase=phase),
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3),
    )


#: The scenario matrix both halves of the identity proof run over. Each entry
#: is (label, outcome, req kwargs) and covers a term some round argued about.
MATRIX = (
    ("boot6_reaped_pp0", PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True,
                                         matched=B6_MATCHED), {}),
    ("boot6_sibling_loaded", PrefetchOutcome(5_971, hit_tokens=B6_STAMP, probed=True,
                                             matched=37), {}),
    ("probed_miss_beside_stamp", PrefetchOutcome(0, hit_tokens=0, probed=True), {}),
    ("reaped_unprobed", PrefetchOutcome(0, hit_tokens=0, probed=False), {}),
    ("header_only", PrefetchOutcome(0, hit_tokens=40, probed=True), {}),
    ("bare_int_zero", 0, {}),
    ("bare_int_positive", 7, {}),
    ("device_resident_whole_prefix", PrefetchOutcome(0, hit_tokens=0, probed=True),
     {"resident": B6_STAMP}),
    ("host_half_only", PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True),
     {"host_hit": B6_STAMP}),
    ("registered_head", PrefetchOutcome(42, hit_tokens=B6_STAMP, probed=True, matched=8),
     {"resident": 100, "host_hit": 5_900, "registered": 6_000}),
    ("cold_no_stamp", PrefetchOutcome(0, hit_tokens=0, probed=True), {"stamp": 0}),
)


class A_TheWitnessIsInert(CustomTestCase):
    """No reachable witness path can stop the group any more."""

    def test_the_control_flow_apparatus_is_deleted_from_the_module(self):
        for name in DELETED:
            with self.subTest(name):
                self.assertFalse(
                    hasattr(phase_purity, name),
                    f"{name} survived round 6; the raise apparatus must be DELETED, "
                    "not neutered in place",
                )

    def test_no_witness_function_contains_a_raise(self):
        """STRUCTURAL, not behavioural: a `raise` anywhere in these three
        functions is a re-introduced group STOP. This is the check the
        re-introduction mutant must trip."""
        tree = ast.parse(inspect.getsource(phase_purity))
        found = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in WITNESS_FUNCS:
                found[node.name] = [
                    n for n in ast.walk(node) if isinstance(n, ast.Raise)
                ]
        self.assertEqual(sorted(found), sorted(WITNESS_FUNCS))
        for name, raises in found.items():
            with self.subTest(name):
                self.assertEqual(
                    raises, [], f"{name} raises -- an observation may never STOP a group"
                )

    def test_no_function_reachable_from_the_witness_contains_a_raise(self):
        """WIDER THAN THE THREE ENTRY POINTS (review round 7). The round-6
        guard scoped itself to three function bodies; a `raise` inside a
        TRANSITIVE callee reaches the admission path just as well and survived
        it. The closure is derived from the AST, so it cannot go stale."""
        reachable = _witness_reachable(phase_purity)
        self.assertIn(
            "_store_witness_allowance",
            reachable,
            "the closure must reach the allowance helper -- that is the callee "
            "a reviewer's mutant raised from, unseen by the three-name guard",
        )
        tree = ast.parse(inspect.getsource(phase_purity))
        bodies = {
            n.name: n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in sorted(reachable):
            with self.subTest(name):
                self.assertEqual(
                    [n for n in ast.walk(bodies[name]) if isinstance(n, ast.Raise)],
                    [],
                    f"{name} is reachable from the observation and raises -- "
                    "an observation may never STOP a group",
                )

    def test_no_orphan_reference_survives_in_production(self):
        """PRODUCTION is where a leftover name would matter: an import of a
        deleted symbol is an ImportError at boot, and a surviving `may_stop=`
        argument is a surviving verdict. Under test/ the same tokens are
        legitimate as NEGATIVE assertions (this file and the #1157 ratchet
        both name them to keep them dead), so test/ is checked for BINDINGS
        instead -- an import or a call, never a mention."""
        root = phase_purity.__file__.split("/python/sglang/")[0]
        prod = subprocess.run(
            ["grep", "-rn", "-e", "StoreWitnessContradiction",
             "-e", "witness_stop_authority", "-e", "assert_store_witness_at_admission",
             "-e", "may_stop", f"{root}/python"],
            capture_output=True, text=True,
        ).stdout
        self.assertEqual(
            [ln for ln in prod.splitlines() if ln.strip()],
            [],
            "the deleted STOP apparatus still has a reference under python/",
        )
        bound = subprocess.run(
            ["grep", "-rnE",
             r"(import|^\s*from).*(StoreWitnessContradiction|witness_stop_authority"
             r"|assert_store_witness_at_admission)"
             r"|(StoreWitnessContradiction|witness_stop_authority"
             r"|assert_store_witness_at_admission)\(|may_stop=",
             f"{root}/test"],
            capture_output=True, text=True,
        ).stdout
        self.assertEqual(
            [
                ln for ln in bound.splitlines()
                if ln.strip() and "test_1176_witness_inert_r6.py" not in ln
            ],
            [],
            "a test still imports or calls the deleted STOP apparatus",
        )

    def test_no_admission_or_probe_path_catches_a_witness_contradiction(self):
        src = inspect.getsource(Scheduler)
        self.assertNotIn("StoreWitnessContradiction", src)
        self.assertNotIn("witness_stop_authority", src)

    def test_the_admission_arms_observe_and_bind_nothing(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertEqual(
            src.count(
                "observe_store_witness(self, req, loaded_tokens, self.tree_cache)"
            ),
            2,
            "both admission arms (PP0 and TP) must OBSERVE with the same call",
        )
        self.assertNotIn(
            "= observe_store_witness(", src,
            "the observation's value must not be bound: it decides nothing",
        )

    def test_the_whole_matrix_classifies_without_raising(self):
        """Every record the six rounds argued about, through both the reader
        and the emitter. Nothing raises; every state is a declared one."""
        for label, outcome, kw in MATRIX:
            with self.subTest(label):
                r = _req(**kw)
                s = _sched([r], outcomes={r.rid: outcome})
                state = store_witness(s, r)
                self.assertIn(state, phase_purity.WITNESS_STATES)
                observe_store_witness(s, r, outcome, s.tree_cache)

    def test_no_state_named_contradiction_survives(self):
        """The verdict word that WAS the STOP is gone from the vocabulary, and
        no record in the matrix can produce it. Round 5 returned
        "contradiction" for `probed_miss_beside_stamp`; the caller turned that
        word into a raise."""
        self.assertNotIn("contradiction", phase_purity.WITNESS_STATES)
        for label, outcome, kw in MATRIX:
            with self.subTest(label):
                r = _req(**kw)
                s = _sched([r], outcomes={r.rid: outcome})
                self.assertNotEqual(store_witness(s, r), "contradiction")

    def test_the_boot6_record_classifies_and_the_premise_stands(self):
        """THE SPECIMEN, as a REGRESSION pin -- and it is green on the parent
        ac4b1d4bf8 too, which is stated here rather than hidden.

        weg1b6 died on this record, but NOT on the local classification: round
        5 already read matched here (p_reg = 0 + 3456, shortfall 2552 <= 4096
        allowance) and answered "hit" rank-locally. The STOP came from the
        group-MIN of `loaded` alone at the admission assertion this round
        DELETES -- a reduction a single-rank hermetic call cannot reach. The
        red-first discrimination for that lives in
        `test_the_control_flow_apparatus_is_deleted_from_the_module` and
        `test_no_witness_function_contains_a_raise`; this one guards the
        classification and the premise against a future regression."""
        r = _req()
        out = PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True, matched=B6_MATCHED)
        s = _sched([r], outcomes={r.rid: out})
        self.assertEqual(store_witness(s, r), "hit")
        self.assertTrue(seam_transport_premise_holds(s))

    def test_a_malformed_record_is_observed_without_raising(self):
        """Defensive by construction: every read is getattr-with-default plus
        int coercion, so junk cannot turn the observation into a boot killer."""
        junk = types.SimpleNamespace(rid="junk")
        observe_store_witness(_sched([]), junk, None, None)
        observe_store_witness(None, junk, PrefetchOutcome(1), None)


class B_TheGateDecidesIdenticallyWithoutTheWitness(CustomTestCase):
    """BYTE-IDENTICAL PROOF OBLIGATION: run the gate both ways."""

    def _verdicts(self):
        out = []
        for label, outcome, kw in MATRIX:
            r = _req(**kw)
            s = _sched([r], outcomes={r.rid: outcome})
            out.append((label, store_witness(s, r), bool(seam_transport_premise_holds(s))))
        return out

    def test_removing_the_observation_entirely_changes_no_decision(self):
        with_witness = self._verdicts()
        saved = phase_purity.observe_store_witness
        try:
            del phase_purity.observe_store_witness
            without_witness = self._verdicts()
        finally:
            phase_purity.observe_store_witness = saved
        self.assertEqual(with_witness, without_witness)

    def test_the_gate_never_calls_the_observation(self):
        """Stronger than removal: an emitter that raises must never be
        reached from the premise path."""
        saved = phase_purity.observe_store_witness

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("the gate called the observation")

        try:
            phase_purity.observe_store_witness = _boom
            self._verdicts()
        finally:
            phase_purity.observe_store_witness = saved


class C_TheObservationMeasuresEveryDisputedTerm(CustomTestCase):
    def setUp(self):
        phase_purity._WITNESS_OBSERVATIONS = 0
        phase_purity._WITNESS_BREACHES = 0

    def _emit(self, req, outcome, sched=None):
        s = sched or _sched([req], outcomes={req.rid: outcome})
        with self.assertLogs("sglang.srt.managers.phase_purity", level="WARNING") as cm:
            observe_store_witness(s, req, outcome, s.tree_cache)
        return "\n".join(cm.output)

    def test_the_line_carries_every_term_the_six_rounds_argued_about(self):
        r = _req(resident=100, host_hit=5_900, registered=6_000)
        out = PrefetchOutcome(42, hit_tokens=B6_STAMP, probed=True, matched=8)
        line = self._emit(r, out)
        for term in (
            "rid=1e95e023", "phase=tp", "pp_rank=0", "stamp=6008", "allowance=4096",
            "resident=100", "host_hit=5900", "registered=6000", "matched=8",
            "loaded=42", "materialized=50", "probed=True", "hit_tokens=6008",
        ):
            with self.subTest(term):
                self.assertIn(term, line)

    def test_all_four_readings_are_named_with_shortfall_and_verdict(self):
        r = _req(resident=100, host_hit=5_900, registered=6_000)
        out = PrefetchOutcome(42, hit_tokens=B6_STAMP, probed=True, matched=8)
        line = self._emit(r, out)
        for name in ("p_span", "p_device", "p_reg", "p_current"):
            with self.subTest(name):
                self.assertRegex(line, rf"{name}=\d+/short=-?\d+/(hit|short-by--?\d+)")

    def test_no_reading_is_called_the_presence_and_the_frame_is_undecided(self):
        line = self._emit(_req(), PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True,
                                                  matched=B6_MATCHED))
        self.assertIn("FRAME UNDECIDED", line)
        self.assertNotRegex(line, r"\bthe presence=")
        for sha in ("4b277fff25", "be3ec1760b", "1634bc3d28",
                    "8e73b2a9cc", "c6fccf75f0", "ac4b1d4bf8"):
            with self.subTest(sha):
                self.assertIn(sha, line)

    def test_the_docstring_names_the_undecided_frame_and_the_six_attempts(self):
        doc = (phase_purity._witness_from_outcome.__doc__ or "") + (
            phase_purity.observe_store_witness.__doc__ or ""
        ) + (phase_purity.witness_readings.__doc__ or "")
        self.assertIn("UNDECIDED", doc)
        self.assertIn("Boot 7", doc)
        for sha in ("4b277fff25", "be3ec1760b", "1634bc3d28",
                    "8e73b2a9cc", "c6fccf75f0", "ac4b1d4bf8"):
            with self.subTest(sha):
                self.assertIn(sha, doc)

    def test_the_readings_arithmetic_on_the_boot6_numbers(self):
        r = _req()
        out = PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True, matched=B6_MATCHED)
        rd = witness_readings(r, out, B6_STAMP, CHUNK)["readings"]
        self.assertEqual(rd["p_span"], (3456, 2552, "hit"))
        self.assertEqual(rd["p_device"], (3456, 2552, "hit"))
        self.assertEqual(rd["p_reg"], (3456, 2552, "hit"))
        self.assertEqual(rd["p_current"], (3456, 2552, "hit"))

    def test_the_readings_disagree_when_the_frame_matters(self):
        """The whole reason four readings are printed: on a request whose
        prefix is device+host resident below the registered span, they differ
        by thousands of tokens and by verdict."""
        r = _req(resident=100, host_hit=5_900, registered=6_000)
        out = PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True, matched=0)
        rd = witness_readings(r, out, B6_STAMP, CHUNK)["readings"]
        self.assertEqual(rd["p_span"][0], 0)
        self.assertEqual(rd["p_device"][0], 100)
        self.assertEqual(rd["p_reg"][0], 6_000)
        self.assertEqual(rd["p_current"][0], 6_000)
        self.assertTrue(rd["p_span"][2].startswith("short-by-"))
        self.assertEqual(rd["p_reg"][2], "hit")

    def test_an_unstamped_registration_head_prints_unset(self):
        line = self._emit(_req(), PrefetchOutcome(0, hit_tokens=1, probed=True))
        self.assertIn("registered=unset", line)

    def test_the_rate_limit_is_named_and_holds(self):
        """ROUND 7: the record here is deliberately WITHIN allowance
        (stamp=0). Round 6 drove the cap with a record whose shortfall was
        6008 against a 4096 allowance -- under the round-7 contract that is a
        BREACH and is always emitted, so the old form would have asserted the
        cap on exactly the population the cap no longer governs."""
        head = phase_purity._WITNESS_OBSERVE_HEAD
        every = phase_purity._WITNESS_OBSERVE_EVERY
        self.assertGreater(head, 0)
        self.assertGreater(every, 1)
        r = _req(stamp=0)
        out = PrefetchOutcome(0, hit_tokens=1, probed=True)
        s = _sched([r], outcomes={r.rid: out})
        n = head + 2 * every + 5
        with self.assertLogs("sglang.srt.managers.phase_purity", level="WARNING") as cm:
            for _ in range(n):
                observe_store_witness(s, r, out, s.tree_cache)
        emitted = [ln for ln in cm.output if "STORE WITNESS OBSERVATION" in ln]
        due = [i for i in range(1, n + 1) if i <= head or i % every == 0]
        self.assertEqual(len(emitted), len(due))
        self.assertIn("(n=1)", emitted[0])
        # The LAST line is the last DUE index, not n: the tail between two
        # multiples is silent by design, and the printed n says so.
        self.assertIn(f"(n={due[-1]})", emitted[-1])
        self.assertLess(due[-1], n)


class D_TheReaderSurvivesTheProductionType(CustomTestCase):
    """ROUND 7, blocking review findings B1 and B2.

    B1: `req.prefix_indices` is a torch.Tensor on every real request
    (schedule_batch.py:1460 constructs `torch.empty((0,), dtype=torch.int64)`,
    :2477 and :2616 re-bind it to a tensor slice). Round 6 read it as
    `len(getattr(req, "prefix_indices", None) or ())`, and `x or ()` asks
    `bool(x)` -- which torch REFUSES with a **RuntimeError** for an empty or
    multi-element tensor, a class `except TypeError` does not catch. The read
    was unconditional on both admission arms (scheduler.py:11331 and :11386),
    so the observation raised on the bare admission path: exactly the fatal
    direction round 6 set out to remove by construction.

    B2: for a ONE-element tensor holding index 0, `tensor or ()` is falsy and
    the reader recorded resident=0 instead of 1 -- a silent mis-measurement on
    the one line the whole round exists to produce.

    The tree documents this idiom twice already (schedule_batch.py:3442-3449,
    scheduler.py:8300). Both prior instances sat inside try/except diagnostics;
    round 6 put it on the admission path.
    """

    PRODUCTION_SHAPES = (
        ("empty tensor -- a freshly arrived request",
         torch.empty((0,), dtype=torch.int64), 0),
        ("one row holding index 0 -- the silent mis-read (B2)",
         torch.tensor([0], dtype=torch.int64), 1),
        ("one row holding index 5",
         torch.tensor([5], dtype=torch.int64), 1),
        ("many rows -- a resident prefix",
         torch.arange(4_096, dtype=torch.int64), 4_096),
    )

    def setUp(self):
        phase_purity._WITNESS_OBSERVATIONS = 0
        phase_purity._WITNESS_BREACHES = 0

    def test_the_reader_counts_tensor_rows_and_never_raises(self):
        out = PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True, matched=B6_MATCHED)
        for label, pi, want in self.PRODUCTION_SHAPES:
            with self.subTest(label):
                req = types.SimpleNamespace(
                    rid="prod0001",
                    cached_prompt_tokens_at_retract=B6_STAMP,
                    prefix_indices=pi,
                    host_hit_length=0,
                )
                r = witness_readings(req, out, B6_STAMP, CHUNK)
                self.assertEqual(r["resident"], want)

    def test_the_emitter_never_raises_on_the_production_type(self):
        """The B1 reproduction end to end: the emitter is what the admission
        arms actually call, and it read `prefix_indices` unconditionally."""
        out = PrefetchOutcome(0, hit_tokens=B6_STAMP, probed=True, matched=B6_MATCHED)
        for label, pi, want in self.PRODUCTION_SHAPES:
            with self.subTest(label):
                req = _req()
                req.prefix_indices = pi
                s = _sched([req], outcomes={req.rid: out})
                with self.assertLogs(
                    "sglang.srt.managers.phase_purity", level="WARNING"
                ) as cm:
                    observe_store_witness(s, req, out, s.tree_cache)
                self.assertIn(f"resident={want}", "\n".join(cm.output))

    def test_a_record_whose_prefix_is_not_sized_is_still_observed(self):
        """The defensive posture survives the fix: junk that has no length is
        read as 0, it does not become a boot killer."""
        req = types.SimpleNamespace(rid="junk0001", prefix_indices=7)
        self.assertEqual(witness_readings(req, 0, 0, CHUNK)["resident"], 0)


class E_ABreachIsNeverSuppressedByTheRateLimit(CustomTestCase):
    """ROUND 7, blocking review finding B5.

    Round 6 incremented the counter and returned BEFORE reading the stamp, so
    the rate limit sampled calls blind to what they carried. A reviewer drove
    300 real observations with a genuine 80009-token over-allowance breach at
    call 100: the emitter printed 65 lines and the breach was not among them,
    and the acceptance read the resulting log as **PASS**. Worse, the head is
    burned by ordinary cold admissions BEFORE the first cutover, so on the
    exact boot shape this work exists to survive the sampled population is
    systematically the pre-cutover one -- where a #939 breach cannot occur.

    The rate limit now governs WITHIN-allowance lines only.
    """

    def setUp(self):
        phase_purity._WITNESS_OBSERVATIONS = 0
        phase_purity._WITNESS_BREACHES = 0

    def _drive(self, n_benign, breach_req, breach_out):
        benign = _req(rid="benign01", stamp=0)
        bout = PrefetchOutcome(0, hit_tokens=0, probed=True)
        bs = _sched([benign], outcomes={benign.rid: bout})
        brs = _sched([breach_req], outcomes={breach_req.rid: breach_out})
        with self.assertLogs(
            "sglang.srt.managers.phase_purity", level="WARNING"
        ) as cm:
            for _ in range(n_benign):
                observe_store_witness(bs, benign, bout, bs.tree_cache)
            observe_store_witness(brs, breach_req, breach_out, brs.tree_cache)
        return [ln for ln in cm.output if "STORE WITNESS OBSERVATION" in ln]

    def test_a_breach_inside_the_silent_tail_is_still_printed(self):
        head = phase_purity._WITNESS_OBSERVE_HEAD
        every = phase_purity._WITNESS_OBSERVE_EVERY
        n_benign = head + 5
        self.assertLess(n_benign + 1, every, "the breach must land in the tail")
        self.assertNotEqual(
            (n_benign + 1) % every, 0, "the breach must not be a due index"
        )
        breach = _req(rid="deadbeef", stamp=80_009)
        out = PrefetchOutcome(0, hit_tokens=0, probed=True)
        lines = self._drive(n_benign, breach, out)
        self.assertTrue(
            any("rid=deadbeef" in ln for ln in lines),
            "an over-allowance shortfall was suppressed by the rate limit -- "
            "the acceptance would read that log as PASS with a real breach in "
            "it (review B5, reproduced on 300 real calls)",
        )
        self.assertEqual(phase_purity._WITNESS_BREACHES, 1)

    def test_every_line_carries_the_breach_counter_as_its_own_denominator(self):
        """A14 compares this printed counter against the breach lines it can
        parse: a truncated log then reads as truncated, never as clean."""
        breach = _req(rid="deadbeef", stamp=80_009)
        out = PrefetchOutcome(0, hit_tokens=0, probed=True)
        lines = self._drive(3, breach, out)
        for ln in lines:
            self.assertRegex(ln, r"breaches=\d+")
        self.assertIn("breaches=1", lines[-1])

    def test_a_within_allowance_line_is_still_rate_limited(self):
        """The cap is real and the fix did not remove it."""
        head = phase_purity._WITNESS_OBSERVE_HEAD
        every = phase_purity._WITNESS_OBSERVE_EVERY
        req = _req(stamp=0)
        out = PrefetchOutcome(0, hit_tokens=0, probed=True)
        s = _sched([req], outcomes={req.rid: out})
        n = head + 2 * every + 5
        with self.assertLogs(
            "sglang.srt.managers.phase_purity", level="WARNING"
        ) as cm:
            for _ in range(n):
                observe_store_witness(s, req, out, s.tree_cache)
        emitted = [ln for ln in cm.output if "STORE WITNESS OBSERVATION" in ln]
        due = [i for i in range(1, n + 1) if i <= head or i % every == 0]
        self.assertEqual(len(emitted), len(due))
        self.assertEqual(phase_purity._WITNESS_BREACHES, 0)


if __name__ == "__main__":
    unittest.main()

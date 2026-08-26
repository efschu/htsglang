"""#894 -- the rest of the #889 sweep: knobs that lose in silence.

#889 fixed ONE instance (``--phase-policy-pp-window-s`` made unreachable by a
declared decode-stall SLO). The sweep that found it found three more of the
same class. This suite pins the two that live under ``srt/managers``; the third
(``SGLANG_GGUF_MMQ_DECODE_THRESHOLD`` beating ``--gguf-mmq-decode-threshold``)
is pinned in ``test/registered/unit/quantization/
test_gguf_mmq_env_supersession_894.py``.

S3 -- PHANTOM FIELD (phase_purity.py:300 at the base)
-----------------------------------------------------
``validate_tp_exit_pair`` read ``getattr(policy_cfg, "tp_window_s", 0.0)``.
``PhasePolicyConfig`` has no such field and never had one, so the read is a
constant ``0.0``, the escape ``tp_window > 0`` on the next line can never be
taken, and the refusal is STRICTER than its own source reads. Worse, the
refusal message it raises told operators to "set a bounded TP window" -- a knob
that does not exist, so following the instruction cannot clear the refusal.

The fix is a DELETION, not a new knob: the only bound on TP residency that
exists is ``decode_stall_slo_s``. Adding ``tp_window_s`` for real would mean
adding a config field with no consumer in ``decide``, i.e. manufacturing a
fresh instance of the very class being closed.

Note the runtime BEHAVIOUR does not change for any real configuration -- the
branch was already dead. What changes is that the code and the message stop
describing an escape that is not there.

S4 -- SILENT NARROWING AND SILENT DISCARD (min_free_slots_delayer.py:16-25)
--------------------------------------------------------------------------
``resolve_min_free_slots`` does two things to an explicit user value without a
word: it caps it to the DFlash formula (``min(user_value, formula)``), and --
when ``max_running_requests < 8`` -- it discards it ENTIRELY and returns
``None``, so ``MinFreeSlotsDelayer`` is never built and the admission gate the
operator asked for does not exist. There is no ``logger`` in that module at
all, and its only caller (``scheduler.py``) logged nothing either, so a
``--min-free-slots-delay 6`` that silently became 4, or nothing, was
indistinguishable from one that was honoured.

WARNING, NOT REFUSAL, for both. The discarding input is
``max_running_requests``, which is NOT a parse-time quantity: #287 has the KV
resolver cut it below the ServerArgs ceiling, so whether the value survives is
decided during scheduler init. A refusal there is a dead boot on a
configuration that booted yesterday, traded against a defect whose blast radius
is a wrong belief about admission batching. Same direction as #889.

CLASS RATCHET
-------------
``TestNoPhantomPolicyFields`` is the part meant to outlive these three fixes.
It walks the AST of the phase modules and requires every
``getattr(<policy cfg>, "<name>", <default>)`` to name a field that actually
exists on ``PhasePolicyConfig``. That single assertion is what S3 was: a typo
or a knob that was designed and never built reads as a defaulted constant and
disables the branch that mentions it, forever and silently. It is RED at the
base commit on ``tp_window_s``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import ast
import re
import pathlib
import types
import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MIN_FREE_SLOTS_CAPPED,
    MIN_FREE_SLOTS_DISABLED_SMALL_POOL,
    MIN_FREE_SLOTS_HONOURED,
    MinFreeSlotsDelayer,
    min_free_slots_verdict,
    narrowed_min_free_slots_warning,
    resolve_min_free_slots,
)
from sglang.srt.managers.phase_policy import PhasePolicyConfig
from sglang.srt.managers.phase_purity import (
    PhasePurityError,
    parse_purity,
    validate_tp_exit_pair,
)
from sglang.test.test_utils import CustomTestCase

#  .../<root>/test/registered/unit/managers/<this file>
_SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
_PHASE_MODULES = (
    _SRT / "managers" / "phase_purity.py",
    _SRT / "managers" / "phase_policy.py",
)
# The identifiers that, in these modules, denote a PhasePolicyConfig.
_POLICY_CFG_NAMES = ("policy_cfg", "cfg", "phase_policy_cfg")


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def _purity_cfg(**kw):
    """A policy-config stand-in that carries ONLY real fields.

    Deliberately not ``PhasePolicyConfig(...)``: the point of S3 is that a
    stand-in which invents a field hides the phantom. The base commit's own
    suite (``test_quiescence_no_carry_858.py``) handed ``tp_window_s=0.0`` to
    this validator, which is precisely why nobody noticed the read was dead.
    """
    base = dict(drain_mode_strict=True, decode_stall_slo_s=0.0)
    unknown = set(base) | set(kw)
    phantom = unknown - set(PhasePolicyConfig.__dataclass_fields__)
    assert not phantom, f"test stub invents phantom fields: {sorted(phantom)}"
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestTheTpExitRefusalNamesOnlyKnobsThatExist(CustomTestCase):
    def test_the_message_does_not_advertise_a_tp_window(self):
        """RED AT BASE. The refusal told operators to set a bounded TP window.

        There is no such knob -- not a CLI flag, not an env var, not a
        ``PhasePolicyConfig`` field. An operator who followed the instruction
        would change nothing and hit the same refusal on the next boot.
        """
        with self.assertRaises(PhasePurityError) as ctx:
            validate_tp_exit_pair(parse_purity("strict"), _purity_cfg())
        message = str(ctx.exception)
        self.assertNotIn("tp_window_s", message)
        self.assertNotIn("bounded TP window", message)
        # It must still name the escape that DOES exist, or the refusal is
        # merely quieter rather than truthful.
        self.assertIn("DECODE_STALL_SLO_S", message)

    def test_a_declared_slo_is_still_the_escape(self):
        validate_tp_exit_pair(
            parse_purity("strict"), _purity_cfg(decode_stall_slo_s=180)
        )

    def test_the_deadlocking_triple_is_still_refused(self):
        with self.assertRaises(PhasePurityError):
            validate_tp_exit_pair(parse_purity("strict"), _purity_cfg())

    def test_non_strict_is_not_refused(self):
        validate_tp_exit_pair(parse_purity("off"), _purity_cfg())

    def test_a_phantom_attribute_on_the_config_cannot_open_the_escape(self):
        """The refusal must not be re-openable by a stray attribute.

        A stand-in that carries ``tp_window_s=99`` -- the exact shape the base
        suite used -- must still be refused, because nothing in ``decide``
        consumes such a field. If this ever passes, the escape has come back
        without a consumer.
        """
        cfg = types.SimpleNamespace(
            drain_mode_strict=True, decode_stall_slo_s=0.0, tp_window_s=99.0
        )
        with self.assertRaises(PhasePurityError):
            validate_tp_exit_pair(parse_purity("strict"), cfg)


# ---------------------------------------------------------------------------
# S4
# ---------------------------------------------------------------------------
class TestMinFreeSlotsSaysWhatItDidWithTheValue(CustomTestCase):
    def test_an_honoured_value_is_reported_as_honoured_and_warns_nothing(self):
        value, reason = min_free_slots_verdict(2, 512, is_dflash_family=False)
        self.assertEqual(value, 2)
        self.assertEqual(reason, MIN_FREE_SLOTS_HONOURED)
        self.assertIsNone(
            narrowed_min_free_slots_warning(2, 512, is_dflash_family=False)
        )

    def test_an_unset_value_never_warns(self):
        """A configuration that asked for nothing must not acquire a warning.

        Both the disabled default and the DFlash auto-enable arrive here with
        ``user_value=None``; neither is a narrowing of anything the operator
        wrote, and a warning on them would train readers to skip the line.
        """
        for dflash in (False, True):
            self.assertIsNone(
                narrowed_min_free_slots_warning(None, 512, is_dflash_family=dflash)
            )
            self.assertIsNone(
                narrowed_min_free_slots_warning(None, 4, is_dflash_family=dflash)
            )

    def test_a_capped_value_names_both_numbers(self):
        """RED AT BASE: 6 silently became 4."""
        value, reason = min_free_slots_verdict(6, 512, is_dflash_family=False)
        self.assertEqual(value, 4)
        self.assertEqual(reason, MIN_FREE_SLOTS_CAPPED)
        warning = narrowed_min_free_slots_warning(6, 512, is_dflash_family=False)
        self.assertIsNotNone(warning)
        self.assertIn("6", warning)
        self.assertIn("4", warning)
        self.assertIn("min-free-slots-delay", warning)

    def test_a_discarded_value_says_the_delayer_was_not_built(self):
        """RED AT BASE: with a small pool the flag produced NOTHING, silently.

        This is the dangerous half. A capped value still delays; a discarded
        one leaves ``min_free_slots_delayer`` at ``None``, so the admission
        gate the operator configured is absent from the scheduler entirely.
        """
        value, reason = min_free_slots_verdict(4, 4, is_dflash_family=False)
        self.assertIsNone(value)
        self.assertEqual(reason, MIN_FREE_SLOTS_DISABLED_SMALL_POOL)
        warning = narrowed_min_free_slots_warning(4, 4, is_dflash_family=False)
        self.assertIsNotNone(warning)
        self.assertIn("4", warning)
        self.assertIn("8", warning)
        # It has to say the thing that is actually different, not just that a
        # number moved: no delayer exists.
        self.assertIn("no admission delay", warning.lower())

    def test_the_small_pool_discard_beats_the_cap(self):
        """Order matters: below 8 the value is gone, not merely narrowed."""
        _, reason = min_free_slots_verdict(6, 4, is_dflash_family=False)
        self.assertEqual(reason, MIN_FREE_SLOTS_DISABLED_SMALL_POOL)

    def test_the_dflash_auto_enable_is_not_reported_as_a_narrowing(self):
        value, reason = min_free_slots_verdict(None, 512, is_dflash_family=True)
        self.assertEqual(value, 4)
        self.assertEqual(reason, MIN_FREE_SLOTS_HONOURED)

    def test_resolve_min_free_slots_is_unchanged_on_every_verdict(self):
        """The legacy entry point keeps its exact contract.

        ``scheduler.py`` and the registered #580 suite both call it; the fix
        adds a reporter beside it and must not move a single resolved value.
        """
        cases = [
            (None, 512, False),
            (None, 512, True),
            (None, 8, True),
            (None, 4, True),
            (1, 512, False),
            (2, 512, False),
            (6, 512, False),
            (6, 4, False),
            (4, 4, True),
            (3, 12, False),
            (3, 0, False),
        ]
        for user_value, max_running, dflash in cases:
            with self.subTest(user_value=user_value, max_running=max_running):
                self.assertEqual(
                    resolve_min_free_slots(
                        user_value, max_running, is_dflash_family=dflash
                    ),
                    min_free_slots_verdict(
                        user_value, max_running, is_dflash_family=dflash
                    )[0],
                )

    def test_the_scheduler_actually_logs_the_warning(self):
        """A reporter nobody calls is the defect wearing a fix.

        Read at source level rather than by booting a ``Scheduler``: the wiring
        is the assertion, and it must hold without a GPU, a model or a rank.
        The three parts have to be present together -- the import, the call
        fed from the SAME three inputs the resolver got, and a ``logger`` line
        that emits it.
        """
        src = (_SRT / "managers" / "scheduler.py").read_text()
        self.assertIn("narrowed_min_free_slots_warning,", src)
        tree = ast.parse(src)

        # The assignment, and the statement that immediately follows it.
        assign = follow = None
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body[:-1]):
                if not isinstance(stmt, ast.Assign):
                    continue
                call = stmt.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "narrowed_min_free_slots_warning"
                ):
                    continue
                assign, follow = stmt, body[i + 1]
        self.assertIsNotNone(assign, "scheduler.py never calls the reporter")

        fed = ast.dump(assign.value)
        for want in (
            "min_free_slots_delay",
            "max_running_requests",
            "is_dflash_family",
        ):
            self.assertIn(want, fed, f"the reporter is not fed {want}")

        # ... and it must be EMITTED. A `logger.warning` sitting under a guard
        # that cannot be true is the exact shape #894 is about, so the guard is
        # asserted, not merely the presence of the call.
        self.assertIsInstance(follow, ast.If, "the reporter's result is not used")
        self.assertIsInstance(follow.test, ast.Name)
        self.assertEqual(follow.test.id, assign.targets[0].id)
        emitted = [
            n
            for n in ast.walk(follow)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "warning"
        ]
        self.assertTrue(emitted, "the narrowing is resolved and then dropped")

    def test_the_delayer_itself_is_untouched(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=3)
        self.assertTrue(delayer.should_delay(running_bs=1, num_allocatable_reqs=2))
        self.assertFalse(delayer.should_delay(running_bs=0, num_allocatable_reqs=0))
        self.assertFalse(delayer.should_delay(running_bs=1, num_allocatable_reqs=3))


# ---------------------------------------------------------------------------
# The class ratchet
# ---------------------------------------------------------------------------
def _phantom_getattr_reads(path: pathlib.Path):
    """Every ``getattr(<policy cfg>, "<literal>", <default>)`` naming a field
    that ``PhasePolicyConfig`` does not have."""
    tree = ast.parse(path.read_text())
    fields = set(PhasePolicyConfig.__dataclass_fields__)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        target, name = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id in _POLICY_CFG_NAMES):
            continue
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            continue
        if name.value not in fields:
            out.append((path.name, node.lineno, name.value))
    return out


class TestNoPhantomPolicyFields(CustomTestCase):
    """RED AT BASE on ``phase_purity.py:300`` -- and the reason S3 could exist.

    ``getattr(cfg, "name", default)`` is silent by construction: a name that is
    not there yields the default, so a guard written against it is not merely
    wrong, it is UNREACHABLE and reports nothing. This walks the phase modules
    and refuses any such read whose name is not a real ``PhasePolicyConfig``
    field, which turns the whole failure mode into a collection-time failure
    instead of a wedge someone finds in a 535 s boot log.
    """

    def test_every_defaulted_policy_read_names_a_real_field(self):
        phantoms = []
        for path in _PHASE_MODULES:
            self.assertTrue(path.exists(), f"missing module: {path}")
            phantoms.extend(_phantom_getattr_reads(path))
        self.assertEqual(
            phantoms,
            [],
            "getattr on a PhasePolicyConfig with a name the dataclass does not "
            "have: the default is returned forever, so every branch guarded by "
            "it is dead and silent. Either add the field WITH a consumer, or "
            "delete the branch and the message that advertises it (#894 S3).",
        )

    def test_the_scan_covers_both_phase_modules(self):
        """A ratchet is only as wide as its file list.

        With the phantom gone, dropping a module from ``_PHASE_MODULES`` costs
        nothing today and silently narrows the guard for everything after --
        which is how a ratchet quietly stops ratcheting. Pin the scope.
        """
        names = sorted(p.name for p in _PHASE_MODULES)
        self.assertEqual(names, ["phase_policy.py", "phase_purity.py"])
        for path in _PHASE_MODULES:
            self.assertTrue(path.exists(), f"missing module: {path}")

    def test_the_walker_finds_a_phantom_when_one_exists_probe(self):
        """The ratchet's own can-fail proof -- an empty result must mean 'none',
        not 'the walker matched nothing'."""
        probe = pathlib.Path(__file__).with_name("_probe_894.py")
        probe.write_text(
            "def f(cfg):\n"
            "    a = getattr(cfg, 'pp_window_s', 0.0)\n"
            "    b = getattr(cfg, 'definitely_not_a_field_894', 0.0)\n"
            "    return a, b\n"
        )
        try:
            found = _phantom_getattr_reads(probe)
        finally:
            probe.unlink()
        self.assertEqual([f[2] for f in found], ["definitely_not_a_field_894"])


# ---------------------------------------------------------------------------
# The second class ratchet: prose that announces a supersession must be
# classified, and the classification must say whether the loser is announced
# AT RUNTIME.
# ---------------------------------------------------------------------------
_SERVER_ARGS = _SRT / "server_args.py"

_SUPERSESSION_PROSE = re.compile(
    r"wins over|takes precedence|supersede[sd]?|is inert|silent no-op"
    r"|ignored when|has no effect|capped to",
    re.I,
)

#: Every ``ServerArgs`` field whose own help text says another knob can beat
#: it, classified. Determined 2026-08-26 at base 2b13ba92d1 (= pin 0cd27d957d
#: + #889); a verdict is a SNAPSHOT and carries its date, so re-read it against
#: the pin before trusting it.
#:
#: WARNED             the loss is announced at runtime; the value names where.
#: NOT_A_PAIR         the prose is not one knob silencing another.
#: KNOWN_SILENT       it IS silent, it is named, and it is not fixed here.
#: UPSTREAM           not this fork's code.
_CLASSIFIED = {
    # -- announced at runtime -------------------------------------------------
    "gguf_mmq_decode_threshold": (
        "WARNED",
        "gguf.py:_announce_mmq_env_override (#894 S5)",
    ),
    "phase_policy_pp_window_s": (
        "WARNED",
        "phase_policy.py:superseded_pp_bound_warning (#889)",
    ),
    "min_free_slots_delay": (
        "WARNED",
        "min_free_slots_delayer.py:narrowed_min_free_slots_warning (#894 S4)",
    ),
    "rank_mlp_ratio": ("WARNED", "server_args.py:_handle_uneven_mlp_ratio (#781)"),
    "rank_moe_ratio": ("WARNED", "server_args.py:_handle_uneven_mlp_ratio (#781)"),
    "rank_vocab_ratio": ("WARNED", "server_args.py:_handle_uneven_mlp_ratio (#781)"),
    "uneven_token_vector_role": (
        "WARNED",
        "phase_flip_boot.resolve_effective_flip_token_vector logs the "
        "three-state verdict every boot, and a 'seed' arms "
        "assert_seed_superseded (#797)",
    ),
    # -- not a knob silencing another knob ------------------------------------
    "kv_session_offload_default_spill_class": (
        "NOT_A_PAIR",
        "a per-request `spill_class` overriding a server-wide DEFAULT is what "
        "a default is; nothing the operator set is lost",
    ),
    "uneven_dcp": (
        "NOT_A_PAIR",
        "'superseded in most cases by --rank-kv-ratio' is design guidance "
        "about feature overlap, not a claim that setting both makes one inert",
    ),
    "cuda_graph_config": (
        "NOT_A_PAIR",
        "the JSON REPLACES the convenience flags it expands into; the "
        "expansion is the documented contract, not a loss",
    ),
    "regime_trace": (
        "NOT_A_PAIR",
        "an artifact path that has nothing to write when its producer "
        "(--regime-controller) is off; no second knob competes for it",
    ),
    # -- named debt -----------------------------------------------------------
    "rank_kv_ratio": (
        "KNOWN_SILENT",
        "FOUND BY THE #894 RE-SWEEP, NOT FIXED HERE. "
        "distributed/utils.py:816-838 (`resolve_cp_token_ratios`) lets "
        "SGLANG_UNEVEN_TOKEN_VECTOR win on presence over an explicit "
        "--rank-kv-ratio and logs nothing -- `grep -n 'logger\\.' ` over the "
        "whole function returns zero. Same class as #894 S5. Left open "
        "deliberately: that resolver is documented as a DETERMINISTIC PURE "
        "FUNCTION called from several sites, so the announcement belongs at a "
        "boot-time site and that placement needs its own red-first, not a "
        "logger bolted inside a pure function.",
    ),
    # -- not this fork --------------------------------------------------------
    "max_queued_requests": (
        "UPSTREAM",
        "upstream sglang #7565 (747dd45077); 'ignored when using "
        "disaggregation-mode' is upstream's own prose",
    ),
}

_VERDICTS = {"WARNED", "NOT_A_PAIR", "KNOWN_SILENT", "UPSTREAM"}


def _fields_with_supersession_prose(path: pathlib.Path):
    """ServerArgs fields whose declaration contains supersession prose.

    Keyed by FIELD NAME, not by line number, so the table survives edits above
    it. Scope, stated rather than implied: declarations inside the ``ServerArgs``
    class body only -- prose in module comments and method docstrings is not
    covered, and a supersession documented only there would not be caught.
    """
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
    )
    found = []
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        text = " ".join(
            c.value
            for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        )
        if _SUPERSESSION_PROSE.search(text):
            found.append(node.target.id)
    return found


class TestEverySupersessionInHelpTextIsClassified(CustomTestCase):
    """The ratchet that outlives S3/S4/S5.

    Neither #889 nor #894 was found by a crash. Each was found by reading help
    text that ANNOUNCED a supersession and then asking whether anything said so
    at runtime -- and the answer was no, four times. This makes that question
    unskippable: a new flag whose help says another knob beats it fails this
    test until someone writes down which of the four verdicts applies.

    It deliberately does NOT require ``WARNED``. Requiring it would push the
    next author to delete the prose instead of classifying it, which is the
    opposite of the goal. What it requires is that nobody can add the prose
    and move on.
    """

    def test_no_unclassified_supersession_prose(self):
        fields = _fields_with_supersession_prose(_SERVER_ARGS)
        self.assertTrue(fields, "the prose scanner matched nothing at all")
        unclassified = [f for f in fields if f not in _CLASSIFIED]
        self.assertEqual(
            unclassified,
            [],
            "ServerArgs field(s) whose help text says another knob can beat "
            "them, with no verdict in _CLASSIFIED. #889 and #894 were both "
            "exactly this: prose that announced a supersession the runtime "
            "never mentioned. Classify each as WARNED (name the reporter), "
            "NOT_A_PAIR (say why), KNOWN_SILENT (name the site) or UPSTREAM.",
        )

    def test_the_table_has_no_entries_for_fields_that_no_longer_say_so(self):
        """A stale verdict is worse than no verdict: it reads as a checked
        clean bill for prose that is gone or was rewritten."""
        fields = set(_fields_with_supersession_prose(_SERVER_ARGS))
        stale = sorted(set(_CLASSIFIED) - fields)
        self.assertEqual(stale, [])

    def test_every_verdict_is_one_of_the_four_and_carries_a_reason(self):
        for field, (verdict, reason) in _CLASSIFIED.items():
            with self.subTest(field=field):
                self.assertIn(verdict, _VERDICTS)
                self.assertGreater(len(reason), 20, "a verdict needs a reason")

    def test_the_known_silent_set_is_exactly_what_894_left_open(self):
        """The debt is pinned, so it cannot grow without someone editing this
        line and noticing what they are doing."""
        silent = sorted(f for f, (v, _) in _CLASSIFIED.items() if v == "KNOWN_SILENT")
        self.assertEqual(silent, ["rank_kv_ratio"])

    def test_the_scanner_finds_new_prose(self):
        """Can-fail proof for the ratchet: an unclassified field must be
        reported, or an empty result means nothing."""
        source = (
            "class ServerArgs:\n"
            "    already_fine: A[int, Arg(help='a plain flag')] = 1\n"
            "    brand_new_894: A[int, Arg(help='SGLANG_X wins over this "
            "flag')] = 2\n"
        )
        probe = pathlib.Path(__file__).with_name("_probe_prose_894.py")
        probe.write_text(source)
        try:
            found = _fields_with_supersession_prose(probe)
        finally:
            probe.unlink()
        self.assertEqual(found, ["brand_new_894"])
        self.assertNotIn("brand_new_894", _CLASSIFIED)


if __name__ == "__main__":
    unittest.main()

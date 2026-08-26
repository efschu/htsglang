"""#901 -- ONE knob-resolution authority, and a ratchet against a fifth silo.

WHY THIS SUITE EXISTS
=====================
On 2026-08-26 four reporters for the same question were built in one day
(#894 S4, #894 S5, #896, #897). Each was correct; four of them are a class.
The operator's challenge was the right one -- "many roots means you are
treating symptoms" -- and the structural answer is a single
``srt/knob_resolution`` that resolves a knob's precedence ladder and prints
the result, with the four sites as its callers.

RED AT BASE (b5f3dcbd46). ``sglang.srt.knob_resolution`` does not exist
there, so every test below fails at import. That is the honest statement of
red-first for a NEW authority, and it is why the suite does not stop at the
module: the migration guards further down fail on a tree where the module
exists but a site still carries its own private reporter, and the ratchet
fails on a tree where a new unrouted env read has been added. Those are the
assertions that keep meaning something after this ticket lands.

WHAT IS DELIBERATELY NOT ASSERTED
=================================
That the four sites resolve knobs in the SAME ORDER. They do not, and they
must not. Env-over-flag is design in this fork -- the server logs "restart
with SGLANG_...=" after a calibration run and the environment re-applies the
measured value without re-parsing ServerArgs (SKILL.md Rule 6, rig-runbook
section 2) -- while #781 deliberately made the flag authoritative for the
phase-policy knobs. The authority takes the order as a parameter. A test
that pinned one order would break the feature this fork ships.

Hermetic: pure AST and pure-function plumbing. No device, no model, no
server, no process group.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import math
import pathlib
import unittest

from sglang.srt import knob_resolution as K
from sglang.test.test_utils import CustomTestCase

#  .../<root>/test/registered/unit/server_args/<this file>
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SRT = _ROOT / "python" / "sglang" / "srt"
_SERVER_ARGS = _SRT / "server_args.py"

_AUTHORITY = _SRT / "knob_resolution.py"

#: THE MIGRATED SCOPE, pinned. Four reporters plus the authority itself.
#:
#: HONEST ABOUT ITS SIZE. Measured 2026-08-26 at b5f3dcbd46: 648
#: ``os.environ`` reads in 209 files under ``python/sglang/srt``. This gate
#: covers five of them. A 209-file gate is not a stronger version of this
#: one -- it would have to ship as an allowlist of hundreds of entries, which
#: is a document rather than a guard, and every one of them would be reviewed
#: by nobody. The scope list is pinned by
#: ``test_the_scope_list_is_pinned`` so widening it is a deliberate act with a
#: diff, and the remaining debt is named in the authority's module docstring
#: and in the open register.
_SCOPE = (
    _AUTHORITY,
    _SRT / "managers" / "phase_policy.py",
    _SRT / "managers" / "min_free_slots_delayer.py",
    _SRT / "layers" / "quantization" / "gguf.py",
    _SRT / "distributed" / "utils.py",
)

#: Direct ``os.environ`` reads of a FLAG-TWINNED variable that are allowed to
#: stay unrouted in the scoped modules, each with a reason. Empty, and an
#: empty expectation is the strongest form of this guard: the next author who
#: parks one has to write its name in here rather than append to a list that
#: already had entries. (Same reasoning as #894's KNOWN_SILENT set, which is
#: empty for the same reason.)
_ALLOWED_UNROUTED = {}


# ---------------------------------------------------------------------------
# The interface: scalar, vector, multi-stage
# ---------------------------------------------------------------------------
class TestTheLadderCarriesEveryShape(CustomTestCase):
    """The bar this interface had to clear is #897's case, not #896's.

    #896's recorder resolved ONE scalar from ``getattr(server_args, field)``
    against one env reader. #897 stated, in its own docstring, that the
    recorder could not carry a five-level vector precedence with gcd
    reduction -- and it was right about that helper. An authority that only
    carried the scalar case would have been the fourth silo with a nicer
    name, so the vector and the five-level walk are asserted first.
    """

    def test_a_scalar_ladder_names_the_winning_source(self):
        res = K.resolve_knob(
            [
                K.KnobSource(K.flag_source("decode_stall_slo_s"), 180.0, present=True),
                K.KnobSource(K.env_source("SGLANG_X"), 12.0, present=True),
                K.KnobSource(K.PROVENANCE_DEFAULT, 0.0, present=True),
            ]
        )
        self.assertEqual(res.value, 180.0)
        self.assertEqual(res.source, "flag --decode-stall-slo-s")
        self.assertEqual(res.verdict, K.VERDICT_SUPERSEDED)
        self.assertEqual(res.top_loser.source, "env SGLANG_X")

    def test_a_vector_ladder_compares_with_the_sites_own_equivalence(self):
        """``6,2`` and ``3,1`` are the same ownership split (#897's gcd rule).

        A generic ``==`` would report a loss where none happened, which is the
        noise that teaches operators to skip the line -- so the equivalence
        test is a parameter, not a built-in.
        """

        def gcd_reduced(v):
            g = math.gcd(*v)
            return [x // g for x in v]

        res = K.resolve_knob(
            [
                K.KnobSource(
                    K.env_source("SGLANG_UNEVEN_TOKEN_VECTOR"), [6, 2], present=True
                ),
                K.KnobSource(K.flag_source("rank_kv_ratio"), [3, 1], present=True),
            ],
            normalize=gcd_reduced,
        )
        self.assertEqual(res.value, [6, 2])
        self.assertFalse(res.lost_anything)
        self.assertEqual(res.verdict, K.VERDICT_SOLE)

        lost = K.resolve_knob(
            [
                K.KnobSource(
                    K.env_source("SGLANG_UNEVEN_TOKEN_VECTOR"), [7, 3], present=True
                ),
                K.KnobSource(K.flag_source("rank_kv_ratio"), [3, 7], present=True),
            ],
            normalize=gcd_reduced,
        )
        self.assertTrue(lost.lost_anything)

    def test_five_rungs_are_not_a_special_case(self):
        """#897's documented precedence, walked by the authority.

        env > explicit flag vector > planner capacity seed > budget estimate >
        weights fallback. The winner is the first PRESENT rung and every
        present, reportable rung below it that carries a different value is a
        loss -- with no arm of the walk keyed on the ladder's length.
        """
        res = K.resolve_knob(
            [
                K.KnobSource("env", present=False),
                K.KnobSource("flag", [3, 7], present=True),
                K.KnobSource("seed", [9, 1], present=True),
                K.KnobSource("estimate", [1, 1], present=True, reportable=False),
                K.KnobSource("weights", [2, 1], present=True, reportable=False),
            ]
        )
        self.assertEqual(res.value, [3, 7])
        self.assertEqual(res.source, "flag")
        self.assertEqual([s.source for s in res.superseded], ["seed"])

    def test_an_internal_derivation_is_present_but_not_a_reportable_loss(self):
        """Present and reportable are different questions.

        Caught live while migrating #897: its ladder ends in two rungs the
        resolver derives internally and nobody ever chose. Reporting them as
        superseded produced a SUPERSEDED line on a configuration where
        nothing was lost -- the registered #897 suite failed on exactly that,
        which is the guard doing its job.
        """
        res = K.resolve_knob(
            [
                K.KnobSource("env", "a", present=True),
                K.KnobSource(
                    "internal derivation", "b", present=True, reportable=False
                ),
            ]
        )
        self.assertFalse(res.lost_anything)
        self.assertEqual(res.verdict, K.VERDICT_SOLE)

    def test_a_losing_rung_is_never_read(self):
        """The lazy reader is not an optimisation, it is a correctness rule.

        ``_env_float`` on a malformed value raises. A rung that LOST has no
        business raising, and a resolver that parsed every rung would turn a
        stale unparseable variable into a dead boot on a command line that
        did not use it.
        """
        calls = []

        def _boom():
            calls.append(1)
            raise ValueError("a losing rung must not be parsed")

        res = K.resolve_knob(
            [
                K.KnobSource("flag", 7, present=True),
                K.KnobSource("env", present=False, reader=_boom),
            ]
        )
        self.assertEqual(res.value, 7)
        self.assertEqual(calls, [])

    def test_a_present_loser_that_cannot_be_normalised_is_reported_not_swallowed(self):
        """It cannot be proven equal, so it is a loss. Silence here would be
        the exact failure the module exists to end."""
        res = K.resolve_knob(
            [
                K.KnobSource("env", [1, 2], present=True),
                K.KnobSource("flag", "not-a-vector", present=True),
            ],
            normalize=lambda v: [x * 2 for x in v],
        )
        self.assertTrue(res.lost_anything)

    def test_the_constraint_answers_the_second_question(self):
        """WHO supplied it and WHAT survived of it are different verdicts.

        #894 S4's knob is SOLE (nothing competed for it) and DISCARDED (a pool
        floor dropped it) at the same time. A single verdict field could not
        have said both, and the pair is precisely what made a discarded
        ``--min-free-slots-delay`` indistinguishable from an honoured one.
        """
        res = K.resolve_knob(
            [K.KnobSource(K.flag_source("min_free_slots_delay"), 6, present=True)],
            constraint=lambda value, winner: (None, K.VERDICT_DISCARDED),
        )
        self.assertEqual(res.verdict, K.VERDICT_SOLE)
        self.assertEqual(res.constraint_verdict, K.VERDICT_DISCARDED)
        self.assertIsNone(res.value)
        self.assertEqual(res.requested, 6)

    def test_an_empty_ladder_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            K.resolve_knob([])

    def test_no_present_rung_falls_through_to_the_last_one(self):
        res = K.resolve_knob(
            [
                K.KnobSource("flag", present=False),
                K.KnobSource(K.PROVENANCE_DEFAULT, 3, present=False),
            ]
        )
        self.assertEqual(res.value, 3)
        self.assertEqual(res.source, K.PROVENANCE_DEFAULT)


class TestTheTwoPresenceRulesAreBothDeclared(CustomTestCase):
    """The sites genuinely differ, and the difference is dangerous.

    ``_env_float`` falls through to the default for ``FOO=``, so an empty
    value is NOT a source there. ``gguf._mmq_decode_threshold_enabled``
    short-circuits on ``is not None`` and then tests ``== "1"``, so an empty
    value IS a source there and reads as OFF. Both rules live in the
    authority so the difference is visible in one place instead of being
    re-decided, differently, per module.
    """

    def test_empty_is_not_a_source_under_the_nonempty_rule(self):
        self.assertFalse(K.env_present_nonempty("X", {"X": ""}))
        self.assertTrue(K.env_present_nonempty("X", {"X": "0"}))
        self.assertFalse(K.env_present_nonempty("X", {}))

    def test_empty_is_a_source_under_the_presence_rule(self):
        self.assertTrue(K.env_present("X", {"X": ""}))
        self.assertFalse(K.env_present("X", {}))

    def test_env_provenance_uses_the_safe_rule(self):
        self.assertEqual(K.env_provenance("X", {"X": ""}), K.PROVENANCE_DEFAULT)
        self.assertEqual(K.env_provenance("X", {"X": "5"}), "env X")


class TestThePrintedForms(CustomTestCase):
    def test_the_provenance_field_is_896s_grammar(self):
        self.assertEqual(
            K.provenance_field("decode_stall_slo_s", 180.0, "flag --x", "g"),
            "decode_stall_slo_s=180 from flag --x",
        )

    def test_the_provenance_line_separator_cannot_be_produced_by_a_field(self):
        """A source may itself contain a comma (the phase-policy seam reports
        seed AND estimator state), so ``", "`` is not a separator."""
        line = K.provenance_line(
            "PHASE-POLICY", ["a=1 from flag --a", "b=2 from default"]
        )
        self.assertEqual(
            line, "PHASE-POLICY knob provenance: a=1 from flag --a | b=2 from default"
        )

    def test_every_remedy_says_remove_and_names_the_empty_string_trap(self):
        """One remedy, and the one mistake none of them may advise.

        #894 S5 shipped "unset SGLANG_GGUF_MMQ_DECODE_THRESHOLD", which does
        not close that site's own trap: its presence rule is ``is not None``,
        so ``export FOO=`` leaves the override present and still reading as
        OFF. An operator following the shorter advice would change nothing.
        """
        remedy = K.removal_remedy("SGLANG_FOO")
        self.assertIn("REMOVE SGLANG_FOO", remedy)
        self.assertIn("not by setting it to an empty string", remedy)
        self.assertIn("server_args.py:5607", remedy)

    def test_the_supersession_skeleton_is_897s_line(self):
        line = K.supersession_line(
            "897",
            winner="W",
            subject="S",
            effective="E",
            loss=K.loss_clause("L", "C"),
            presence_rule="P.",
            remedy="R.",
        )
        self.assertEqual(
            line,
            "#897 SUPERSEDED KNOB: W decided S -- E -- and L did not decide "
            "it: C. P. Documented precedence, announced rather than refused: R.",
        )

    def test_the_narrowed_head_spells_the_flag(self):
        self.assertEqual(
            K.narrowed_head("894", "MIN-FREE-SLOTS", "min_free_slots_delay", 6),
            "MIN-FREE-SLOTS #894 NARROWED KNOB: --min-free-slots-delay=6",
        )


class TestTheSharedLatch(CustomTestCase):
    def test_it_says_it_once_and_the_reset_hook_clears_it(self):
        said = []

        class _L:
            def warning(self, fmt, msg):
                said.append(fmt % msg)

        announcer = K.Announcer("probe")
        self.assertTrue(announcer.say(_L(), "hello"))
        self.assertFalse(announcer.say(_L(), "hello"))
        self.assertEqual(said, ["hello"])
        announcer.reset()
        self.assertTrue(announcer.say(_L(), "hello"))

    def test_the_record_message_is_the_line_itself(self):
        """The migrated sites' tests read ``record.getMessage()``; a
        pre-formatted line keeps them byte-comparable with the hand-built
        lines they replace."""
        import logging

        seen = []

        class _H(logging.Handler):
            def emit(self, record):
                seen.append(record.getMessage())

        logger = logging.getLogger("test.knob_resolution.901")
        handler = _H()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            K.Announcer("probe").say(logger, "a line with a % in it")
        finally:
            logger.removeHandler(handler)
        self.assertEqual(seen, ["a line with a % in it"])


# ---------------------------------------------------------------------------
# Migration guards: a site that grows its own reporter back must fail
# ---------------------------------------------------------------------------
def _imports_authority(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "sglang.srt",
            "srt",
        ):
            if any(a.name == "knob_resolution" for a in node.names):
                return True
        if isinstance(node, ast.Import):
            if any(a.name == "sglang.srt.knob_resolution" for a in node.names):
                return True
    return False


class TestEveryMigratedSiteGoesThroughTheAuthority(CustomTestCase):
    """A reporter "simplified" back into its own module is the regression.

    Not a style rule: the whole ticket is that four private mechanisms are a
    class, so the assertion has to be that each site still ROUTES rather than
    that each site still logs.
    """

    def test_all_four_sites_import_it(self):
        for path in _SCOPE:
            if path == _AUTHORITY:
                continue
            with self.subTest(module=path.name):
                self.assertTrue(
                    _imports_authority(path),
                    f"{path.name} no longer imports the knob-resolution authority",
                )
        self.assertTrue(_imports_authority(_SERVER_ARGS), "server_args.py dropped it")

    def test_no_site_keeps_a_hand_rolled_announcement_latch(self):
        """Three of the four had ``_..._announced``/``_..._logged`` module
        booleans plus a reset. One implementation, one reset contract."""
        for name, path in (
            ("gguf.py", _SRT / "layers" / "quantization" / "gguf.py"),
            ("distributed/utils.py", _SRT / "distributed" / "utils.py"),
        ):
            with self.subTest(module=name):
                src = path.read_text()
                self.assertNotIn("_mmq_env_override_logged", src)
                self.assertNotIn("_kv_ratio_supersession_announced", src)

    def test_the_authority_never_reaches_up_into_a_caller(self):
        """It sits below ``managers``, ``distributed`` and ``layers``, and
        importing any of them would recreate the dependency #897 refused."""
        src = _AUTHORITY.read_text()
        for forbidden in (
            "sglang.srt.managers",
            "sglang.srt.distributed",
            "sglang.srt.layers",
            "sglang.srt.server_args",
        ):
            self.assertNotIn(forbidden, src)

    def test_the_authority_logs_nothing_by_itself(self):
        """Resolver purity, which is what let #897's zero-logger contract
        survive the migration. The authority MELDET; the site decides when."""
        tree = ast.parse(_AUTHORITY.read_text())
        bare = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("warning", "info", "error", "debug")
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "logger"
        ]
        self.assertEqual(bare, [], "the authority acquired a module logger")


class TestThe781PublishIsSymmetric(CustomTestCase):
    """The vector and its ROLE are two halves of one decision (#901).

    The role was published UNCONDITIONALLY and the vector only when its flag
    was set, so with the flag unset this process stamped its own default role
    onto a vector some earlier process had left in the shell. The remedy is
    not to stop publishing the role -- its own comment gives the reason it is
    unconditional -- but to VISIT the vector on both branches: publish the
    flag's truth when there is one, and announce the inheritance when there
    is not.
    """

    def test_the_publisher_visits_the_vector_on_both_branches(self):
        src = _SERVER_ARGS.read_text()
        tree = ast.parse(src)
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
        )
        names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("_resolve_token_vector_publication_901", names)

        publisher = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_publish_promoted_781_flags"
        )
        called = {
            n.func.attr
            for n in ast.walk(publisher)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        self.assertIn("_resolve_token_vector_publication_901", called)

        # And the raw conditional write must be gone from the publisher, or
        # both spellings exist and can diverge -- the failure #786 named for
        # the barlink publish, one knob over.
        body = ast.get_source_segment(src, publisher)
        self.assertNotIn('os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"]', body)

    def test_the_role_publish_stays_unconditional(self):
        """Symmetry means both halves are VISITED, not that the role became
        conditional. Making the role conditional would reintroduce exactly
        the stale-shell ambiguity #797 published it unconditionally to close.
        """
        src = _SERVER_ARGS.read_text()
        tree = ast.parse(src)
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
        )
        publisher = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_publish_promoted_781_flags"
        )
        role_writes = [
            n
            for n in ast.walk(publisher)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"
                for t in n.targets
            )
        ]
        self.assertEqual(len(role_writes), 1, "the role publish moved or was dropped")
        # Its enclosing statement must be the function body itself, not an If.
        for node in ast.walk(publisher):
            if isinstance(node, ast.If):
                self.assertNotIn(
                    role_writes[0],
                    node.body,
                    "the role publish became conditional",
                )


# ---------------------------------------------------------------------------
# THE RATCHET
# ---------------------------------------------------------------------------
def _server_args_fields(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
    )
    return {
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def flag_twin(env_name: str, fields) -> str:
    """The ServerArgs field an env var shadows, by this tree's own convention.

    ``SGLANG_FOO_BAR`` <-> ``foo_bar``. Stated rather than implied: this is a
    CONVENTION check, not a registry lookup, so a pair whose spellings differ
    (``SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK`` vs
    ``enable_tp_memory_imbalance_check`` -- note the typo the env kept) is NOT
    detected and will not be caught. Saying so is the point: a guard whose
    reach is overstated is worse than a narrow one, because the next reader
    treats a green run as coverage it does not have.
    """
    for prefix in ("SGLANG_", "HTSGLANG_", "SGL_"):
        if env_name.startswith(prefix):
            candidate = env_name[len(prefix) :].lower()
            if candidate in fields:
                return candidate
    return ""


def _env_reads(path: pathlib.Path):
    """``os.environ`` reads of a NAMEABLE variable in one module.

    Both spellings, ``os.environ.get(X)`` and ``os.environ[X]``, and both
    forms of X: a string literal, or a module-level constant bound to one.
    Resolving the constant matters -- every one of the four migrated sites
    read through a constant (``_MMQ_THRESHOLD_ENV``, ``ENV_REST_STATE``,
    ``ENV_TP_TOK_S``), so a literal-only scan would have found nothing at all
    and reported a clean tree.
    """
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value.value

    def _name_of(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    found = []
    for node in ast.walk(tree):
        target = None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "setdefault")
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and node.args
        ):
            target = node.args[0]
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
        ):
            target = node.slice
        if target is None:
            continue
        name = _name_of(target)
        if name:
            found.append((path.name, node.lineno, name))
    return found


class TestNoNewUnroutedTwinnedEnvRead(CustomTestCase):
    """THE RATCHET, and the part meant to outlive #901.

    A knob with BOTH a CLI flag and an env var is a precedence decision. Every
    one of the four defects fixed on 2026-08-26 was such a pair read directly,
    with the losing half never named. In the migrated modules that read must
    go through the authority, or be written into ``_ALLOWED_UNROUTED`` with a
    reason -- which is the same shape as #894's classification ratchet, and
    for the same reason: the goal is that nobody can add the read and move
    on, not that the read is forbidden.
    """

    def test_no_scoped_module_reads_a_twinned_env_var_directly(self):
        fields = _server_args_fields(_SERVER_ARGS)
        offenders = []
        for path in _SCOPE:
            for module, lineno, env_name in _env_reads(path):
                twin = flag_twin(env_name, fields)
                if not twin:
                    continue
                if (module, env_name) in _ALLOWED_UNROUTED:
                    continue
                offenders.append(f"{module}:{lineno} {env_name} (flag --{twin})")
        self.assertEqual(
            offenders,
            [],
            "a knob that has BOTH a --flag and an env var is a precedence "
            "decision, and reading it directly in a migrated module is how "
            "all four of the 2026-08-26 defects went silent. Route it through "
            "sglang.srt.knob_resolution.resolve_knob, or add it to "
            "_ALLOWED_UNROUTED with a reason.",
        )

    def test_the_scope_list_is_pinned(self):
        """A ratchet is only as wide as its file list, and this one is
        deliberately narrow -- five files out of the 209 that read os.environ.
        Pinning the list is what makes both facts reviewable: dropping a
        module costs nothing today, and the debt cannot be quietly restated as
        finished."""
        self.assertEqual(
            sorted(p.name for p in _SCOPE),
            [
                "gguf.py",
                "knob_resolution.py",
                "min_free_slots_delayer.py",
                "phase_policy.py",
                "utils.py",
            ],
        )
        for path in _SCOPE:
            self.assertTrue(path.exists(), f"missing module: {path}")

    def test_the_allowlist_has_no_stale_entries(self):
        fields = _server_args_fields(_SERVER_ARGS)
        live = set()
        for path in _SCOPE:
            for module, _, env_name in _env_reads(path):
                if flag_twin(env_name, fields):
                    live.add((module, env_name))
        self.assertEqual(sorted(set(_ALLOWED_UNROUTED) - live), [])

    def test_the_authority_itself_never_names_an_env_var(self):
        """It reads only names it is HANDED. A literal in here would be the
        authority growing a private knob of its own, which is the class it
        exists to close, in the one file that must not have it."""
        reads = _env_reads(_AUTHORITY)
        self.assertEqual(
            [f"{m}:{n} {e}" for m, n, e in reads],
            [],
            "knob_resolution.py names an environment variable directly",
        )

    def test_the_scanner_finds_an_unrouted_read_probe(self):
        """CAN-FAIL PROOF. An empty offender list must mean "none", not "the
        walker matched nothing" -- and the constant-resolution half is proven
        separately, because a literal-only walker would report a clean tree on
        every one of the four real sites."""
        fields = _server_args_fields(_SERVER_ARGS)
        self.assertIn("gguf_mmq_decode_threshold", fields)

        probe = pathlib.Path(__file__).with_name("_probe_901.py")
        probe.write_text(
            "import os\n"
            "_E = 'SGLANG_GGUF_MMQ_DECODE_THRESHOLD'\n"
            "_UNTWINNED = 'SGLANG_NOT_A_FLAG_ANYWHERE_901'\n"
            "def by_constant():\n"
            "    return os.environ.get(_E)\n"
            "def by_literal():\n"
            "    return os.environ['SGLANG_UNEVEN_TOKEN_VECTOR']\n"
            "def untwinned():\n"
            "    return os.environ.get(_UNTWINNED)\n"
        )
        try:
            reads = _env_reads(probe)
            twinned = sorted({e for _, _, e in reads if flag_twin(e, fields)})
        finally:
            probe.unlink()
        self.assertEqual(
            twinned,
            ["SGLANG_GGUF_MMQ_DECODE_THRESHOLD", "SGLANG_UNEVEN_TOKEN_VECTOR"],
        )
        # The untwinned name is seen by the walker and correctly NOT reported:
        # an env var with no flag is not a precedence decision.
        self.assertIn("SGLANG_NOT_A_FLAG_ANYWHERE_901", {e for _, _, e in reads})

    def test_the_twin_convention_is_stated_and_bounded(self):
        fields = _server_args_fields(_SERVER_ARGS)
        self.assertEqual(
            flag_twin("SGLANG_GGUF_MMQ_DECODE_THRESHOLD", fields),
            "gguf_mmq_decode_threshold",
        )
        self.assertEqual(flag_twin("SGLANG_NO_SUCH_FLAG_901", fields), "")
        # The documented blind spot, asserted so it is a known bound rather
        # than an assumption: the env kept the "INBALANCE" typo, the field did
        # not, so the convention cannot pair them.
        self.assertEqual(
            flag_twin("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", fields), ""
        )
        self.assertIn("enable_tp_memory_imbalance_check", fields)


class TestTheMigrationDebtIsNamed(CustomTestCase):
    """648 reads in 209 files, five migrated. Named, not implied.

    #880 measured what happens to work that is finished and not written down:
    five posts determined closed on 2026-08-17 stood unchanged nine days
    later. A migration this partial that does not carry its own remainder in
    the module reads, to the next author, as a completed sweep.
    """

    def test_the_authority_docstring_carries_the_remainder(self):
        doc = _AUTHORITY.read_text()
        self.assertIn("648", doc)
        self.assertIn("209", doc)


if __name__ == "__main__":
    unittest.main()

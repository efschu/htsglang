"""#861l: EVERY instrument must be reachable, not just the ones somebody audited.

THE CLASS: desk-written-never-executed -- an instrument is built, unit-tested
against hand-made inputs, and never wired to the path it judges. It then
reports green by never running, and its passing unit tests are exactly what
hides it. #861k's ``layout_conformance.work_layout_verdict`` is the measured
instance: written after W37-D's 258 cold transport batches, it had ZERO
production callers, so W37-G's 27 recompute transports passed with a
conformance count of 0 -- the falsifier for that shape was mute through the
very boot it existed for.

WHY A NEW FILE WHEN ``test_unwired_features_421.py`` ALREADY EXISTS. It does
already carry the doctrine ("a reachability assertion is the only kind of test
that can see the gap between 'the unit works' and 'the product uses it'") and
the AST machinery. Its hole is not its logic but its SHAPE: it is
ENUMERATIVE. Each pin names one feature that a human audit
(``docs/dev/AUDIT_421_UNWIRED.md``) had already found. Anything built after
that audit is outside its reach by construction, which is why four separate
instruments -- ``work_layout_verdict`` (#861), ``thrash_verdict`` (#861e),
``phase_corridor_verdict`` (#784) and the whole ``progress_liveness`` module
(#699) -- went dark under a green suite. This file is the EXHAUSTIVE
complement: it sweeps the whole instrument surface and requires that the dark
set equal a declared allowlist, so a NEW dark instrument fails a gate run
instead of waiting for the next metal boot to not-report something.

The two files divide cleanly and neither is redundant: #421 pins named
absences with per-feature reasoning; this one pins the SIZE AND MEMBERSHIP of
the dark set. Deleting either would restore a real hole.

THE INSTRUMENT WAS TESTED AGAINST A KNOWN STATE, IN BOTH DIRECTIONS, BEFORE
ITS NUMBERS WERE USED -- and it failed the first time. Its first cut indexed
only ``ast.Name`` and ``ast.Attribute`` references and reported
``Scheduler._check_layout_policy_conformance``, ``_check_layout_conformance``
and ``PhaseFlipRuntime._census_ownership_audit`` as dark. All three are
WIRED, through this tree's deliberate defensive idiom

    getattr(self, "_check_layout_policy_conformance", lambda *_: None)(...)

where the callee name exists only as a STRING LITERAL. Publishing that would
have been a false alarm on three live checkers, and a detector that cries
wolf is worse than a missing one. ``_references_in`` therefore counts string
constants as references, and :class:`TestTheSweepItselfCanFail` pins that
behaviour with a synthetic dispatch site so the defect cannot come back.

KNOWN AND DELIBERATE LIMITATION, stated because a silent one is how
instruments rot: matching is BY NAME, not by resolved binding. If the same
function name is defined twice and one of the two is called, both count as
wired. That biases the sweep toward false NEGATIVES (missing a dark
instrument) and never toward false positives (accusing a live one), which is
the correct direction for a gate that people must keep trusting.

Source-structure assertions (AST over the repo tree): hermetic, no torch, no
CUDA, no GPU, no imports of the code under inspection.
"""

import ast
import pathlib
import re
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PKG = _REPO_ROOT / "python" / "sglang"

#: What counts as an INSTRUMENT: a function whose name declares that it judges
#: something rather than computes something. Deliberately name-based -- the
#: property that makes this class dangerous is that the thing READS like a
#: check, so a reader assumes it runs.
_INSTRUMENT_NAME = re.compile(
    r"(_verdict$|^verdict|_violation$|^detect_|_audit$|^audit_"
    r"|_conformance$|_invariant$|^assert_)"
)


def _is_test_path(path: pathlib.Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    padded = "/" + rel
    return (
        rel.startswith("test/")
        or "/test/" in padded
        # ``/tests/`` too: python/sglang/jit_kernel/tests/ is test support that
        # the #421 helper's ``/test/`` check walks straight past. Excluded by
        # PATH rather than by an allowlist entry, because it is not a dark
        # instrument -- it is not production source at all, and allowlisting
        # it would have recorded a fiction.
        or "/tests/" in padded
        or path.name.startswith("test_")
        or rel.startswith("benchmark/")
        or "/benchmark" in padded
    )


def _references_in(tree: ast.AST) -> set:
    """Every name this module could be reaching a function by.

    Attribute access and bare names, PLUS string constants: this tree calls
    optional mixin methods through ``getattr(self, "name", fallback)()``, and
    an index blind to that reports live checkers as dark. See the module
    docstring -- that was this instrument's own first defect.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def _sweep(root: pathlib.Path, repo_root: pathlib.Path = None):
    """``(dark, total)`` -- dark instruments as ``"relpath::name"`` keys.

    An instrument is DARK when its name appears NOWHERE in production source
    outside its own ``def`` line -- its own module included, because a helper
    called only by its own module is still reached by whatever reaches that
    module.
    """
    repo_root = repo_root or _REPO_ROOT
    refs = {}
    defs = []
    for path in sorted(root.rglob("*.py")):
        if repo_root is _REPO_ROOT and _is_test_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        refs[rel] = _references_in(tree)
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and _INSTRUMENT_NAME.search(node.name):
                defs.append((rel, node.name, node.lineno))
    dark = {}
    for rel, name, lineno in defs:
        if not any(name in seen for seen in refs.values()):
            dark[f"{rel}::{name}"] = lineno
    return dark, len(defs)


#: THE DECLARED DARK SET. Every entry is an instrument that exists and is not
#: reached from production. Two kinds live here and the distinction is the
#: whole point of the field:
#:
#:   EXEMPT  -- not an instance of the class. The name matches the pattern but
#:              the thing is not a check over a production path.
#:   DARK    -- a real instance, filed with an owner. Wiring it is work that
#:              has not been done, and the entry is the debt marker.
#:
#: RULE, inherited from ``test_unwired_features_421.py``: when an entry starts
#: failing because somebody WIRED it, DELETE the entry. Never widen it, and
#: never re-key it to keep the gate quiet -- a silently relaxed pin re-creates
#: precisely the failure this file exists to catch.
_DECLARED_DARK = {
    # ---- EXEMPT: matches the name pattern, is not an instrument -------------
    "python/sglang/srt/layers/quantization/utils.py::assert_fp8_all_close": (
        "EXEMPT. Upstream numerical debug helper for comparing two fp8 "
        "tensors by hand. It judges no production path and is not part of "
        "this fork's instrument surface."
    ),
    "python/sglang/srt/model_loader/gguf_deepseek4.py::audit_name_map": (
        "EXEMPT BY DESIGN, and it says so: 'Exposed for the offline audit "
        "(test/registered/unit/model_loader/test_gguf_deepseek4_name_map.py): "
        "a machine with the file but without the kernels can still check that "
        "the mapping covers it.' A test-facing entry point is reached by the "
        "test that is its declared caller."
    ),
    "python/sglang/srt/model_loader/gguf_dflash.py::audit_dflash_name_map": (
        "EXEMPT. Offline comparison tool, run by hand against a GGUF file "
        "during a port ('the round-7c lesson' in its docstring). Not on any "
        "serving path and not claimed to be."
    ),
    # ---- DARK: the class, filed with an owner ------------------------------
    "python/sglang/srt/managers/progress_liveness.py::thrash_verdict": (
        "DARK -- #861e, owner: the strict-batch strand (#857/#861). The "
        "purest instance in the tree: before this gate existed it had exactly "
        "ONE reference repo-wide, its own def line, and no test either. Its "
        "docstring describes W37-D verbatim -- '102 flips, 69 decode batches, "
        "GPU at 98/47/57 %, ZERO completions in seven minutes ... "
        "decode_steps STRICTLY INCREASING, completions FLAT'. That is the "
        "W37-G shape and the #857 acceptance discriminator (COMPLETIONS > 0), "
        "so the detector for the failure the strand is chasing has never run. "
        "Its whole module is dark with it: NOTHING in production imports "
        "sglang.srt.managers.progress_liveness, so #699's assess(), "
        "sample_from_scheduler() and build_liveness_is_active() are dark too "
        "-- and that module exists because '/health answers is the process "
        "up, not is work moving' and 'the existing watchdog is blind to "
        "admission wedges BY CONSTRUCTION' (its own header). The replacement "
        "for two blind instruments was built and never installed. NOTE: the "
        "job it was built for is today done OUT of process by "
        "/root/bin/serving-liveness-monitor.sh -- one payload, two movers, "
        "the in-process one inert; that reconciliation is owed with the "
        "wiring."
    ),
    "python/sglang/srt/managers/corridor_guard.py::phase_corridor_verdict": (
        "DARK -- #784, owner: the VRAM-corridor strand. Seven test call sites "
        "in test_corridor_arming_credit_784.py and zero production callers. "
        "corridor_guard's CONSTANTS are imported widely (corridor_trace, "
        "kv_vmm_backing, model_runner_kv_cache_mixin, server_args) which is "
        "what makes the module read as live; its VERDICT -- the phase-aware "
        "boot acceptance that #784 built after a boot was graded in the "
        "layout that sizes nothing -- is computed by no machine."
    ),
    "python/sglang/srt/disaggregation/draft_kv_canonical.py::assert_compatible": (
        "DARK -- owner: disaggregation. Raises DraftKvLayoutMismatch unless a "
        "peer means the same bytes ('Loud and specific'). Nothing calls it, "
        "so the draft-KV layout version is never actually refused."
    ),
    "python/sglang/srt/disaggregation/nccl/contract.py::assert_compatible": (
        "DARK -- owner: disaggregation. Sibling of the above on the KV "
        "transport identity: raises IncompatiblePeer, and no handshake asks "
        "it to."
    ),
    "python/sglang/srt/distributed/device_communicators/barlink_uniformity.py"
    "::assert_sequences_uniform": (
        "DARK -- owner: barlink. Raises CollectiveSequenceDivergence on the "
        "first mismatch between ranks' collective sequences. Divergence is "
        "the failure mode barlink is most exposed to, and the raiser is "
        "unreached; first_divergence() below it is the half that gets used."
    ),
    "python/sglang/srt/parser/template_detection.py::detect_reasoning_parser": (
        "DARK -- owner: parser. Auto-detection of the reasoning parser from "
        "the chat template is implemented and never consulted, so the "
        "auto path resolves by other means or not at all."
    ),
    "python/sglang/srt/parser/template_detection.py::detect_tool_call_parser": (
        "DARK -- owner: parser. Sibling of the above for the tool-call "
        "parser, same shape, same absence."
    ),
    "python/sglang/srt/planner/lse_merge_gate.py::assert_deterministic": (
        "DARK -- owner: planner. Announces itself as 'GATE 1' and demands "
        "bit-identical results across runs to catch an order-dependent "
        "reduction. Gate 1 is not installed at any gate."
    ),
    "python/sglang/srt/planner/replayssm_identity.py::gate_verdict": (
        "DARK -- owner: planner. 'Decide the enable, with a printable reason "
        "for every refusal' -- and nothing production-side asks for the "
        "decision."
    ),
}


class TestInstrumentSurfaceIsReachable(CustomTestCase):
    """The dark set must be exactly what is declared -- no more, no fewer."""

    def setUp(self):
        self.dark, self.total = _sweep(_PKG)

    def test_the_sweep_actually_found_instruments(self):
        """Guard the guard: a broken walk would pass everything silently."""
        self.assertGreater(
            self.total,
            100,
            "the instrument sweep found almost nothing to judge, which means "
            "the tree walk or the name pattern broke -- not that the tree is "
            f"clean. Found {self.total} instrument-shaped defs.",
        )

    def test_no_undeclared_dark_instrument(self):
        undeclared = {k: v for k, v in self.dark.items() if k not in _DECLARED_DARK}
        self.assertEqual(
            undeclared,
            {},
            "NEW DARK INSTRUMENT(S). Each of these is a function that reads "
            "like a check and that no production code reaches, so it reports "
            "green by never running -- the #861k class, caught this time by a "
            "gate instead of by a metal boot that failed to report anything.\n"
            f"  {sorted(undeclared)}\n"
            "WIRE IT to the path it judges (that is the fix), or -- if it is "
            "genuinely not an instrument -- add it to _DECLARED_DARK with an "
            "EXEMPT reason that says who reaches it instead. Do not delete "
            "the function to quiet the gate: an instrument nobody wired is a "
            "missing check, and deleting it makes the gap permanent.",
        )

    def test_no_declared_entry_has_been_wired(self):
        wired = sorted(k for k in _DECLARED_DARK if k not in self.dark)
        self.assertEqual(
            wired,
            [],
            "GOOD NEWS: these declared-dark instruments now have production "
            f"callers ({wired}). Per this file's rule, DELETE their entries "
            "from _DECLARED_DARK rather than widening anything, and replace "
            "each with a POSITIVE wiring test that pins the CALL SITE -- "
            "otherwise a later refactor can drop the call and leave the "
            "function importable, unreached, and undeclared again.",
        )


class TestTheSweepItselfCanFail(CustomTestCase):
    """Can-fail proofs on synthetic trees -- the gate must move in BOTH
    directions, or its green is worth nothing."""

    def _tree(self, files: dict):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        for name, body in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        return root

    def test_it_reports_an_unwired_instrument(self):
        root = self._tree(
            {
                "inst.py": "def thing_verdict(x):\n    return x\n",
                "user.py": "def run():\n    return 1\n",
            }
        )
        dark, total = _sweep(root, repo_root=root)
        self.assertEqual(total, 1)
        self.assertEqual(sorted(dark), ["inst.py::thing_verdict"])

    def test_it_goes_silent_once_the_instrument_is_called(self):
        root = self._tree(
            {
                "inst.py": "def thing_verdict(x):\n    return x\n",
                "user.py": (
                    "from inst import thing_verdict\n\n"
                    "def run():\n    return thing_verdict(1)\n"
                ),
            }
        )
        dark, _ = _sweep(root, repo_root=root)
        self.assertEqual(dark, {})

    def test_a_same_module_caller_counts_as_wired(self):
        """A helper reached only by its own module is reached."""
        root = self._tree(
            {
                "inst.py": (
                    "def thing_verdict(x):\n    return x\n\n"
                    "def entry(x):\n    return thing_verdict(x)\n"
                ),
            }
        )
        dark, _ = _sweep(root, repo_root=root)
        self.assertEqual(dark, {})

    def test_string_dispatch_counts_as_wired(self):
        """THE INSTRUMENT'S OWN FIRST DEFECT, pinned.

        Blind to string literals, this sweep called three live checkers dark
        (``_check_layout_policy_conformance``, ``_check_layout_conformance``,
        ``_census_ownership_audit``) -- each reached through
        ``getattr(self, "<name>", fallback)(...)``. A gate that accuses live
        code is worse than no gate, so the idiom is pinned here.
        """
        root = self._tree(
            {
                "inst.py": (
                    "class M:\n" "    def _thing_verdict(self, x):\n        return x\n"
                ),
                "user.py": (
                    "def run(obj, x):\n"
                    '    return getattr(obj, "_thing_verdict", lambda *_: None)(x)\n'
                ),
            }
        )
        dark, total = _sweep(root, repo_root=root)
        self.assertEqual(total, 1)
        self.assertEqual(
            dark,
            {},
            "the sweep is blind to getattr-by-string dispatch again; it will "
            "report live checkers as dark. See the module docstring.",
        )

    def test_a_declared_reason_is_never_empty(self):
        """An allowlist entry without a reason is a hole with a name on it."""
        for key, reason in _DECLARED_DARK.items():
            self.assertTrue(
                reason.strip().startswith(("DARK", "EXEMPT")),
                f"{key}: every _DECLARED_DARK reason must open with DARK "
                "(the class, with an owner) or EXEMPT (not an instrument).",
            )
            self.assertGreater(
                len(reason.strip()), 60, f"{key}: reason too thin to audit."
            )


if __name__ == "__main__":
    unittest.main()

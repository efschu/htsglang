"""The planner's answer must come from the PROFILE, not from the rig it was
written on (#434).

The user directive this suite enforces: *the optimum per task must always be
selected automatically by the planner -- for every hardware combination,
model and quant format; nothing tailored to one rig*. A solver that has
absorbed a development rig's numbers still produces plausible output on a
foreign machine, which is why the defect survives review: the answer looks
solved. The only way to catch it is to feed the solver hardware it has never
seen and check that the answer MOVES with the hardware and with nothing else.

Every rig below is SYNTHETIC and foreign -- no card here exists, and the
reference rig this fork was developed on (one large card plus two smaller
ones over PCIe) is deliberately not among the shapes. Every checkpoint is
written to a temp dir. Nothing reads NVML, a model cache or the local machine.

Four properties, each with its own falsifier:

``TestTheSolveFollowsTheHardware``
    Change the profile, the vector changes; change the checkpoint's weight
    format, the vector changes. Anti-vacuity: the two answers differ.

``TestSymmetryHasNoLever``
    On a rig whose cards are identical, no concentration can be justified by
    anything IN the profile -- so a vector appearing there came from
    somewhere else. This is the cheapest leak detector in the suite.

``TestNoLocalRigConstantLeaks``
    Two invariances the solver must have and a leak cannot:
    RELABELING (permute the ranks, the vector permutes with them -- an
    assumption that "rank 0 is the fast card", true on the reference box
    where CUDA's FASTEST_FIRST order puts the big card first, breaks this)
    and SCALE (multiply every measured rate by a constant, the vector is
    unchanged -- a threshold fitted at one rig's absolute magnitudes breaks
    this).

``TestNoReferenceRigLiteralInTheSolvePath``
    Static guard on ``uneven_perf.py``: after AST docstring stripping (the
    #421 detector-B2 technique -- prose is not a data source), no card name
    or architecture tag of the development rig may appear in EXECUTABLE code.
    Anti-vacuity: the same scan without stripping finds many, so the guard is
    measuring the stripper's work and not an empty corpus.
"""

import ast
import json
import math
import os
import re
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


# --- synthetic checkpoints -------------------------------------------------

_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}
#: On-disk checkpoint size the sizing math reads (sparse file, no real bytes).
_CHECKPOINT_MIB = 26000

#: Two single-scheme quantization configs that select DIFFERENT lane tables in
#: ``_FORMAT_LANES``. This is the "quant diversity" axis of the directive: the
#: same cards score differently per format, so the same rig must solve two
#: different vectors for two different checkpoints.
_QUANT_CONFIGS = {
    "fp8": {"quant_method": "compressed-tensors", "format": "float-quantized"},
    "int8": {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True},
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "dynamic": True,
                },
            }
        },
    },
}


def _write_checkpoint(root, name, quant=None):
    """A checkpoint directory the cost model can size, with no real weights."""
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    config = dict(_CONFIG)
    if quant is not None:
        config["quantization_config"] = _QUANT_CONFIGS[quant]
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump(config, handle)
    with open(os.path.join(path, "model-00001-of-00001.safetensors"), "wb") as f:
        f.truncate(_CHECKPOINT_MIB * 2**20)
    return path


# --- synthetic foreign rigs ------------------------------------------------


class Card(types.SimpleNamespace):
    """One synthetic card: identity, VRAM and its MEASURED rates per lane."""


def _card(uuid, index, name, total_mib, lanes, membw, gemv):
    return Card(
        uuid=uuid,
        index=index,
        name=name,
        total_mib=total_mib,
        lanes=dict(lanes),
        membw=float(membw),
        gemv=float(gemv),
    )


def _profile(cards, links=None, group=None):
    """A v3 hardware profile over synthetic cards.

    ``gemm_tflops`` (the checkpoint-agnostic scalar) is pinned to the dense
    ``bf16`` lane, exactly as a real probe writes it.
    """
    gpus = {}
    for c in cards:
        gpus[c.uuid] = {
            "name": c.name,
            "cuda_index": c.index,
            "total_mib": c.total_mib,
            "gemm_tflops": c.lanes["bf16"],
            "membw_gbs": c.membw,
            "membw_gemv_gbs": c.gemv,
            "gemm_lanes": dict(c.lanes),
            "gemm_lane_notes": {},
        }
    link_block = dict(links or {})
    link_block["__group__"] = dict(group or {"ar_10kb_us": 30.0, "ar_1mb_us": 350.0})
    profile = {
        "version": 3,
        "driver": "999.00.00",
        "gpus": gpus,
        "links": link_block,
    }
    inventory = [
        {
            "uuid": c.uuid,
            "cuda_index": c.index,
            "name": c.name,
            "total_mib": c.total_mib,
        }
        for c in cards
    ]
    return profile, inventory


def _uniform_links(cards, gbs):
    out = {}
    for i, a in enumerate(cards):
        for b in cards[i + 1 :]:
            out[f"{a.uuid}|{b.uuid}"] = {"p2p_gbs": float(gbs)}
    return out


def _island_links(cards, inside_gbs, across_gbs):
    """Two islands: cards pair up (0,1) and (2,3) on a fast local fabric and
    reach the other island over a slow one."""
    out = {}
    for i, a in enumerate(cards):
        for j, b in enumerate(cards):
            if j <= i:
                continue
            same = (i // 2) == (j // 2)
            out[f"{a.uuid}|{b.uuid}"] = {
                "p2p_gbs": float(inside_gbs if same else across_gbs)
            }
    return out


# RIG MIX: the workhorse. A large card whose fp8 lane is by far the strongest
# and two smaller cards that are the strongest on int8. VRAM order and compute
# order deliberately DISAGREE per format, so a solver that reads the profile
# has to answer differently for an fp8 and an int8 checkpoint on the same
# cards, and a solver that reads a habit answers the same twice.
def _mix_cards(fp8_big=900.0):
    return [
        _card(
            "SYNTH-MIX-0",
            0,
            "SYNTH Accel L 48G",
            49140,
            {"bf16": 300.0, "fp8_native": fp8_big, "int8_native": 320.0},
            1400.0,
            1300.0,
        ),
        _card(
            "SYNTH-MIX-1",
            1,
            "SYNTH Accel Q 24G",
            24564,
            {"bf16": 120.0, "fp8_native": 130.0, "int8_native": 640.0},
            700.0,
            660.0,
        ),
        _card(
            "SYNTH-MIX-2",
            2,
            "SYNTH Accel Q 24G",
            24564,
            {"bf16": 118.0, "fp8_native": 128.0, "int8_native": 630.0},
            690.0,
            650.0,
        ),
    ]


#: RIG ISLAND: four IDENTICAL cards on a two-island fabric (an NVLink pair
#: reaching the other pair over the host bus).
def _island_cards():
    return [
        _card(
            f"SYNTH-ISL-{i}",
            i,
            "SYNTH Accel U 40G",
            40960,
            {"bf16": 400.0, "fp8_native": 800.0, "int8_native": 800.0},
            1200.0,
            1100.0,
        )
        for i in range(4)
    ]


#: RIG EIGHT: eight identical cards, the commodity shape this fork is NOT
#: developed on and the one most of its users have.
def _eight_cards():
    return [
        _card(
            f"SYNTH-8X-{i}",
            i,
            "SYNTH Accel E 24G",
            24564,
            {"bf16": 250.0, "fp8_native": 500.0, "int8_native": 500.0},
            900.0,
            860.0,
        )
        for i in range(8)
    ]


# --- the planner call ------------------------------------------------------


def _args(model_path, cards, *, tune="phase-prefill", reserve_mib=3000, context=131072):
    """Duck-typed ServerArgs with exactly the fields the optimizer reads.

    Same shape as ``test_phase_kv_coupling._args`` (#435); kept local so a
    change to that fixture cannot silently move this one.

    The default target is ``phase-prefill`` rather than ``enc``: the two share
    one objective and one ladder, and the phase arm treats the decode-knee
    guard as ADVISORY (#357), so a test about WHICH vector the profile implies
    is not silently answered by a gate. Tests that are about the gates name
    their own target.
    """
    tp = len(cards)
    budgets = [c.total_mib - reserve_mib for c in cards]
    sa = types.SimpleNamespace(
        model_path=model_path,
        tp_size=tp,
        rank_gpu_id=[c.index for c in cards],
        rank_gpu_memory_mib=list(budgets),
        rank_tp_ratio=list(budgets),
        rank_mlp_ratio=None,
        rank_vocab_ratio=None,
        rank_moe_ratio=None,
        rank_kv_ratio="coupled",
        rank_kv_capacity_seed=None,
        rank_auto_reserve_mib=",".join(str(reserve_mib) for _ in cards),
        rank_perf_tune=tune,
        rank_perf_loose_ctx_percent=0.0,
        kv_cache_dtype="fp8_e4m3",
        context_length=context,
        page_size=1,
        quantization=None,
        max_running_requests=16,
        chunked_prefill_size=2048,
        mem_fraction_static=0.74,
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_num_draft_tokens=None,
        speculative_adaptive=False,
        speculative_adaptive_config=None,
        speculative_cross_algorithm=False,
        speculative_draft_placement="split",
        disable_cuda_graph=False,
        dcp_size=tp,
        _derived_rank_auto_reserve_per_gpu={c.index: reserve_mib for c in cards},
        _measured_kv_budget_registry_path="/nonexistent/registry.json",
        cuda_graph_config=types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24)
        ),
    )
    sa.uneven_kv_flag_active = lambda: sa.rank_kv_ratio != "coupled"
    sa.uneven_kv_capacity_mode = lambda: sa.rank_kv_ratio == "capacity"
    sa.uneven_kv_speed_mode = lambda: sa.rank_kv_ratio == "speed"
    sa.uneven_kv_derived_mode = lambda: (
        sa.uneven_kv_capacity_mode() or sa.uneven_kv_speed_mode()
    )
    return sa


def _plan(model_path, cards, links=None, **kwargs):
    """Run the boot optimizer against a synthetic profile; return (args, log)."""
    sa = _args(model_path, cards, **kwargs)
    profile, inventory = _profile(cards, links or _uniform_links(cards, 12.0))
    captured = []
    with mock.patch.object(
        uneven_perf,
        "get_hardware_profile",
        return_value=(profile, "synthetic foreign fixture", inventory),
    ), mock.patch.object(
        uneven_perf.logger,
        "info",
        lambda *a, **k: captured.append(a[0] if a else ""),
    ), mock.patch.object(
        uneven_perf.logger, "warning", lambda *a, **k: None
    ):
        uneven_perf.apply_auto_performance(sa)
    return sa, "\n".join(captured)


def _ladder(log):
    """Every candidate MLP vector the objective priced, as int tuples."""
    out = []
    for line in log.splitlines():
        marker = "candidate MLP vector "
        if marker in line:
            out.append(
                tuple(int(x) for x in line.split(marker, 1)[1].split(":", 1)[0].split(","))
            )
    return out


class GeneralityTestCase(CustomTestCase):
    """One temp dir of synthetic checkpoints for the whole class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = {
            fmt: _write_checkpoint(cls._tmp.name, f"Synth-{fmt}", fmt)
            for fmt in ("fp8", "int8")
        }
        cls.ckpt["bf16"] = _write_checkpoint(cls._tmp.name, "Synth-bf16", None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()


class TestTheSolveFollowsTheHardware(GeneralityTestCase):
    """The answer is a function of the profile and the checkpoint."""

    def test_the_same_cards_solve_differently_per_weight_format(self):
        """Quant diversity: on the mixed rig the fp8-strong card and the
        int8-strong cards are DIFFERENT cards, so one profile must yield two
        vectors. A planner carrying a habit yields one."""
        cards = _mix_cards()
        fp8, fp8_log = _plan(self.ckpt["fp8"], cards)
        int8, int8_log = _plan(self.ckpt["int8"], cards)
        self.assertIsNotNone(fp8.rank_mlp_ratio, f"fp8 did not solve:\n{fp8_log}")
        self.assertIsNotNone(int8.rank_mlp_ratio, f"int8 did not solve:\n{int8_log}")
        self.assertNotEqual(fp8.rank_mlp_ratio, int8.rank_mlp_ratio)
        # ... and each concentrates onto the card that is strong in ITS
        # format: the big card for fp8, the two small ones for int8.
        fp8_share = fp8.rank_mlp_ratio[0] / sum(fp8.rank_mlp_ratio)
        int8_share = int8.rank_mlp_ratio[0] / sum(int8.rank_mlp_ratio)
        self.assertGreater(fp8_share, int8_share)

    def test_perturbing_one_lane_moves_the_vector_monotonically(self):
        """FALSIFIER for "the vector is transferred, not solved": make the
        big card's fp8 lane progressively stronger and its share of the MLP
        family must not shrink, and must grow somewhere across the sweep."""
        shares = []
        for strength in (400.0, 900.0, 2000.0):
            cards = _mix_cards(fp8_big=strength)
            sa, log = _plan(self.ckpt["fp8"], cards)
            self.assertIsNotNone(
                sa.rank_mlp_ratio, f"no solve at fp8={strength}:\n{log}"
            )
            vec = sa.rank_mlp_ratio
            shares.append(vec[0] / sum(vec))
        for lo, hi in zip(shares, shares[1:]):
            self.assertGreaterEqual(hi, lo - 1e-9)
        self.assertGreater(
            shares[-1],
            shares[0],
            "a 5x stronger lane on one card did not move the split at all",
        )

    def test_the_link_fabric_reaches_the_objective(self):
        """The candidate ladder must be built from link-adjusted scores: put
        the compute-strong card behind a much narrower fabric and the priced
        ladder has to change."""
        cards = _mix_cards()
        wide = _plan(self.ckpt["fp8"], cards, links=_uniform_links(cards, 300.0))[1]
        narrow_links = _uniform_links(cards, 300.0)
        for key in narrow_links:
            if cards[0].uuid in key:
                narrow_links[key] = {"p2p_gbs": 2.0}
        narrow = _plan(self.ckpt["fp8"], cards, links=narrow_links)[1]
        self.assertNotEqual(_ladder(wide), _ladder(narrow))


class TestSymmetryHasNoLever(GeneralityTestCase):
    """Identical cards justify no concentration. A vector on a symmetric rig
    was not derived from the profile, because the profile does not contain a
    reason for one."""

    def test_eight_equal_cards_get_no_concentration(self):
        cards = _eight_cards()
        for tune in ("enc", "both", "dec", "phase-prefill"):
            with self.subTest(tune=tune):
                sa, log = _plan(self.ckpt["fp8"], cards, tune=tune)
                self.assertIsNone(
                    sa.rank_mlp_ratio,
                    f"tune={tune} concentrated on identical cards:\n{log}",
                )

    def test_an_island_fabric_alone_does_not_concentrate_identical_cards(self):
        """The two-island fabric is a real asymmetry, but it is symmetric
        BETWEEN the ranks -- every card has one fast and two slow peers -- so
        the per-rank link term is uniform and there is still no lever. The
        planner must not invent one."""
        cards = _island_cards()
        sa, log = _plan(
            self.ckpt["fp8"],
            cards,
            links=_island_links(cards, 300.0, 8.0),
        )
        self.assertIsNone(sa.rank_mlp_ratio, log)

    def test_the_symmetric_refusal_is_named_not_silent(self):
        cards = _eight_cards()
        _sa, log = _plan(self.ckpt["fp8"], cards, tune="enc")
        self.assertTrue(
            "no representable concentration candidate" in log
            or "no effective lever" in log
            or "no candidate survives" in log,
            f"the refusal is not named in the plan log:\n{log}",
        )


class TestNoLocalRigConstantLeaks(GeneralityTestCase):
    """Behavioural leak guards. Both are invariances the profile-driven solve
    has by construction and a smuggled constant does not."""

    def test_relabeling_the_ranks_permutes_the_vector(self):
        """The reference rig puts its big card at CUDA index 0 (FASTEST_FIRST
        does that on a mixed box), so "rank 0 is the strong one" is true there
        and false in general. Reverse the rank order and the solved vector
        must reverse with it -- nothing else about the rig changed."""
        cards = _mix_cards()
        forward, fwd_log = _plan(self.ckpt["fp8"], cards)
        self.assertIsNotNone(forward.rank_mlp_ratio, fwd_log)

        reversed_cards = [
            _card(c.uuid, i, c.name, c.total_mib, c.lanes, c.membw, c.gemv)
            for i, c in enumerate(reversed(cards))
        ]
        backward, bwd_log = _plan(self.ckpt["fp8"], reversed_cards)
        self.assertIsNotNone(backward.rank_mlp_ratio, bwd_log)
        self.assertEqual(
            list(reversed(backward.rank_mlp_ratio)),
            list(forward.rank_mlp_ratio),
            "the solved vector does not follow the cards when the ranks are "
            "relabeled -- something outside the profile is deciding which "
            "rank gets the mass",
        )

    def test_the_answer_is_invariant_to_the_absolute_compute_scale(self):
        """The prefill objective is a RATIO between ranks. Multiply every
        measured rate on every card by the same factor and the answer must be
        the same vector. A threshold fitted at one rig's absolute TFLOPS --
        the classic shape of a smuggled constant -- fails this."""
        base = _mix_cards()
        reference, _ = _plan(self.ckpt["fp8"], base)
        self.assertIsNotNone(reference.rank_mlp_ratio)
        for factor in (0.1, 10.0):
            with self.subTest(factor=factor):
                scaled = [
                    _card(
                        c.uuid,
                        c.index,
                        c.name,
                        c.total_mib,
                        {k: v * factor for k, v in c.lanes.items()},
                        c.membw * factor,
                        c.gemv * factor,
                    )
                    for c in base
                ]
                sa, log = _plan(self.ckpt["fp8"], scaled)
                self.assertEqual(
                    sa.rank_mlp_ratio,
                    reference.rank_mlp_ratio,
                    f"scaling every rate by {factor} moved the answer:\n{log}",
                )

    def test_the_card_names_do_not_reach_the_answer(self):
        """Identity is UUID and measured rates. Renaming every card to a
        model this fork has never heard of must not change the plan."""
        base = _mix_cards()
        reference, _ = _plan(self.ckpt["fp8"], base)
        renamed = [
            _card(
                c.uuid,
                c.index,
                f"UNKNOWN VENDOR PART {i}",
                c.total_mib,
                c.lanes,
                c.membw,
                c.gemv,
            )
            for i, c in enumerate(base)
        ]
        sa, _log = _plan(self.ckpt["fp8"], renamed)
        self.assertEqual(sa.rank_mlp_ratio, reference.rank_mlp_ratio)


# --- static guard ----------------------------------------------------------

#: Card names, model numbers and architecture tags of the box this fork is
#: developed on, plus the neighbouring parts its notes quote.
_REFERENCE_RIG_TOKENS = re.compile(
    r"\b(RTX|GeForce|5090|3080|2080|Ti)\b|sm_?120\b|sm_?86\b", re.I
)

#: The concrete split vectors solved on that box. They are correct AS
#: EXAMPLES and are quoted all over the prose; in executable code they would
#: be one rig's answer applied to every rig.
_REFERENCE_RIG_VECTORS = (
    (10, 1, 1),
    (16, 1, 1),
    (8, 1, 1),
    (6, 1, 1),
    (3, 2, 2),
    (2, 11, 10),
    (16, 2, 3),
)

_SOLVE_PATH = "python/sglang/srt/uneven_perf.py"


def _repo_root():
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "sglang")):
            return here
    raise AssertionError("could not locate the repository root from the test file")


class _StripDocstrings(ast.NodeTransformer):
    """Prose is not a data source (#421 detector B2). A module that explains
    which rig a number came from is doing the right thing; the guard is about
    numbers the CODE reads."""

    def _strip(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def _executable_source(path):
    """The module's source with docstrings and comments removed.

    ``ast.unparse`` drops comments on its own; the transformer above removes
    the docstrings ``unparse`` would otherwise keep.
    """
    with open(path) as handle:
        tree = ast.parse(handle.read())
    stripped = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(stripped)
    return ast.unparse(stripped)


def _int_tuples(source):
    """Every literal list/tuple of small ints in the source, as tuples."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        values = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, int):
                values.append(element.value)
            else:
                values = None
                break
        if values and len(values) >= 3:
            out.append(tuple(values))
    return out


class TestNoReferenceRigLiteralInTheSolvePath(CustomTestCase):
    """The boot solver may not name the development rig in code it executes.

    Scoped to ``uneven_perf.py`` because that is the module a boot runs; the
    planner CLI's evidence registers legitimately quote the rig they measured
    (``planner/crossover.py``, ``planner/rejected.py``) and are audited by
    name in ``docs/dev/AUDIT_434_planner_constants.md`` instead.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.path = os.path.join(_repo_root(), _SOLVE_PATH)
        cls.code = _executable_source(cls.path)
        with open(cls.path) as handle:
            cls.raw = handle.read()

    def test_no_reference_card_name_in_executable_code(self):
        hits = [
            line.strip()
            for line in self.code.splitlines()
            if _REFERENCE_RIG_TOKENS.search(line)
        ]
        self.assertEqual(
            hits,
            [],
            "the boot solver names the development rig's hardware in code it "
            "executes; a card model may only appear in prose or in a lookup "
            "keyed by the DETECTED card",
        )

    def test_no_reference_solved_vector_as_a_code_literal(self):
        found = set(_int_tuples(self.code)) & set(_REFERENCE_RIG_VECTORS)
        self.assertEqual(
            found,
            set(),
            "a vector solved on the development rig appears as a literal in "
            "executable code -- it is one rig's answer, not a default",
        )

    def test_the_stripper_is_doing_work(self):
        """Anti-vacuity for both guards above: the module DOES discuss the
        reference rig at length in its prose, so a green result means the
        docstring stripping worked, not that the corpus was empty."""
        prose_hits = [
            line
            for line in self.raw.splitlines()
            if _REFERENCE_RIG_TOKENS.search(line)
        ]
        self.assertGreater(
            len(prose_hits),
            5,
            "the reference rig is no longer discussed in this module's prose, "
            "so this guard can no longer tell a working stripper from an "
            "empty scan -- re-anchor it on a module that does",
        )
        self.assertLess(
            len(self.code.splitlines()),
            len(self.raw.splitlines()),
            "stripping removed nothing",
        )


class TestTheDecodeTargetSolvesOnAnyProfile(GeneralityTestCase):
    """#434 cut 1: ``--rank-perf-tune dec`` used to return the base split
    without evaluating a candidate, on the strength of one rig's measurement
    that decode is flat across weight splits. It now solves.

    FALSIFIER: this class fails on the pre-#434 tree, where the branch returns
    before the ladder is built.
    """

    def _bandwidth_skewed_cards(self):
        """A rig where the VRAM order and the BANDWIDTH order disagree: the
        big card is the slow one. The capacity-first base plan therefore puts
        most of the weight mass on the rank least able to stream it, which is
        precisely the decode lever the old branch could not see."""
        return [
            _card(
                "SYNTH-BW-0",
                0,
                "SYNTH Accel Wide 48G",
                49140,
                {"bf16": 200.0, "fp8_native": 400.0},
                420.0,
                400.0,
            ),
            _card(
                "SYNTH-BW-1",
                1,
                "SYNTH Accel Fast 24G",
                24564,
                {"bf16": 210.0, "fp8_native": 420.0},
                1800.0,
                1700.0,
            ),
            _card(
                "SYNTH-BW-2",
                2,
                "SYNTH Accel Fast 24G",
                24564,
                {"bf16": 208.0, "fp8_native": 416.0},
                1780.0,
                1690.0,
            ),
        ]

    def test_dec_finds_the_lever_when_vram_and_bandwidth_disagree(self):
        cards = self._bandwidth_skewed_cards()
        sa, log = _plan(self.ckpt["fp8"], cards, tune="dec")
        self.assertTrue(_ladder(log), f"dec priced no candidate:\n{log}")
        self.assertIn("predicted decode gain", log)
        self.assertIsNotNone(
            sa.rank_mlp_ratio,
            "on a rig whose bandwidth order contradicts its VRAM order the "
            f"decode weight lever is real and dec found nothing:\n{log}",
        )
        # It moved mass AWAY from the slow-but-large rank 0, which is the
        # whole point: the base plan is capacity-proportional.
        base = _args(self.ckpt["fp8"], cards).rank_tp_ratio
        self.assertLess(
            sa.rank_mlp_ratio[0] / sum(sa.rank_mlp_ratio),
            base[0] / sum(base),
        )

    def test_dec_is_scored_on_bandwidth_and_not_on_the_gemm_lanes(self):
        """Anti-confound: change only the compute lanes and dec must not
        move. The same change moves ``enc`` (asserted in
        ``test_gemm_lane_format``), so the invariance is a property of the
        decode objective and not of an inert planner."""
        cards = self._bandwidth_skewed_cards()
        first, _ = _plan(self.ckpt["fp8"], cards, tune="dec")
        boosted = [
            _card(
                c.uuid,
                c.index,
                c.name,
                c.total_mib,
                {k: v * (3.0 if c.index == 0 else 1.0) for k, v in c.lanes.items()},
                c.membw,
                c.gemv,
            )
            for c in cards
        ]
        second, _ = _plan(self.ckpt["fp8"], boosted, tune="dec")
        self.assertEqual(first.rank_mlp_ratio, second.rank_mlp_ratio)

    def test_a_flat_profile_reports_a_result_not_an_assumption(self):
        """Where VRAM share and bandwidth share already agree, dec must keep
        the base split AND say that this is what the solve found."""
        cards = _eight_cards()
        sa, log = _plan(self.ckpt["fp8"], cards, tune="dec")
        self.assertIsNone(sa.rank_mlp_ratio)
        self.assertIn("tune=dec:", log)
        self.assertIn("DECODE round time", log)
        self.assertNotIn("documented no-op", log)


class TestTheCapacityFirstDefaultNamesItsAlternative(CustomTestCase):
    """#434 cut 1: plain ``--rank-tp-ratio auto`` stays capacity-first, but it
    may not present itself as the optimum without naming the flag that solves
    for one.

    FALSIFIER: fails on the pre-#434 tree, where no such notice exists.
    """

    def test_the_notice_names_the_flag_and_the_targets(self):
        from sglang.srt.server_args import ServerArgs

        notice = ServerArgs.CAPACITY_FIRST_DEFAULT_NOTICE
        self.assertIn("auto-performance", notice)
        self.assertIn("--rank-perf-tune", notice)
        for target in ("enc", "dec", "maxkv", "phase-prefill", "phase-decode"):
            self.assertIn(target, notice)
        # It must also say the solve is per operating point, which is the
        # half that stops the notice from becoming a new copy-paste default.
        self.assertIn("operating point", notice)

    def test_plain_auto_emits_the_notice(self):
        from sglang.srt.server_args import ServerArgs

        view = types.SimpleNamespace(
            rank_tp_ratio=[3, 2, 2],
            rank_mlp_ratio=None,
            rank_vocab_ratio=None,
            rank_moe_ratio=None,
        )
        with mock.patch("sglang.srt.server_args.logger") as log:
            ServerArgs._announce_capacity_first_default(view)
        emitted = "\n".join(str(call) for call in log.info.call_args_list)
        self.assertIn("auto-performance", emitted)

    def test_a_pinned_family_vector_is_called_out(self):
        from sglang.srt.server_args import ServerArgs

        view = types.SimpleNamespace(
            rank_tp_ratio=[3, 2, 2],
            rank_mlp_ratio=[10, 1, 1],
            rank_vocab_ratio=None,
            rank_moe_ratio=None,
        )
        with mock.patch("sglang.srt.server_args.logger") as log:
            ServerArgs._announce_capacity_first_default(view)
        emitted = "\n".join(str(call) for call in log.info.call_args_list)
        self.assertIn("--rank-mlp-ratio", emitted)
        self.assertIn("re-solve", emitted.lower())
        self.assertEqual(
            log.info.call_count,
            2,
            "the pin call-out must be its own line, not folded into the "
            "default notice every boot prints",
        )


class TestTheFixtureItselfIsForeign(GeneralityTestCase):
    """Guard on this file: if someone re-anchors these rigs onto the
    development box the whole suite stops proving anything."""

    def test_no_synthetic_card_carries_a_real_model_number(self):
        for factory in (_mix_cards, _island_cards, _eight_cards):
            for card in factory():
                with self.subTest(card=card.name):
                    self.assertIsNone(_REFERENCE_RIG_TOKENS.search(card.name))

    def test_the_rigs_have_shapes_the_reference_box_does_not(self):
        self.assertEqual(len(_eight_cards()), 8)
        self.assertEqual(len(_island_cards()), 4)
        # The mixed rig is TP=3 like the reference box, so its distinguishing
        # property is the disagreement between VRAM order and per-format
        # compute order -- assert that, rather than the rank count.
        cards = _mix_cards()
        by_vram = sorted(range(3), key=lambda i: -cards[i].total_mib)
        by_int8 = sorted(range(3), key=lambda i: -cards[i].lanes["int8_native"])
        self.assertNotEqual(by_vram, by_int8)

    def test_the_checkpoints_are_synthetic(self):
        for path in self.ckpt.values():
            self.assertTrue(path.startswith(tempfile.gettempdir()))
            self.assertTrue(os.path.isfile(os.path.join(path, "config.json")))

    def test_the_gcd_reduction_holds_on_every_solved_vector(self):
        """Cheap well-formedness pin so a solve that starts returning
        unreduced vectors is caught here rather than in a boot."""
        cards = _mix_cards()
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                sa, _log = _plan(self.ckpt[fmt], cards)
                if sa.rank_mlp_ratio is None:
                    continue
                self.assertEqual(math.gcd(*sa.rank_mlp_ratio), 1)


if __name__ == "__main__":
    unittest.main()

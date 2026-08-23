"""#500 registry contract: every ``planner/flags.py`` edge, against the RUNTIME.

WHY THIS FILE EXISTS
--------------------
Audit #500's structural finding (§3) is that this tree has TWO capability
registries and they disagree. ``planner/flags.py`` calls itself "the single
source of truth for EVERY sglang ``ServerArgs`` flag plus EVERY fork-specific
flag/env var" and carries ``requires`` / ``mutually_exclusive_with`` /
``tuple_len_flag`` edges that drive the dashboard's Runner tab.
``FEATURE_CATALOG.md`` never mentions it. Neither is authoritative -- only the
predicate in ``server_args.py`` is, which is what CLAUDE.md's MECHANISM REACH
law says.

The curated overlay's own preamble claimed its edges "were verified against
server_args.py's own validation". They were verified by READING, once, and two
of them had drifted into forbidding configurations the server supports:

* ``rank_gpu_id`` was declared ``mutually_exclusive_with=(..., "pp_size", ...)``
  while the runtime does the OPPOSITE -- with a pipeline and an uneven plan it
  REQUIRES the placement (``server_args.py:9596``) and validates the
  world-length ``pp_size * tp_size`` form (``:9698-9712``). The dashboard could
  not express §1's TPxPPxTP feature at all (#500-I2).
* ``rank_tp_ratio`` was declared ``requires=("rank_gpu_id",)`` while the
  runtime decouples them on purpose -- the whole ratio validation block is
  hoisted ABOVE the ``if self.rank_gpu_id is None: ... return`` early return,
  because the cross-vendor two-launcher bring-up places its own ranks and
  ``--rank-gpu-id`` cannot describe the ROCm rank at all (``:9540-9548``).
  The dashboard blocked the cross-vendor arm (#500-I3).

So this file replaces "verified by reading" with "asserted by executing".
Every edge declared on the uneven-TP flags is DRIVEN through the real
``ServerArgs`` validation, and the runtime's verdict is the expected value.
A declared edge the runtime does not enforce is a registry that forbids more
than the server does; an enforced rule with no declared edge is a dashboard
that offers a boot which then dies.

SCOPE, STATED HONESTLY
----------------------
The uneven-TP / placement group (``rank_gpu_id``, ``rank_gpu_memory_mib``,
``rank_tp_ratio``, ``rank_kv_ratio`` and their exclusion partners) -- the
flags audit #500 examined. The edge ENUMERATION is generic, so a new edge
added to any of these flags is checked automatically; the driver table is not,
so a new FLAG must be added to ``_ACTIVATION`` (and
``test_every_audited_flag_has_a_driver`` fails until it is).

#837 EXTENDS THIS FILE TO A SECOND FAMILY, and states plainly why it is a
second family rather than four more entries in ``_AUDITED``: the promoted
phase-flip seam knobs are validated by ``_handle_seam_shrink_flags_837``, not
by ``_handle_uneven_tp``. Putting them in ``_AUDITED`` would drive them
through a handler that has never heard of them, which accepts every value --
a whole class of tests passing green while measuring nothing. So they get
their own driver (``seam_refusal``), their own can-discriminate check, and
their own classes below. What is checked for them is the same kind of thing:
declared DOMAINS against the runtime's, the deliberate NON-edge between the
master and its two per-half overrides, and that the family emits as argv
rather than env.

CAN-FAIL PROOF
--------------
Put ``"pp_size"`` back into ``rank_gpu_id``'s ``mutually_exclusive_with`` and
``test_declared_exclusions_are_runtime_refusals`` goes red naming pp_size;
restore ``requires=("rank_gpu_id",)`` on ``rank_tp_ratio`` and
``test_declared_requirements_are_runtime_refusals`` goes red. Both were
executed red before the fix (that is where the two rows above come from).
"""

import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.planner import flags
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


#: The flags whose edges this contract covers.
_AUDITED = ("rank_gpu_id", "rank_gpu_memory_mib", "rank_tp_ratio", "rank_kv_ratio")

#: How to make each flag ACTIVE on a 2-rank launch, and the CLI spelling its
#: refusal is expected to name. ``tp_size`` is 2 everywhere so one base works.
_ACTIVATION = {
    "rank_gpu_id": ([0, 1], "--rank-gpu-id"),
    "rank_gpu_memory_mib": (15000, "--rank-gpu-memory-mib"),
    "rank_tp_ratio": ([3, 1], "--rank-tp-ratio"),
    "rank_kv_ratio": ("speed", "--rank-kv-ratio"),
    "mem_fraction_static": (0.8, "--mem-fraction-static"),
    "dp_size": (2, "--dp-size"),
    "ep_size": (2, "--ep-size"),
    "nnodes": (2, "--nnodes"),
    "base_gpu_id": (1, "--base-gpu-id"),
    "gpu_id_step": (2, "--gpu-id-step"),
    "pp_size": (2, "--pp-size"),
    # #837: the promoted phase-flip seam knobs. Their driver is NOT
    # `_handle_uneven_tp` (see `_SEAM_837` below), so they are not in
    # `_AUDITED`; the rows are here because `_ACTIVATION` is this file's one
    # table of "how do you make this flag active, and what does its refusal
    # call it", and a second table would drift from this one.
    "seam_shrink": (True, "--seam-shrink"),
    "seam_shrink_prearm_quiesce": (1, "--seam-shrink-prearm-quiesce"),
    "seam_shrink_defer_grow": (1, "--seam-shrink-defer-grow"),
    "seam_shrink_grow_debt_rounds": (32, "--seam-shrink-grow-debt-rounds"),
    "flip_seam_drain_budget_ms": (1094, "--flip-seam-drain-budget-ms"),
    "hicache_read_buffers": (8, "--hicache-read-buffers"),
}

#: #837's family, and the value that is OUTSIDE each one's declared domain.
#: The tri-states are a closed set (-1/0/1); the three counts are >= 0.
_SEAM_837 = {
    "seam_shrink_prearm_quiesce": 2,
    "seam_shrink_defer_grow": -2,
    "seam_shrink_grow_debt_rounds": -1,
    "flip_seam_drain_budget_ms": -1,
    "hicache_read_buffers": -1,
}


#: Partners that change the SHAPE the other flags must have, so an exclusion
#: claim can be tested against a well-formed configuration rather than one that
#: refuses on geometry. Under a pipeline ``--rank-gpu-id`` is world-length.
_SHAPE_FIXUP = {
    "pp_size": {"rank_gpu_id": [0, 1, 2, 3]},
}


def _fake_nvml(gpu_ids):
    return {gpu_id: (32768, 30000) for gpu_id in sorted(set(gpu_ids))}


def drive(**kwargs):
    """Run the REAL uneven-TP validation on a bare ServerArgs.

    ``model_path='dummy'`` short-circuits ``__post_init__``, so the handler is
    exercised in isolation with exactly the fields under test and no NVML,
    no device and no model on the machine.
    """
    args = ServerArgs(model_path="dummy", **kwargs)
    with patch.object(
        server_args_module, "_query_rank_gpu_memory_mib", _fake_nvml
    ):
        args._handle_uneven_tp()
    return args


def refusal(**kwargs):
    """The runtime's refusal message, or None if it accepts."""
    try:
        drive(**kwargs)
        return None
    except ValueError as e:
        return str(e)


class TestTheDriverIsHonest(CustomTestCase):
    """The instrument must be able to discriminate before its verdicts count
    (CLAUDE.md: an instrument's verdict counts only after a can-discriminate
    check on known-different inputs)."""

    def test_a_legal_configuration_is_accepted(self):
        self.assertIsNone(refusal(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=15000))

    def test_a_known_illegal_configuration_is_refused(self):
        msg = refusal(tp_size=2, rank_gpu_id=[0, 1, 2], rank_gpu_memory_mib=15000)
        self.assertIsNotNone(msg)
        self.assertIn("--rank-gpu-id", msg)

    def test_every_audited_flag_has_a_driver(self):
        for fid in _AUDITED:
            self.assertIn(fid, _ACTIVATION, f"{fid} has no activation value")


class TestDeclaredEdgesAreRuntimePredicates(CustomTestCase):
    def setUp(self):
        self.cat = flags.catalog()

    def _base(self, **overrides):
        kw = dict(tp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=15000)
        kw.update(overrides)
        return kw

    def test_declared_exclusions_are_runtime_refusals(self):
        """Every ``mutually_exclusive_with`` edge ON an audited flag must be a
        real refusal. An edge the runtime does not enforce greys a field the
        server would have taken.

        An exclusion is DISPROVED by one accepted well-formed configuration,
        so every well-formed shape has to be tried -- otherwise a refusal for
        an unrelated reason reads as proof of the edge. That is not
        hypothetical: with ``pp_size`` the plain base refuses on VECTOR LENGTH
        (``--rank-gpu-id`` wants ``pp_size x tp_size`` entries under a
        pipeline), so a length-shaped false pass is exactly what would have
        hidden #500-I2 here. ``_SHAPE_FIXUP`` supplies the corrected shape.
        """
        for fid in _AUDITED:
            spec = self.cat[fid]
            for partner in spec.mutually_exclusive_with:
                with self.subTest(flag=fid, excludes=partner):
                    self.assertIn(partner, _ACTIVATION, f"no driver for {partner}")
                    value, _ = _ACTIVATION[partner]
                    shapes = [self._base(**{partner: value})]
                    fixup = _SHAPE_FIXUP.get(partner)
                    if fixup is not None:
                        shapes.append(self._base(**{partner: value}, **fixup))
                    for kw in shapes:
                        self.assertIsNotNone(
                            refusal(**kw),
                            f"flags.py declares {fid} x {partner} exclusive, "
                            f"but the runtime accepts {kw}",
                        )

    def test_declared_requirements_are_runtime_refusals(self):
        """Every ``requires`` edge must be a real refusal when the required
        flag is absent."""
        for fid in _AUDITED:
            spec = self.cat[fid]
            for needed in spec.requires:
                with self.subTest(flag=fid, requires=needed):
                    value, _ = _ACTIVATION[fid]
                    kw = {"tp_size": 2, fid: value}
                    # supply every OTHER declared requirement so the refusal
                    # under test is the one being asserted
                    for other in spec.requires:
                        if other != needed:
                            kw[other] = _ACTIVATION[other][0]
                    msg = refusal(**kw)
                    self.assertIsNotNone(
                        msg,
                        f"flags.py declares {fid} requires {needed}, but the "
                        f"runtime accepts {fid} without it",
                    )

    def test_declared_requires_any_groups_are_runtime_refusals(self):
        for fid in _AUDITED:
            for group in self.cat[fid].requires_any:
                with self.subTest(flag=fid, requires_any=group):
                    value, _ = _ACTIVATION[fid]
                    msg = refusal(tp_size=2, **{fid: value})
                    self.assertIsNotNone(
                        msg,
                        f"flags.py declares {fid} requires one of {group}, but "
                        f"the runtime accepts it with none of them",
                    )


class TestTheTwoNamedInversions(CustomTestCase):
    """The two rows that made this file necessary, pinned by name so a revert
    is loud rather than a silently narrower dashboard."""

    def setUp(self):
        self.cat = flags.catalog()

    def test_rank_gpu_id_is_not_exclusive_with_pp_size(self):
        """#500-I2. The runtime ACCEPTS the pair -- it is the TPxPPxTP
        feature's own shape."""
        self.assertNotIn("pp_size", self.cat["rank_gpu_id"].mutually_exclusive_with)
        self.assertNotIn("rank_gpu_id", self.cat["pp_size"].mutually_exclusive_with)
        self.assertIsNone(
            refusal(
                tp_size=2, pp_size=2, rank_gpu_id=[0, 1, 2, 3], rank_gpu_memory_mib=15000
            )
        )

    def test_the_runtime_requires_the_placement_under_a_pipeline(self):
        """The opposite edge, and the reason the exclusion was so wrong."""
        msg = refusal(tp_size=2, pp_size=2, rank_tp_ratio=[3, 1])
        self.assertIsNotNone(msg)
        self.assertIn("requires --rank-gpu-id", msg)

    def test_the_world_length_rule_is_pp_times_tp(self):
        """``tuple_len_flag='tp_size'`` alone is wrong under a pipeline: the
        runtime validates ``pp_size * tp_size`` entries."""
        spec = self.cat["rank_gpu_id"]
        self.assertEqual(spec.tuple_len_flag, "tp_size")
        self.assertEqual(spec.tuple_len_times_flag, "pp_size")
        # runtime: a tp_size-length vector under a pipeline is refused
        msg = refusal(tp_size=2, pp_size=2, rank_gpu_id=[0, 1], rank_gpu_memory_mib=15000)
        self.assertIsNotNone(msg)
        self.assertIn("pp-size", msg)
        # registry: resolve() reports the same requirement
        res = flags.resolve(
            {"tp_size": 2, "pp_size": 2, "rank_gpu_id": [0, 1], "rank_gpu_memory_mib": 15000}
        )
        self.assertIsNotNone(res["rank_gpu_id"]["error"])
        self.assertIn("4 entries", res["rank_gpu_id"]["error"])

    def test_rank_tp_ratio_does_not_require_a_placement(self):
        """#500-I3. An explicit vector is a PARTITION; the cross-vendor
        two-launcher arm has no --rank-gpu-id to give."""
        self.assertNotIn("rank_gpu_id", self.cat["rank_tp_ratio"].requires)
        self.assertIsNone(refusal(tp_size=2, rank_tp_ratio=[3, 1]))

    def test_but_the_auto_modes_still_do(self):
        """The narrower rule that IS real: the auto modes derive weights from
        per-card budgets (``server_args.py:8971``)."""
        for mode in ("auto", "auto-performance"):
            with self.subTest(mode=mode):
                msg = refusal(tp_size=2, rank_tp_ratio=mode)
                self.assertIsNotNone(msg)
                self.assertIn("--rank-gpu-id", msg)
                err = flags.resolve({"tp_size": 2, "rank_tp_ratio": mode})
                self.assertIn("--rank-gpu-id", err["rank_tp_ratio"]["error"] or "")

    def test_the_dp_ep_nnodes_edges_the_audit_left_unverified_do_hold(self):
        """Audit #500 recorded ``rank_gpu_id`` x dp/ep/nnodes as declared but
        unverified. Executed here: all three are real runtime refusals, so the
        edges stay."""
        for partner, value in (("dp_size", 2), ("ep_size", 2), ("nnodes", 2)):
            with self.subTest(partner=partner):
                msg = refusal(
                    tp_size=2,
                    rank_gpu_id=[0, 1],
                    rank_gpu_memory_mib=15000,
                    **{partner: value},
                )
                self.assertIsNotNone(msg)
                self.assertIn("--rank-gpu-id", msg)


class TestRankKvRatioEdgeMatchesTheRuntime(CustomTestCase):
    """#500-B2's fix is what makes this edge true, so it is asserted from both
    sides: the registry says ``requires=("rank_gpu_id", "rank_tp_ratio")`` and
    the runtime now refuses the flag on the no-placement path by name instead
    of accepting it and doing nothing."""

    def test_the_declared_edge(self):
        self.assertEqual(
            tuple(flags.catalog()["rank_kv_ratio"].requires),
            ("rank_gpu_id", "rank_tp_ratio"),
        )

    def test_the_runtime_refuses_without_a_placement(self):
        msg = refusal(tp_size=2, rank_tp_ratio=[3, 1], rank_kv_ratio="speed")
        self.assertIsNotNone(msg)
        self.assertIn("--rank-gpu-id", msg)

    def test_the_pair_is_accepted_with_one(self):
        self.assertIsNone(
            refusal(
                tp_size=2,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=15000,
                rank_tp_ratio=[3, 1],
                rank_kv_ratio="speed",
            )
        )


def seam_refusal(**kwargs):
    """The #837 family's refusal message, or None if the runtime accepts.

    A SECOND driver, not a reuse of `drive`: `_handle_uneven_tp` knows nothing
    about these fields, so running them through it would accept every value
    and the whole class below would pass by driving the wrong handler. The
    same trap `TestTheDriverIsHonest` was written against, one handler over.
    """
    args = ServerArgs(model_path="dummy", **kwargs)
    try:
        args._handle_seam_shrink_flags_837()
        return None
    except ValueError as e:
        return str(e)


class TestTheSeam837DriverIsHonest(CustomTestCase):
    """#837. Same rule as `TestTheDriverIsHonest`: this driver's verdicts do
    not count until it is shown to discriminate on known-different inputs."""

    def test_a_legal_configuration_is_accepted(self):
        self.assertIsNone(
            seam_refusal(
                seam_shrink=True,
                seam_shrink_prearm_quiesce=1,
                seam_shrink_defer_grow=0,
                seam_shrink_grow_debt_rounds=32,
                flip_seam_drain_budget_ms=1094,
                hicache_read_buffers=8,
            )
        )

    def test_a_known_illegal_configuration_is_refused(self):
        msg = seam_refusal(seam_shrink_prearm_quiesce=2)
        self.assertIsNotNone(msg)
        self.assertIn("--seam-shrink-prearm-quiesce", msg)

    def test_every_seam_837_flag_has_a_driver(self):
        for fid in _SEAM_837:
            self.assertIn(fid, _ACTIVATION, f"{fid} has no activation value")


class TestSeam837DeclaredDomainsAreRuntimePredicates(CustomTestCase):
    """The catalog now DECLARES a domain for these flags (the tri-states carry
    `allowed=(-1, 0, 1)`), and a declared domain the runtime does not enforce
    is the same class of drift as a declared edge it does not enforce -- the
    dashboard offering a value the boot then rejects, or greying one it would
    have taken. Driven, not read.
    """

    def setUp(self):
        self.cat = flags.catalog()

    def test_declared_allowed_values_are_all_accepted(self):
        for fid, spec in ((f, self.cat[f]) for f in _SEAM_837):
            for value in spec.allowed or ():
                with self.subTest(flag=fid, value=value):
                    self.assertIsNone(
                        seam_refusal(**{fid: value}),
                        f"flags.py offers {fid}={value}, the runtime refuses it",
                    )

    def test_out_of_domain_values_are_refused_by_cli_name(self):
        for fid, bad in _SEAM_837.items():
            with self.subTest(flag=fid, value=bad):
                msg = seam_refusal(**{fid: bad})
                self.assertIsNotNone(
                    msg, f"the runtime accepts {fid}={bad}, which is out of domain"
                )
                self.assertIn(_ACTIVATION[fid][1], msg)

    def test_unset_is_always_legal(self):
        """None is "the operator said nothing", never a value to validate --
        it is what keeps the default path byte-identical."""
        self.assertIsNone(seam_refusal())


class TestTheSeam837NonEdge(CustomTestCase):
    """THE EDGE THAT MUST NOT EXIST, pinned in both directions.

    `_handle_seam_shrink_flags_837` says it in a comment rather than leaving it
    to be inferred: "A per-half override without the master is not an error --
    -1 follows a master that is off, and 1 is exactly how a window turns ONE
    half on without the other. Refusing it would forbid the attribution run the
    overrides exist for."

    A `requires=("seam_shrink",)` in the curated overlay would therefore grey
    out, in the dashboard, exactly the single-half configuration W13b's
    criteria 8-14 need in order to attribute a cutover change to one half --
    the same shape as #500-I3 on `rank_tp_ratio`, and this file exists because
    that one was found by executing rather than by reading.

    CAN-FAIL PROOF: add `requires=("seam_shrink",)` to either override in
    `flags.py::_CURATED` and `test_no_requires_edge_is_declared` goes red
    naming the flag.
    """

    def setUp(self):
        self.cat = flags.catalog()

    def test_no_requires_edge_is_declared(self):
        for fid in ("seam_shrink_prearm_quiesce", "seam_shrink_defer_grow"):
            with self.subTest(flag=fid):
                self.assertNotIn("seam_shrink", self.cat[fid].requires)
                self.assertNotIn(
                    "seam_shrink", self.cat[fid].mutually_exclusive_with
                )

    def test_the_runtime_accepts_a_half_without_the_master(self):
        for fid in ("seam_shrink_prearm_quiesce", "seam_shrink_defer_grow"):
            for value in (-1, 0, 1):
                with self.subTest(flag=fid, value=value):
                    self.assertIsNone(seam_refusal(**{fid: value}))


class TestSeam837IsEmittedAsFlagsNotEnv(CustomTestCase):
    """#837's point: the flag is how the value reaches the server. A bool that
    argparse registers with `store_true` must emit as a BARE token.

    WHY THE CURATED `type="bool"` IS LOAD-BEARING, precisely. `_infer_type`
    maps `Optional[bool]` to "str" (it tests `base_type is bool`, and
    `Union[bool, None]` is not `bool`). `profile_argv` survives that for a
    hand-built profile because it also checks `isinstance(v, bool)` -- but the
    dashboard never sends a Python bool. `webui.py:11325` renders a checkbox
    only for `type==='bool'`, and `webui.py:11475` reads back
    `spec.type==='bool' ? el.checked : el.value.trim()`. Under the inferred
    "str" the flag is a TEXT BOX whose value arrives as the string "1", and
    `profile_argv` then emits `--seam-shrink 1`, which argparse's `store_true`
    rejects. So the type is asserted directly, and the string path is driven
    below rather than left to the bool path to cover.
    """

    def test_seam_shrink_is_a_bool_in_the_catalog(self):
        self.assertEqual(flags.catalog()["seam_shrink"].type, "bool")

    def test_a_dashboard_shaped_value_still_emits_the_bare_token(self):
        """The string "1" is what `webui.py:11475` produces for a non-bool
        type. It must NOT become a positional value."""
        prof = flags.Profile(
            "t", "custom", settings={"model_path": "/m", "seam_shrink": "1"}
        )
        argv = flags.profile_argv(prof)
        self.assertIn("--seam-shrink", argv)
        after = argv[argv.index("--seam-shrink") + 1 :]
        self.assertTrue(not after or after[0].startswith("--"), argv)

    def test_an_armed_profile_emits_the_bare_token(self):
        prof = flags.Profile(
            "t", "custom", settings={"model_path": "/m", "seam_shrink": True}
        )
        argv = flags.profile_argv(prof)
        self.assertIn("--seam-shrink", argv)
        # BARE: nothing follows it but another flag (or nothing at all). The
        # inferred "str" type would have put a "1" here.
        after = argv[argv.index("--seam-shrink") + 1 :]
        self.assertTrue(not after or after[0].startswith("--"), argv)

    def test_an_unset_profile_emits_nothing(self):
        prof = flags.Profile("t", "custom", settings={"model_path": "/m"})
        joined = " ".join(flags.profile_argv(prof))
        for fid in ("seam_shrink", *_SEAM_837):
            with self.subTest(flag=fid):
                self.assertNotIn(_ACTIVATION[fid][1], joined)

    def test_the_planner_seat_reaches_a_generated_profile(self):
        """THE #837 SEAT, driven end to end.

        `_seam_shrink_planner_default` returns `{}` today (W13b has not been
        run), so on its own it proves nothing about being WIRED -- a seat that
        is never called and a seat that declines look identical from outside.
        Patching it to arm the flag separates the two: the decision has to
        reach `_mk`, land in the profile's settings, and come out of
        `profile_argv` as a bare token. This is what makes the operator's
        post-measurement change a one-line edit rather than a hope.
        """
        gpus = [{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607}]
        base = {"model_path": "/m"}

        declined = flags.profiles(None, gpus, base=dict(base))
        self.assertTrue(declined)
        for prof in declined:
            self.assertIsNone(prof.settings.get("seam_shrink"))
            self.assertNotIn("--seam-shrink", flags.profile_argv(prof))

        with patch.object(
            flags, "_seam_shrink_planner_default", lambda s, m, g: {"seam_shrink": True}
        ):
            armed = flags.profiles(None, gpus, base=dict(base))
        self.assertTrue(armed)
        for prof in armed:
            self.assertIs(prof.settings.get("seam_shrink"), True)
            self.assertIn("--seam-shrink", flags.profile_argv(prof))
            self.assertTrue(
                any("phase-flip seam policy" in i for i in prof.info), prof.info
            )

    def test_an_explicit_operator_value_outranks_the_planner(self):
        """The planner holds the seat; it does not overrule the operator who
        sat down in it. An explicit `base` value survives an armed default."""
        gpus = [{"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607}]
        with patch.object(
            flags, "_seam_shrink_planner_default", lambda s, m, g: {"seam_shrink": True}
        ):
            profs = flags.profiles(
                None, gpus, base={"model_path": "/m", "seam_shrink": False}
            )
        for prof in profs:
            self.assertIs(prof.settings.get("seam_shrink"), False)
            self.assertNotIn("--seam-shrink", flags.profile_argv(prof))

    def test_the_family_is_never_env_typed(self):
        """`profile_env` must not carry them: they are argv now, and the env
        keys survive only as the deprecated fallback the SERVER publishes to
        itself from its own argv."""
        cat = flags.catalog()
        for fid in ("seam_shrink", *_SEAM_837):
            with self.subTest(flag=fid):
                self.assertFalse(cat[fid].is_env)
                self.assertEqual(cat[fid].source, "fork")


if __name__ == "__main__":
    unittest.main()

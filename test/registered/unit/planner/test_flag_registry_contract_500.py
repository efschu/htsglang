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


if __name__ == "__main__":
    unittest.main()

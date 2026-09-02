"""#1068 WEG 1 slice 1 (WEG1_BUILD_SPEC_0901 section 4.1, L11/L13): the
phase-flip HiCache rebind requires the staging role and BOTH absolute knobs,
and prices the second phase's pools on the parse-time host ledger.

WHY A REFUSAL. Under ``--phase-flip-rebind-hicache`` there are TWO host
pools per rank (the pp pool and the tp pin) plus two mamba anchor pools, and
since #1068 the tp pin's rows are coupled to the pp pool that --hicache-size
built. A ratio-sized pp pool would make the pin ratio-sized too, and a
default (0) anchor budget would size the anchors from the device pool --
rank-divergently. So the three flags are one contract: role staging
(DESIGN_706_BOOT: the host tier is staging, the disk tier is retention),
--hicache-size > 0, --hicache-mamba-host-mib > 0. No ratio fallback.

WHY A LEDGER POST. Every scheduler process allocates its own tp pin and its
own two anchor pools. A parse-time ledger that prices only the staging tier
under-counts a rebind boot by the second phase, and the machine finds out at
the OOM killer (#721).

Retention boots and staging boots WITHOUT the rebind flag reach none of this
(byte-identical, pinned below).

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/managers/test_server_args_rebind_requires_staging_1068.py -q
"""

import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1_000_000_000
MIB = 1024**2
ROOMY = (128 * GB, 100 * GB)


class _LedgerRecorder:
    """Captures the posts handed to joint_pinned_host_error and accepts."""

    def __init__(self):
        self.posts = None

    def __call__(self, posts, total_bytes, available_bytes, *a, **k):
        self.posts = list(posts)
        return None


def _run(*, host_memory=ROOMY, recorder=None, **kwargs):
    base = dict(enable_hierarchical_cache=True, hicache_storage_backend="file")
    base.update(kwargs)
    args = ServerArgs(model_path="dummy", **base)
    patches = [
        mock.patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            lambda: host_memory,
        )
    ]
    if recorder is not None:
        patches.append(
            mock.patch(
                "sglang.srt.mem_cache.pinned_host_budget.joint_pinned_host_error",
                recorder,
            )
        )
    with patches[0]:
        if recorder is not None:
            with patches[1]:
                args._handle_hicache_host_role()
        else:
            args._handle_hicache_host_role()
    return args


def _rebind(**kwargs):
    base = dict(
        phase_flip_rebind_hicache=True,
        hicache_host_role="staging",
        hicache_size=6,
        hicache_mamba_host_mib=2400,
        tp_size=1,
        pp_size=3,
    )
    base.update(kwargs)
    return _run(**base)


class TestTheRebindRefusesAnIncompleteSizing(CustomTestCase):
    def _assert_l13(self, cm, role, size, mib):
        msg = str(cm.exception)
        self.assertIn("--phase-flip-rebind-hicache requires --hicache-host-role staging", msg)
        self.assertIn("--hicache-size > 0", msg)
        self.assertIn("--hicache-mamba-host-mib > 0", msg)
        self.assertIn(f"got role={role} size={size} mamba_mib={mib}", msg)

    def test_retention_role_is_refused(self):
        # T5a. The role is the whole point: a retention tier is ratio-shaped.
        with self.assertRaises(ValueError) as cm:
            _rebind(hicache_host_role="retention", hicache_size=0, hicache_mamba_host_mib=0)
        self._assert_l13(cm, "retention", 0, 0)

    def test_a_missing_size_is_refused(self):
        # T5b. No ratio fallback: the tp pin is row-coupled to this number.
        with self.assertRaises(ValueError) as cm:
            _rebind(hicache_size=0)
        self._assert_l13(cm, "staging", 0, 2400)

    def test_a_missing_anchor_budget_is_refused(self):
        # T5c. 0 would derive the anchor pool from the device pool per rank.
        with self.assertRaises(ValueError) as cm:
            _rebind(hicache_mamba_host_mib=0)
        self._assert_l13(cm, "staging", 6, 0)

    def test_the_refusal_names_the_reasons(self):
        with self.assertRaises(ValueError) as cm:
            _rebind(hicache_host_role="retention")
        msg = str(cm.exception)
        self.assertIn("row-coupled", msg)
        self.assertIn("MIN-synced", msg)


class TestTheSecondPhaseIsOnTheLedger(CustomTestCase):
    def test_the_tp_pin_and_the_anchor_pools_are_posted(self):
        # T5d. staging + 6 GB + 2400 MiB on 3 ranks: two extra named posts.
        rec = _LedgerRecorder()
        with self.assertLogs("sglang.srt.server_args", level="INFO") as logs:
            _rebind(recorder=rec)
        names = [p.name for p in rec.posts]
        tp_pin = [p for p in rec.posts if "phase-flip tp pin" in p.name]
        anchors = [p for p in rec.posts if "mamba anchor" in p.name]
        self.assertEqual(len(tp_pin), 1, names)
        self.assertEqual(len(anchors), 1, names)
        # ceiling estimate 2x (cell_tp / cell_pp0 = 2 on this cut), x ranks
        self.assertEqual(tp_pin[0].nbytes, 6 * GB * 3 * 2)
        self.assertIn("--hicache-size", tp_pin[0].flag)
        # MiB x ranks x 2 phases
        self.assertEqual(anchors[0].nbytes, 2400 * MIB * 3 * 2)
        self.assertIn("--hicache-mamba-host-mib", anchors[0].flag)
        # L11: one ledger line naming every term with its denominator
        ledger = [line for line in logs.output if "#810 host ledger" in line]
        self.assertEqual(len(ledger), 1, logs.output)
        self.assertIn("phase-flip tp pin ceiling 36.00 GB (2x)", ledger[0])
        self.assertIn("mamba anchor pools 15.10 GB (x3 ranks x2 phases)", ledger[0])
        self.assertIn("minus reserve", ledger[0])

    def test_the_joint_check_still_refuses_what_cannot_fit(self):
        # The posts are not decorative: 40 GB x 3 x 2 = 240 GB is refused
        # on a 100 GB machine, and the refusal names the pin.
        with self.assertRaises(ValueError) as cm:
            _rebind(hicache_size=40)
        self.assertIn("phase-flip tp pin", str(cm.exception))

    def test_a_staging_boot_without_the_rebind_is_byte_identical(self):
        # The default path: no rebind, no extra posts, the old line.
        rec = _LedgerRecorder()
        with self.assertLogs("sglang.srt.server_args", level="INFO") as logs:
            _run(hicache_host_role="staging", hicache_size=6, tp_size=1, pp_size=3, recorder=rec)
        self.assertEqual(len(rec.posts), 1, [p.name for p in rec.posts])
        self.assertIn("HiCache staging host tier", rec.posts[0].name)
        ledger = [line for line in logs.output if "#810 host ledger" in line]
        self.assertEqual(len(ledger), 1)
        self.assertNotIn("phase-flip", ledger[0])

    def test_a_retention_boot_without_the_rebind_runs_no_validation(self):
        args = _run(hicache_ratio=1.5)
        self.assertEqual(args.hicache_host_role, "retention")


if __name__ == "__main__":
    unittest.main()

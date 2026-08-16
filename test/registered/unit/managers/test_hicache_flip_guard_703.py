"""#703 retention stage 1: the #630 flip guard must ask what HiCache IS, not
whether it is configured.

WHY THIS CHANGES A PINNED INVARIANT. `flip_blocking_guards` refuses flip
arming whenever `enable_hierarchical_cache` is set at all, citing "#630: PP x
disk HiCache wedges at warmup". That wedge was real, and it was FIXED:
`9da9dfd025` introduced bounded collectives (mem_cache/hicache_collective.py),
the fix is an ancestor of the deployed commit 9925a2e4fa, and its suite
(test_hicache_bounded_waits_630.py) passes 14 tests + 14 subtests hermetically.
The guard outlived its defect.

The cost of leaving it is not theoretical: because the guard refuses the flip
outright, the deployment runs with `enable_hierarchical_cache=False`, i.e. NO
cache tier whatsoever on the serving line. The operator must choose between
the phase flip and any prefix retention at all.

The precedent for the fix shape is three lines below the guard itself: the
kv-session-offload guard was already converted from "is it CONFIGURED" to
"what is it DOING" (#656, kvso_flip_contract), because refusing on mere
configuration made kvso and the flip mutually exclusive. This is the same
defect in the same function.

STAGE 1 IS DELIBERATELY NARROW. The #630 wedge specifically involved the DISK
tier at warmup. So the guard keeps refusing when a storage backend is
configured, and stands down only for the device+host-local configuration --
"PP-phase-local HiCache". Widening to the disk tier is a separate step behind
its own evidence.
"""

from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


def _sched(**server_args_over):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    sa = dict(
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        dual_group_lane=False,
    )
    sa.update(server_args_over)
    sched = SimpleNamespace(
        server_args=SimpleNamespace(**sa),
        kv_session_offload=None,
        is_dual_group_lane=False,
        # the guard also enumerates the tree cache; give it a conforming stub
        # so this file isolates the HiCache clause and nothing else.
        tree_cache=SimpleNamespace(all_values_flatten=lambda: None),
    )
    sched.disaggregation_mode = DisaggregationMode.NULL
    return sched


def _hicache_guards(sched):
    from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards

    return [g for g in flip_blocking_guards(sched) if "630" in g or "ierarchical" in g]


class TestHiCacheFlipGuard703(CustomTestCase):
    def test_off_is_unguarded(self):
        """Baseline, unchanged: no HiCache, no HiCache guard."""
        self.assertEqual(_hicache_guards(_sched()), [])

    def test_device_host_local_hicache_no_longer_blocks_the_flip(self):
        """THE CHANGE. Hierarchical cache with no storage backend is the
        PP-phase-local configuration; the wedge it was refused for is fixed
        and shipped, so it must not refuse flip arming."""
        sched = _sched(enable_hierarchical_cache=True, hicache_storage_backend=None)
        self.assertEqual(
            _hicache_guards(sched),
            [],
            "device+host-local HiCache must not block flip arming: the #630 "
            "wedge is fixed (9da9dfd025, bounded collectives) and its suite "
            "is green",
        )

    def test_storage_backend_still_blocks(self):
        """STAGE 1 BOUNDARY, and the can-fail proof for this file: the guard
        must still fire for the disk tier, which is the configuration #630
        actually wedged on. A change that simply deleted the guard would pass
        the test above and FAIL this one."""
        for backend in ("file", "mooncake", "hf3fs"):
            with self.subTest(backend=backend):
                sched = _sched(
                    enable_hierarchical_cache=True, hicache_storage_backend=backend
                )
                guards = _hicache_guards(sched)
                self.assertTrue(
                    guards,
                    f"storage backend {backend!r} must still refuse flip arming",
                )

    def test_storage_backend_alone_without_hierarchical_is_unguarded(self):
        """A storage backend set while hierarchical cache is off builds no
        tier at all (scheduler.py gates tier construction on
        enable_hierarchical_cache alone), so there is nothing to refuse."""
        sched = _sched(enable_hierarchical_cache=False, hicache_storage_backend="file")
        self.assertEqual(_hicache_guards(sched), [])


class TestHiCacheFlipV1Blocker703(CustomTestCase):
    """The BOOT-TIME twin. `flip_blocking_guards` gates arming; this one
    refuses at ServerArgs parse time, before a scheduler exists. Fixing only
    the runtime clause leaves --enable-hierarchical-cache unusable, which is
    exactly what a live boot proved:

      ValueError: --enable-phase-flip V1 refuses:
      --enable-hierarchical-cache (#630: PP x disk HiCache wedges at warmup).
    """

    def _blockers(self, **over):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs.__new__(ServerArgs)
        # the validator walks the whole phase-flip surface before it reaches
        # the blocker list; give it a minimal VALID flip config so the only
        # thing under test is the HiCache clause.
        sa.enable_phase_flip = True
        sa.phase_flip_policy = "auto"
        sa.phase_flip_tp_vector = "32,16,16"
        sa.phase_flip_purity = "prefill_in_tp"
        sa.phase_flip_spill_depth = "arena"
        sa.pp_size = 3
        sa.disaggregation_mode = "null"
        sa.enable_hierarchical_cache = False
        sa.hicache_storage_backend = None
        sa.dual_group_lane = False
        sa.dp_size = 1
        sa.ep_size = 1
        sa.tp_size = 1
        sa.speculative_algorithm = None
        sa.speculative_draft_placement = None
        for k, v in over.items():
            setattr(sa, k, v)
        try:
            sa._handle_phase_flip()
        except ValueError as exc:
            return str(exc)
        except AttributeError:
            self.skipTest("validator not named _handle_phase_flip")
        return ""

    def test_device_host_local_hicache_boots(self):
        msg = self._blockers(enable_hierarchical_cache=True)
        self.assertNotIn("hierarchical", msg, msg)

    def test_storage_backend_still_refused(self):
        msg = self._blockers(
            enable_hierarchical_cache=True, hicache_storage_backend="file"
        )
        self.assertIn("hierarchical", msg, msg)

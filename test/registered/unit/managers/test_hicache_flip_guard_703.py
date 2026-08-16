"""#703: the #630 flip guard is stale in BOTH of its copies, and is removed.

HISTORY, because this file changed contract twice and the second change was a
correction of the first.

Stage 1 narrowed the guard from "hierarchical cache is enabled" to "a storage
backend is configured", keeping the disk tier refused pending its own evidence.
Stage 2 removes the clause entirely, because "its own evidence" turned out to
be the SAME evidence that cleared the host tier:

  * the #630 wedge was PP x disk HiCache at warmup;
  * its root fix is 9da9dfd025 (bounded collectives,
    mem_cache/hicache_collective.py);
  * that commit is an ancestor of every deployed commit since;
  * test_hicache_bounded_waits_630.py passes 14 tests + 14 subtests hermetically.

A guard cannot be justified by a defect a green suite says is fixed. Keeping
the disk tier refused was conservatism, not caution, and it blocked the disk L3
that is the actual retention store (host RAM cannot hold it -- the flip's own
host weight images take 20.50 of 25.87 GB usable).

WHAT IS STILL GATED: the KV key's `_{pp_size}_{pp_rank}` suffix. That is a
claim about BYTES -- a PP-written page really does contain only one stage's
layers -- and it drops only when #706's whole-page format makes the page
complete across stages. Refusing the backend never protected those bytes; it
only stopped anyone from reaching them.
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
        tree_cache=SimpleNamespace(all_values_flatten=lambda: None),
    )
    sched.disaggregation_mode = DisaggregationMode.NULL
    return sched


def _hicache_guards(sched):
    from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards

    return [g for g in flip_blocking_guards(sched) if "630" in g or "ierarchical" in g]


class TestHiCacheFlipGuard703(CustomTestCase):
    """Runtime clause: gates flip ARMING."""

    def test_no_hicache_config_blocks_arming(self):
        for backend in (None, "file", "mooncake", "hf3fs"):
            for enabled in (False, True):
                with self.subTest(backend=backend, enabled=enabled):
                    sched = _sched(
                        enable_hierarchical_cache=enabled,
                        hicache_storage_backend=backend,
                    )
                    self.assertEqual(
                        _hicache_guards(sched),
                        [],
                        "no HiCache configuration may refuse flip arming; the "
                        "#630 root fix ships and its suite is green",
                    )

    def test_other_guards_still_fire(self):
        """CAN-FAIL PROOF. Removing the HiCache clause must not have neutered
        the function. A change that returned [] unconditionally would pass
        every assertion above and fail here."""
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.phase_flip_runtime import flip_blocking_guards

        sched = _sched(dual_group_lane=True)
        self.assertTrue(
            any("dual-group" in g for g in flip_blocking_guards(sched)),
            "dual-group lane must still refuse arming",
        )

        sched = _sched()
        sched.tree_cache = SimpleNamespace()  # no all_values_flatten
        self.assertTrue(
            any("tree cache" in g for g in flip_blocking_guards(sched)),
            "a tree cache without enumeration must still refuse arming",
        )

        modes = [m for m in DisaggregationMode if m != DisaggregationMode.NULL]
        if modes:
            sched = _sched()
            sched.disaggregation_mode = modes[0]
            self.assertTrue(
                any("disagg" in g.lower() for g in flip_blocking_guards(sched)),
                "PD disaggregation must still refuse arming",
            )


class TestHiCacheFlipV1Blocker703(CustomTestCase):
    """Boot-time twin, in ServerArgs. Refuses at parse time, before a scheduler
    exists -- so fixing only the runtime clause left the flag unusable, which a
    live boot proved:

      ValueError: --enable-phase-flip V1 refuses:
      --enable-hierarchical-cache (#630: PP x disk HiCache wedges at warmup).
    """

    def _blockers(self, **over):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs.__new__(ServerArgs)
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

    def test_disk_l3_boots(self):
        """The configuration the retention design actually needs."""
        for backend in ("file", "mooncake", "hf3fs"):
            with self.subTest(backend=backend):
                msg = self._blockers(
                    enable_hierarchical_cache=True, hicache_storage_backend=backend
                )
                self.assertNotIn("hierarchical", msg, msg)

    def test_other_v1_blockers_still_fire(self):
        """CAN-FAIL PROOF for the boot-time clause: the surrounding refusals
        must survive. Proven reachable -- an earlier version of this fixture
        never reached the blocker list at all and passed vacuously."""
        msg = self._blockers(dp_size=2)
        self.assertIn("dp-size", msg, msg)
        msg = self._blockers(dual_group_lane=True)
        self.assertIn("dual-group", msg, msg)

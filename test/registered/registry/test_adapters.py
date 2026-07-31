"""The three lane adapters: costing, launch arguments, and refused rungs.

Hermetic. Nothing boots. What is checked is everything that decides *whether*
a boot is attempted and *with which arguments* -- the part that is expensive to
get wrong and cheap to test, and the part a card window cannot cover more than
once or twice.

    python -m pytest test/registered/registry/test_adapters.py -v
"""

import sys
import unittest

from sglang.srt.registry.adapter import AdapterContext, AdapterError, EstimateError
from sglang.srt.registry.adapters.class1_srt import Class1SrtAdapter
from sglang.srt.registry.adapters.class2_diffusion import Class2DiffusionAdapter
from sglang.srt.registry.adapters.class3_utility import build as build_class3
from sglang.srt.registry.ledger import MIB
from sglang.srt.registry.spec import EngineClass, EngineSpec, ResidencyState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024 * MIB
CARD_A = "GPU-aaaa0000-0000-0000-0000-00000000000a"
CARD_B = "GPU-bbbb0000-0000-0000-0000-00000000000b"
TOTALS = {CARD_A: 20 * GIB, CARD_B: 32 * GIB}


def context():
    return AdapterContext(card_totals=TOTALS)


def class1_spec(engine_id="llm", placement=(CARD_A,), **launch):
    base = {"model_path": "/models/qwen", "port": 30100}
    base.update(launch)
    return EngineSpec(
        engine_id=engine_id,
        klass=EngineClass.AUTOREGRESSIVE,
        adapter="class1_srt",
        placement=tuple(placement),
        launch=base,
    )


class Class1EstimateTest(unittest.TestCase):
    def test_absolute_mib_budget_is_taken_verbatim(self):
        """No ceiling, no safety factor, no rounding down. The whole budget."""
        spec = class1_spec(rank_gpu_memory_mib=15_000)
        adapter = Class1SrtAdapter(spec, context())
        profile = adapter.estimate(spec, (CARD_A,))
        self.assertEqual(profile.peak_bytes[CARD_A], 15_000 * MIB)
        self.assertEqual(profile.posts[CARD_A]["declared_rank_budget"], 15_000 * MIB)

    def test_two_ranks_on_one_card_reserve_twice(self):
        """The physical-impossibility check needs the per-card sum, not per rank."""
        spec = class1_spec(
            tp_size=2,
            placement=(CARD_B,),
            rank_cards=[CARD_B, CARD_B],
            rank_gpu_memory_mib=15_000,
        )
        adapter = Class1SrtAdapter(spec, context())
        profile = adapter.estimate(spec, (CARD_B,))
        self.assertEqual(profile.peak_bytes[CARD_B], 30_000 * MIB)

    def test_a_fraction_resolves_against_each_cards_own_total(self):
        """The same fraction is different bytes on unlike cards. That is the point."""
        spec = class1_spec(
            tp_size=2,
            placement=(CARD_A, CARD_B),
            gpu_memory_utilization=0.5,
        )
        adapter = Class1SrtAdapter(spec, context())
        profile = adapter.estimate(spec, (CARD_A, CARD_B))
        self.assertEqual(profile.peak_bytes[CARD_A], 10 * GIB)
        self.assertEqual(profile.peak_bytes[CARD_B], 16 * GIB)

    def test_a_spec_without_a_budget_is_refused_with_both_options_named(self):
        with self.assertRaises(EstimateError) as ctx:
            Class1SrtAdapter(class1_spec(), context())
        message = str(ctx.exception)
        self.assertIn("rank_gpu_memory_mib", message)
        self.assertIn("gpu_memory_utilization", message)

    def test_pipeline_and_data_parallelism_are_refused_by_name(self):
        for flag in ("pp_size", "dp_size", "ep_size"):
            with self.assertRaises(EstimateError) as ctx:
                Class1SrtAdapter(
                    class1_spec(rank_gpu_memory_mib=1000, **{flag: 2}), context()
                )
            self.assertIn(flag, str(ctx.exception))
            self.assertIn("tensor parallelism", str(ctx.exception))

    def test_rank_cards_must_have_one_entry_per_rank(self):
        with self.assertRaises(EstimateError):
            Class1SrtAdapter(
                class1_spec(
                    tp_size=3, rank_cards=[CARD_A, CARD_B], rank_gpu_memory_mib=1000
                ),
                context(),
            )

    def test_rank_cards_outside_the_placement_is_an_error(self):
        spec = class1_spec(
            tp_size=1,
            placement=(CARD_A,),
            rank_cards=[CARD_B],
            rank_gpu_memory_mib=1000,
        )
        adapter = Class1SrtAdapter(spec, context())
        with self.assertRaises(EstimateError) as ctx:
            adapter.estimate(spec, (CARD_A,))
        self.assertIn(CARD_B, str(ctx.exception))

    def test_ranks_cannot_be_spread_over_fewer_cards_without_saying_how(self):
        spec = class1_spec(
            tp_size=3, placement=(CARD_A, CARD_B), rank_gpu_memory_mib=1000
        )
        adapter = Class1SrtAdapter(spec, context())
        with self.assertRaises(EstimateError) as ctx:
            adapter.estimate(spec, (CARD_A, CARD_B))
        self.assertIn("rank_cards", str(ctx.exception))


class Class1LaunchTest(unittest.TestCase):
    def build(self, spec, cards):
        adapter = Class1SrtAdapter(spec, context())
        adapter.bind(cards)
        return adapter

    def test_rank_gpu_ids_index_into_the_pinned_visible_set(self):
        """UUIDs are the registry's identity; the CLI wants an integer.

        There is no single right integer for a UUID: CUDA enumerates
        FASTEST_FIRST and NVML by PCI bus, and this rig disagrees between the
        two (the 5090 is CUDA 0 and NVML 1). Emitting either one is a coin
        flip that boots the engine on the wrong card. The adapter instead
        pins the child to its own cards in placement order, so the index is
        defined by construction.
        """
        spec = class1_spec(
            tp_size=3,
            placement=(CARD_A, CARD_B),
            rank_cards=[CARD_B, CARD_B, CARD_A],
            rank_gpu_memory_mib=15_000,
        )
        adapter = self.build(spec, (CARD_A, CARD_B))
        argv = adapter.build_argv()
        self.assertEqual(argv[:3], [sys.executable, "-m", "sglang.launch_server"])
        self.assertIn("--rank-gpu-id", argv)
        self.assertEqual(argv[argv.index("--rank-gpu-id") + 1], "1,1,0")
        self.assertEqual(argv[argv.index("--rank-gpu-memory-mib") + 1], "15000")
        self.assertNotIn("--gpu-memory-utilization", argv)
        self.assertEqual(adapter.visible_devices(), f"{CARD_A},{CARD_B}")

    def test_the_visible_set_is_exactly_the_placement(self):
        spec = class1_spec(placement=(CARD_B,), rank_gpu_memory_mib=8000)
        adapter = self.build(spec, (CARD_B,))
        self.assertEqual(adapter.visible_devices(), CARD_B)
        self.assertEqual(adapter.visible_rank_indices(), (0,))

    def test_the_memory_saver_is_on_so_the_rungs_can_free_anything(self):
        spec = class1_spec(rank_gpu_memory_mib=8000)
        self.assertIn("--enable-memory-saver", self.build(spec, (CARD_A,)).build_argv())

    def test_the_memory_saver_can_be_opted_out_of(self):
        spec = class1_spec(rank_gpu_memory_mib=8000, enable_memory_saver=False)
        self.assertNotIn(
            "--enable-memory-saver", self.build(spec, (CARD_A,)).build_argv()
        )

    def test_the_global_utilisation_flag_is_used_only_without_a_mib_budget(self):
        spec = class1_spec(gpu_memory_utilization=0.82)
        argv = self.build(spec, (CARD_A,)).build_argv()
        self.assertIn("--gpu-memory-utilization", argv)
        self.assertNotIn("--rank-gpu-id", argv)

    def test_hibernate_flags_appear_as_the_mutually_required_pair(self):
        spec = class1_spec(rank_gpu_memory_mib=8000, hibernate_dir="/tmp/hib")
        argv = self.build(spec, (CARD_A,)).build_argv()
        self.assertIn("--enable-weights-disk-backup", argv)
        self.assertEqual(argv[argv.index("--hibernate-dir") + 1], "/tmp/hib")

    def test_extra_args_are_appended_verbatim(self):
        spec = class1_spec(
            rank_gpu_memory_mib=8000,
            extra_args=["--max-model-len", "-1", "--enable-prefix-caching"],
        )
        argv = self.build(spec, (CARD_A,)).build_argv()
        self.assertEqual(
            argv[-3:], ["--max-model-len", "-1", "--enable-prefix-caching"]
        )

    def test_extra_args_accept_a_shell_string(self):
        spec = class1_spec(rank_gpu_memory_mib=8000, extra_args="--kv-cache-dtype fp8")
        argv = self.build(spec, (CARD_A,)).build_argv()
        self.assertEqual(argv[-2:], ["--kv-cache-dtype", "fp8"])


class Class1LadderTest(unittest.TestCase):
    def test_warm_host_is_refused_with_the_reason(self):
        spec = class1_spec(rank_gpu_memory_mib=1000)
        adapter = Class1SrtAdapter(spec, context())
        for call in (adapter.promote, adapter.demote):
            with self.assertRaises(AdapterError) as ctx:
                call(ResidencyState.WARM_HOST)
            self.assertIn("WARM_HOST", str(ctx.exception))
        self.assertIn("HOT / WARM_GPU / COLD", self._promote_error(adapter))

    def _promote_error(self, adapter):
        try:
            adapter.promote(ResidencyState.WARM_HOST)
        except AdapterError as exc:
            return str(exc)
        raise AssertionError("expected a refusal")

    def test_demoting_a_cold_engine_to_cold_is_a_no_op(self):
        adapter = Class1SrtAdapter(class1_spec(rank_gpu_memory_mib=1000), context())
        adapter.demote(ResidencyState.COLD)
        self.assertEqual(adapter.state(), ResidencyState.COLD)

    def test_cold_engine_is_healthy_and_holds_nothing(self):
        adapter = Class1SrtAdapter(class1_spec(rank_gpu_memory_mib=1000), context())
        self.assertTrue(adapter.health().ok)
        self.assertEqual(adapter.pids(), ())
        self.assertEqual(adapter.measured(), {})


class Class2Test(unittest.TestCase):
    def spec(self, **posts):
        declared = {
            "weights_bytes": 8000,
            "activation_peak_bytes": 4000,
            "latent_bytes": 200,
            "text_encoder_bytes": 1500,
            "vae_bytes": 300,
            "ctx_overhead_bytes": 500,
        }
        declared.update(posts)
        return EngineSpec(
            engine_id="flux",
            klass=EngineClass.DIFFUSION,
            adapter="class2_diffusion",
            placement=(CARD_B,),
            launch={"posts_mib": declared},
        )

    def test_the_estimate_is_the_sum_of_the_declared_posts(self):
        spec = self.spec()
        profile = Class2DiffusionAdapter(spec, context()).estimate(spec, (CARD_B,))
        self.assertEqual(profile.peak_bytes[CARD_B], 14_500 * MIB)
        # Steady drops the transient activation peak; the reservation keeps it.
        self.assertEqual(profile.steady_bytes[CARD_B], 10_500 * MIB)
        self.assertEqual(profile.slack_bytes()[CARD_B], 4000 * MIB)

    def test_a_missing_post_is_named_rather_than_defaulted(self):
        declared = self.spec().launch["posts_mib"]
        del declared["activation_peak_bytes"]
        spec = EngineSpec(
            engine_id="flux",
            klass=EngineClass.DIFFUSION,
            adapter="class2_diffusion",
            placement=(CARD_B,),
            launch={"posts_mib": declared},
        )
        with self.assertRaises(EstimateError) as ctx:
            Class2DiffusionAdapter(spec, context())
        self.assertIn("activation_peak_bytes", str(ctx.exception))

    def test_warm_gpu_rung_is_refused_with_the_ladder(self):
        # M3 launches, but Class 2 exposes no WARM_GPU endpoint; the ladder is
        # HOT / WARM_HOST / COLD and the refusal says so.
        adapter = Class2DiffusionAdapter(self.spec(), context())
        with self.assertRaises(AdapterError) as ctx:
            adapter.promote(ResidencyState.WARM_GPU)
        self.assertIn("WARM_HOST", str(ctx.exception))

    def test_promotion_needs_a_model_path_to_boot(self):
        # Estimate-only registration works from posts alone (§7.4); booting does
        # not -- it needs a model to load.
        adapter = Class2DiffusionAdapter(self.spec(), context())
        with self.assertRaises(AdapterError) as ctx:
            adapter.promote(ResidencyState.HOT)
        self.assertIn("model_path", str(ctx.exception))

    def test_multi_card_diffusion_needs_opt_in_and_says_so(self):
        # Without enable_uneven_sp a multi-card tenant is out of scope; the
        # message names M4, where the uneven-SP collective wiring lands.
        spec = self.spec()
        with self.assertRaises(EstimateError) as ctx:
            Class2DiffusionAdapter(spec, context()).estimate(spec, (CARD_A, CARD_B))
        self.assertIn("M4", str(ctx.exception))


class Class3Test(unittest.TestCase):
    def pooling_spec(self, **launch):
        base = {"model_path": "/models/bge", "port": 30200, "rank_gpu_memory_mib": 3000}
        base.update(launch)
        return EngineSpec(
            engine_id="embed",
            klass=EngineClass.UTILITY,
            adapter="class3_utility",
            placement=(CARD_A,),
            launch=base,
        )

    def test_a_pooling_tenant_launches_with_is_embedding(self):
        spec = self.pooling_spec()
        adapter = build_class3(spec, context())
        adapter.bind((CARD_A,))
        self.assertIn("--is-embedding", adapter.build_argv())
        self.assertEqual(adapter.klass, 3)

    def test_a_pooling_tenant_has_no_warm_gpu_rung(self):
        adapter = build_class3(self.pooling_spec(), context())
        with self.assertRaises(AdapterError) as ctx:
            adapter.promote(ResidencyState.WARM_GPU)
        self.assertIn("HOT / COLD", str(ctx.exception))

    def test_an_opaque_process_tenant_declares_its_own_budget(self):
        spec = EngineSpec(
            engine_id="video",
            klass=EngineClass.UTILITY,
            adapter="class3_utility",
            placement=(CARD_B,),
            launch={"mode": "process", "argv": ["/bin/true"], "budget_mib": 12_000},
        )
        adapter = build_class3(spec, context())
        profile = adapter.estimate(spec, (CARD_B,))
        self.assertEqual(profile.peak_bytes[CARD_B], 12_000 * MIB)

    def test_an_opaque_tenant_without_a_budget_is_refused(self):
        spec = EngineSpec(
            engine_id="video",
            klass=EngineClass.UTILITY,
            adapter="class3_utility",
            placement=(CARD_B,),
            launch={"mode": "process", "argv": ["/bin/true"]},
        )
        with self.assertRaises(EstimateError) as ctx:
            build_class3(spec, context())
        self.assertIn("budget_mib", str(ctx.exception))

    def test_an_unknown_mode_names_the_known_ones(self):
        spec = EngineSpec(
            engine_id="x",
            klass=EngineClass.UTILITY,
            adapter="class3_utility",
            placement=(CARD_B,),
            launch={"mode": "telepathy"},
        )
        with self.assertRaises(EstimateError) as ctx:
            build_class3(spec, context())
        self.assertIn("pooling", str(ctx.exception))
        self.assertIn("process", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

"""Task #114 (24.09.): group P's prefill transient is priced from the chunk.

fnFL2x121/x122 booted the 5090 stage at residency 0.45/0.40 with chunk 16384,
passed every launch gate (W120/W122, the runtime's KV sizing) and died in the
FIRST chunk's forward with a CUDA OOM 320-384 MiB short. Both books priced the
reference boots' 1024 MiB activation post (chunk 4096) on a 16384-token chunk,
while the boot's own ``[vram-peak]`` instrument measured 3.58 GiB on that stage
-- growing with the chunk on every stage (H41 support points, PP0/PP1/PP2:
512 -> 0.13/0.13/0.13, 4096 -> 0.92/0.82/0.85, 8192 -> 1.82/1.63/1.60, 16384
-> 3.61/3.21/3.19 GiB; linear between them, W131 outside).

Two seams, one measurement:

* the pool model's ``activation_reserve_mib`` (the dry-run's cut solver and
  its pool floor = cap + chunk) follows the chunk P boots with;
* group P's rank env carries the runtime's own residual post
  (``SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB``), so the sizer subtracts the
  transient before the KV pool instead of sizing the pool into its room.

Hermetic: no GPU, no torch device.
"""

import os

import pytest

from sglang.srt.weg2 import launcher


@pytest.fixture
def scrub_env(monkeypatch):
    monkeypatch.delenv(launcher.P_PREFILL_TRANSIENT_ENV, raising=False)
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)


def _build(group: str):
    return launcher.build_env(
        tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
        debug_hold=False, tag="probe", group=group,
    )


class TestMeasuredVector:
    def test_reproduces_the_measured_points_per_stage(self):
        # H41: the support points, GiB from the [vram-peak] lines (max over
        # the boots of each point). PP2 at 16384 is 3.19, not #114's 2.43
        # (that was the draft forward's line after a peak reset, x118).
        measured = {
            512: (0.13, 0.13, 0.13),
            4096: (0.92, 0.82, 0.85),
            8192: (1.82, 1.63, 1.60),
            16384: (3.61, 3.21, 3.19),
        }
        for chunk, gib in measured.items():
            vec = launcher.p_prefill_transient_vector_mib(chunk)
            assert len(vec) == 3
            for got_mib, want_gib in zip(vec, gib):
                assert abs(got_mib / 1024.0 - want_gib) < 0.006, (chunk, vec)

    def test_linear_between_the_points(self):
        a = launcher.p_prefill_transient_vector_mib(8192)
        b = launcher.p_prefill_transient_vector_mib(16384)
        mid = launcher.p_prefill_transient_vector_mib(12288)
        for x, y, m in zip(a, b, mid):
            assert abs(m - (x + y) / 2.0) <= 0.1

    def test_outside_the_support_is_refused_by_name(self):
        with pytest.raises(launcher.Weg2LaunchRefused, match="W131"):
            launcher.p_prefill_transient_vector_mib(32768)
        with pytest.raises(launcher.Weg2LaunchRefused, match="W131"):
            launcher.p_prefill_transient_vector_mib(256)

    def test_the_binding_stage_is_the_5090_stage(self):
        vec = launcher.p_prefill_transient_vector_mib(16384)
        assert vec[0] == max(vec)


class TestPoolModelPost:
    def test_reference_chunk_prices_byte_identically_to_1286(self):
        # chunk 4096: measured 942 MiB < 1024 -> the reference post stands.
        assert launcher.p_prefill_activation_reserve_mib(None, 4096) == 1024.0

    def test_below_the_support_the_reference_post_dominates(self):
        # the transient grows with the chunk: below 512 it is <= 133 MiB.
        assert launcher.p_prefill_activation_reserve_mib(None, 256) == 1024.0

    def test_wide_chunk_prices_the_measured_transient(self):
        post = launcher.p_prefill_activation_reserve_mib(None, 16384)
        assert post == max(launcher.p_prefill_transient_vector_mib(16384))
        assert 3690.0 < post < 3700.0  # 3.61 GiB, max of x118/x141/x145/x146

    def test_unmeasured_wide_chunk_is_refused(self):
        with pytest.raises(launcher.Weg2LaunchRefused, match="W131"):
            launcher.p_prefill_activation_reserve_mib(None, 32768)

    def test_operator_pin_wins(self):
        assert launcher.p_prefill_activation_reserve_mib(2048.0, 16384) == 2048.0
        assert launcher.p_prefill_activation_reserve_mib(0.0, 16384) == 0.0

    def test_parser_default_is_unpinned(self):
        ap = launcher.build_parser()
        assert ap.get_default("pp_cut_activation_reserve_mib") is None


class TestRuntimeSeam:
    def test_env_name_is_the_runtime_constant(self):
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

        assert launcher.P_PREFILL_TRANSIENT_ENV == mx.PREFILL_TRANSIENT_ENV

    def test_group_p_carries_the_vector_for_its_chunk(self, scrub_env):
        env = _build("P")
        text = env[launcher.P_PREFILL_TRANSIENT_ENV]
        want = launcher.p_prefill_transient_vector_mib(launcher.P_CHUNKED_PREFILL_TOKENS)
        parts = [float(x) for x in text.split(",")]
        assert len(parts) == 3
        for got, exp in zip(parts, want):
            assert abs(got - exp) <= 0.5

    def test_the_runtime_reads_back_the_same_number_per_rank(self, scrub_env):
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as mx

        env = _build("P")
        text = env[launcher.P_PREFILL_TRANSIENT_ENV]
        want = launcher.p_prefill_transient_vector_mib(launcher.P_CHUNKED_PREFILL_TOKENS)
        for rank in range(3):
            assert abs(mx.prefill_transient_mib_for_rank(text, rank) - want[rank]) <= 0.5
        assert mx.prefill_transient_mib_for_rank(text, 3) == 0.0

    def test_operator_value_wins_for_p(self, scrub_env, monkeypatch):
        monkeypatch.setenv(launcher.P_PREFILL_TRANSIENT_ENV, "1,2,3")
        assert _build("P")[launcher.P_PREFILL_TRANSIENT_ENV] == "1,2,3"

    def test_other_groups_never_inherit_the_post(self, scrub_env, monkeypatch):
        monkeypatch.setenv(launcher.P_PREFILL_TRANSIENT_ENV, "1,2,3")
        assert launcher.P_PREFILL_TRANSIENT_ENV not in _build("D")
        assert launcher.P_PREFILL_TRANSIENT_ENV not in _build("")

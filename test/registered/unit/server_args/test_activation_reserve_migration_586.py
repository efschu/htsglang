"""#586: the legacy KV-sizing consumers now read the calibrated activation.

Three sites in model_runner_kv_cache_mixin subtract a prefill-activation
reserve from the memory available to the KV pool. All three used
mamba_pre_capture_reserve_mb, i.e. the heuristic the 2026-08-05 window
falsified (3968 MiB booked, <= 1766 MiB actually free on the binding card while
it completed a 70018-token prefill).

BEHAVIOUR DELTA, deliberate and tested here:
  * calibrated rig  -> the sites see the MEASURED activation, so the KV pool
                       grows by (heuristic - measured) per rank.
  * uncalibrated rig -> unchanged numbers, plus a one-time loud warning. It
                       does NOT refuse: these are on every existing recipe's
                       boot path, and making an unprobed rig unbootable would
                       be a worse regression than an over-reserve.
"""

import pytest

from sglang.srt.mem_ledger.activation import FootprintProvenance, PhaseFootprint
from sglang.srt.server_args import ServerArgs

HEURISTIC = 512 + 2048 * 1.5 + 3 * 1 / 8 * 1024  # 3968


def make_args(**kw):
    args = ServerArgs.__new__(ServerArgs)
    defaults = dict(
        model_path="/nonexistent/model",
        chunked_prefill_size=2048,
        max_prefill_tokens=16384,
        tp_size=3,
        pp_size=1,
        max_running_requests=4,
        disaggregation_mode="null",
        kv_cache_dtype="fp8_e4m3",
        speculative_num_draft_tokens=4,
        trust_remote_code=True,
        revision=None,
        model_config_parser="auto",
    )
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


@pytest.fixture(autouse=True)
def _reset_warn_latch():
    ServerArgs._activation_heuristic_warned = False
    yield
    ServerArgs._activation_heuristic_warned = False


def install(monkeypatch, footprint):
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.activation.resolve_phase_footprint",
        lambda uuid, **kw: footprint,
    )
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.calibration.live_fingerprint",
        lambda **kw: ("fp586", [], "drv"),
    )
    monkeypatch.setattr("sglang.srt.registry.nvml.current_device_uuid", lambda: "GPU-x")
    monkeypatch.setattr(
        "sglang.srt.mem_ledger.engine._model_architectures", lambda sa: ()
    )


MEASURED = PhaseFootprint(
    activation_mib=1766,
    capture_mib=640,
    provenance=FootprintProvenance.MEASURED_PEAK,
    source="test",
    card_uuid="GPU-x",
)


def test_calibrated_rig_sees_the_measured_activation(monkeypatch):
    args = make_args()
    install(monkeypatch, MEASURED)
    assert args.activation_reserve_mb(20480) == 1766


def test_the_falsified_number_is_unreachable_on_a_calibrated_rig(monkeypatch):
    args = make_args()
    install(monkeypatch, MEASURED)
    got = args.activation_reserve_mb(20480)
    assert got != HEURISTIC
    assert got < HEURISTIC, "the whole point is that it over-reserved"
    # ...and the KV pool gains exactly the difference, per rank.
    assert HEURISTIC - got == 2202


def test_uncalibrated_rig_keeps_booting_but_warns_once(monkeypatch, caplog):
    args = make_args()
    install(monkeypatch, None)
    with caplog.at_level("WARNING"):
        first = args.activation_reserve_mb(20480)
        second = args.activation_reserve_mb(20480)
    assert first == HEURISTIC == second, "unprobed rigs must still boot"
    warnings = [r for r in caplog.records if "INHERITED activation" in r.getMessage()]
    assert len(warnings) == 1, "the warning is per process, not per call site"
    text = warnings[0].getMessage()
    assert "falsified" in text
    assert "probe_activation.py" in text


def test_the_three_kv_sizing_sites_call_the_resolver_not_the_heuristic():
    """Pin the migration itself: a future edit that reaches past the resolver
    would silently restore the over-reserve."""
    import inspect

    from sglang.srt.model_executor import model_runner_kv_cache_mixin as m

    src = inspect.getsource(m)
    assert "activation_reserve_mb(" in src
    assert "mamba_pre_capture_reserve_mb(" not in src, (
        "a KV-sizing site still calls the falsified heuristic directly"
    )


def test_resolver_failure_degrades_to_the_heuristic_rather_than_crashing(monkeypatch):
    """A boot must not die because a probe cache is unreadable."""
    args = make_args()

    def boom(*a, **k):
        raise RuntimeError("cache on fire")

    monkeypatch.setattr(
        "sglang.srt.mem_ledger.activation.resolve_phase_footprint", boom
    )
    assert args.activation_reserve_mb(20480) == HEURISTIC

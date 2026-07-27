"""Unit tests for the dashboard's MLP-split crossover panel.

    python3 -m pytest test_crossover_panel.py -q

The panel has one job beyond showing numbers: never show a crossover as this
rig's when it is not one, and never keep computing silently on a finding that
is stale or was taken under throttling.
"""

import json
import os
import tempfile
import time

from server import crossover_state


def _write(tmp, **over):
    payload = {
        "rig": {
            "cards": ["card A", "card B", "card B"],
            "model": "a-model",
            "quant": "fp8",
            "tp_size": 3,
        },
        "points": [
            {
                "vector": [3, 1, 1],
                "prefill_ms_per_prompt_token_saved": 0.05,
                "decode_ms_per_output_token_cost": 1.0,
                "prefill_gain_pct": 7.2,
                "decode_cost_pct": 6.0,
            }
        ],
        "provenance": "measured_here",
        "measured_at": time.time(),
        "cache_bypass_proven": True,
    }
    payload.update(over)
    path = os.path.join(tmp, "mlp_crossover.json")
    with open(path, "w") as f:
        json.dump({"findings": [payload]}, f)
    return path


def test_no_local_finding_offers_the_measurement():
    st = crossover_state("/nonexistent/mlp_crossover.json")
    assert st["state"] == "unmeasured"
    assert st["usable"] is False
    assert {t["key"] for t in st["offer"]} == {"quick", "thorough"}
    assert all(t["est_runtime_min"] > 0 for t in st["offer"])


def test_a_local_finding_is_shown_with_its_rig_and_age():
    with tempfile.TemporaryDirectory() as tmp:
        st = crossover_state(_write(tmp))
    assert st["state"] == "measured_here"
    assert st["usable"] is True
    assert "card A" in st["rig"]
    assert st["age_days"] >= 0
    assert st["table"][0]["break_even_prompt_to_output"] == 20.0


def test_a_stale_finding_is_refused_and_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        st = crossover_state(
            _write(tmp, measured_at=time.time() - 400 * 86400)
        )
    assert st["usable"] is False
    assert any("stale" in c.lower() for c in st["caveats"])
    assert st["offer"], "a refused finding must still offer the measurement"


def test_a_throttled_finding_is_kept_and_marked():
    with tempfile.TemporaryDirectory() as tmp:
        st = crossover_state(
            _write(tmp, throttled=True, throttle_reason="sw_thermal_slowdown")
        )
    assert st["usable"] is True
    assert any("throttl" in c.lower() for c in st["caveats"])


def test_a_finding_without_cache_proof_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        st = crossover_state(_write(tmp, cache_bypass_proven=False))
    assert st["usable"] is False
    assert any("cache" in c.lower() for c in st["caveats"])


def test_the_state_is_json_serialisable():
    with tempfile.TemporaryDirectory() as tmp:
        st = crossover_state(_write(tmp))
    json.dumps(st)


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)

# SPDX-License-Identifier: Apache-2.0
"""#452 -- the capturable MoE offload decode path refuses again, by name.

#443 ported the DeepSeek-V4 GGUF expert fetch onto the capturable path and
shipped it BOOT-PENDING behind ``SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1``. The boot ran
on 2026-08-02 and refuted it
(``/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/RESULTS.md``):
capture succeeds (B1), but the replayed arm decodes DIFFERENT text from the same
greedy prompt (B2) and is 6.60x slower per decoded token (B4).

This file pins the protective re-gate. It is deliberately about the DECISION
and its message, not about the fetch mechanism -- that mechanism stays in-tree
behind the refusal and keeps its own suite
(``tests/moe_offload/test_capture_desync_port.py``), because a refusal that
deleted the code under it could not be measured again.

What is pinned:

1. The refusal fires for the one configuration that reaches it (offload
   fraction < 1.0 AND the opt-in) and names B1/B2/B4 plus the evidence path.
2. Every other configuration -- and that is every default launch -- returns
   False without touching the refusal at all.
3. The development override converts the refusal into a warning, so a card
   window can still measure a candidate fix.
4. ``FusedMoE.__init__`` reaches the shipped decision function rather than a
   paraphrase of it, so reverting the gate cannot leave this file green.

Run:  python -m pytest tests/moe_offload/test_capture_regate_452.py -q
"""

import inspect
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import offload_capture_gate  # noqa: E402
from sglang.srt.layers.moe.offload_capture_gate import (  # noqa: E402
    ENV_GRAPH_REFUTED_OVERRIDE,
    REFUTATION,
    CapturableOffloadRefuted,
    refuse_capturable_offload_decode,
    resolve_graph_mode,
)


@pytest.fixture(autouse=True)
def _override_off(monkeypatch):
    """Every test states its own override value; none inherits the shell's."""
    monkeypatch.delenv(ENV_GRAPH_REFUTED_OVERRIDE, raising=False)


# --------------------------------------------------------------------------
# 1. the refusal fires, and says which claim killed the path
# --------------------------------------------------------------------------


def test_the_opt_in_refuses_by_name():
    with pytest.raises(CapturableOffloadRefuted) as excinfo:
        resolve_graph_mode(0.42, True, layer_id=7)
    msg = str(excinfo.value)
    assert "layer 7" in msg
    # The two measured failures, not a generic "unsupported".
    assert "6.60x" in msg
    assert "984.4" in msg and "149.1" in msg
    assert "diverging at character 5 of 533" in msg
    # And where to check the claim.
    assert REFUTATION["evidence"] in msg
    assert "2026-08-02_desync_graph_proof" in msg
    assert "NOTE_452_desync_boot_refutation.md" in msg
    # And the way out that works today.
    assert "--disable-cuda-graph" in msg


def test_the_refusal_is_its_own_exception_type():
    """A named type, so a caller can distinguish "refuted" from "misconfigured".

    ``OffloadCaptureBreach`` (the #394 replay-boundary gate) is a different
    failure with a different remedy; both are ``RuntimeError`` subclasses and
    an ``except RuntimeError`` that conflated them would report the wrong one.
    """
    assert issubclass(CapturableOffloadRefuted, RuntimeError)
    assert not issubclass(
        CapturableOffloadRefuted, offload_capture_gate.OffloadCaptureBreach
    )
    with pytest.raises(CapturableOffloadRefuted):
        refuse_capturable_offload_decode()


# --------------------------------------------------------------------------
# 2. the default path is untouched -- the refusal is not even consulted
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fraction,opt_in",
    [
        (1.0, False),  # the default launch
        (1.0, True),  # opt-in without an offload: nothing to capture
        (0.42, False),  # the shipped V4 recipe: eager offload
        (0.0, False),  # everything offloaded, still eager
    ],
)
def test_every_other_configuration_returns_false_without_refusing(
    fraction, opt_in, monkeypatch
):
    called = []
    monkeypatch.setattr(
        offload_capture_gate,
        "refuse_capturable_offload_decode",
        lambda *a, **k: called.append(a),
    )
    assert resolve_graph_mode(fraction, opt_in) is False
    assert called == []


def test_the_refusal_cannot_fire_without_an_offload():
    """Guards the ordering inside the decision, not just its result.

    ``fraction >= 1.0`` means the experts are fully resident and the offload
    never installs, so a capture is not merely allowed -- there is no capturable
    path to refuse. A gate that tested only the opt-in would abort a launch
    that has nothing to do with expert offload at all.
    """
    assert resolve_graph_mode(1.0, True) is False


# --------------------------------------------------------------------------
# 3. the development override, for the window that measures a fix
# --------------------------------------------------------------------------


def test_the_override_warns_instead_of_refusing(monkeypatch, caplog):
    monkeypatch.setenv(ENV_GRAPH_REFUTED_OVERRIDE, "1")
    with caplog.at_level(logging.WARNING, logger=offload_capture_gate.__name__):
        assert resolve_graph_mode(0.42, True) is True
    text = caplog.text
    assert "REFUTED" in text
    assert "6.60x" in text
    assert REFUTATION["evidence"] in text


@pytest.mark.parametrize("raw", ["0", "false", "off", "no", ""])
def test_a_falsey_override_still_refuses(raw, monkeypatch):
    monkeypatch.setenv(ENV_GRAPH_REFUTED_OVERRIDE, raw)
    with pytest.raises(CapturableOffloadRefuted):
        resolve_graph_mode(0.42, True)


# --------------------------------------------------------------------------
# 4. the layer reaches the shipped decision, not a copy of it
# --------------------------------------------------------------------------


def test_the_layer_selects_the_mode_through_this_gate():
    """The can-fail hook.

    Reverting the re-gate means putting the ``fraction < 1.0 and opt_in``
    expression back inline in ``FusedMoE.__init__``. That leaves tests 1-3
    green -- they exercise the gate module directly -- and only this test goes
    red, which is the point of having it.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    src = inspect.getsource(FusedMoE.__init__)
    # #462 widened the gate to three outcomes, so the layer now calls
    # resolve_offload_graph_mode() and derives the capturable boolean from it.
    # The property this pins is unchanged: the mode is decided in the gate
    # module, never inline in the layer.
    assert "resolve_offload_graph_mode(" in src
    assert "self._moe_offload_mode = resolve_offload_graph_mode(" in src
    assert "self._moe_offload_graph_mode = self._moe_offload_mode == MODE_CAPTURABLE" in src
    # ...and the inline expression it replaced is gone, so a half-revert that
    # leaves both in place (the #444 lesson: a revert must be complete) is red
    # too.
    assert "envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()\n        )" not in src


def test_the_refutation_facts_have_one_source():
    """The boot message, the docs and these tests quote ``REFUTATION``.

    Three hand-written paraphrases of the same measurement drift apart; the
    first one to drift is the one an operator reads at 3am.
    """
    assert set(REFUTATION) == {"evidence", "b1", "b2", "b4"}
    msg = str(
        pytest.raises(CapturableOffloadRefuted, refuse_capturable_offload_decode).value
    )
    for key in ("b1", "b2", "b4"):
        assert REFUTATION[key] in msg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

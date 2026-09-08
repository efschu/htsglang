# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#1244: the gapped forward's TWO ENDS, and what happens when one is loose.

WHAT THIS FALSIFIES. A pipeline's ends are fixed by RANK -- the token embedding
on the first stage, the final norm and the head on the last -- while a layer SET
fixes ownership by LAYER. Under contiguous ownership those two always name the
same stage, so nothing ever had to say it. A gapped set can separate them, and
when it does the final layer's output is computed and dropped (no crossing
follows the last layer: `crossing_schedule` iterates `range(num_layers - 1)`)
while the last stage normalises a MID-STACK activation and samples from it.

Everything here is CPU-only, hermetic, and drives the REAL modules --
`crossing_schedule`, `CrossingWire`, `parse_pp_layer_sets` -- through an
in-process link. No checkpoint, no process group, no device. The toy model
carries the two properties that make this failure invisible on inspection:

  * the two-tensor `(hidden, residual)` contract every Qwen3.5-class decoder
    layer returns, so a fix that carries only one of the pair is caught;
  * a GDN-like STATEFUL layer whose output depends on a per-layer recurrent
    carry, so a stage that silently skips a peer's layers still produces
    plausible, finite, fluent numbers rather than an obvious NaN.
"""

from __future__ import annotations

import os
import unittest
from typing import Dict, FrozenSet, List, Sequence, Tuple

import torch

from sglang.srt.distributed.pp_crossing_schedule import crossing_schedule
from sglang.srt.distributed.pp_crossing_wire import build_crossing_wire
from sglang.srt.distributed.utils import (
    PP_LAYER_SET_ENV,
    PPLayerSetError,
    parse_pp_layer_sets,
)

HIDDEN = 8


# -- the toy model ------------------------------------------------------


class ToyLayer:
    """One decoder layer: the `(hidden, residual)` contract, plus GDN-like state.

    Deterministic and cheap, but NOT commutative and NOT idempotent -- skipping
    a layer, or applying two in the wrong order, changes the output without
    changing its shape, dtype or finiteness. That is the property the assertion
    below needs: a wrong forward here must look exactly as healthy as a right
    one.
    """

    def __init__(self, layer_id: int, stateful: bool) -> None:
        self.layer_id = layer_id
        self.stateful = stateful
        gen = torch.Generator().manual_seed(735_000_001 + layer_id)
        self.w = torch.randn(HIDDEN, HIDDEN, generator=gen, dtype=torch.float32)
        self.state = torch.zeros(HIDDEN, dtype=torch.float32)

    def reset(self) -> None:
        self.state = torch.zeros(HIDDEN, dtype=torch.float32)

    def __call__(self, hidden, residual):
        # The first layer of the model receives residual=None and seeds it,
        # exactly as GemmaRMSNorm's fused-add path does.
        if residual is None:
            residual = hidden
            hidden = torch.tanh(hidden)
        else:
            residual = residual + hidden
            hidden = torch.tanh(residual)
        hidden = hidden @ self.w
        if self.stateful:
            # A recurrent carry: the layer's own output depends on what it saw
            # on the previous token/chunk, so a dropped layer perturbs every
            # later pass too rather than only this one.
            self.state = 0.5 * self.state + hidden.mean(dim=0)
            hidden = hidden + self.state
        return hidden, residual


def make_model(num_layers: int, attention_every: int) -> List[ToyLayer]:
    """`attention_every`-th layer is 'attention', the rest are GDN-like."""
    return [
        ToyLayer(i, stateful=(i % attention_every != attention_every - 1))
        for i in range(num_layers)
    ]


def run_contiguous(layers: Sequence[ToyLayer], x: torch.Tensor) -> torch.Tensor:
    """The reference: every layer in order on one device, head at the end."""
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(hidden, residual)
    return hidden + residual  # stands in for `norm(hidden, residual)`


# -- the in-process link ------------------------------------------------


class DictLink:
    """`send`/`recv` over an in-process mailbox, keyed like the real channel.

    The real `PpGroupLink` matches a receive to a send by `(src, kind)` and lets
    per-peer wire ORDER do the rest; `slot` is accepted and unused there. This
    double keeps exactly that contract -- a FIFO per ordered pair -- so a
    schedule that would mispair on metal mispairs here too, and one that would
    not, does not.
    """

    def __init__(self) -> None:
        self.mailbox: Dict[Tuple[int, int], List[dict]] = {}
        self.me = 0

    def send(self, dst: int, slot: int, payload, timeout_s: float) -> None:
        self.mailbox.setdefault((self.me, dst), []).append(
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in payload.items()}
        )

    def recv(self, src: int, slot: int, timeout_s: float):
        queue = self.mailbox.get((src, self.me))
        if not queue:
            raise AssertionError(
                f"no crossing queued from rank {src} to rank {self.me}"
            )
        return queue.pop(0)


def run_gapped(
    layers: Sequence[ToyLayer],
    owned: Sequence[FrozenSet[int]],
    x: torch.Tensor,
) -> torch.Tensor:
    """Drive the SAME layer objects through the real wire, stage by stage.

    A faithful single-threaded rendering of `qwen3_5.py`'s loop and of the
    scheduler's gapped protocol: every rank enters together (the stage-boundary
    proxy is suppressed under a gapped set), each rank walks ONLY the layers it
    owns in ascending order, and the head is applied by the LAST RANK to
    whatever that rank is holding when its own loop ends -- which is the whole
    point of the test.

    Ranks are stepped in layer order rather than concurrently, which is exactly
    what the per-iteration lockstep barrier buys on metal: one pass in flight,
    every stage inside it.
    """
    num_layers = len(layers)
    pp_size = len(owned)
    link = DictLink()
    wires = [
        build_crossing_wire(list(owned), num_layers, r, link) for r in range(pp_size)
    ]
    owner = {layer: r for r, s in enumerate(owned) for layer in s}

    # Per-rank carried activations. Only rank 0 has the embedding; every other
    # rank's entry arrives over the wire (`provides_entry_activations`).
    carried: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
        r: (x if r == 0 else None, None) for r in range(pp_size)
    }

    for layer_id in range(num_layers):
        rank = owner[layer_id]
        link.me = rank
        hidden, residual = carried[rank]
        hidden, residual = wires[rank].before_layer(layer_id, hidden, residual)
        hidden, residual = layers[layer_id](hidden, residual)
        wires[rank].after_layer(layer_id, hidden, residual)
        carried[rank] = (hidden, residual)

    # THE HEAD IS THE LAST RANK'S, not the last layer's. `qwen3_5.py:1642`.
    hidden, residual = carried[pp_size - 1]
    return hidden + residual


# -- the tests ----------------------------------------------------------


class GappedForwardTwoEnds(unittest.TestCase):
    """64 layers, 3 stages, the family shape: GDN on one card, attention split."""

    NUM_LAYERS = 64
    INTERVAL = 4  # layers 3, 7, ..., 63 are 'attention'

    def setUp(self) -> None:
        torch.manual_seed(735_000_001)
        self.x = torch.randn(5, HIDDEN, dtype=torch.float32)
        self._env = os.environ.get(PP_LAYER_SET_ENV)

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop(PP_LAYER_SET_ENV, None)
        else:
            os.environ[PP_LAYER_SET_ENV] = self._env

    def _fresh(self):
        return make_model(self.NUM_LAYERS, self.INTERVAL)

    def _attn(self) -> List[int]:
        return [
            i for i in range(self.NUM_LAYERS) if i % self.INTERVAL == self.INTERVAL - 1
        ]

    def _linear(self) -> List[int]:
        return [
            i for i in range(self.NUM_LAYERS) if i % self.INTERVAL != self.INTERVAL - 1
        ]

    # -- the map the rig actually asked for: 48 GDN + 8/8 attention ----

    def _user_map(self) -> List[FrozenSet[int]]:
        attn = self._attn()
        return [frozenset(self._linear()), frozenset(attn[:8]), frozenset(attn[8:])]

    def test_the_wire_itself_is_exact_on_the_users_map(self):
        """CONTROL. The 48/8/8 map is bit-equal to the contiguous forward.

        This is the half that must stay GREEN through any fix: it says the
        crossing wire carries BOTH tensors of the pair, at the right loop
        boundary, in the right order. It is also the desk counterpart of the
        2026-08-18 22:16-22:43Z metal probe (docs/dev/NOTE_753_gapped_probe.md),
        which answered the determined-answer prompt correctly on this same map.
        """
        owned = self._user_map()
        self.assertEqual(len(crossing_schedule(list(owned), self.NUM_LAYERS)), 31)

        ref = run_contiguous(self._fresh(), self.x)
        got = run_gapped(self._fresh(), owned, self.x)
        self.assertTrue(
            torch.equal(ref, got),
            f"the 48/8/8 gapped forward diverged from the contiguous one; "
            f"max|delta| = {(ref - got).abs().max().item():.6e}",
        )

    # -- the defect ----------------------------------------------------

    def _terminal_off_last_stage(self) -> List[FrozenSet[int]]:
        """A map whose FINAL layer sits on stage 1, not on the last stage.

        Every property the parser checked before #1244 holds: full cover, no
        duplicate, in range, and the crossing schedule builds and routes.
        """
        attn = self._attn()
        return [
            frozenset(self._linear()),
            frozenset(attn[8:]),  # holds layer 63 -- and is NOT the last stage
            frozenset(attn[:8]),
        ]

    def test_the_damage_is_silent_and_numeric(self):
        """The head samples a mid-stack activation. No error, finite, wrong.

        This is the falsifier's evidence half rather than its gate half: it
        shows WHAT the missing invariant costs, and it holds before and after
        the fix because it drives the wire directly rather than the parser.
        """
        owned = self._terminal_off_last_stage()
        ref = run_contiguous(self._fresh(), self.x)
        got = run_gapped(self._fresh(), owned, self.x)

        self.assertEqual(ref.shape, got.shape, "the damage must not be a shape error")
        self.assertTrue(torch.isfinite(got).all(), "the damage must not be a NaN")
        self.assertFalse(
            torch.equal(ref, got),
            "a map whose final layer is off the last stage cannot produce the "
            "contiguous answer -- if it does, this test no longer measures the "
            "hazard it was written for",
        )

    def test_the_parser_refuses_a_loose_terminal_end(self):
        """RED before #1244: the parser accepted this map without a word."""
        owned = self._terminal_off_last_stage()
        raw = ";".join(",".join(str(i) for i in sorted(s)) for s in owned)
        with self.assertRaises(PPLayerSetError) as ctx:
            parse_pp_layer_sets(raw, self.NUM_LAYERS, 3, allow_gapped=True)
        message = str(ctx.exception)
        self.assertIn("FINAL layer 63", message)
        self.assertIn("last stage (2)", message)

    def test_the_parser_refuses_a_loose_entry_end(self):
        """The mirror case: layer 0 off stage 0.

        It raised before #1244 too -- but on three ranks, inside the model's
        forward, after the weights had loaded. Refusing it in the parser is the
        same invariant read at the only place ownership is derived.
        """
        attn = self._attn()
        linear = self._linear()
        owned = [
            frozenset(attn[:8]),
            frozenset(linear),  # holds layer 0 -- and is NOT the first stage
            frozenset(attn[8:]),
        ]
        raw = ";".join(",".join(str(i) for i in sorted(s)) for s in owned)
        with self.assertRaises(PPLayerSetError) as ctx:
            parse_pp_layer_sets(raw, self.NUM_LAYERS, 3, allow_gapped=True)
        self.assertIn("layer 0 is owned by stage 1", str(ctx.exception))

    def test_the_users_map_still_parses(self):
        """The invariant must not cost the layout #1244 exists to enable."""
        owned = self._user_map()
        raw = ";".join(",".join(str(i) for i in sorted(s)) for s in owned)
        parsed = parse_pp_layer_sets(raw, self.NUM_LAYERS, 3, allow_gapped=True)
        self.assertEqual([set(s) for s in parsed], [set(s) for s in owned])

    def test_the_contiguous_default_is_untouched(self):
        """An ordinary ascending contiguous set-form map still parses."""
        raw = "0-21;22-42;43-63"
        parsed = parse_pp_layer_sets(raw, self.NUM_LAYERS, 3)
        self.assertEqual(min(parsed[0]), 0)
        self.assertEqual(max(parsed[2]), 63)


if __name__ == "__main__":
    unittest.main()

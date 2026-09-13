# SPDX-License-Identifier: Apache-2.0
"""#1332 B1b -- the PRODUCER for ``widest_layer_bytes``, measured from the checkpoint.

THE OPEN DECIDE THIS CLOSES, in seat 3's own words (WEG2_REUSE_SPEC_0908 §10.8,
"OPEN DECIDES for the next seat", item 1): *"``widest_layer_bytes`` has no
producer yet. It must be MEASURED (per-tag, per-layer device bytes) and not
derived from the mean -- the whole refusal rests on it."*

``bounce_terms`` already refuses to be called without it
(``widest_layer_bytes must be positive ... Absent is NOT 'use the mean'``), and
``under_coverage_refusal`` grades the assemble buffer against it. So the number
has to come from somewhere that MEASURES, once per launch, before either group
starts -- and the only thing available at that moment that knows every
parameter's true extent is the CHECKPOINT ITSELF.

THE INSTRUMENT: the safetensors HEADER of every shard. A header carries
``dtype``, ``shape`` and ``data_offsets`` per tensor, so the on-disk byte count
is ``data_offsets[1] - data_offsets[0]`` -- exact, and read without touching a
single byte of tensor data, without torch, without a GPU and without a model
load. The read side of this already existed as a desk tool
(``weg2/tools/class_byte_census.py``, written for boot weg2xsn8's sizing
addendum); B1b turns it into a product producer with a refusal.

GROUP-WIDE, PER LAYER, MAX -- three decisions, all from §10.2 and the user law
of 2026-09-11 ("das Layer wird VOLLSTAENDIG in einem kleinen Host-Puffer
zusammengesetzt, jede Karte nimmt ihren Teil"):
* GROUP-WIDE because the buffer holds ONE COMPLETE layer that all three cards
  then slice, not one card's share of it;
* PER LAYER because that is the assembly unit;
* MAX and never MEAN, because a buffer sized on the mean dies on the first
  layer above it -- the danger direction of the whole slice, and the mutant
  this file runs.

RED ON `140825d754`: the module does not exist (collection error), which is the
same red-first shape seat 3 used for `xchg_bounce.py` itself.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import checkpoint_census as cc  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402


def _write_shard(path, tensors):
    """A REAL safetensors file: 8-byte header length, JSON header, then data.

    Written rather than mocked because the producer's whole claim is that it
    reads THE CHECKPOINT's own headers; a monkeypatched reader would prove
    nothing about the format, and the format is where an off-by-one lives.
    """
    header, off = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * off)


#: Three layers of DELIBERATELY DIFFERENT size, so max and mean are far apart
#: and a mean-sized bound is measurably wrong: 1 MiB, 4 MiB, 2 MiB.
L0, L1, L2 = 1 << 20, 1 << 22, 1 << 21
#: Unlayered bytes, which must NOT land in any layer's total.
EMBED = 1 << 23


@pytest.fixture()
def ckpt(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    _write_shard(d / "model-00001-of-00002.safetensors", {
        "model.layers.0.mlp.gate_proj.weight": ("I8", (1024, 1024), L0 // 2),
        "model.layers.0.mlp.up_proj.weight": ("I8", (1024, 1024), L0 // 2),
        "model.layers.1.mlp.gate_proj.weight": ("I8", (2048, 2048), L1 - 4096),
        "model.layers.1.input_layernorm.weight": ("BF16", (2048,), 4096),
    })
    _write_shard(d / "model-00002-of-00002.safetensors", {
        "model.layers.2.self_attn.qkv_proj.weight": ("I8", (2048, 1024), L2),
        "model.embed_tokens.weight": ("BF16", (4096, 1024), EMBED),
    })
    return d


# ===========================================================================
# THE MEASUREMENT.
# ===========================================================================


def test_the_census_reads_bytes_per_layer_from_the_headers(ckpt):
    """Exact on-disk bytes per layer, group-wide, from `data_offsets`."""
    census = cc.layer_census_from_headers(str(ckpt))
    assert census.n_layers == 3, census.n_layers
    assert dict(census.layer_bytes) == {0: L0, 1: L1, 2: L2}, census.layer_bytes
    assert census.files == 2
    # The unlayered bytes are counted, separately, and never inside a layer.
    assert census.unlayered_bytes == EMBED
    assert census.layer_total_bytes == L0 + L1 + L2


def test_the_widest_layer_is_the_MAX_and_never_the_mean(ckpt):
    """THE MUTANT DIRECTION, and the reason the producer exists at all.

    mean = (1 + 4 + 2) / 3 MiB = 2.33 MiB, max = 4 MiB. A producer returning
    the mean is off by 1.71 MiB on a three-layer toy and by whatever the real
    checkpoint's spread is on metal -- and `bounce_terms` would then build a
    buffer that cannot hold layer 1 while `covers_widest_layer` still said yes,
    because it would be grading the bound against the same wrong number.
    """
    census = cc.layer_census_from_headers(str(ckpt))
    layer, nbytes, classes = cc.widest_layer(census)
    assert layer == 1, layer
    assert nbytes == L1, nbytes
    mean = (L0 + L1 + L2) // 3
    assert nbytes > mean, (nbytes, mean)
    # The classes of THAT layer, so a reader can see what the buffer holds.
    assert set(classes) == {"gate_proj", "input_layernorm"}, classes


def test_the_widest_line_carries_the_layer_its_bytes_and_its_classes(ckpt):
    """The line a boot record greps."""
    census = cc.layer_census_from_headers(str(ckpt))
    line = cc.widest_line(census)
    assert line.startswith("WEG2-XCHG WIDEST "), line
    assert "layer=1 " in line, line
    assert f"bytes={L1} " in line, line
    assert "classes=gate_proj,input_layernorm" in line, line
    # The denominators belong on the line too: how many layers were measured
    # and what the mean is, so nobody has to recompute them to read the bound.
    assert "layers=3 " in line, line
    assert f"mean_bytes={(L0 + L1 + L2) // 3} " in line, line


def test_an_unreadable_checkpoint_REFUSES_BY_NAME_and_never_defaults(tmp_path):
    """Absent is not 'use a default' -- the refusal is the deliverable.

    Three shapes, all named: a directory with no shards, a shard whose header
    is truncated, and a checkpoint with no layer-indexed tensor at all (which
    would otherwise produce `n_layers=0` and a division by zero one call later).
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cc.Weg2XchgWidestLayerUnreadable) as e1:
        cc.layer_census_from_headers(str(empty))
    assert "no safetensors" in str(e1.value).lower(), str(e1.value)

    trunc = tmp_path / "trunc"
    trunc.mkdir()
    (trunc / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", 4096) + b"{")
    with pytest.raises(cc.Weg2XchgWidestLayerUnreadable) as e2:
        cc.layer_census_from_headers(str(trunc))
    assert "header" in str(e2.value).lower(), str(e2.value)

    flat = tmp_path / "flat"
    flat.mkdir()
    _write_shard(flat / "model-00001-of-00001.safetensors",
                 {"model.embed_tokens.weight": ("BF16", (16, 16), 512)})
    with pytest.raises(cc.Weg2XchgWidestLayerUnreadable) as e3:
        cc.layer_census_from_headers(str(flat))
    assert "no layer" in str(e3.value).lower(), str(e3.value)
    # Every refusal names the path it read, or the next reader cannot check it.
    for exc in (e1, e2, e3):
        assert str(tmp_path) in str(exc.value)


def test_the_census_feeds_bounce_terms_and_the_bound_grades_against_the_max(ckpt):
    """The seam: B1's sizing function, fed by B1b's producer.

    `bounce_terms` sizes `buffer_bytes = widest x depth` and
    `covers_widest_layer` compares one depth-slot against the widest layer. Fed
    the MEASURED max, a depth-2 buffer covers it; fed the mean, the same buffer
    is under-covered -- which is the refusal working and is asserted here so
    the two halves cannot drift apart silently.
    """
    census = cc.layer_census_from_headers(str(ckpt))
    _layer, widest, _classes = cc.widest_layer(census)
    terms = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers, widest_layer_bytes=widest,
        pairs=3, depth=2)
    assert terms.buffer_bytes == widest * 2
    assert terms.covers_widest_layer is True
    assert terms.mean_layer_bytes < widest
    # THE MUTANT'S SHAPE, stated as the grading actually works: a buffer sized
    # on the mean is graded per DEPTH-SLOT, and one slot of it cannot hold the
    # real widest layer.  (Its total is not the test: 2 slots of the mean
    # happen to exceed one widest layer on this toy, which is exactly how a
    # mean-sized buffer can look sufficient while the assembly of a single
    # layer does not fit.)
    mean_terms = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers,
        widest_layer_bytes=terms.mean_layer_bytes, pairs=3, depth=2)
    per_slot = mean_terms.buffer_bytes // mean_terms.depth
    assert per_slot < widest, (per_slot, widest)
    # And the refusal text names both numbers, so a reader cannot mistake one
    # for the other.
    msg = xb.under_coverage_refusal(mean_terms, widest_layer_name="layer 1")
    assert "W71" in msg and "layer 1" in msg


# ===========================================================================
# THE WIRING. A producer nothing calls is the desk-written-never-executed law's
# own example, so the launcher's arm predicate is driven here directly.
# ===========================================================================


def _launcher():
    from sglang.srt.weg2 import launcher

    return launcher


def test_the_arms_that_pin_nothing_charge_nothing_and_print_nothing(ckpt):
    """`ring` and `ipc` must be byte-identical to before this slice.

    The bounce FILE exists only on the host arm; charging elsewhere is the
    mirror of the omission the term was added to close (S6 refuter, must_fix 4),
    and printing a WIDEST line on an arm that assembles nothing would put a
    number in a boot record that nothing spends.
    """
    lz = _launcher()
    assert lz.xchg_bounce_terms_for_arm("ring", "host", str(ckpt)) == (0, [])
    assert lz.xchg_bounce_terms_for_arm("shadow", "ipc", str(ckpt)) == (0, [])
    assert lz.xchg_bounce_terms_for_arm("exchange", "ipc", str(ckpt)) == (0, [])


def test_the_host_arm_charges_the_MEASURED_bounce_and_prints_both_lines(ckpt):
    """The charge is the expression's own total, and the lines carry it."""
    lz = _launcher()
    charged, lines = lz.xchg_bounce_terms_for_arm("shadow", "host", str(ckpt))
    census = cc.layer_census_from_headers(str(ckpt))
    _idx, widest, _cls = cc.widest_layer(census)
    expected = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers, widest_layer_bytes=widest,
        pairs=3, depth=xb.ASSEMBLE_DEPTH_DEFAULT)
    assert charged == expected.total_bytes, (charged, expected.total_bytes)
    assert charged > 0, "the whole point of the arm change is a priced bounce"
    assert len(lines) == 3, lines
    assert lines[0].startswith(cc.WIDEST_LINE_PREFIX), lines[0]
    assert lines[1].startswith("WEG2-XCHG-BOUNCE "), lines[1]
    # THE PROVENANCE OF THE DEPTH-SLOT, so a boot log says where the size came
    # from instead of leaving a later reader to assume a default. The size
    # itself is unchanged -- this asserts only that its ORIGIN is printed.
    assert lines[2].startswith("WEG2-XCHG DEPTH-SLOT "), lines[2]
    assert "source=census" in lines[2], lines[2]
    assert f"bytes={expected.widest_layer_bytes}" in lines[2], lines[2]
    # The ARM line's total and the charge are ONE number, not two readings.
    assert f"bounce_total_mib={expected.total_bytes // xb.MIB}" in lines[1]


def test_the_host_arm_with_no_checkpoint_REFUSES_and_never_charges_a_default():
    """An arm that pins bytes without a checkpoint to measure is refused."""
    lz = _launcher()
    with pytest.raises(Exception) as e:
        lz.xchg_bounce_terms_for_arm("shadow", "host", "")
    assert "W14" in str(e.value), str(e.value)  # W74=SourceMissing, W88=scheduler.py:6314
    with pytest.raises(cc.Weg2XchgWidestLayerUnreadable):
        lz.xchg_bounce_terms_for_arm("shadow", "host", "/nonexistent/ckpt")


def test_the_under_coverage_GUARD_is_present_and_currently_unreachable(ckpt):
    """HONEST SCOPE, so nobody reads a green test as coverage it is not.

    `bounce_terms` sizes the buffer FROM the widest layer
    (`buffer_bytes = widest x depth`), so one depth-slot equals the widest
    layer exactly and `covers_widest_layer` is True BY CONSTRUCTION on every
    input this call site can produce. The launcher's W71 branch is therefore a
    RATCHET, not a live path: it starts firing the moment a CAP is introduced
    (step 3 raises the slot to 64 MiB and step 5 may bound the buffer), which
    is precisely when an under-covered buffer becomes possible. Asserting the
    guard exists AND that it does not fire today is the only claim the code
    supports.
    """
    lz = _launcher()
    charged, _lines = lz.xchg_bounce_terms_for_arm("shadow", "host", str(ckpt))
    assert charged > 0
    src = __import__("inspect").getsource(lz.xchg_bounce_terms_for_arm)
    assert "covers_widest_layer" in src and "under_coverage_refusal" in src
    # And the refusal it would raise is the W71 form, proven on hand-built
    # terms whose buffer is deliberately one slot too small.
    census = cc.layer_census_from_headers(str(ckpt))
    _i, widest, _c = cc.widest_layer(census)
    capped = xb.bounce_terms(
        bytes_per_direction=census.layer_total_bytes,
        n_layers=census.n_layers, widest_layer_bytes=widest // 4,
        pairs=3, depth=2)
    assert (capped.buffer_bytes // capped.depth) < widest
    assert "W71" in xb.under_coverage_refusal(capped)

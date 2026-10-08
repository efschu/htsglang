# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-c: a P stage rebuilds a held END anchor from the D ranks' head
shares (canonical blob round trip on a small GDN spec)."""

from __future__ import annotations

import torch

from flliper.srt.mem_cache.hicache_migrate import MambaBlobSpec
from flliper.srt.pdflip.l15_anchor_plan import rank_ranges
from flliper.srt.pdflip.l15_share_take import L15TakeError, take_anchor

SPEC = MambaBlobSpec(num_layers=2, num_heads=4, head_dim=1, state_size=2,
                     conv_dim=8, conv_width=1, key_dim=2, value_dim=4, units=2,
                     temporal_itemsize=1, conv_itemsize=1)
RATIOS = [1, 1]
SLOTS = 4
A = 2                      # D's held anchor row (same on every rank)


def _canon():
    return torch.arange(SPEC.total_bytes, dtype=torch.uint8) + 1


def _d_shares(canon):
    reg, shares, nfd = {}, {}, 0
    for r in range(len(RATIOS)):
        rs = SPEC.shard_for_rank(RATIOS, r)
        blob = torch.cat([canon[o:o + n] for o, n in rank_ranges(SPEC, RATIOS, r)])
        tl, cl = rs.temporal_layer_bytes, rs.conv_layer_bytes
        temporal = torch.zeros(SPEC.num_layers, SLOTS, tl, dtype=torch.uint8)
        conv = torch.zeros(SPEC.num_layers, SLOTS, cl, dtype=torch.uint8)
        for l in range(SPEC.num_layers):
            temporal[l, A] = blob[l * tl:(l + 1) * tl]
            base = SPEC.num_layers * tl
            conv[l, A] = blob[base + l * cl: base + (l + 1) * cl]
        bases, fds = [], []
        for name, t in (("mamba_temporal", temporal), ("mamba_conv0", conv)):
            reg[nfd] = t.reshape(-1)
            for l in range(SPEC.num_layers):
                bases.append({"role": name, "layer": l,
                              "view_off": l * t.stride(0), "unit": t.stride(1),
                              "extents": [[0, t.numel()]]})
                fds.append(nfd)
            nfd += 1
        # one fd per extent entry, in base order (same registry tensor)
        shares[r] = ({"spans": [{"rid": "r1", "anchor_slot": A, "slots": []}],
                      "bases": bases}, fds)
    return shares, reg


def test_stage_slot_gets_the_full_heads_canonical_state():
    canon = _canon()
    shares, reg = _d_shares(canon)
    pt = torch.zeros(SPEC.num_layers, SLOTS, SPEC.temporal_layer_bytes, dtype=torch.uint8)
    pc = torch.zeros(SPEC.num_layers, SLOTS, SPEC.conv_layer_bytes, dtype=torch.uint8)
    n = take_anchor(shares, rid="r1", spec=SPEC, ratios=RATIOS, stage=(0, 2),
                    p_temporal=pt, p_conv=pc, p_slot=1,
                    map_extent=lambda fd, size: reg[fd])
    assert n == SPEC.total_bytes
    tl, cl = SPEC.temporal_layer_bytes, SPEC.conv_layer_bytes
    for l in range(SPEC.num_layers):
        assert torch.equal(pt[l, 1], canon[l * tl:(l + 1) * tl])
        base = SPEC.num_layers * tl
        assert torch.equal(pc[l, 1], canon[base + l * cl: base + (l + 1) * cl])
    assert int(pt[:, 0].sum()) == 0          # other slots untouched


def test_missing_anchor_refuses():
    shares, reg = _d_shares(_canon())
    for r in shares:
        shares[r][0]["spans"][0]["anchor_slot"] = -1
    pt = torch.zeros(2, SLOTS, 8, dtype=torch.uint8)
    pc = torch.zeros(2, SLOTS, 8, dtype=torch.uint8)
    try:
        take_anchor(shares, rid="r1", spec=SPEC, ratios=RATIOS, stage=(0, 2),
                    p_temporal=pt, p_conv=pc, p_slot=1,
                    map_extent=lambda fd, size: reg[fd])
    except L15TakeError as exc:
        assert "anchor" in str(exc)
    else:
        raise AssertionError("a missing anchor must refuse")

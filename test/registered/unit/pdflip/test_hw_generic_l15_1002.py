# SPDX-License-Identifier: Apache-2.0
"""HW-GENERIC 1002 (L15): the L1.5 caps keyed by CARD IDENTITY.

User order 02.10.: the pdflip release must run on any sm_86/sm_120 card set.
``FLLIPER_PDFLIP_L15_MIB="c1=7616,c2=1792"`` names a card by its budget ORDINAL
(card order), a position that moves with the inventory. Pinned here:

* identity keys (``GPU-<uuid>=``, ``pci:<bus id>=``, ``nvml<i>=``) resolve
  ONCE in the launcher to the canonical ordinal form the ranks read, so the
  rank path (``l15_shadow.caps_from_env``) is unchanged;
* the legacy ``c1=7616,c2=1792`` gives this rig exactly today's posts/caps
  and is handed on byte-identical;
* a key naming no card, two keys naming one card, mixing ordinal and
  identity keys, or a malformed key is a NAMED refusal (W-L15-CARDKEY);
* the NOCAP refusal names cards by identity, not "c0 = the 5090";
* 0, 1 and 2 cap-0 ranks through caps_from_env / select_hold / retain, and
  the launcher's ``L15-CAP0`` warning when the count is not exactly one.

Hermetic: no GPU, no NVML, no boot. Plain pytest functions.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os

import pytest

from flliper.srt.pdflip import l15_plan, l15_policy, l15_retain, l15_shadow
from flliper.srt.pdflip import launcher as L

_HERE = os.path.dirname(__file__)

# the reference rig (test_pdflip_l15_plan_wiring_0930.py fixture)
U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
PCI = {U0: "00000000:01:00.0", U5090: "00000000:02:00.0", U2: "00000000:03:00.0"}
MASTER = {"FLLIPER_PDFLIP_L15": "1"}
LEGACY = "c1=7616,c2=1792"


def _rig():
    """Card order: [5090 (nvml1), 3080 (nvml0), 3080 (nvml2)]."""
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U0, name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425, cc=(8, 6)),
        L.Card(nvml_index=1, uuid=U5090, name="NVIDIA GeForce RTX 5090",
               total_mib=32607, reserved_mib=518, cc=(12, 0)),
        L.Card(nvml_index=2, uuid=U2, name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425, cc=(8, 6)),
    ])


def _other_rig():
    """Another inventory: nvml0 = RTX A6000 (49140), nvml1/2 = RTX 3090."""
    return L.order_cards([
        L.Card(nvml_index=0, uuid="GPU-aaaa", name="NVIDIA RTX A6000",
               total_mib=49140, cc=(8, 6)),
        L.Card(nvml_index=1, uuid="GPU-bbbb", name="NVIDIA GeForce RTX 3090",
               total_mib=24576, cc=(8, 6)),
        L.Card(nvml_index=2, uuid="GPU-cccc", name="NVIDIA GeForce RTX 3090",
               total_mib=24576, cc=(8, 6)),
    ])


def _env(mib):
    return dict(MASTER, FLLIPER_PDFLIP_L15_MIB=mib)


# ---------------------------------------------------------------------------
# legacy: byte-identical on this rig


def test_rig_order_is_the_reference_order():
    assert [c.uuid for c in _rig()] == [U5090, U0, U2]


def test_legacy_value_is_handed_on_unchanged():
    res = l15_plan.resolve_l15_mib(LEGACY, _rig())
    assert res.value == LEGACY and res.rewritten is False
    assert [(k, o, m) for k, o, m in res.rows] == [("c1", 1, 7616), ("c2", 2, 1792)]


def test_legacy_posts_and_caps_are_todays():
    budgets, peaks = [20000, 9000, 4000], [None, None, None]
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, budgets, peaks, _env(LEGACY))
    assert [(p.card, p.mib, p.src) for p in posts] == [
        (0, 0, "OVERRIDE-UNNAMED"), (1, 7616, "OVERRIDE"), (2, 1792, "OVERRIDE")]
    cell = 4096
    assert l15_shadow.caps_from_env(_env(LEGACY), 3, [cell] * 3, [0, 1, 2]) == (
        0, 7616 * 2**20 // cell, 1792 * 2**20 // cell)


def test_auto_and_absent_are_untouched():
    for v in (None, "", "auto", "AUTO"):
        res = l15_plan.resolve_l15_mib(v, _rig())
        assert res.mode == "auto" and res.rewritten is False and res.value == v


# ---------------------------------------------------------------------------
# identity keys -> the canonical ordinal form


@pytest.mark.parametrize("value", [
    f"{U0}=7616,{U2}=1792",
    f"{U2.upper().replace('GPU-', 'gpu-')}=1792,{U0}=7616",  # order and case free
    "nvml0=7616,nvml2=1792",
    "pci:00000000:01:00.0=7616,pci:00000000:03:00.0=1792",
    "pci:0000:01:00.0=7616,pci:03:00.0=1792",
    f"pci:01:00.0=7616,{U2}=1792",  # identity kinds may mix
])
def test_identity_keys_canonicalise_to_todays_value(value):
    res = l15_plan.resolve_l15_mib(value, _rig(), PCI)
    assert res.rewritten is True
    assert res.value == LEGACY
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [20000, 9000, 4000],
                                   [None] * 3, _env(res.value))
    assert [p.mib for p in posts] == [0, 7616, 1792]


def test_pci_from_the_card_attribute():
    cards = [dict(uuid=c.uuid, nvml_index=c.nvml_index, pci_bus_id=PCI[c.uuid],
                  name=c.name, total_mib=c.total_mib) for c in _rig()]
    assert l15_plan.resolve_l15_mib("pci:02:00.0=5", cards).value == "c0=5"


def test_identity_follows_the_card_on_another_inventory():
    cards = _other_rig()
    assert [c.uuid for c in cards] == ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]
    res = l15_plan.resolve_l15_mib("GPU-cccc=3000,GPU-aaaa=100", cards)
    assert res.value == "c0=100,c2=3000"
    assert l15_shadow.caps_from_env(_env(res.value), 3, [1] * 3) == (
        100 * 2**20, 0, 3000 * 2**20)


def test_mapping_line_names_key_ordinal_and_card():
    res = l15_plan.resolve_l15_mib(f"{U0}=7616,nvml2=1792", _rig())
    line = l15_plan.mib_cards_line(res, _rig())
    assert line.startswith("L15-MIB-CARDS ")
    assert f"{U0} -> c1 -> nvml0" in line and "nvml2 -> c2 -> nvml2" in line
    assert "'c1=7616,c2=1792'" in line and "rewritten" in line


# ---------------------------------------------------------------------------
# named refusals


@pytest.mark.parametrize("value, needle", [
    (f"c1=7616,{U2}=1792", "mixed"),
    ("GPU-deadbeef=100", "'GPU-deadbeef' matches no card"),
    ("nvml7=100", "'nvml7' matches no card"),
    ("pci:09:00.0=100", "'pci:09:00.0' matches no card"),
    (f"{U0}=1,nvml0=2", "both name card ordinal 1"),
    ("c1=1,c1=2", "both name card ordinal 1"),
    ("c5=100", "'c5' names budget ordinal 5"),
    ("cuda0=100", "malformed pair 'cuda0=100'"),
    ("pci:zz:00.0=100", "malformed pair"),
    (f"{U0}=7.5k", "malformed pair"),
])
def test_bad_keys_refuse_by_name(value, needle):
    with pytest.raises(ValueError) as ei:
        l15_plan.resolve_l15_mib(value, _rig(), PCI)
    msg = str(ei.value)
    assert msg.startswith(l15_plan.CARDKEY_REFUSAL_CODE), msg
    assert needle in msg, msg


def test_pci_key_without_a_known_bus_id_refuses():
    with pytest.raises(ValueError, match="no PCI bus id is known"):
        l15_plan.resolve_l15_mib("pci:01:00.0=100", _rig())


def test_rank_parser_still_refuses_identity_keys():
    # the ranks only ever see the ordinal form; an unresolved value must not
    # silently become cap 0 on a rank either
    with pytest.raises(ValueError, match="resolved by the launcher"):
        l15_plan.parse_l15_mib(f"{U0}=100")


# ---------------------------------------------------------------------------
# NOCAP text: by identity, never "c0 = the 5090"


def test_nocap_text_names_cards_by_identity():
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [9000] * 3, [None] * 3, MASTER)
    msg = l15_plan.refuse_no_caps(posts, MASTER, _rig())
    assert msg is not None and msg.startswith(l15_plan.NOCAP_REFUSAL_CODE)
    assert "c0 = the 5090" not in msg
    assert f"c0 = nvml1 NVIDIA GeForce RTX 5090" in msg and U5090 in msg
    assert "GPU-<uuid>=" in msg
    bare = l15_plan.refuse_no_caps(posts, MASTER)
    assert "5090" not in bare and "c1=" in bare


# ---------------------------------------------------------------------------
# cap-0 multiplicity: 0, 1, 2


@pytest.mark.parametrize("mib, cap0", [
    ("c0=10,c1=10,c2=10", ()),
    (LEGACY, (0,)),
    ("c1=7616", (0, 2)),
])
def test_caps_from_env_cap0_multiplicity(mib, cap0):
    caps = l15_shadow.caps_from_env(_env(mib), 3, [4096] * 3)
    assert tuple(r for r, c in enumerate(caps) if c == 0) == cap0
    assert l15_shadow.cap0_ranks(_env(mib), 3) == cap0


def test_cap0_line_only_when_not_exactly_one():
    assert l15_shadow.cap0_line((0,)) is None
    assert l15_shadow.cap0_line((1,)) is None
    assert l15_shadow.cap0_line(()) == (
        "L15-CAP0 ranks=- (metal-proven only for exactly one cap-0 rank)")
    assert l15_shadow.cap0_line((0, 2)) == (
        "L15-CAP0 ranks=0,2 (metal-proven only for exactly one cap-0 rank)")


def _cand(rid, rows, t):
    return l15_policy.Candidate(rid=rid, kind="seat", last_active=t,
                                rows_by_rank=rows, anchor_depth=1, kv_depth=1)


@pytest.mark.parametrize("caps, admitted", [
    ((4, 4, 4), ("a",)),       # 0 cap-0 ranks: b no longer fits on any rank
    ((0, 4, 4), ("a",)),       # 1 cap-0 rank: rank 0 never blocks
    ((0, 4, 0), ("a", "b")),   # 2 cap-0 ranks: only rank 1 is charged
])
def test_select_hold_cap0_multiplicity(caps, admitted):
    cands = [_cand("a", (3, 2, 3), 2.0), _cand("b", (3, 2, 3), 1.0)]
    hs = l15_policy.select_hold(cands, caps, 8)
    assert hs.rids == admitted
    assert hs.rows_by_rank == tuple(sum(c.rows_by_rank[r] for c in cands
                                        if c.rid in admitted) for r in range(3))


def test_select_hold_all_ranks_cap0_admits_up_to_the_anchor_cap():
    cands = [_cand("a", (3, 2, 3), 2.0), _cand("b", (3, 2, 3), 1.0)]
    hs = l15_policy.select_hold(cands, (0, 0, 0), 1)
    assert hs.rids == ("a",) and dict(hs.excluded)["b"] == "anchor_full"


def _retain(tmp_path, caps):
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_retain_0930", os.path.join(_HERE, "test_pdflip_l15_retain_0930.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sc = m.make_scenario(tmp_path, [])
    kw = dict(sc["kwargs"])
    kw["caps_rows_by_rank"] = caps
    kw["candidates"] = [c for c in kw["candidates"] if c.rid in ("r_seat", "r_parked")]
    res = l15_retain.retain_at_sleep(**kw)
    keeps = [c for c in sc["set_keep_calls"] if c[0] == "set_keep"]
    return res, keeps, sc["manifest_path"], m.RANK


@pytest.mark.parametrize("caps, this_rank_cap0", [
    ((10, 10), False),   # 0 cap-0 ranks
    ((0, 10), False),    # 1 cap-0 rank, not this one: this rank keeps
    ((0, 0), True),      # 2 cap-0 ranks: this one keeps nothing
])
def test_retain_keep_windows_follow_this_ranks_cap(tmp_path, caps, this_rank_cap0):
    res, keeps, mpath, rank = _retain(tmp_path, caps)
    assert rank == 1
    assert res is not None and keeps and os.path.exists(mpath)
    assert all(c[2] == () for c in keeps) is this_rank_cap0


# ---------------------------------------------------------------------------
# the launcher hook


def _ns(env_p="", env_d=""):
    return argparse.Namespace(env_p=env_p, env_d=env_d)


def test_launcher_rewrites_identity_keys_for_the_ranks():
    env = _env(f"{U0}=7616,{U2}=1792")
    lines = []
    L.l15_mib_by_card_identity(_ns(), _rig(), lines.append, env)
    assert env["FLLIPER_PDFLIP_L15_MIB"] == LEGACY
    assert any(x.startswith("L15-MIB-CARDS source=env ") for x in lines)
    assert not any(x.startswith("L15-CAP0") for x in lines)  # exactly one: rank 0


def test_launcher_leaves_legacy_byte_identical():
    env = _env(LEGACY)
    lines = []
    L.l15_mib_by_card_identity(_ns(), _rig(), lines.append, env)
    assert env == _env(LEGACY)
    assert len(lines) == 1 and "rewritten" not in lines[0]


def test_launcher_without_the_variable_says_nothing():
    env = dict(MASTER)
    lines = []
    L.l15_mib_by_card_identity(_ns(), _rig(), lines.append, env)
    assert env == MASTER
    # master on and no override: caps_from_env holds every rank 0
    assert lines == ["L15-CAP0 ranks=0,1,2 (metal-proven only for exactly one cap-0 rank)"]


def test_launcher_rewrites_env_d_too():
    ns = _ns(env_d=f"FLLIPER_X=1;FLLIPER_PDFLIP_L15_MIB=nvml0=7616")
    env = dict(MASTER)
    lines = []
    L.l15_mib_by_card_identity(ns, _rig(), lines.append, env)
    assert ns.env_d == "FLLIPER_X=1;FLLIPER_PDFLIP_L15_MIB=c1=7616"
    assert any("source=env_d" in x for x in lines)
    # D runs c1 only: ranks 0 and 2 are cap 0
    assert "L15-CAP0 ranks=0,2 (metal-proven only for exactly one cap-0 rank)" in lines


def test_launcher_refuses_a_bad_key_by_name():
    env = _env("GPU-deadbeef=100")
    with pytest.raises(L.PdFlipLaunchRefused, match="W-L15-CARDKEY"):
        L.l15_mib_by_card_identity(_ns(), _rig(), lambda s: None, env)


def test_launcher_master_off_prints_no_cap0_line():
    env = {"FLLIPER_PDFLIP_L15_MIB": "c1=7616"}
    lines = []
    L.l15_mib_by_card_identity(_ns(), _rig(), lines.append, env)
    assert not any(x.startswith("L15-CAP0") for x in lines)


def test_launcher_main_resolves_before_posts_and_before_build_env():
    src = inspect.getsource(L.main)
    i = src.index("l15_mib_by_card_identity(ns, cards, log)")
    assert i < src.index("l15_plan.resolve_posts(")
    assert i < src.index("env_p = build_env(")
    assert i < src.index("env_d = build_env(")
    assert "l15_plan.refuse_no_caps(l15_posts, os.environ, cards)" in src


# ---------------------------------------------------------------------------
# comments: no card named as THE cap-0 rank


@pytest.mark.parametrize("mod", [l15_policy, l15_retain])
def test_no_5090_tp0_as_the_cap0_rank(mod):
    src = inspect.getsource(mod)
    assert "5090 / TP0" not in src and "the 5090" not in src

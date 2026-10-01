# SPDX-License-Identifier: Apache-2.0
"""L15 caps come ONLY from the SGLANG_WEG2_L15_MIB override (desk proof for the
next L15=1 boot).

N3c (10012013) and N3e (10012112) booted with SGLANG_WEG2_L15=1 but WITHOUT
SGLANG_WEG2_L15_MIB: the launcher printed "L15-POST card=0/1/2 mib=0
src=UNMEASURED" (no P_AWAKE_PEAK_MIB record exists for any profile, so the
record path prices 0) and every D rank logged "L15-SHADOW ... cap=0,0,0"
(caps_from_env honours ONLY the override; "auto"/absent is cap 0 on every
rank). A retain with n>0 was structurally impossible.

The shadow boots of the same line that carried SGLANG_WEG2_L15_MIB="c1=7616,
c2=1792" (e.g. 10011251, 10011623, 10011803, 10011917) logged
cap=0,243712,57344 -- these numbers are pinned here. Card number = budget
ordinal (order_cards: c0 = the 5090 = TP0, c1/c2 = the 3080s), which is also
the rank index the D hook passes as card_of_rank (card_map=[0, 1, 2]).
"""

from sglang.srt.weg2 import l15_plan, l15_shadow

# 27B D KV cell: 7616 MiB -> 243712 rows (measured in the shadow boots above)
CELL_BYTES_27B = 7616 * 2**20 // 243712

ARMED = {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "c1=7616,c2=1792"}


def test_cell_size_is_32_kib():
    assert CELL_BYTES_27B == 32768


def test_override_gives_the_shadow_boot_caps():
    caps = l15_shadow.caps_from_env(ARMED, 3, [CELL_BYTES_27B] * 3, [0, 1, 2])
    assert caps == (0, 243712, 57344)


def test_master_without_override_is_cap_zero_everywhere():
    # the N3c/N3e arm: master on, no MIB -> nothing can ever be held
    caps = l15_shadow.caps_from_env({"SGLANG_WEG2_L15": "1"}, 3,
                                    [CELL_BYTES_27B] * 3, [0, 1, 2])
    assert caps == (0, 0, 0)
    caps_auto = l15_shadow.caps_from_env(
        {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "auto"}, 3,
        [CELL_BYTES_27B] * 3, [0, 1, 2])
    assert caps_auto == (0, 0, 0)


def test_launcher_posts_follow_the_override():
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [20000, 9000, 4000],
                                   [None, None, None], ARMED)
    assert [(p.card, p.mib, p.src) for p in posts] == [
        (0, 0, "OVERRIDE-UNNAMED"), (1, 7616, "OVERRIDE"), (2, 1792, "OVERRIDE")]
    unmeasured = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [20000, 9000, 4000],
                                        [None, None, None], {"SGLANG_WEG2_L15": "1"})
    assert [(p.mib, p.src) for p in unmeasured] == [(0, "UNMEASURED")] * 3

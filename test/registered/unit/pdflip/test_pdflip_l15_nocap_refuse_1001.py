# SPDX-License-Identifier: Apache-2.0
"""L15 fail-fast (N3f): an L1.5-armed launch whose every card holds 0 MiB is
refused BY NAME before any launch.

N3c (10012013) and N3e (10012112) ran FLLIPER_PDFLIP_L15=1 without
FLLIPER_PDFLIP_L15_MIB: "L15-POST card=0/1/2 mib=0 src=UNMEASURED" and
"L15-SHADOW ... cap=0,0,0" on every D rank -- a whole boot window spent on an
L1.5 boot that could never hold a single row, noticed only by reading the
logs. The refusal turns that into a launch error with the fix in the text.
"""

from flliper.srt.pdflip import l15_plan
from flliper.srt.pdflip.l15_plan import L15Post


def _posts(*mibs):
    return [L15Post(card=i, mib=m, src="X")
            for i, m in enumerate(mibs)]


def test_master_off_never_refuses():
    assert l15_plan.refuse_no_caps(_posts(0, 0, 0), {}) is None
    assert l15_plan.refuse_no_caps(_posts(0, 0, 0), {"FLLIPER_PDFLIP_L15": "0"}) is None


def test_master_on_all_zero_posts_refuses_by_name():
    msg = l15_plan.refuse_no_caps(_posts(0, 0, 0), {"FLLIPER_PDFLIP_L15": "1"})
    assert msg is not None
    assert msg.startswith(l15_plan.NOCAP_REFUSAL_CODE)
    assert "FLLIPER_PDFLIP_L15_MIB" in msg
    assert "c1=" in msg  # the fix is named in the text


def test_master_on_override_all_zero_refuses():
    env = {"FLLIPER_PDFLIP_L15": "1", "FLLIPER_PDFLIP_L15_MIB": "c1=0,c2=0"}
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [9000, 9000, 9000],
                                   [None, None, None], env)
    assert l15_plan.refuse_no_caps(posts, env) is not None


def test_master_on_without_override_refuses_through_resolve_posts():
    env = {"FLLIPER_PDFLIP_L15": "1"}
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [9000, 9000, 9000],
                                   [None, None, None], env)
    assert [p.src for p in posts] == ["UNMEASURED"] * 3
    assert l15_plan.refuse_no_caps(posts, env) is not None


def test_master_on_with_a_held_card_passes():
    env = {"FLLIPER_PDFLIP_L15": "1", "FLLIPER_PDFLIP_L15_MIB": "c1=7616,c2=1792"}
    posts = l15_plan.resolve_posts(l15_plan.LINE_QWEN27B, [20000, 9000, 4000],
                                   [None, None, None], env)
    assert l15_plan.refuse_no_caps(posts, env) is None


def test_launcher_raises_on_the_refusal():
    # wiring: the launcher's L15-POST block must raise PdFlipLaunchRefused on it
    import inspect

    from flliper.srt.pdflip import launcher as L

    src = inspect.getsource(L)
    i = src.index("l15_posts = l15_plan.resolve_posts(")
    block = src[i:i + 2500]   # S4 grew the L15-POOL boot-line block between the two
    assert "l15_plan.refuse_no_caps(" in block
    assert "raise PdFlipLaunchRefused(" in block

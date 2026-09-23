"""xsn392 (19.09.): the prefetch claim reduce must have ONE form on every
rank. Under --speculative-draft-placement solo the host arms the draft tier
(packed [hit, -hit] MIN) and the shadows do not (scalar MIN): a 2-vector
against two scalars timed every probe out at its budget with 0 pages on
every rank. A solo shadow answers the packed form."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.cache_controller import HiCacheController  # noqa: E402
from sglang.srt.mem_cache import kv_cache_builder as kb  # noqa: E402


def _stub(armed: bool, shadow):
    st = types.SimpleNamespace(draft_tier_armed=lambda direction: armed)
    if shadow is not None:
        st.solo_draft_shadow = shadow
    return st


def test_the_packed_form_follows_the_draft_tier_or_the_solo_shadow_mark():
    f = HiCacheController.draft_claim_packed
    assert f(_stub(True, None)) is True          # the host / any draft-armed rank
    assert f(_stub(False, None)) is False        # a plain rank without a draft
    assert f(_stub(False, False)) is False
    assert f(_stub(False, True)) is True         # the solo shadow


def test_the_builder_marks_a_solo_shadow_and_leaves_others_alone():
    class _Ctl:
        solo_draft_shadow = False
    ctl = _Ctl()
    tree = types.SimpleNamespace(cache_controller=ctl)
    runner = types.SimpleNamespace(token_to_kv_pool=None, is_draft_solo_shadow=True)
    dw = types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner))
    sa = types.SimpleNamespace(enable_multi_layer_eagle=False)
    spec = types.SimpleNamespace(is_ngram=lambda: False)
    kb.maybe_register_hicache_draft(tree_cache=tree, draft_worker=dw, spec_algorithm=spec,
                                    server_args=sa, enable_hierarchical_cache=True, page_size=1)
    assert ctl.solo_draft_shadow is True
    ctl2 = _Ctl()
    runner2 = types.SimpleNamespace(token_to_kv_pool=None, is_draft_solo_shadow=False)
    dw2 = types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner2))
    kb.maybe_register_hicache_draft(tree_cache=types.SimpleNamespace(cache_controller=ctl2),
                                    draft_worker=dw2, spec_algorithm=spec, server_args=sa,
                                    enable_hierarchical_cache=True, page_size=1)
    assert ctl2.solo_draft_shadow is False

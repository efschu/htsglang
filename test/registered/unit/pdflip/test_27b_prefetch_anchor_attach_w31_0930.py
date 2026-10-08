"""27B z30y8 (650238fdd2, 30.09.): 22 W50 reroutes x_refusal_midstream, 17 of them path=held -- NF's y5a class.

The chain on 27B, pdflip-18-64 (D log ...09301808_650238fdd2):
* the front priced it SHORT on presence_span=88994 (d_leg2_cached);
* D: ``HiCache prefetch success matched=91553 loaded=0 ... deliverable=91553``, PDFLIP-LOAD-DEVICE 91553 tokens --
  the span was ALREADY in the tree (a sibling's device load), so the read inserted no host node and
  ``MambaComponent.commit_hicache_transfer(PREFETCH)`` released the read Mamba anchor;
* the admission match found no state on the path -> ``X-GATE uncached=91738`` > X -> W31 -> W50 (to P).

Fix: NF d5d75f328a PREFETCH-ANCHOR-ATTACH (the node the read's key ended at takes the anchor when it has no
state). This test runs the chain end to end on the real UnifiedRadixCache (NF's builder, scaled: a 128-token
device path, a read of 96 with its anchor, a 120-token admission, X=32 standing for 12288): with the attach the
match resumes at the anchor (96) and the uncached extent (24) is admitted; without it D has no state on the
path, the whole prompt is uncached (120 > X) -> W31, exactly the z30y8 verdict.
"""
from __future__ import annotations

import importlib.util
import os
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from flliper.srt.mem_cache.radix_cache import RadixKey  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_anchor_attach_nf", os.path.join(os.path.dirname(__file__), "..", "mem_cache",
                                        "test_pdflip_prefetch_anchor_attach_0930.py"))
NF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NF)

X = 32          # stands for group D's --tp-prefill-max-tokens 12288
PROMPT = 120    # the held request's prompt (pdflip-18-64: 91738 tokens)
READ = 96       # its store read, ending at its anchor (91553 tokens)


def _x_gate(attach: bool):
    """The z30y8 chain: sibling on the device, the held read over it, commit, admission match, X gate."""
    with envs.FLLIPER_PDFLIP_PREFETCH_ANCHOR_ATTACH.override(attach):
        cache, _pool, toks = NF._device_path(128, anchor=None)
        res = NF._read_span(cache, toks, READ)
        assert res.inserted_host_node is None       # "matched=91553 loaded=0": the span was in the tree
        NF._commit(cache, res)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", toks[:PROMPT]))))
    # a hybrid (Mamba/GDN) D resumes only at a state anchor (#928): what it must prefill is the rest
    uncached = PROMPT - int(m.state_anchor_depth)
    return ("W31" if uncached > X else "admit"), uncached, m


def test_with_the_attach_the_held_read_is_admitted_on_d():
    verdict, uncached, m = _x_gate(True)
    assert (verdict, uncached) == ("admit", PROMPT - READ)
    assert m.state_anchor_depth == READ and m.mamba_host_hit_length == 1


def test_without_it_the_same_chain_is_the_z30y8_w31():
    verdict, uncached, m = _x_gate(False)
    assert (verdict, uncached) == ("W31", PROMPT)
    assert m.state_anchor_depth == 0


def test_the_switch_is_on_by_default():
    assert envs.FLLIPER_PDFLIP_PREFETCH_ANCHOR_ATTACH.get() is True

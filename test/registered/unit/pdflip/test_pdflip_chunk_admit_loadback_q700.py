"""Q-700 P PREFILL-OOM RANK-DEATH (NF y9nf, boot ...10040027 c7b2690e2a, P PP0 00:42:31Z).

Group P, per-chunk admission (Punkt 2, chunked_prefill_size 16384, pool 262208 rows). One pass
admitted pdflip-12-86 (continuation, 13341), pdflip-10-83 (host load-back 88768, extend 193) and
pdflip-11-84 (host load-back 154176 + 4864 on device, extend 204): the adder charged each the NEXT
CHUNK only -- min(extend, rem_chunk) -- but ``init_load_back`` puts the whole host hit on the
device at admission. ``PDFLIP-ARENA-LOAD rows=242944 ... dst=[16448,259391] of 262208``, then
``Prefill out of memory ... Try to allocate 13738 tokens. Available full tokens: 2816
(... full_evictable_size_=0)`` -> ``PDFLIP RANK-DEATH (RANK_EXCEPTION)``, PP1/PP2 followed.

Pinned: the load-back is charged whole, the chunk only on the rest; replaying the pass, the
second load-back is refused (NO_TOKEN, it waits) and the batch's extend fits.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import park as pk  # noqa: E402

POOL = 262208
USED = 16448          # pdflip-12-85/86's first chunk (dst starts at 16448)
CHUNK = 16384
PAGE = 64


def _pass(load_back_charged: bool):
    """The adder's per-chunk gate over the specimen's three candidates; returns
    (admitted rids, rows the batch's extend still needs, rows free after the load-backs)."""
    free = POOL - USED
    offset = 0            # rem_total_token_offset: the extends promised this pass
    rem_chunk = CHUNK
    admitted, extend_need = [], 0
    # rid, extend (fill - device prefix), host load-back extent
    for rid, ext, lb in (("pdflip-12-86", 13341, 0), ("pdflip-10-83", 88961, 88768),
                         ("pdflip-11-84", 154380, 154176)):
        charge = pk.chunk_admit_tokens(ext, rem_chunk, load_back=lb if load_back_charged else 0)
        total = charge + PAGE
        if total >= free - offset:
            continue      # NO_TOKEN: waits
        free -= lb        # init_load_back: the whole host hit lands on the device
        rest = ext - lb
        take = min(rest, rem_chunk)
        rem_chunk -= take
        offset += take
        extend_need += take
        admitted.append(rid)
    return admitted, extend_need, free


def test_the_specimen_pass_ooms_without_the_load_back_charge():
    admitted, need, free = _pass(load_back_charged=False)
    assert admitted == ["pdflip-12-86", "pdflip-10-83", "pdflip-11-84"]
    assert free == 2816 and need > free          # 'Available full tokens: 2816'


def test_the_load_back_is_charged_whole_and_the_pass_fits():
    admitted, need, free = _pass(load_back_charged=True)
    assert admitted == ["pdflip-12-86", "pdflip-10-83"]   # pdflip-11-84 waits (NO_TOKEN)
    assert need <= free


def test_chunk_admit_tokens_load_back():
    assert pk.chunk_admit_tokens(88961, 3043, load_back=88768) == 88768 + 193
    assert pk.chunk_admit_tokens(154380, 2850, load_back=154176) == 154176 + 204
    assert pk.chunk_admit_tokens(99572, 4096, load_back=50000) == 50000 + 4096
    assert pk.chunk_admit_tokens(900, 4096, load_back=5000) == 900     # clamped to the extend
    assert pk.chunk_admit_tokens(99572, None, load_back=50000) == 99572  # chunking off: whole
    assert pk.chunk_admit_tokens(99572, 4096, anchor_gap=1) == 4097      # no load-back: unchanged


def test_the_adder_passes_the_load_back_extent():
    from flliper.srt.managers import schedule_policy as sp

    src = open(sp.__file__).read()
    i = src.index("chunk_admit_tokens as _cat")
    assert "load_back=int(_pp_load_back_extent(req) or 0)" in src[i:i + 600]

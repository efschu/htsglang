"""L15-02b: unit tests for the L1.5 SHADOW instrument (pure parts only).

Plain pytest functions (CustomTestCase's retry() hides the real failure).
Hermetic: no scheduler/front imports, no GPU, env passed explicitly as a dict.
"""

from flliper.srt.pdflip.l15_shadow import (
    ShadowLedger,
    candidates_from,
    caps_from_env,
    rows_split,
    shadow_on,
)


def _entry(**kw):
    base = dict(
        rid="r",
        kind="seat",
        last_active=0,
        rows_by_rank=(1,),
        anchor_depth=5,
        kv_depth=5,
    )
    base.update(kw)
    return base


def test_shadow_on_parses_the_switch():
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "1"}) is True
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "true"}) is True
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "on"}) is True
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "ON"}) is True
    assert shadow_on({}) is False
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": ""}) is False
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "0"}) is False
    assert shadow_on({"FLLIPER_PDFLIP_L15_SHADOW": "yes"}) is False


def test_caps_from_env_override_named_cards_only():
    env = {"FLLIPER_PDFLIP_L15_MIB": "c0=100,c2=3000"}
    # cell bytes: card0 262144, card1 1048576, card2 131072
    caps = caps_from_env(env, 3, (262144, 1048576, 131072))
    # 100 * 2**20 / 262144 = 400; card 1 absent -> 0; 3000 * 2**20 / 131072 = 24000
    assert caps == (400, 0, 24000)


def test_caps_from_env_absent_or_auto_is_all_zero():
    assert caps_from_env({}, 2, (1024, 1024)) == (0, 0)
    assert caps_from_env({"FLLIPER_PDFLIP_L15_MIB": "auto"}, 2, (1024, 1024)) == (0, 0)


def test_caps_from_env_rounds_down():
    # 1 MiB / 7 bytes = 149796.57... -> floor
    caps = caps_from_env({"FLLIPER_PDFLIP_L15_MIB": "c0=1"}, 1, (7,))
    assert caps == (1048576 // 7,)


def test_caps_from_env_follows_rank_order():
    env = {"FLLIPER_PDFLIP_L15_MIB": "c1=5"}
    caps = caps_from_env(env, 3, (1048576, 1048576, 1048576))
    assert caps == (0, 5, 0)


def test_caps_from_env_card_of_rank_maps_rank_to_card():
    # TP ranks do not sit on cards 0,1,2 in order (TP0 is the 5090):
    # rank 0 -> card 1, rank 1 -> card 0, rank 2 -> card 2.
    env = {"FLLIPER_PDFLIP_L15_MIB": "c1=100"}
    caps = caps_from_env(env, 3, (262144, 1048576, 1048576), card_of_rank=(1, 0, 2))
    # only rank 0 (on card 1) gets the cap: 100 * 2**20 / 262144 = 400
    assert caps == (400, 0, 0)


def test_candidates_from_accepts_valid_entries():
    cands = candidates_from([_entry(rid="a", rows_by_rank=[2, 3])])
    assert len(cands) == 1
    assert cands[0].rid == "a"
    assert cands[0].rows_by_rank == (2, 3)


def test_candidates_from_skips_invalid_entries():
    bad = [
        _entry(rid=""),                       # empty rid
        _entry(rid=None),                     # non-str rid
        _entry(kind="weird"),                 # unknown kind
        _entry(kind=None),                    # non-str kind
        _entry(rows_by_rank="abc"),           # not a sequence of ints
        _entry(rows_by_rank=(-1,)),           # negative rows
        _entry(anchor_depth="x"),             # non-int anchor
        _entry(kv_depth=-2),                  # negative kv depth
        {"rid": "x"},                         # missing fields
        "not-a-dict",                         # not even a dict
    ]
    assert candidates_from(bad) == []
    mixed = candidates_from([_entry(rid="ok"), "not-a-dict"])
    assert [c.rid for c in mixed] == ["ok"]


def test_ledger_at_load_only_for_held_rids():
    led = ShadowLedger()
    assert led.note_load("a", 12.0, 5.0) is None  # nothing held before a sleep
    led.note_sleep(["a", "b"])
    assert led.note_load("a", 12.0, 5.0) == "L15-SHADOW at=load rid=a ms=12.0 read_ms=5.0"
    assert led.note_load("c", 9.0, 1.0) is None   # never held
    led.note_sleep(["b"])                          # reset per sleep
    assert led.note_load("a", 1.0, 1.0) is None   # dropped by the reset
    assert led.note_load("b", 1.0, 1.0) == "L15-SHADOW at=load rid=b ms=1.0 read_ms=1.0"


def test_ledger_note_sleep_accepts_a_holdset():
    from flliper.srt.pdflip.l15_policy import Candidate, select_hold

    cands = [
        Candidate(
            rid="x",
            kind="seat",
            last_active=0,
            rows_by_rank=(1,),
            anchor_depth=2,
            kv_depth=2,
        )
    ]
    led = ShadowLedger()
    led.note_sleep(select_hold(cands, (10,), 4))
    assert led.note_load("x", 3, 1) == "L15-SHADOW at=load rid=x ms=3 read_ms=1"


def test_rows_split_proportional_floor_remainder_in_order():
    assert rows_split(10, 3, [1, 2, 2]) == (2, 4, 4)
    assert sum(rows_split(11, 3, [1, 2, 2])) == 11  # remainder to rank 0
    assert rows_split(30, 3, [1, 2, 2]) == (6, 12, 12)


def test_rows_split_even_sums_exactly():
    parts = rows_split(7, 3)
    assert len(parts) == 3
    assert sum(parts) == 7
    assert parts == (3, 2, 2)  # remainder to the leading ranks in order


def test_select_hold_skips_cap0_but_not_capped_ranks_for_split_rows():
    from flliper.srt.pdflip.l15_policy import select_hold

    rows = rows_split(30, 3, [1, 2, 2])
    assert rows == (6, 12, 12)
    hs = select_hold(
        candidates_from([_entry(rid="a", rows_by_rank=rows)]), (0, 5, 5), 1
    )
    assert hs.rids == ()  # 12 > 5 on ranks 1/2: not admitted
    assert ("a", "no_room") in hs.excluded
    # Old hook shape (whole request as rank-0-only rows): rank 0 has cap 0,
    # select_hold skips cap-0 ranks, so the request WAS admitted -> the
    # shadow hold set was "everything" and the N1 price meaningless.
    old = select_hold(
        candidates_from([_entry(rid="a", rows_by_rank=(30,))]), (0,), 1
    )
    assert old.rids == ("a",)


def test_front_shadow_block_reads_the_deque_with_islice():
    # front.py's self.queue is a collections.deque: queue[:8] raises
    # TypeError and the bare except swallowed it, so the line never
    # appeared. Check the source without importing front (not hermetic).
    import pathlib

    import flliper.srt.pdflip.l15_shadow as _m

    src = pathlib.Path(_m.__file__).with_name("front.py").read_text(encoding="utf-8")
    i = src.index("HOT-HANDOVER-SHADOW")
    block = src[i : i + 600]
    assert "islice" in block
    assert "self.queue[:8]" not in block


# --- N1 (dkr27browauthoritybar1fs10011036): cap=0,0,0 on the hybrid pool ----------------


def _hybrid_pool(layers=2, rows=1024, heads=4, dim=8):
    from types import SimpleNamespace

    import torch

    inner = SimpleNamespace(
        k_buffer=[torch.zeros(rows, heads, dim, dtype=torch.float16) for _ in range(layers)],
        v_buffer=[torch.zeros(rows, heads, dim, dtype=torch.float16) for _ in range(layers)],
    )
    return SimpleNamespace(full_kv_pool=inner)  # HybridLinearKVPool: no k_buffer of its own


def test_cell_bytes_unwraps_the_hybrid_pool_and_prices_one_token():
    """RED on 5ab4b12b28: 0 for the wrapper (no k_buffer), and the inner pool
    priced the WHOLE layer tensor (rows x heads x dim) instead of one row."""
    from flliper.srt.pdflip.l15_shadow import cell_bytes_from

    pool = _hybrid_pool(layers=2, rows=1024, heads=4, dim=8)
    # one token: 2 layers x (K + V) x 4 heads x 8 dims x 2 bytes = 256
    assert cell_bytes_from(pool) == 256
    assert cell_bytes_from(pool.full_kv_pool) == 256


def test_caps_from_the_n1_env_are_nonzero_on_the_named_cards():
    from flliper.srt.pdflip.l15_shadow import caps_from_env, cell_bytes_from

    cell = cell_bytes_from(_hybrid_pool())
    caps = caps_from_env({"FLLIPER_PDFLIP_L15_MIB": "c1=7616,c2=1792"}, 3, [cell] * 3)
    assert caps[0] == 0 and caps[1] == 7616 * 2**20 // cell and caps[2] == 1792 * 2**20 // cell

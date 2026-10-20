"""F3: the seat-row SHRINK without one ``.cpu()`` -- byte for byte the host path.

rc26 (20.10.): every rows-OFF move read 4-6 tables per layer back to the host (48 layers) between two
device-wide syncs. ``seat_rows_off_device`` does the coldest-first pairs, the byte moves, the ``hot_phys``
drops and the OFF writes as device ops. What must hold, over random tables (bijection kept):
* tables (row_key, row_use, hot_phys, pf_row), the bank bytes and seat_on equal the host path's;
* the departed experts read back from the device snapshots equal the host path's ``departed_out``;
* no host read: ``Tensor.cpu`` / ``tolist`` / ``item`` are never called (they raise here);
* the masked warm commit leaves invalid / meanwhile-hot experts' rows free and keeps the bijection.
"""
import os
import random
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.moe import expert_pool_device as epd  # noqa: E402

E, R, S, C, X, W = 64, 4, 3, 20, 24, 3


def _tables(seed: int, k_old: int):
    rng = random.Random(seed)
    t = epd.allocate_pool_tables("cpu", E, R + C + X, R + S, S, {e: e for e in range(R)},
                                 [-1] * R + list(range(E - R)), seat_rows=X)
    epd.set_seat_rows_on(t, k_old, device_write=True)
    lru, top = int(t.lru_start), int(t.seat_base) + k_old
    rows = list(range(lru, int(t.seat_base))) + list(range(int(t.seat_base), top))
    experts = list(range(R, E))
    rng.shuffle(experts)
    for r in rows:
        if rng.random() < 0.2 or not experts:
            continue  # a free row
        e = experts.pop()
        t.row_key[r] = e
        t.hot_phys[e] = r
        t.row_use[r] = rng.randrange(1, 1000)
        t.pf_row[r] = rng.choice([-1, 5])
    t.clock[0] = rng.choice([500, 2000])
    bank = torch.arange(int(t.row_key.numel()) * W, dtype=torch.float32).reshape(-1, W)
    return t, bank


def _clone(t, bank):
    import copy

    t2 = copy.copy(t)
    for f in ("row_key", "row_use", "hot_phys", "pf_row", "clock"):
        setattr(t2, f, getattr(t, f).clone())
    return t2, bank.clone()


@pytest.mark.parametrize("seed", range(40))
def test_the_device_shrink_equals_the_host_shrink(seed):
    rng = random.Random(1000 + seed)
    k_old = rng.randrange(1, X + 1)
    k = rng.randrange(0, k_old)
    th, bh = _tables(seed, k_old)
    td, bd = _clone(th, bh)
    departed_host = []

    def host_mover(moves):
        for s, d in moves:
            bh[d] = bh[s]

    epd.set_seat_rows_on(th, k, device_write=True, move_rows=host_mover, departed_out=departed_host)

    def dev_mover(src, dst):
        bd.index_copy_(0, dst, bd.index_select(0, src))

    snaps = []
    with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")), \
            mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")), \
            mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")):
        old = epd.seat_rows_off_device(td, k, move_rows=dev_mover, recall=lambda *a: snaps.append(a))
    assert old == k_old and int(td.seat_on) == int(th.seat_on) == k
    for f in ("row_key", "row_use", "hot_phys", "pf_row"):
        assert torch.equal(getattr(td, f), getattr(th, f)), f
    assert torch.equal(bd, bh)
    keys_b, uses_b, hot_a = snaps[0]
    assert epd.departed_experts(keys_b.tolist(), uses_b.tolist(), hot_a.tolist(), num_experts=E) == departed_host
    assert epd.bijection_breaks(td) == 0


def test_a_grow_is_the_old_grow():
    t, _ = _tables(3, 4)
    assert epd.seat_rows_off_device(t, 9) == 4 and int(t.seat_on) == 9
    base = int(t.seat_base)
    assert (t.row_key[base + 4:base + 9] == -1).all()


def test_the_masked_commit_keeps_the_bijection():
    t, _ = _tables(7, 10)
    base = int(t.seat_base)
    rows = torch.arange(base + 10, base + 14)
    epd.set_seat_rows_on(t, 14, device_write=True)
    epd.reserve_warm_rows(t, rows)
    cold = [e for e in range(R, E) if int(t.hot_phys[e]) < 0][:3]
    hot_e = next(e for e in range(R, E) if int(t.hot_phys[e]) >= 0)
    experts = torch.tensor(cold + [hot_e])
    valid = torch.tensor([True, False, True, True])
    epd.commit_warm_rows_masked(t, experts, rows, valid)
    assert int(t.hot_phys[cold[0]]) == base + 10 and int(t.row_key[base + 10]) == cold[0]
    assert int(t.hot_phys[cold[1]]) == -1 and int(t.row_key[base + 11]) == -1     # invalid: free
    assert int(t.row_key[base + 13]) == -1                                          # hot elsewhere: free
    assert epd.bijection_breaks(t) == 0

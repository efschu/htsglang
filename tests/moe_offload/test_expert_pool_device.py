"""Device-planned expert pool (vLLM #56177 port, 19.09.): the reference step
program on CPU pins the semantics the Triton program must match."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.layers.moe import expert_pool_device as ep

E, ROWS, R, S = 8, 7, 2, 3  # residents 0,1 in rows 0,1; LRU rows 2,3; staging 4,5,6
HOST = [-1, -1, 0, 1, 2, 3, 4, 5]


def _pool():
    t = ep.allocate_pool_tables("cpu", E, ROWS, R, S, {0: 0, 1: 1}, HOST)
    b = ep.allocate_step_buffers("cpu", E, 8)
    return t, b


def _ids(*v):
    return torch.tensor(v, dtype=torch.int32)


def test_misses_take_free_rows_first_and_routes_point_at_them():
    t, b = _pool()
    pairs, _ = ep.step_reference(t, _ids(2, 2, 3), b)
    assert pairs == [(0, 2), (1, 3)]  # host rows of experts 2,3 into free LRU rows
    assert b.routes[:3].tolist() == [2, 2, 3]
    assert int(b.gather_count[0]) == 2 and int(b.promoted_count[0]) == 2
    assert t.hot_phys.tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert t.row_key.tolist() == [0, 1, 2, 3, -1, -1, -1]
    pairs, _ = ep.step_reference(t, _ids(0, -1, 3), b)  # a resident hit, padding, an LRU hit
    assert pairs == [] and b.routes[:3].tolist() == [0, -1, 3]
    with pytest.raises(ValueError):
        ep.step_reference(t, _ids(2, 3, 4, 5), b)  # more lanes than staging rows


def test_lru_evicts_the_least_recently_used_row_and_residents_never_move():
    t, b = _pool()
    ep.step_reference(t, _ids(2, 3), b)       # clock 1: rows 2,3 used
    ep.step_reference(t, _ids(3), b)          # clock 2: row 3 refreshed
    pairs, _ = ep.step_reference(t, _ids(4), b)  # clock 3: expert 4 evicts row 2 (LRU)
    assert pairs == [(2, 2)]
    assert t.hot_phys.tolist()[2] == -1 and t.hot_phys.tolist()[4] == 2
    for _ in range(6):
        ep.step_reference(t, _ids(5, 6, 7), b)
    assert t.row_key.tolist()[:2] == [0, 1]  # residents untouched
    assert t.hot_phys.tolist()[:2] == [0, 1]


def test_a_row_used_this_step_is_never_the_victim():
    t, b = _pool()
    ep.step_reference(t, _ids(2, 3), b)
    pairs, _ = ep.step_reference(t, _ids(2, 4, 5), b)  # 2 hits row 2; 4 evicts row 3; 5 staged
    assert pairs[0] == (2, 3)
    assert int(b.promoted_count[0]) == 1 and int(b.staged_count[0]) == 1
    assert pairs[1] == (3, 4)  # expert 5 (host row 3) into staging row 4
    assert b.routes[:3].tolist() == [2, 3, 4]
    assert t.hot_phys.tolist()[5] == -1  # staged only, never resident


def test_closed_gate_stages_every_miss_and_moves_nothing():
    t, b = _pool()
    t.gate.fill_(0)
    pairs, _ = ep.step_reference(t, _ids(2, 3), b)
    assert pairs == [(0, 4), (1, 5)] and int(b.promoted_count[0]) == 0
    assert t.row_key.tolist() == [0, 1, -1, -1, -1, -1, -1] and int(t.clock[0]) == 0


def test_bad_ids_set_the_sticky_error_and_are_skipped():
    t, b = _pool()
    ep.step_reference(t, _ids(9, 2), b)
    assert int(t.error[0]) == 1 and b.routes[:2].tolist() == [-1, 2]


def test_copy_rows_reference_moves_every_tensor_row():
    src = [torch.arange(12.0).view(6, 2), torch.arange(6).view(6, 1)]
    dst = [torch.zeros(6, 2), torch.zeros(6, 1, dtype=torch.int64)]
    ep.copy_rows(src, dst, torch.tensor([4, 1]), torch.tensor([2, 3]), torch.tensor([2]))
    assert dst[0][2].tolist() == [8.0, 9.0] and int(dst[1][3]) == 1 and dst[0][0].sum() == 0


def test_sync_tables_takes_the_hosts_lru_truth_and_frees_the_rest():
    t, b = _pool()
    ep.step_reference(t, _ids(2, 3), b)
    ep.sync_tables(t, {2: 6, 5: 7})  # row 2 now holds expert 6; row 5 is staging -> ignored
    assert t.hot_phys.tolist() == [0, 1, -1, -1, -1, -1, 2, -1]
    assert t.row_key.tolist() == [0, 1, 6, -1, -1, -1, -1]


def test_allocation_refuses_a_resident_with_a_host_row_and_an_empty_lru():
    with pytest.raises(ValueError):
        ep.allocate_pool_tables("cpu", E, ROWS, R, S, {0: 0, 2: 1}, HOST)
    with pytest.raises(ValueError):
        ep.allocate_pool_tables("cpu", E, 4, 2, 2, {0: 0, 1: 1}, HOST)  # no LRU row


def test_pool_mode_is_selected_by_name_and_only_with_an_offload(monkeypatch):
    from sglang.srt.layers.moe import offload_capture_gate as g

    monkeypatch.setenv(g.ENV_GRAPH_MODE, "pool")
    assert g.resolve_offload_graph_mode(0.5, False) == g.MODE_POOL
    assert g.resolve_offload_graph_mode(1.0, False) == g.MODE_EAGER
    monkeypatch.delenv(g.ENV_GRAPH_MODE)
    assert g.resolve_offload_graph_mode(0.5, False) == g.MODE_EAGER


def test_take_report_counts_misses_since_the_last_report_and_resets():
    t, b = _pool()
    ep.step_reference(t, _ids(2, 3), b)
    ep.step_reference(t, _ids(2), b)
    assert ep.take_report(t) == (2, 2)
    assert ep.take_report(t) == (0, 0)


def test_hotset_path_template_names_the_ranks_own_file():
    from sglang.srt.layers.moe.expert_offload import hotset_path_for_rank

    assert hotset_path_for_rank("/x/hot_tp{rank}.json", 2) == "/x/hot_tp2.json"
    assert hotset_path_for_rank("/x/hot.json", 2) == "/x/hot.json"


def test_hotset_file_never_covers_a_draft_layer():
    # fn4j 19.09.: the draft's layer 0 shares key "0" and (after the expert
    # split) the local expert count with the target's layer 0.
    from types import SimpleNamespace

    from sglang.srt.layers.moe.expert_offload import hotset_covers_layer

    assert hotset_covers_layer(SimpleNamespace(_sglang_prefix="model.layers.0.mlp.experts"))
    assert hotset_covers_layer(SimpleNamespace())
    assert not hotset_covers_layer(SimpleNamespace(_sglang_prefix="mtp.layers.0.mlp.experts"))
    assert not hotset_covers_layer(SimpleNamespace(_sglang_prefix="model.mtp.layers.0.mlp"))

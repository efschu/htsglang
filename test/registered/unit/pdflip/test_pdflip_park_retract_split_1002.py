"""PARK-RETRACT-SPLIT (02.10.): the flip park's retract phase split per
request into cache_finished_req / write_backup / cache_controller.write, one
line per park; the wrappers are gone after the retract."""

import logging
import time

import pytest

from flliper.srt.pdflip import park_retract_split as prs


class _Controller:
    def __init__(self):
        self.calls = 0

    def write(self, device_indices=None, node_id=None, extra_pools=None):
        self.calls += 1
        time.sleep(0.004)
        return [1]


class _Tree:
    def __init__(self):
        self.cache_controller = _Controller()
        self.backups = 0

    def write_backup(self, node, write_back=False):
        self.backups += 1
        self.cache_controller.write(device_indices=node, node_id=node)
        time.sleep(0.002)
        return 1

    def cache_finished_req(self, req, is_insert=True):
        for node in range(req.nodes):
            self.write_backup(node)
        time.sleep(0.003)


class _Req:
    def __init__(self, rid, nodes):
        self.rid = rid
        self.nodes = nodes


def _retract(tree, reqs):
    def run():
        for r in reqs:
            tree.cache_finished_req(r, is_insert=True)
            time.sleep(0.005)  # reset_for_retract and the rest
        return reqs

    return run


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_RETRACT_SPLIT", "1")


def test_split_line_names_release_backup_write_per_rid(switch_on, caplog):
    tree = _Tree()
    reqs = [_Req("pdflip-18-76", 3), _Req("pdflip-18-77", 1)]
    with caplog.at_level(logging.INFO, logger=prs.__name__):
        out = prs.run_split(tree, _retract(tree, reqs), epoch=18)
    assert out is reqs
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-PARK-RETRACT-SPLIT" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "epoch=18 n=2" in line
    assert "backup_n=4" in line and "write_n=4" in line
    assert "pdflip-18-76:" in line and "pdflip-18-77:" in line


def test_split_sums_are_consistent(switch_on):
    tree = _Tree()
    reqs = [_Req("a", 2), _Req("b", 2)]
    split = prs.RetractSplit()
    split.arm(tree)
    t0 = time.perf_counter()
    try:
        _retract(tree, reqs)()
    finally:
        split.total_ms = (time.perf_counter() - t0) * 1000.0
        split.disarm()
    release = sum(split.release_ms.values())
    assert split.write_n == 4 and split.backup_n == 4
    assert split.write_ms <= split.backup_ms <= release <= split.total_ms
    # 4 controller writes at >= 4 ms each, 4 backups add >= 2 ms each
    assert split.write_ms >= 16.0
    assert split.backup_ms - split.write_ms >= 8.0
    # the per-request rest (3 ms each) and the retract rest (5 ms each)
    assert release - split.backup_ms >= 6.0
    assert split.total_ms - release >= 10.0


def test_wrappers_removed_after_retract_even_on_error(switch_on):
    tree = _Tree()

    def boom():
        tree.cache_finished_req(_Req("x", 1))
        raise RuntimeError("retract failed")

    with pytest.raises(RuntimeError):
        prs.run_split(tree, boom, epoch=1)
    assert "cache_finished_req" not in tree.__dict__
    assert "write_backup" not in tree.__dict__
    assert "write" not in tree.cache_controller.__dict__
    # class behaviour intact afterwards
    tree.cache_finished_req(_Req("y", 2))
    assert tree.backups == 3


def test_switch_off_runs_retract_untouched(monkeypatch, caplog):
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_RETRACT_SPLIT", "0")
    tree = _Tree()
    reqs = [_Req("a", 1)]
    with caplog.at_level(logging.INFO, logger=prs.__name__):
        out = prs.run_split(tree, _retract(tree, reqs), epoch=3)
    assert out is reqs
    assert not [r for r in caplog.records if "PDFLIP-PARK-RETRACT-SPLIT" in r.getMessage()]


def test_no_tree_is_harmless(switch_on):
    assert prs.run_split(None, lambda: ["r"], epoch=0) == ["r"]

"""H106 (rc12z15 f49f7bddd2, D 10:22:39, TP1+TP2 at once): the QSA sidecar of
a Form A worker is addressed with ids its byteless KV anchor GREW (#249).

rc12z15 D log: ``#249 BYTELESS-GROW pool=MHATokenToKVPoolHost rows 353600 ->
373504`` at the wake that read six held prompts (63360 + 17536 + 105664 +
17664 + 63488 + 105792 = 373504), the QSA host pool stayed at its assembly
size (``V4 paged pool 'qsa_indexer' (layers=12, pages=5525)`` = 353600 ids),
and the load of weg2-0-4's tail died in ``transfer_kv_direct``:
``output with shape [1, 4096] doesn't match the broadcast shape [0, 4096]``
-- the host slice of page >= 5525 is empty.

Hermetic: the real ``QSAPagedHostPool`` (layer_first, direct), its real
``load_to_device_per_layer`` / ``backup_from_device_all_layer``; the CUDA
op ``transfer_kv_direct`` is replaced by the same slice-copy it performs
(sgl-kernel ``transfer_page_direct``: ``dst.slice(..).copy_(src.slice(..))``
over contiguous runs), which reproduces the metal message on CPU."""
import pytest
import torch

import sglang.srt.mem_cache.memory_pool_host as mph
import sglang.srt.rank_role as rank_role
from sglang.srt.mem_cache.qsa_pool_host import QSAPagedHostPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")

RATIO, PAGE, HEADS, DIM = 4, 64, 1, 128
HOST_PAGES = 4  # the assembly size of the sidecar (rc12z15: 5525)


class _Dev:
    def __init__(self, layers=2, slots=256):
        self.qsa_compress_ratio = RATIO
        self.qsa_compressed_k_buffer_pool = [
            torch.zeros(slots, HEADS, DIM, dtype=torch.bfloat16) for _ in range(layers)
        ]
        self.full_attention_layer_id_mapping = {g: i for i, g in enumerate(range(layers))}


def _direct(src_layers, dst_layers, src_indices, dst_indices, page_size):
    """sgl-kernel transfer_kv_direct on the CPU: contiguous runs, slice copy."""
    s, d = src_indices.tolist(), dst_indices.tolist()
    start = 0
    for i in range(len(s)):
        if i < len(s) - 1 and s[i + 1] - s[i] == 1 and d[i + 1] - d[i] == 1:
            continue
        n = i + 1 - start
        for src, dst in zip(src_layers, dst_layers):
            dst[d[start]:d[start] + n].copy_(src[s[start]:s[start] + n])
        start = i + 1


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(mph, "transfer_kv_direct", _direct)
    dev = _Dev()
    host = QSAPagedHostPool([dev], num_host_tokens=HOST_PAGES * PAGE, page_size=PAGE,
                            layout="layer_first", pin_memory=False)
    for l, buf in enumerate(host.kv_buffer):
        buf.copy_(torch.arange(HOST_PAGES, dtype=torch.uint8)[:, None].expand_as(buf) + 10 * l)
    return dev, host


def _ids(pages):
    return torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in pages])


def _load(host, dev, host_pages, dev_pages):
    for l in range(host.layer_num):
        host.load_to_device_per_layer(dev, _ids(host_pages), _ids(dev_pages), l, "direct")


def test_worker_load_with_a_grown_id_is_skipped_not_a_crash(pool, monkeypatch):
    """The rc12z15 shape: a Form A worker loads a KV-grown id (page 4 of a
    4-page sidecar). RED on f49f7bddd2: RuntimeError '[1, 4096] ... [0, 4096]'.
    GREEN: nothing is transferred (the worker never reads its QSA rows), the
    device rows stay as they were, the skip is counted by name."""
    dev, host = pool
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    before = [b.clone() for b in host.device_buffers]
    n0 = getattr(QSAPagedHostPool, "_formA_skip_n", 0)
    _load(host, dev, [HOST_PAGES - 1, HOST_PAGES], [5, 6])
    assert all(torch.equal(a, b) for a, b in zip(before, host.device_buffers))
    assert getattr(QSAPagedHostPool, "_formA_skip_n", 0) == n0 + host.layer_num


def test_worker_backup_with_a_grown_id_is_skipped(pool, monkeypatch):
    """The write direction of the same ids: the direct copy into an empty host
    slice writes nothing silently, the kernel branches would write past the
    pinned buffer. GREEN: skipped by name, the host rows untouched."""
    dev, host = pool
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    before = [b.clone() for b in host.kv_buffer]
    n0 = getattr(QSAPagedHostPool, "_formA_skip_n", 0)
    host.backup_from_device_all_layer(dev, _ids([HOST_PAGES + 2]), _ids([1]), "direct")
    assert all(torch.equal(a, b) for a, b in zip(before, host.kv_buffer))
    assert getattr(QSAPagedHostPool, "_formA_skip_n", 0) == n0 + 1


def test_a_rank_with_real_qsa_bytes_stops_by_name(pool, monkeypatch):
    """Not a Form A worker: the same out-of-range id is a wrong index for bytes
    the attention reads -- a named stop, never a silent skip. RED on
    f49f7bddd2 (the anonymous torch error), GREEN: 'H106 SIDECAR HOST INDEX
    OUT OF RANGE'."""
    dev, host = pool
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    with pytest.raises(RuntimeError, match="H106 SIDECAR HOST INDEX OUT OF RANGE"):
        _load(host, dev, [HOST_PAGES], [3])


def test_in_range_transfers_are_unchanged(pool, monkeypatch):
    """Ids inside the sidecar copy exactly as before, on a worker too."""
    dev, host = pool
    for worker in (True, False):
        monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda w=worker: w)
        for b in host.device_buffers:
            b.zero_()
        _load(host, dev, [2, 0], [7, 3])
        for l, buf in enumerate(host.device_buffers):
            assert torch.all(buf[7] == 2 + 10 * l) and torch.all(buf[3] == 0 + 10 * l)


def _skip_lines(caplog):
    return [r.getMessage() for r in caplog.records if "H106 FORM-A SIDECAR SKIP" in r.getMessage()]


def test_h106b_one_line_per_transfer_and_direction(pool, monkeypatch, caplog):
    """H106b (rc12z17 10:50:22Z: n=1..11+ in one second -- one line per layer
    call). A load over L layers is ONE transfer: one line; the next transfer
    (a backup) prints its own line carrying the previous transfer's sums."""
    import logging

    dev, host = pool
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    monkeypatch.setattr(QSAPagedHostPool, "_h106_open", None)
    caplog.set_level(logging.WARNING, logger="sglang.srt.mem_cache.qsa_pool_host")
    _load(host, dev, [HOST_PAGES, HOST_PAGES + 1], [5, 6])  # one transfer, layer_num calls
    lines = _skip_lines(caplog)
    assert len(lines) == 1, lines
    assert "dir=load" in lines[0] and "pages=2" in lines[0]
    host.backup_from_device_all_layer(dev, _ids([HOST_PAGES + 3]), _ids([1]), "direct")
    lines = _skip_lines(caplog)
    assert len(lines) == 2, lines
    assert "dir=backup" in lines[1]
    assert f"prev(dir=load calls={host.layer_num} pages_total={2 * host.layer_num})" in lines[1]


def test_h106b_periodic_counter_line(pool, monkeypatch, caplog):
    """Folded calls still surface: every H106_PERIODIC folded calls one line
    with suppressed_since_last_print."""
    import logging

    dev, host = pool
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    monkeypatch.setattr(QSAPagedHostPool, "_h106_open", None)
    monkeypatch.setattr(QSAPagedHostPool, "_h106_suppressed", 0)
    monkeypatch.setattr(QSAPagedHostPool, "H106_PERIODIC", 3)
    caplog.set_level(logging.WARNING, logger="sglang.srt.mem_cache.qsa_pool_host")
    for _ in range(4):
        host.load_to_device_per_layer(dev, _ids([HOST_PAGES]), _ids([5]), 0, "direct")
    lines = _skip_lines(caplog)
    assert len(lines) == 2, lines  # the transfer line + one periodic line
    assert "(periodic)" in lines[1] and "suppressed_since_last_print=3" in lines[1]


# ---- PA review of 1bab093912: the fold key names the pool; the load and backup
# threads share the fold state and each keeps its own direction ------------------

def test_h106b_two_pools_with_the_same_ids_are_two_transfers(pool, monkeypatch, caplog):
    """RED on 1bab093912: the key was (dir, first id, pages) class-wide, so a
    second sidecar pool skipping the same ids was folded into the first."""
    import logging

    dev, host = pool
    other = QSAPagedHostPool([dev], num_host_tokens=HOST_PAGES * PAGE, page_size=PAGE,
                             layout="layer_first", pin_memory=False)
    other.pool_name = "qsa_other"
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    monkeypatch.setattr(QSAPagedHostPool, "_h106_open", None)
    caplog.set_level(logging.WARNING, logger="sglang.srt.mem_cache.qsa_pool_host")
    host.load_to_device_per_layer(dev, _ids([HOST_PAGES]), _ids([5]), 0, "direct")
    other.load_to_device_per_layer(dev, _ids([HOST_PAGES]), _ids([5]), 0, "direct")
    lines = _skip_lines(caplog)
    assert len(lines) == 2, lines
    assert "pool=qsa_other" in lines[1]


def test_h106b_a_backup_in_another_thread_does_not_rename_a_load(pool, monkeypatch, caplog):
    """RED on 1bab093912: the direction was an instance attribute; a backup the
    backup thread starts while the load thread is between its direction and
    its skip made the load's line say dir=backup."""
    import logging
    import threading

    dev, host = pool
    state = {"inside": False}

    def _worker():
        if not state["inside"] and threading.current_thread() is threading.main_thread():
            state["inside"] = True
            t = threading.Thread(target=lambda: host.backup_from_device_all_layer(
                dev, _ids([HOST_PAGES + 3]), _ids([1]), "direct"))
            t.start()
            t.join()
        return True

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", _worker)
    monkeypatch.setattr(QSAPagedHostPool, "_h106_open", None)
    caplog.set_level(logging.WARNING, logger="sglang.srt.mem_cache.qsa_pool_host")
    host.load_to_device_per_layer(dev, _ids([HOST_PAGES]), _ids([5]), 0, "direct")
    lines = _skip_lines(caplog)
    import re

    dirs = [re.search(r"SIDECAR SKIP pool=\S+ dir=(\w+)", ln).group(1) for ln in lines]
    assert dirs == ["backup", "load"], lines

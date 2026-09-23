"""23.09. (fnFL2x33/x34): D TP0 died with SIGSEGV inside cudaMemcpyAsync at
unit 2 of the draft band, ``lm_head.weight_packed`` (636 MB), over BAR1
(x33) and over the SEQ host lane (x34) alike. The draft's head on D IS the
target's head (MEASURED-SHARED, data_ptr identity), which lives in region
``weights`` -- PAUSED while ``weights_draft`` is collected. The join skipped
the name from the DRAFT PLAN (35 descs), but the lane derives its unit list
from the raw join (38 units, so both sides agree without metadata), and the
collect wrote unit 2 into a reserved, unmapped range: a fault with no name.

Two guards: the measured head share joins the embed's ``no_write`` set
(consumed, never written -- eec31d1d12's form for the embed), and every
collect asks the driver about each destination ONCE before the first copy
into it and refuses by name.
"""
from __future__ import annotations

import ctypes
import os
import re
import threading
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402


#: the six directed pairs of the 3x3 rig (test_weg2_bar1_lanes_xsn365); p4 = (2, 0)
PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


class _HostOps:
    def create_stream(self, device):
        return 0

    def synchronize(self, stream):
        pass

    def memcpy_async(self, dst, src, n, stream):
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


class _ProbingOps(_HostOps):
    """Host copies plus a driver-style probe: every address is 'device'
    except the ones listed as unknown (rc 0, type 0, device -1: the
    reserved-but-unmapped reading of weg2xsn61 and x33/x34)."""

    def __init__(self, unknown):
        self.unknown = set(unknown)
        self.copied = 0

    def ptr_attrs(self, addr):
        if any(lo <= addr < hi for lo, hi in self.unknown):
            return (0, 0, -1)
        return (0, 2, 0)

    def memcpy_async(self, dst, src, n, stream):
        self.copied += 1
        super().memcpy_async(dst, src, n, stream)


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def test_an_unmapped_destination_is_refused_by_name_before_any_copy(tmp_path):
    slot, ring = 4096, 2
    window = bytearray(slot * ring)
    dep = _lanes(tmp_path, "P", 2)
    col = _lanes(tmp_path, "D", 0)
    col.recv["p4"] = SimpleNamespace(dptr=_addr(window), slot_bytes=slot, ring=ring)
    dep.peers["p4"] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=slot, ring=ring)
    good_src, good_dst = bytearray(os.urandom(3000)), bytearray(3000)
    bad_src, bad_dst = bytearray(os.urandom(3000)), bytearray(3000)
    descs = [
        SimpleNamespace(kind=tp.FLAT, nbytes=3000, src_off=0, dst_off=0, param_name="fc_embedding.weight",
                        tag="weights_draft", src_ptr=_addr(good_src), dst_ptr=_addr(good_dst)),
        SimpleNamespace(kind=tp.FLAT, nbytes=3000, src_off=0, dst_off=0, param_name="lm_head.weight_packed",
                        tag="weights_draft", src_ptr=_addr(bad_src), dst_ptr=_addr(bad_dst)),
    ]
    ops = _ProbingOps(unknown=[(_addr(bad_dst), _addr(bad_dst) + 3000)])
    out = {}

    def _collect():
        out["c"] = b1.run_bar1_units(descs, ops, lanes=col, lane_key="p4", role="dst",
                                     seq="1-weights_draft", phase="collect", budget_s=5.0,
                                     log=lambda *_a: None)

    th = threading.Thread(target=_collect)
    th.start()
    b1.run_bar1_units(descs, _HostOps(), lanes=dep, lane_key="p4", role="src",
                      seq="1-weights_draft", phase="deposit", budget_s=5.0, log=lambda *_a: None)
    th.join(10)
    why = out["c"]
    assert "lm_head.weight_packed" in why and "not mapped device memory" in why
    assert "type=0" in why and "device=-1" in why
    # the good piece before it was copied, the unknown one never touched
    assert bytes(good_dst) == bytes(good_src)
    assert bytes(bad_dst) == bytes(3000)
    assert ops.copied == 1


def test_host_fakes_without_a_probe_keep_copying():
    """The desk's ops carry no probe: nothing is asked, bytes move as before."""
    assert tp.dst_pointer_probe(_HostOps()) is None
    assert tp.dst_pointer_probe(SimpleNamespace(name="cudart")) is not None


def test_the_measured_head_share_is_in_the_no_write_set():
    """Bookkeeping (x34): `skip_names` narrows only the draft PLAN; the
    LANE's unit list comes from the raw join, so a head share that is not
    ALSO in `_weg2_xchg_no_write` is written into the paused target head.
    A site that builds the set from the embed share alone is the x33/x34
    mutant."""
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    m = re.search(r"self\._weg2_xchg_no_write = frozenset\(\s*"
                  r"\(str\(draft_region_tag\), str\(n\)\)\s*"
                  r"for n in set\(_shared\) \| set\(_lm_head_excluded\)\)", src)
    assert m, "the no_write set no longer carries the measured lm_head share"

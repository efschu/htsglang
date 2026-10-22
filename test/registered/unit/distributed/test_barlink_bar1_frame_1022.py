"""Fix A (22.10.): framed BAR1 slots -- no assumption about PCIe write order.

switch-p2p-wurzel-1022.md: on the PEX880xx topology the receiver read the end
of its slot before the sender's bytes were visible there, after the flag had
arrived (3080 GPU_DIRECT_RDMA_WRITES_ORDERING = NONE, posted writes with
Relaxed Ordering; E1: RO off is not enough). The fix frames every slot: 31 data
packets + 1 trailer {hash, round} per 512 bytes, and a receiver consumes a tile
only when the trailer carries THIS round and the hash of the bytes it read.

Three layers here:

* DESK (no card): the arithmetic (stride, capacity, window), the host
  reference of the hash and its rejection properties, and source invariants
  of the kernels (every slot read goes through ``frRecvTile``; status 3 is an
  abort; the unframed pipe is declined).
* GPU (one card, skipped without CUDA): the device hash is bit-identical to the
  host reference; a stale, torn, foreign-salt or old-generation tile is never
  accepted; a tile whose TRAILER lands first and whose data lands 30 ms later
  is accepted only after the data is there ("flag before data" adversary).
* MUTANTS (GPU): the same scenarios against builds with ``FRAME_MUTANT=1``
  (old generation accepted: round left out) and ``FRAME_MUTANT=2`` (no check:
  the pre-22.10. protocol). Each mutant must FAIL a named scenario -- that is
  the evidence that the scenario guards the danger direction.

The two-card regression (bench with the round term, >= 32 runs over 24/96/128
MiB x window 16/40) runs outside pytest, in a gpuq window:
``scripts/pdflip/barlink_frame_regress.sh``.
"""

import os
import re
import unittest
from pathlib import Path

from flliper.srt.distributed.device_communicators import barlink_bar1 as b1
from flliper.srt.distributed.device_communicators import barlink_bar1_frame as fr
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_COMM = (
    Path(__file__).resolve().parents[4]
    / "python/flliper/srt/distributed/device_communicators"
)
_EXT = _COMM / "barlink_bar1_ext.py"
_BAR1 = _COMM / "barlink_bar1.py"


def _raw_cuda_src() -> str:
    import ast

    tree = ast.parse(_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_CUDA_SRC":
            return node.value.value
    raise AssertionError("_CUDA_SRC not found")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ===========================================================================
# DESK
# ===========================================================================


class TestArithmetic(CustomTestCase):
    def test_framed_bytes_is_32_per_31(self):
        self.assertEqual(fr.framed_bytes(0), 0)
        self.assertEqual(fr.framed_bytes(1), 512)
        self.assertEqual(fr.framed_bytes(496), 512)
        self.assertEqual(fr.framed_bytes(497), 1024)
        self.assertEqual(fr.framed_bytes(31 * 16 * 1000), 512 * 1000)

    def test_capacity_inverts_framed_bytes(self):
        for stride in (512, 4096, 1 << 20, 2793472):
            cap = fr.capacity_of(stride)
            self.assertLessEqual(fr.framed_bytes(cap), stride)
            self.assertGreater(fr.framed_bytes(cap + 16), stride - 511)

    def test_geometry_lays_slots_out_at_the_framed_stride(self):
        for world in (2, 3, 4, 8):
            for mib in (16, 24, 40, 64):
                n = b1.max_payload(world, mib << 20, True, False, 0, 0)
                g = b1.geometry(world, n, True, False, 0, 0)
                self.assertLessEqual(g["region_bytes"], mib << 20,
                                     msg=f"R={world} w={mib}")
                self.assertGreaterEqual(g["slot_stride"],
                                        fr.framed_bytes(g["chunk_max"]))
                self.assertEqual(g["slot_stride"] % b1.PAGE_SIZE, 0)
                slots = 2 * (world - 1)
                self.assertEqual(g["off_ring"], slots * g["slot_stride"])
                self.assertEqual(g["off_a2a"], 2 * slots * g["slot_stride"])
                self.assertEqual(g["a2a_slot"], g["chunk_max"])
                # the largest chunk of the largest payload fits the capacity
                share = -(-(n // 16) // world) * 16
                self.assertLessEqual(share, g["chunk_max"])

    def test_max_payload_is_tight(self):
        """One more packet per rank must no longer fit the window."""
        for world in (2, 3):
            for mib in (16, 40):
                n = b1.max_payload(world, mib << 20, True, False, 0, 0)
                g = b1.geometry(world, n + world * 16 * 300, True, False, 0, 0)
                self.assertGreater(g["region_bytes"], (mib << 20) - 64 * 4096)

    def test_window_requirement_counts_framed_bytes(self):
        self.assertEqual(b1.window_requirement("mesh", 2 * 496, 2),
                         2 * 1 * 512)
        self.assertEqual(b1.window_requirement("oneshot", 496, 3), 2 * 512)


class TestHostReference(CustomTestCase):
    """The host twin of the device hash, and the properties the fix rests on."""

    def setUp(self):
        import random

        rnd = random.Random(1022)
        self.payload = bytes(rnd.getrandbits(8) for _ in range(16 * 100))
        self.n4 = 100
        self.salt = fr.salt(fr.KIND_RING, 3, 1)

    def test_fresh_slot_verifies(self):
        slot = fr.frame_slot(self.payload, 7, self.salt)
        self.assertEqual(len(slot), fr.framed_bytes(len(self.payload)))
        self.assertEqual(fr.verify_slot(slot, self.n4, 7, self.salt), [])

    def test_old_generation_is_rejected_even_with_identical_data(self):
        old = fr.frame_slot(self.payload, 6, self.salt)
        self.assertEqual(fr.verify_slot(old, self.n4, 7, self.salt),
                         list(range(4)))

    def test_every_single_stale_packet_is_rejected(self):
        """One stale packet anywhere -> its tile is rejected, deterministically."""
        new = bytearray(fr.frame_slot(self.payload, 7, self.salt))
        old = fr.frame_slot(bytes(len(self.payload)), 6, self.salt)
        for d in range(self.n4):
            t, lane = divmod(d, fr.FRAME_DATA)
            pos = (t * fr.FRAME_LANES + lane) * 16
            torn = bytearray(new)
            torn[pos:pos + 16] = old[pos:pos + 16]
            if bytes(torn) == bytes(new):
                continue
            self.assertEqual(fr.verify_slot(bytes(torn), self.n4, 7, self.salt), [t],
                             msg=f"packet {d}")

    def test_torn_packet_is_rejected(self):
        new = bytearray(fr.frame_slot(self.payload, 7, self.salt))
        new[5 * 16 + 4] ^= 0x01
        self.assertEqual(fr.verify_slot(bytes(new), self.n4, 7, self.salt), [0])

    def test_untouched_zero_slot_is_rejected(self):
        zero = bytes(fr.framed_bytes(len(self.payload)))
        self.assertEqual(fr.verify_slot(zero, self.n4, 7, self.salt), [0, 1, 2, 3])

    def test_foreign_slot_stream_is_rejected(self):
        other = fr.frame_slot(self.payload, 7, fr.salt(fr.KIND_RING, 3, 2))
        self.assertEqual(fr.verify_slot(other, self.n4, 7, self.salt), [0, 1, 2, 3])


class TestKernelSourceInvariants(CustomTestCase):
    def setUp(self):
        self.raw = _strip_comments(_raw_cuda_src())

    def test_no_slot_is_read_outside_the_framed_receive(self):
        """readV4 is the slot read; it may appear only in its own definition.

        Every receive path goes through frRecvTile (spliced in from
        barlink_bar1_frame.FRAME_CUDA_SRC), which consumes nothing unverified.
        A bare readV4 on a receive slot would be the pre-22.10. protocol back.
        """
        uses = [m.start() for m in re.finditer(r"\breadV4\s*\(", self.raw)]
        self.assertEqual(len(uses), 1, msg="readV4 used outside its definition")
        self.assertIn("readB(", _raw_cuda_src())   # kept, but unused:
        self.assertNotIn("= readB(", self.raw)
        self.assertNotIn("readB(q", self.raw)

    def test_every_kernel_receive_is_verified(self):
        for name, n in (("bar1_mesh_kernel", 2), ("bar1_oneshot_kernel", 1),
                        ("bar1_ring_kernel", 2)):
            body = self.raw[self.raw.index(f"__global__ void {name}"):]
            body = body[:body.index("\n}\n")]
            self.assertEqual(body.count("FR_CHECK()"), n, msg=name)
        a2a = self.raw[self.raw.index("__global__ void bar1_a2a_kernel"):]
        self.assertIn("frRecvTile(", a2a[:a2a.index("\n}\n")])
        self.assertIn("frSendTile(", a2a[:a2a.index("\n}\n")])

    def test_the_frame_source_is_spliced_exactly_once(self):
        from flliper.srt.distributed.device_communicators import barlink_bar1_ext as ext

        self.assertEqual(ext._CUDA_SRC.count("__device__ __forceinline__ bool frRecvTile"), 1)
        self.assertNotIn("@@BARLINK_FRAME_SRC@@", ext._CUDA_SRC)

    def test_the_frame_receive_has_deadline_and_host_probe(self):
        src = _strip_comments(fr.FRAME_CUDA_SRC)
        body = src[src.index("bool frRecvTile"):]
        self.assertIn("> capCycles", body)
        self.assertIn("abortHost != nullptr", body)
        self.assertIn("& 1023u", body)
        self.assertIn("__shfl_sync(FR_FULL, stop, 0)", body,
                      msg="give-up must be warp-uniform")
        self.assertIn("frReadMmio(q)", body, msg="retries must bypass the caches")
        self.assertIn("__nanosleep(", body, msg="retries must back off")
        self.assertIn("if (attempt == 0) v = readV4(q);", body)

    def test_mutants_are_compile_time_and_off_by_default(self):
        self.assertIn("#define FRAME_MUTANT 0", fr.FRAME_CUDA_SRC)
        src = _EXT.read_text(encoding="utf-8")
        self.assertNotIn("FRAME_MUTANT=", src)


class TestHostSideStatusAndPipe(CustomTestCase):
    def test_status_three_is_an_abort(self):
        src = _BAR1.read_text(encoding="utf-8")
        self.assertEqual(src.count("not in (1, 2, _frame.STATUS_FRAME)"), 2)
        self.assertEqual(fr.STATUS_FRAME, 3)
        self.assertIn("#define FR_STATUS 3u", fr.FRAME_CUDA_SRC)

    def test_the_unframed_pipe_is_declined_by_name(self):
        src = _BAR1.read_text(encoding="utf-8")
        i = src.index('self.pipe_on = os.environ.get("FLLIPER_BARLINK_BAR1_PIPE"')
        tail = src[i:i + 2500]
        self.assertIn("DECLINED", tail)
        self.assertIn("self.pipe_on = False", tail)

    def test_both_launchers_pass_the_stride(self):
        src = _BAR1.read_text(encoding="utf-8")
        self.assertEqual(src.count('int(self._geo["slot_stride"]),'), 2)


# ===========================================================================
# GPU (one card) and MUTANTS
# ===========================================================================

_PRELUDE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
using u64 = unsigned long long;
__device__ __forceinline__ uint4 readV4(const void *p)
{
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void writeV4(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
"""

_HARNESS = r"""
__global__ void packK(const uint4 *in, uint4 *slot, int n4, unsigned salt, u64 round)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
    const int lane = tid & 31;
    const long long nT = ((long long)n4 + FR_DATA - 1) / FR_DATA;
    for (long long t = tid >> 5; t < nT; t += nth >> 5) {
        const long long d = t * FR_DATA + lane;
        const bool valid = lane < FR_DATA && d < n4;
        const uint4 v = valid ? in[d] : make_uint4(0u, 0u, 0u, 0u);
        frSendTile(slot, t, lane, valid, v, frKey(round, salt, t), round);
    }
}
__global__ void verifyK(const uint4 *slot, uint4 *out, int n4, unsigned salt, u64 round,
                        u64 cap, int *status)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
    const int lane = tid & 31;
    const long long nT = ((long long)n4 + FR_DATA - 1) / FR_DATA;
    for (long long t = tid >> 5; t < nT; t += nth >> 5) {
        const long long d = t * FR_DATA + lane;
        const bool valid = lane < FR_DATA && d < n4;
        uint4 v;
        if (!frRecvTile(slot, t, lane, valid, frKey(round, salt, t), round, &v, cap, nullptr)) {
            if (lane == 0) atomicExch(status, 1);
            return;
        }
        if (valid) out[d] = v;
    }
}
// The adversary: after `delay` cycles copy [lo, hi) packets of `src` into
// `dst` with ordinary stores, then fence.
__global__ void lateCopyK(uint4 *dst, const uint4 *src, long long lo, long long hi,
                          long long delay)
{
    if (threadIdx.x == 0) { long long t0 = clock64(); while (clock64() - t0 < delay) {} }
    __syncthreads();
    for (long long j = lo + threadIdx.x; j < hi; j += blockDim.x) writeV4(dst + j, src[j]);
    __threadfence_system();
}

void pack(at::Tensor in, at::Tensor slot, int64_t n4, int64_t salt, int64_t round)
{
    packK<<<4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const uint4 *)in.data_ptr(), (uint4 *)slot.data_ptr(), (int)n4, (unsigned)salt, (u64)round);
}
void verify(at::Tensor slot, at::Tensor out, int64_t n4, int64_t salt, int64_t round,
            int64_t cap, at::Tensor status, int64_t stream)
{
    verifyK<<<2, 128, 0, (cudaStream_t)stream>>>(
        (const uint4 *)slot.data_ptr(), (uint4 *)out.data_ptr(), (int)n4, (unsigned)salt,
        (u64)round, (u64)cap, (int *)status.data_ptr());
}
void late_copy(at::Tensor dst, at::Tensor src, int64_t lo, int64_t hi, int64_t delay, int64_t stream)
{
    lateCopyK<<<1, 256, 0, (cudaStream_t)stream>>>(
        (uint4 *)dst.data_ptr(), (const uint4 *)src.data_ptr(), lo, hi, delay);
}
"""

_CPP = """
void pack(at::Tensor in, at::Tensor slot, int64_t n4, int64_t salt, int64_t round);
void verify(at::Tensor slot, at::Tensor out, int64_t n4, int64_t salt, int64_t round,
            int64_t cap, at::Tensor status, int64_t stream);
void late_copy(at::Tensor dst, at::Tensor src, int64_t lo, int64_t hi, int64_t delay, int64_t stream);
"""

_BUILT = {}


def _gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _build(mutant: int):
    if mutant in _BUILT:
        return _BUILT[mutant]
    import torch
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    ext = load_inline(
        name=f"barlink_frame_harness_m{mutant}_{major}{minor}",
        cpp_sources=_CPP,
        cuda_sources=_PRELUDE + f"#define FRAME_MUTANT {mutant}\n" + fr.FRAME_CUDA_SRC + _HARNESS,
        functions=["pack", "verify", "late_copy"],
        extra_cuda_cflags=[f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        verbose=False,
    )
    _BUILT[mutant] = ext
    return ext


def _scenarios(ext) -> dict:
    """Outcome of every scenario: True = the receiver behaved correctly."""
    import torch

    dev = torch.device("cuda")
    clk = int(torch.cuda.get_device_properties(0).clock_rate) * 1000  # Hz
    n4 = 31 * 7 + 5                      # 8 tiles, the last one partial
    salt = fr.salt(fr.KIND_MESH, 0, 1)
    rnd = 1000
    g = torch.Generator(device="cpu").manual_seed(1022)
    data = torch.randint(-2**31, 2**31 - 1, (n4 * 4,), dtype=torch.int32, generator=g).to(dev)
    stale = torch.randint(-2**31, 2**31 - 1, (n4 * 4,), dtype=torch.int32, generator=g).to(dev)
    nslot = fr.framed_packets(n4) * 4
    res = {}

    def run_verify(slot, round_=rnd, salt_=salt, cap_s=0.02, stream=None):
        out = torch.zeros(n4 * 4, dtype=torch.int32, device=dev)
        st = torch.zeros(1, dtype=torch.int32, device=dev)
        s = stream or torch.cuda.current_stream()
        # the zero-fills above ran on the current stream; a side stream must
        # not start the verify before them (torch side streams are non-blocking)
        s.wait_stream(torch.cuda.current_stream())
        ext.verify(slot, out, n4, salt_, round_, int(cap_s * clk), st, s.cuda_stream)
        return out, st

    fresh = torch.zeros(nslot, dtype=torch.int32, device=dev)
    ext.pack(data, fresh, n4, salt, rnd)
    torch.cuda.synchronize()
    ref = fr.frame_slot(data.cpu().numpy().tobytes(), rnd, salt)
    res["device_hash_matches_host_reference"] = fresh.cpu().numpy().tobytes() == ref

    out, st = run_verify(fresh)
    torch.cuda.synchronize()
    res["fresh_accepted"] = int(st.item()) == 0 and torch.equal(out, data)

    # old generation, SAME data: only the round tells them apart
    old = torch.zeros_like(fresh)
    ext.pack(data, old, n4, salt, rnd - 1)
    out, st = run_verify(old)
    torch.cuda.synchronize()
    res["old_generation_rejected"] = int(st.item()) == 1

    # old generation, DIFFERENT data (the stale slot of the previous round)
    old2 = torch.zeros_like(fresh)
    ext.pack(stale, old2, n4, salt, rnd - 1)
    out, st = run_verify(old2)
    torch.cuda.synchronize()
    res["stale_slot_rejected"] = int(st.item()) == 1 and not torch.equal(out, stale)

    # trailer new, one data packet stale -- in the first, a middle and the last tile
    ok = True
    for d in (0, 30, 31 * 3 + 17, n4 - 1):
        t, lane = divmod(d, fr.FRAME_DATA)
        torn = fresh.clone()
        p = (t * fr.FRAME_LANES + lane) * 4
        torn[p:p + 4] = old2[p:p + 4]
        out, st = run_verify(torn)
        torch.cuda.synchronize()
        ok = ok and int(st.item()) == 1
    res["one_stale_packet_rejected"] = ok

    # untouched slot (zeros) -- the 22.10. specimen
    out, st = run_verify(torch.zeros_like(fresh))
    torch.cuda.synchronize()
    res["untouched_slot_rejected"] = int(st.item()) == 1

    # a tile written for another slot stream in the same round
    out, st = run_verify(fresh, salt_=fr.salt(fr.KIND_MESH, 0, 2))
    torch.cuda.synchronize()
    res["foreign_stream_rejected"] = int(st.item()) == 1

    # FLAG BEFORE DATA: the slot holds the previous round; the adversary first
    # writes ONLY the trailers of the new round, 30 ms later the data. The
    # receiver is already spinning. It must end with the NEW data, not before.
    slot = old2.clone()
    s_rx = torch.cuda.Stream()
    s_tx = torch.cuda.Stream()
    # Warm the adversary kernel first: under CUDA lazy module loading its FIRST
    # launch waits for the running (spinning) verify kernel, which would turn
    # the race into a serial run and the scenario into a timeout (measured).
    ext.late_copy(slot, fresh, 0, 0, 0, s_tx.cuda_stream)
    torch.cuda.synchronize()
    out, st = run_verify(slot, cap_s=2.0, stream=s_rx)
    tiles = -(-n4 // fr.FRAME_DATA)
    for t in range(tiles):     # trailers first
        lo = t * fr.FRAME_LANES + fr.FRAME_DATA
        ext.late_copy(slot, fresh, lo, lo + 1, int(0.002 * clk), s_tx.cuda_stream)
    for t in range(tiles):     # data 30 ms later
        lo = t * fr.FRAME_LANES
        ext.late_copy(slot, fresh, lo, lo + fr.FRAME_DATA, int(0.030 * clk) if t == 0 else 0,
                      s_tx.cuda_stream)
    torch.cuda.synchronize()
    res["trailer_before_data_waits_for_data"] = int(st.item()) == 0 and torch.equal(out, data)
    res["_tbd_detail"] = (int(st.item()), bool(torch.equal(out, data)),
                          bool(torch.equal(out, stale)), int((out == 0).sum().item()))
    return res


@unittest.skipUnless(_gpu(), "needs one CUDA card (run inside a gpuq window)")
class TestFrameOnDevice(CustomTestCase):
    def test_real_protocol_passes_every_scenario(self):
        res = _scenarios(_build(0))
        bad = [k for k, v in res.items() if not k.startswith("_") and not v]
        self.assertFalse(bad, msg=f"real protocol failed: {bad} ({res})")


@unittest.skipUnless(_gpu(), "needs one CUDA card (run inside a gpuq window)")
class TestFrameMutants(CustomTestCase):
    """Each mutant must be KILLED by a scenario of the danger direction."""

    def test_old_generation_mutant_is_killed(self):
        res = _scenarios(_build(1))
        self.assertFalse(res["old_generation_rejected"],
                         msg="mutant 1 (round ignored) survived: the old-generation "
                             "scenario does not guard the round")

    def test_no_check_mutant_is_killed(self):
        res = _scenarios(_build(2))
        killed = [k for k in ("trailer_before_data_waits_for_data", "stale_slot_rejected",
                              "untouched_slot_rejected") if not res[k]]
        self.assertTrue(killed, msg=f"mutant 2 (no check) survived every scenario: {res}")


if __name__ == "__main__":
    unittest.main()

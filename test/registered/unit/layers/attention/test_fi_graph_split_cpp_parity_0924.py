"""fi_graph_split against flashinfer 0.6.14's OWN planner -- at the desk, no GPU.

The prefill planner is host code: PrefillPlan (scheduler.cuh) reaches CUDA only
through cudaGetDevice / cudaDeviceGetAttribute (the SM count, the compute
capability for the CTA tile) and one cudaMemcpyAsync of the int arrays. This
test compiles the venv's own header with exactly those three calls redirected
to host stubs (nvcc, ~4 s, ~0.4 GB, address space capped at 4 GiB, into a
temp dir; flashinfer is LOCATED, never imported, nothing is written under
~/.cache) and compares with the Python port array by array:

* the three shapes of the GPU parity test, and why its first window run
  (2026-09-24 20:23Z) failed: at 98k / 7 chunks the eager REFERENCE plan asks
  for 506 MiB of split partials (the 0.6.14 GQA over-reservation, #5177) and
  the test gave it 384 MiB, so the C++ planner raised "Buffer overflow ...
  batch_prefill_tmp_v" before any array was compared -- no array differs;
* stock_eager_split_float_bytes == the planner's exact refusal boundary (the
  GPU test now sizes its reference workspace with it);
* the 27B P path (512 / 16 / 1 new rows, prefix up to 256k, 1..7 chunks) and
  seeded random multi-request batches (empty sentinel slots, gqa 1..8,
  head_dim 64..256, uneven chunks);
* the graph-mode stock plan that layout_from_stock reads (5090 and 3080), the
  valid-item mask our region reproduces, and the one form our arrays do NOT
  cover -- a sliding window, which flashinfer_contract_ok refuses.
"""

import importlib.util
import json
import math
import os
import random
import resource
import shutil
import subprocess
import tempfile
import unittest

from sglang.srt.layers.attention import fi_graph_split as G
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QO, HQ, HKV, HD = 512, 24, 4, 256
WS_384 = 384 * 1024 * 1024  # the GPU test's (and the server's) float workspace

_HARNESS = r"""
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

static int g_num_sm = 170;
static int g_cc_major = 12;
static cudaError_t host_get_device(int* d) { *d = 0; return cudaSuccess; }
static cudaError_t host_device_get_attribute(int* v, cudaDeviceAttr a, int) {
  if (a == cudaDevAttrMultiProcessorCount) { *v = g_num_sm; }
  else if (a == cudaDevAttrComputeCapabilityMajor) { *v = g_cc_major; }
  else if (a == cudaDevAttrComputeCapabilityMinor) { *v = 0; }
  else { std::fprintf(stderr, "unexpected cudaDeviceGetAttribute(%d)\n", int(a)); std::abort(); }
  return cudaSuccess;
}
static cudaError_t host_memcpy_async(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
  std::memcpy(dst, src, n);
  return cudaSuccess;
}
// after the CUDA headers (include guards keep their declarations intact)
#define cudaGetDevice host_get_device
#define cudaDeviceGetAttribute host_device_get_attribute
#define cudaMemcpyAsync host_memcpy_async
#include <flashinfer/attention/scheduler.cuh>

static std::vector<int32_t> csv(const char* s) {
  std::vector<int32_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(int32_t(std::stoll(tok)));
  return out;
}
template <typename T>
static void dump(const char* name, const T* p, size_t n, bool last = false) {
  std::printf("\"%s\":[", name);
  for (size_t i = 0; i < n; ++i) std::printf(i ? ",%lld" : "%lld", (long long)p[i]);
  std::printf("]%s", last ? "" : ",");
}
// argv: num_sm cc_major graph hq hkv head_dim page fixed_split disable_split
//       float_ws_bytes total_rows(0=qo_indptr[-1]) qo_indptr kv_indptr_pages window_left
int main(int argc, char** argv) {
  if (argc != 15) { std::fprintf(stderr, "14 arguments, got %d\n", argc - 1); return 2; }
  g_num_sm = std::atoi(argv[1]);
  g_cc_major = std::atoi(argv[2]);
  const bool graph = std::atoi(argv[3]) != 0;
  const uint32_t hq = std::atoi(argv[4]), hkv = std::atoi(argv[5]), hd = std::atoi(argv[6]);
  const uint32_t page = std::atoi(argv[7]);
  const int32_t fixed = std::atoi(argv[8]);
  const bool disable = std::atoi(argv[9]) != 0;
  const size_t float_ws = std::strtoull(argv[10], nullptr, 10);
  uint32_t rows = std::atoi(argv[11]);
  std::vector<int32_t> qo = csv(argv[12]), kv = csv(argv[13]);
  const int32_t window_left = std::atoi(argv[14]);
  const uint32_t batch = qo.size() - 1;
  if (!rows) rows = qo.back();
  const size_t int_ws = 8u << 20;  // the wrapper's int workspace
  std::vector<uint8_t> int_buf(int_ws, 0), pinned(int_ws, 0);
  void* float_buf = reinterpret_cast<void*>(uintptr_t(1) << 40);  // offsets only
  flashinfer::PrefillPlanInfo info;
  try {
    cudaError_t st = flashinfer::PrefillPlan<int32_t>(
        float_buf, float_ws, int_buf.data(), pinned.data(), int_ws, info, qo.data(), kv.data(),
        rows, batch, hq, hkv, hd, hd, page, graph, 2, window_left, fixed, disable, 0, nullptr);
    if (st != cudaSuccess) { std::printf("{\"error\":\"cuda status %d\"}\n", int(st)); return 0; }
  } catch (const std::exception& e) {
    std::string msg = e.what();
    for (auto& c : msg) if (c == '"' || c == '\\' || c == '\n') c = ' ';
    std::printf("{\"error\":\"%s\"}\n", msg.c_str());
    return 0;
  }
  const size_t padded = info.padded_batch_size;
  auto at = [&](int64_t off) { return reinterpret_cast<const int32_t*>(int_buf.data() + off); };
  std::printf("{");
  std::vector<int64_t> vec = info.ToVector();
  dump("plan_info", vec.data(), vec.size());
  dump("request_indices", at(info.request_indices_offset), padded);
  dump("qo_tile_indices", at(info.qo_tile_indices_offset), padded);
  dump("kv_tile_indices", at(info.kv_tile_indices_offset), padded);
  dump("o_indptr", at(info.o_indptr_offset), batch + 1);
  dump("kv_chunk_size", at(info.kv_chunk_size_ptr_offset), 1);
  if (info.enable_cuda_graph) dump("total_num_rows_dev", at(info.total_num_rows_offset), 1);
  if (info.split_kv) {
    dump("merge_indptr", at(info.merge_indptr_offset), info.total_num_rows + 1);
    dump("block_valid_mask", int_buf.data() + info.block_valid_mask_offset, padded);
  }
  const int64_t f[2] = {info.v_offset, info.s_offset};
  dump("float_offsets", f, 2, true);
  std::printf("}\n");
  return 0;
}
"""


def _flashinfer_data():
    spec = importlib.util.find_spec("flashinfer")  # locates the package, does not import it
    if spec is None or not spec.submodule_search_locations:
        return None
    data = os.path.join(list(spec.submodule_search_locations)[0], "data")
    if not os.path.isfile(os.path.join(data, "include/flashinfer/attention/scheduler.cuh")):
        return None
    return data


def _nvcc():
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    for cand in (shutil.which("nvcc"), os.path.join(cuda_home, "bin", "nvcc")):
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


def _cap_address_space():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))


def _first_diff(got, want):
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return i
    return None if len(got) == len(want) else min(len(got), len(want))


class TestCppPlannerParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The harness calls PrefillPlan with the signature of the version
        # fi_graph_split mirrors; under any other flashinfer the module stands
        # down (flashinfer_contract_ok), so parity with that header is moot.
        from importlib.metadata import PackageNotFoundError, version

        try:
            fi_ver = version("flashinfer-python")
        except PackageNotFoundError:
            fi_ver = ""
        if not fi_ver.startswith(G.FLASHINFER_VERSION):
            raise unittest.SkipTest(
                "flashinfer %r installed, fi_graph_split mirrors %s (module stands down)"
                % (fi_ver, G.FLASHINFER_VERSION)
            )
        data, nvcc = _flashinfer_data(), _nvcc()
        if data is None or nvcc is None:
            raise unittest.SkipTest("needs nvcc and the flashinfer headers (data=%s nvcc=%s)" % (data, nvcc))
        cls.tmp = tempfile.mkdtemp(prefix="fi_plan_host_")
        src = os.path.join(cls.tmp, "fi_plan_host.cu")
        cls.exe = os.path.join(cls.tmp, "fi_plan_host")
        with open(src, "w") as f:
            f.write(_HARNESS)
        inc = [
            "-I" + os.path.join(data, "cccl", "cub"),
            "-I" + os.path.join(data, "cccl", "libcudacxx", "include"),
            "-I" + os.path.join(data, "cccl", "thrust"),
            "-I" + os.path.join(data, "include"),
        ]
        cmd = [nvcc, "-std=c++17", "-O1", "--expt-relaxed-constexpr", "-arch=sm_80", *inc, "-o", cls.exe, src]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=cls.tmp)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, preexec_fn=_cap_address_space)
        if r.returncode:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            # the header no longer builds with three stubs: the contract moved
            raise AssertionError("flashinfer's planner header does not build host-only:\n" + r.stderr[-4000:])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def plan(self, qo_lens, kv_lens, *, chunk=-1, graph=False, float_ws=1 << 40, num_sm=170,
             cc_major=12, hq=HQ, hkv=HKV, hd=HD, window_left=-1):
        qi, ki = [0], [0]
        for q, k in zip(qo_lens, kv_lens):
            qi.append(qi[-1] + q)
            ki.append(ki[-1] + k)
        argv = [self.exe, num_sm, cc_major, int(graph), hq, hkv, hd, 1, chunk, 0, float_ws, 0,
                ",".join(map(str, qi)), ",".join(map(str, ki)), window_left]
        r = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=60,
                           env={"CUDA_VISIBLE_DEVICES": ""}, check=True)
        return json.loads(r.stdout)

    def diffs(self, d, qo_lens, kv_lens, chunk, gqa):
        """Every array's first difference, "" when identical."""
        info = d["plan_info"]
        want = G.split_arrays(qo_lens, kv_lens, gqa=gqa, cta_tile_q=info[3], kv_chunk=chunk)
        n = len(want["request_indices"])
        out = []
        if info[0] != n:
            out.append("work items cpp %d ours %d" % (info[0], n))
        for name in ("request_indices", "qo_tile_indices", "kv_tile_indices", "o_indptr", "kv_chunk_size"):
            got = d[name][:n] if name.endswith("_indices") else d[name]
            i = _first_diff(got, want[name])
            if i is not None:
                out.append("%s[%d] cpp %s ours %s" % (name, i, got[max(0, i - 2): i + 3], want[name][max(0, i - 2): i + 3]))
        if info[14]:
            i = _first_diff(d["merge_indptr"], want["merge_indptr"])
            if i is not None:
                out.append("merge_indptr[%d]" % i)
            if d["block_valid_mask"] != [1] * n:
                out.append("block_valid_mask not all valid")
        elif any(want["kv_tile_indices"]):
            out.append("cpp did not split, ours did")
        return "; ".join(out)

    def test_the_gpu_tests_three_shapes_are_identical(self):
        for kv, n in ((36000 + QO, 5), (98304 + QO, 7), (2048 + QO, 4)):
            chunk = math.ceil(kv / n)
            d = self.plan([QO], [kv], chunk=chunk)
            self.assertNotIn("error", d, (kv, n))
            self.assertEqual(self.diffs(d, [QO], [kv], chunk, HQ // HKV), "", (kv, n, d["plan_info"]))
            self.assertEqual(d["plan_info"][0], 48 * n)

    def test_the_window_failure_was_the_reference_plans_workspace(self):
        kv = 98304 + QO
        d = self.plan([QO], [kv], chunk=math.ceil(kv / 7), float_ws=WS_384)
        self.assertIn("batch_prefill_tmp_v with size 528482304", d.get("error", ""))
        self.assertEqual(G.stock_eager_split_float_bytes(HQ, 48 * 7, 64, HD), 528482304 + 2064384)
        ok = self.plan([QO], [36000 + QO], chunk=math.ceil((36000 + QO) / 5), float_ws=WS_384)
        self.assertNotIn("error", ok)  # the first shape fit (379 MB), so it was compared -- and equal

    def test_reservation_formula_is_the_planners_exact_boundary(self):
        rnd = random.Random(7)
        for hkv, gqa, hd, qo, kv, n in [(4, 6, 256, 512, 98816, 7), (4, 6, 256, 16, 50000, 3)] + [
            (rnd.choice([1, 2, 4, 8]), rnd.choice([1, 2, 4, 6, 8]), rnd.choice([64, 128, 256]),
             rnd.randint(1, 1024), rnd.randint(2048, 200000), rnd.randint(2, 8)) for _ in range(6)
        ]:
            kv = max(kv, qo)
            chunk = math.ceil(kv / n)
            kw = dict(chunk=chunk, hq=hkv * gqa, hkv=hkv, hd=hd)
            info = self.plan([qo], [kv], **kw)["plan_info"]
            if not info[14]:
                continue
            need = G.stock_eager_split_float_bytes(hkv * gqa, info[0], info[3], hd)
            self.assertNotIn("error", self.plan([qo], [kv], float_ws=need, **kw))
            self.assertIn("batch_prefill_tmp_s", self.plan([qo], [kv], float_ws=need - 1, **kw).get("error", ""))

    def test_p_path_every_depth_and_chunk_count(self):
        bad = []
        for qo in (512, 16, 1):
            for prefix in range(0, 262144 + 1, 16384):
                kv = prefix + qo
                for n in range(1, 8):
                    chunk = max(1, math.ceil(kv / n))
                    why = self.diffs(self.plan([qo], [kv], chunk=chunk), [qo], [kv], chunk, HQ // HKV)
                    if why:
                        bad.append((qo, kv, n, why))
        self.assertEqual(bad, [])

    def test_random_multi_request_batches(self):
        rnd = random.Random(924)
        bad = []
        for _ in range(300):
            hkv, gqa, hd = rnd.choice([1, 2, 4, 8]), rnd.choice([1, 2, 4, 6, 8]), rnd.choice([64, 128, 256])
            qo = [rnd.choice([0, 1, 2, 7, 16, 63, 64, 65, 255, 511, 512, rnd.randint(0, 2048)])
                  for _ in range(rnd.randint(1, 6))]
            qo[0] = qo[0] or 1
            kv = [q + rnd.choice([0, 1, 511, 4096, rnd.randint(0, 300000)]) for q in qo]
            top = max(max(kv), 1)
            lo = -(-top // 16)  # <= 16 chunks per request keeps the int arrays in 8 MiB
            chunk = rnd.choice([lo, -(-top // rnd.randint(1, 16)), rnd.randint(lo, top + 10)])
            d = self.plan(qo, kv, chunk=chunk, hq=hkv * gqa, hkv=hkv, hd=hd)
            why = d.get("error") or self.diffs(d, qo, kv, chunk, gqa)
            if why:
                bad.append((qo, kv, hkv * gqa, hkv, hd, chunk, why))
        self.assertEqual(bad, [])

    def test_graph_stock_plan_feeds_the_capture_layout(self):
        d = self.plan([QO], [4096 + QO], graph=True, float_ws=WS_384)
        info = d["plan_info"]
        # padded = max(2 x 170 / 4 = 85, 48 tiles); graph plans always split_kv
        self.assertEqual([info[0], info[1], info[3], info[13], info[14]], [85, QO, 64, 1, 1])
        self.assertEqual(d["total_num_rows_dev"], [QO])
        self.assertEqual(d["block_valid_mask"], [1] * 48 + [0] * 37)
        lay, why = G.layout_from_stock(
            info, slots=1, max_chunks=7, num_qo_heads=HQ, num_kv_heads=HKV, head_dim_vo=HD,
            out_bytes=2, int_workspace_bytes=8 << 20, float_workspace_bytes=WS_384)
        self.assertIsNotNone(lay, why)
        self.assertEqual(lay.padded, 48 * 7)
        kv = 98304 + QO
        a = G.split_arrays([QO], [kv], gqa=HQ // HKV, cta_tile_q=info[3], kv_chunk=math.ceil(kv / 5))
        region = G.int_region_bytes(lay, a)
        at = lay.offsets["block_valid_mask"] - lay.int_base
        self.assertEqual(list(region[at: at + lay.padded]), [1] * 240 + [0] * (lay.padded - 240))
        # the tiny bucket's capture (16 rows) and the 3080 (68 SMs: padded = max(34, 48))
        self.assertEqual(self.plan([16], [4096 + 16], graph=True, float_ws=WS_384)["plan_info"][3], 64)
        self.assertEqual(self.plan([QO], [4096 + QO], graph=True, float_ws=WS_384, num_sm=68, cc_major=8)["plan_info"][0], 48)

    def test_sliding_window_is_the_form_these_arrays_do_not_cover(self):
        kv = 36000 + QO
        chunk = math.ceil(kv / 5)
        d = self.plan([QO], [kv], chunk=chunk, window_left=4096)
        ours = G.split_arrays([QO], [kv], gqa=HQ // HKV, cta_tile_q=d["plan_info"][3], kv_chunk=chunk)
        # the planner (and the kernel) count chunks over window + CTA_TILE_Q = 4160 -> 1
        self.assertEqual(d["plan_info"][0], 48)
        self.assertEqual(len(ours["request_indices"]), 240)


if __name__ == "__main__":
    unittest.main()

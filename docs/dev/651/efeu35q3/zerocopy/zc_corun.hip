// efeu-TP14: co-run bandwidth -- iGPU and all CPU cores stream DIFFERENT halves of
// one hipHostMalloc'd buffer at the same time. Answers: does a CPU share add
// bandwidth on this single DDR5 bus, or does it only take it from the iGPU?
//   HSA_OVERRIDE_GFX_VERSION=11.0.0 OMP_NUM_THREADS=8 ./zc_corun [MiB] [seconds]
#include <hip/hip_runtime.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
__global__ void gread(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) atomicAdd(out, 1ull);
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
  size_t half = (size_t)(argc > 1 ? atoi(argv[1]) : 1024) << 20;
  double secs = argc > 2 ? atof(argv[2]) : 4.0;
  char* p;
  (void)hipHostMalloc((void**)&p, 2 * half, hipHostMallocNonCoherent);
  memset(p, 1, 2 * half);
  unsigned long long* d;
  (void)hipMalloc(&d, 8);
  for (int mode = 0; mode < 3; ++mode) {  // 0 gpu only, 1 cpu only, 2 both
    std::atomic<bool> stop{false};
    double gbytes = 0, cbytes = 0, t0 = now();
    std::thread gt([&] {
      if (mode == 1) return;
      while (!stop) {
        gread<<<768, 256>>>((const uint4*)p, half / 16, d);
        (void)hipDeviceSynchronize();
        gbytes += half;
      }
    });
    std::thread ct([&] {
      if (mode == 0) return;
      const uint64_t* q = (const uint64_t*)(p + half);
      size_t n = half / 8;
      while (!stop) {
        uint64_t acc = 0;
#pragma omp parallel for reduction(^ : acc) schedule(static)
        for (size_t i = 0; i < n; ++i) acc ^= q[i];
        if (acc == 42) printf(" ");
        cbytes += half;
      }
    });
    while (now() - t0 < secs) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop = true;
    gt.join();
    ct.join();
    double dt = now() - t0;
    printf("%-9s GPU %6.1f GB/s  CPU %6.1f GB/s  total %6.1f GB/s\n",
           mode == 0 ? "gpu-only" : mode == 1 ? "cpu-only" : "both", gbytes / dt / 1e9, cbytes / dt / 1e9,
           (gbytes + cbytes) / dt / 1e9);
  }
  return 0;
}

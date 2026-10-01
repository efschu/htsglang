// efeu-TP14 2026-10-01: can iGPU and CPU read the SAME bytes at full speed?
//
// One buffer per allocation kind, streamed by (a) a GPU read kernel and (b) all
// CPU cores (OpenMP, AVX-512 friendly 64-bit sums). Reports GB/s, best of N.
//   hipMalloc            : the default device allocation (GTT on this APU)
//   hipHostMalloc coh    : pinned host, fine-grained (coherent, snooped)
//   hipHostMalloc noncoh : pinned host, coarse-grained (hipHostMallocNonCoherent)
//   malloc+hipHostRegister: ordinary pageable memory registered for the GPU
//   hipMallocManaged     : managed / unified
//   malloc (CPU only)    : reference CPU bandwidth
// CPU access to hipMalloc is attempted only if the pointer is host-accessible
// (hipPointerGetAttributes), otherwise reported "n/a".
//
//   hipcc -O3 --offload-arch=gfx1100 -fopenmp zc_bench.hip -o zc_bench
//   HSA_OVERRIDE_GFX_VERSION=11.0.0 OMP_NUM_THREADS=8 ./zc_bench [MiB]
#include <hip/hip_runtime.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("ERR %s: %s\n", #x, hipGetErrorString(e)); return nullptr; } } while (0)

__global__ void gread(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) atomicAdd(out, 1ull);
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double gpu_bw(void* p, size_t bytes, unsigned long long* d_out) {
  size_t n = bytes / sizeof(uint4);
  double best = 0;
  for (int r = 0; r < 6; ++r) {
    hipDeviceSynchronize();
    double t0 = now();
    gread<<<12 * 64, 256>>>((const uint4*)p, n, d_out);
    hipDeviceSynchronize();
    double dt = now() - t0;
    if (r > 0 && bytes / dt / 1e9 > best) best = bytes / dt / 1e9;
  }
  return best;
}

static double cpu_bw(void* p, size_t bytes) {
  size_t n = bytes / 8;
  const uint64_t* q = (const uint64_t*)p;
  double best = 0;
  for (int r = 0; r < 6; ++r) {
    uint64_t acc = 0;
    double t0 = now();
#pragma omp parallel for reduction(^ : acc) schedule(static)
    for (size_t i = 0; i < n; ++i) acc ^= q[i];
    double dt = now() - t0;
    if (acc == 42) printf(" ");
    if (r > 0 && bytes / dt / 1e9 > best) best = bytes / dt / 1e9;
  }
  return best;
}

static bool host_accessible(void* p) {
  hipPointerAttribute_t a;
  if (hipPointerGetAttributes(&a, p) != hipSuccess) return true;  // plain host memory
  return a.hostPointer != nullptr;
}

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? atoi(argv[1]) : 1024;
  size_t bytes = mib << 20;
  unsigned long long* d_out;
  hipMalloc(&d_out, 8);
  hipDeviceProp_t pr;
  hipGetDeviceProperties(&pr, 0);
  printf("device %s integrated=%d canMapHostMemory=%d pageableMemoryAccess=%d buffer=%zu MiB threads=%d\n",
         pr.gcnArchName, pr.integrated, pr.canMapHostMemory, pr.pageableMemoryAccess, mib, omp_get_max_threads());

  struct K { const char* name; int kind; };
  K kinds[] = {{"hipMalloc", 0}, {"hipHostMalloc coherent", 1}, {"hipHostMalloc noncoherent", 2},
               {"malloc+hipHostRegister", 3}, {"hipMallocManaged", 4}, {"malloc (cpu only)", 5}};
  for (auto& k : kinds) {
    void* p = nullptr;
    void* dp = nullptr;
    hipError_t e = hipSuccess;
    switch (k.kind) {
      case 0: e = hipMalloc(&p, bytes); dp = p; break;
      case 1: e = hipHostMalloc(&p, bytes, hipHostMallocDefault); if (!e) e = hipHostGetDevicePointer(&dp, p, 0); break;
      case 2: e = hipHostMalloc(&p, bytes, hipHostMallocNonCoherent); if (!e) e = hipHostGetDevicePointer(&dp, p, 0); break;
      case 3: p = aligned_alloc(4096, bytes); memset(p, 1, bytes); e = hipHostRegister(p, bytes, hipHostRegisterMapped);
              if (!e) e = hipHostGetDevicePointer(&dp, p, 0); break;
      case 4: e = hipMallocManaged(&p, bytes); dp = p; break;
      case 5: p = aligned_alloc(4096, bytes); memset(p, 1, bytes); break;
    }
    if (e != hipSuccess) { printf("%-28s alloc ERR %s\n", k.name, hipGetErrorString(e)); continue; }
    if (k.kind == 0) hipMemset(p, 1, bytes);
    else if (k.kind != 3 && k.kind != 5) memset(p, 1, bytes);
    double g = k.kind == 5 ? -1 : gpu_bw(dp, bytes, d_out);
    double c = (k.kind == 0 && !host_accessible(p)) ? -1 : cpu_bw(p, bytes);
    printf("%-28s GPU read %7.1f GB/s   CPU read %7.1f GB/s   same_ptr=%d\n", k.name, g, c, p == dp);
    switch (k.kind) {
      case 0: case 4: hipFree(p); break;
      case 1: case 2: hipHostFree(p); break;
      case 3: hipHostUnregister(p); free(p); break;
      case 5: free(p); break;
    }
  }
  return 0;
}

// "Kann die 5090 nativ INT4?" -- synthetic tensor-core throughput of
//   mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32   (INT4)
// against
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32   (INT8)
// in the same loop shape. Each warp keeps ILP independent accumulator chains
// busy; the result is written so the compiler cannot drop the loop.
// Build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a (5090) / -arch=sm_86 (3080).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define ILP 8

template <bool INT4>
__global__ void mma_loop(int iters, int32_t* out) {
  uint32_t a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 0x5a5a5a5au, a2 = a0 + 7, a3 = a0 * 3;
  uint32_t b0 = threadIdx.x ^ 0x33333333u, b1 = b0 + 11;
  int32_t c[ILP][4];
#pragma unroll
  for (int j = 0; j < ILP; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = j;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      if (INT4) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c[j][0]), "+r"(c[j][1]), "+r"(c[j][2]), "+r"(c[j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  int32_t s = 0;
#pragma unroll
  for (int j = 0; j < ILP; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <bool INT4>
double run(int sms, int warps_per_block, int blocks_per_sm, int iters) {
  int blocks = sms * blocks_per_sm, threads = 32 * warps_per_block;
  int32_t* out;
  cudaMalloc(&out, sizeof(int32_t) * blocks * threads);
  mma_loop<INT4><<<blocks, threads>>>(iters / 10, out);  // warmup
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  mma_loop<INT4><<<blocks, threads>>>(iters, out);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaError_t err = cudaGetLastError();
  cudaFree(out);
  if (err != cudaSuccess) {
    printf("  error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  double ops_per_mma = 2.0 * 16 * 8 * (INT4 ? 64 : 32);
  double total = ops_per_mma * ILP * (double)iters * blocks * warps_per_block;
  return total / (ms * 1e-3) / 1e12;
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int sms = p.multiProcessorCount;
  printf("device %s sm_%d%d SMs=%d clock=%d MHz\n", p.name, p.major, p.minor, sms, p.clockRate / 1000);
  int iters = 20000;
  for (int wpb : {4, 8}) {
    double t8 = run<false>(sms, wpb, 2, iters);
    double t4 = run<true>(sms, wpb, 2, iters);
    printf("warps/block=%d  INT8 m16n8k32: %.1f TOPS   INT4 m16n8k64: %.1f TOPS   ratio INT4/INT8 = %.2f\n", wpb, t8, t4,
           t8 > 0 ? t4 / t8 : 0.0);
  }
  return 0;
}

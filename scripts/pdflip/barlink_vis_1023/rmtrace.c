// rmtrace.c -- barlink-visibility-1023: LD_PRELOAD shim that logs how CUDA asks RM/UVM
// to map the peer-BAR1 IoMemory registration on the SENDER GPU, i.e. whether that
// GMMU mapping can be cached in the sender's L2 (vol=0) or not (vol=1).
//
// Why: RM decides the PTE volatile bit from (a) the OS descriptor's GPU cacheability
// (NVOS32 ALLOC_OS_DESCRIPTOR attr2 GPU_CACHEABLE -> osCreateOsDescriptorFromIoMemory,
// osmemdesc.c:730) and (b) an override at map time: NVOS46 flags GPU_CACHEABLE
// (virt_mem_allocator_gm107.c:1390) or, for UVM external mappings, gpuCachingType
// (nv_gpu_ops.c:3698). This shim prints all three, nothing else. Read-only: it
// forwards every ioctl unchanged.
//
// Build: gcc -O2 -shared -fPIC -o rmtrace.so rmtrace.c -ldl (plus)
//          -I$NVO/src/common/sdk/nvidia/inc -I$NVO/kernel-open/common/inc
//          -I$NVO/src/nvidia/arch/nvalloc/unix/include -I$NVO/kernel-open/nvidia-uvm
// Run:   RMTRACE_LOG=/tmp/rmtrace.log LD_PRELOAD=$PWD/rmtrace.so ./vis_probe --arms=E --n=1
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <pthread.h>

#include <nvtypes.h>
#include <nvstatus.h>
#include <nvos.h>
#include <nv_escape.h>
#include <nv-ioctl-numbers.h>
#include <nv-ioctl.h>
#include <nv-unix-nvos-params-wrappers.h>
#include "uvm_ioctl.h"

static int (*real_ioctl)(int, unsigned long, ...);
static FILE *logf_;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void init(void)
{
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    if (!logf_) {
        const char *p = getenv("RMTRACE_LOG");
        logf_ = p ? fopen(p, "a") : stderr;
        if (!logf_) logf_ = stderr;
    }
}

static int fd_kind(int fd)   // 1 = nvidiactl/nvidiaN, 2 = nvidia-uvm, 0 = other
{
    char link[64], path[256];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, sizeof(path) - 1);
    if (n <= 0) return 0;
    path[n] = 0;
    if (!strcmp(path, "/dev/nvidia-uvm")) return 2;
    if (!strncmp(path, "/dev/nvidia", 11)) return 1;
    return 0;
}

static const char *cache2(unsigned v)   // 2-bit GPU_CACHEABLE fields
{
    switch (v) { case 0: return "DEFAULT"; case 1: return "YES"; case 2: return "NO"; default: return "INVALID"; }
}

static void log_rm_pre(int nr, void *arg)
{
    if (nr == NV_ESC_RM_VID_HEAP_CONTROL) {
        NVOS32_PARAMETERS *p = arg;
        if (p->function == NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR) {
            unsigned a2 = (p->data.AllocOsDesc.attr2 >> 2) & 3u;   // NVOS32_ATTR2_GPU_CACHEABLE 3:2
            fprintf(logf_, "RM  VID_HEAP_CONTROL ALLOC_OS_DESCRIPTOR descType=%u (2=OS_IO_MEMORY) "
                    "attr=0x%08x attr2=0x%08x GPU_CACHEABLE=%s flags=0x%08x limit=0x%llx\n",
                    p->data.AllocOsDesc.descriptorType, p->data.AllocOsDesc.attr, p->data.AllocOsDesc.attr2,
                    cache2(a2), p->data.AllocOsDesc.flags,
                    (unsigned long long)p->data.AllocOsDesc.limit);
        }
    } else if (nr == NV_ESC_RM_MAP_MEMORY_DMA) {
        NVOS46_PARAMETERS *p = arg;
        unsigned c = (p->flags >> 17) & 3u;                       // NVOS46_FLAGS_GPU_CACHEABLE 18:17
        unsigned snoop = (p->flags >> 4) & 1u;                    // NVOS46_FLAGS_CACHE_SNOOP 4:4
        fprintf(logf_, "RM  MAP_MEMORY_DMA hMemory=0x%x len=0x%llx flags=0x%08x GPU_CACHEABLE=%s CACHE_SNOOP=%u flags2=0x%08x\n",
                p->hMemory, (unsigned long long)p->length, p->flags, cache2(c), snoop, p->flags2);
    } else if (nr == NV_ESC_RM_ALLOC_MEMORY) {
        nv_ioctl_nvos02_parameters_with_fd *w = arg;
        NVOS02_PARAMETERS *p = &w->params;
        fprintf(logf_, "RM  ALLOC_MEMORY class=0x%x flags=0x%08x GPU_CACHEABLE(18:18)=%u limit=0x%llx\n",
                p->hClass, p->flags, (p->flags >> 18) & 1u, (unsigned long long)p->limit);
    }
}

static void log_rm_post(int nr, void *arg)
{
    if (nr == NV_ESC_RM_VID_HEAP_CONTROL) {
        NVOS32_PARAMETERS *p = arg;
        if (p->function == NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR)
            fprintf(logf_, "    -> status=0x%x hMemory=0x%x\n", p->status, p->data.AllocOsDesc.hMemory);
    } else if (nr == NV_ESC_RM_MAP_MEMORY_DMA) {
        NVOS46_PARAMETERS *p = arg;
        fprintf(logf_, "    -> status=0x%x dmaOffset=0x%llx\n", p->status, (unsigned long long)p->dmaOffset);
    }
}

int ioctl(int fd, unsigned long req, ...)
{
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    init();
    int kind = fd_kind(fd);
    int nr = -1; void *inner = arg;
    if (kind == 1 && _IOC_TYPE(req) == NV_IOCTL_MAGIC) {
        nr = _IOC_NR(req);
        if (nr == NV_ESC_IOCTL_XFER_CMD) {
            nv_ioctl_xfer_t *x = arg;
            nr = (int)x->cmd; inner = (void *)(uintptr_t)x->ptr;
        }
        pthread_mutex_lock(&mu); log_rm_pre(nr, inner); fflush(logf_); pthread_mutex_unlock(&mu);
    } else if (kind == 2 && req == UVM_MAP_EXTERNAL_ALLOCATION) {
        UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *p = arg;
        pthread_mutex_lock(&mu);
        fprintf(logf_, "UVM MAP_EXTERNAL_ALLOCATION base=0x%llx len=0x%llx hMemory=0x%x gpus=%llu\n",
                (unsigned long long)p->base, (unsigned long long)p->length, p->hMemory,
                (unsigned long long)p->gpuAttributesCount);
        for (unsigned long long g = 0; g < p->gpuAttributesCount && g < UVM_MAX_GPUS; ++g)
            fprintf(logf_, "    gpu[%llu] mappingType=%u cachingType=%u (0=Default 1=ForceUncached 2=ForceCached) "
                    "formatType=%u compression=%u\n", g,
                    p->perGpuAttributes[g].gpuMappingType, p->perGpuAttributes[g].gpuCachingType,
                    p->perGpuAttributes[g].gpuFormatType, p->perGpuAttributes[g].gpuCompressionType);
        fflush(logf_);
        pthread_mutex_unlock(&mu);
    }
    int rc = real_ioctl(fd, req, arg);
    if (kind == 1 && nr >= 0) { pthread_mutex_lock(&mu); log_rm_post(nr, inner); fflush(logf_); pthread_mutex_unlock(&mu); }
    if (kind == 2 && req == UVM_MAP_EXTERNAL_ALLOCATION) {
        pthread_mutex_lock(&mu);
        fprintf(logf_, "    -> rmStatus=0x%x\n", ((UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *)arg)->rmStatus);
        fflush(logf_); pthread_mutex_unlock(&mu);
    }
    return rc;
}

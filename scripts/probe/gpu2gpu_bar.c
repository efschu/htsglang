/* SPDX-License-Identifier: MIT
 * Falsifikationsprobe #280: GPU->GPU-Direktschreiben ueber BAR-Mapping +
 * cudaHostRegisterIoMemory. Reine Userspace-Route, kein Treiber-Patch.
 *
 * F1  CPU-Mapping von GPU-B-VRAM beschaffen
 *     (a) mmap() auf den dmabuf-fd der VMM-Allokation
 *     (b) mmap() von /sys/bus/pci/devices/<pci>/resource1 (BAR1) + Musterscan
 * F2  cuMemHostRegister(..., CU_MEMHOSTREGISTER_IOMEMORY) im Kontext von GPU A
 * F3  4-KiB-Schreibprobe GPU A -> Bs VRAM, Verifikation per D2H aus der
 *     Original-Allokation auf B
 * F4  Leiter + Parallelpunkt
 *
 * Die dmabuf-Kette (VMM -> Objekt-fd -> RM-Client -> dmabuf-fd) ist
 * unveraendert aus scripts/probe/gpurdma_00_smoke.c uebernommen.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdint.h>

#include <infiniband/verbs.h>

#include <cuda.h>
#include <nvtypes.h>
#include <nvos.h>
#include <nv-ioctl.h>
#include <nv_escape.h>
#include <class/cl0000.h>
#include <class/cl0080.h>
#include <ctrl/ctrl0000/ctrl0000gpu.h>
#include <ctrl/ctrl0000/ctrl0000unix.h>

#define CHK(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) { \
    const char *_s = NULL; cuGetErrorName(_r, &_s); \
    fprintf(stderr, "[FAIL] %s -> %d (%s)\n", #x, _r, _s ? _s : "?"); exit(1); } } while (0)
#define TRY(x) ({ CUresult _r = (x); const char *_s = NULL, *_d = NULL; \
    cuGetErrorName(_r, &_s); cuGetErrorString(_r, &_d); \
    fprintf(stderr, "[try] %s -> %d (%s: %s)\n", #x, _r, _s?_s:"?", _d?_d:"?"); _r; })

static int nv_ioctl(int fd, int nr, void *p, size_t size)
{ return ioctl(fd, _IOC(_IOC_READ|_IOC_WRITE, NV_IOCTL_MAGIC, nr, (unsigned)size), p); }

#define MAGIC 0x2801BA1FEEDF00D5ull

/* ---- dmabuf-Kette, 1:1 aus gpurdma_00_smoke.c ---- */
static int gpu_dmabuf(int ord, size_t *sizep, CUdeviceptr *dptr,
                      CUmemGenericAllocationHandle *mh, char *namebuf, size_t nblen,
                      CUcontext *ctxout)
{
    CUdevice dev; int bus = -1; size_t size = *sizep;
    CHK(cuInit(0));
    CHK(cuDeviceGet(&dev, ord));
    CHK(cuDeviceGetName(namebuf, (int)nblen, dev));
    CHK(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CUcontext c; CHK(cuDevicePrimaryCtxRetain(&c, dev)); CHK(cuCtxSetCurrent(c));
    if (ctxout) *ctxout = c;

    CUmemAllocationProp prop; memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    CHK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    fprintf(stderr, "[i] VMM-Granularitaet minimum = %zu B\n", gran);
    if (size < gran) size = gran;
    if (size % gran) size = ((size / gran) + 1) * gran;
    *sizep = size;
    CHK(cuMemCreate(mh, size, &prop, 0));
    CHK(cuMemAddressReserve(dptr, size, 0, 0, 0));
    CHK(cuMemMap(*dptr, size, 0, *mh, 0));
    CUmemAccessDesc ad; memset(&ad, 0, sizeof(ad));
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHK(cuMemSetAccess(*dptr, size, &ad, 1));

    int objfd = -1;
    CHK(cuMemExportToShareableHandle(&objfd, *mh,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    int ctl = open("/dev/nvidiactl", O_RDWR);
    if (ctl < 0) { perror("nvidiactl"); return -1; }
    { nv_ioctl_rm_api_version_t v; char b[256]={0}, ver[64]={0}, *p;
      FILE *f = fopen("/proc/driver/nvidia/version","r");
      memset(&v,0,sizeof(v)); v.cmd = NV_RM_API_VERSION_CMD_RELAXED;
      if (f) { if(!fgets(b,sizeof(b),f)) b[0]=0; fclose(f); }
      p = strstr(b,"for x86_64");
      if (p && sscanf(p,"for x86_64 %63s",ver)==1) strncpy(v.versionString,ver,sizeof(v.versionString)-1);
      if (nv_ioctl(ctl, NV_ESC_CHECK_VERSION_STR, &v, sizeof(v)) < 0) return -1; }
    NvU32 gpu_id=0, minor=0; int found=0;
    { nv_ioctl_card_info_t ci[32]; memset(ci,0,sizeof(ci));
      if (nv_ioctl(ctl, NV_ESC_CARD_INFO, ci, sizeof(ci)) < 0) return -1;
      for (int i=0;i<32;i++){ if(!ci[i].valid) continue;
        if ((int)ci[i].pci_info.bus==bus){ gpu_id=ci[i].gpu_id; minor=ci[i].minor_number; found=1; } }
      if (!found) return -1; }
    NvHandle hCl=0, hDev=0xbee00001, hMem=0xbee00010;
    { NVOS21_PARAMETERS a; memset(&a,0,sizeof(a)); a.hClass=NV01_ROOT;
      if (nv_ioctl(ctl, NV_ESC_RM_ALLOC,&a,sizeof(a))<0||a.status) return -1;
      hCl=a.hObjectNew; }
    NvU32 di=0;
    { NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS p; NVOS54_PARAMETERS c2;
      memset(&p,0,sizeof(p)); memset(&c2,0,sizeof(c2)); p.gpuId=gpu_id;
      c2.hClient=hCl; c2.hObject=hCl; c2.cmd=NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2;
      c2.params=(NvP64)(uintptr_t)&p; c2.paramsSize=sizeof(p);
      if (nv_ioctl(ctl, NV_ESC_RM_CONTROL,&c2,sizeof(c2))<0||c2.status) return -1;
      di=p.deviceInstance; }
    { NV0080_ALLOC_PARAMETERS dp; NVOS21_PARAMETERS a;
      memset(&dp,0,sizeof(dp)); memset(&a,0,sizeof(a));
      dp.deviceId=di; dp.hClientShare=hCl;
      a.hRoot=hCl; a.hObjectParent=hCl; a.hObjectNew=hDev; a.hClass=NV01_DEVICE_0;
      a.pAllocParms=(NvP64)(uintptr_t)&dp;
      if (nv_ioctl(ctl, NV_ESC_RM_ALLOC,&a,sizeof(a))<0||a.status) return -1; }
    { NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS p; NVOS54_PARAMETERS c2;
      memset(&p,0,sizeof(p)); memset(&c2,0,sizeof(c2));
      p.fd=objfd; p.object.type=NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
      p.object.data.rmObject.hDevice=hDev; p.object.data.rmObject.hParent=hDev;
      p.object.data.rmObject.hObject=hMem;
      c2.hClient=hCl; c2.hObject=hCl; c2.cmd=NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD;
      c2.params=(NvP64)(uintptr_t)&p; c2.paramsSize=sizeof(p);
      if (nv_ioctl(ctl, NV_ESC_RM_CONTROL,&c2,sizeof(c2))<0||c2.status) {
          fprintf(stderr,"IMPORT status 0x%x\n", c2.status); return -1; } }
    { char dp[64]; snprintf(dp,sizeof(dp),"/dev/nvidia%u",minor);
      int dfd = open(dp,O_RDWR); if (dfd<0){perror(dp);return -1;}
      nv_ioctl_export_to_dma_buf_fd_t p; memset(&p,0,sizeof(p));
      p.fd=-1; p.hClient=hCl; p.totalObjects=1; p.numObjects=1; p.index=0;
      p.totalSize=size; p.numObjects=1;
      p.handles[0]=hMem; p.offsets[0]=0; p.sizes[0]=size;
      if (nv_ioctl(dfd, NV_ESC_EXPORT_TO_DMABUF_FD, &p, sizeof(p))<0) { perror("EXPORT_TO_DMABUF_FD"); return -1; }
      if (p.status) { fprintf(stderr,"EXPORT status 0x%x\n", p.status); return -1; }
      return p.fd; }
}

/* ---- Musterfuellung: 16-B-Records [MAGIC][offset] ---- */
static void fill_pattern(void *buf, size_t n)
{
    uint64_t *p = (uint64_t*)buf;
    for (size_t o = 0; o + 16 <= n; o += 16) { p[o/8] = MAGIC; p[o/8+1] = o; }
}

static sigjmp_buf jb; static volatile int in_scan = 0;
static void bushdl(int s) { (void)s; if (in_scan) siglongjmp(jb, 1); _exit(97); }


/* Fensterweiser BAR1-Scan. sysfs-resource-mmap akzeptiert auf diesem Kernel
 * nur Fenster <= 32 MiB, beliebige Offsets; also 2-MiB-Fenster. */
static size_t bar_scan(int rfd, size_t barsz, size_t *basep, const char *tag)
{
    const size_t WIN = 2u<<20, stride = 65536;
    size_t hit = (size_t)-1, base = 0; long wfail = 0, wok = 0;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t w = 0; w < barsz && hit == (size_t)-1; w += WIN) {
        size_t len = (barsz - w < WIN) ? (barsz - w) : WIN;
        void *m = mmap(NULL, len, PROT_READ, MAP_SHARED, rfd, (off_t)w);
        if (m == MAP_FAILED) { wfail++; continue; }
        in_scan = 1;
        if (sigsetjmp(jb, 1) == 0) {
            volatile uint64_t *v = (volatile uint64_t*)m;
            for (size_t o = 0; o + 16 <= len; o += stride)
                if (v[o/8] == MAGIC) { hit = w + o; base = hit - (size_t)v[o/8+1]; break; }
        } else fprintf(stderr, "[!] SIGBUS im Fenster 0x%zx\n", w);
        in_scan = 0; wok++;
        munmap(m, len);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sc = (t1.tv_sec-t0.tv_sec) + 1e-9*(t1.tv_nsec-t0.tv_nsec);
    fprintf(stderr, "[scan:%s] hit=%s base=0x%zx %.2fs Fenster ok=%ld fail=%ld\n",
            tag, hit==(size_t)-1 ? "none" : "yes", base, sc, wok, wfail);
    if (basep) *basep = base;
    return hit;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "verify";
    int ordB   = argc > 2 ? atoi(argv[2]) : 1;          /* Ziel-GPU (B) */
    const char *pciB = argc > 3 ? argv[3] : "0000:0a:00.0";
    int ordA   = argc > 4 ? atoi(argv[4]) : 2;          /* Quell-GPU (A) */
    size_t xfer = argc > 5 ? (size_t)atoll(argv[5]) : 4096;
    double secs = argc > 6 ? atof(argv[6]) : 5.0;
    const char *nic = argc > 7 ? argv[7] : "rocep4s0f0";

    size_t allocsz = xfer < (2u<<20) ? (2u<<20) : xfer;
    CUdeviceptr dB = 0; CUmemGenericAllocationHandle mh = 0;
    char nameB[128] = {0}; CUcontext ctxB = NULL;

    int dfd = gpu_dmabuf(ordB, &allocsz, &dB, &mh, nameB, sizeof(nameB), &ctxB);
    if (dfd < 0) { printf("RESULT F1 FAIL errno=%d dmabuf-Kette\n", errno); return 1; }
    fprintf(stderr, "[i] B=ord%d '%s' alloc=%zu dmabuf_fd=%d\n", ordB, nameB, allocsz, dfd);

    /* ---------- F1a: mmap auf den dmabuf-fd ---------- */
    errno = 0;
    void *dm = mmap(NULL, allocsz, PROT_READ|PROT_WRITE, MAP_SHARED, dfd, 0);
    if (dm == MAP_FAILED) {
        printf("RESULT F1a FAIL errno=%d (%s) mmap(dmabuf_fd)\n", errno, strerror(errno));
    } else {
        printf("RESULT F1a PASS errno=0 mmap(dmabuf_fd)=%p\n", dm);
    }

    /* Muster in Bs VRAM schreiben (Host -> B, ganz normal) */
    void *hb = malloc(allocsz); fill_pattern(hb, allocsz);
    CHK(cuCtxSetCurrent(ctxB));
    CHK(cuMemcpyHtoD(dB, hb, allocsz));
    CHK(cuCtxSynchronize());

    /* ---------- F1b: BAR1 mappen + Muster suchen ---------- */
    char rp[256]; { const char *bn = getenv("BARNODE"); snprintf(rp, sizeof(rp), "/sys/bus/pci/devices/%s/%s", pciB, bn ? bn : "resource1_wc"); }
    int rfd = open(rp, O_RDWR|O_SYNC);
    if (rfd < 0) { printf("RESULT F1b FAIL errno=%d open(%s)\n", errno, rp); return 1; }
    struct stat st; fstat(rfd, &st);
    size_t barsz = (size_t)st.st_size;
    /* sysfs-resource-mmap akzeptiert auf diesem Kernel nur Fenster <= 32 MiB
     * (>=64 MiB -> EINVAL), beliebige Offsets sind erlaubt. Also fensterweise
     * scannen statt die BAR am Stueck zu mappen. */
    size_t hit = (size_t)-1, base = 0;
    signal(SIGBUS, bushdl); signal(SIGSEGV, bushdl);

    /* ---------- F1c-0: Scan POSITIV KALIBRIEREN ----------
     * Bekanntes Muster per CPU auf BAR1-Offset 0 schreiben, scannen, wieder
     * loeschen. Findet der Scan das nicht, ist er kein Beweis. */
    {
        void *c = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, rfd, 0);
        if (c == MAP_FAILED) { printf("RESULT F1c0 FAIL errno=%d mmap(cal)\n", errno); return 1; }
        volatile uint64_t *cv = (volatile uint64_t*)c;
        cv[0] = MAGIC; cv[1] = 0;
        size_t cb = 0, ch = bar_scan(rfd, barsz, &cb, "kalibrierung");
        printf("RESULT F1c0 %s errno=0 Scan-Kalibrierung: erwartet hit=0x0, bekommen hit=0x%zx\n",
               (ch == 0) ? "PASS" : "FAIL", ch);
        cv[0] = 0; cv[1] = 0;
        munmap(c, 4096);
        if (ch != 0) return 1;
    }

    /* ---------- F1c-1: BAR1-Backing per ibv_reg_dmabuf_mr erzwingen ---------- */
    struct ibv_mr *mr = NULL; struct ibv_pd *pd = NULL; struct ibv_context *ibc = NULL;
    {
        int nd = 0; struct ibv_device **dl = ibv_get_device_list(&nd);
        struct ibv_device *dev = NULL;
        for (int i = 0; i < nd; i++) if (!strcmp(ibv_get_device_name(dl[i]), nic)) dev = dl[i];
        if (!dev) { printf("RESULT F1c1 FAIL errno=0 NIC '%s' nicht gefunden\n", nic); return 1; }
        ibc = ibv_open_device(dev);
        if (!ibc) { printf("RESULT F1c1 FAIL errno=%d ibv_open_device\n", errno); return 1; }
        pd = ibv_alloc_pd(ibc);
        if (!pd) { printf("RESULT F1c1 FAIL errno=%d ibv_alloc_pd\n", errno); return 1; }
        errno = 0;
        mr = ibv_reg_dmabuf_mr(pd, 0, allocsz, 0, dfd,
                               IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
        if (!mr) { printf("RESULT F1c1 FAIL errno=%d (%s) ibv_reg_dmabuf_mr\n",
                          errno, strerror(errno)); return 1; }
        printf("RESULT F1c1 PASS errno=0 MR lebt: lkey=0x%x rkey=0x%x addr=%p len=%zu\n",
               mr->lkey, mr->rkey, mr->addr, mr->length);
    }

    /* ---------- F1b (Wiederholung): Scan bei LEBENDER MR ---------- */
    hit = bar_scan(rfd, barsz, &base, "mit-lebender-MR");
    if (hit == (size_t)-1) {
        printf("RESULT F1b FAIL errno=0 Muster trotz lebender MR nicht in BAR1 gefunden\n");
        return 1;
    }
    printf("RESULT F1b PASS errno=0 hit=0x%zx base=0x%zx (mit lebender MR)\n", hit, base);

    /* Fenster exakt auf die Allokation legen und Muster vollstaendig pruefen */
    void *bar = mmap(NULL, allocsz, PROT_READ|PROT_WRITE, MAP_SHARED, rfd, (off_t)base);
    if (bar == MAP_FAILED) { printf("RESULT F1 FAIL errno=%d mmap(base=0x%zx,len=%zu)\n",
                                    errno, base, allocsz); return 1; }
    {
        volatile uint64_t *b64 = (volatile uint64_t*)bar;
        size_t bad = 0;
        for (size_t o = 0; o + 16 <= allocsz; o += 4096)
            if (b64[o/8] != MAGIC || b64[o/8+1] != o) bad++;
        printf("RESULT F1 %s errno=0 CPU-Lesepruefung durch BAR: %zu von %zu 4K-Punkten falsch\n",
               bad ? "FAIL" : "PASS", bad, allocsz/4096);
        if (bad) return 1;
    }
    /* ---------- F1c-2: ueberlebt das BAR1-Mapping ibv_dereg_mr? ---------- */
    {
        if (ibv_dereg_mr(mr)) fprintf(stderr, "[!] ibv_dereg_mr fehlgeschlagen\n");
        mr = NULL;
        size_t b2 = 0, h2 = bar_scan(rfd, barsz, &b2, "nach-dereg");
        printf("RESULT F1c2 INFO errno=0 nach ibv_dereg_mr: %s (hit=0x%zx, vorher base=0x%zx)\n",
               (h2 == (size_t)-1) ? "Mapping WEG - MR muss gehalten werden"
                                  : "Mapping BLEIBT - einmalige Registrierung reicht",
               h2, base);
    }
    if (!strcmp(mode, "f1")) return 0;

    /* ---------- F2: cuMemHostRegister IOMEMORY im Kontext von A ---------- */
    CUdevice devA; CUcontext ctxA;
    CHK(cuDeviceGet(&devA, ordA));
    char nameA[128]; CHK(cuDeviceGetName(nameA, sizeof(nameA), devA));
    CHK(cuDevicePrimaryCtxRetain(&ctxA, devA)); CHK(cuCtxSetCurrent(ctxA));
    fprintf(stderr, "[i] A=ord%d '%s'\n", ordA, nameA);

    void *win = bar;   /* Fenster liegt bereits exakt auf der Allokation */
    CUresult r = TRY(cuMemHostRegister(win, allocsz,
                     CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_PORTABLE));
    if (r != CUDA_SUCCESS) {
        const char *n=NULL; cuGetErrorName(r,&n);
        printf("RESULT F2 FAIL cuda=%d (%s) cuMemHostRegister IOMEMORY|PORTABLE\n", r, n?n:"?");
        /* Umgehungsversuch 1: nur IOMEMORY */
        r = TRY(cuMemHostRegister(win, allocsz, CU_MEMHOSTREGISTER_IOMEMORY));
        if (r != CUDA_SUCCESS) {
            cuGetErrorName(r,&n);
            printf("RESULT F2b FAIL cuda=%d (%s) cuMemHostRegister IOMEMORY\n", r, n?n:"?");
            /* Umgehungsversuch 2: kleineres Fenster, 64 KiB */
            r = TRY(cuMemHostRegister(win, 65536, CU_MEMHOSTREGISTER_IOMEMORY));
            if (r != CUDA_SUCCESS) {
                cuGetErrorName(r,&n);
                printf("RESULT F2c FAIL cuda=%d (%s) cuMemHostRegister IOMEMORY 64KiB\n", r, n?n:"?");
                return 1;
            }
            allocsz = 65536;
        }
    }
    printf("RESULT F2 PASS cuda=0 registriert win=%p len=%zu\n", win, allocsz);

    CUdeviceptr dwin = 0;
    r = TRY(cuMemHostGetDevicePointer(&dwin, win, 0));
    if (r != CUDA_SUCCESS) { printf("RESULT F2d FAIL cuda=%d cuMemHostGetDevicePointer\n", r); return 1; }
    printf("RESULT F2d PASS dwin=0x%llx\n", (unsigned long long)dwin);

    /* ---------- F3: 4-KiB-Schreibprobe A -> B ---------- */
    if (xfer > allocsz) xfer = allocsz;
    CUdeviceptr srcA = 0; CHK(cuMemAlloc(&srcA, allocsz));
    void *hs = malloc(allocsz);
    for (size_t i = 0; i < allocsz; i++) ((unsigned char*)hs)[i] = (unsigned char)((i*7+3) & 0xff);
    CHK(cuMemcpyHtoD(srcA, hs, allocsz));
    CHK(cuCtxSynchronize());

    r = TRY(cuMemcpyDtoD(dwin, srcA, xfer));
    if (r != CUDA_SUCCESS) { printf("RESULT F3 FAIL cuda=%d cuMemcpyDtoD in BAR-Fenster\n", r); return 1; }
    CHK(cuCtxSynchronize());

    /* Verifikation auf B, durch die ORIGINAL-Allokation, nicht durchs Mapping */
    CHK(cuCtxSetCurrent(ctxB));
    void *hv = malloc(xfer);
    CHK(cuMemcpyDtoH(hv, dB, xfer));
    CHK(cuCtxSynchronize());
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < xfer; i++)
        if (((unsigned char*)hv)[i] != ((unsigned char*)hs)[i]) { bad++; if (first==(size_t)-1) first=i; }
    printf("RESULT F3 %s errno=0 bytes=%zu mismatch=%zu first=%zd\n",
           bad ? "FAIL" : "PASS", xfer, bad, (ssize_t)first);
    if (bad) return 1;

    if (!strcmp(mode, "verify")) return 0;

    /* ---------- F4: Durchsatz/Latenz ---------- */
    CHK(cuCtxSetCurrent(ctxA));
    CUstream s; CHK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    /* Aufwaermen */
    for (int i = 0; i < 20; i++) CHK(cuMemcpyDtoDAsync(dwin, srcA, xfer, s));
    CHK(cuStreamSynchronize(s));
    long iters = 0; struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (;;) {
        for (int i = 0; i < 32; i++) CHK(cuMemcpyDtoDAsync(dwin, srcA, xfer, s));
        CHK(cuStreamSynchronize(s));
        iters += 32;
        clock_gettime(CLOCK_MONOTONIC, &b);
        double el = (b.tv_sec-a.tv_sec)+1e-9*(b.tv_nsec-a.tv_nsec);
        if (el >= secs) {
            printf("RESULT F4 PASS bytes=%zu iters=%ld sec=%.3f us_per_write=%.3f GBps=%.3f\n",
                   xfer, iters, el, el*1e6/iters, (double)xfer*iters/el/1e9);
            break;
        }
    }
    return 0;
}

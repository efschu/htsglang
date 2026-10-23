// vis_probe.cu -- barlink-visibility-1023, PHASE-2 probe (built in phase 1, NOT run yet)
//
// ONE question per arm: where are the sender's BAR1 bytes while they are not
// visible on the receiving card (barlink-ll-ordering-1022.md: >= 2.3-4.7 s,
// i.e. until the cycle cap / kernel end, even with ld.mmio retries)?
//
// Setup is reused VERBATIM from nvidia-smallbar-p2p/sonden/peerstore_probe.cu
// (VMM alloc on the receiver -> dma-buf export -> dmabuf_holder attach as the
// sender BDF -> BAR1 offset from the sg table -> resource1_wc mmap ->
// cudaHostRegister(IoMemory) on the sender). That is the same mapping chain
// barlink_bar1.py builds, in one process with two contexts.
//
// Every trial writes a slot of --bytes (default 2 649 040 B = the 96-MiB
// half-slot of the 22.10. reproducer) with a pattern that depends on (round,
// packet) -- stale data can never match -- then a flag. The receiver kernel
// waits for the flag with ld.mmio and then SCANS the whole slot with ld.mmio
// (cache-bypassing, LDG.E.64.MMIO.SYS) until every packet matches or capMs.
// It records when the slot became complete, and how much was missing at the
// first scan, at 10us/100us/1ms/10ms/100ms/1s after the flag, and at the end.
// After both kernels ended the host copies the slot out (copy engine): that is
// the "arrived eventually" truth.
//
// ARMS (letters for --arms):
//   E  unidirectional, sender stores st.global.wt.v4 (= barlink writeV4),
//      sender kernel EXITS right after the flag
//   S  as E, but the sender kernel SPINS spinMs after the flag on clock64 only
//      (no memory traffic at all, so the fabric is quiet)
//   M  as S, but the payload is stored with st.mmio.relaxed.sys (cache-bypass
//      on the SENDER, never combined)
//   R  as S, but before the flag the sender reads its last packet back from
//      the peer (ld.mmio) and fences, so the flag cannot leave before that read
//      completed (PCIe: a read never passes a posted write on the same path)
//   A  as S, and during the spin the sender polls its own last packet at the
//      peer (ld.mmio) and records when it reads back what it wrote
//   F  as S, and the host calls cuFlushGPUDirectRDMAWrites(TO_OWNER) on the
//      RECEIVER context flushMs after launch while both kernels still run
//   B  bidirectional (both cards send and receive at once), both senders SPIN
//   X  as S, but the receiver kernel gives up after --x-cap-ms (300) while the
//      sender kernel is still spinning; then, WHILE THE SENDER STILL RUNS, three
//      other observers look at the slot: (1) the copy engine (D2H), (2) the CPU
//      through the BAR1 mapping (a PCIe read on the same last hop as the P2P
//      writes, so it pushes anything parked in the switch/receiver ingress),
//      (3) a FRESH receiver kernel (kernel boundary on the receiver only)
//
// Decision table (phase 1 report barlink-visibility-1023.md, section 4):
//   E complete in us/ms but S complete only at ~spinMs   -> bytes held in the SENDER until its kernel ends
//   M complete early while S is late                     -> held in a sender CACHE stage (L2 / vol=0 peer-IO PTE)
//   R complete early while S is late                     -> bytes were in flight (fabric/receiver ingress), a read pushes them
//   F completes at ~flushMs                              -> held at the RECEIVER (inbound buffer), host flush releases
//   E and S both early, B late                           -> full-duplex class (bug 1571948 / RC hairpin under counter traffic)
//   X: fresh kernel / CE see it, running kernel did not  -> RECEIVER-side visibility (running kernel only)
//   X: only the CPU BAR1 read sees it (and after it, CE)  -> parked in switch/receiver ingress, a read pushes it
//   X: nobody sees it while the sender spins              -> not at the receiver: SENDER (or upstream fabric)
//
// Build (no GPU needed):
//   NVO=/spinning/nvidia-open-595 SP=/spinning/nvidia-smallbar-p2p
//   nvcc -O3 -std=c++14 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 \
//     -I$SP -I$SP/sonden -I$NVO/src/common/sdk/nvidia/inc -I$NVO/kernel-open/common/inc \
//     -I$NVO/src/nvidia/arch/nvalloc/unix/include -o vis_probe vis_probe.cu -lcuda
//
// Run (phase 2 only, inside an own gpuq window, cards 0,1, card locks held):
//   ./vis_probe --a=0 --b=1 --arms=ESXMRAFB --n=6 [--bytes=2649040 --spin-ms=1500 --cap-ms=3000 --flush-ms=500]
//
// Rules kept from the base probe: every device spin has a clock/globaltimer
// cap, the host never waits unbounded (cudaStreamQuery loops), nothing is
// loaded, unloaded, reset or killed. PCI config is never touched.

#define main peerstore_probe_main_unused
#include "peerstore_probe.cu"
#undef main

#include <cinttypes>

typedef unsigned long long u64;

// ---------------------------------------------------------------------------
// pattern: depends on round and packet index, never all-zero
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ u64 vpMix(u64 v)
{
    v ^= v >> 30; v *= 0xBF58476D1CE4E5B9ull;
    v ^= v >> 27; v *= 0x94D049BB133111EBull;
    v ^= v >> 31;
    return v;
}

__host__ __device__ __forceinline__ uint4 vpPattern(u64 round, u64 i)
{
    const u64 a = vpMix(round * 0x9E3779B97F4A7C15ull ^ (i * 0xD6E8FEB86659FD93ull)) | 1ull;
    const u64 b = vpMix(a ^ 0xA24BAED4963EE407ull);
    return make_uint4((unsigned)a, (unsigned)(a >> 32), (unsigned)b, (unsigned)(b >> 32));
}

__device__ __forceinline__ u64 vpNow()
{
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void vpStWt(uint4 *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void vpStMmio(uint4 *p, uint4 v)
{
    const u64 a = ((u64)v.y << 32) | v.x, b = ((u64)v.w << 32) | v.z;
    asm volatile("st.mmio.relaxed.sys.global.u64 [%0], %1;" :: "l"(p), "l"(a) : "memory");
    asm volatile("st.mmio.relaxed.sys.global.u64 [%0], %1;" :: "l"((char *)p + 8), "l"(b) : "memory");
}

__device__ __forceinline__ uint4 vpLdMmio(const uint4 *p)
{
    u64 a, b;
    asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];" : "=l"(a) : "l"(p) : "memory");
    asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];" : "=l"(b) : "l"((const char *)p + 8) : "memory");
    return make_uint4((unsigned)a, (unsigned)(a >> 32), (unsigned)b, (unsigned)(b >> 32));
}

__device__ __forceinline__ u64 vpLdFlag(const u64 *p)
{
    u64 v;
    asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void vpStFlag(u64 *p, u64 v)
{
    // identical to barlink writeU64
    asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__host__ __device__ __forceinline__ bool vpEq(uint4 a, uint4 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

enum { ST_WT = 0, ST_MMIO = 1 };
enum { TAIL_EXIT = 0, TAIL_SPIN = 1, TAIL_SPIN_MONITOR = 2 };

struct SendArgs {
    uint4    *dst;        // sender's device pointer into the peer BAR1 (cudaHostRegister IoMemory)
    u64       n4;
    u64       round;
    u64      *flag;       // peer flag word, same mapping
    int       storeMode;
    int       tailMode;
    int       readbackBeforeFlag;
    u64       spinNs;
    unsigned *done;       // sender-local counter, zeroed per trial
    u64      *ts;         // sender-local: [0] t_flag [1] t_end [2] readback_ok [3] t_arrive (monitor)
};

__global__ void vp_send(SendArgs A)
{
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 nth = (u64)gridDim.x * blockDim.x;
    for (u64 i = tid; i < A.n4; i += nth) {
        const uint4 v = vpPattern(A.round, i);
        if (A.storeMode == ST_MMIO) vpStMmio(A.dst + i, v);
        else                        vpStWt(A.dst + i, v);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(A.done, 1u);
    if (!(blockIdx.x == 0 && threadIdx.x == 0)) return;

    const u64 t0 = vpNow();
    while (*(volatile unsigned *)A.done < gridDim.x) {
        if (vpNow() - t0 > 2000000000ull) { A.ts[0] = 0; return; }   // cap 2 s
    }
    __threadfence_system();

    if (A.readbackBeforeFlag) {
        const uint4 rb = vpLdMmio(A.dst + (A.n4 - 1));
        A.ts[2] = vpEq(rb, vpPattern(A.round, A.n4 - 1)) ? 1ull : 2ull;   // depends on the load
        __threadfence_system();                                           // the load is performed before the flag
    }
    vpStFlag(A.flag, A.round);
    __threadfence_system();
    const u64 tf = vpNow();
    A.ts[0] = tf;

    if (A.tailMode == TAIL_SPIN) {
        long long c0 = clock64();
        while (vpNow() - tf < A.spinNs) { (void)c0; }                    // no memory traffic
    } else if (A.tailMode == TAIL_SPIN_MONITOR) {
        const uint4 want = vpPattern(A.round, A.n4 - 1);
        A.ts[3] = 0;
        while (vpNow() - tf < A.spinNs) {
            if (A.ts[3] == 0 && vpEq(vpLdMmio(A.dst + (A.n4 - 1)), want)) A.ts[3] = vpNow();
            __nanosleep(1000);
        }
    }
    A.ts[1] = vpNow();
}

#define VP_MARKS 6
struct RecvArgs {
    const uint4 *slot;    // receiver's OWN VMM pointer
    u64          n4;
    u64          round;
    const u64   *flag;    // receiver's OWN VMM pointer
    u64          capNs;
    u64         *res;     // receiver-local, see VR_*
};
enum {
    VR_T_ENTRY = 0, VR_T_FLAG, VR_MISS_FIRST, VR_FIRST_IDX, VR_LAST_IDX, VR_T_COMPLETE,
    VR_SCANS, VR_MISS_END, VR_FIRST_END, VR_LAST_END, VR_MARK0, VR_N = VR_MARK0 + VP_MARKS
};

__global__ void vp_recv(RecvArgs A)
{
    __shared__ unsigned long long sMiss, sFirst, sLast;
    __shared__ int sStop;
    const u64 marksNs[VP_MARKS] = {10000ull, 100000ull, 1000000ull, 10000000ull, 100000000ull, 1000000000ull};
    if (threadIdx.x == 0) {
        A.res[VR_T_ENTRY] = vpNow();
        for (int k = 0; k < VP_MARKS; ++k) A.res[VR_MARK0 + k] = ~0ull;
        A.res[VR_T_COMPLETE] = 0; A.res[VR_T_FLAG] = 0;
        sStop = 0;
        const u64 t0 = vpNow();
        while (vpLdFlag(A.flag) != A.round) {
            if (vpNow() - t0 > A.capNs) { sStop = 1; break; }
        }
        A.res[VR_T_FLAG] = vpNow();
    }
    __syncthreads();
    if (sStop) return;
    const u64 tFlag = A.res[VR_T_FLAG];
    int markNext = 0;
    u64 scans = 0;
    for (;;) {
        if (threadIdx.x == 0) { sMiss = 0; sFirst = ~0ull; sLast = 0; }
        __syncthreads();
        unsigned long long miss = 0, first = ~0ull, last = 0;
        for (u64 i = threadIdx.x; i < A.n4; i += blockDim.x) {
            if (!vpEq(vpLdMmio(A.slot + i), vpPattern(A.round, i))) {
                ++miss; if (i < first) first = i; last = i;
            }
        }
        if (miss) { atomicAdd(&sMiss, miss); atomicMin(&sFirst, first); atomicMax(&sLast, last); }
        __syncthreads();
        ++scans;
        if (threadIdx.x == 0) {
            const u64 now = vpNow(), dt = now - tFlag;
            if (scans == 1) {
                A.res[VR_MISS_FIRST] = sMiss; A.res[VR_FIRST_IDX] = sFirst; A.res[VR_LAST_IDX] = sLast;
            }
            while (markNext < VP_MARKS && dt >= marksNs[markNext]) A.res[VR_MARK0 + markNext++] = sMiss;
            A.res[VR_MISS_END] = sMiss; A.res[VR_FIRST_END] = sFirst; A.res[VR_LAST_END] = sLast;
            A.res[VR_SCANS] = scans;
            sStop = 0;
            if (sMiss == 0) { A.res[VR_T_COMPLETE] = now; sStop = 1; }
            else if (dt > A.capNs) sStop = 1;
        }
        __syncthreads();
        if (sStop) break;
        __nanosleep(2000);
    }
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
struct Side {               // one directed pair: src writes into a region on dst
    int ord_src = 0, ord_dst = 0;
    Aufbau A;
    Region reg;
    int barFd = -1;
    u64 n4 = 0, slotBytes = 0, flagOff = 0;
    unsigned *done = nullptr; u64 *ts = nullptr;   // on src
    u64 *res = nullptr;                             // on dst
    cudaStream_t ss = nullptr, sr = nullptr;
    CUcontext ctxDst = nullptr;
    std::string name;
};

static int sideBuild(Side &S, int src, int dst, u64 bytes, const std::string &holderPath)
{
    S.ord_src = src; S.ord_dst = dst;
    char bus[64] = {0};
    RT(cudaDeviceGetPCIBusId(bus, sizeof(bus), dst)); std::string dstBDF = lower(bus);
    std::memset(bus, 0, sizeof(bus));
    RT(cudaDeviceGetPCIBusId(bus, sizeof(bus), src)); std::string srcBDF = lower(bus);
    cudaDeviceProp ps, pd;
    RT(cudaGetDeviceProperties(&ps, src)); RT(cudaGetDeviceProperties(&pd, dst));
    S.name = std::string(ps.name) + " -> " + pd.name;
    CUdevice dstDev = 0; DRV(cuDeviceGet(&dstDev, dst));

    std::string barPath = "/sys/bus/pci/devices/" + dstBDF + "/resource1_wc";
    S.barFd = open(barPath.c_str(), O_RDWR | O_SYNC);
    if (S.barFd < 0) { std::fprintf(stderr, "open(%s): %s\n", barPath.c_str(), std::strerror(errno)); return 2; }
    struct stat st; if (fstat(S.barFd, &st) != 0) return 2;
    uint64_t b1s = 0, b1e = 0; bool have = readBarRange(dstBDF, 1, &b1s, &b1e);

    S.A.dstOrd = dst; S.A.srcOrd = src; S.A.dstDev = dstDev;
    S.A.dstBDF = dstBDF; S.A.srcBDF = srcBDF;
    S.A.barFd = S.barFd; S.A.barSize = (uint64_t)st.st_size;
    S.A.bar1Start = b1s; S.A.bar1End = b1e; S.A.haveBar1Phys = have;
    S.A.imp = IMP_MODUL; S.A.ibv = nullptr; S.A.ibPd = nullptr;
    S.A.holderPath = holderPath;

    S.slotBytes = (bytes + 15) / 16 * 16;
    S.n4 = S.slotBytes / 16;
    S.flagOff = (S.slotBytes + SEITE - 1) / SEITE * SEITE;
    S.reg.name = "slot+flag";
    S.reg.wunsch = S.flagOff + SEITE;
    S.reg.nonce = ((uint64_t)getpid() << 32) ^ (uint64_t)time(nullptr) ^ ((uint64_t)dst << 8) ^ src;
    int rc = regionAufbauen(S.A, S.reg);
    if (rc) return rc;

    RT(cudaSetDevice(src));
    RT(cudaMalloc(&S.done, sizeof(unsigned)));
    RT(cudaMalloc(&S.ts, 8 * sizeof(u64)));
    RT(cudaStreamCreateWithFlags(&S.ss, cudaStreamNonBlocking));
    RT(cudaSetDevice(dst));
    RT(cudaMalloc(&S.res, VR_N * sizeof(u64)));
    RT(cudaStreamCreateWithFlags(&S.sr, cudaStreamNonBlocking));
    DRV(cuCtxGetCurrent(&S.ctxDst));
    // driver statement on inbound remote writes for the RECEIVER
    int ord = -1, fl = -1;
    cuDeviceGetAttribute(&ord, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, dstDev);
    cuDeviceGetAttribute(&fl, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, dstDev);
    std::printf("PAIR %s: receiver WRITES_ORDERING=%d FLUSH_OPTIONS=0x%x, slot %llu B, "
                "BAR1 offset 0x%llx, sender ptr %p\n",
                S.name.c_str(), ord, fl, S.slotBytes, (unsigned long long)S.reg.barVersatz, S.reg.ioDev);
    return 0;
}

static int sideLaunchRecv(Side &S, u64 round, u64 capNs)
{
    RT(cudaSetDevice(S.ord_dst));
    RecvArgs R;
    R.slot = (const uint4 *)S.reg.dptr; R.n4 = S.n4; R.round = round;
    R.flag = (const u64 *)(S.reg.dptr + S.flagOff); R.capNs = capNs; R.res = S.res;
    RT(cudaMemsetAsync(S.res, 0, VR_N * sizeof(u64), S.sr));
    vp_recv<<<1, 1024, 0, S.sr>>>(R);
    RT(cudaGetLastError());
    return 0;
}

static int sideLaunchSend(Side &S, u64 round, int storeMode, int tailMode, int readback, u64 spinNs)
{
    RT(cudaSetDevice(S.ord_src));
    int sms = 0; RT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, S.ord_src));
    SendArgs A;
    A.dst = (uint4 *)S.reg.ioDev; A.n4 = S.n4; A.round = round;
    A.flag = (u64 *)((char *)S.reg.ioDev + S.flagOff);
    A.storeMode = storeMode; A.tailMode = tailMode; A.readbackBeforeFlag = readback;
    A.spinNs = spinNs; A.done = S.done; A.ts = S.ts;
    RT(cudaMemsetAsync(S.done, 0, sizeof(unsigned), S.ss));
    RT(cudaMemsetAsync(S.ts, 0, 8 * sizeof(u64), S.ss));
    vp_send<<<sms, 256, 0, S.ss>>>(A);
    RT(cudaGetLastError());
    return 0;
}

static int report(const char *arm, Side &S, int trial, double flushAtMs, double hostMs)
{
    u64 r[VR_N], t[8];
    RT(cudaSetDevice(S.ord_dst)); RT(cudaMemcpy(r, S.res, sizeof(r), cudaMemcpyDeviceToHost));
    RT(cudaSetDevice(S.ord_src)); RT(cudaMemcpy(t, S.ts, sizeof(t), cudaMemcpyDeviceToHost));
    // eventual truth: slot copied out by the copy engine after both kernels ended
    std::vector<uint4> img(S.n4);
    RT(cudaSetDevice(S.ord_dst));
    RT(cudaMemcpy(img.data(), (const void *)S.reg.dptr, S.slotBytes, cudaMemcpyDeviceToHost));
    u64 badAfter = 0;
    // round is not stored on the host side here: recompute from the receiver's flag word
    u64 roundSeen = 0;
    RT(cudaMemcpy(&roundSeen, (const void *)(S.reg.dptr + S.flagOff), 8, cudaMemcpyDeviceToHost));
    for (u64 i = 0; i < S.n4; ++i) if (!vpEq(img[i], vpPattern(roundSeen, i))) ++badAfter;

    const double flagUs = r[VR_T_FLAG] ? (r[VR_T_FLAG] - r[VR_T_ENTRY]) / 1e3 : -1;
    const double compMs = r[VR_T_COMPLETE] ? (r[VR_T_COMPLETE] - r[VR_T_FLAG]) / 1e6 : -1;
    const double sendTailMs = (t[0] && t[1]) ? (t[1] - t[0]) / 1e6 : -1;
    const double arriveMs = (t[0] && t[3]) ? (t[3] - t[0]) / 1e6 : -1;
    std::printf("ARM %s | %s | trial %d | flag@%.1fus | miss_first=%llu [%llu..%llu] | "
                "complete=%s%.3f ms | marks(10us,100us,1ms,10ms,100ms,1s)=",
                arm, S.name.c_str(), trial, flagUs, r[VR_MISS_FIRST],
                r[VR_FIRST_IDX] == ~0ull ? 0 : r[VR_FIRST_IDX], r[VR_LAST_IDX],
                compMs < 0 ? "NEVER/" : "", compMs);
    for (int k = 0; k < VP_MARKS; ++k) {
        if (r[VR_MARK0 + k] == ~0ull) std::printf("-%s", k + 1 < VP_MARKS ? "," : "");
        else std::printf("%llu%s", r[VR_MARK0 + k], k + 1 < VP_MARKS ? "," : "");
    }
    std::printf(" | miss_end=%llu [%llu..%llu] scans=%llu"
                " | sender_tail=%.1f ms readback=%llu self_arrive=%.3f ms | bad_after_kernels=%llu"
                " | host_flush@%.0f ms | host %.0f ms\n",
                r[VR_MISS_END], r[VR_FIRST_END] == ~0ull ? 0 : r[VR_FIRST_END], r[VR_LAST_END], r[VR_SCANS],
                sendTailMs, t[2], arriveMs, badAfter, flushAtMs, hostMs);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char **argv)
{
    int a = 0, b = 1, n = 6;
    u64 bytes = 2649040ull;
    double spinMs = 1500, capMs = 3000, flushMs = 500, xCapMs = 300;
    std::string arms = "ESXMRAFB", holder = "/dev/dmabuf_holder", dirs = "both";
    for (int i = 1; i < argc; ++i) {
        const char *s = argv[i];
        if (!std::strncmp(s, "--a=", 4)) a = atoi(s + 4);
        else if (!std::strncmp(s, "--b=", 4)) b = atoi(s + 4);
        else if (!std::strncmp(s, "--n=", 4)) n = atoi(s + 4);
        else if (!std::strncmp(s, "--bytes=", 8)) bytes = strtoull(s + 8, nullptr, 0);
        else if (!std::strncmp(s, "--spin-ms=", 10)) spinMs = atof(s + 10);
        else if (!std::strncmp(s, "--cap-ms=", 9)) capMs = atof(s + 9);
        else if (!std::strncmp(s, "--flush-ms=", 11)) flushMs = atof(s + 11);
        else if (!std::strncmp(s, "--x-cap-ms=", 11)) xCapMs = atof(s + 11);
        else if (!std::strncmp(s, "--arms=", 7)) arms = s + 7;
        else if (!std::strncmp(s, "--dirs=", 7)) dirs = s + 7;      // both | ab | ba
        else if (!std::strncmp(s, "--holder=", 9)) holder = s + 9;
        else { std::fprintf(stderr, "unknown option %s\n", s); return 1; }
    }
    if (capMs <= spinMs + 200) { std::fprintf(stderr, "cap-ms must exceed spin-ms by >200 ms\n"); return 1; }
    std::printf("== barlink visibility probe 1023 == a=%d b=%d n=%d bytes=%llu spin=%.0fms cap=%.0fms "
                "flush@%.0fms arms=%s dirs=%s\n", a, b, n, bytes, spinMs, capMs, flushMs, arms.c_str(), dirs.c_str());
    RT(cudaSetDevice(a)); RT(cudaFree(nullptr));
    RT(cudaSetDevice(b)); RT(cudaFree(nullptr));

    Side AB, BA;
    int rc = sideBuild(AB, a, b, bytes, holder);
    if (rc) return rc;
    rc = sideBuild(BA, b, a, bytes, holder);
    if (rc) { regionAbbauen(AB.reg, nullptr); return rc; }


    const u64 spinNs = (u64)(spinMs * 1e6), capNs = (u64)(capMs * 1e6);
    const double waitMs = capMs + 3000;
    u64 roundCtr = ((u64)time(nullptr) << 20);
    Side *dirsList[2]; int nd = 0;
    if (dirs == "both" || dirs == "ab") dirsList[nd++] = &AB;
    if (dirs == "both" || dirs == "ba") dirsList[nd++] = &BA;

    for (int trial = 0; trial < n; ++trial) {
        for (char arm : arms) {
            for (int d = 0; d < nd; ++d) {
                Side &S = *dirsList[d];
                Side &O = (&S == &AB) ? BA : AB;
                const u64 round = ++roundCtr;
                const double t0 = jetztMs();
                double flushAt = -1;
                const char tag[2] = {arm, 0};
                if (arm == 'B') {
                    if (d) continue;                       // one bidirectional trial per round of arms
                    const u64 round2 = ++roundCtr;
                    sideLaunchRecv(S, round, capNs); sideLaunchRecv(O, round2, capNs);
                    sideLaunchSend(S, round, ST_WT, TAIL_SPIN, 0, spinNs);
                    sideLaunchSend(O, round2, ST_WT, TAIL_SPIN, 0, spinNs);
                    int w = stromWarten(S.sr, waitMs, "recv") | stromWarten(O.sr, waitMs, "recv")
                          | stromWarten(S.ss, waitMs, "send") | stromWarten(O.ss, waitMs, "send");
                    if (w) return w;
                    report(tag, S, trial, -1, jetztMs() - t0);
                    report(tag, O, trial, -1, jetztMs() - t0);
                    continue;
                }
                if (arm == 'X') {
                    if (xCapMs + 300 >= spinMs) { std::fprintf(stderr, "arm X needs spin-ms > x-cap-ms + 300\n"); return 1; }
                    sideLaunchRecv(S, round, (u64)(xCapMs * 1e6));
                    sideLaunchSend(S, round, ST_WT, TAIL_SPIN, 0, spinNs);
                    int w = stromWarten(S.sr, waitMs, "recv");
                    if (w) return w;
                    const double tObs = jetztMs() - t0;
                    const bool aliveBefore = cudaStreamQuery(S.ss) == cudaErrorNotReady;
                    // (1) copy engine
                    std::vector<uint4> img(S.n4);
                    RT(cudaSetDevice(S.ord_dst));
                    RT(cudaMemcpy(img.data(), (const void *)S.reg.dptr, S.slotBytes, cudaMemcpyDeviceToHost));
                    u64 badCE = 0; for (u64 i = 0; i < S.n4; ++i) if (!vpEq(img[i], vpPattern(round, i))) ++badCE;
                    // (2) CPU through BAR1 (uncached PCIe reads on the receiver's last hop)
                    u64 badCPU = 0;
                    { const volatile uint4 *bp = (const volatile uint4 *)S.reg.barHost;
                      for (u64 i = 0; i < S.n4; ++i) { uint4 g; g.x = bp[i].x; g.y = bp[i].y; g.z = bp[i].z; g.w = bp[i].w;
                                                       if (!vpEq(g, vpPattern(round, i))) ++badCPU; } }
                    // (1b) copy engine again, after the CPU read
                    RT(cudaMemcpy(img.data(), (const void *)S.reg.dptr, S.slotBytes, cudaMemcpyDeviceToHost));
                    u64 badCE2 = 0; for (u64 i = 0; i < S.n4; ++i) if (!vpEq(img[i], vpPattern(round, i))) ++badCE2;
                    // (3) fresh receiver kernel (flag is already there)
                    u64 r1[VR_N]; RT(cudaMemcpy(r1, S.res, sizeof(r1), cudaMemcpyDeviceToHost));
                    sideLaunchRecv(S, round, 50000000ull);
                    w = stromWarten(S.sr, waitMs, "recv2"); if (w) return w;
                    u64 r2[VR_N]; RT(cudaMemcpy(r2, S.res, sizeof(r2), cudaMemcpyDeviceToHost));
                    const bool aliveAfter = cudaStreamQuery(S.ss) == cudaErrorNotReady;
                    const double tEnd = jetztMs() - t0;
                    w = stromWarten(S.ss, waitMs, "send"); if (w) return w;
                    std::printf("ARM X | %s | trial %d | running-kernel miss_end=%llu (first scan %llu) | at %.0f ms, sender alive=%d: "
                                "CE bad=%llu, CPU-BAR1 bad=%llu, CE-after-CPU bad=%llu, fresh-kernel miss_first=%llu complete=%d | "
                                "observers done at %.0f ms, sender alive=%d\n",
                                S.name.c_str(), trial, r1[VR_MISS_END], r1[VR_MISS_FIRST], tObs, (int)aliveBefore,
                                badCE, badCPU, badCE2, r2[VR_MISS_FIRST], r2[VR_T_COMPLETE] ? 1 : 0, tEnd, (int)aliveAfter);
                    std::fflush(stdout);
                    continue;
                }
                sideLaunchRecv(S, round, capNs);
                switch (arm) {
                case 'E': sideLaunchSend(S, round, ST_WT,   TAIL_EXIT, 0, 0);      break;
                case 'S': sideLaunchSend(S, round, ST_WT,   TAIL_SPIN, 0, spinNs); break;
                case 'M': sideLaunchSend(S, round, ST_MMIO, TAIL_SPIN, 0, spinNs); break;
                case 'R': sideLaunchSend(S, round, ST_WT,   TAIL_SPIN, 1, spinNs); break;
                case 'A': sideLaunchSend(S, round, ST_WT,   TAIL_SPIN_MONITOR, 0, spinNs); break;
                case 'F': sideLaunchSend(S, round, ST_WT,   TAIL_SPIN, 0, spinNs); break;
                default: std::fprintf(stderr, "unknown arm %c\n", arm); return 1;
                }
                if (arm == 'F') {
                    while (jetztMs() - t0 < flushMs) usleep(200);
                    {
                        DRV(cuCtxPushCurrent(S.ctxDst));
                        CUresult fr = cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                                                 CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
                        CUcontext dummy; cuCtxPopCurrent(&dummy);
                        flushAt = jetztMs() - t0;
                        if (fr != CUDA_SUCCESS) std::printf("  cuFlushGPUDirectRDMAWrites -> %d\n", (int)fr);
                    }
                }
                int w = stromWarten(S.sr, waitMs, "recv") | stromWarten(S.ss, waitMs, "send");
                if (w) return w;
                report(tag, S, trial, flushAt, jetztMs() - t0);
            }
        }
    }
    regionAbbauen(BA.reg, nullptr);
    regionAbbauen(AB.reg, nullptr);
    std::printf("== done ==\n");
    return 0;
}

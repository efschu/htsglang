/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 efschu
 *
 * Eigenstaendiger Code. Verwendet ausschliesslich oeffentliche APIs:
 *  - CUDA Driver API (NVIDIA CUDA Toolkit, normal installiert)
 *  - die Ioctl-Schnittstelle der quelloffenen NVIDIA-Kernelmodule
 *    (NVIDIA/open-gpu-kernel-modules, Dual MIT/GPL) -- Header werden
 *    per #include eingebunden, NICHT mit diesem Code verteilt.
 *  - libibverbs / rdma-core (Dual BSD/GPL)
 *
 * Es wird nichts gepatcht, ersetzt oder nachgebaut.
 *
 * Herkunft: uebernommen aus der Uebergabe /spinning/gdr-uebergabe/, unter dem
 * dortigen MIT-Header. Erweiterung fuer den GDR-Fensterlauf (probe/gdr-window):
 * optionaler --ro-Schalter fuer IBV_ACCESS_RELAXED_ORDERING auf der
 * Nutzlast-MR, siehe Abschnitt weiter unten. Ohne --ro ist das Verhalten
 * byte-identisch zum uebernommenen Original.
 */

/*
  * Latenz-Benchmark: GDR (RDMA direkt in/aus GPU-VRAM) gegen
 * HOST-STAGING (der heute gefahrene Weg: cuMemcpyDtoH -> RDMA -> cuMemcpyHtoD).
 *
 * Ping-Pong ueber RC-QP. Die Nutzlast liegt je nach Modus in einer GPU-MR
 * (Weg-B-dmabuf) oder in einer Host-MR. Das Synchronisations-Flag liegt in
 * BEIDEN Modi im Host-Speicher - sonst waere der Vergleich unfair, weil die
 * CPU GPU-Speicher nicht sinnvoll pollen kann.
 *
 * Gemessen wird der halbe Round-trip (einseitige Latenz) je Iteration;
 * berichtet werden Median, p10 und p90 ueber alle Iterationen sowie der
 * A-vs-A-Rauschboden (zwei Laeufe desselben Modus).
 *
 * Bauen (beide Rollen brauchen CUDA nur, wenn ihre Nutzlast im VRAM liegt):
 *   gcc -O2 -DWITH_CUDA ... -lcuda -libverbs
 *
 * Aufruf:
 *   Server: ./wegB_bench server <cuda-ord|-1> <nic> <port> <gdr|stage> [--ro]
 *   Client: ./wegB_bench client <server-ip> <nic> <port> <gdr|stage> <cuda-ord|-1> [--ro]
 *   cuda-ord = -1 bedeutet: Nutzlast im Host-Speicher (kein CUDA noetig).
 *   --ro = optional, darf an beliebiger Position stehen, wird vor der
 *          sonst unveraenderten Positions-Auswertung herausgefiltert.
 *          Setzt IBV_ACCESS_RELAXED_ORDERING auf der Nutzlast-MR (nicht auf
 *          der Flag-MR -- deren Reihenfolge muss fuer das Polling strikt
 *          bleiben). Default: aus, heutiges Verhalten unveraendert. Wird
 *          --ro verlangt, aber das installierte rdma-core kennt das Symbol
 *          IBV_ACCESS_RELAXED_ORDERING nicht (Buildzeit-Check), laeuft das
 *          Programm sauber ohne RO weiter und meldet das einmalig auf stderr
 *          und in der Statuszeile -- kein Abbruch.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#ifdef WITH_CUDA
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
    printf("[FAIL] %s -> %d (%s)\n", #x, _r, _s ? _s : "?"); exit(1); } } while (0)
static int nv_ioctl(int fd, int nr, void *p, size_t size)
{ return ioctl(fd, _IOC(_IOC_READ|_IOC_WRITE, NV_IOCTL_MAGIC, nr, (unsigned)size), p); }
#endif

#define MAXSZ    (4u*1024u*1024u)
#define WARMUP   200
#define ITERS    2000

static const size_t SIZES[] = { 8, 4096, 65536, 1048576 };
#define NSIZES (sizeof(SIZES)/sizeof(SIZES[0]))

/* ---- --ro: IBV_ACCESS_RELAXED_ORDERING, Buildzeit-Symbol-Check ----
 * IBV_ACCESS_RELAXED_ORDERING ist in verbs.h ein enum-Wert, kein Makro --
 * "#if defined(...)" waere hier immer falsch, unabhaengig von der
 * installierten rdma-core-Version, weil defined() nur #define-Makros sieht.
 * Der eigentliche Check laeuft daher im Buildskript (build.sh) als kleine
 * Testkompilierung; das Ergebnis kommt als -DHAVE_IBV_ACCESS_RELAXED_ORDERING=1
 * herein. Ohne diesen Schalter (aelteres rdma-core) bleibt es bei 0 -- der
 * Code kompiliert trotzdem unveraendert, der Effekt ist schlicht "kein RO". */
#ifndef HAVE_IBV_ACCESS_RELAXED_ORDERING
#define HAVE_IBV_ACCESS_RELAXED_ORDERING 0
#endif
#if HAVE_IBV_ACCESS_RELAXED_ORDERING
#define HAVE_RO_SYMBOL 1
#define RO_ACCESS_BIT  IBV_ACCESS_RELAXED_ORDERING
#else
#define HAVE_RO_SYMBOL 0
#define RO_ACCESS_BIT  0
#endif

static int g_ro_requested = 0;   /* --ro auf der Kommandozeile gesehen */

/* Effektives Access-Bit fuer die Nutzlast-MR: nur gesetzt, wenn sowohl vom
 * Nutzer angefordert als auch zur Buildzeit verfuegbar. */
static uint32_t ro_payload_bit(void)
{
    return (g_ro_requested && HAVE_RO_SYMBOL) ? (uint32_t)RO_ACCESS_BIT : 0u;
}

/* Filtert "--ro" aus argv heraus (beliebige Position), damit die sonst
 * unveraenderte Positions-Auswertung darunter exakt wie im Original laeuft.
 * Gibt das neue argc zurueck. */
static int strip_ro_flag(int argc, char **argv)
{
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ro") == 0) {
            g_ro_requested = 1;
            continue;
        }
        argv[out++] = argv[i];
    }
    return out;
}

struct conn_info {
    uint32_t qpn, psn, rkey_pay, rkey_flag;
    uint64_t addr_pay, addr_flag;
    uint8_t  gid[16];
};

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int gid_is_v2(const char *dev, int idx)
{
    char p[256], b[64] = {0}; FILE *f;
    snprintf(p, sizeof(p), "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", dev, idx);
    f = fopen(p, "r"); if (!f) return 0;
    if (!fgets(b, sizeof(b), f)) b[0] = 0;
    fclose(f); return strstr(b, "RoCE v2") != NULL;
}

static int find_gid_index(struct ibv_context *ctx, const char *dev, uint8_t out[16])
{
    int fb = -1;
    for (int i = 0; i < 64; i++) {
        union ibv_gid g;
        if (ibv_query_gid(ctx, 1, i, &g)) continue;
        int pz = 1; for (int k = 0; k < 10; k++) if (g.raw[k]) { pz = 0; break; }
        if (!pz || g.raw[10] != 0xff || g.raw[11] != 0xff) continue;
        int nz = 0; for (int k = 12; k < 16; k++) if (g.raw[k]) nz = 1;
        if (!nz) continue;
        if (gid_is_v2(dev, i)) { memcpy(out, g.raw, 16); return i; }
        if (fb < 0) { fb = i; memcpy(out, g.raw, 16); }
    }
    return fb;
}

static int tcp_exchange(int is_server, const char *ip, int port,
                        const struct conn_info *mine, struct conn_info *peer)
{
    int sock = -1, lst = -1, one = 1;
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET; a.sin_port = htons(port);
    if (is_server) {
        lst = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(lst, (struct sockaddr*)&a, sizeof(a))) { perror("bind"); return -1; }
        if (listen(lst, 1)) { perror("listen"); return -1; }
        sock = accept(lst, NULL, NULL);
        if (sock < 0) { perror("accept"); return -1; }
        close(lst);
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        a.sin_addr.s_addr = inet_addr(ip);
        if (connect(sock, (struct sockaddr*)&a, sizeof(a))) { perror("connect"); return -1; }
    }
    if (write(sock, mine, sizeof(*mine)) != (ssize_t)sizeof(*mine)) return -1;
    if (read(sock, peer, sizeof(*peer)) != (ssize_t)sizeof(*peer)) return -1;
    return sock;
}

static int qp_rtr_rts(struct ibv_qp *qp, const struct conn_info *peer,
                      int sgid, uint32_t psn)
{
    struct ibv_qp_attr at; memset(&at, 0, sizeof(at));
    at.qp_state = IBV_QPS_RTR; at.path_mtu = IBV_MTU_1024;
    at.dest_qp_num = peer->qpn; at.rq_psn = peer->psn;
    at.max_dest_rd_atomic = 1; at.min_rnr_timer = 12;
    at.ah_attr.is_global = 1; at.ah_attr.port_num = 1;
    at.ah_attr.grh.hop_limit = 16; at.ah_attr.grh.sgid_index = sgid;
    memcpy(at.ah_attr.grh.dgid.raw, peer->gid, 16);
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|
                               IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|
                               IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)) {
        perror("RTR"); return -1; }
    memset(&at, 0, sizeof(at));
    at.qp_state = IBV_QPS_RTS; at.timeout = 14; at.retry_cnt = 7;
    at.rnr_retry = 7; at.sq_psn = psn; at.max_rd_atomic = 1;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                               IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)) {
        perror("RTS"); return -1; }
    return 0;
}

#ifdef WITH_CUDA
/* dmabuf-Kette: VMM-Puffer -> Objekt-fd -> eigener RM-Client -> dmabuf-fd */
static int gpu_dmabuf(int ord, size_t size, CUdeviceptr *dptr,
                      CUmemGenericAllocationHandle *mh, char *namebuf, size_t nblen)
{
    CUdevice dev; int bus = -1;
    CHK(cuInit(0));
    CHK(cuDeviceGet(&dev, ord));
    CHK(cuDeviceGetName(namebuf, (int)nblen, dev));
    CHK(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CUcontext c; CHK(cuDevicePrimaryCtxRetain(&c, dev)); CHK(cuCtxSetCurrent(c));

    CUmemAllocationProp prop; memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    CHK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (size < gran) size = gran;
    CHK(cuMemCreate(mh, size, &prop, 0));
    CHK(cuMemAddressReserve(dptr, size, 0, 0, 0));
    CHK(cuMemMap(*dptr, size, 0, *mh, 0));
    CUmemAccessDesc ad; memset(&ad, 0, sizeof(ad));
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHK(cuMemSetAccess(*dptr, size, &ad, 1));
    CHK(cuMemsetD8(*dptr, 0, size)); CHK(cuCtxSynchronize());

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
      if (nv_ioctl(ctl, NV_ESC_RM_ALLOC,&a,sizeof(a))<0||a.status) return -1; hCl=a.hObjectNew; }
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
      p.totalSize=size; p.mappingType=NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
      p.bAllowMmap=NV_FALSE; p.handles[0]=hMem; p.offsets[0]=0; p.sizes[0]=size;
      if (nv_ioctl(dfd, NV_ESC_EXPORT_TO_DMABUF_FD,&p,sizeof(p))<0||p.status) {
          fprintf(stderr,"DMABUF status 0x%x\n", p.status); return -1; }
      return p.fd; }
}
#endif

int main(int argc, char **argv)
{
    argc = strip_ro_flag(argc, argv);   /* --ro raus, Rest unveraendert positional */

    if (argc < 6) {
        fprintf(stderr,
          "Server: %s server <cuda-ord|-1> <nic> <port> <gdr|stage> [--ro]\n"
          "Client: %s client <server-ip> <nic> <port> <gdr|stage> <cuda-ord|-1> [--ro]\n",
          argv[0], argv[0]);
        return 1;
    }
    if (g_ro_requested && !HAVE_RO_SYMBOL) {
        fprintf(stderr,
          "[warn] --ro angefordert, aber IBV_ACCESS_RELAXED_ORDERING ist beim "
          "Bauen nicht definiert (rdma-core zu alt) -- laeuft ohne RO weiter\n");
    }
    int is_server = strcmp(argv[1], "server") == 0;
    const char *nic = argv[3];
    int port = atoi(argv[4]);
    int gdr = strcmp(argv[5], "gdr") == 0;
    int ord = is_server ? atoi(argv[2]) : (argc > 6 ? atoi(argv[6]) : -1);

    int ndev = 0;
    struct ibv_device **list = ibv_get_device_list(&ndev), *ch = NULL;
    if (!list) { perror("dev_list"); return 1; }
    for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(list[i]), nic)) ch = list[i];
    if (!ch) { fprintf(stderr, "NIC %s fehlt\n", nic); return 1; }
    struct ibv_context *vctx = ibv_open_device(ch);
    struct ibv_pd *pd = ibv_alloc_pd(vctx);
    if (!vctx || !pd) { perror("open/pd"); return 1; }

    /* ---- Nutzlast-Region: GPU (gdr) oder Host (stage / ord<0) ---- */
    struct ibv_mr *mr_pay = NULL;
    void *host_pay = NULL;
    uint64_t pay_addr = 0;
    char gpuname[128] = "(host)";
#ifdef WITH_CUDA
    CUdeviceptr dptr = 0; CUmemGenericAllocationHandle mh = 0;
#endif

    if (gdr && ord >= 0) {
#ifdef WITH_CUDA
        int dfd = gpu_dmabuf(ord, MAXSZ, &dptr, &mh, gpuname, sizeof(gpuname));
        if (dfd < 0) { fprintf(stderr, "GPU-dmabuf fehlgeschlagen\n"); return 1; }
        mr_pay = ibv_reg_dmabuf_mr(pd, 0, MAXSZ, 0, dfd,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ
                    |ro_payload_bit());
        if (!mr_pay) { fprintf(stderr, "reg_dmabuf errno=%d\n", errno); return 1; }
        pay_addr = 0;   /* iova 0 */
#else
        fprintf(stderr, "gdr braucht -DWITH_CUDA\n"); return 1;
#endif
    } else {
        if (posix_memalign(&host_pay, 4096, MAXSZ)) { perror("memalign"); return 1; }
        memset(host_pay, 0, MAXSZ);
        mr_pay = ibv_reg_mr(pd, host_pay, MAXSZ,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ
                    |ro_payload_bit());
        if (!mr_pay) { perror("reg_mr pay"); return 1; }
        pay_addr = (uint64_t)(uintptr_t)host_pay;
#ifdef WITH_CUDA
        /* stage-Modus mit GPU: separater VRAM-Puffer, der pro Iteration
         * ueber den Host kopiert wird - das ist der heutige Weg. */
        if (ord >= 0) {
            CUdevice dv; CUcontext cc;
            CHK(cuInit(0)); CHK(cuDeviceGet(&dv, ord));
            CHK(cuDeviceGetName(gpuname, sizeof(gpuname), dv));
            CHK(cuDevicePrimaryCtxRetain(&cc, dv)); CHK(cuCtxSetCurrent(cc));
            CHK(cuMemAlloc(&dptr, MAXSZ));
            CHK(cuMemsetD8(dptr, 0, MAXSZ)); CHK(cuCtxSynchronize());
        }
#endif
    }

    /* ---- Flag-Region: IMMER Host (fair fuer beide Modi) ---- */
    void *flagbuf = NULL;
    if (posix_memalign(&flagbuf, 4096, 4096)) { perror("memalign flag"); return 1; }
    memset(flagbuf, 0, 4096);
    struct ibv_mr *mr_flag = ibv_reg_mr(pd, flagbuf, 4096,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
    if (!mr_flag) { perror("reg_mr flag"); return 1; }
    volatile uint64_t *flag = (volatile uint64_t *)flagbuf;

    struct ibv_cq *cq = ibv_create_cq(vctx, 64, NULL, NULL, 0);
    struct ibv_qp_init_attr ia; memset(&ia, 0, sizeof(ia));
    ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 64; ia.cap.max_recv_wr = 64;
    ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &ia);
    if (!cq || !qp) { perror("cq/qp"); return 1; }
    { struct ibv_qp_attr at; memset(&at, 0, sizeof(at));
      at.qp_state = IBV_QPS_INIT; at.pkey_index = 0; at.port_num = 1;
      at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
      if (ibv_modify_qp(qp,&at,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) {
          perror("INIT"); return 1; } }

    uint8_t mygid[16];
    int gidx = find_gid_index(vctx, nic, mygid);
    if (gidx < 0) { fprintf(stderr, "kein IPv4-GID\n"); return 1; }

    struct conn_info mine, peer;
    memset(&mine, 0, sizeof(mine)); memset(&peer, 0, sizeof(peer));
    mine.qpn = qp->qp_num; mine.psn = 0x4321;
    mine.rkey_pay = mr_pay->rkey;   mine.addr_pay = pay_addr;
    mine.rkey_flag = mr_flag->rkey; mine.addr_flag = (uint64_t)(uintptr_t)flagbuf;
    memcpy(mine.gid, mygid, 16);

    int sock = tcp_exchange(is_server, is_server ? NULL : argv[2], port, &mine, &peer);
    if (sock < 0) return 1;
    if (qp_rtr_rts(qp, &peer, gidx, mine.psn)) return 1;

    printf("[%s] Modus=%s  GPU=%s  sgid=%d  RO=%s\n", is_server ? "srv" : "cli",
           gdr ? "GDR" : "STAGE", gpuname, gidx,
           !g_ro_requested ? "off" : (HAVE_RO_SYMBOL ? "on" : "unsupported"));
    fflush(stdout);

    double *lat = malloc(sizeof(double) * ITERS);

    for (size_t si = 0; si < NSIZES; si++) {
        size_t sz = SIZES[si];
        uint64_t seq = 0;

        for (int it = 0; it < WARMUP + ITERS; it++) {
            if (!is_server) {
                double t0 = now_us();
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) {           /* heutiger Weg: DtoH */
                    cuMemcpyDtoH(host_pay, dptr, sz);
                }
#endif
                struct ibv_sge sg; struct ibv_send_wr wr, *bad;
                if (sz > 0) {
                    memset(&sg, 0, sizeof(sg));
                    sg.addr = gdr ? 0 : (uint64_t)(uintptr_t)host_pay;
                    sg.length = sz; sg.lkey = mr_pay->lkey;
                    memset(&wr, 0, sizeof(wr));
                    wr.wr_id = 1; wr.sg_list = &sg; wr.num_sge = 1;
                    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = 0;
                    wr.wr.rdma.remote_addr = peer.addr_pay;
                    wr.wr.rdma.rkey = peer.rkey_pay;
                    if (ibv_post_send(qp, &wr, &bad)) { perror("post pay"); return 1; }
                }
                seq++;
                *(uint64_t *)((char *)flagbuf + 64) = seq;
                struct ibv_sge fs; struct ibv_send_wr fw;
                memset(&fs, 0, sizeof(fs));
                fs.addr = (uint64_t)(uintptr_t)flagbuf + 64;
                fs.length = 8; fs.lkey = mr_flag->lkey;
                memset(&fw, 0, sizeof(fw));
                fw.wr_id = 2; fw.sg_list = &fs; fw.num_sge = 1;
                fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = IBV_SEND_SIGNALED;
                /* Client -> Server: Offset 64, dort pollt der Server (flag[8]) */
                fw.wr.rdma.remote_addr = peer.addr_flag + 64;
                fw.wr.rdma.rkey = peer.rkey_flag;
                if (ibv_post_send(qp, &fw, &bad)) { perror("post flag"); return 1; }
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) == 0) { }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "WC %s\n", ibv_wc_status_str(wc.status)); return 1; }
                /* auf Antwort warten */
                while (flag[0] != seq) { __builtin_ia32_pause(); }
                double t1 = now_us();
                if (it >= WARMUP) lat[it - WARMUP] = (t1 - t0) / 2.0;
            } else {
                seq++;
                while (flag[8] != seq) { __builtin_ia32_pause(); }   /* Offset 64B = idx 8 */
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) {           /* heutiger Weg: HtoD */
                    cuMemcpyHtoD(dptr, host_pay, sz);
                }
#endif
                *(uint64_t *)flagbuf = seq;
                struct ibv_sge fs; struct ibv_send_wr fw, *bad;
                memset(&fs, 0, sizeof(fs));
                fs.addr = (uint64_t)(uintptr_t)flagbuf; fs.length = 8; fs.lkey = mr_flag->lkey;
                memset(&fw, 0, sizeof(fw));
                fw.wr_id = 3; fw.sg_list = &fs; fw.num_sge = 1;
                fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = IBV_SEND_SIGNALED;
                fw.wr.rdma.remote_addr = peer.addr_flag; fw.wr.rdma.rkey = peer.rkey_flag;
                if (ibv_post_send(qp, &fw, &bad)) { perror("srv post"); return 1; }
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) == 0) { }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "srv WC %s\n", ibv_wc_status_str(wc.status)); return 1; }
            }
        }

        if (!is_server) {
            qsort(lat, ITERS, sizeof(double), cmp_d);
            printf("%-9s %8zu B   p10 %8.2f us   MEDIAN %8.2f us   p90 %8.2f us\n",
                   gdr ? "GDR" : "STAGE", sz,
                   lat[ITERS/10], lat[ITERS/2], lat[(ITERS*9)/10]);
            fflush(stdout);
        }
    }

    close(sock);
    return 0;
}

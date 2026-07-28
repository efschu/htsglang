/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 efschu (dmabuf-Kette, Verifikationsmuster) /
 *                     Erweiterung fuer probe/gdr-window
 *
 * Eigenstaendiger Code, wie die uebrigen Dateien in diesem Verzeichnis.
 * Verwendet ausschliesslich oeffentliche APIs (CUDA Driver API, die Ioctl-
 * Schnittstelle der quelloffenen NVIDIA-Kernelmodule, libibverbs). Es wird
 * nichts gepatcht, ersetzt oder nachgebaut.
 *
 * STUFE 0 -- Smoke-Test "geht es hier ueberhaupt", VOR jeder Leiter:
 *
 *  (a) regcheck: ibv_reg_dmabuf_mr auf einer kleinen VMM-Allokation einer
 *      per PCI-Ordinal gewaehlten Karte -- ohne Netzwerk, ohne Gegenstelle,
 *      reiner Registrierungs-Test. Deckt "die dmabuf-Kette + rdma-core auf
 *      dieser Karte ueberhaupt" ab, unabhaengig von jeder Transferfrage.
 *
 *  (b) server/client, GPU = Server (ord>=0): Ziel-Richtung. Client (Host-
 *      Speicher) schreibt ein Muster per RDMA-WRITE in den per dmabuf
 *      registrierten GPU-Speicher des Servers. Verifikation wie im
 *      Original-Muster aus gpurdma_03_transfer.c: GPU-Speicher vorher mit
 *      FILLVAL vorbefuellt, Client schreibt PATVAL + Sequenzmarken pro
 *      4-KiB-Seite, Server liest per cuMemcpyDtoH zurueck und vergleicht.
 *
 *  (c) server/client, GPU = Client (ord>=0): Quell-Richtung. Der Client haelt
 *      die Nutzlast in seinem eigenen dmabuf-registrierten GPU-Speicher
 *      (dort per cuMemcpyHtoD VOR dem Registrieren mit PATVAL+Marken befuellt)
 *      und schreibt per RDMA-WRITE in den Host-Speicher des Servers. Der
 *      Server prueft direkt per memcmp -- kein GPU-Readback noetig, da das
 *      Ziel hier Host-Speicher ist.
 *
 * In beiden Faellen (b)/(c) schreibt IMMER der Client (gleiche Rollen-
 * Konvention wie gpurdma_03/04); welche Seite die GPU haelt, entscheidet die
 * getestete Richtung -- kein separates "Modus"-Argument noetig.
 *
 * Aufruf:
 *   Registrierungs-Check (kein Netzwerk):
 *     ./gpurdma_00_smoke regcheck <cuda-ord> <nic>
 *
 *   Transfer, Ziel-Richtung (NIC -> GPU):
 *     Server: ./gpurdma_00_smoke server <cuda-ord> <nic> <port>
 *     Client: ./gpurdma_00_smoke client <server-ip> <nic> <port> -1
 *
 *   Transfer, Quell-Richtung (GPU -> NIC):
 *     Server: ./gpurdma_00_smoke server -1 <nic> <port>
 *     Client: ./gpurdma_00_smoke client <server-ip> <nic> <port> <cuda-ord>
 *
 * Ausgabe: eine Zeile "RESULT <name> PASS|FAIL errno=<n> <detail>" auf
 * stdout, maschinenlesbar fuer den Runner.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
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
static int nv_ioctl(int fd, int nr, void *p, size_t size)
{ return ioctl(fd, _IOC(_IOC_READ|_IOC_WRITE, NV_IOCTL_MAGIC, nr, (unsigned)size), p); }

#define BUFSZ    16384u    /* "wenige KiB", Stufe-0-Nutzlast */
#define FILLVAL  0xAA      /* Vorzustand im GPU-Speicher (Zielrichtung) */
#define PATVAL   0x5A      /* was geschrieben wird */

struct conn_info {
    uint32_t qpn, psn, rkey;
    uint64_t addr;
    uint8_t  gid[16];
};

/* ---- identisch zu gpurdma_03/04_*.c: GID/TCP/QP-Handshake ---- */
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

/* ---- dmabuf-Kette: identisch zu gpurdma_04_bench.c::gpu_dmabuf() ----
 * (VMM-Puffer -> Objekt-fd -> eigener RM-Client -> dmabuf-fd). Nicht neu
 * erfunden, nur wiederverwendet, damit Stufe 0 exakt denselben Pfad prueft,
 * den die Leitern danach benutzen. */
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
      p.totalSize=size; p.mappingType=NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
      p.bAllowMmap=NV_FALSE; p.handles[0]=hMem; p.offsets[0]=0; p.sizes[0]=size;
      if (nv_ioctl(dfd, NV_ESC_EXPORT_TO_DMABUF_FD,&p,sizeof(p))<0||p.status) {
          fprintf(stderr,"DMABUF status 0x%x\n", p.status); return -1; }
      return p.fd; }
}

static void result(const char *name, int pass, int err, const char *detail)
{
    printf("RESULT %s %s errno=%d %s\n", name, pass ? "PASS" : "FAIL", err, detail);
    fflush(stdout);
}

/* (a) reiner Registrierungs-Check, kein Netzwerk, keine Gegenstelle. */
static int mode_regcheck(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "regcheck <cuda-ord> <nic>\n"); return 1; }
    int ord = atoi(argv[2]);
    const char *nic = argv[3];

    int ndev = 0;
    struct ibv_device **list = ibv_get_device_list(&ndev), *ch = NULL;
    if (!list) { result("regcheck", 0, errno, "ibv_get_device_list"); return 1; }
    for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(list[i]), nic)) ch = list[i];
    if (!ch) { result("regcheck", 0, 0, "NIC nicht gefunden"); return 1; }
    struct ibv_context *vctx = ibv_open_device(ch);
    struct ibv_pd *pd = vctx ? ibv_alloc_pd(vctx) : NULL;
    if (!vctx || !pd) { result("regcheck", 0, errno, "open_device/alloc_pd"); return 1; }

    CUdeviceptr dptr = 0; CUmemGenericAllocationHandle mh = 0; char name[128] = {0};
    int dfd = gpu_dmabuf(ord, BUFSZ, &dptr, &mh, name, sizeof(name));
    if (dfd < 0) { result("regcheck", 0, errno, "gpu_dmabuf (VMM/Export/Import-Kette)"); return 1; }

    struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, BUFSZ, 0, dfd,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
    if (!mr) { result("regcheck", 0, errno, "ibv_reg_dmabuf_mr"); return 1; }

    char detail[128];
    snprintf(detail, sizeof(detail), "GPU=%s rkey=0x%08x", name, mr->rkey);
    result("regcheck", 1, 0, detail);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
          "Regcheck: %s regcheck <cuda-ord> <nic>\n"
          "Server:   %s server <cuda-ord|-1> <nic> <port>\n"
          "Client:   %s client <server-ip> <nic> <port> <cuda-ord|-1>\n",
          argv[0], argv[0], argv[0]);
        return 1;
    }
    if (!strcmp(argv[1], "regcheck")) return mode_regcheck(argc, argv);

    if (argc < 5) { fprintf(stderr, "zu wenige Argumente, siehe Usage ohne Argumente\n"); return 1; }
    int is_server = !strcmp(argv[1], "server");
    const char *nic = argv[3];
    int port = atoi(argv[4]);
    int ord = is_server ? atoi(argv[2]) : (argc > 5 ? atoi(argv[5]) : -1);

    /* Konvention (wie gpurdma_04_bench): genau eine Seite hat ord>=0. Diese
     * Seite bestimmt die getestete Richtung -- unabhaengig davon, ob es die
     * eigene Rolle (server/client) ist. Wahrheitstabelle:
     *   is_server=1, ord>=0 (Server=GPU)  -> target (NIC schreibt INS GPU)
     *   is_server=1, ord<0  (Server=Host) -> source (Client=GPU schreibt RAUS)
     *   is_server=0, ord>=0 (Client=GPU)  -> source
     *   is_server=0, ord<0  (Client=Host) -> target
     * d.h. target genau dann, wenn (is_server == (ord>=0)). */
    const char *testname = (is_server == (ord >= 0))
        ? "target(NIC->GPU)" : "source(GPU->NIC)";

    int ndev = 0;
    struct ibv_device **list = ibv_get_device_list(&ndev), *ch = NULL;
    if (!list) { perror("dev_list"); return 1; }
    for (int i = 0; i < ndev; i++)
        if (!strcmp(ibv_get_device_name(list[i]), nic)) ch = list[i];
    if (!ch) { fprintf(stderr, "NIC %s fehlt\n", nic); return 1; }
    struct ibv_context *vctx = ibv_open_device(ch);
    struct ibv_pd *pd = ibv_alloc_pd(vctx);
    if (!vctx || !pd) { perror("open/pd"); return 1; }

    struct ibv_mr *mr = NULL;
    void *host_buf = NULL;
    uint64_t reg_addr = 0;
    char gpuname[128] = "(host)";
    CUdeviceptr dptr = 0; CUmemGenericAllocationHandle mh = 0;

    if (ord >= 0) {
        int dfd = gpu_dmabuf(ord, BUFSZ, &dptr, &mh, gpuname, sizeof(gpuname));
        if (dfd < 0) {
            result(testname, 0, errno, "gpu_dmabuf");
            return 1;
        }
        if (is_server) {
            /* Ziel-Richtung: Vorzustand setzen, damit ein spaeteres Muster
             * eindeutig neu ist (identisch zum Original-Muster). */
            CHK(cuMemsetD8(dptr, FILLVAL, BUFSZ)); CHK(cuCtxSynchronize());
        } else {
            /* Quell-Richtung: Muster + Sequenzmarken VOR dem RDMA-Write ins
             * eigene GPU-Memory schreiben (via Host-Staging-Puffer + HtoD --
             * das ist Setup, nicht Teil des gemessenen Pfads). */
            unsigned char *seed = malloc(BUFSZ);
            memset(seed, PATVAL, BUFSZ);
            for (size_t i = 0; i < BUFSZ; i += 4096)
                ((unsigned int *)(seed + i))[0] = 0xC0DE0000u + (unsigned)(i / 4096);
            CHK(cuMemcpyHtoD(dptr, seed, BUFSZ)); CHK(cuCtxSynchronize());
            free(seed);
        }
        mr = ibv_reg_dmabuf_mr(pd, 0, BUFSZ, 0, dfd,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
        if (!mr) {
            result(testname, 0, errno, "ibv_reg_dmabuf_mr");
            return 1;
        }
        reg_addr = 0;
    } else {
        if (posix_memalign(&host_buf, 4096, BUFSZ)) { perror("memalign"); return 1; }
        if (is_server) {
            memset(host_buf, 0, BUFSZ);   /* wird vom Client ueberschrieben */
        } else {
            memset(host_buf, PATVAL, BUFSZ);
            for (size_t i = 0; i < BUFSZ; i += 4096)
                ((unsigned int *)((char *)host_buf + i))[0] = 0xC0DE0000u + (unsigned)(i / 4096);
        }
        mr = ibv_reg_mr(pd, host_buf, BUFSZ,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
        if (!mr) { perror("reg_mr"); return 1; }
        reg_addr = (uint64_t)(uintptr_t)host_buf;
    }

    struct ibv_cq *cq = ibv_create_cq(vctx, 16, NULL, NULL, 0);
    struct ibv_qp_init_attr ia; memset(&ia, 0, sizeof(ia));
    ia.send_cq = cq; ia.recv_cq = cq; ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = 16; ia.cap.max_recv_wr = 16;
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
    mine.qpn = qp->qp_num; mine.psn = 0x2345;
    mine.rkey = mr->rkey; mine.addr = reg_addr;
    memcpy(mine.gid, mygid, 16);

    int sock = tcp_exchange(is_server, is_server ? NULL : argv[2], port, &mine, &peer);
    if (sock < 0) return 1;
    if (qp_rtr_rts(qp, &peer, gidx, mine.psn)) return 1;

    if (!is_server) {
        struct ibv_sge sge; memset(&sge, 0, sizeof(sge));
        sge.addr = (ord >= 0) ? 0 : (uint64_t)(uintptr_t)host_buf;
        sge.length = BUFSZ; sge.lkey = mr->lkey;
        struct ibv_send_wr wr, *bad = NULL; memset(&wr, 0, sizeof(wr));
        wr.wr_id = 1; wr.sg_list = &sge; wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = peer.addr; wr.wr.rdma.rkey = peer.rkey;
        if (ibv_post_send(qp, &wr, &bad)) {
            result(testname, 0, errno, "ibv_post_send"); return 1; }
        struct ibv_wc wc; int spins = 0;
        while (ibv_poll_cq(cq, 1, &wc) == 0) {
            if (++spins > 200000000) { result(testname, 0, ETIMEDOUT, "CQ-Timeout"); return 1; } }
        if (wc.status != IBV_WC_SUCCESS) {
            result(testname, 0, 0, ibv_wc_status_str(wc.status)); return 1; }
        char done = 'D';
        if (write(sock, &done, 1) != 1) { result(testname, 0, errno, "done-signal"); return 1; }
        /* Client meldet Erfolg des Sendens; das ENDGUELTIGE Urteil (Inhalt
         * angekommen) faellt der Server, der als einziger die Empfangsseite
         * sieht -- der Client druckt hier trotzdem PASS fuer "Write ok". */
        result(testname, 1, 0, "RDMA_WRITE abgeschlossen (Inhaltspruefung siehe Server)");
    } else {
        char d = 0;
        if (read(sock, &d, 1) != 1) { result(testname, 0, errno, "warte auf done-signal"); return 1; }
        if (ord >= 0) {
            unsigned char *check = malloc(BUFSZ);
            CHK(cuMemcpyDtoH(check, dptr, BUFSZ)); CHK(cuCtxSynchronize());
            size_t marks_ok = 0, marks_total = 0, bad_bytes = 0;
            for (size_t i = 0; i < BUFSZ; i += 4096) {
                unsigned int want = 0xC0DE0000u + (unsigned)(i / 4096);
                unsigned int got = ((unsigned int *)(check + i))[0];
                marks_total++; if (got == want) marks_ok++;
            }
            for (size_t i = 0; i < BUFSZ; i++) {
                if ((i % 4096) < 4) continue;
                if (check[i] != PATVAL) bad_bytes++;
            }
            char detail[128];
            snprintf(detail, sizeof(detail), "marks_ok=%zu/%zu bad_bytes=%zu",
                     marks_ok, marks_total, bad_bytes);
            result(testname, marks_ok == marks_total && bad_bytes == 0, 0, detail);
            free(check);
        } else {
            size_t marks_ok = 0, marks_total = 0, bad_bytes = 0;
            unsigned char *check = (unsigned char *)host_buf;
            for (size_t i = 0; i < BUFSZ; i += 4096) {
                unsigned int want = 0xC0DE0000u + (unsigned)(i / 4096);
                unsigned int got = ((unsigned int *)(check + i))[0];
                marks_total++; if (got == want) marks_ok++;
            }
            for (size_t i = 0; i < BUFSZ; i++) {
                if ((i % 4096) < 4) continue;
                if (check[i] != PATVAL) bad_bytes++;
            }
            char detail[128];
            snprintf(detail, sizeof(detail), "marks_ok=%zu/%zu bad_bytes=%zu",
                     marks_ok, marks_total, bad_bytes);
            result(testname, marks_ok == marks_total && bad_bytes == 0, 0, detail);
        }
    }
    close(sock);
    return 0;
}

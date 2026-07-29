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
 *
 * Zweite Erweiterung (Cross-Rig-Groessenleiter): freie Groessenliste,
 * Pipelining-Tiefe (mehrere ausstehende Nutzlast-WRs je Runde), frei
 * waehlbare Iterations-/Aufwaermzahl, und ein CUDA-freier Build fuer die
 * Seite, deren Nutzlast im Hauptspeicher liegt. Siehe Abschnitt
 * "Pipelining-Tiefe" weiter unten.
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
 *   gcc -O2 -DWITH_CUDA ... -lcuda -libverbs      (Seite mit GPU-Nutzlast)
 *   gcc -O2 ... -libverbs                          (Seite nur mit Host-Nutzlast)
 * Der zweite Fall braucht weder CUDA-Toolkit noch die Kernelmodul-Header;
 * alles GPU-Bezogene haengt an -DWITH_CUDA. Ohne CUDA ist nur <cuda-ord> = -1
 * zulaessig, ein GPU-Ordinal wird mit klarer Meldung abgelehnt.
 *
 * Aufruf:
 *   Server: ./wegB_bench server <cuda-ord|-1> <nic> <port> <gdr|stage> [opts]
 *   Client: ./wegB_bench client <server-ip> <nic> <port> <gdr|stage> <cuda-ord|-1> [opts]
 *   cuda-ord = -1 bedeutet: Nutzlast im Host-Speicher (kein CUDA noetig).
 *
 * Optionen ([opts], duerfen an beliebiger Position stehen, werden vor der
 * sonst unveraenderten Positions-Auswertung herausgefiltert):
 *   --ro          Setzt IBV_ACCESS_RELAXED_ORDERING auf der Nutzlast-MR (nicht
 *                 auf der Flag-MR -- deren Reihenfolge muss fuer das Polling
 *                 strikt bleiben). Default: aus. Wird --ro verlangt, aber das
 *                 installierte rdma-core kennt das Symbol
 *                 IBV_ACCESS_RELAXED_ORDERING nicht (Buildzeit-Check), laeuft
 *                 das Programm sauber ohne RO weiter und meldet das einmalig
 *                 auf stderr und in der Statuszeile -- kein Abbruch.
 *   --sizes=A,B,C Groessenleiter in Bytes. Default 8,4096,65536,1048576.
 *   --depth=N     Pipelining-Tiefe, siehe unten. Default 1.
 *   --iters=N     Messiterationen je Groesse. Default 2000.
 *   --warmup=N    Aufwaermiterationen je Groesse. Default 200.
 * BEIDE Seiten muessen dieselben --sizes/--depth/--iters/--warmup bekommen;
 * ein Missverhaeltnis wird beim Verbindungsaufbau erkannt und hart gemeldet
 * (Magic/Version/Parameter werden mitausgetauscht), nicht stillschweigend
 * durchgelassen.
 *
 * ---- Pipelining-Tiefe (--depth) ----
 * Mit Tiefe N schickt der Client je Runde N Nutzlast-Schreiber (jeder an einen
 * eigenen Versatz d*sz im Zielpuffer, also N unabhaengige Transfers) als EINE
 * verkettete WR-Liste ab, gefolgt vom signalisierten Flag-Schreiber, und pollt
 * erst danach. Auf einem RC-QP werden die Completions in Reihenfolge
 * abgeliefert, das Flag-Completion impliziert also die vorangegangenen
 * Nutzlast-Schreiber.
 *
 * Wozu: Das ist der direkte Test "weiche Warteschlangen-Grenze gegen harten
 * Bandbreiten-Deckel". Sinkt die Zeit je Nachricht (Rundenzeit/N) mit
 * steigender Tiefe, war der Engpass die Zahl ausstehender Transaktionen
 * (Rueckstau, mit Knoepfen milderbar). Bleibt sie konstant, ist der Pfad
 * bandbreitenbegrenzt und die Tiefe hilft nicht.
 *
 * Wichtig fuer die Vergleichbarkeit der Tiefen-Achse: auch bei --depth=1 wird
 * exakt derselbe verkettete Pfad benutzt (eine Postung fuer Nutzlast+Flag).
 * Gegenueber dem uebernommenen Original, das Nutzlast und Flag mit zwei
 * getrennten ibv_post_send-Aufrufen abschickte, ist das eine Aenderung auch
 * im Default -- bewusst, denn sonst waere der Schritt von Tiefe 1 auf Tiefe 4
 * mit einem Codepfad-Wechsel vermengt. Die Neutralitaet dieses Schritts wird
 * separat gemessen (Alt-Binary gegen Neu-Binary bei Tiefe 1).
 *
 * Der Nutzlastpuffer waechst entsprechend auf max(groesste Groesse * Tiefe,
 * 4 MiB); bei den Default-Parametern bleibt es exakt bei den bisherigen 4 MiB.
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

#define MINBUF   (4u*1024u*1024u)   /* Untergrenze Nutzlastpuffer = bisheriges MAXSZ */
#define MAX_SIZES 32
#define MAX_DEPTH 32

/* Default-Leiter/Parameter: identisch zum uebernommenen Original. */
static size_t g_sizes[MAX_SIZES] = { 8, 4096, 65536, 1048576 };
static int    g_nsizes = 4;
static int    g_depth  = 1;
static int    g_iters  = 2000;
static int    g_warmup = 200;
static double g_secs   = 0.0;   /* >0: Zeitbudget je Groesse statt fester Iterationszahl */

/* ---- Mix-Szenario (--mix) ------------------------------------------------
 * Ein uniformer Sweep misst jede Groesse fuer sich und sieht deshalb NICHT,
 * was echter Inferencing-Verkehr macht: kleine Kollektive laufen in Kadenz,
 * und dazwischen schiebt sich ein grosser Prefill-Chunk auf DENSELBEN Pfad.
 * Genau dort entsteht Head-of-Line-Blocking -- die kleine Nachricht wartet,
 * bis der Brocken vor ihr durch ist. Sichtbar wird das nur am p99 der
 * KLEINEN Nachrichten, nicht am Median und nicht in einem Sweep.
 *
 * Fahrplan je Messrunde i (beide Seiten rechnen ihn identisch aus i aus --
 * er darf NICHT von der lokalen Uhr abhaengen, sonst laufen die Seiten
 * auseinander und das Ping-Pong haengt):
 *   i % burst_every == burst_every-1  -> gross   (Prefill-Chunk-Analog)
 *   i % g_mix_mid_every == ...-1      -> mittel  (Verify-Analog)
 *   sonst                             -> klein   (Decode-all_reduce-Analog)
 * burst_every wird vom Client nach dem Warmup aus der gemessenen Rundenzeit
 * bestimmt (Ziel: alle --mix-burst-ms Millisekunden ein Burst) und zusammen
 * mit der Rundenzahl an den Server geschickt. --mix-no-burst setzt es auf 0
 * und liefert damit den Vergleichsfall "gleiche Kadenz, keine Brocken".
 * Ausgewertet werden die drei Klassen GETRENNT, mit p50 und p99. */
static int    g_mix = 0;
static int    g_mix_mid_every = 10;
static double g_mix_burst_ms  = 100.0;
static int    g_mix_no_burst  = 0;
#define SECS_MIN_ITERS 50
#define SECS_MAX_ITERS 2000000

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

/* --sizes=A,B,C -> g_sizes/g_nsizes. Gibt 0 zurueck bei Erfolg. */
static int parse_sizes(const char *s)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        unsigned long long v = strtoull(p, &end, 0);
        if (end == p || v == 0 || v > (1ull << 30) || n >= MAX_SIZES) return -1;
        g_sizes[n++] = (size_t)v;
        p = end;
        if (*p == ',') p++;
        else if (*p) return -1;
    }
    if (n == 0) return -1;
    g_nsizes = n;
    return 0;
}

/* Filtert die Optionen aus argv heraus (beliebige Position), damit die sonst
 * unveraenderte Positions-Auswertung darunter exakt wie im Original laeuft.
 * Gibt das neue argc zurueck, oder -1 bei einer unbrauchbaren Option. */
static int strip_opts(int argc, char **argv)
{
    int out = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--ro") == 0) {
            g_ro_requested = 1;
            continue;
        }
        if (strncmp(a, "--sizes=", 8) == 0) {
            if (parse_sizes(a + 8)) {
                fprintf(stderr, "[FAIL] unbrauchbares --sizes: %s\n", a + 8);
                return -1;
            }
            continue;
        }
        if (strncmp(a, "--depth=", 8) == 0) {
            g_depth = atoi(a + 8);
            if (g_depth < 1 || g_depth > MAX_DEPTH) {
                fprintf(stderr, "[FAIL] --depth muss 1..%d sein\n", MAX_DEPTH);
                return -1;
            }
            continue;
        }
        if (strncmp(a, "--iters=", 8) == 0) {
            g_iters = atoi(a + 8);
            if (g_iters < 1) { fprintf(stderr, "[FAIL] --iters < 1\n"); return -1; }
            continue;
        }
        if (strcmp(a, "--mix") == 0)          { g_mix = 1; continue; }
        if (strcmp(a, "--mix-no-burst") == 0) { g_mix_no_burst = 1; continue; }
        if (strncmp(a, "--mix-mid-every=", 16) == 0) {
            g_mix_mid_every = atoi(a + 16);
            if (g_mix_mid_every < 2) { fprintf(stderr, "[FAIL] --mix-mid-every < 2\n"); return -1; }
            continue;
        }
        if (strncmp(a, "--mix-burst-ms=", 15) == 0) {
            g_mix_burst_ms = atof(a + 15);
            if (g_mix_burst_ms <= 0) { fprintf(stderr, "[FAIL] --mix-burst-ms <= 0\n"); return -1; }
            continue;
        }
        if (strncmp(a, "--secs=", 7) == 0) {
            g_secs = atof(a + 7);
            if (g_secs < 0) { fprintf(stderr, "[FAIL] --secs < 0\n"); return -1; }
            continue;
        }
        if (strncmp(a, "--warmup=", 9) == 0) {
            g_warmup = atoi(a + 9);
            if (g_warmup < 0) { fprintf(stderr, "[FAIL] --warmup < 0\n"); return -1; }
            continue;
        }
        if (strncmp(a, "--", 2) == 0) {
            fprintf(stderr, "[FAIL] unbekannte Option: %s\n", a);
            return -1;
        }
        argv[out++] = argv[i];
    }
    return out;
}

/* Beim Verbindungsaufbau ausgetauscht. magic/version fangen den Fall ab, dass
 * die beiden Seiten mit unterschiedlichen Staenden gebaut wurden; die
 * Parameterfelder fangen ab, dass sie mit unterschiedlichen Leitern/Tiefen
 * gestartet wurden. Beides waere sonst ein stiller Messfehler. */
#define CONN_MAGIC   0x47445231u   /* "GDR1" */
#define CONN_VERSION 2u

struct conn_info {
    uint32_t magic, version;
    uint32_t qpn, psn, rkey_pay, rkey_flag;
    uint64_t addr_pay, addr_flag;
    uint64_t bufsz;
    uint32_t depth, nsizes, iters, warmup, secs_ms;
    uint32_t mix, mix_mid_every, mix_no_burst;
    uint64_t sizes_hash;
    uint8_t  gid[16];
};

static uint64_t sizes_hash(void)
{
    uint64_t h = 1469598103934665603ull;   /* FNV-1a */
    for (int i = 0; i < g_nsizes; i++) {
        uint64_t v = (uint64_t)g_sizes[i];
        for (int b = 0; b < 8; b++) {
            h ^= (v >> (b * 8)) & 0xff;
            h *= 1099511628211ull;
        }
    }
    return h;
}

/* Gibt 0 zurueck, wenn die Gegenseite zu uns passt. */
static int check_peer(const struct conn_info *p)
{
    if (p->magic != CONN_MAGIC || p->version != CONN_VERSION) {
        fprintf(stderr, "[FAIL] Gegenseite spricht ein anderes Protokoll "
                "(magic=0x%x version=%u, erwartet 0x%x/%u) -- vermutlich ein "
                "alter Binary-Stand auf einer Seite. Beide Seiten neu bauen.\n",
                p->magic, p->version, CONN_MAGIC, CONN_VERSION);
        return -1;
    }
    if (p->depth != (uint32_t)g_depth || p->nsizes != (uint32_t)g_nsizes ||
        p->iters != (uint32_t)g_iters || p->warmup != (uint32_t)g_warmup ||
        p->secs_ms != (uint32_t)(g_secs * 1000.0 + 0.5) ||
        p->mix != (uint32_t)g_mix ||
        p->mix_mid_every != (uint32_t)g_mix_mid_every ||
        p->mix_no_burst != (uint32_t)g_mix_no_burst ||
        p->sizes_hash != sizes_hash()) {
        fprintf(stderr, "[FAIL] Gegenseite laeuft mit anderen Messparametern "
                "(peer depth=%u nsizes=%u iters=%u warmup=%u, lokal depth=%d "
                "nsizes=%d iters=%d warmup=%d, Groessenliste %s) -- beide "
                "Seiten brauchen identische --sizes/--depth/--iters/--warmup.\n",
                p->depth, p->nsizes, p->iters, p->warmup,
                g_depth, g_nsizes, g_iters, g_warmup,
                p->sizes_hash == sizes_hash() ? "gleich" : "verschieden");
        return -1;
    }
    return 0;
}

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
    /* Empfangs-Timeout: sonst haengt ein Stand-Missverhaeltnis zwischen den
     * beiden Seiten (unterschiedlich grosse conn_info) hier ewig im read()
     * statt sauber zu melden. */
    { struct timeval tv = { .tv_sec = 15, .tv_usec = 0 };
      setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)); }
    if (write(sock, mine, sizeof(*mine)) != (ssize_t)sizeof(*mine)) return -1;
    { char *dst = (char *)peer; size_t got = 0;
      while (got < sizeof(*peer)) {
          ssize_t n = read(sock, dst + got, sizeof(*peer) - got);
          if (n > 0) { got += (size_t)n; continue; }
          fprintf(stderr, "[FAIL] Parameter-Austausch mit der Gegenseite "
                  "unvollstaendig (%zu von %zu Bytes) -- unterschiedliche "
                  "Binary-Staende oder Gegenseite abgestuerzt.\n",
                  got, sizeof(*peer));
          return -1;
      } }
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
    argc = strip_opts(argc, argv);   /* Optionen raus, Rest unveraendert positional */
    if (argc < 0) return 1;

    if (argc < 6) {
        fprintf(stderr,
          "Server: %s server <cuda-ord|-1> <nic> <port> <gdr|stage> [opts]\n"
          "Client: %s client <server-ip> <nic> <port> <gdr|stage> <cuda-ord|-1> [opts]\n"
          "opts: --ro --sizes=A,B,C --depth=N --iters=N --warmup=N\n"
          "      (beide Seiten muessen dieselben opts bekommen)\n",
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

#ifndef WITH_CUDA
    if (ord >= 0) {
        fprintf(stderr, "[FAIL] cuda-ord=%d verlangt eine GPU, dieser Build ist "
                "aber ohne -DWITH_CUDA gebaut (Host-Zweig). Nur -1 moeglich.\n", ord);
        return 1;
    }
#endif

    /* Nutzlastpuffer: groesste Leitergroesse mal Pipelining-Tiefe, mindestens
     * die bisherigen 4 MiB. Bei Default-Parametern (1 MiB, Tiefe 1) bleibt es
     * exakt bei 4 MiB wie bisher. */
    size_t maxsz = 0;
    for (int i = 0; i < g_nsizes; i++) if (g_sizes[i] > maxsz) maxsz = g_sizes[i];
    size_t bufsz = (size_t)MINBUF;
    { size_t need = maxsz * (size_t)g_depth;
      if (need > bufsz) bufsz = (need + 4095u) & ~(size_t)4095u; }

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
        int dfd = gpu_dmabuf(ord, bufsz, &dptr, &mh, gpuname, sizeof(gpuname));
        if (dfd < 0) { fprintf(stderr, "GPU-dmabuf fehlgeschlagen\n"); return 1; }
        mr_pay = ibv_reg_dmabuf_mr(pd, 0, bufsz, 0, dfd,
                    IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ
                    |ro_payload_bit());
        if (!mr_pay) { fprintf(stderr, "reg_dmabuf errno=%d\n", errno); return 1; }
        pay_addr = 0;   /* iova 0 */
#else
        fprintf(stderr, "gdr braucht -DWITH_CUDA\n"); return 1;
#endif
    } else {
        if (posix_memalign(&host_pay, 4096, bufsz)) { perror("memalign"); return 1; }
        memset(host_pay, 0, bufsz);
        mr_pay = ibv_reg_mr(pd, host_pay, bufsz,
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
            CHK(cuMemAlloc(&dptr, bufsz));
            CHK(cuMemsetD8(dptr, 0, bufsz)); CHK(cuCtxSynchronize());
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
    /* 64 wie bisher, sofern die Tiefe (+1 fuer den Flag-WR) hineinpasst. */
    ia.cap.max_send_wr = (g_depth + 1 > 64) ? (uint32_t)(g_depth + 1) : 64;
    ia.cap.max_recv_wr = 64;
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
    mine.magic = CONN_MAGIC; mine.version = CONN_VERSION;
    mine.qpn = qp->qp_num; mine.psn = 0x4321;
    mine.rkey_pay = mr_pay->rkey;   mine.addr_pay = pay_addr;
    mine.rkey_flag = mr_flag->rkey; mine.addr_flag = (uint64_t)(uintptr_t)flagbuf;
    mine.bufsz = (uint64_t)bufsz;
    mine.depth = (uint32_t)g_depth; mine.nsizes = (uint32_t)g_nsizes;
    mine.iters = (uint32_t)g_iters; mine.warmup = (uint32_t)g_warmup;
    mine.secs_ms = (uint32_t)(g_secs * 1000.0 + 0.5);
    mine.mix = (uint32_t)g_mix;
    mine.mix_mid_every = (uint32_t)g_mix_mid_every;
    mine.mix_no_burst = (uint32_t)g_mix_no_burst;
    mine.sizes_hash = sizes_hash();
    memcpy(mine.gid, mygid, 16);

    int sock = tcp_exchange(is_server, is_server ? NULL : argv[2], port, &mine, &peer);
    if (sock < 0) return 1;
    if (check_peer(&peer)) return 1;
    if (peer.bufsz < (uint64_t)bufsz) {
        fprintf(stderr, "[FAIL] Zielpuffer der Gegenseite zu klein (%llu < %zu)\n",
                (unsigned long long)peer.bufsz, bufsz);
        return 1;
    }
    if (qp_rtr_rts(qp, &peer, gidx, mine.psn)) return 1;

    printf("[%s] Modus=%s  GPU=%s  sgid=%d  RO=%s  depth=%d  iters=%d  warmup=%d  buf=%zu\n",
           is_server ? "srv" : "cli",
           gdr ? "GDR" : "STAGE", gpuname, gidx,
           !g_ro_requested ? "off" : (HAVE_RO_SYMBOL ? "on" : "unsupported"),
           g_depth, g_iters, g_warmup, bufsz);
    fflush(stdout);

    /* Bei Zeitbudget (--secs) steht die Iterationszahl erst nach dem Warmup
     * fest, deshalb wird der Puffer auf die Obergrenze ausgelegt. */
    int lat_cap = (g_secs > 0.0) ? SECS_MAX_ITERS : g_iters;
    double *lat = malloc(sizeof(double) * (size_t)lat_cap);
    /* +1 fuer den Flag-WR am Ende der Kette */
    struct ibv_sge     sges[MAX_DEPTH + 1];
    struct ibv_send_wr wrs[MAX_DEPTH + 1];
    if (!lat) { perror("malloc lat"); return 1; }

    /* =====================================================================
     * MIX-Szenario: EINE Messreihe, in der sich die Groessen nach dem oben
     * beschriebenen Fahrplan abwechseln. Bewusst keine Groessenschleife --
     * der ganze Sinn ist, dass alles auf demselben Pfad durcheinanderlaeuft.
     * ===================================================================== */
    if (g_mix) {
        if (g_nsizes != 3) {
            fprintf(stderr, "[FAIL] --mix braucht genau drei Groessen "
                    "(klein,mittel,gross) in --sizes\n");
            return 1;
        }
        size_t sz_small = g_sizes[0], sz_mid = g_sizes[1], sz_large = g_sizes[2];
        uint64_t seq = 0;
        int burst_every = 0;
        int n_meas = g_iters;
        int total = g_warmup + ((g_secs > 0.0) ? 0 : g_iters);
        double t_warm0 = 0.0;
        /* Getrennte Sammler je Klasse -- der Sinn der Uebung ist ja gerade,
         * die kleinen Nachrichten NICHT mit den grossen zu vermischen. */
        double *l_small = malloc(sizeof(double) * (size_t)lat_cap);
        double *l_mid   = malloc(sizeof(double) * (size_t)lat_cap);
        double *l_large = malloc(sizeof(double) * (size_t)lat_cap);
        int n_small = 0, n_mid = 0, n_large = 0;
        if (!l_small || !l_mid || !l_large) { perror("malloc mix"); return 1; }

        for (int it = 0; it < total; it++) {
            if (it == 0) t_warm0 = now_us();
            int i = it - g_warmup;           /* <0 waehrend des Warmups */
            size_t sz = sz_small; int cls = 0;
            if (i >= 0) {
                if (burst_every > 0 && (i % burst_every) == burst_every - 1) {
                    sz = sz_large; cls = 2;
                } else if ((i % g_mix_mid_every) == g_mix_mid_every - 1) {
                    sz = sz_mid; cls = 1;
                }
            }

            if (!is_server) {
                double t0 = now_us();
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) cuMemcpyDtoH(host_pay, dptr, sz * (size_t)g_depth);
#endif
                struct ibv_send_wr *bad;
                int nwr = 0;
                for (int d = 0; d < g_depth; d++) {
                    memset(&sges[nwr], 0, sizeof(sges[0]));
                    sges[nwr].addr = pay_addr + (uint64_t)d * sz;
                    sges[nwr].length = sz; sges[nwr].lkey = mr_pay->lkey;
                    memset(&wrs[nwr], 0, sizeof(wrs[0]));
                    wrs[nwr].wr_id = 1;
                    wrs[nwr].sg_list = &sges[nwr]; wrs[nwr].num_sge = 1;
                    wrs[nwr].opcode = IBV_WR_RDMA_WRITE; wrs[nwr].send_flags = 0;
                    wrs[nwr].wr.rdma.remote_addr = peer.addr_pay + (uint64_t)d * sz;
                    wrs[nwr].wr.rdma.rkey = peer.rkey_pay;
                    nwr++;
                }
                seq++;
                *(uint64_t *)((char *)flagbuf + 64) = seq;
                memset(&sges[nwr], 0, sizeof(sges[0]));
                sges[nwr].addr = (uint64_t)(uintptr_t)flagbuf + 64;
                sges[nwr].length = 8; sges[nwr].lkey = mr_flag->lkey;
                memset(&wrs[nwr], 0, sizeof(wrs[0]));
                wrs[nwr].wr_id = 2;
                wrs[nwr].sg_list = &sges[nwr]; wrs[nwr].num_sge = 1;
                wrs[nwr].opcode = IBV_WR_RDMA_WRITE;
                wrs[nwr].send_flags = IBV_SEND_SIGNALED;
                wrs[nwr].wr.rdma.remote_addr = peer.addr_flag + 64;
                wrs[nwr].wr.rdma.rkey = peer.rkey_flag;
                nwr++;
                for (int k = 0; k < nwr - 1; k++) wrs[k].next = &wrs[k + 1];
                wrs[nwr - 1].next = NULL;
                if (ibv_post_send(qp, &wrs[0], &bad)) { perror("mix post"); return 1; }
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) == 0) { }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "mix WC %s\n", ibv_wc_status_str(wc.status)); return 1; }
                while (flag[0] != seq) { __builtin_ia32_pause(); }
                double t1 = now_us();
                if (i >= 0) {
                    double v = (t1 - t0) / 2.0;
                    if (cls == 0 && n_small < lat_cap) l_small[n_small++] = v;
                    else if (cls == 1 && n_mid < lat_cap) l_mid[n_mid++] = v;
                    else if (cls == 2 && n_large < lat_cap) l_large[n_large++] = v;
                }
            } else {
                seq++;
                while (flag[8] != seq) { __builtin_ia32_pause(); }
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) cuMemcpyHtoD(dptr, host_pay, sz * (size_t)g_depth);
#endif
                *(uint64_t *)flagbuf = seq;
                struct ibv_sge fs; struct ibv_send_wr fw, *bad;
                memset(&fs, 0, sizeof(fs));
                fs.addr = (uint64_t)(uintptr_t)flagbuf; fs.length = 8; fs.lkey = mr_flag->lkey;
                memset(&fw, 0, sizeof(fw));
                fw.wr_id = 3; fw.sg_list = &fs; fw.num_sge = 1;
                fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = IBV_SEND_SIGNALED;
                fw.wr.rdma.remote_addr = peer.addr_flag; fw.wr.rdma.rkey = peer.rkey_flag;
                if (ibv_post_send(qp, &fw, &bad)) { perror("mix srv post"); return 1; }
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) == 0) { }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "mix srv WC %s\n", ibv_wc_status_str(wc.status)); return 1; }
            }

            /* Nach dem Warmup: Rundenzahl UND Burst-Periode aushandeln. Die
             * Periode kommt aus der gemessenen Rundenzeit der kleinen
             * Nachricht (Warmup faehrt ausschliesslich klein), damit der
             * Burst-Abstand in MILLISEKUNDEN stimmt -- der Fahrplan selbst
             * bleibt aber reine Rundenarithmetik und damit auf beiden Seiten
             * identisch, ohne jede Uhrabhaengigkeit. */
            if (it == g_warmup - 1) {
                uint32_t msg[2] = { 0, 0 };
                if (!is_server) {
                    double per_iter = (now_us() - t_warm0) / (double)g_warmup;
                    double want = (g_secs > 0.0)
                        ? ((per_iter > 0.0) ? (g_secs * 1e6 / per_iter) : SECS_MIN_ITERS)
                        : (double)g_iters;
                    if (want < SECS_MIN_ITERS) want = SECS_MIN_ITERS;
                    if (want > SECS_MAX_ITERS) want = SECS_MAX_ITERS;
                    msg[0] = (uint32_t)want;
                    if (!g_mix_no_burst && per_iter > 0.0) {
                        double be = g_mix_burst_ms * 1000.0 / per_iter;
                        if (be < 2) be = 2;
                        if (be > 1e9) be = 1e9;
                        msg[1] = (uint32_t)be;
                    }
                    if (write(sock, msg, sizeof(msg)) != (ssize_t)sizeof(msg)) {
                        fprintf(stderr, "[FAIL] Mix-Fahrplan senden\n"); return 1; }
                } else {
                    char *d = (char *)msg; size_t got = 0;
                    while (got < sizeof(msg)) {
                        ssize_t r = read(sock, d + got, sizeof(msg) - got);
                        if (r <= 0) { fprintf(stderr, "[FAIL] Mix-Fahrplan empfangen\n"); return 1; }
                        got += (size_t)r;
                    }
                }
                n_meas = (int)msg[0];
                burst_every = (int)msg[1];
                total = g_warmup + n_meas;
            }
        }

        if (!is_server) {
            struct { const char *name; double *a; int n; size_t sz; } cl[3] = {
                { "klein",  l_small, n_small, sz_small },
                { "mittel", l_mid,   n_mid,   sz_mid   },
                { "gross",  l_large, n_large, sz_large },
            };
            printf("MIX-Fahrplan: klein=%zu B, mittel=%zu B alle %d Runden, "
                   "gross=%zu B alle %d Runden (%s), Runden=%d\n",
                   sz_small, sz_mid, g_mix_mid_every, sz_large, burst_every,
                   burst_every ? "mit Bursts" : "OHNE Bursts (Vergleichsfall)",
                   n_meas);
            for (int c = 0; c < 3; c++) {
                if (cl[c].n < 2) continue;
                qsort(cl[c].a, cl[c].n, sizeof(double), cmp_d);
                double p50 = cl[c].a[cl[c].n / 2];
                double p99 = cl[c].a[(int)((double)cl[c].n * 0.99)];
                printf("MIXDATA\t%s\t%s\t%zu\t%d\t%d\t%s\t%.3f\t%.3f\t%.3f\n",
                       gdr ? "gdr" : "stage", cl[c].name, cl[c].sz, cl[c].n,
                       burst_every, burst_every ? "mit" : "ohne",
                       cl[c].a[cl[c].n / 10], p50, p99);
            }
            fflush(stdout);
        }
        free(l_small); free(l_mid); free(l_large);
        close(sock);
        return 0;
    }

    for (int si = 0; si < g_nsizes; si++) {
        size_t sz = g_sizes[si];
        uint64_t seq = 0;
        /* n_meas steht bei fester Iterationszahl sofort fest; mit --secs erst
         * nach dem Warmup (siehe unten), bis dahin ist total nur das Warmup. */
        int n_meas = g_iters;
        int total  = g_warmup + ((g_secs > 0.0) ? 0 : g_iters);
        double t_warm0 = 0.0;

        for (int it = 0; it < total; it++) {
            if (it == 0) t_warm0 = now_us();
            if (!is_server) {
                double t0 = now_us();
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) {           /* heutiger Weg: DtoH */
                    /* Die Staging-Kopie muss genauso viele Bytes bewegen wie
                     * die Runde ueber den Draht schickt, sonst waere der
                     * Vergleich bei Tiefe > 1 zugunsten von STAGE verzerrt. */
                    cuMemcpyDtoH(host_pay, dptr, sz * (size_t)g_depth);
                }
#endif
                struct ibv_send_wr *bad;
                int nwr = 0;
                if (sz > 0) {
                    for (int d = 0; d < g_depth; d++) {
                        memset(&sges[nwr], 0, sizeof(sges[0]));
                        /* LOKALE Adresse der Nutzlast-MR. Nicht "gdr ? 0 : host":
                         * der Modus gdr sagt nur, dass die GEGENSEITE eine GPU-MR
                         * haelt. Liegt die eigene Nutzlast im Host (Arm
                         * gpu_is_server: Client laeuft mit cuda-ord = -1, also
                         * ibv_reg_mr auf host_pay), waere addr=0 zusammen mit dem
                         * Host-lkey eine ungueltige lokale Adresse -> "local
                         * protection error". pay_addr ist genau die richtige
                         * Groesse: 0 (iova) fuer die dmabuf-MR, host_pay sonst.
                         * Der Versatz d*sz macht die Tiefen-WRs zu unabhaengigen
                         * Transfers statt zu ueberlappenden Schreibern auf
                         * dieselbe Stelle. */
                        sges[nwr].addr = pay_addr + (uint64_t)d * sz;
                        sges[nwr].length = sz; sges[nwr].lkey = mr_pay->lkey;
                        memset(&wrs[nwr], 0, sizeof(wrs[0]));
                        wrs[nwr].wr_id = 1;
                        wrs[nwr].sg_list = &sges[nwr]; wrs[nwr].num_sge = 1;
                        wrs[nwr].opcode = IBV_WR_RDMA_WRITE; wrs[nwr].send_flags = 0;
                        wrs[nwr].wr.rdma.remote_addr = peer.addr_pay + (uint64_t)d * sz;
                        wrs[nwr].wr.rdma.rkey = peer.rkey_pay;
                        nwr++;
                    }
                }
                seq++;
                *(uint64_t *)((char *)flagbuf + 64) = seq;
                memset(&sges[nwr], 0, sizeof(sges[0]));
                sges[nwr].addr = (uint64_t)(uintptr_t)flagbuf + 64;
                sges[nwr].length = 8; sges[nwr].lkey = mr_flag->lkey;
                memset(&wrs[nwr], 0, sizeof(wrs[0]));
                wrs[nwr].wr_id = 2;
                wrs[nwr].sg_list = &sges[nwr]; wrs[nwr].num_sge = 1;
                wrs[nwr].opcode = IBV_WR_RDMA_WRITE;
                wrs[nwr].send_flags = IBV_SEND_SIGNALED;
                /* Client -> Server: Offset 64, dort pollt der Server (flag[8]) */
                wrs[nwr].wr.rdma.remote_addr = peer.addr_flag + 64;
                wrs[nwr].wr.rdma.rkey = peer.rkey_flag;
                nwr++;
                /* EINE verkettete Postung: alle Nutzlast-WRs plus das Flag,
                 * danach erst pollen. Nur das Flag ist signalisiert -- auf
                 * einem RC-QP kommen die Completions in Reihenfolge, sein
                 * Completion impliziert also die Nutzlast-WRs davor. */
                for (int i = 0; i < nwr - 1; i++) wrs[i].next = &wrs[i + 1];
                wrs[nwr - 1].next = NULL;
                if (ibv_post_send(qp, &wrs[0], &bad)) { perror("post chain"); return 1; }
                struct ibv_wc wc;
                while (ibv_poll_cq(cq, 1, &wc) == 0) { }
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "WC %s\n", ibv_wc_status_str(wc.status)); return 1; }
                /* auf Antwort warten */
                while (flag[0] != seq) { __builtin_ia32_pause(); }
                double t1 = now_us();
                if (it >= g_warmup && (it - g_warmup) < lat_cap)
                    lat[it - g_warmup] = (t1 - t0) / 2.0;
            } else {
                seq++;
                while (flag[8] != seq) { __builtin_ia32_pause(); }   /* Offset 64B = idx 8 */
#ifdef WITH_CUDA
                if (!gdr && ord >= 0) {           /* heutiger Weg: HtoD */
                    cuMemcpyHtoD(dptr, host_pay, sz * (size_t)g_depth);
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

            /* ---- Zeitbudget: Iterationszahl nach dem Warmup aushandeln ----
             * Das Warmup laeuft mit fester, BEIDEN Seiten bekannter Laenge --
             * es ist damit gleichzeitig die Kalibrierung. Der Client rechnet
             * daraus, wie viele Messrunden in das Budget passen, und schickt
             * die Zahl ueber den bereits offenen TCP-Socket. Beide Seiten
             * fahren danach exakt gleich viele Runden, der strikte Gleichtakt
             * des Ping-Pongs bleibt also erhalten -- eine Seite, die selbst
             * "nach Uhr" abbricht, wuerde die andere im Poll haengen lassen.
             * Der Socket ist hier garantiert ruhig (die Runde ist fertig). */
            if (g_secs > 0.0 && it == g_warmup - 1) {
                uint32_t n = 0;
                if (!is_server) {
                    double per_iter = (now_us() - t_warm0) / (double)g_warmup;
                    double want = (per_iter > 0.0)
                                ? (g_secs * 1e6 / per_iter) : (double)SECS_MIN_ITERS;
                    if (want < SECS_MIN_ITERS) want = SECS_MIN_ITERS;
                    if (want > SECS_MAX_ITERS) want = SECS_MAX_ITERS;
                    n = (uint32_t)want;
                    if (write(sock, &n, sizeof(n)) != (ssize_t)sizeof(n)) {
                        fprintf(stderr, "[FAIL] Iterationszahl senden\n"); return 1; }
                } else {
                    char *d = (char *)&n; size_t got = 0;
                    while (got < sizeof(n)) {
                        ssize_t r = read(sock, d + got, sizeof(n) - got);
                        if (r <= 0) { fprintf(stderr,
                            "[FAIL] Iterationszahl empfangen\n"); return 1; }
                        got += (size_t)r;
                    }
                }
                n_meas = (int)n;
                total  = g_warmup + n_meas;
            }
        }

        if (!is_server) {
            qsort(lat, n_meas, sizeof(double), cmp_d);
            /* Menschliche Zeile: Format unveraendert, damit der bestehende
             * Parser in gdr_window_run.sh weiter greift. */
            printf("%-9s %8zu B   p10 %8.2f us   MEDIAN %8.2f us   p90 %8.2f us\n",
                   gdr ? "GDR" : "STAGE", sz,
                   lat[n_meas/10], lat[n_meas/2], lat[(n_meas*9)/10]);
            /* Maschinenzeile fuer die Cross-Rig-Leiter: alle Achsen explizit.
             * Die us-Werte sind wie bisher der halbe Round-trip EINER RUNDE;
             * eine Runde traegt depth Nachrichten, die Zeit je Nachricht ist
             * also median/depth. Bewusst nicht vorgeteilt, damit die Rohgroesse
             * dieselbe Bedeutung wie in allen frueheren Tabellen behaelt. */
            /* Spalte 11 ist p99: fuer Schwanz-Fragen (Nebenlast, Blocking)
             * ist p90 zu grob. Angehaengt, nicht eingeschoben -- bestehende
             * Parser mit festen Feldnummern bleiben gueltig. */
            printf("DATA\t%s\t%zu\t%d\t%s\t%d\t%.3f\t%.3f\t%.3f\t%zu\t%.3f\t%.3f\n",
                   gdr ? "gdr" : "stage", sz, g_depth,
                   !g_ro_requested ? "off" : (HAVE_RO_SYMBOL ? "on" : "unsupported"),
                   n_meas,
                   lat[n_meas/10], lat[n_meas/2], lat[(n_meas*9)/10],
                   sz * (size_t)g_depth,
                   lat[(int)((double)n_meas * 0.99)],
                   lat[n_meas - 1]);
            fflush(stdout);
        }
    }

    close(sock);
    return 0;
}

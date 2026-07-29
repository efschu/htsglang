// SPDX-License-Identifier: MIT
//
// ramrelay -- wie schnell ist der RAM-Pfad zwischen zwei Karten WIRKLICH?
//
// ===========================================================================
// DIE FRAGE
// ===========================================================================
// Die Messkampagne vom 2026-07-28 zeigt: NIC-Relay (GPU -> ConnectX-4 -> GPU,
// beidseitig dmabuf) schlaegt den heutigen Pfad (NCCL ueber System-RAM) bei
// 20 KiB um Faktor 5,08 (7,37 gegen 37,41 us halber Round-trip). Das ist
// erklaerungsbeduerftig: die NIC haengt an Gen3 x4, die Karten an x16 bzw. x4.
// Eine Zwischenstation kann den direkten Weg nicht schlagen, wenn beide
// dieselbe Software-Qualitaet haben.
//
// Aufloesung der Vermutung: der Vergleich misst NICHT "NIC-Pfad gegen
// RAM-Pfad", sondern "DMA-Engine mit fertigem Deskriptor gegen ein
// Software-Kollektiv". 20 KiB sind auf Gen3 x4 rund 5,9 us reine Drahtzeit --
// die 7,37 us des NIC-Arms sind also fast vollstaendig Draht. Durch den
// System-RAM sind dieselben 20 KiB in unter 2 us kopiert. Die 37 us koennen
// damit gar keine Transportkosten sein; sie muessen fast vollstaendig
// Protokoll-, Start- und Synchronisationskosten sein.
//
// Diese Sonde beziffert das, statt es zu behaupten: sie misst den RAM-Pfad
// OHNE NCCL und ohne jedes Kollektiv-Protokoll, auf zwei Arten:
//
//   ce   Der naive Weg, wie ihn ein Staging-Transport heute geht:
//        cudaMemcpyAsync DtoH auf der Sendekarte, Synchronisation,
//        cudaMemcpyAsync HtoD auf der Empfangskarte. Vier Sync-Punkte je
//        Runde. Beziffert, was Copy-Engine-Start und Stream-Synchronisation
//        allein kosten.
//   zc   Der schlanke Weg: je Karte EIN dauerhaft laufender Kernel, der
//        gemappten Host-Speicher direkt beschreibt und liest und auf ein
//        Flag im Host-RAM wartet. Kein Kernel-Start je Nachricht, keine
//        Stream-Synchronisation, keine Copy-Engine. Das ist der Boden, den
//        der RAM-Pfad auf dieser Hardware hergibt.
//
// Ist zc deutlich schneller als die 37 us von NCCL, dann ist der Vorsprung
// des NIC-Arms ueberwiegend ein Software-Vorsprung und mit einem besseren
// eigenen Transport (HTCCL) einzuholen -- ohne NIC, ohne Treiber-Patch.
// Bleibt zc bei ~37 us, ist der RAM-Pfad tatsaechlich strukturell teuer und
// der NIC-Umweg verdient seinen Namen nicht.
//
// ===========================================================================
// AUFBAU
// ===========================================================================
// Striktes Ping-Pong zwischen zwei Karten, wie im RDMA-Bench: A schickt die
// Nutzlast, B schickt sie zurueck; berichtet wird der HALBE Round-trip, damit
// die Zahlen direkt neben gpurdma_04_bench und nccl_reference.py stehen.
// Zeit wird auf Karte A mit clock64() genommen und ueber die gemeldete
// Taktrate in Mikrosekunden umgerechnet.
//
// P2P ist auf diesem Rig treiberseitig gesperrt -- gemappter Host-Speicher
// (cudaHostAllocPortable|Mapped) ist deshalb das einzige gemeinsame Medium.
// Genau das ist der Punkt: es ist derselbe System-RAM, den NCCL benutzt.
//
// SICHERHEIT: der zc-Kernel wartet NIE unbegrenzt. Jede Warteschleife hat
// einen clock64()-Deckel; laeuft er ab, bricht der Kernel ab und meldet
// timeout. Ein haengender Dauerkernel wuerde die Karte blockieren, und die
// Karten sind geteilt.
//
// Bauen:  nvcc -O3 -arch=sm_75 ... siehe build_ramrelay.sh
// Aufruf: ramrelay <ord_a> <ord_b> <ce|zc> [--sizes=..] [--secs=..]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[FAIL] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); return 1; } } while (0)

// Warte-Deckel je Runde. Grosszuegig (Sekundenbereich bei ~1 GHz Shader-Takt),
// aber endlich -- ein Dauerkernel ohne Deckel wuerde die geteilte Karte
// blockieren, wenn die Gegenseite stirbt.
#define SPIN_LIMIT_CYCLES 4000000000ULL

// ---------------------------------------------------------------------------
// zc: ein Dauerkernel je Karte. Rolle 0 beginnt die Runde (sendet zuerst),
// Rolle 1 antwortet. Beide arbeiten auf denselben Host-Puffern; die Richtung
// unterscheidet nur, welcher Puffer aus- und welcher eingelesen wird.
//
// Ein Block, damit __syncthreads() als Ordnungsmittel reicht. Die Nutzlast
// wandert in uint4-Schritten (16 B), das ist die breiteste Store-Einheit, die
// ueber PCIe sauber koaleszent bleibt.
// ---------------------------------------------------------------------------
__global__ void zc_agent(uint4 *__restrict__ vram,
                         uint4 *__restrict__ h_out, uint4 *__restrict__ h_in,
                         volatile unsigned int *flag_out,
                         volatile unsigned int *flag_in,
                         int n16, int rounds, int role,
                         long long *cyc_out, int *status)
{
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int r = 1; r <= rounds; ++r) {
        long long t0 = 0;
        if (role == 0 && tid == 0) t0 = clock64();

        // --- senden: VRAM -> gemappter Host-Puffer, dann Flag setzen -------
        if (role == 1) {
            // Rolle 1 antwortet erst, wenn sie empfangen hat: Empfang zuerst.
            if (tid == 0) {
                long long s = clock64();
                while (*flag_in != (unsigned int)r) {
                    if ((unsigned long long)(clock64() - s) > SPIN_LIMIT_CYCLES) {
                        *status = 1; return;
                    }
                }
                // Leseseitige Barriere: ohne sie darf die Karte Nutzlast-Loads
                // vor die Flag-Beobachtung ziehen und alte Daten liefern.
                __threadfence_system();
            }
            __syncthreads();
            for (int i = tid; i < n16; i += nth) vram[i] = h_in[i];
            __syncthreads();
        }

        for (int i = tid; i < n16; i += nth) h_out[i] = vram[i];
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            *flag_out = (unsigned int)r;
            __threadfence_system();
        }

        if (role == 0) {
            // --- empfangen: auf Antwort warten, dann in das eigene VRAM ----
            if (tid == 0) {
                long long s = clock64();
                while (*flag_in != (unsigned int)r) {
                    if ((unsigned long long)(clock64() - s) > SPIN_LIMIT_CYCLES) {
                        *status = 1; return;
                    }
                }
                // Leseseitige Barriere: ohne sie darf die Karte Nutzlast-Loads
                // vor die Flag-Beobachtung ziehen und alte Daten liefern.
                __threadfence_system();
            }
            __syncthreads();
            for (int i = tid; i < n16; i += nth) vram[i] = h_in[i];
            __syncthreads();
            if (tid == 0) cyc_out[r - 1] = clock64() - t0;
        }
    }
    if (tid == 0) *status = 0;
}

struct Args {
    int ord_a = 0, ord_b = 1;
    std::string mode = "zc";
    std::vector<size_t> sizes{20480, 81920, 1048576};
    double secs = 5.0;
    int threads = 256;
};

// Inhaltspruefung. Im Ping-Pong echot B die Nutzlast, am Ende muss also B's
// VRAM byte-gleich zu A's Ausgangsmuster sein. Projektregel: eine
// Transportzahl ohne Byte-Beleg ist keine Transportzahl.
static int fill_and_check(void *va, void *vb, size_t bytes, int ord_a, int ord_b,
                          bool fill, size_t *bad_out)
{
    std::vector<unsigned char> pat(bytes), got(bytes);
    for (size_t i = 0; i < bytes; ++i) pat[i] = (unsigned char)(i * 31 + 7);
    if (fill) {
        CK(cudaSetDevice(ord_a));
        CK(cudaMemcpy(va, pat.data(), bytes, cudaMemcpyHostToDevice));
        CK(cudaSetDevice(ord_b));
        CK(cudaMemset(vb, 0, bytes));
        return 0;
    }
    CK(cudaSetDevice(ord_b));
    CK(cudaMemcpy(got.data(), vb, bytes, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < bytes; ++i) if (got[i] != pat[i]) ++bad;
    *bad_out = bad;
    return 0;
}

static void parse_sizes(const char *s, std::vector<size_t> &out)
{
    out.clear();
    const char *p = s;
    while (*p) {
        out.push_back(strtoull(p, nullptr, 10));
        const char *c = strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
}

// ---------------------------------------------------------------------------
// ce: der naive Staging-Weg. Bewusst genau so gebaut, wie ein einfacher
// Transport es taete -- Copy-Engine plus Stream-Synchronisation je Schritt.
// Die Zahl ist keine Schwaeche der Hardware, sondern der Preis dieses Musters.
// ---------------------------------------------------------------------------
static int run_ce(const Args &a, size_t bytes, std::vector<double> &lat_us,
                  size_t &bad)
{
    void *va = nullptr, *vb = nullptr, *hab = nullptr, *hba = nullptr;
    cudaStream_t sa, sb;

    CK(cudaSetDevice(a.ord_a));
    CK(cudaMalloc(&va, bytes));
    CK(cudaStreamCreate(&sa));
    CK(cudaHostAlloc(&hab, bytes, cudaHostAllocPortable));
    CK(cudaHostAlloc(&hba, bytes, cudaHostAllocPortable));
    CK(cudaSetDevice(a.ord_b));
    CK(cudaMalloc(&vb, bytes));
    CK(cudaStreamCreate(&sb));

    if (fill_and_check(va, vb, bytes, a.ord_a, a.ord_b, true, nullptr)) return 1;

    const int warmup = 50;
    for (int i = 0; i < warmup; ++i) {
        CK(cudaSetDevice(a.ord_a));
        CK(cudaMemcpyAsync(hab, va, bytes, cudaMemcpyDeviceToHost, sa));
        CK(cudaStreamSynchronize(sa));
        CK(cudaSetDevice(a.ord_b));
        CK(cudaMemcpyAsync(vb, hab, bytes, cudaMemcpyHostToDevice, sb));
        CK(cudaStreamSynchronize(sb));
    }

    // Nach Zeit begrenzen, nicht nach Iterationen (Projektregel).
    cudaEvent_t e0, e1;
    CK(cudaSetDevice(a.ord_a));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    lat_us.clear();
    double spent = 0.0;
    while (spent < a.secs * 1e6) {
        CK(cudaEventRecord(e0, sa));
        // Hinweg A -> Host -> B
        CK(cudaSetDevice(a.ord_a));
        CK(cudaMemcpyAsync(hab, va, bytes, cudaMemcpyDeviceToHost, sa));
        CK(cudaStreamSynchronize(sa));
        CK(cudaSetDevice(a.ord_b));
        CK(cudaMemcpyAsync(vb, hab, bytes, cudaMemcpyHostToDevice, sb));
        CK(cudaStreamSynchronize(sb));
        // Rueckweg B -> Host -> A
        CK(cudaMemcpyAsync(hba, vb, bytes, cudaMemcpyDeviceToHost, sb));
        CK(cudaStreamSynchronize(sb));
        CK(cudaSetDevice(a.ord_a));
        CK(cudaMemcpyAsync(va, hba, bytes, cudaMemcpyHostToDevice, sa));
        CK(cudaEventRecord(e1, sa));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        lat_us.push_back(ms * 1000.0 / 2.0);   // halber Round-trip
        spent += ms * 1000.0;
    }

    if (fill_and_check(va, vb, bytes, a.ord_a, a.ord_b, false, &bad)) return 1;

    CK(cudaSetDevice(a.ord_a)); cudaFree(va); cudaStreamDestroy(sa);
    CK(cudaSetDevice(a.ord_b)); cudaFree(vb); cudaStreamDestroy(sb);
    cudaFreeHost(hab); cudaFreeHost(hba);
    return 0;
}

static int run_zc(const Args &a, size_t bytes, std::vector<double> &lat_us,
                  size_t &bad)
{
    const int n16 = (int)(bytes / 16);
    void *va = nullptr, *vb = nullptr;
    void *hab = nullptr, *hba = nullptr, *hflags = nullptr;

    CK(cudaSetDevice(a.ord_a));
    CK(cudaMalloc(&va, bytes));
    CK(cudaSetDevice(a.ord_b));
    CK(cudaMalloc(&vb, bytes));

    const unsigned flg = cudaHostAllocPortable | cudaHostAllocMapped;
    CK(cudaHostAlloc(&hab, bytes, flg));
    CK(cudaHostAlloc(&hba, bytes, flg));
    CK(cudaHostAlloc(&hflags, 4096, flg));
    memset(hflags, 0, 4096);

    // Geraetezeiger auf denselben Host-Speicher, je Karte eigener.
    void *dab_a, *dba_a, *dfl_a, *dab_b, *dba_b, *dfl_b;
    CK(cudaSetDevice(a.ord_a));
    CK(cudaHostGetDevicePointer(&dab_a, hab, 0));
    CK(cudaHostGetDevicePointer(&dba_a, hba, 0));
    CK(cudaHostGetDevicePointer(&dfl_a, hflags, 0));
    CK(cudaSetDevice(a.ord_b));
    CK(cudaHostGetDevicePointer(&dab_b, hab, 0));
    CK(cudaHostGetDevicePointer(&dba_b, hba, 0));
    CK(cudaHostGetDevicePointer(&dfl_b, hflags, 0));

    // Rundenzahl aus einem kurzen Kalibrierlauf, damit das Zeitbudget passt.
    // Beide Karten MUESSEN exakt dieselbe Zahl fahren -- sonst wartet eine
    // Seite in ihrer Warteschleife bis zum Deckel. (Dieselbe Fehlerfamilie
    // wie der NCCL-Deadlock im Projekt: rang-lokale Bedingung vor einem
    // gemeinsamen Ablauf.)
    int rounds_cal = 200;

    cudaStream_t sa, sb;
    long long *cyc_a = nullptr; int *st_a = nullptr, *st_b = nullptr;
    CK(cudaSetDevice(a.ord_a));
    CK(cudaStreamCreate(&sa));
    CK(cudaSetDevice(a.ord_b));
    CK(cudaStreamCreate(&sb));

    int clk_khz = 0;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, a.ord_a));
    if (clk_khz <= 0) { fprintf(stderr, "[FAIL] Taktrate unbekannt\n"); return 1; }

    auto launch_pair = [&](int rounds) -> int {
        CK(cudaSetDevice(a.ord_a));
        CK(cudaMalloc(&cyc_a, sizeof(long long) * (size_t)rounds));
        CK(cudaMallocManaged(&st_a, sizeof(int)));
        CK(cudaSetDevice(a.ord_b));
        CK(cudaMallocManaged(&st_b, sizeof(int)));
        *st_a = -1; *st_b = -1;

        CK(cudaSetDevice(a.ord_b));
        zc_agent<<<1, a.threads, 0, sb>>>((uint4 *)vb, (uint4 *)dba_b,
            (uint4 *)dab_b, (unsigned int *)dfl_b + 16, (unsigned int *)dfl_b,
            n16, rounds, 1, nullptr, st_b);
        CK(cudaGetLastError());
        CK(cudaSetDevice(a.ord_a));
        zc_agent<<<1, a.threads, 0, sa>>>((uint4 *)va, (uint4 *)dab_a,
            (uint4 *)dba_a, (unsigned int *)dfl_a, (unsigned int *)dfl_a + 16,
            n16, rounds, 0, cyc_a, st_a);
        CK(cudaGetLastError());

        CK(cudaSetDevice(a.ord_a)); CK(cudaStreamSynchronize(sa));
        CK(cudaSetDevice(a.ord_b)); CK(cudaStreamSynchronize(sb));
        if (*st_a != 0 || *st_b != 0) {
            fprintf(stderr, "[FAIL] zc timeout/abbruch (status a=%d b=%d) bei %zu B\n",
                    *st_a, *st_b, bytes);
            return 2;
        }
        return 0;
    };

    if (fill_and_check(va, vb, bytes, a.ord_a, a.ord_b, true, nullptr)) return 1;

    memset(hflags, 0, 4096);
    int rc = launch_pair(rounds_cal);
    if (rc) return rc;
    std::vector<long long> cy(rounds_cal);
    CK(cudaSetDevice(a.ord_a));
    CK(cudaMemcpy(cy.data(), cyc_a, sizeof(long long) * rounds_cal, cudaMemcpyDeviceToHost));
    cudaFree(cyc_a); cudaFree(st_a); cudaFree(st_b);
    double per_us = (double)cy[rounds_cal / 2] / (double)clk_khz * 1000.0;
    int rounds = (per_us > 0) ? (int)(a.secs * 1e6 / per_us) : 2000;
    rounds = std::max(200, std::min(200000, rounds));

    memset(hflags, 0, 4096);
    rc = launch_pair(rounds);
    if (rc) return rc;
    cy.assign(rounds, 0);
    CK(cudaSetDevice(a.ord_a));
    CK(cudaMemcpy(cy.data(), cyc_a, sizeof(long long) * rounds, cudaMemcpyDeviceToHost));
    cudaFree(cyc_a); cudaFree(st_a); cudaFree(st_b);

    lat_us.clear();
    for (int i = 0; i < rounds; ++i)
        lat_us.push_back((double)cy[i] / (double)clk_khz * 1000.0 / 2.0);  // halber RT

    if (fill_and_check(va, vb, bytes, a.ord_a, a.ord_b, false, &bad)) return 1;

    CK(cudaSetDevice(a.ord_a)); cudaFree(va); cudaStreamDestroy(sa);
    CK(cudaSetDevice(a.ord_b)); cudaFree(vb); cudaStreamDestroy(sb);
    cudaFreeHost(hab); cudaFreeHost(hba); cudaFreeHost(hflags);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Aufruf: %s <ord_a> <ord_b> <ce|zc> [--sizes=a,b,c] [--secs=S] [--threads=N]\n"
            "  Ordinale bei CUDA_DEVICE_ORDER=PCI_BUS_ID (= nvidia-smi-Index).\n", argv[0]);
        return 2;
    }
    Args a;
    a.ord_a = atoi(argv[1]);
    a.ord_b = atoi(argv[2]);
    a.mode  = argv[3];
    for (int i = 4; i < argc; ++i) {
        if (!strncmp(argv[i], "--sizes=", 8))        parse_sizes(argv[i] + 8, a.sizes);
        else if (!strncmp(argv[i], "--secs=", 7))    a.secs = atof(argv[i] + 7);
        else if (!strncmp(argv[i], "--threads=", 10)) a.threads = atoi(argv[i] + 10);
        else { fprintf(stderr, "[FAIL] unbekanntes Argument: %s\n", argv[i]); return 2; }
    }
    if (a.mode != "ce" && a.mode != "zc") {
        fprintf(stderr, "[FAIL] Modus muss ce oder zc sein\n"); return 2;
    }
    if (a.ord_a == a.ord_b) {
        fprintf(stderr, "[FAIL] zwei verschiedene Karten noetig\n"); return 2;
    }

    cudaDeviceProp pa, pb;
    CK(cudaGetDeviceProperties(&pa, a.ord_a));
    CK(cudaGetDeviceProperties(&pb, a.ord_b));
    printf("[info] A=ord%d %s (%02x:%02x.0)  B=ord%d %s (%02x:%02x.0)  modus=%s secs=%.1f threads=%d\n",
           a.ord_a, pa.name, pa.pciBusID, pa.pciDeviceID,
           a.ord_b, pb.name, pb.pciBusID, pb.pciDeviceID,
           a.mode.c_str(), a.secs, a.threads);
    fflush(stdout);

    for (size_t bytes : a.sizes) {
        if (a.mode == "zc" && (bytes % 16)) {
            fprintf(stderr, "[FAIL] zc braucht Vielfache von 16 B (%zu)\n", bytes);
            return 2;
        }
        std::vector<double> lat;
        size_t bad = (size_t)-1;
        int rc = (a.mode == "ce") ? run_ce(a, bytes, lat, bad)
                                  : run_zc(a, bytes, lat, bad);
        if (rc) return rc;
        if (lat.empty()) { fprintf(stderr, "[FAIL] keine Messwerte\n"); return 1; }
        std::sort(lat.begin(), lat.end());
        size_t n = lat.size();
        // Spaltenfolge wie DATA im RDMA-Bench, damit beide Tabellen
        // nebeneinander lesbar sind. us = HALBER Round-trip.
        // bad_bytes != 0 entwertet die Zeile -- eine Transportzahl ohne
        // Byte-Beleg zaehlt in diesem Projekt nicht.
        printf("RAMDATA\t%s\t%zu\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\tbad_bytes=%zu\n",
               a.mode.c_str(), bytes, n,
               lat[n / 10], lat[n / 2], lat[(n * 9) / 10],
               lat[(size_t)((double)n * 0.99)],
               (double)bytes / (2.0 * lat[n / 2]), bad);
        fflush(stdout);
    }
    return 0;
}

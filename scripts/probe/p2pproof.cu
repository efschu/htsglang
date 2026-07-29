// SPDX-License-Identifier: MIT
//
// p2pproof -- der Byte-Beleg fuer GPU-zu-GPU direkt ueber PCIe.
//
// ===========================================================================
// WOZU
// ===========================================================================
// Auf diesem Rig ist Peer-to-Peer zwischen GeForce-Karten treiberseitig
// gesperrt; jedes Byte zwischen zwei Karten laeuft heute durch den
// System-RAM (NCCL/SHM) oder ueber das NIC-Relay. Mit dem
// SMALLBAR_P2P-Patch soll der direkte Weg ueber die BAR1-Apertur offen
// sein. Diese Sonde beantwortet in dieser Reihenfolge:
//
//   1. Meldet der Treiber ueberhaupt Peer-Zugriff (cudaDeviceCanAccessPeer)?
//   2. Kommen die Bytes UNVERAENDERT an? Muster schreiben, auf der
//      ZIELKARTE zurueoklesen, jedes Byte vergleichen. Ohne bad_bytes=0
//      zaehlt keine Zeitmessung.
//   3. Erst dann: wie schnell, auf denselben Groessen und in derselben
//      Konvention wie gpurdma_04_bench und nccl_reference.py, damit die
//      Zahlen direkt nebeneinander stehen.
//
// Vergleichswerte auf diesem Rig (halber Round-trip, 5090 <-> 3080, solo):
//
//   Groesse   NCCL send/recv     NIC-Relay direkt
//   20 KiB      37,41 us            7,37 us
//   80 KiB      44,27 us           16,56 us
//    1 MiB     220,81 us          169,88 us
//
// ===========================================================================
// KONVENTION
// ===========================================================================
// Gemessen wird ein striktes Ping-Pong A -> B -> A und der HALBE Round-trip
// berichtet, exakt wie im C-Bench. Zeit wird mit CUDA-Events auf Karte A
// genommen. Nach Zeitbudget begrenzt, nicht nach Iterationen (Projektregel).
//
// Geraete werden ueber die PCI-Adresse benannt, nie ueber eine feste
// Ordinal-Annahme -- CUDA-Ordinal, NVML-Index und PCI-Reihenfolge sind auf
// diesem Rig verschieden. CUDA_DEVICE_ORDER=PCI_BUS_ID setzen.
//
// Bauen:  nvcc -O3 -gencode ... -o bin/p2pproof p2pproof.cu
// Aufruf: p2pproof [--sizes=a,b,c] [--secs=S] [--pairs=i:j,...]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[FAIL] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(e_)); return -1; } } while (0)

static std::string pci_of(int ord)
{
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, ord) != cudaSuccess) return "??";
    char b[32];
    snprintf(b, sizeof b, "%02x:%02x.0", p.pciBusID, p.pciDeviceID);
    return b;
}

// Byte-Beleg: Muster von A nach B, auf B zurueoklesen, jedes Byte pruefen.
// Der Vergleich passiert bewusst ueber den Umweg Host, damit er NICHT
// denselben Pfad benutzt wie der Transfer -- sonst wuerde ein defekter
// Pfad seinen eigenen Fehler verdecken.
static long long byte_proof(int a, int b, size_t bytes)
{
    void *va = nullptr, *vb = nullptr;
    std::vector<unsigned char> pat(bytes), got(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) pat[i] = (unsigned char)(i * 131 + 17);

    // Haengengebliebenen Fehler aus einem frueheren Aufruf abraeumen, sonst
    // meldet ihn die naechste Funktion und man diagnostiziert die falsche
    // Stelle.
    cudaGetLastError();

    CK(cudaSetDevice(a));
    cudaError_t ea = cudaMalloc(&va, bytes);
    if (ea != cudaSuccess) {
        fprintf(stderr, "[FAIL] cudaMalloc auf ord %d (%s): %s\n",
                a, pci_of(a).c_str(), cudaGetErrorString(ea));
        return -1;
    }
    CK(cudaSetDevice(b));
    cudaError_t eb = cudaMalloc(&vb, bytes);
    if (eb != cudaSuccess) {
        fprintf(stderr, "[FAIL] cudaMalloc auf ord %d (%s): %s\n",
                b, pci_of(b).c_str(), cudaGetErrorString(eb));
        cudaSetDevice(a); cudaFree(va);
        return -1;
    }
    CK(cudaSetDevice(a));
    CK(cudaMemcpy(va, pat.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaSetDevice(b)); CK(cudaMemset(vb, 0, bytes));

    CK(cudaMemcpyPeer(vb, b, va, a, bytes));
    CK(cudaDeviceSynchronize());

    CK(cudaSetDevice(b));
    CK(cudaMemcpy(got.data(), vb, bytes, cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (size_t i = 0; i < bytes; ++i) if (got[i] != pat[i]) ++bad;

    CK(cudaSetDevice(a)); cudaFree(va);
    CK(cudaSetDevice(b)); cudaFree(vb);
    return bad;
}

static int ladder(int a, int b, size_t bytes, double secs, double out[4])
{
    void *va = nullptr, *vb = nullptr;
    cudaEvent_t e0, e1;
    CK(cudaSetDevice(a)); CK(cudaMalloc(&va, bytes));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaSetDevice(b)); CK(cudaMalloc(&vb, bytes));
    CK(cudaSetDevice(a));

    for (int i = 0; i < 50; ++i) {           // Arbeitspunkt anfahren
        cudaMemcpyPeer(vb, b, va, a, bytes);
        cudaMemcpyPeer(va, a, vb, b, bytes);
    }
    CK(cudaDeviceSynchronize());

    std::vector<double> lat;
    double spent = 0.0;
    while (spent < secs * 1e6) {
        CK(cudaEventRecord(e0));
        CK(cudaMemcpyPeer(vb, b, va, a, bytes));   // Hinweg
        CK(cudaMemcpyPeer(va, a, vb, b, bytes));   // Rueckweg
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        lat.push_back(ms * 1000.0 / 2.0);          // halber Round-trip
        spent += ms * 1000.0;
    }
    std::sort(lat.begin(), lat.end());
    size_t n = lat.size();
    out[0] = lat[n / 10];
    out[1] = lat[n / 2];
    out[2] = lat[(n * 9) / 10];
    out[3] = lat[(size_t)((double)n * 0.99)];

    CK(cudaSetDevice(a)); cudaFree(va);
    CK(cudaSetDevice(b)); cudaFree(vb);
    return (int)n;
}

int main(int argc, char **argv)
{
    std::vector<size_t> sizes{20480, 81920, 1048576};
    double secs = 3.0;
    std::string pairs;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--sizes=", 8)) {
            sizes.clear();
            for (const char *p = argv[i] + 8; p && *p; ) {
                sizes.push_back(strtoull(p, nullptr, 10));
                const char *c = strchr(p, ','); p = c ? c + 1 : nullptr;
            }
        } else if (!strncmp(argv[i], "--secs=", 7)) secs = atof(argv[i] + 7);
        else if (!strncmp(argv[i], "--pairs=", 8)) pairs = argv[i] + 8;
        else { fprintf(stderr, "[FAIL] unbekannt: %s\n", argv[i]); return 2; }
    }

    int n = 0;
    CK(cudaGetDeviceCount(&n));
    printf("[info] %d Karten\n", n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i));
        printf("[info]   ord=%d pci=%s %s\n", i, pci_of(i).c_str(), p.name);
    }
    if (n < 2) { fprintf(stderr, "[FAIL] P2P braucht mindestens zwei Karten\n"); return 1; }

    // --- Stufe 1: meldet der Treiber Peer-Zugriff? -------------------------
    printf("\n[stufe1] Peer-Faehigkeit laut Treiber\n");
    std::vector<std::pair<int,int>> ok_pairs;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            int can = 0;
            CK(cudaDeviceCanAccessPeer(&can, a, b));
            printf("CANACCESS\t%s\t->\t%s\t%s\n", pci_of(a).c_str(),
                   pci_of(b).c_str(), can ? "JA" : "nein");
            if (can) ok_pairs.push_back({a, b});
        }
    }
    if (ok_pairs.empty()) {
        printf("\n[verdikt] Kein Paar meldet Peer-Zugriff -- P2P ist zu.\n");
        return 0;
    }

    for (auto &pr : ok_pairs) {
        CK(cudaSetDevice(pr.first));
        cudaError_t e = cudaDeviceEnablePeerAccess(pr.second, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
            printf("[warn] EnablePeerAccess %s -> %s: %s\n",
                   pci_of(pr.first).c_str(), pci_of(pr.second).c_str(),
                   cudaGetErrorString(e));
        }
    }

    // --- Stufe 2: Byte-Beleg. Ohne ihn zaehlt keine Zeitmessung. ----------
    printf("\n[stufe2] Byte-Beleg (Muster schreiben, auf der Zielkarte zurueoklesen)\n");
    std::vector<std::pair<int,int>> proven;
    for (auto &pr : ok_pairs) {
        long long bad = byte_proof(pr.first, pr.second, 1u << 20);
        if (bad < 0) { printf("PROOF\t%s\t->\t%s\tFEHLER\n",
                              pci_of(pr.first).c_str(), pci_of(pr.second).c_str()); continue; }
        printf("PROOF\t%s\t->\t%s\t%s\tbad_bytes=%lld\n",
               pci_of(pr.first).c_str(), pci_of(pr.second).c_str(),
               bad == 0 ? "PASS" : "FAIL", bad);
        if (bad == 0) proven.push_back(pr);
    }
    if (proven.empty()) {
        printf("\n[verdikt] Kein Paar besteht den Byte-Beleg -- keine Zeitmessung.\n");
        return 1;
    }

    // --- Stufe 3: Groessenleiter, halber Round-trip ------------------------
    printf("\n[stufe3] Groessenleiter, halber Round-trip in us\n");
    printf("# pair\tsize_bytes\tn\tp10\tp50\tp90\tp99\tMB_per_s\n");
    for (auto &pr : proven) {
        for (size_t sz : sizes) {
            double r[4];
            int cnt = ladder(pr.first, pr.second, sz, secs, r);
            if (cnt <= 0) { printf("[warn] Leiter fehlgeschlagen\n"); continue; }
            printf("P2PDATA\t%s->%s\t%zu\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.1f\n",
                   pci_of(pr.first).c_str(), pci_of(pr.second).c_str(),
                   sz, cnt, r[0], r[1], r[2], r[3],
                   (double)sz / (2.0 * r[1]));
            fflush(stdout);
        }
    }
    return 0;
}

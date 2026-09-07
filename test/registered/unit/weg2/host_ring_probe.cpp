// T1 harness: drive HostBackupRing WITHOUT a GPU.
//
// The ring's state machine -- the bitmap, the robust mutex, the blocking
// acquire, the stale-pid sweep, the foreign-release refusal -- is pure shared
// memory and pure pthreads.  Only three call sites touch the driver
// (cuDeviceGetUuid to name the card, cudaHostRegister to page-lock a span,
// cudaGetErrorString to report), and this file stubs exactly those, so the
// state machine is testable on a box with no free card and inside a hermetic
// CUDA_VISIBLE_DEVICES="" test run.  The stubs are the ONLY thing faked; the
// ring code under test is the shipping .cpp, compiled unmodified.
//
// Subcommands, each printing a line the python test asserts on.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

#include "host_ring.h"

// ------------------------------- driver stubs -------------------------------

//: Set WEG2_TEST_REGISTER_FAILS=1 to make cudaHostRegister refuse, which is the
//: "the published form does not hold on this driver" path (W33).
extern "C" CUresult cuDeviceGetUuid(CUuuid* out, CUdevice dev) {
    (void)dev;
    for (int i = 0; i < 16; ++i) {
        out->bytes[i] = (char)(0x10 + i);
    }
    return CUDA_SUCCESS;
}
extern "C" CUresult cuGetErrorString(CUresult r, const char** s) {
    (void)r;
    *s = "stub";
    return CUDA_SUCCESS;
}
extern "C" cudaError_t cudaHostRegister(void* p, size_t n, unsigned int f) {
    (void)p; (void)n; (void)f;
    const char* fail = getenv("WEG2_TEST_REGISTER_FAILS");
    if (fail != nullptr && fail[0] == '1') {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t e) { (void)e; return "stub"; }
extern "C" cudaError_t cudaGetLastError(void) { return cudaSuccess; }

// ---------------------------------- cases -----------------------------------

static const char* UUID = "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f";

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "";

    if (cmd == "uuid") {
        // The card is named from cuDeviceGetUuid, never from an ordinal.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) {
            printf("NO_RING\n");
            return 3;
        }
        printf("CARD %s\n", r->card_uuid().c_str());
        return 0;
    }

    if (cmd == "no_ring") {
        // No TMS_HOST_RING_* published at all => nullptr, the stock path.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        printf(r == nullptr ? "NO_RING\n" : "RING\n");
        return r == nullptr ? 0 : 1;
    }

    if (cmd == "granules") {
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        HostRingStats s0 = r->stats();
        // A size that is NOT a granule multiple must still be covered.
        size_t bytes = 3 * TMS_RING_GRANULE_BYTES + 1;
        std::vector<void*> g = r->acquire(bytes, TMS_RING_FAMILY_FLIP_BACKUP, "weights_0");
        HostRingStats s1 = r->stats();
        r->release(g);
        HostRingStats s2 = r->stats();
        printf("GRANULES total=%u got=%zu free0=%u free1=%u free2=%u peak=%u acquires=%llu\n",
               s0.granules_total, g.size(), s0.granules_free, s1.granules_free,
               s2.granules_free, s1.granules_peak_taken,
               (unsigned long long)s1.acquires);
        return 0;
    }

    if (cmd == "scatter") {
        // A scatter list covering a size no CONTIGUOUS run could: take
        // everything, give back every other granule, then ask for half.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        HostRingStats s = r->stats();
        std::vector<void*> all = r->acquire(
            (size_t)s.granules_total * TMS_RING_GRANULE_BYTES,
            TMS_RING_FAMILY_FLIP_BACKUP, "weights_0");
        std::vector<void*> odd;
        for (size_t i = 1; i < all.size(); i += 2) {
            odd.push_back(all[i]);
        }
        r->release(odd);
        size_t want = odd.size();
        std::vector<void*> got = r->acquire(want * TMS_RING_GRANULE_BYTES,
                                            TMS_RING_FAMILY_CARRIER, "carrier");
        // Contiguity check: no two returned pointers are adjacent.
        int adjacent = 0;
        for (size_t i = 1; i < got.size(); ++i) {
            if ((char*)got[i] - (char*)got[i - 1] == (long)TMS_RING_GRANULE_BYTES) {
                adjacent = 1;
            }
        }
        printf("SCATTER want=%zu got=%zu adjacent=%d\n", want, got.size(), adjacent);
        return 0;
    }

    if (cmd == "family") {
        // Two tag families share the ONE region -- the carrier's API surface.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        std::vector<void*> a = r->acquire(TMS_RING_GRANULE_BYTES,
                                          TMS_RING_FAMILY_FLIP_BACKUP, "weights_0");
        std::vector<void*> b = r->acquire(TMS_RING_GRANULE_BYTES,
                                          TMS_RING_FAMILY_CARRIER, "hicache_page");
        HostRingStats s = r->stats();
        printf("FAMILY a=%zu b=%zu same=%d free=%u\n", a.size(), b.size(),
               a[0] == b[0] ? 1 : 0, s.granules_free);
        return 0;
    }

    if (cmd == "blocking") {
        // A child takes everything, then releases; the parent's acquire must
        // BLOCK and then succeed -- not fall back to a second allocator.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        HostRingStats s = r->stats();
        std::vector<void*> all = r->acquire(
            (size_t)s.granules_total * TMS_RING_GRANULE_BYTES,
            TMS_RING_FAMILY_FLIP_BACKUP, "hog");
        // The waiter is a CHILD using the INHERITED mapping (fork keeps a
        // MAP_SHARED mapping at the same address, and gives the child its own
        // pid -- which is exactly the two-co-located-ranks shape).  It must
        // BLOCK until this process releases; a non-blocking acquire would come
        // back empty instead of waiting, which is the whole point of R5.
        pid_t peer = fork();
        if (peer == 0) {
            std::vector<void*> g = r->acquire(TMS_RING_GRANULE_BYTES,
                                              TMS_RING_FAMILY_FLIP_BACKUP, "waiter");
            printf("BLOCKED_THEN_GOT %zu\n", g.size());
            fflush(stdout);
            _exit(g.size() == 1 ? 0 : 1);
        }
        usleep(400 * 1000);
        r->release(all);
        int st2 = 0;
        waitpid(peer, &st2, 0);
        HostRingStats s2 = r->stats();
        printf("BLOCKING waiter_rc=%d blocked_ms=%.0f\n", WEXITSTATUS(st2), s2.blocked_ms);
        return WEXITSTATUS(st2);
    }

    if (cmd == "foreign_release") {
        // A granule owned by another pid may never be released here.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        int fds[2];
        if (pipe(fds) != 0) return 9;
        // Same inherited mapping (so the ADDRESS is meaningful in both
        // processes), different pid (so the OWNER differs) -- the exact shape
        // in which a co-located rank could free the peer's live backup.
        pid_t pid = fork();
        if (pid == 0) {
            std::vector<void*> g = r->acquire(TMS_RING_GRANULE_BYTES,
                                              TMS_RING_FAMILY_FLIP_BACKUP, "peer");
            void* p = g[0];
            ssize_t w = write(fds[1], &p, sizeof(p));
            (void)w;
            usleep(2000 * 1000);
            _exit(0);
        }
        void* peer_ptr = nullptr;
        ssize_t rd = read(fds[0], &peer_ptr, sizeof(peer_ptr));
        (void)rd;
        printf("FOREIGN_RELEASE_ATTEMPT\n");
        fflush(stdout);
        std::vector<void*> steal;
        steal.push_back(peer_ptr);
        r->release(steal);  // must exit(1)
        printf("FOREIGN_RELEASE_ACCEPTED\n");
        return 0;
    }

    if (cmd == "stale_sweep") {
        // A dead pid's granules -- and ONLY those -- come back on the next attach.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        std::vector<void*> mine = r->acquire(2 * TMS_RING_GRANULE_BYTES,
                                             TMS_RING_FAMILY_FLIP_BACKUP, "mine");
        pid_t pid = fork();
        if (pid == 0) {
            HostBackupRing* r2 = HostBackupRing::open_from_env(0);
            r2->acquire(3 * TMS_RING_GRANULE_BYTES, TMS_RING_FAMILY_FLIP_BACKUP, "doomed");
            _exit(0);  // dies holding three granules
        }
        int st = 0;
        waitpid(pid, &st, 0);
        HostRingStats before = r->stats();
        // Any acquire wakes the sweep through the timed wait; force it by
        // asking for everything that should be free.
        uint32_t want = before.granules_total - 2;  // all but "mine"
        std::vector<void*> got = r->acquire((size_t)want * TMS_RING_GRANULE_BYTES,
                                            TMS_RING_FAMILY_FLIP_BACKUP, "after");
        HostRingStats after = r->stats();
        printf("SWEEP before_free=%u got=%zu after_free=%u swept=%llu mine=%zu\n",
               before.granules_free, got.size(), after.granules_free,
               (unsigned long long)after.swept_stale, mine.size());
        return got.size() == want ? 0 : 1;
    }

    if (cmd == "exhausted") {
        // More than the whole region, with no peer to release: W31 + exit(1)
        // after the bounded wait.  The budget is shortened by the harness only
        // through the number of granules, never by an env knob -- the test asks
        // for a size that can NEVER be satisfied.
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        HostRingStats s = r->stats();
        printf("EXHAUST_ATTEMPT total=%u\n", s.granules_total);
        fflush(stdout);
        r->acquire((size_t)(s.granules_total + 1) * TMS_RING_GRANULE_BYTES,
                   TMS_RING_FAMILY_FLIP_BACKUP, "too_big");
        printf("EXHAUST_ACCEPTED\n");
        return 0;
    }

    if (cmd == "register") {
        HostBackupRing* r = HostBackupRing::open_from_env(0);
        if (r == nullptr) { printf("NO_RING\n"); return 3; }
        r->register_span(0);
        r->register_span(1);
        HostRingStats s = r->stats();
        printf("REGISTER spans=%u\n", s.spans_registered);
        return 0;
    }

    fprintf(stderr, "unknown case %s\n", cmd.c_str());
    return 2;
}

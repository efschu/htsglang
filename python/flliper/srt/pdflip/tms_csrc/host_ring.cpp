#include "host_ring.h"

#if defined(USE_CUDA)

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

// Deliberately NOT "utils.h": this module uses none of its macros, and pulling
// it in would drag CUDAUtils' driver calls (cuMemCreate, cuCtxGetDevice, ...)
// into the translation unit.  Keeping the ring free of them is what lets T1
// link it against three stubs and exercise the whole state machine with no GPU.

namespace {

double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

const char* env_or_null(const char* key) {
    const char* v = getenv(key);
    if (v == nullptr || v[0] == '\0') {
        return nullptr;
    }
    return v;
}

//: The NVML spelling of a device UUID, which is what the launcher publishes and
//: what ``nvidia-smi`` prints.  ``cuDeviceGetUuid`` hands back the same 16
//: bytes; only the rendering differs, so rendering it here is what lets the
//: card be resolved without ever touching a CVD ordinal.
std::string uuid_string(CUdevice device) {
    CUuuid raw;
    CUresult rc = cuDeviceGetUuid(&raw, device);
    if (rc != CUDA_SUCCESS) {
        const char* err = nullptr;
        cuGetErrorString(rc, &err);
        std::cerr << "[host_ring.cpp] cuDeviceGetUuid failed: " << rc << " ("
                  << (err ? err : "unknown") << ") -- the card of a ring "
                  << "attach is NEVER guessed from a CVD ordinal" << std::endl;
        exit(1);
    }
    static const char* hex = "0123456789abcdef";
    std::string out = "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out.push_back('-');
        }
        unsigned char b = static_cast<unsigned char>(raw.bytes[i]);
        out.push_back(hex[(b >> 4) & 0xF]);
        out.push_back(hex[b & 0xF]);
    }
    return out;
}

uint32_t tag_hash_of(const std::string& tag) {
    // FNV-1a, 32 bit.  Attribution only (L3/L4), never an identity.
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < tag.size(); ++i) {
        h ^= static_cast<unsigned char>(tag[i]);
        h *= 16777619u;
    }
    return h;
}

bool pid_alive(int pid) {
    if (pid <= 0) {
        return false;
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d", pid);
    struct stat st;
    return stat(path, &st) == 0;
}

//: One entry of ``TMS_HOST_RING_MAP``: ``<uuid>=<bytes>:<span1_bytes>[:fd=<n>]``.
struct MapEntry {
    bool found = false;
    uint64_t bytes = 0;
    uint64_t span1_bytes = 0;
};

MapEntry parse_map(const char* spec, const std::string& uuid) {
    MapEntry out;
    std::string s(spec);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t comma = s.find(',', pos);
        std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t eq = item.find('=');
        if (eq != std::string::npos && item.substr(0, eq) == uuid) {
            std::string rest = item.substr(eq + 1);
            size_t c1 = rest.find(':');
            if (c1 == std::string::npos) {
                std::cerr << "[host_ring.cpp] TMS_HOST_RING_MAP entry for " << uuid
                          << " has no ':<span1_bytes>': " << item << std::endl;
                exit(1);
            }
            out.bytes = strtoull(rest.substr(0, c1).c_str(), nullptr, 10);
            std::string tail = rest.substr(c1 + 1);
            size_t c2 = tail.find(':');
            out.span1_bytes = strtoull(tail.substr(0, c2).c_str(), nullptr, 10);
            if (c2 != std::string::npos) {
                // The map is sizes only.  A third field comes from a publisher
                // that still believed an fd number survives spawn; refuse by
                // name rather than ignore it.
                std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: "
                          << "TMS_HOST_RING_MAP entry for " << uuid
                          << " carries a third field '" << tail.substr(c2 + 1)
                          << "'.  The map is '<uuid>=<bytes>:<span1>' and nothing "
                          << "else: an fd number cannot address this region from a "
                          << "spawn-started rank." << std::endl;
                exit(1);
            }
            out.found = true;
            return out;
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

void init_header_locked(HostRingHeader* h, uint64_t epoch, uint32_t granules, uint32_t span1) {
    memset(h, 0, sizeof(*h));
    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
    // ROBUST is the whole reason a peer's exit(1) mid-flip is survivable rather
    // than a wedge: the next locker gets EOWNERDEAD and sweeps.
    pthread_mutexattr_setrobust(&ma, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&h->mutex, &ma);
    pthread_mutexattr_destroy(&ma);
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&h->cond, &ca);
    pthread_condattr_destroy(&ca);
    h->granule_bytes = static_cast<uint32_t>(TMS_RING_GRANULE_BYTES);
    h->epoch = epoch;
    h->granules_total = granules;
    h->span1_granules = span1;
    h->granules_free = granules;
    h->granules_free_min = granules;
    h->granules_peak_taken = 0;
    h->version = TMS_RING_VERSION;
    // magic and initialised go LAST: a concurrent attacher that sees the magic
    // must be able to trust every field before it.
    h->magic = TMS_RING_MAGIC;
    h->initialised = 1;
}

}  // namespace

HostBackupRing* HostBackupRing::open_from_env(CUdevice device) {
    const char* dir = env_or_null("TMS_HOST_RING_DIR");
    const char* map = env_or_null("TMS_HOST_RING_MAP");
    if (dir == nullptr && map == nullptr) {
        // No ring published: the stock cudaMallocHost path runs unchanged.
        // This is the ONLY fallback, it is boot-level, and it is silent by
        // design (spec section 10.4).
        return nullptr;
    }
    const char* form = env_or_null("TMS_HOST_RING_FORM");
    if (form == nullptr) {
        std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: TMS_HOST_RING_DIR is "
                  << "published but TMS_HOST_RING_FORM is not.  The launcher "
                  << "publishes the form the step-0 probe proved, or publishes "
                  << "nothing at all and the OLD flip form runs." << std::endl;
        exit(1);
    }
    std::string form_s(form);
    if (form_s != "MAP_SHARED") {
        // MAP_SHARED is the only form built.  The step-0 metal probe
        // (WEG2_BUILD_DECISIONS_0906 section 1p, 2026-09-07T23:13:23Z) proved
        // cudaHostRegister on a cross-process /dev/shm MAP_SHARED range and
        // recorded that the memfd fallback is NOT NEEDED; a second
        // implementation of a settled question is a defect surface, not an
        // option.
        std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: TMS_HOST_RING_FORM="
                  << form_s << " is not MAP_SHARED, the only form this build "
                  << "implements (spec R16, settled on the metal in section 1p)"
                  << std::endl;
        exit(1);
    }
    if (map == nullptr) {
        std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: TMS_HOST_RING_DIR "
                  << "without TMS_HOST_RING_MAP -- the per-card sizes are solved by "
                  << "the launcher from the previous boot's own lines, never here"
                  << std::endl;
        exit(1);
    }

    std::string uuid = uuid_string(device);
    MapEntry entry = parse_map(map, uuid);
    if (!entry.found) {
        // A card the launcher did not size is not this rank's business: the
        // stock path runs for it.  Named rather than silent.
        std::cerr << "[host_ring.cpp] host ring: no TMS_HOST_RING_MAP entry for card "
                  << uuid << " -- stock cudaMallocHost path for this device" << std::endl;
        return nullptr;
    }
    uint64_t epoch = 0;
    const char* ep = env_or_null("TMS_HOST_RING_EPOCH");
    if (ep != nullptr) {
        epoch = strtoull(ep, nullptr, 10);
    }

    uint32_t granules = static_cast<uint32_t>(entry.bytes / TMS_RING_GRANULE_BYTES);
    uint32_t span1 = static_cast<uint32_t>(entry.span1_bytes / TMS_RING_GRANULE_BYTES);
    if (granules == 0 || granules > TMS_RING_MAX_GRANULES) {
        std::cerr << "[host_ring.cpp] host ring size for " << uuid << " is "
                  << entry.bytes << " B = " << granules << " granules, outside 1.."
                  << TMS_RING_MAX_GRANULES << std::endl;
        exit(1);
    }
    if (span1 > granules) {
        span1 = granules;
    }

    HostBackupRing* ring = new HostBackupRing();
    ring->card_uuid_ = uuid;
    ring->form_ = form_s;
    ring->map_bytes_ = TMS_RING_HEADER_BYTES + static_cast<size_t>(granules) * TMS_RING_GRANULE_BYTES;

    // A PATH, never an inherited fd number.  The ranks that reach this code are
    // scheduler processes started with mp.set_start_method("spawn", force=True)
    // (entrypoints/engine.py), which passes only its own handles: an fd number
    // published by the launcher names a DIFFERENT object here -- a closed slot,
    // a socket, a log file -- and mmapping it would write a ring header into
    // whatever that is.  The map therefore carries sizes only.
    ring->path_ = std::string(dir ? dir : "") + "/" + uuid + ".ring";
    ring->fd_ = open(ring->path_.c_str(), O_RDWR);
    if (ring->fd_ < 0) {
        std::cerr << "[host_ring.cpp] host ring open(" << ring->path_
                  << ") failed: " << strerror(errno)
                  << " -- the launcher creates and ftruncates every per-card file "
                  << "BEFORE either group starts (spec C18)" << std::endl;
        exit(1);
    }
    struct stat st;
    if (fstat(ring->fd_, &st) != 0 ||
        static_cast<uint64_t>(st.st_size) < static_cast<uint64_t>(ring->map_bytes_)) {
        std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: " << ring->path_
                  << " is " << st.st_size << " B but the map claims "
                  << ring->map_bytes_ << " B -- refusing to mmap and initialise an "
                  << "object whose identity is not established" << std::endl;
        exit(1);
    }

    ring->base_ = mmap(nullptr, ring->map_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, ring->fd_, 0);
    if (ring->base_ == MAP_FAILED) {
        std::cerr << "[host_ring.cpp] host ring mmap(" << ring->map_bytes_ << " B, "
                  << ring->path_ << ") failed: " << strerror(errno) << std::endl;
        exit(1);
    }
    ring->h_ = static_cast<HostRingHeader*>(ring->base_);

    // Initialisation race: both co-located ranks attach independently.  flock on
    // the shared fd is the one arbiter, and it is held only across the memset +
    // mutex init, never across a copy.
    if (flock(ring->fd_, LOCK_EX) != 0) {
        std::cerr << "[host_ring.cpp] host ring flock failed: " << strerror(errno) << std::endl;
        exit(1);
    }
    bool fresh = false;
    if (ring->h_->magic != TMS_RING_MAGIC || ring->h_->initialised != 1 ||
        ring->h_->version != TMS_RING_VERSION || ring->h_->epoch != epoch) {
        // Epoch mismatch = a previous boot's leftovers in the same tmpfs file.
        // Zero rather than adopt: adopting would inherit owner_pids of processes
        // that no longer exist and free bits that mean nothing.
        init_header_locked(ring->h_, epoch, granules, span1);
        fresh = true;
    }
    uint32_t swept = 0;
    if (!fresh) {
        int rc = pthread_mutex_lock(&ring->h_->mutex);
        if (rc == EOWNERDEAD) {
            swept = ring->sweep_stale_locked();
            pthread_mutex_consistent(&ring->h_->mutex);
        } else if (rc != 0) {
            std::cerr << "[host_ring.cpp] host ring mutex lock failed on attach: "
                      << strerror(rc) << std::endl;
            exit(1);
        } else {
            swept = ring->sweep_stale_locked();
        }
        pthread_mutex_unlock(&ring->h_->mutex);
    }
    flock(ring->fd_, LOCK_UN);

    std::cerr << "[host_ring.cpp] WEG2-RING-ATTACH card=" << uuid << " path=" << ring->path_
              << " form=" << form_s << " epoch=" << epoch << " bytes=" << entry.bytes
              << " granules=" << granules << " span1_granules=" << span1
              << " fresh=" << (fresh ? 1 : 0) << " swept_stale=" << swept
              << " pid=" << getpid() << std::endl;
    return ring;
}

void HostBackupRing::lock() const {
    int rc = pthread_mutex_lock(&h_->mutex);
    if (rc == EOWNERDEAD) {
        // The peer died holding the lock.  Its granules are exactly the ones
        // whose owner_pid is gone, and the sweep is the recovery -- there is no
        // second place where that state could have been recorded.
        const_cast<HostBackupRing*>(this)->sweep_stale_locked();
        pthread_mutex_consistent(&h_->mutex);
        return;
    }
    if (rc != 0) {
        std::cerr << "[host_ring.cpp] host ring mutex lock failed: " << strerror(rc)
                  << " card=" << card_uuid_ << std::endl;
        exit(1);
    }
}

void HostBackupRing::unlock() const { pthread_mutex_unlock(&h_->mutex); }

uint32_t HostBackupRing::sweep_stale_locked() const {
    uint32_t swept = 0;
    int self = static_cast<int>(getpid());
    for (uint32_t i = 0; i < h_->granules_total; ++i) {
        int owner = h_->owner_pid[i];
        if (owner == 0 || owner == self) {
            continue;
        }
        if (!pid_alive(owner)) {
            h_->owner_pid[i] = 0;
            h_->family[i] = TMS_RING_FAMILY_FREE;
            h_->tag_hash[i] = 0;
            h_->granules_free += 1;
            swept += 1;
        }
    }
    h_->swept_stale += swept;
    return swept;
}

uint32_t HostBackupRing::granule_index(void* ptr) const {
    size_t off = static_cast<char*>(ptr) - (static_cast<char*>(base_) + TMS_RING_HEADER_BYTES);
    return static_cast<uint32_t>(off / TMS_RING_GRANULE_BYTES);
}

void HostBackupRing::register_span(int index) {
    if (index < 0 || index > 1 || span_registered_[index]) {
        return;
    }
    uint32_t first = (index == 0) ? 0u : h_->span1_granules;
    uint32_t last = (index == 0) ? h_->span1_granules : h_->granules_total;
    if (last <= first) {
        span_registered_[index] = true;
        return;
    }
    void* addr = granule_ptr(first);
    size_t bytes = static_cast<size_t>(last - first) * TMS_RING_GRANULE_BYTES;
    double t0 = now_s();
    // cudaHostRegisterPortable so the pages are page-locked for EVERY context of
    // THIS process; the peer process registers the same pages independently
    // (spec R16 -- the metal fact the step-0 probe settles).
    cudaError_t rc = cudaHostRegister(addr, bytes, cudaHostRegisterPortable);
    double ms = (now_s() - t0) * 1000.0;
    if (rc == cudaErrorHostMemoryAlreadyRegistered) {
        // Same-process overlap only; not an error for us.
        cudaGetLastError();
        rc = cudaSuccess;
    }
    if (rc != cudaSuccess) {
        std::cerr << "[host_ring.cpp] W33 Weg2RingFormUnproven: cudaHostRegister(span="
                  << (index + 1) << ", " << bytes << " B, form=" << form_ << ") on card "
                  << card_uuid_ << " failed: " << cudaGetErrorString(rc)
                  << " -- the published form does not hold on this driver; the boot "
                  << "must fall back to the OLD flip form" << std::endl;
        exit(1);
    }
    span_registered_[index] = true;
    std::cerr << "[host_ring.cpp] WEG2-RING-OPEN card=" << card_uuid_ << " path=" << path_
              << " span=" << (index + 1) << "/2 bytes=" << (bytes / (1024 * 1024))
              << " MiB granule=2 MiB granules=" << (last - first) << " register_ms="
              << static_cast<long>(ms) << " peer_pid=" << peer_pid_hint_
              << " swept_stale=" << h_->swept_stale << std::endl;
}

void HostBackupRing::ensure_span_for_locked(uint32_t granule_index_) {
    // Registration is lazy per span (R7): span 1 at the first pause, span 2 the
    // first time a request reaches past it.  Registering both at launch is what
    // costs the M=1200 arm 2.98 GiB it does not have.
    int span = (granule_index_ < h_->span1_granules) ? 0 : 1;
    if (!span_registered_[span]) {
        register_span(span);
    }
}

std::vector<void*> HostBackupRing::acquire(size_t bytes, uint8_t family, const std::string& tag) {
    std::vector<void*> out;
    uint32_t need = static_cast<uint32_t>(bytes_to_granules(bytes));
    if (need == 0) {
        return out;
    }
    uint32_t hash = tag_hash_of(tag);
    int self = static_cast<int>(getpid());
    double t0 = now_s();
    double waited_ms = 0.0;

    lock();
    while (true) {
        if (h_->granules_free >= need) {
            break;
        }
        double elapsed = now_s() - t0;
        if (elapsed >= TMS_RING_ACQUIRE_BUDGET_S) {
            // W31: the backstop.  Provably unreachable given the launch check
            // (L6 / W32, R15) -- reaching it means the check was violated, so
            // the message carries the arithmetic that would have caught it.
            int holder = 0;
            for (uint32_t i = 0; i < h_->granules_total; ++i) {
                if (h_->owner_pid[i] != 0 && h_->owner_pid[i] != self) {
                    holder = h_->owner_pid[i];
                    break;
                }
            }
            uint32_t free_now = h_->granules_free;
            unlock();
            std::cerr << "[host_ring.cpp] W31 Weg2HostRingExhausted card=" << card_uuid_
                      << " tag=" << tag << " need=" << (need * 2) << " free="
                      << (free_now * 2) << " MiB -- the launch check (L6) was violated."
                      << " waiter=pid " << self << " waited_ms=" << static_cast<long>(elapsed * 1000)
                      << " peer_holder_pid=" << holder << " family=" << static_cast<int>(family)
                      << std::endl;
            exit(1);
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        // 250 ms slices rather than one long wait: a condvar cannot be robust,
        // so a peer that dies between our two checks must not hold us past the
        // budget.  Each wake re-sweeps.
        ts.tv_nsec += 250L * 1000L * 1000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_nsec -= 1000000000L;
            ts.tv_sec += 1;
        }
        int rc = pthread_cond_timedwait(&h_->cond, &h_->mutex, &ts);
        if (rc == EOWNERDEAD) {
            sweep_stale_locked();
            pthread_mutex_consistent(&h_->mutex);
        } else if (rc == ETIMEDOUT) {
            sweep_stale_locked();
        }
        waited_ms = (now_s() - t0) * 1000.0;
    }

    for (uint32_t i = 0; i < h_->granules_total && out.size() < need; ++i) {
        if (h_->owner_pid[i] != 0) {
            continue;
        }
        h_->owner_pid[i] = self;
        h_->family[i] = family;
        h_->tag_hash[i] = hash;
        h_->granules_free -= 1;
        ensure_span_for_locked(i);
        out.push_back(granule_ptr(i));
    }
    if (h_->granules_free < h_->granules_free_min) {
        h_->granules_free_min = h_->granules_free;
    }
    uint32_t taken = h_->granules_total - h_->granules_free;
    if (taken > h_->granules_peak_taken) {
        h_->granules_peak_taken = taken;
    }
    h_->acquires += 1;
    h_->blocked_us += static_cast<uint64_t>(waited_ms * 1000.0);
    unlock();
    return out;
}

void HostBackupRing::release(const std::vector<void*>& granules) {
    if (granules.empty()) {
        return;
    }
    int self = static_cast<int>(getpid());
    lock();
    for (size_t k = 0; k < granules.size(); ++k) {
        uint32_t i = granule_index(granules[k]);
        if (i >= h_->granules_total) {
            unlock();
            std::cerr << "[host_ring.cpp] host ring release of a pointer outside the "
                      << "region on card " << card_uuid_ << " (index " << i << " of "
                      << h_->granules_total << ")" << std::endl;
            exit(1);
        }
        if (h_->owner_pid[i] != self) {
            // A foreign release is never a compensation: the granule's content
            // belongs to the peer and freeing it would hand the peer's live
            // backup to the next acquirer.
            int owner = h_->owner_pid[i];
            unlock();
            std::cerr << "[host_ring.cpp] host ring REFUSES a foreign release: granule "
                      << i << " on card " << card_uuid_ << " is owned by pid " << owner
                      << ", this process is " << self << std::endl;
            exit(1);
        }
        h_->owner_pid[i] = 0;
        h_->family[i] = TMS_RING_FAMILY_FREE;
        h_->tag_hash[i] = 0;
        h_->granules_free += 1;
    }
    h_->releases += 1;
    pthread_cond_broadcast(&h_->cond);
    unlock();
}

HostRingStats HostBackupRing::stats() const {
    HostRingStats s;
    lock();
    s.granules_total = h_->granules_total;
    s.granules_free = h_->granules_free;
    s.granules_free_min = h_->granules_free_min;
    s.granules_peak_taken = h_->granules_peak_taken;
    s.acquires = h_->acquires;
    s.releases = h_->releases;
    s.blocked_ms = static_cast<double>(h_->blocked_us) / 1000.0;
    s.swept_stale = h_->swept_stale;
    s.span1_granules = h_->span1_granules;
    unlock();
    s.spans_registered = (span_registered_[0] ? 1u : 0u) + (span_registered_[1] ? 1u : 0u);
    return s;
}

#endif  // USE_CUDA

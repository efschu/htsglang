#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include "utils.h"
#include "macro.h"
#include "host_ring.h"

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    // WEG2 flip-cost C2: the host backup is a SCATTER LIST of 2 MiB granules,
    // not one contiguous pinned block.  ``cpu_backup_from_ring`` says which
    // allocator owns them -- the shared per-card ring (the boot published
    // TMS_HOST_RING_*) or stock ``cudaMallocHost`` (it did not).  There is no
    // third state and no per-allocation fallback between the two: the choice is
    // made once per boot (spec section 10.4).
    std::vector<void*> cpu_backup_granules;
    bool cpu_backup_from_ring;

#if defined(USE_CUDA)
    CUmemGenericAllocationHandle allocHandle;
#elif defined(USE_ROCM)
    size_t aligned_size;
    std::vector<hipMemGenericAllocationHandle_t> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    #error "USE_PLATFORM is not set"
#endif
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t free(void* ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

    //: C7: the planner's sizing input.  Sum of ``metadata.size`` over the
    //: allocations carrying ``tag``.  This REPLACES the RssShmem delta as the
    //: per-tag instrument: ring granules are shared pages mapped by both
    //: co-located processes, so RssShmem collapses to ~0 and a per-process sum
    //: double-counts (spec R8).
    uint64_t tag_bytes(const std::string& tag);
    //: C7: the live ring counters of this rank's card, or false when this boot
    //: published no ring.
    bool ring_stats(HostRingStats* out, std::string* card_uuid);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    //: R12 (NEW HAZARD): this stream is created with ``cudaStreamCreate``, i.e.
    //: DEFAULT (blocking) flags, and never with ``cudaStreamNonBlocking``.  The
    //: copies it carries replace ``cudaMemcpy`` on the legacy default stream,
    //: which implicitly ordered against PyTorch's default stream; a
    //: non-blocking stream silently drops that ordering and copies pages a
    //: pending kernel is still writing.  Pinned by T4.
    cudaStream_t backup_stream_ = nullptr;
    void ensure_backup_stream();
    //: The per-card shared host granule ring, or nullptr for the stock path.
    //: Opened lazily at the first cpu-backed pause, when a CUDA context (and
    //: therefore a resolvable device UUID) certainly exists.
    HostBackupRing* ring_ = nullptr;
    bool ring_open_attempted_ = false;
    HostBackupRing* ensure_ring(CUdevice device);
    //: C6 / R20: the ONE gate through which host bytes may be taken for an
    //: allocation.  kv_cache is paused WITHOUT enable_cpu_backup
    //: (weight_updater.py:686-687) and must therefore never reach either host
    //: allocator.
    static void assert_host_backup_eligible(const AllocationMetadata& metadata);

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
};